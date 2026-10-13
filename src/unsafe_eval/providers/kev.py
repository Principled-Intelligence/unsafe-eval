"""Inspect model provider for Kev, an open reconstruction of Jev's System One API.

Kev serves TypeSafe's System One API, so this provider sends the same request as the
`typesafe` provider and runs under the `jev` protocol. It differs in where the model
comes from:

* It starts `kev.serve` on the pinned checkpoint for each run. Kev requires an older
  torch than this package's vLLM, so `uv` runs it in an isolated environment built from
  the pinned Kev commit, with the kernel packages at the versions Kev's own lock tests.
* `kev.serve` answers to `kev-latest` whatever it loaded, so the provider reads the
  served run from `/v1/models` and fails unless it is the pinned `<repo>@<revision>`.
* `quantization="fp8"` stores a full-weight checkpoint's linear layers in FP8, which fits
  Kev-27B in one 32 GB GPU. `kev` has no such option, so the server starts through
  `_kev_fp8.py`, which loads the backbone with transformers' FP8 config.
"""

import atexit
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelOutput, modelapi
from inspect_ai.tool import ToolChoice, ToolInfo

from unsafe_eval.providers._pins import commit
from unsafe_eval.providers.typesafe import (
    TypeSafeAPI,
    model_output_from_response,
    questions_from_messages,
    state_from_messages,
)

KEV_REPOSITORY = "https://github.com/jaredpalmer/kev"
# Kev's lock at its `kev-1.0` tag, plus the fused-kernel package its serving image installs.
KERNEL_PINS = (
    "torch==2.8.0",
    "transformers==5.17.0",
    "peft==0.21.0",
    "flash-linear-attention==0.5.2",
)
FP8_LAUNCHER = Path(__file__).with_name("_kev_fp8.py")
# The name `kev.serve` answers to, whichever checkpoint it loaded.
SERVED_NAME = "kev-latest"
# A first start builds the environment and downloads the base model and the adapter.
SERVER_START_SECONDS = 1800


def served_runs(url: str) -> set[str]:
    """The runs a Kev server reports loading."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as response:
        return {model["run"] for model in json.load(response)["models"]}


def serve(run: str, kev_revision: str, quantization: str | None = None) -> str:
    """Start `kev.serve` on `run` and return its URL once it reports serving exactly `run`."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    fp8 = quantization == "fp8"
    pins = [argument for pin in KERNEL_PINS for argument in ("--with", pin)]
    kev = f"kev[serve] @ git+{KEV_REPOSITORY}@{kev_revision}"
    # With KEV_API_KEY set, the server would refuse the client's placeholder key.
    env = {name: value for name, value in os.environ.items() if name != "KEV_API_KEY"}
    # `-I` keeps the launcher's directory off `sys.path`, where this module would shadow Kev's.
    server = ["-I", str(FP8_LAUNCHER)] if fp8 else ["-m", "kev.serve"]
    if fp8:
        # Kev's fused kernels concatenate the linear weights, which FP8 layers cannot, and the
        # CUDA graphs' preallocated buffers do not fit next to Kev-27B's FP8 weights in 32 GB.
        env["KEV_FUSED"] = "0"
        env["KEV_CUDA_GRAPHS"] = "0"
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            kev,
            *pins,
            "python",
            *server,
            "--run",
            run,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        # `uv run` starts the server as a child, so stopping it means stopping the group.
        start_new_session=True,
    )
    atexit.register(_stop, process)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"kev.serve exited with code {process.returncode} before it was ready"
            )
        try:
            runs = served_runs(url)
        except OSError:
            time.sleep(2)
            continue
        if runs != {run}:
            _stop(process)
            raise ValueError(f"kev.serve loaded {sorted(runs)!r} instead of the pinned {run!r}")
        return url
    _stop(process)
    raise TimeoutError(f"kev.serve was not ready after {SERVER_START_SECONDS} s")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)


class KevAPI(TypeSafeAPI):
    """Evaluate one message per request against a Kev server this provider starts."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        revision: str | None = None,
        kev_revision: str | None = None,
        quantization: str | None = None,
    ) -> None:
        # Not TypeSafeAPI's own initializer: it pins a hosted version name and reads
        # TYPESAFE_API_KEY, while a local Kev server takes any key.
        ModelAPI.__init__(
            self,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key or "local",
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        if base_url is not None:
            raise ValueError("The kev provider starts its own server and takes no base_url")
        if quantization not in (None, "fp8"):
            raise ValueError(f"Kev `quantization` must be fp8 when set; got {quantization!r}")
        run = f"{model_name}@{commit('kev', 'revision', revision)}"
        kev_commit = commit("kev", "kev_revision", kev_revision)
        self._metadata = {
            "served_model": run + (f"+{quantization}" if quantization else ""),
            "kev_revision": kev_commit,
        }
        self._client = self._build_client(serve(run, kev_commit, quantization))

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Issue one System One request for this conversation."""
        response = await self._client.system_one(
            state_from_messages(input), questions_from_messages(input), model=SERVED_NAME
        )
        output = model_output_from_response(response, model_name=SERVED_NAME)
        output.model = self.model_name
        output.metadata = {**(output.metadata or {}), **self._metadata}
        return output


@modelapi(name="kev")
def kev() -> type[ModelAPI]:
    """Register the Kev provider with Inspect."""
    return KevAPI
