"""Inspect model provider for ScopeGuard through its own SDK, `orbitals`.

The SDK builds ScopeGuard's prompt and parses its answer, so a run evaluates the SDK rather
than this package's rendering of the prompt. The completion is the JSON the protocol's
scorer parses for a `vllm/` run of the same model; like the other providers, this one knows
nothing about safety.

* Each run starts its own `vllm serve` on the pinned checkpoint with the configuration's
  flags and sends every sample as its own request through the SDK's `vllm-api` client. A
  run's latency is then measured as a `vllm/` run's is: one server per run, a cold prefix
  cache, and the time of each request.
* `guard: scope-guard` serves the `scope-guard-2601` protocol and `guard: scope-guard-v2`
  the `scope-guard-2609` one. The provider reads the benchmark prompt and the AI service
  description from the metadata the protocol's setup solvers attach to the user message,
  because the SDK takes them raw.
* `method: validate` generates the fields the protocol selects. `method: classify`,
  ScopeGuard v2 only, reads the class off the first answer token's log-probabilities and
  records the class probabilities.
* The SDK's default safety principles are turned off, so the benchmark's description
  reaches the model unchanged.
* `orbitals` is imported only when the model is constructed, because `protocols/_builtins`
  imports this module on every install.
"""

import atexit
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import version
from typing import Any

from huggingface_hub import snapshot_download
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from unsafe_eval.profiles.scope_guard import SCOPE_GUARD_REQUEST_METADATA
from unsafe_eval.profiles.scope_guard_2609 import SCOPE_GUARD_2609_REQUEST_METADATA
from unsafe_eval.providers._pins import commit

# Each guard's request metadata key and the protocol that attaches it.
GUARDS = {
    "scope-guard": (SCOPE_GUARD_REQUEST_METADATA, "scope-guard-2601"),
    "scope-guard-v2": (SCOPE_GUARD_2609_REQUEST_METADATA, "scope-guard-2609"),
}
METHODS = ("validate", "classify")

# The SDK raises ValueError both for bad inputs and for a model answer it cannot read; only
# these messages are about the answer.
_INVALID_ANSWER = (
    "was not in the top logprobs",
    "Failed to parse generated text",
    "Failed to validate generated text",
)

SERVER_START_SECONDS = 900


def request_from_messages(input: list[ChatMessage], guard: str) -> dict[str, Any]:
    """Read the raw prompt and service description the protocol attached."""
    key, protocol = GUARDS[guard]
    for message in reversed(input):
        if message.role == "user":
            request = (message.metadata or {}).get(key)
            if not isinstance(request, dict):
                raise ValueError(f"orbitals `guard: {guard}` runs only the {protocol} protocol")
            return request
    raise ValueError("The orbitals provider requires a user message to classify")


def model_output(
    answer: dict[str, Any] | None, usage: Any, *, model_name: str, metadata: dict[str, Any]
) -> ModelOutput:
    """Present the SDK's answer as an Inspect completion.

    `None` stands for an answer the SDK could not read, which becomes an empty, and so
    invalid, completion rather than a failed sample.
    """
    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="" if answer is None else json.dumps(answer), model=model_name
                ),
                stop_reason="stop",
            )
        ],
        usage=ModelUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        if usage is not None
        else ModelUsage(),
        metadata=metadata,
    )


def serve(path: str, server_args: dict[str, Any]) -> str:
    """Start `vllm serve` on a local checkpoint and return its URL once it is healthy.

    Each `server_args` entry becomes a flag: `max_model_len=30000` is passed as
    `--max-model-len 30000`, and `enable_prefix_caching=True` as `--enable-prefix-caching`.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    flags: list[str] = []
    for name, value in server_args.items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            flags += [f"--{flag}", str(value)]
    process = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", path]
        + ["--host", "127.0.0.1", "--port", str(port), *flags],
        # `vllm serve` would require VLLM_API_KEY, which Inspect's `vllm` provider reads from
        # `.env`, and the SDK's client sends no key.
        env={name: value for name, value in os.environ.items() if name != "VLLM_API_KEY"},
    )
    atexit.register(process.terminate)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited with code {process.returncode} before it was ready"
            )
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5):
                return url
        except OSError:
            time.sleep(2)
    process.terminate()
    raise TimeoutError(f"vllm serve was not ready after {SERVER_START_SECONDS} s")


class OrbitalsAPI(ModelAPI):
    """Ask ScopeGuard about one message per call with the `orbitals` SDK."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        revision: str | None = None,
        guard: str | None = None,
        method: str = "validate",
        **server_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        revision = commit("orbitals", "revision", revision)
        if guard not in GUARDS:
            raise ValueError(f"orbitals `guard` must be one of {list(GUARDS)}, got {guard!r}")
        if method not in METHODS:
            raise ValueError(f"orbitals `method` must be one of {list(METHODS)}, got {method!r}")
        if method == "classify" and guard != "scope-guard-v2":
            raise ValueError("orbitals `method: classify` requires `guard: scope-guard-v2`")
        # A server kept across runs would carry its prefix cache into the next one.
        if base_url:
            raise ValueError("orbitals starts its own vllm serve for each run; drop `base_url`")
        self._guard_name = guard
        self._v2 = guard == "scope-guard-v2"
        self._method = method
        self._guard, sdk = self._load(model_name, revision, self._v2, server_args)
        self._metadata = {"served_model": f"{model_name}@{revision}", **sdk}

    @staticmethod
    def _load(
        model_name: str, revision: str, v2: bool, server_args: dict[str, Any]
    ) -> tuple[Any, dict[str, str]]:
        """Download the pinned checkpoint, serve it, and connect the SDK's client to it."""
        try:
            if v2:
                from orbitals.scope_guard_v2 import AsyncScopeGuardV2 as Guard
                from orbitals.scope_guard_v2.prompting import SYSTEM_PROMPT
            else:
                from orbitals.scope_guard import AsyncScopeGuard as Guard
                from orbitals.scope_guard.prompting import SYSTEM_PROMPT
        except ImportError as error:
            raise ImportError(
                "The orbitals provider requires the `orbitals` extra: "
                "uv sync --extra orbitals (or pip install 'unsafe-eval[orbitals]')"
            ) from error

        path = snapshot_download(model_name, revision=revision)
        guard = Guard(
            backend="vllm-api",
            model=path,
            vllm_serving_url=serve(path, server_args),
            include_default_safety_principles=False,
            count_system_prompt_in_usage=True,
        )
        return guard, {
            "orbitals_version": version("orbitals"),
            "orbitals_system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        }

    async def _ask(self, request: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        """Return the scorer's fields, the token usage, and any extra output metadata."""
        conversation = request["prompt"]
        description = request["ai_service_description"]
        if self._method == "classify":
            if request["output_fields"] != ["scope_class"]:
                raise ValueError(
                    "orbitals `method: classify` returns only the class: set the protocol "
                    "option output_fields: [scope_class]"
                )
            classification = await self._guard.classify(
                conversation, ai_service_description=description, resolve_predefined=False
            )
            return (
                {"scope_class": classification.scope_class.value},
                classification.usage,
                {
                    "probabilities": dict(classification.probabilities),
                    "confidence": classification.confidence,
                },
            )
        if self._v2:
            fields = request["output_fields"]
            output = await self._guard.validate(
                conversation,
                ai_service_description=description,
                output_fields=fields,
                resolve_predefined=False,
            )
        else:
            fields = (
                ["evidences", "scope_class"] if request["include_evidence"] else ["scope_class"]
            )
            output = await self._guard.validate(
                conversation,
                ai_service_description=description,
                skip_evidences=not request["include_evidence"],
            )
        answer = output.model_dump(mode="json", include=set(fields))
        return answer, output.usage, {}

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Ask ScopeGuard about the conversation's last user message."""
        request = request_from_messages(input, self._guard_name)
        started = time.monotonic()
        try:
            answer, usage, details = await self._ask(request)
        except ValueError as error:
            if not any(marker in str(error) for marker in _INVALID_ANSWER):
                raise
            answer, usage, details = None, None, {}
        elapsed = time.monotonic() - started
        output = model_output(
            answer, usage, model_name=self.model_name, metadata={**self._metadata, **details}
        )
        return output, ModelCall.create(
            request=request, response={"completion": output.completion}, time=elapsed
        )


@modelapi(name="orbitals")
def orbitals() -> type[ModelAPI]:
    """Register the orbitals provider with Inspect."""
    return OrbitalsAPI
