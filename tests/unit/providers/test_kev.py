"""Tests for the Kev provider.

The server process, its HTTP endpoint and the SDK client are stubbed, so no test starts
Kev or reaches the network.
"""

import asyncio
import io
import json
import signal
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from typesafe_sdk import ChoiceAnswer, SystemOneResponse, Usage

from unsafe_eval.providers import kev as kev_module
from unsafe_eval.providers.kev import (
    FP8_LAUNCHER,
    KERNEL_PINS,
    SERVED_NAME,
    KevAPI,
    kev,
    serve,
    served_runs,
)

MODEL = "jaredpalmer/kev-4b"
REVISION = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
KEV_REVISION = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
RUN = f"{MODEL}@{REVISION}"
QUESTIONS = {"safety": {"type": "choice", "criteria": {"safe": None, "unsafe": None}}}


class StubProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.pid = 4242
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class Launch:
    """Records what `serve` starts and stops, and answers its readiness checks."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, runs: list[Any], process: StubProcess
    ) -> None:
        self.popen: dict[str, Any] = {}
        self.killed: list[tuple[int, int]] = []
        self._runs = iter(runs)
        monkeypatch.setattr(kev_module.subprocess, "Popen", self._popen(process))
        monkeypatch.setattr(kev_module.atexit, "register", lambda *args: None)
        monkeypatch.setattr(kev_module.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(
            kev_module.os, "killpg", lambda pid, sig: self.killed.append((pid, sig))
        )
        monkeypatch.setattr(kev_module, "served_runs", self._served_runs)

    def _popen(self, process: StubProcess) -> Any:
        def popen(command: list[str], **kwargs: Any) -> StubProcess:
            self.popen = {"command": command, **kwargs}
            return process

        return popen

    def _served_runs(self, url: str) -> set[str]:
        runs = next(self._runs)
        if isinstance(runs, Exception):
            raise runs
        return runs


def test_served_runs_reads_the_run_of_every_model_card(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"models": [{"name": "kev-latest", "run": RUN}, {"name": "jev-latest", "run": RUN}]}
    monkeypatch.setattr(
        kev_module.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(json.dumps(body).encode()),
    )

    assert served_runs("http://127.0.0.1:1") == {RUN}


def test_serve_waits_until_the_server_reports_the_pinned_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEV_API_KEY", "secret")
    launch = Launch(monkeypatch, [ConnectionRefusedError(), {RUN}], StubProcess())

    url = serve(RUN, KEV_REVISION)

    command = launch.popen["command"]
    assert url.startswith("http://127.0.0.1:")
    assert f"kev[serve] @ git+https://github.com/jaredpalmer/kev@{KEV_REVISION}" in command
    assert all(command[command.index(pin) - 1] == "--with" for pin in KERNEL_PINS)
    assert command[-6:-2] == ["--run", RUN, "--host", "127.0.0.1"]
    assert command[-9:-6] == ["python", "-m", "kev.serve"]
    assert "KEV_API_KEY" not in launch.popen["env"]
    assert launch.popen["start_new_session"]
    assert not launch.killed


def test_fp8_serves_through_the_launcher_without_fused_kernels_or_cuda_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = Launch(monkeypatch, [{RUN}], StubProcess())

    serve(RUN, KEV_REVISION, "fp8")

    command = launch.popen["command"]
    assert command[-9:-6] == ["python", "-I", str(FP8_LAUNCHER)]
    assert FP8_LAUNCHER.is_file()
    assert launch.popen["env"]["KEV_FUSED"] == "0"
    assert launch.popen["env"]["KEV_CUDA_GRAPHS"] == "0"


def test_a_server_serving_another_checkpoint_is_stopped_and_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = Launch(monkeypatch, [{f"{MODEL}@main"}], StubProcess())

    with pytest.raises(ValueError, match="instead of the pinned"):
        serve(RUN, KEV_REVISION)
    assert launch.killed == [(4242, signal.SIGTERM)]


def test_a_server_that_exits_before_it_is_ready_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Launch(monkeypatch, [], StubProcess(returncode=1))

    with pytest.raises(RuntimeError, match="exited with code 1"):
        serve(RUN, KEV_REVISION)


def test_a_server_that_never_gets_ready_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = Launch(monkeypatch, [], StubProcess())
    monkeypatch.setattr(kev_module, "SERVER_START_SECONDS", 0)

    with pytest.raises(TimeoutError):
        serve(RUN, KEV_REVISION)
    assert launch.killed == [(4242, signal.SIGTERM)]


class StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, state: Any, questions: Any, *, model: str | None = None) -> Any:
        self.calls.append({"state": state, "questions": questions, "model": model})
        return SystemOneResponse(
            model=SERVED_NAME,
            answers={
                "safety": ChoiceAnswer(
                    choice="unsafe", confidence=0.8, probabilities={"safe": 0.2, "unsafe": 0.8}
                )
            },
            usage=Usage(input_tokens=40, output_tokens=1),
        )


def stub_server(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[StubClient, list[tuple[str, str, str | None]]]:
    client = StubClient()
    started: list[tuple[str, str, str | None]] = []

    def fake_serve(run: str, kev_revision: str, quantization: str | None = None) -> str:
        started.append((run, kev_revision, quantization))
        return "http://127.0.0.1:1"

    monkeypatch.setattr(kev_module, "serve", fake_serve)
    monkeypatch.setattr(KevAPI, "_build_client", lambda self, base_url: client)
    return client, started


def test_the_provider_serves_the_pinned_run_and_records_it_on_every_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, started = stub_server(monkeypatch)
    api = KevAPI(MODEL, revision=REVISION, kev_revision=KEV_REVISION)
    state = {"message": "prompt-sentinel"}
    messages = [
        ChatMessageSystem(content=json.dumps(QUESTIONS)),
        ChatMessageUser(content="prompt-sentinel", metadata={"state": state}),
    ]

    output = asyncio.run(api.generate(messages, [], "none", GenerateConfig()))

    assert started == [(RUN, KEV_REVISION, None)]
    assert client.calls == [{"state": state, "questions": QUESTIONS, "model": SERVED_NAME}]
    assert output.completion == "unsafe"
    assert output.model == MODEL
    assert output.metadata is not None
    assert output.metadata["served_model"] == RUN
    assert output.metadata["kev_revision"] == KEV_REVISION
    assert output.metadata["probabilities"] == {"safe": 0.2, "unsafe": 0.8}


@pytest.mark.parametrize(
    "args",
    [
        {"kev_revision": KEV_REVISION},
        {"revision": "main", "kev_revision": KEV_REVISION},
        {"revision": REVISION},
        {"revision": REVISION, "kev_revision": "kev-1.0"},
    ],
)
def test_both_commits_must_be_pinned(monkeypatch: pytest.MonkeyPatch, args: dict[str, str]) -> None:
    stub_server(monkeypatch)

    with pytest.raises(ValueError, match="40-character commit hash"):
        KevAPI(MODEL, **args)


def test_fp8_is_passed_to_the_server_and_recorded_on_the_served_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, started = stub_server(monkeypatch)

    api = KevAPI(MODEL, revision=REVISION, kev_revision=KEV_REVISION, quantization="fp8")

    assert started == [(RUN, KEV_REVISION, "fp8")]
    assert api._metadata["served_model"] == f"{RUN}+fp8"


def test_quantization_other_than_fp8_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_server(monkeypatch)

    with pytest.raises(ValueError, match="must be fp8"):
        KevAPI(MODEL, revision=REVISION, kev_revision=KEV_REVISION, quantization="int4")


def test_a_base_url_is_refused_because_the_provider_starts_its_own_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_server(monkeypatch)

    with pytest.raises(ValueError, match="takes no base_url"):
        KevAPI(MODEL, base_url="http://elsewhere", revision=REVISION, kev_revision=KEV_REVISION)


def test_the_registered_factory_builds_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_server(monkeypatch)

    api = kev(MODEL, revision=REVISION, kev_revision=KEV_REVISION)

    assert isinstance(api, KevAPI)
    assert api.model_name == MODEL
