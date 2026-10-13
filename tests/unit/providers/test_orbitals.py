"""Translation tests for the orbitals provider.

Stub guards stand in for the `orbitals` SDK and stub processes for `vllm serve`, so the
offline suite never imports vllm.
"""

import asyncio
import contextlib
import sys
import types
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelName
from inspect_ai.solver import Solver, TaskState
from pydantic import BaseModel

from unsafe_eval.profiles.scope_guard import scope_guard_prompt
from unsafe_eval.profiles.scope_guard_2609 import (
    scope_guard_2609_metadata_policy_prompt,
    scope_guard_2609_prompt,
)
from unsafe_eval.providers.orbitals import OrbitalsAPI, serve
from unsafe_eval.scorers.scope_guard_2609_safety import parse_scope_guard_2609_completion
from unsafe_eval.scorers.scope_guard_safety import parse_scope_guard_completion

MODEL = "principled-intelligence/scope-guard-v2-4B-q-2609"
REVISION = "0110d5d0f50c537513096f597d46ff4a1ee18377"
SDK = {"orbitals_version": "0.6.1", "orbitals_system_prompt_sha256": "prompt-digest"}
CLASS_ONLY = ("scope_class",)


class Usage(BaseModel):
    prompt_tokens: int = 40
    completion_tokens: int = 12
    total_tokens: int = 52


class Output(BaseModel):
    """The SDK's validate answer, with every field filled whatever was requested."""

    evidences: list[str] | None = ["evidence"]
    reasoning: str | None = "reasoning"
    scope_class: str = "Restricted"
    suggested_response: str | None = "reply"
    model: str = "/snapshot"
    usage: Usage | None = Usage()


CLASSIFICATION = types.SimpleNamespace(
    scope_class=types.SimpleNamespace(value="Restricted"),
    probabilities={"Restricted": 0.9, "Directly Supported": 0.1},
    confidence=0.9,
    usage=Usage(completion_tokens=1, total_tokens=41),
)


class StubGuard:
    """The SDK's async guard over `vllm serve`."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    async def _answer(self, method: str, conversation: Any, kwargs: dict[str, Any]) -> Any:
        self.calls.append((method, conversation, kwargs))
        if self.error is not None:
            raise self.error
        return CLASSIFICATION if method == "classify" else Output()

    async def validate(self, conversation: Any, **kwargs: Any) -> Any:
        return await self._answer("validate", conversation, kwargs)

    async def classify(self, conversation: Any, **kwargs: Any) -> Any:
        return await self._answer("classify", conversation, kwargs)


def api(
    guard: StubGuard | None = None, *, guard_name: str = "scope-guard-v2", **model_args: Any
) -> OrbitalsAPI:
    args: dict[str, Any] = {"revision": REVISION, "guard": guard_name, **model_args}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(OrbitalsAPI, "_load", staticmethod(lambda *_: (guard or StubGuard(), SDK)))
        return OrbitalsAPI(MODEL, **args)


def prompted(setup: Solver, metadata: dict[str, Any] | None = None) -> list[Any]:
    state = TaskState(
        model=ModelName("orbitals/model"),
        sample_id=1,
        epoch=1,
        input="prompt-sentinel",
        messages=[ChatMessageUser(content="prompt-sentinel")],
        metadata=metadata,
    )

    async def generate(state: TaskState, **_: Any) -> TaskState:
        return state

    return asyncio.run(setup(state, generate)).messages


def generate(model: OrbitalsAPI, messages: list[Any]) -> Any:
    output, call = asyncio.run(model.generate(messages, [], "none", GenerateConfig()))
    assert call.time is not None
    return output


@pytest.mark.parametrize(
    ("setup", "metadata", "description"),
    [
        (scope_guard_2609_prompt("description-sentinel"), None, "description-sentinel"),
        (
            scope_guard_2609_metadata_policy_prompt("policy: {safety_policy}"),
            {"safety_policy": "no-sentinel"},
            "policy: no-sentinel",
        ),
    ],
)
def test_v2_validate_answers_with_exactly_the_selected_fields(
    setup: Solver, metadata: dict[str, Any] | None, description: str
) -> None:
    guard = StubGuard()

    output = generate(api(guard), prompted(setup, metadata))

    assert guard.calls == [
        (
            "validate",
            "prompt-sentinel",
            {
                "ai_service_description": description,
                "output_fields": ["reasoning", "scope_class"],
                "resolve_predefined": False,
            },
        )
    ]
    # The 2609 parser rejects any key outside the selection.
    selection = ("reasoning", "scope_class")
    assert parse_scope_guard_2609_completion(output.completion, selection).scope_class == (
        "Restricted"
    )
    assert output.usage.input_tokens == 40
    assert output.metadata == {"served_model": f"{MODEL}@{REVISION}", **SDK}


@pytest.mark.parametrize(("include_evidence", "evidences"), [(True, ["evidence"]), (False, None)])
def test_v1_validate_reports_evidence_only_when_the_protocol_asks(
    include_evidence: bool, evidences: list[str] | None
) -> None:
    guard = StubGuard()
    setup = scope_guard_prompt("description", include_evidence=include_evidence)

    output = generate(api(guard, guard_name="scope-guard"), prompted(setup))

    assert guard.calls == [
        (
            "validate",
            "prompt-sentinel",
            {"ai_service_description": "description", "skip_evidences": not include_evidence},
        )
    ]
    result = parse_scope_guard_completion(output.completion)
    assert (result.scope_class, result.evidences) == ("Restricted", evidences)


def test_classify_reads_the_class_and_records_its_probabilities() -> None:
    guard = StubGuard()

    output = generate(
        api(guard, method="classify"), prompted(scope_guard_2609_prompt("description", CLASS_ONLY))
    )

    assert guard.calls == [
        (
            "classify",
            "prompt-sentinel",
            {"ai_service_description": "description", "resolve_predefined": False},
        )
    ]
    assert output.completion == '{"scope_class": "Restricted"}'
    assert output.metadata["probabilities"] == {"Restricted": 0.9, "Directly Supported": 0.1}
    assert output.metadata["confidence"] == 0.9


def test_an_answer_the_sdk_cannot_read_is_an_invalid_completion() -> None:
    error = ValueError("Failed to validate generated text: scope_class missing")
    messages = prompted(scope_guard_2609_prompt("description", CLASS_ONLY))

    output = generate(api(StubGuard(error=error)), messages)

    assert parse_scope_guard_2609_completion(output.completion, CLASS_ONLY).scope_class == (
        "invalid"
    )


def test_any_other_sdk_error_fails_the_sample() -> None:
    messages = prompted(scope_guard_2609_prompt("description", CLASS_ONLY))

    with pytest.raises(ValueError, match="bad conversation"):
        generate(api(StubGuard(error=ValueError("bad conversation"))), messages)


@pytest.mark.parametrize(
    ("model_args", "messages", "message"),
    [
        ({}, [ChatMessageUser(content="prompt")], "only the scope-guard-2609 protocol"),
        (
            {"guard_name": "scope-guard"},
            prompted(scope_guard_2609_prompt("description")),
            "only the scope-guard-2601 protocol",
        ),
        (
            {"method": "classify"},
            prompted(scope_guard_2609_prompt("description")),
            "output_fields: \\[scope_class\\]",
        ),
    ],
)
def test_a_request_the_guard_cannot_answer_is_refused(
    model_args: dict[str, Any], messages: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate(api(**model_args), messages)


@pytest.mark.parametrize(
    ("model_args", "message"),
    [
        ({"revision": "main"}, "40-character commit hash"),
        ({"guard_name": "scope-guard-pro"}, "`guard`"),
        ({"method": "score"}, "`method`"),
        ({"guard_name": "scope-guard", "method": "classify"}, "requires `guard: scope-guard-v2`"),
        ({"base_url": "http://127.0.0.1:8000"}, "`base_url`"),
    ],
)
def test_an_unpinned_unknown_or_shared_server_configuration_is_refused(
    model_args: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        api(**model_args)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    recorded: dict[str, list[Any]] = {"downloads": [], "served": [], "guards": []}

    def snapshot_download(repo_id: str, revision: str) -> str:
        recorded["downloads"].append((repo_id, revision))
        return "/snapshot"

    def serve(path: str, server_args: dict[str, Any]) -> str:
        recorded["served"].append((path, server_args))
        return "http://127.0.0.1:1234"

    def guard_class(name: str) -> Any:
        def construct(**kwargs: Any) -> StubGuard:
            recorded["guards"].append((name, kwargs))
            return StubGuard()

        return construct

    monkeypatch.setattr("unsafe_eval.providers.orbitals.snapshot_download", snapshot_download)
    monkeypatch.setattr("unsafe_eval.providers.orbitals.serve", serve)
    monkeypatch.setattr("unsafe_eval.providers.orbitals.version", lambda _: "0.6.1")
    for package, name in (
        ("scope_guard", "AsyncScopeGuard"),
        ("scope_guard_v2", "AsyncScopeGuardV2"),
    ):
        module = types.SimpleNamespace(**{name: guard_class(name)})
        monkeypatch.setitem(sys.modules, f"orbitals.{package}", module)
        monkeypatch.setitem(
            sys.modules, f"orbitals.{package}.prompting", types.SimpleNamespace(SYSTEM_PROMPT="s")
        )
    return recorded


@pytest.mark.parametrize(
    ("guard", "sdk_class"),
    [("scope-guard", "AsyncScopeGuard"), ("scope-guard-v2", "AsyncScopeGuardV2")],
)
def test_each_run_serves_the_pinned_download_to_the_sdk_client(
    sdk: dict[str, list[Any]], guard: str, sdk_class: str
) -> None:
    model = OrbitalsAPI(MODEL, revision=REVISION, guard=guard, max_model_len=8192)

    assert sdk["downloads"] == [(MODEL, REVISION)]
    assert sdk["served"] == [("/snapshot", {"max_model_len": 8192})]
    assert sdk["guards"] == [
        (
            sdk_class,
            {
                "backend": "vllm-api",
                "model": "/snapshot",
                "vllm_serving_url": "http://127.0.0.1:1234",
                "include_default_safety_principles": False,
                "count_system_prompt_in_usage": True,
            },
        )
    ]
    assert model._metadata["orbitals_version"] == "0.6.1"


class StubProcess:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        pass


def test_serve_returns_the_url_once_vllm_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    health: list[str] = []

    def popen(command: list[str], env: dict[str, str]) -> StubProcess:
        assert "VLLM_API_KEY" not in env
        commands.append(command)
        return StubProcess(None)

    def urlopen(url: str, timeout: float) -> Any:
        health.append(url)
        if len(health) == 1:
            raise ConnectionRefusedError
        return contextlib.nullcontext()

    monkeypatch.setattr("unsafe_eval.providers.orbitals.subprocess.Popen", popen)
    monkeypatch.setattr("unsafe_eval.providers.orbitals.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("unsafe_eval.providers.orbitals.time.sleep", lambda _: None)
    monkeypatch.setenv("VLLM_API_KEY", "key-for-inspect-vllm")
    monkeypatch.setattr("unsafe_eval.providers.orbitals.atexit.register", lambda _: None)

    url = serve(
        "/snapshot",
        {"max_model_len": 30000, "enable_prefix_caching": True, "enforce_eager": False},
    )

    (command,) = commands
    port = command[command.index("--port") + 1]
    assert url == f"http://127.0.0.1:{port}"
    assert health == [f"{url}/health"] * 2
    assert command[-4:] == [
        "--max-model-len",
        "30000",
        "--enable-prefix-caching",
        "--no-enforce-eager",
    ]
    assert command[3:5] == ["serve", "/snapshot"]


def test_serve_fails_when_vllm_exits_before_it_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unsafe_eval.providers.orbitals.subprocess.Popen", lambda *_, **__: StubProcess(1)
    )
    monkeypatch.setattr("unsafe_eval.providers.orbitals.atexit.register", lambda _: None)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        serve("/snapshot", {})


def test_a_missing_orbitals_extra_fails_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[str] = []
    monkeypatch.setattr(
        "unsafe_eval.providers.orbitals.snapshot_download",
        lambda repo_id, **_: downloads.append(repo_id),
    )
    monkeypatch.setitem(sys.modules, "orbitals.scope_guard_v2", None)

    with pytest.raises(ImportError, match="uv sync --extra orbitals"):
        OrbitalsAPI(MODEL, revision=REVISION, guard="scope-guard-v2")

    assert downloads == []
