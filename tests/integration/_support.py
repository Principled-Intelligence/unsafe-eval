"""Shared model and evaluation harness for mocked integration tests."""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    Logprob,
    Logprobs,
    ModelOutput,
    TopLogprob,
    get_model,
)

PROTOCOLS = {
    "shieldstral": "shieldstral-1.0-3b",
    "scope_guard": "scope-guard-2601",
    "nemotron": "nemotron-3.5-content-safety",
    # The same protocol over OpenRouter, which forwards no chat-template arguments.
    "nemotron_openrouter": "nemotron-3.5-content-safety",
    "polyguard": "polyguard-qwen",
    "qwen3_guard": "qwen3-guard-gen",
    "gpt": "gpt",
    # Claude over Anthropic's API, with the gpt format.
    "claude": "claude",
    # Gemini over Google's API.
    "gemini": "gemini",
}

# Options a profile needs to resolve. Only the routed protocols have any.
PROFILE_OPTIONS: dict[str, dict[str, object]] = {
    "nemotron": {"route": "vllm"},
    "nemotron_openrouter": {"route": "openrouter", "request_categories": False},
    "gpt": {"route": "openrouter"},
    "claude": {"route": "anthropic"},
    "gemini": {"route": "google"},
}

# Options a profile adds on a policy-conditioned benchmark. OpenRouter cannot reach
# Nemotron's `custom_policy` template argument, so the policy goes in the user turn.
POLICY_PROFILE_OPTIONS: dict[str, dict[str, object]] = {
    "nemotron_openrouter": {"inline_policy": True},
}


def task_with_dataset(
    *,
    module_name: str,
    loader_name: str,
    task_name: str,
    samples: list[Sample],
    protocol: str,
    monkeypatch: pytest.MonkeyPatch,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Build a public task with its network loader replaced by local samples."""
    benchmark = import_module(f"unsafe_eval.benchmarks.{module_name}")
    monkeypatch.setattr(
        benchmark,
        loader_name,
        lambda _revision: MemoryDataset(samples),
    )
    task_factory = getattr(benchmark, task_name)
    task: Task = task_factory(protocol=protocol, protocol_options=protocol_options)
    return task


def configure_test_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / f"{profile}-xdg-data"))
    monkeypatch.setenv("INSPECT_TRACE_FILE", str(tmp_path / f"{profile}-inspect-trace.log"))


def _output_for(profile: str, messages: list[ChatMessage]) -> ModelOutput:
    prompt = messages[-1].text
    unsafe = "unsafe-example" in prompt
    if profile == "shieldstral":
        verdict = "yes" if unsafe else "no"
        other = "no" if unsafe else "yes"
        return ModelOutput(
            model="mockllm/model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=verdict),
                    stop_reason="stop",
                    logprobs=Logprobs(
                        content=[
                            Logprob(
                                token=verdict,
                                logprob=-0.1,
                                top_logprobs=[
                                    TopLogprob(token=verdict, logprob=-0.1),
                                    TopLogprob(token=other, logprob=-2.1),
                                ],
                            )
                        ]
                    ),
                )
            ],
        )
    if profile == "scope_guard":
        scope_class = "Restricted" if unsafe else "Directly Supported"
        completion = f'{{"scope_class":"{scope_class}","evidences":[]}}'
    elif profile in ("nemotron", "nemotron_openrouter"):
        completion = f"User Safety: {'unsafe' if unsafe else 'safe'}"
    elif profile == "polyguard":
        completion = (
            f"Harmful request: {'yes' if unsafe else 'no'}\nRequest safety violations: None"
        )
    elif profile in ("gpt", "claude", "gemini"):
        # One response shape for all three, because they share the same schema.
        label = "unsafe" if unsafe else "safe"
        completion = f'{{"reasoning":"mock","user_safety":"{label}"}}'
    else:
        completion = f"Safety: {'Unsafe' if unsafe else 'Safe'}\nCategories: None"
    return ModelOutput.from_content(model="mockllm/model", content=completion)


def run_task(
    task: Task,
    *,
    profile: str,
    tmp_path: Path,
    observed: dict[str, Any],
) -> Any:
    def model_output(
        messages: list[ChatMessage],
        _tools: list[Any],
        _tool_choice: Any,
        config: Any,
    ) -> ModelOutput:
        observed["messages"] = messages
        observed["config"] = config
        return _output_for(profile, messages)

    model = get_model("mockllm/model", custom_outputs=model_output)
    logs = eval(
        task,
        model=model,
        log_dir=str(tmp_path),
        display="none",
        ctl_server=False,
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].results is not None
    return logs[0].results
