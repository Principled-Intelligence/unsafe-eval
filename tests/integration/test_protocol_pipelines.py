from pathlib import Path
from typing import Any

import pytest
from inspect_ai.dataset import Sample

from tests.integration._support import (
    POLICY_PROFILE_OPTIONS,
    PROFILE_OPTIONS,
    PROTOCOLS,
    configure_test_environment,
    run_task,
    task_with_dataset,
)
from unsafe_eval.profiles.claude import CLAUDE_SYSTEM_MESSAGE
from unsafe_eval.profiles.gemini import GEMINI_SYSTEM_MESSAGE
from unsafe_eval.profiles.gpt import GPT_SYSTEM_MESSAGE
from unsafe_eval.profiles.polyguard import POLYGUARD_SYSTEM_MESSAGE
from unsafe_eval.profiles.scope_guard import SCOPE_GUARD_SYSTEM_MESSAGE
from unsafe_eval.profiles.shieldstral import SHIELDSTRAL_SYSTEM_MESSAGE


@pytest.mark.parametrize(
    "profile",
    [
        "shieldstral",
        "scope_guard",
        "nemotron",
        "polyguard",
        "qwen3_guard",
        "gpt",
        "claude",
        "gemini",
    ],
)
def test_protocol_pipeline_applies_native_messages_and_generation_config(
    profile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_test_environment(tmp_path, monkeypatch, profile)
    task = task_with_dataset(
        module_name="simple_safety_tests",
        loader_name="load_simple_safety_tests",
        task_name="simple_safety_tests",
        samples=[
            Sample(
                input="unsafe-example",
                target="yes",
                metadata={"harm_area": "Violence", "category": "Instructions"},
            )
        ],
        protocol=PROTOCOLS[profile],
        protocol_options=PROFILE_OPTIONS.get(profile),
        monkeypatch=monkeypatch,
    )
    observed: dict[str, Any] = {}

    results = run_task(task, profile=profile, tmp_path=tmp_path, observed=observed)

    scores = {score.name: score for score in results.scores}
    assert scores["label"].metrics["unsafe_eval/unsafe_recall"].value == 1.0
    assert scores["verdict"].metrics.values()
    messages = observed["messages"]
    config = observed["config"]
    if profile == "shieldstral":
        assert messages[0].text == SHIELDSTRAL_SYSTEM_MESSAGE
        assert "unsafe-example" in messages[1].text
        assert config.max_tokens == 1
        assert config.logprobs is True
        assert scores["unsafe_probability"].metrics["mean"].value > 0.5
    elif profile == "scope_guard":
        assert messages[0].text == SCOPE_GUARD_SYSTEM_MESSAGE
        assert "unsafe-example" in messages[1].text
        assert config.max_tokens == 3000
        assert config.response_schema.name == "scope_guard_response"
    elif profile == "nemotron":
        assert len(messages) == 1
        assert messages[0].text == "unsafe-example"
        assert config.max_tokens == 100
        assert config.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    elif profile == "polyguard":
        assert messages[0].text == POLYGUARD_SYSTEM_MESSAGE
        assert messages[1].text == "Human user:\nunsafe-example\n\nAI assistant:"
        assert config.max_tokens == 100
    elif profile == "gpt":
        assert messages[0].text == GPT_SYSTEM_MESSAGE
        assert "unsafe-example" in messages[1].text
        assert config.max_tokens == 16384
        assert config.response_schema.name == "gpt_response"
        # A hosted API treats the schema as advisory unless `strict` is set.
        assert config.response_schema.strict is True
        assert config.reasoning_effort == "low"
        assert config.seed == 0
        # The GPT-5.6 reasoning family rejects a sampling temperature, so this
        # protocol pins `seed` in its place. Every other protocol sets 0.0 here.
        assert config.temperature is None
    elif profile == "claude":
        # Claude's own prompt, with no seed: Anthropic's API takes none.
        assert messages[0].text == CLAUDE_SYSTEM_MESSAGE
        assert "unsafe-example" in messages[1].text
        assert config.max_tokens == 8192
        assert config.response_schema.name == "claude_response"
        assert config.response_schema.strict is True
        assert config.reasoning_effort == "low"
        assert config.seed is None
        assert config.temperature is None
        assert config.cache_prompt is False
    elif profile == "gemini":
        # Gemini's own prompt and no temperature. No seed on the google route:
        # Inspect's google provider never sends one.
        assert messages[0].text == GEMINI_SYSTEM_MESSAGE
        assert "unsafe-example" in messages[1].text
        assert config.max_tokens == 32768
        assert config.response_schema.name == "gemini_response"
        assert config.response_schema.strict is True
        assert config.reasoning_effort == "low"
        assert config.seed is None
        assert config.temperature is None
    else:
        assert len(messages) == 1
        assert messages[0].text == "unsafe-example"
        assert config.max_tokens == 128


@pytest.mark.parametrize(
    "profile",
    [
        "shieldstral",
        "scope_guard",
        "nemotron",
        "nemotron_openrouter",
        "gpt",
        "claude",
        "gemini",
    ],
)
def test_policy_conditioned_pipeline_populates_native_protocol(
    profile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_test_environment(tmp_path, monkeypatch, f"policy-{profile}")
    policy = "Permit the scenario-specific activity but prohibit its unsafe variant."
    task = task_with_dataset(
        module_name="cosapien_prompt_safety",
        loader_name="load_cosapien_prompt_safety",
        task_name="cosapien_prompt_safety",
        samples=[
            Sample(
                input="unsafe-example",
                target="yes",
                metadata={
                    "safety_policy": policy,
                    "prompt_type": "partial",
                    "scenario": "game_development",
                },
            ),
            Sample(
                input="safe-example",
                target="no",
                metadata={
                    "safety_policy": policy,
                    "prompt_type": "allowed",
                    "scenario": "game_development",
                },
            ),
        ],
        protocol=PROTOCOLS[profile],
        protocol_options={
            **PROFILE_OPTIONS.get(profile, {}),
            **POLICY_PROFILE_OPTIONS.get(profile, {}),
        },
        monkeypatch=monkeypatch,
    )
    observed: dict[str, Any] = {}

    results = run_task(task, profile=profile, tmp_path=tmp_path, observed=observed)

    metrics = {score.name: score for score in results.scores}["label"].metrics
    assert metrics["unsafe_eval/balanced_accuracy"].value == 1.0
    grouped_values = [
        metric.value for name, metric in metrics.items() if "scenario_game_development" in name
    ]
    assert grouped_values == [1.0]
    messages = observed["messages"]
    if profile in ("shieldstral", "scope_guard", "gpt", "claude", "gemini"):
        # The sample's own policy reaches the model in the user turn.
        assert policy in messages[1].text
    elif profile == "nemotron_openrouter":
        # Inlined ahead of the prompt, since the template argument is out of reach.
        assert len(messages) == 1
        assert policy in messages[0].text
        assert messages[0].text.endswith("### Prompt to classify\nsafe-example")
        assert "chat_template_kwargs" not in observed["config"].extra_body
    else:
        assert messages[0].text == "safe-example"
        config = observed["config"]
        assert policy in config.extra_body["chat_template_kwargs"]["custom_policy"]
