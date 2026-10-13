import pytest
from inspect_ai.model import GenerateConfig
from inspect_ai.model._providers.google import GoogleGenAIAPI

from unsafe_eval.profiles.gemini import GEMINI_PROMPT_RESOURCE_HASHES
from unsafe_eval.protocols import resolve_protocol
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols.gemini import REASONING_EFFORTS


def test_gemini_records_its_options_and_the_digests_of_its_own_prompt_resources() -> None:
    protocol = resolve_protocol(
        "gemini",
        benchmark="aegis-2.0",
        options={"route": "openrouter", "reasoning_effort": "high"},
    )
    _, instruction_hashes = adaptation_resource("aegis-2.0", "gemini", "instruction.txt")

    assert protocol.metadata["protocol"] == "gemini"
    assert protocol.metadata["protocol_options"] == {
        "route": "openrouter",
        "output_fields": ["reasoning", "user_safety"],
        "reasoning_effort": "high",
        "validate_prompt_resource_hashes": True,
    }
    assert protocol.metadata["prompt_resource_hashes"] == {
        **GEMINI_PROMPT_RESOURCE_HASHES,
        **instruction_hashes,
    }


def test_inspect_sends_each_effort_to_gemini_as_the_same_thinking_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenRouter maps low, medium and high to Gemini's thinking level of the same name.
    # The google route matches it only while Inspect sends a level, not a budget.
    # Inspect treats every Gemini 3 model name alike here, so one configured model
    # stands for all of them.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    api = GoogleGenAIAPI(
        model_name="vertex/gemini-3.8-flash",
        base_url=None,
        api_key=None,
        config=GenerateConfig(),
    )

    offenders = []
    for effort in REASONING_EFFORTS:
        thinking = api.chat_thinking_config(GenerateConfig(reasoning_effort=effort))
        if (
            thinking is None
            or thinking.thinking_budget is not None
            or thinking.thinking_level is None
            or thinking.thinking_level.value != effort.upper()
        ):
            offenders.append(effort)

    assert not offenders
