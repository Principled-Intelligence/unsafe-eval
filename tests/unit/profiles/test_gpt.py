import asyncio

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.gpt import (
    GPT_FORMAT,
    GPT_SYSTEM_MESSAGE,
    gpt_metadata_policy_prompt,
    gpt_prompt,
)


async def _noop_generate(state: TaskState, **kwargs: object) -> TaskState:
    return state


def test_prompt_carries_the_policy_and_the_content() -> None:
    state = TaskState(
        model="mockllm/model",
        sample_id="sample",
        epoch=1,
        input="prompt-sentinel",
        messages=[ChatMessageUser(content="prompt-sentinel")],
    )

    state = asyncio.run(gpt_prompt("policy-sentinel")(state, _noop_generate))

    assert state.messages[0].text == GPT_SYSTEM_MESSAGE
    assert state.messages[1].text == (
        "<policy>\npolicy-sentinel\n</policy>\n\n<content>\nprompt-sentinel\n</content>"
    )


def test_prompt_solvers_reject_an_empty_policy() -> None:
    with pytest.raises(ValueError, match="non-empty benchmark policy"):
        gpt_prompt("")
    with pytest.raises(ValueError, match="non-empty template"):
        gpt_metadata_policy_prompt("")


def test_system_message_defines_no_taxonomy_and_defers_to_the_policy() -> None:
    # The frame must not compete with the injected policy: each benchmark's gold
    # labels follow its own taxonomy, so a general harm definition here would be
    # scored against eight different ones at once.
    assert "sole definition of `unsafe`" in GPT_SYSTEM_MESSAGE
    assert "not widen or narrow the policy" in GPT_SYSTEM_MESSAGE
    # The item under judgement is "content", never "a request": some benchmarks
    # classify arbitrary text, for which "would fulfilling this cause harm" has no
    # meaning. The one permitted use of the word is the output contract.
    lowered = GPT_SYSTEM_MESSAGE.lower()
    assert "one item of user content" in lowered
    assert "fulfilling" not in lowered
    assert lowered.count("request") == 1
    assert "the requested json object" in lowered


def test_non_sequence_selections_are_rejected() -> None:
    with pytest.raises(ValueError, match="output_fields must be a list"):
        GPT_FORMAT.normalize_selection({"user_safety": True})


def test_unknown_output_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown output field"):
        GPT_FORMAT.normalize_selection(["user_safety", "scope_class"])


def test_response_schema_constrains_the_label_to_the_native_vocabulary() -> None:
    selection = GPT_FORMAT.canonical_order
    schema = GPT_FORMAT.inspect_response_schema(selection).model_dump(exclude_none=True)

    assert schema["properties"]["user_safety"]["enum"] == ["safe", "unsafe"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["reasoning", "user_safety"]
