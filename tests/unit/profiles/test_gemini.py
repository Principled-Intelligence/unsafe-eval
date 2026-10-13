import asyncio

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.gemini import (
    GEMINI_SYSTEM_MESSAGE,
    gemini_metadata_policy_prompt,
    gemini_prompt,
)


def _state() -> TaskState:
    return TaskState(
        model="mockllm/model",
        sample_id="sample",
        epoch=1,
        input="Prompt {braces}",
        messages=[ChatMessageUser(content="Prompt {braces}")],
    )


async def _noop_generate(state: TaskState, **kwargs: object) -> TaskState:
    return state


def test_gemini_prompt_wraps_the_content_in_the_policy_template() -> None:
    state = asyncio.run(gemini_prompt("POLICY")(_state(), _noop_generate))

    assert state.messages[0].text == GEMINI_SYSTEM_MESSAGE
    assert "POLICY" in state.messages[1].text
    assert "Prompt {braces}" in state.messages[1].text


def test_gemini_prompt_solvers_reject_an_empty_policy() -> None:
    with pytest.raises(ValueError, match="non-empty benchmark policy"):
        gemini_prompt("")
    with pytest.raises(ValueError, match="non-empty template"):
        gemini_metadata_policy_prompt("")
