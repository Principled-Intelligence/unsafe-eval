import asyncio

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.claude import (
    CLAUDE_SYSTEM_MESSAGE,
    claude_metadata_policy_prompt,
    claude_prompt,
)


def _state(metadata: dict[str, object] | None = None) -> TaskState:
    return TaskState(
        model="mockllm/model",
        sample_id="sample",
        epoch=1,
        input="Prompt {braces}",
        messages=[ChatMessageUser(content="Prompt {braces}")],
        metadata=metadata or {},
    )


async def _noop_generate(state: TaskState, **kwargs: object) -> TaskState:
    return state


def test_claude_prompt_wraps_the_content_in_the_policy_template() -> None:
    state = asyncio.run(claude_prompt("POLICY")(_state(), _noop_generate))

    assert state.messages[0].text == CLAUDE_SYSTEM_MESSAGE
    assert "POLICY" in state.messages[1].text
    assert "Prompt {braces}" in state.messages[1].text


def test_claude_metadata_policy_prompt_requires_a_policy() -> None:
    solver = claude_metadata_policy_prompt("Apply: {safety_policy}")

    with pytest.raises(ValueError, match="Claude sample metadata field 'safety_policy'"):
        asyncio.run(solver(_state({"safety_policy": None}), _noop_generate))


def test_claude_prompt_solvers_reject_an_empty_policy() -> None:
    with pytest.raises(ValueError, match="non-empty benchmark policy"):
        claude_prompt("")
    with pytest.raises(ValueError, match="non-empty template"):
        claude_metadata_policy_prompt("")
