import asyncio

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.nemotron import nemotron_inline_policy_generate

TEMPLATE = "Policy:\n{safety_policy}"


def _state(metadata: dict[str, object]) -> TaskState:
    return TaskState(
        model="mockllm/model",
        sample_id="sample",
        epoch=1,
        input="Prompt",
        messages=[ChatMessageUser(content="Prompt")],
        metadata=metadata,
    )


def _recording_generate(seen: dict[str, object]):
    async def generate(state: TaskState, **kwargs: object) -> TaskState:
        seen["text"] = state.user_prompt.text
        seen["kwargs"] = kwargs
        state.output = ModelOutput.from_content(model="mockllm/model", content="User Safety: safe")
        return state

    return generate


def test_inline_policy_requires_a_non_empty_policy() -> None:
    solve = nemotron_inline_policy_generate(custom_policy_template=TEMPLATE)

    with pytest.raises(ValueError, match="'safety_policy' must be a non-empty string"):
        asyncio.run(solve(_state({}), _recording_generate({})))
