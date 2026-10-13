import asyncio
import json
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.jev import jev_policy_prompt, jev_prompt

# The questions carry JSON braces, which is exactly what must survive delivery.
QUESTIONS = json.dumps(
    {
        "safety": {
            "type": "choice",
            "instructions": {"question": "q-sentinel", "focus": "f-sentinel"},
            "criteria": {"unsafe": {"what": "u-sentinel"}, "safe": {"what": "s-sentinel"}},
        }
    }
)


def task_state(prompt: str) -> TaskState:
    return TaskState(
        model="typesafe/jev-1.13.0",
        sample_id="sample",
        epoch=1,
        input=prompt,
        messages=[ChatMessageUser(content=prompt)],
    )


async def _generate(state: TaskState, **kwargs: Any) -> TaskState:
    return state


def run(solver: Any, state: TaskState) -> TaskState:
    result: TaskState = asyncio.run(solver(state, _generate))
    return result


def test_questions_are_delivered_verbatim_so_their_hash_still_describes_them() -> None:
    state = run(jev_prompt(QUESTIONS), task_state("prompt-sentinel"))

    system = state.messages[0]
    assert system.role == "system"
    # Byte-identical: templating would have eaten the JSON braces.
    assert system.text == QUESTIONS
    assert json.loads(system.text)["safety"]["instructions"]["question"] == "q-sentinel"


@pytest.mark.parametrize("factory", [jev_prompt, jev_policy_prompt])
def test_empty_questions_are_refused(factory: Any) -> None:
    with pytest.raises(ValueError, match="non-empty questions"):
        factory("")


def test_a_conversation_without_a_user_message_is_refused() -> None:
    state = TaskState(
        model="typesafe/jev-1.13.0",
        sample_id="sample",
        epoch=1,
        input="Prompt",
        messages=[],
    )

    with pytest.raises(ValueError, match="user message"):
        run(jev_prompt(QUESTIONS), state)
