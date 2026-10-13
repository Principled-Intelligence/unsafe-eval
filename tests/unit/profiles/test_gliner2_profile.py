import asyncio
import json
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.gliner2 import gliner2_prompt

SCHEMA = json.dumps(
    {"tasks": {"safety": {"labels": ["unsafe", "safe"], "instruction": "Under: {safety_policy}"}}}
)
POLICY = 'No "medical" advice (dosages).\nNever {explain} them.'


def run(schema: str, text_template: str = "{message}", **metadata: Any) -> TaskState:
    state = TaskState(
        model="gliner2/fastino/GLiNER2.5-Decide",
        sample_id="sample",
        epoch=1,
        input="prompt-sentinel",
        messages=[ChatMessageUser(content="prompt-sentinel")],
        metadata=metadata,
    )

    async def generate(state: TaskState, **kwargs: Any) -> TaskState:
        return state

    return asyncio.run(gliner2_prompt(schema, text_template)(state, generate))


def test_a_policy_reaches_the_instruction_and_the_text_verbatim() -> None:
    result = run(SCHEMA, "Policy: {safety_policy}\nPrompt: {message}", safety_policy=POLICY)

    schema = json.loads(result.messages[0].text)
    assert result.messages[0].role == "system"
    # gliner2 rejects parentheses in an instruction but not in the text.
    instruction = 'Under: No "medical" advice [dosages].\nNever {explain} them.'
    assert schema["tasks"]["safety"]["instruction"] == instruction
    assert result.messages[-1].text == f"Policy: {POLICY}\nPrompt: prompt-sentinel"


def test_a_policy_placeholder_without_a_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="safety_policy"):
        run(SCHEMA)
