import asyncio
import json
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.profiles.gliner2 import GLIGUARD_SCHEMA
from unsafe_eval.protocols import resolve_protocol


def setup(protocol: str, benchmark: str, **metadata: Any) -> TaskState:
    state = TaskState(
        model="gliner2/fastino/test",
        sample_id="sample",
        epoch=1,
        input="prompt-sentinel",
        messages=[ChatMessageUser(content="prompt-sentinel")],
        metadata=metadata,
    )

    async def generate(state: TaskState, **kwargs: Any) -> TaskState:
        return state

    selected = resolve_protocol(protocol, benchmark=benchmark)
    return asyncio.run(selected.setup(state, generate))


@pytest.mark.parametrize(
    ("protocol", "hashes"),
    [
        ("gliner2.5-decide", {"protocol/schema.json"}),
        ("gliguard", {"profile/schema.json", "profile/policy_text.txt"}),
    ],
)
def test_metadata_records_the_protocol_its_options_and_its_digests(
    protocol: str, hashes: set[str]
) -> None:
    selected = resolve_protocol(protocol, benchmark="aegis-2.0")

    assert selected.metadata["protocol"] == protocol
    assert set(selected.metadata["prompt_resource_hashes"]) == hashes
    assert selected.metadata["protocol_options"] == {"validate_prompt_resource_hashes": True}


def test_decide_carries_the_policy_in_its_instruction() -> None:
    result = setup("gliner2.5-decide", "cosapien", safety_policy="policy-sentinel")

    (task,) = json.loads(result.messages[0].text)["tasks"].values()
    assert task["instruction"].endswith("Policy: policy-sentinel")
    assert list(task["labels"]) == ["unsafe", "safe"]
    assert result.messages[-1].text == "prompt-sentinel"


def test_gliguard_carries_the_policy_in_its_text_and_nothing_else_elsewhere() -> None:
    policy = setup("gliguard", "cosapien", safety_policy="policy-sentinel")
    plain = setup("gliguard", "aegis-2.0", safety_policy="ignored")

    assert policy.messages[-1].text == "Policy: policy-sentinel\nPrompt: prompt-sentinel"
    assert plain.messages[-1].text == "prompt-sentinel"
    assert json.loads(plain.messages[0].text) == json.loads(GLIGUARD_SCHEMA)
