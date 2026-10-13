import asyncio
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from unsafe_eval.protocols import resolve_protocol


def task_state(metadata: dict[str, Any] | None = None) -> TaskState:
    return TaskState(
        model="typesafe/jev-1.13.0",
        sample_id="sample",
        epoch=1,
        input="prompt-sentinel",
        messages=[ChatMessageUser(content="prompt-sentinel")],
        metadata=metadata or {},
    )


async def _generate(state: TaskState, **kwargs: Any) -> TaskState:
    return state


def run_setup(benchmark: str, metadata: dict[str, Any] | None = None) -> TaskState:
    selected = resolve_protocol("jev", benchmark=benchmark)
    result: TaskState = asyncio.run(selected.setup(task_state(metadata), _generate))
    return result


def test_metadata_records_the_resolved_options_and_the_questions_digest() -> None:
    selected = resolve_protocol("jev", benchmark="aegis-2.0")

    hashes = selected.metadata["prompt_resource_hashes"]
    assert set(hashes) == {"protocol/questions.json"}
    assert all(len(digest) == 64 for digest in hashes.values())
    assert selected.metadata["protocol_options"] == {"validate_prompt_resource_hashes": True}


def test_a_policy_conditioned_benchmark_requires_the_sample_policy() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_setup("cosapien")

    state = run_setup("cosapien", {"safety_policy": "policy-sentinel"})
    payload = state.messages[-1].metadata["state"]
    assert payload == {"policy": "policy-sentinel", "message": "prompt-sentinel"}
