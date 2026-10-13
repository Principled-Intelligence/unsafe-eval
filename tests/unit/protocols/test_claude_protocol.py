import pytest
from inspect_ai.model._reasoning import effort_to_reasoning_tokens

from unsafe_eval.profiles.claude import CLAUDE_PROMPT_RESOURCE_HASHES
from unsafe_eval.protocols import resolve_protocol
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols.claude import (
    ANSWER_TOKENS,
    REASONING_BUDGETS,
    REASONING_EFFORTS,
)


def test_claude_requires_a_route() -> None:
    with pytest.raises(ValueError, match="'route' is required"):
        resolve_protocol("claude", benchmark="aegis-2.0")


def test_claude_records_its_options_and_the_digests_of_its_own_prompt_resources() -> None:
    protocol = resolve_protocol(
        "claude",
        benchmark="aegis-2.0",
        options={"route": "openrouter", "reasoning_effort": "none"},
    )
    # Claude's prompt is its own and may diverge from any other family's, so what a
    # run records is the digest of what Claude was sent, loaded from its own files.
    _, instruction_hashes = adaptation_resource("aegis-2.0", "claude", "instruction.txt")

    assert protocol.metadata["protocol"] == "claude"
    assert protocol.metadata["protocol_options"] == {
        "route": "openrouter",
        "output_fields": ["reasoning", "user_safety"],
        "reasoning_effort": "none",
        "validate_prompt_resource_hashes": True,
    }
    assert protocol.metadata["prompt_resource_hashes"] == {
        **CLAUDE_PROMPT_RESOURCE_HASHES,
        **instruction_hashes,
    }


def test_max_tokens_leaves_room_above_the_thinking_budget() -> None:
    # Anthropic rejects an extended-thinking budget that is not below max_tokens,
    # which is how Claude Haiku 4.5 receives a reasoning effort. The budgets are
    # copied from Inspect's effort table, so they must still match what it sends.
    offenders = []
    for effort in REASONING_EFFORTS:
        protocol = resolve_protocol(
            "claude",
            benchmark="aegis-2.0",
            options={"route": "anthropic", "reasoning_effort": effort},
        )
        if (
            REASONING_BUDGETS[effort] != (effort_to_reasoning_tokens(effort) or 0)
            or protocol.config.reasoning_effort != effort
            or protocol.config.max_tokens != ANSWER_TOKENS + REASONING_BUDGETS[effort]
        ):
            offenders.append(effort)

    assert not offenders
