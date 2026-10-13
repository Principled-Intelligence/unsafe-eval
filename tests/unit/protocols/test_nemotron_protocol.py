"""Nemotron 3.5 Content Safety over its vLLM and OpenRouter routes."""

import pytest

from unsafe_eval.protocols import resolve_protocol

PROTOCOL = "nemotron-3.5-content-safety"
OPENROUTER = {"route": "openrouter", "request_categories": False}


def test_vllm_steers_the_template_through_its_arguments() -> None:
    selected = resolve_protocol(
        PROTOCOL,
        benchmark="aegis-2.0",
        options={"route": "vllm", "request_categories": True},
    )

    assert selected.config.extra_body == {
        "chat_template_kwargs": {"request_categories": "/categories", "enable_thinking": False}
    }
    assert selected.solver is None


def test_openrouter_sends_no_template_arguments_and_disables_reasoning() -> None:
    selected = resolve_protocol(PROTOCOL, benchmark="aegis-2.0", options=OPENROUTER)

    assert selected.config.extra_body == {"reasoning": {"enabled": False}}
    assert selected.config.max_tokens == 100
    assert selected.config.temperature == 0.0
    assert selected.solver is None


def test_openrouter_refuses_categories_it_cannot_request() -> None:
    # `request_categories` defaults to true, so the refusal also covers an unset option.
    with pytest.raises(ValueError, match="'request_categories' must be false"):
        resolve_protocol(PROTOCOL, benchmark="aegis-2.0", options={"route": "openrouter"})


def test_openrouter_refuses_a_policy_it_cannot_deliver() -> None:
    with pytest.raises(ValueError, match="need 'inline_policy'"):
        resolve_protocol(PROTOCOL, benchmark="cosapien", options=OPENROUTER)


def test_inline_policy_applies_only_to_policy_conditioned_benchmarks() -> None:
    options = {"route": "vllm", "request_categories": False, "inline_policy": True}

    with pytest.raises(ValueError, match="applies only to policy-conditioned"):
        resolve_protocol(PROTOCOL, benchmark="aegis-2.0", options=options)


def test_metadata_records_the_route_and_every_resolved_option() -> None:
    selected = resolve_protocol(PROTOCOL, benchmark="aegis-2.0", options=OPENROUTER)

    assert selected.metadata["protocol"] == PROTOCOL
    assert selected.metadata["protocol_options"] == {
        "route": "openrouter",
        "request_categories": False,
        "inline_policy": False,
        "validate_prompt_resource_hashes": True,
    }
