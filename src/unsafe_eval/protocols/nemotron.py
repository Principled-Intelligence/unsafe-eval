"""Nemotron 3.5 Content Safety benchmark protocol construction, over a selected route.

Nemotron is steered through its chat template's arguments (`request_categories`,
`enable_thinking`, `custom_policy`). vLLM forwards them and OpenRouter silently discards
them, so the protocol refuses a setting the route cannot deliver, and a run never records
an argument the model did not see. With the same options, both routes build the
same prompt.
"""

from inspect_ai.model import GenerateConfig

from unsafe_eval.profiles.nemotron import (
    nemotron_inline_policy_generate,
    nemotron_metadata_policy_generate,
)
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    boolean_option,
    protocol_options,
    route_option,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.nemotron_safety import nemotron_safety

PROTOCOL: ProtocolName = "nemotron-3.5-content-safety"

SUPPORTED_ROUTES: tuple[str, ...] = ("vllm", "openrouter")


def build_nemotron_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build Nemotron 3.5 Content Safety components for one benchmark."""
    values = protocol_options(PROTOCOL, options, {"inline_policy", "request_categories", "route"})
    route = route_option(values, SUPPORTED_ROUTES)
    request_categories = boolean_option(values, "request_categories", True)
    inline_policy = boolean_option(values, "inline_policy", False)
    validate_hashes = validate_prompt_resource_hashes_option(values)
    template_kwargs = route.forwards_chat_template_kwargs
    if request_categories and not template_kwargs:
        # Without template arguments the template falls back to "/no_categories".
        raise ValueError(
            f"Route {route.name!r} forwards no chat-template arguments, so "
            "'request_categories' must be false"
        )
    if inline_policy and not spec.policy_conditioned:
        raise ValueError(
            "Protocol option 'inline_policy' applies only to policy-conditioned benchmarks"
        )
    if spec.policy_conditioned and not inline_policy and not template_kwargs:
        # The per-sample policy travels in `custom_policy`; dropping it would score the
        # model's default taxonomy under the benchmark's name.
        raise ValueError(
            f"Route {route.name!r} forwards no chat-template arguments, so "
            "policy-conditioned benchmarks need 'inline_policy'"
        )
    solver = None
    custom_policy_hashes: dict[str, str] = {}
    if spec.policy_conditioned:
        custom_policy, custom_policy_hashes = adaptation_resource(
            benchmark, PROTOCOL, "custom-policy.txt", validate_hash=validate_hashes
        )
        solver = (
            nemotron_inline_policy_generate(custom_policy_template=custom_policy)
            if inline_policy
            else nemotron_metadata_policy_generate(
                custom_policy_template=custom_policy,
                request_categories=request_categories,
            )
        )
    extra_body: dict[str, object]
    if template_kwargs:
        extra_body = {
            "chat_template_kwargs": {
                "request_categories": ("/categories" if request_categories else "/no_categories"),
                "enable_thinking": False,
            }
        }
    else:
        # OpenRouter enables reasoning by default and maps it onto `enable_thinking`,
        # which swaps "/no_think" for "/think" and pushes the answer past `max_tokens`.
        extra_body = {"reasoning": {"enabled": False}}
    return SafetyProtocol(
        setup=None,
        scorer=nemotron_safety(),
        config=GenerateConfig(max_tokens=100, temperature=0.0, extra_body=extra_body),
        metrics=build_protocol_metrics(spec),
        metadata={
            "protocol": PROTOCOL,
            "protocol_options": {
                "route": route.name,
                "request_categories": request_categories,
                "inline_policy": inline_policy,
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": custom_policy_hashes,
        },
        solver=solver,
    )
