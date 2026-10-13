"""ScopeGuard benchmark protocol construction."""

from inspect_ai.model import GenerateConfig, ResponseSchema

from unsafe_eval.profiles.scope_guard import (
    INSPECT_SCOPE_GUARD_RESPONSE_SCHEMA,
    SCOPE_GUARD_PROMPT_RESOURCE_HASHES,
    scope_guard_metadata_policy_prompt,
    scope_guard_prompt,
)
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    boolean_option,
    protocol_options,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.scope_guard_safety import SCORED_SCOPE_CLASSES, scope_guard_safety

PROTOCOL: ProtocolName = "scope-guard-2601"


def build_scope_guard_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build ScopeGuard components for one benchmark."""
    values = protocol_options(PROTOCOL, options, {"include_evidence"})
    include_evidence = boolean_option(values, "include_evidence", True)
    validate_hashes = validate_prompt_resource_hashes_option(values)
    service_description, adaptation_hashes = adaptation_resource(
        benchmark,
        PROTOCOL,
        "ai-service-description.txt",
        validate_hash=validate_hashes,
    )
    return SafetyProtocol(
        setup=(
            scope_guard_metadata_policy_prompt(
                ai_service_description_template=service_description,
                include_evidence=include_evidence,
            )
            if spec.policy_conditioned
            else scope_guard_prompt(
                ai_service_description=service_description,
                include_evidence=include_evidence,
            )
        ),
        scorer=scope_guard_safety(),
        config=GenerateConfig(
            max_tokens=3000,
            temperature=0.0,
            response_schema=ResponseSchema(
                name="scope_guard_response",
                description="ScopeGuard classification and supporting policy evidence.",
                json_schema=INSPECT_SCOPE_GUARD_RESPONSE_SCHEMA,
            ),
        ),
        metrics=build_protocol_metrics(spec, scope_classes=list(SCORED_SCOPE_CLASSES)),
        metadata={
            "protocol": PROTOCOL,
            "protocol_options": {
                "include_evidence": include_evidence,
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": {
                **SCOPE_GUARD_PROMPT_RESOURCE_HASHES,
                **adaptation_hashes,
            },
        },
    )
