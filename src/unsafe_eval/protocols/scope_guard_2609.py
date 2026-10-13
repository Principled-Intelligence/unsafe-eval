"""ScopeGuard 2609 benchmark protocol construction."""

from inspect_ai.model import GenerateConfig, ResponseSchema

from unsafe_eval.profiles.scope_guard_2609 import (
    SCOPE_GUARD_2609_FORMAT,
    SCOPE_GUARD_2609_PROMPT_RESOURCE_HASHES,
    scope_guard_2609_metadata_policy_prompt,
    scope_guard_2609_prompt,
)
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    choice_option,
    protocol_options,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.scope_guard_2609_safety import (
    SCORED_SCOPE_CLASSES_2609,
    UNSAFE_AXES,
    scope_guard_2609_safety,
)

PROTOCOL: ProtocolName = "scope-guard-2609"

# `reasoning` precedes `scope_class`, so it acts as chain-of-thought. Training measured
# this as the best must-refuse shape, which is why it is preferred to class-only.
DEFAULT_OUTPUT_FIELDS: tuple[str, ...] = ("reasoning", "scope_class")


def build_scope_guard_2609_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build ScopeGuard 2609 components for one benchmark."""
    values = protocol_options(PROTOCOL, options, {"output_fields", "unsafe_axis"})
    unsafe_axis = choice_option(values, "unsafe_axis", tuple(UNSAFE_AXES), "pass")
    output_fields = SCOPE_GUARD_2609_FORMAT.normalize_selection(
        values.get("output_fields", DEFAULT_OUTPUT_FIELDS)
    )
    validate_hashes = validate_prompt_resource_hashes_option(values)
    service_description, adaptation_hashes = adaptation_resource(
        benchmark,
        PROTOCOL,
        "ai-service-description.txt",
        validate_hash=validate_hashes,
    )
    return SafetyProtocol(
        setup=(
            scope_guard_2609_metadata_policy_prompt(
                ai_service_description_template=service_description,
                output_fields=output_fields,
            )
            if spec.policy_conditioned
            else scope_guard_2609_prompt(
                ai_service_description=service_description,
                output_fields=output_fields,
            )
        ),
        scorer=scope_guard_2609_safety(output_fields=output_fields, unsafe_axis=unsafe_axis),
        config=GenerateConfig(
            # The longest field selection measures p99 234 tokens in training.
            max_tokens=3000,
            temperature=0.0,
            response_schema=ResponseSchema(
                name="scope_guard_2609_response",
                description=(
                    "ScopeGuard 2609 classification, emitting exactly the "
                    "requested output fields in canonical order."
                ),
                json_schema=SCOPE_GUARD_2609_FORMAT.inspect_response_schema(output_fields),
            ),
        ),
        metrics=build_protocol_metrics(spec, scope_classes=list(SCORED_SCOPE_CLASSES_2609)),
        metadata={
            "protocol": PROTOCOL,
            "protocol_options": {
                "output_fields": list(output_fields),
                "unsafe_axis": unsafe_axis,
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": {
                **SCOPE_GUARD_2609_PROMPT_RESOURCE_HASHES,
                **adaptation_hashes,
            },
        },
    )
