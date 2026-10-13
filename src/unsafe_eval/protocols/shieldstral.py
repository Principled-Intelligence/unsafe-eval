"""Shieldstral benchmark protocol construction."""

from inspect_ai.model import GenerateConfig

from unsafe_eval.profiles.shieldstral import (
    SHIELDSTRAL_PROMPT_RESOURCE_HASHES,
    shieldstral_metadata_policy_prompt,
    shieldstral_prompt,
)
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    protocol_options,
    threshold_option,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.shieldstral_safety import shieldstral_safety

PROTOCOL: ProtocolName = "shieldstral-1.0-3b"


def build_shieldstral_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build Shieldstral components for one benchmark."""
    values = protocol_options(PROTOCOL, options, {"threshold"})
    threshold = threshold_option(values)
    validate_hashes = validate_prompt_resource_hashes_option(values)
    instruction, instruction_hash = adaptation_resource(
        benchmark, PROTOCOL, "instruction.txt", validate_hash=validate_hashes
    )
    query, query_hash = adaptation_resource(
        benchmark, PROTOCOL, "query.txt", validate_hash=validate_hashes
    )
    return SafetyProtocol(
        setup=(
            shieldstral_metadata_policy_prompt(
                instruction_template=instruction,
                query=query,
            )
            if spec.policy_conditioned
            else shieldstral_prompt(instruction=instruction, query=query)
        ),
        scorer=shieldstral_safety(threshold=threshold),
        config=GenerateConfig(
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        ),
        metrics=build_protocol_metrics(spec, probability=True),
        metadata={
            "protocol": PROTOCOL,
            "protocol_options": {
                "threshold": threshold,
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": {
                **SHIELDSTRAL_PROMPT_RESOURCE_HASHES,
                **instruction_hash,
                **query_hash,
            },
        },
    )
