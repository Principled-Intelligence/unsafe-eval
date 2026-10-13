"""PolyGuard-Qwen benchmark protocol construction."""

from inspect_ai.model import GenerateConfig

from unsafe_eval.profiles.polyguard import (
    POLYGUARD_PROMPT_RESOURCE_HASHES,
    polyguard_prompt,
)
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    protocol_options,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.polyguard_safety import polyguard_safety

PROTOCOL: ProtocolName = "polyguard-qwen"


def build_polyguard_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build PolyGuard-Qwen components for one compatible benchmark."""
    values = protocol_options(PROTOCOL, options, set())
    validate_hashes = validate_prompt_resource_hashes_option(values)
    if spec.policy_conditioned:
        raise ValueError(f"{PROTOCOL} does not support policy-conditioned benchmarks")
    return SafetyProtocol(
        setup=polyguard_prompt(),
        scorer=polyguard_safety(),
        config=GenerateConfig(max_tokens=100, temperature=0.0),
        metrics=build_protocol_metrics(spec),
        metadata={
            "protocol": PROTOCOL,
            "protocol_options": {
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": POLYGUARD_PROMPT_RESOURCE_HASHES,
        },
    )
