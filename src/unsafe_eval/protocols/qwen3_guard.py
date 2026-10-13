"""Qwen3Guard-Gen family benchmark protocol construction."""

from inspect_ai.model import GenerateConfig

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
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.qwen3_guard_safety import qwen3_guard_safety

PROTOCOL: ProtocolName = "qwen3-guard-gen"


def build_qwen3_guard_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build Qwen3Guard-Gen components for one compatible benchmark."""
    values = protocol_options(PROTOCOL, options, {"controversial_as_unsafe"})
    controversial_as_unsafe = boolean_option(values, "controversial_as_unsafe", True)
    validate_hashes = validate_prompt_resource_hashes_option(values)
    if spec.policy_conditioned:
        raise ValueError(f"{PROTOCOL} does not support policy-conditioned benchmarks")
    return SafetyProtocol(
        setup=None,
        scorer=qwen3_guard_safety(controversial_as_unsafe=controversial_as_unsafe),
        config=GenerateConfig(max_tokens=128, temperature=0.0),
        metrics=build_protocol_metrics(spec),
        metadata={
            "protocol": PROTOCOL,
            "protocol_options": {
                "controversial_as_unsafe": controversial_as_unsafe,
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": {},
        },
    )
