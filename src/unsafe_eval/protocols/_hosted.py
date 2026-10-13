"""Protocol construction shared by the hosted structured-label families."""

from collections.abc import Callable, Mapping
from typing import Literal

from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from unsafe_eval.profiles._structured_label import StructuredLabelFormat
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    choice_option,
    protocol_options,
    route_option,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


def build_hosted_protocol(
    protocol: ProtocolName,
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
    *,
    family: str,
    label_format: StructuredLabelFormat,
    prompt: Callable[[str], Solver],
    metadata_policy_prompt: Callable[[str], Solver],
    scorer: Callable[..., Scorer],
    solver: Solver,
    prompt_resource_hashes: Mapping[str, str],
    supported_routes: tuple[str, ...],
    default_output_fields: tuple[str, ...],
    reasoning_efforts: tuple[ReasoningEffort, ...],
    default_reasoning_effort: ReasoningEffort,
    max_tokens: int,
    reasoning_budgets: Mapping[str, int] | None = None,
    seed: int | None = None,
) -> SafetyProtocol:
    """Build one hosted family's components for a benchmark, over the requested route."""
    values = protocol_options(protocol, options, {"output_fields", "reasoning_effort", "route"})
    route = route_option(values, supported_routes)
    output_fields = label_format.normalize_selection(
        values.get("output_fields", default_output_fields)
    )
    reasoning_effort = choice_option(
        values, "reasoning_effort", reasoning_efforts, default_reasoning_effort
    )
    validate_hashes = validate_prompt_resource_hashes_option(values)
    instruction, instruction_hashes = adaptation_resource(
        benchmark, protocol, "instruction.txt", validate_hash=validate_hashes
    )
    return SafetyProtocol(
        setup=(
            metadata_policy_prompt(instruction) if spec.policy_conditioned else prompt(instruction)
        ),
        scorer=scorer(output_fields=output_fields),
        solver=solver,
        config=GenerateConfig(
            max_tokens=max_tokens + (reasoning_budgets or {}).get(reasoning_effort, 0),
            seed=seed if route.honours_seed else None,
            reasoning_effort=reasoning_effort,
            # Every prompt is unique, so a cache write is never read back and only adds cost.
            cache_prompt=False,
            response_schema=ResponseSchema(
                name=f"{protocol}_response",
                description=(
                    f"{family} prompt-safety classification, emitting exactly the "
                    "requested output fields in canonical order."
                ),
                json_schema=label_format.inspect_response_schema(output_fields),
                # OpenAI and OpenRouter treat a json_schema response format as advisory
                # unless `strict` is set, and the model then invents keys the scorer rejects.
                strict=True,
            ),
        ),
        metrics=build_protocol_metrics(spec),
        metadata={
            "protocol": protocol,
            "protocol_options": {
                "route": route.name,
                "output_fields": list(output_fields),
                "reasoning_effort": reasoning_effort,
                "validate_prompt_resource_hashes": validate_hashes,
            },
            "prompt_resource_hashes": {**prompt_resource_hashes, **instruction_hashes},
        },
    )
