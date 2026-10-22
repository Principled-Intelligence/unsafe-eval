"""Public task composition helpers."""

from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import generate

from unsafe_eval.protocols import BenchmarkName, ProtocolName, resolve_protocol


def build_prompt_safety_task(
    *,
    dataset: Dataset,
    protocol: ProtocolName,
    benchmark: BenchmarkName,
    protocol_options: dict[str, object] | None,
    version: int,
    metadata: dict[str, Any],
) -> Task:
    """Build an Inspect task using a registered benchmark and model protocol.

    `protocol_options` configures the protocol's prompt setup, generation, and scoring.
    The protocol supplies those components and the benchmark's aggregate metrics. If it
    supplies no generation solver, the task uses Inspect's generator with tools disabled.

    `version` is the Inspect task version. Protocol metadata overrides matching keys in
    `metadata`, and `benchmark_id` is always set to `benchmark`. Unknown identifiers or
    unsupported protocol options raise `ValueError` during protocol resolution.
    """
    selected = resolve_protocol(
        protocol,
        benchmark=benchmark,
        options=protocol_options,
    )
    return Task(
        dataset=dataset,
        setup=selected.setup,
        solver=selected.solver or generate(tool_calls="none"),
        scorer=selected.scorer,
        metrics=selected.metrics,
        config=selected.config,
        version=version,
        metadata={**metadata, **selected.metadata, "benchmark_id": benchmark},
    )


__all__ = ["build_prompt_safety_task"]
