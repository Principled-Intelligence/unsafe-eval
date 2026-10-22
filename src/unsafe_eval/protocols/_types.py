"""Shared types for versioned safety protocol construction."""

from dataclasses import dataclass
from typing import Any

from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Metric, Scorer
from inspect_ai.solver import Solver

ProtocolName = str


@dataclass(frozen=True)
class SafetyProtocol:
    """Resolved Inspect components and metadata for one benchmark/protocol pairing.

    `setup` prepares sample messages, `solver` runs the model, and `scorer` interprets its
    output. `config` holds generation settings; `metrics` maps score keys to aggregate
    metrics. `metadata` records protocol options and resource hashes for the task log.
    A missing `solver` selects the default generator in `build_prompt_safety_task`.
    """

    setup: Solver | None
    scorer: Scorer
    config: GenerateConfig
    metrics: dict[str, list[Metric]]
    metadata: dict[str, Any]
    solver: Solver | None = None
