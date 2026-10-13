"""Shared types for versioned safety protocol construction."""

from dataclasses import dataclass
from typing import Any

from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Metric, Scorer
from inspect_ai.solver import Solver

ProtocolName = str


@dataclass(frozen=True)
class SafetyProtocol:
    """Inspect components required for one benchmark and model protocol pairing."""

    setup: Solver | None
    scorer: Scorer
    config: GenerateConfig
    metrics: dict[str, list[Metric]]
    metadata: dict[str, Any]
    solver: Solver | None = None
