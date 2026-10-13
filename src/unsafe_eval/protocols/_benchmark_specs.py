"""Canonical evaluation semantics for supported safety benchmarks."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from inspect_ai.scorer import Metric

from unsafe_eval.scorers.metrics import mixed_label_metrics, unsafe_only_metrics

BenchmarkName = str


class BenchmarkMetricBuilder(Protocol):
    """Build the score-to-metrics mapping for a benchmark and model protocol."""

    def __call__(
        self,
        *,
        probability: bool,
        scope_classes: list[str] | None,
    ) -> dict[str, list[Metric]]: ...


@dataclass(frozen=True)
class BenchmarkProtocolSpec:
    """Protocol-facing evaluation semantics for one benchmark."""

    mixed_labels: bool = True
    primary_metric: str = "unsafe_f1"
    metric_group_by: str | None = None
    unsafe_group_by: tuple[str, ...] = ()
    policy_conditioned: bool = False
    metric_builder: BenchmarkMetricBuilder | None = None


_BENCHMARK_PROTOCOL_SPECS: dict[BenchmarkName, BenchmarkProtocolSpec] = {
    "aegis-2.0": BenchmarkProtocolSpec(),
    "cosapien": BenchmarkProtocolSpec(metric_group_by="scenario", policy_conditioned=True),
    "dynaguardrail": BenchmarkProtocolSpec(
        metric_group_by="policy_domain", policy_conditioned=True
    ),
    "harmbench-text-test": BenchmarkProtocolSpec(
        mixed_labels=False,
        primary_metric="unsafe_recall",
        unsafe_group_by=("functional_category", "semantic_category"),
    ),
    "openai-moderation": BenchmarkProtocolSpec(),
    "simple-safety-tests": BenchmarkProtocolSpec(
        mixed_labels=False,
        primary_metric="unsafe_recall",
        unsafe_group_by=("harm_area", "category"),
    ),
    "toxic-chat-0124": BenchmarkProtocolSpec(),
    "wildguardmix-test": BenchmarkProtocolSpec(metric_group_by="prompt_type"),
}


def register_benchmark(
    name: BenchmarkName,
    spec: BenchmarkProtocolSpec,
) -> None:
    """Register protocol-facing semantics for a benchmark identifier."""
    if not name:
        raise ValueError("Benchmark name must not be empty")
    if name in _BENCHMARK_PROTOCOL_SPECS:
        raise ValueError(f"Benchmark is already registered: {name}")
    _BENCHMARK_PROTOCOL_SPECS[name] = spec


def registered_benchmarks() -> Mapping[BenchmarkName, BenchmarkProtocolSpec]:
    """Return a read-only snapshot of registered benchmark semantics."""
    return MappingProxyType(dict(_BENCHMARK_PROTOCOL_SPECS))


def benchmark_protocol_spec(benchmark: BenchmarkName) -> BenchmarkProtocolSpec:
    """Return the immutable protocol semantics for a supported benchmark."""
    try:
        return _BENCHMARK_PROTOCOL_SPECS[benchmark]
    except KeyError as error:
        raise ValueError(f"Unsupported benchmark: {benchmark}") from error


def build_protocol_metrics(
    spec: BenchmarkProtocolSpec,
    *,
    probability: bool = False,
    scope_classes: list[str] | None = None,
) -> dict[str, list[Metric]]:
    """Build metrics from the canonical semantics for one benchmark."""
    if spec.metric_builder is not None:
        return spec.metric_builder(
            probability=probability,
            scope_classes=scope_classes,
        )
    if spec.mixed_labels:
        if spec.primary_metric != "unsafe_f1":
            raise ValueError("Mixed-label benchmarks must use unsafe F1")
        return mixed_label_metrics(
            group_by=spec.metric_group_by,
            scope_classes=scope_classes,
        )
    if spec.primary_metric != "unsafe_recall":
        raise ValueError("Unsafe-only benchmarks must use unsafe recall")
    return unsafe_only_metrics(
        probability=probability,
        scope_classes=scope_classes,
        group_by=spec.unsafe_group_by,
    )
