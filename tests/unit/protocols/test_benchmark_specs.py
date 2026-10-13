import pytest
from inspect_ai.scorer import Metric, mean

from unsafe_eval.protocols import register_benchmark
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkProtocolSpec,
    benchmark_protocol_spec,
    build_protocol_metrics,
)


def test_unknown_benchmark_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark"):
        benchmark_protocol_spec("unknown")


def test_benchmark_registration_rejects_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_benchmark("", BenchmarkProtocolSpec())
    with pytest.raises(ValueError, match="already registered"):
        register_benchmark("aegis-2.0", BenchmarkProtocolSpec())


@pytest.mark.parametrize(
    "spec",
    [
        BenchmarkProtocolSpec(mixed_labels=True, primary_metric="balanced_accuracy"),
        BenchmarkProtocolSpec(mixed_labels=False, primary_metric="balanced_accuracy"),
    ],
)
def test_metric_builder_rejects_inconsistent_primary(spec: BenchmarkProtocolSpec) -> None:
    with pytest.raises(ValueError, match="must use"):
        build_protocol_metrics(spec)


def test_custom_metric_builder_is_not_limited_to_builtin_primary_metrics() -> None:
    observed: tuple[bool, list[str] | None] | None = None

    def custom_metrics(
        *, probability: bool, scope_classes: list[str] | None
    ) -> dict[str, list[Metric]]:
        nonlocal observed
        observed = probability, scope_classes
        return {"custom_score": [mean()]}

    metrics = build_protocol_metrics(
        BenchmarkProtocolSpec(
            primary_metric="custom_mean",
            metric_builder=custom_metrics,
        ),
        probability=True,
        scope_classes=["custom"],
    )

    assert set(metrics) == {"custom_score"}
    assert observed == (True, ["custom"])
