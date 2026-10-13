from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.cosapien_prompt_safety")


@pytest.fixture
def upstream_record() -> dict[str, Any]:
    return {
        "prompt": "An observable test prompt",
        "scenario": "A scenario-specific safety policy.",
        "type": "allowed",
    }


@pytest.mark.parametrize(
    ("prompt_type", "target"),
    [
        ("safe", "no"),
        ("allowed", "no"),
        ("disallowed", "yes"),
        ("partial", "yes"),
    ],
)
def test_record_maps_every_source_type_without_filtering(
    upstream_record: dict[str, Any], prompt_type: str, target: str
) -> None:
    upstream_record["type"] = prompt_type

    sample = benchmark.record_to_sample(upstream_record)

    assert sample.input == "An observable test prompt"
    assert sample.target == target
    assert sample.metadata == {
        "safety_policy": "A scenario-specific safety policy.",
        "prompt_type": prompt_type,
    }


def test_record_rejects_an_unknown_type(upstream_record: dict[str, Any]) -> None:
    upstream_record["type"] = None

    with pytest.raises(ValueError, match="Unsupported CoSApien type"):
        benchmark.record_to_sample(upstream_record)


def test_loader_combines_all_pinned_scenario_splits(
    monkeypatch: pytest.MonkeyPatch, upstream_record: dict[str, Any]
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_hf_dataset(**kwargs: Any) -> Dataset:
        captured.append(kwargs)
        converter: Callable[[dict[str, Any]], Sample] = kwargs["sample_fields"]
        return MemoryDataset([converter(upstream_record)])

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)

    dataset = benchmark.load_cosapien_prompt_safety()

    assert [call["split"] for call in captured] == list(benchmark.COSAPIEN_SPLITS)
    assert all(call["path"] == "microsoft/CoSApien" for call in captured)
    assert all(call["name"] == "default" for call in captured)
    assert all(call["revision"] == benchmark.COSAPIEN_REVISION for call in captured)
    assert all(call["trust"] is False for call in captured)
    assert [sample.id for sample in dataset] == [
        f"{scenario}-1" for scenario in benchmark.COSAPIEN_SPLITS
    ]
    assert [sample.metadata["scenario"] for sample in dataset] == list(benchmark.COSAPIEN_SPLITS)
