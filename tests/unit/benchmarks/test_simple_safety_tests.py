from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import MemoryDataset

benchmark = import_module("unsafe_eval.benchmarks.simple_safety_tests")


def upstream_record() -> dict[str, Any]:
    return {
        "id": "sst_001",
        "prompt": "A sensitive test prompt",
        "harm_area": "Physical Harm and Violence",
        "category": "Instructions and actions",
        "counter": 1,
    }


def test_record_maps_all_fields_and_assigns_unsafe_target() -> None:
    sample = benchmark.record_to_sample(upstream_record())

    assert sample.id == "sst_001"
    assert sample.input == "A sensitive test prompt"
    assert sample.target == "yes"
    assert sample.metadata == {
        "harm_area": "Physical Harm and Violence",
        "category": "Instructions and actions",
        "counter": 1,
    }


def test_loader_uses_inspect_field_mapping_and_pinned_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_dataset(**kwargs: Any) -> MemoryDataset:
        captured.update(kwargs)
        converter = kwargs["sample_fields"]
        return MemoryDataset([converter(upstream_record())])

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)
    dataset = benchmark.load_simple_safety_tests()

    assert len(dataset) == 1
    assert captured["path"] == "Bertievidgen/SimpleSafetyTests"
    assert captured["name"] == "default"
    assert captured["split"] == "test"
    assert captured["revision"] == benchmark.SIMPLE_SAFETY_TESTS_REVISION
    assert captured["trust"] is False
    assert captured["sample_fields"] is benchmark.record_to_sample
    assert dataset[0].target == "yes"
