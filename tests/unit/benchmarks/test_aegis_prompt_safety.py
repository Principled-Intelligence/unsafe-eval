from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.aegis_prompt_safety")


@pytest.fixture
def upstream_record() -> dict[str, Any]:
    return {
        "id": "aegis-001",
        "reconstruction_id_if_redacted": None,
        "prompt": "An observable test prompt",
        "response": "This response must be ignored.",
        "prompt_label": "unsafe",
        "response_label": "safe",
        "violated_categories": "Violence, Harassment",
        "prompt_label_source": "human",
        "response_label_source": "llm_jury",
    }


@pytest.mark.parametrize(("label", "target"), [("unsafe", "yes"), ("safe", "no")])
def test_record_maps_prompt_label_and_ignores_response(
    upstream_record: dict[str, Any], label: str, target: str
) -> None:
    upstream_record["prompt_label"] = label
    samples = benchmark.record_to_samples(upstream_record)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "aegis-001"
    assert sample.input == "An observable test prompt"
    assert sample.target == target
    assert sample.metadata == {
        "violated_categories": "Violence, Harassment",
    }
    assert "response" not in sample.metadata
    assert "response_label" not in sample.metadata


def test_record_excludes_redacted_prompt(upstream_record: dict[str, Any]) -> None:
    upstream_record["prompt"] = "REDACTED"
    upstream_record["reconstruction_id_if_redacted"] = 123

    assert benchmark.record_to_samples(upstream_record) == []


def test_record_rejects_unknown_prompt_label(upstream_record: dict[str, Any]) -> None:
    upstream_record["prompt_label"] = "SAFE"
    with pytest.raises(ValueError, match="Unsupported Aegis prompt_label"):
        benchmark.record_to_samples(upstream_record)


def test_record_requires_human_prompt_label(upstream_record: dict[str, Any]) -> None:
    upstream_record["prompt_label_source"] = "llm_jury"
    with pytest.raises(ValueError, match="must be 'human'"):
        benchmark.record_to_samples(upstream_record)


def test_loader_uses_pinned_test_contract(
    monkeypatch: pytest.MonkeyPatch, upstream_record: dict[str, Any]
) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_dataset(**kwargs: Any) -> Dataset:
        captured.update(kwargs)
        converter: Callable[[dict[str, Any]], list[Sample]] = kwargs["sample_fields"]
        return MemoryDataset(converter(upstream_record))

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)
    dataset = benchmark.load_aegis_prompt_safety()

    assert captured["path"] == "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    assert captured["name"] == "default"
    assert captured["split"] == "test"
    assert captured["revision"] == benchmark.AEGIS_REVISION
    assert captured["trust"] is False
    assert dataset[0].metadata == {"violated_categories": "Violence, Harassment"}
