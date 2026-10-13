from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.wildguard_mix_prompt_safety")


@pytest.fixture
def upstream_record() -> dict[str, Any]:
    return {
        "prompt": "An observable test prompt",
        "response": "This response must be ignored.",
        "adversarial": True,
        "prompt_harm_label": "harmful",
        "response_refusal_agreement": 3.0,
        "response_refusal_label": "compliance",
        "response_harm_label": "harmful",
        "subcategory": "cyberattack",
        "prompt_harm_agreement": 2.0,
        "response_harm_agreement": 3.0,
    }


@pytest.mark.parametrize(("label", "target"), [("harmful", "yes"), ("unharmful", "no")])
def test_record_maps_prompt_label_and_ignores_response_fields(
    upstream_record: dict[str, Any], label: str, target: str
) -> None:
    upstream_record["prompt_harm_label"] = label

    samples = benchmark.record_to_samples(upstream_record)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.input == "An observable test prompt"
    assert sample.target == target
    assert sample.metadata == {
        "adversarial": True,
        "prompt_type": "adversarial",
        "subcategory": "cyberattack",
        "prompt_harm_agreement": 2,
    }
    assert "response" not in sample.metadata
    assert "response_harm_label" not in sample.metadata
    assert "response_refusal_label" not in sample.metadata


def test_record_marks_non_adversarial_prompt_as_vanilla(
    upstream_record: dict[str, Any],
) -> None:
    upstream_record["adversarial"] = False

    sample = benchmark.record_to_samples(upstream_record)[0]

    assert sample.metadata["prompt_type"] == "vanilla"


def test_record_excludes_prompt_without_agreement(
    upstream_record: dict[str, Any],
) -> None:
    upstream_record["prompt_harm_label"] = None
    upstream_record["prompt_harm_agreement"] = None

    assert benchmark.record_to_samples(upstream_record) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adversarial", 1, "adversarial must be a boolean"),
        ("prompt_harm_label", "unsafe", "Unsupported WildGuardMix prompt_harm_label"),
        ("prompt_harm_label", [], "Unsupported WildGuardMix prompt_harm_label"),
        ("prompt_harm_agreement", 1.0, "prompt_harm_agreement must be 2 or 3"),
        ("prompt_harm_agreement", None, "prompt_harm_agreement must be 2 or 3"),
    ],
)
def test_record_rejects_malformed_fields(
    upstream_record: dict[str, Any], field: str, value: Any, message: str
) -> None:
    upstream_record[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.record_to_samples(upstream_record)


def test_record_requires_null_agreement_for_missing_label(
    upstream_record: dict[str, Any],
) -> None:
    upstream_record["prompt_harm_label"] = None

    with pytest.raises(ValueError, match="must be null"):
        benchmark.record_to_samples(upstream_record)


def test_loader_uses_pinned_official_test_contract(
    monkeypatch: pytest.MonkeyPatch, upstream_record: dict[str, Any]
) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_dataset(**kwargs: Any) -> Dataset:
        captured.update(kwargs)
        converter: Callable[[dict[str, Any]], list[Sample]] = kwargs["sample_fields"]
        return MemoryDataset(converter(upstream_record))

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)
    dataset = benchmark.load_wildguard_mix_prompt_safety()

    assert captured["path"] == "allenai/wildguardmix"
    assert captured["name"] == "wildguardtest"
    assert captured["split"] == "test"
    assert captured["revision"] == benchmark.WILDGUARD_MIX_REVISION
    assert captured["auto_id"] is True
    assert captured["trust"] is False
    assert len(dataset) == 1
