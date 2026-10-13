from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.dynaguardrail_prompt_safety")


@pytest.fixture
def upstream_record() -> dict[str, Any]:
    return {
        "prompt": "An observable test prompt",
        "label": "safe",
        "label_detailed": "",
        "label_ai_predicted": "unsafe",
        "classes": "['category']",
        "trickiness": "borderline",
        "num_words": 4,
        "abnormal_syntax": False,
        "source": "test-source",
        "dataset_type": "finance",
    }


def test_policy_resources_are_packaged_and_hashed() -> None:
    assert set(benchmark.DYNA_GUARDRAIL_POLICIES) == {
        "finance",
        "injection",
        "safety",
        "tax",
    }
    assert all(policy for policy in benchmark.DYNA_GUARDRAIL_POLICIES.values())
    assert set(benchmark.DYNA_GUARDRAIL_POLICY_RESOURCE_HASHES) == {
        "benchmark-policy/finance.txt",
        "benchmark-policy/injection.txt",
        "benchmark-policy/safety.txt",
        "benchmark-policy/tax.txt",
    }


@pytest.mark.parametrize(
    ("domain", "label", "target"),
    [
        ("finance", "safe", "no"),
        ("injection", "unsafe", "yes"),
    ],
)
def test_record_maps_each_label_with_its_domain_policy(
    upstream_record: dict[str, Any], domain: str, label: str, target: str
) -> None:
    upstream_record["label"] = label
    upstream_record["dataset_type"] = domain

    sample = benchmark.record_to_sample(upstream_record)

    assert sample.input == "An observable test prompt"
    assert sample.target == target
    assert sample.metadata == {
        "safety_policy": benchmark.DYNA_GUARDRAIL_POLICIES[domain],
        "policy_domain": domain,
        "classes": "['category']",
        "trickiness": "borderline",
        "num_words": 4,
        "abnormal_syntax": False,
        "source": "test-source",
    }
    assert "label_detailed" not in sample.metadata
    assert "label_ai_predicted" not in sample.metadata


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", "harmful", "Unsupported DynaGuardrail label"),
        ("dataset_type", "medical", "Unsupported DynaGuardrail dataset_type"),
        ("num_words", -1, "num_words must be a non-negative integer"),
        ("abnormal_syntax", 0, "abnormal_syntax must be a boolean"),
    ],
)
def test_record_rejects_malformed_fields(
    upstream_record: dict[str, Any], field: str, value: Any, message: str
) -> None:
    upstream_record[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.record_to_sample(upstream_record)


def test_loader_uses_pinned_aggregate_split(
    monkeypatch: pytest.MonkeyPatch, upstream_record: dict[str, Any]
) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_dataset(**kwargs: Any) -> Dataset:
        captured.update(kwargs)
        converter: Callable[[dict[str, Any]], Sample] = kwargs["sample_fields"]
        return MemoryDataset([converter(upstream_record)])

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)

    dataset = benchmark.load_dynaguardrail_prompt_safety()

    assert len(dataset) == 1
    assert captured == {
        "path": "parquet",
        "split": "train",
        "data_files": {
            "train": (
                "hf://datasets/dynamofl/DynaGuardrail"
                "@3c21998805b692819e527f88702f761e33f6a013/"
                "data/train-00000-of-00001.parquet"
            )
        },
        "sample_fields": benchmark.record_to_sample,
        "auto_id": True,
        "trust": False,
    }
