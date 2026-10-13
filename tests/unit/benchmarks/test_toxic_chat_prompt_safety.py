from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.toxic_chat_prompt_safety")


@pytest.fixture
def upstream_record() -> dict[str, Any]:
    return {
        "conv_id": "toxic-chat-001",
        "user_input": "An observable test prompt",
        "model_output": "This response must be ignored.",
        "human_annotation": True,
        "toxicity": 1,
        "jailbreaking": 0,
        "openai_moderation": "This moderation result must be ignored.",
    }


@pytest.mark.parametrize(("label", "target"), [(1, "yes"), (0, "no")])
def test_record_maps_toxicity_and_ignores_response_fields(
    upstream_record: dict[str, Any], label: int, target: str
) -> None:
    upstream_record["toxicity"] = label

    sample = benchmark.record_to_sample(upstream_record)

    assert sample.id == "toxic-chat-001"
    assert sample.input == "An observable test prompt"
    assert sample.target == target
    assert sample.metadata == {"human_annotation": True, "jailbreaking": 0}
    assert "model_output" not in sample.metadata
    assert "openai_moderation" not in sample.metadata


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("human_annotation", 1, "human_annotation must be a boolean"),
        ("toxicity", True, "Unsupported ToxicChat toxicity label"),
        ("toxicity", 2, "Unsupported ToxicChat toxicity label"),
        ("jailbreaking", -1, "Unsupported ToxicChat jailbreaking label"),
    ],
)
def test_record_rejects_malformed_fields(
    upstream_record: dict[str, Any], field: str, value: Any, message: str
) -> None:
    upstream_record[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.record_to_sample(upstream_record)


def test_loader_uses_pinned_0124_test_contract(
    monkeypatch: pytest.MonkeyPatch, upstream_record: dict[str, Any]
) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_dataset(**kwargs: Any) -> Dataset:
        captured.update(kwargs)
        converter: Callable[[dict[str, Any]], Sample] = kwargs["sample_fields"]
        automatically_labeled_record = {
            **upstream_record,
            "conv_id": "toxic-chat-automatic",
            "human_annotation": False,
            "toxicity": 0,
        }
        return MemoryDataset([converter(upstream_record), converter(automatically_labeled_record)])

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)
    dataset = benchmark.load_toxic_chat_prompt_safety()

    assert captured["path"] == "lmsys/toxic-chat"
    assert captured["name"] == "toxicchat0124"
    assert captured["split"] == "test"
    assert captured["revision"] == benchmark.TOXIC_CHAT_REVISION
    assert captured["trust"] is False
    assert len(dataset) == 1
    assert dataset[0].id == "toxic-chat-001"
    assert dataset[0].metadata == {"human_annotation": True, "jailbreaking": 0}
