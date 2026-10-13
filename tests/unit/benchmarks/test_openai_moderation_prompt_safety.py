from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.openai_moderation_prompt_safety")


def upstream_record(**overrides: object) -> dict[str, Any]:
    record: dict[str, Any] = {
        "prompt": "An observable test prompt",
        "S": 0,
        "H": 0,
        "V": 0,
        "HR": 0,
        "SH": 0,
        "S3": 0,
        "H2": 0,
        "V2": 0,
    }
    record.update(overrides)
    return record


def fake_dataset() -> MemoryDataset:
    return MemoryDataset(
        [
            Sample(
                id="1",
                input="An observable test prompt",
                target="no",
                metadata={
                    "category_labels": {
                        category: 0 for category in benchmark.OPENAI_MODERATION_CATEGORIES
                    }
                },
            )
        ]
    )


def test_known_positive_record_is_unsafe_even_with_unknown_categories() -> None:
    samples = benchmark.record_to_samples({"prompt": "Prompt", "H": 1, "V": 0})

    assert len(samples) == 1
    assert samples[0].input == "Prompt"
    assert samples[0].target == "yes"
    assert samples[0].metadata == {"category_labels": {"H": 1, "V": 0}}


def test_fully_known_negative_record_is_safe() -> None:
    samples = benchmark.record_to_samples(upstream_record())

    assert len(samples) == 1
    assert samples[0].target == "no"
    assert samples[0].metadata == {
        "category_labels": {category: 0 for category in benchmark.OPENAI_MODERATION_CATEGORIES}
    }


def test_partial_all_negative_record_is_excluded_as_ambiguous() -> None:
    assert benchmark.record_to_samples({"prompt": "Prompt", "S": 0, "H": 0}) == []
    assert benchmark.record_to_samples({"prompt": "Prompt", "S": 0, "H": None}) == []


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"S": 1}, "missing required field: prompt"),
        ({"prompt": None}, "prompt must be a non-empty string"),
        ({"prompt": ""}, "prompt must be a non-empty string"),
    ],
)
def test_record_rejects_invalid_prompt(record: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.record_to_samples(record)


@pytest.mark.parametrize("label", [2, 1.0])
def test_record_rejects_malformed_category_label(label: object) -> None:
    with pytest.raises(ValueError, match="Unsupported OpenAI Moderation S label"):
        benchmark.record_to_samples({"prompt": "Prompt", "S": label})


def test_loader_uses_pinned_compressed_json_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_hf_dataset(**kwargs: Any) -> Dataset:
        captured.update(kwargs)
        converter: Callable[[dict[str, Any]], list[Sample]] = kwargs["sample_fields"]
        return MemoryDataset(converter(upstream_record()))

    monkeypatch.setattr(benchmark, "hf_dataset", fake_hf_dataset)
    dataset = benchmark.load_openai_moderation_prompt_safety()

    assert captured["path"] == "json"
    assert captured["split"] == "train"
    assert captured["data_files"] == (
        "https://raw.githubusercontent.com/openai/moderation-api-release/"
        f"{benchmark.OPENAI_MODERATION_REVISION}/data/samples-1680.jsonl.gz"
    )
    assert captured["sample_fields"] is benchmark.record_to_samples
    assert captured["auto_id"] is True
    assert captured["trust"] is False
    assert len(dataset) == 1
    assert dataset[0].target == "no"

    benchmark.load_openai_moderation_prompt_safety("custom-revision")

    assert "/custom-revision/" in captured["data_files"]


def test_task_omits_pinned_digest_for_revision_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark, "load_openai_moderation_prompt_safety", lambda revision: fake_dataset()
    )

    task = benchmark.openai_moderation_prompt_safety(
        protocol="nemotron-3.5-content-safety",
        dataset_revision="custom-revision",
        protocol_options={"route": "vllm"},
    )

    assert "dataset_file_sha256" not in task.metadata
