from importlib import import_module
from typing import Any

import pytest
from inspect_ai.dataset import MemoryDataset, Sample

benchmark = import_module("unsafe_eval.benchmarks.harmbench_prompt_safety")


def upstream_record(**overrides: object) -> dict[str, Any]:
    record: dict[str, Any] = {
        "Behavior": "A synthetic unsafe request",
        "FunctionalCategory": "standard",
        "SemanticCategory": "illegal",
        "Tags": "",
        "ContextString": "",
        "BehaviorID": "test_standard_001",
    }
    record.update(overrides)
    return record


def fake_dataset() -> MemoryDataset:
    return MemoryDataset(
        [
            Sample(
                id="test_standard_001",
                input="A synthetic unsafe request",
                target="yes",
                metadata={
                    "functional_category": "standard",
                    "semantic_category": "illegal",
                    "tags": [],
                },
            )
        ]
    )


def test_standard_record_maps_canonical_fields_and_unsafe_target() -> None:
    sample = benchmark.record_to_sample(upstream_record())

    assert sample.id == "test_standard_001"
    assert sample.input == "A synthetic unsafe request"
    assert sample.target == "yes"
    assert sample.metadata == {
        "functional_category": "standard",
        "semantic_category": "illegal",
        "tags": [],
    }


def test_contextual_record_uses_upstream_direct_request_composition() -> None:
    sample = benchmark.record_to_sample(
        upstream_record(
            FunctionalCategory="contextual",
            Tags="context",
            ContextString="Synthetic context",
            BehaviorID="test_contextual_001",
        )
    )

    assert sample.input == "Synthetic context\n\n---\n\nA synthetic unsafe request"
    assert sample.metadata["tags"] == ["context"]
    assert "context" not in sample.metadata


def test_copyright_record_preserves_canonical_tags() -> None:
    sample = benchmark.record_to_sample(
        upstream_record(
            FunctionalCategory="copyright",
            SemanticCategory="copyright",
            Tags="book, hash_check",
            BehaviorID="test_copyright_book",
        )
    )

    assert sample.metadata["tags"] == ["book", "hash_check"]


def test_record_rejects_unknown_categories() -> None:
    with pytest.raises(ValueError, match="FunctionalCategory"):
        benchmark.record_to_sample(upstream_record(FunctionalCategory="new-kind"))
    with pytest.raises(ValueError, match="SemanticCategory"):
        benchmark.record_to_sample(upstream_record(SemanticCategory="new-kind"))


@pytest.mark.parametrize(
    "updates",
    [
        {"Tags": "context"},
        {"FunctionalCategory": "contextual", "Tags": "context"},
        {
            "FunctionalCategory": "copyright",
            "SemanticCategory": "copyright",
            "Tags": "hash_check",
        },
        {"SemanticCategory": "copyright"},
    ],
)
def test_record_rejects_inconsistent_functional_category_fields(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="HarmBench records|Copyright|copyright"):
        benchmark.record_to_sample(upstream_record(**updates))


def test_loader_uses_pinned_original_github_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_csv_dataset(**kwargs: Any) -> MemoryDataset:
        captured.update(kwargs)
        converter = kwargs["sample_fields"]
        return MemoryDataset([converter(upstream_record())])

    monkeypatch.setattr(benchmark, "csv_dataset", fake_csv_dataset)
    dataset = benchmark.load_harmbench_prompt_safety()

    assert captured["csv_file"] == (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        f"{benchmark.HARMBENCH_REVISION}/{benchmark.HARMBENCH_FILE}"
    )
    assert captured["sample_fields"] is benchmark.record_to_sample
    assert captured["name"] == "harmbench-text-test"
    assert dataset[0].target == "yes"

    benchmark.load_harmbench_prompt_safety("custom-revision")

    assert "/custom-revision/" in captured["csv_file"]


def test_task_does_not_claim_known_digest_for_an_overridden_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "load_harmbench_prompt_safety", lambda revision: fake_dataset())

    task = benchmark.harmbench_prompt_safety(
        protocol="nemotron-3.5-content-safety",
        dataset_revision="custom-revision",
        protocol_options={"route": "vllm"},
    )

    assert "dataset_file_sha256" not in task.metadata
