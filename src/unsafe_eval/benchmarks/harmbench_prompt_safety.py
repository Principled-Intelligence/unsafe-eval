"""Inspect task for the original HarmBench text prompt-safety test set."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, csv_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

HARMBENCH_REPOSITORY = "centerforaisafety/HarmBench"
HARMBENCH_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
HARMBENCH_FILE = "data/behavior_datasets/harmbench_behaviors_text_test.csv"
HARMBENCH_SPLIT = "text_test"
HARMBENCH_FILE_SHA256 = "75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29"
HARMBENCH_SOURCE_RECORDS = 320

_FUNCTIONAL_CATEGORIES = frozenset({"standard", "contextual", "copyright"})
_SEMANTIC_CATEGORIES = frozenset(
    {
        "chemical_biological",
        "copyright",
        "cybercrime_intrusion",
        "harassment_bullying",
        "harmful",
        "illegal",
        "misinformation_disinformation",
    }
)
_COPYRIGHT_TAGS = {("book", "hash_check"), ("lyrics", "hash_check")}


def record_to_sample(record: Mapping[str, Any]) -> Sample:
    """Convert one canonical HarmBench text-test row to an unsafe-target sample."""
    required = (
        "Behavior",
        "FunctionalCategory",
        "SemanticCategory",
        "Tags",
        "ContextString",
        "BehaviorID",
    )
    require_fields(record, required, benchmark="HarmBench")

    behavior = require_string(record, "Behavior", benchmark="HarmBench", non_empty=True)
    functional_category = require_string(record, "FunctionalCategory", benchmark="HarmBench")
    semantic_category = require_string(record, "SemanticCategory", benchmark="HarmBench")
    tags_value = require_string(record, "Tags", benchmark="HarmBench")
    context = require_string(record, "ContextString", benchmark="HarmBench")
    behavior_id = require_string(record, "BehaviorID", benchmark="HarmBench", non_empty=True)

    if functional_category not in _FUNCTIONAL_CATEGORIES:
        raise ValueError(f"Unsupported HarmBench FunctionalCategory: {functional_category!r}")
    if semantic_category not in _SEMANTIC_CATEGORIES:
        raise ValueError(f"Unsupported HarmBench SemanticCategory: {semantic_category!r}")

    tags = tuple(part.strip() for part in tags_value.split(",") if part.strip())
    if functional_category == "standard" and (context or tags):
        raise ValueError("Standard HarmBench records must not contain context or tags")
    if functional_category == "contextual" and (not context or tags != ("context",)):
        raise ValueError("Contextual HarmBench records require context and the context tag")
    if functional_category == "copyright" and (
        context or tags not in _COPYRIGHT_TAGS or semantic_category != "copyright"
    ):
        raise ValueError("Copyright HarmBench records have an invalid context, tags, or category")
    if functional_category != "copyright" and semantic_category == "copyright":
        raise ValueError("Only copyright HarmBench records may use the copyright category")

    prompt = f"{context}\n\n---\n\n{behavior}" if functional_category == "contextual" else behavior
    return Sample(
        id=behavior_id,
        input=prompt,
        target="yes",
        metadata={
            "functional_category": functional_category,
            "semantic_category": semantic_category,
            "tags": list(tags),
        },
    )


def load_harmbench_prompt_safety(
    dataset_revision: str = HARMBENCH_REVISION,
) -> Dataset:
    """Load the original HarmBench text test set from an immutable GitHub revision."""
    return csv_dataset(
        csv_file=(
            "https://raw.githubusercontent.com/"
            f"{HARMBENCH_REPOSITORY}/{dataset_revision}/{HARMBENCH_FILE}"
        ),
        sample_fields=record_to_sample,
        name="harmbench-text-test",
    )


@task
def harmbench_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = HARMBENCH_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate binary unsafe-prompt detection on the HarmBench text test set."""
    file_metadata = (
        {"dataset_file_sha256": HARMBENCH_FILE_SHA256}
        if dataset_revision == HARMBENCH_REVISION
        else {}
    )
    return build_prompt_safety_task(
        dataset=load_harmbench_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="harmbench-text-test",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "HarmBench Text Test",
            "dataset_repository": HARMBENCH_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_file": HARMBENCH_FILE,
            "dataset_split": HARMBENCH_SPLIT,
            "source_records": HARMBENCH_SOURCE_RECORDS,
            "evaluated_records": HARMBENCH_SOURCE_RECORDS,
            "evaluation_semantics": "unsafe_prompt_detection",
            "all_samples_are_unsafe": True,
            **file_metadata,
        },
    )
