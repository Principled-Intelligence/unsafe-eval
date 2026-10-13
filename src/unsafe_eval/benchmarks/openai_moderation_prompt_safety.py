"""Prompt-only safety evaluation on the OpenAI Moderation test set."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

OPENAI_MODERATION_REPOSITORY = "openai/moderation-api-release"
OPENAI_MODERATION_REVISION = "f4ab51b5edd3bfbcb349a56324274235b674e0e4"
OPENAI_MODERATION_FILE = "data/samples-1680.jsonl.gz"
OPENAI_MODERATION_SPLIT = "test"
OPENAI_MODERATION_FILE_SHA256 = "ef3f7de86ee394fe337b0a566ba4b9c7db058f3caa8d6a4a0006138b663ab9a8"

OPENAI_MODERATION_CATEGORIES = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")


def record_to_samples(record: Mapping[str, Any]) -> list[Sample]:
    """Convert an OpenAI Moderation row to zero or one binary Inspect samples."""
    require_fields(record, ("prompt",), benchmark="OpenAI Moderation")

    prompt = require_string(record, "prompt", benchmark="OpenAI Moderation", non_empty=True)

    category_labels: dict[str, int] = {}
    for category in OPENAI_MODERATION_CATEGORIES:
        value = record.get(category)
        if value is None:
            continue
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"Unsupported OpenAI Moderation {category} label: {value!r}")
        category_labels[category] = value

    if any(label == 1 for label in category_labels.values()):
        target = "yes"
    elif len(category_labels) == len(OPENAI_MODERATION_CATEGORIES):
        target = "no"
    else:
        return []

    return [
        Sample(
            input=prompt,
            target=target,
            metadata={"category_labels": category_labels},
        )
    ]


def load_openai_moderation_prompt_safety(
    dataset_revision: str = OPENAI_MODERATION_REVISION,
) -> Dataset:
    """Load binary-determinate rows from the pinned OpenAI Moderation test set."""
    return hf_dataset(
        path="json",
        split="train",
        data_files=(
            "https://raw.githubusercontent.com/"
            f"{OPENAI_MODERATION_REPOSITORY}/{dataset_revision}/{OPENAI_MODERATION_FILE}"
        ),
        sample_fields=record_to_samples,
        auto_id=True,
        trust=False,
    )


@task
def openai_moderation_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = OPENAI_MODERATION_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate prompt-only binary safety on the OpenAI Moderation test set."""
    file_metadata = (
        {"dataset_file_sha256": OPENAI_MODERATION_FILE_SHA256}
        if dataset_revision == OPENAI_MODERATION_REVISION
        else {}
    )
    return build_prompt_safety_task(
        dataset=load_openai_moderation_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="openai-moderation",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "OpenAI Moderation",
            "dataset_repository": OPENAI_MODERATION_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_file": OPENAI_MODERATION_FILE,
            "dataset_split": OPENAI_MODERATION_SPLIT,
            "evaluation_semantics": "prompt_only_binary_safety",
            "positive_class": "unsafe",
            "dataset_filter": (
                "any known category positive, or all eight categories known negative"
            ),
            "source_records": 1680,
            "evaluated_records": 859,
            "unsafe_records": 522,
            "safe_records": 337,
            **file_metadata,
        },
    )
