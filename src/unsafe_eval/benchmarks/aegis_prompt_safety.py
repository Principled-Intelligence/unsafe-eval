"""Prompt-only safety evaluation on NVIDIA's Aegis 2.0 test split."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

AEGIS_REPOSITORY = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
AEGIS_CONFIG = "default"
AEGIS_SPLIT = "test"
AEGIS_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"

_LABEL_TO_TARGET = {"unsafe": "yes", "safe": "no"}


def record_to_samples(record: Mapping[str, Any]) -> list[Sample]:
    """Convert one human-labeled Aegis prompt to a sample, or omit a redacted prompt.

    Return an empty list when `prompt` is exactly `REDACTED`; otherwise return one sample.
    Map `unsafe` to target `yes` and `safe` to `no`. Response text and response labels
    are excluded. Missing fields, unsupported labels, and non-human label sources raise
    `ValueError`, including on rows whose prompt is redacted.
    """
    required = (
        "id",
        "reconstruction_id_if_redacted",
        "prompt",
        "prompt_label",
        "violated_categories",
        "prompt_label_source",
    )
    require_fields(record, required, benchmark="Aegis")

    prompt = require_string(record, "prompt", benchmark="Aegis")

    prompt_label = record["prompt_label"]
    if prompt_label not in _LABEL_TO_TARGET:
        raise ValueError(f"Unsupported Aegis prompt_label: {prompt_label!r}")
    if record["prompt_label_source"] != "human":
        raise ValueError("Aegis prompt_label_source must be 'human'")

    violated_categories = require_string(record, "violated_categories", benchmark="Aegis")

    if prompt == "REDACTED":
        return []

    return [
        Sample(
            id=str(record["id"]),
            input=prompt,
            target=_LABEL_TO_TARGET[prompt_label],
            metadata={
                "violated_categories": violated_categories,
            },
        )
    ]


def load_aegis_prompt_safety(dataset_revision: str = AEGIS_REVISION) -> Dataset:
    """Load the Aegis 2.0 test split, excluding prompts marked `REDACTED`.

    `dataset_revision` selects the source revision and defaults to the pinned commit.
    """
    return hf_dataset(
        path=AEGIS_REPOSITORY,
        name=AEGIS_CONFIG,
        split=AEGIS_SPLIT,
        revision=dataset_revision,
        sample_fields=record_to_samples,
        trust=False,
    )


@task
def aegis_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = AEGIS_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate prompt-only safe/unsafe classification on Aegis 2.0."""
    return build_prompt_safety_task(
        dataset=load_aegis_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="aegis-2.0",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "Aegis-AI-Content-Safety-Dataset-2.0",
            "dataset_repository": AEGIS_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_split": AEGIS_SPLIT,
            "evaluation_semantics": "prompt_only_binary_safety",
            "positive_class": "unsafe",
            "source_test_records": 1964,
            "excluded_redacted_records": 36,
            "expected_evaluated_records": 1928,
        },
    )
