"""Inspect task for the SimpleSafetyTests harmful-prompt suite."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

SIMPLE_SAFETY_TESTS_REPOSITORY = "Bertievidgen/SimpleSafetyTests"
SIMPLE_SAFETY_TESTS_CONFIG = "default"
SIMPLE_SAFETY_TESTS_SPLIT = "test"
SIMPLE_SAFETY_TESTS_REVISION = "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5"


def record_to_sample(record: Mapping[str, Any]) -> Sample:
    """Convert a SimpleSafetyTests row to an unsafe-target Inspect sample."""
    required = ("id", "prompt", "harm_area", "category", "counter")
    require_fields(record, required, benchmark="SimpleSafetyTests")

    prompt = require_string(record, "prompt", benchmark="SimpleSafetyTests")

    return Sample(
        id=str(record["id"]),
        input=prompt,
        target="yes",
        metadata={
            "harm_area": record["harm_area"],
            "category": record["category"],
            "counter": record["counter"],
        },
    )


def load_simple_safety_tests(
    dataset_revision: str = SIMPLE_SAFETY_TESTS_REVISION,
) -> Dataset:
    """Load SimpleSafetyTests from an immutable Hugging Face revision."""
    return hf_dataset(
        path=SIMPLE_SAFETY_TESTS_REPOSITORY,
        name=SIMPLE_SAFETY_TESTS_CONFIG,
        split=SIMPLE_SAFETY_TESTS_SPLIT,
        revision=dataset_revision,
        sample_fields=record_to_sample,
        trust=False,
    )


@task
def simple_safety_tests(
    protocol: ProtocolName,
    dataset_revision: str = SIMPLE_SAFETY_TESTS_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate binary unsafe-prompt detection on SimpleSafetyTests."""
    return build_prompt_safety_task(
        dataset=load_simple_safety_tests(dataset_revision),
        protocol=protocol,
        benchmark="simple-safety-tests",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "SimpleSafetyTests",
            "dataset_repository": SIMPLE_SAFETY_TESTS_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_split": SIMPLE_SAFETY_TESTS_SPLIT,
            "evaluation_semantics": "unsafe_prompt_detection",
            "all_samples_are_unsafe": True,
        },
    )
