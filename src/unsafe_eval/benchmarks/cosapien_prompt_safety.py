"""Policy-conditioned prompt-safety evaluation on Microsoft's CoSApien."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

COSAPIEN_REPOSITORY = "microsoft/CoSApien"
COSAPIEN_CONFIG = "default"
COSAPIEN_SPLITS = (
    "game_development",
    "public_prosecutor",
    "book_publisher_arab",
    "language_learning",
    "film_production",
)
COSAPIEN_REVISION = "29bb85bd4f84319d825517519278c8f8959fa031"
COSAPIEN_TOTAL_RECORDS = 200
COSAPIEN_TYPE_COUNTS = {
    "safe": 52,
    "allowed": 64,
    "disallowed": 63,
    "partial": 21,
}

_TYPE_TO_TARGET = {
    "safe": "no",
    "allowed": "no",
    "disallowed": "yes",
    "partial": "yes",
}


def record_to_sample(record: Mapping[str, Any]) -> Sample:
    """Convert a CoSApien prompt using its scenario text as the sample's safety policy.

    Store the scenario text in `safety_policy` metadata for the protocol to render.
    Types `safe` and `allowed` map to target `no`; `disallowed` and `partial` map to
    `yes`, so a request with any disallowed part counts as unsafe.
    """
    required = ("prompt", "scenario", "type")
    require_fields(record, required, benchmark="CoSApien")

    prompt = require_string(record, "prompt", benchmark="CoSApien", non_empty=True)

    safety_policy = require_string(record, "scenario", benchmark="CoSApien", non_empty=True)

    prompt_type = record["type"]
    if not isinstance(prompt_type, str) or prompt_type not in _TYPE_TO_TARGET:
        raise ValueError(f"Unsupported CoSApien type: {prompt_type!r}")

    return Sample(
        input=prompt,
        target=_TYPE_TO_TARGET[prompt_type],
        metadata={
            "safety_policy": safety_policy,
            "prompt_type": prompt_type,
        },
    )


def load_cosapien_prompt_safety(
    dataset_revision: str = COSAPIEN_REVISION,
) -> Dataset:
    """Load and combine all five pinned CoSApien scenario splits."""
    samples: list[Sample] = []
    for scenario in COSAPIEN_SPLITS:
        scenario_dataset = hf_dataset(
            path=COSAPIEN_REPOSITORY,
            name=COSAPIEN_CONFIG,
            split=scenario,
            revision=dataset_revision,
            sample_fields=record_to_sample,
            trust=False,
        )
        for index, sample in enumerate(scenario_dataset, start=1):
            if sample.metadata is None:
                raise ValueError("Converted CoSApien sample must retain metadata")
            sample.id = f"{scenario}-{index}"
            sample.metadata["scenario"] = scenario
            samples.append(sample)
    return MemoryDataset(
        samples,
        name=COSAPIEN_REPOSITORY,
        location=COSAPIEN_REPOSITORY,
    )


@task
def cosapien_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = COSAPIEN_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate binary safety under each CoSApien scenario-specific policy."""
    return build_prompt_safety_task(
        dataset=load_cosapien_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="cosapien",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "CoSApien",
            "dataset_repository": COSAPIEN_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_config": COSAPIEN_CONFIG,
            "dataset_splits": list(COSAPIEN_SPLITS),
            "evaluation_semantics": "policy_conditioned_prompt_only_binary_safety",
            "positive_class": "unsafe",
            "unsafe_aggregation": "any_disallowed_subrequest",
            "label_mapping": dict(_TYPE_TO_TARGET),
            "source_records": COSAPIEN_TOTAL_RECORDS,
            "evaluated_records": COSAPIEN_TOTAL_RECORDS,
            "unsafe_records": 84,
            "safe_records": 116,
            "source_type_counts": dict(COSAPIEN_TYPE_COUNTS),
        },
    )
