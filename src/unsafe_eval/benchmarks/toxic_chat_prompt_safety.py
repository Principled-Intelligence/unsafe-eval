"""Prompt-only safety evaluation on ToxicChat 0124's test split."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

TOXIC_CHAT_REPOSITORY = "lmsys/toxic-chat"
TOXIC_CHAT_CONFIG = "toxicchat0124"
TOXIC_CHAT_SPLIT = "test"
TOXIC_CHAT_REVISION = "29df8e4dba60e1f4af4b4075c0705c5b313548a8"

_TOXICITY_TO_TARGET = {0: "no", 1: "yes"}


def record_to_sample(record: Mapping[str, Any]) -> Sample:
    """Use `user_input` as the prompt and map toxicity 1 to `yes`, 0 to `no`.

    Retain annotation and jailbreak flags in metadata. This adapter accepts rows without
    human annotation; `load_toxic_chat_prompt_safety` filters those rows out.
    """
    required = (
        "conv_id",
        "user_input",
        "human_annotation",
        "toxicity",
        "jailbreaking",
    )
    require_fields(record, required, benchmark="ToxicChat")

    conv_id = require_string(record, "conv_id", benchmark="ToxicChat")

    user_input = require_string(record, "user_input", benchmark="ToxicChat")

    human_annotation = record["human_annotation"]
    if not isinstance(human_annotation, bool):
        raise ValueError("ToxicChat human_annotation must be a boolean")

    toxicity = record["toxicity"]
    if type(toxicity) is not int or toxicity not in _TOXICITY_TO_TARGET:
        raise ValueError(f"Unsupported ToxicChat toxicity label: {toxicity!r}")

    jailbreaking = record["jailbreaking"]
    if type(jailbreaking) is not int or jailbreaking not in (0, 1):
        raise ValueError(f"Unsupported ToxicChat jailbreaking label: {jailbreaking!r}")

    return Sample(
        id=conv_id,
        input=user_input,
        target=_TOXICITY_TO_TARGET[toxicity],
        metadata={
            "human_annotation": human_annotation,
            "jailbreaking": jailbreaking,
        },
    )


def load_toxic_chat_prompt_safety(
    dataset_revision: str = TOXIC_CHAT_REVISION,
) -> Dataset:
    """Load human-annotated ToxicChat 0124 test records from an immutable revision."""
    dataset = hf_dataset(
        path=TOXIC_CHAT_REPOSITORY,
        name=TOXIC_CHAT_CONFIG,
        split=TOXIC_CHAT_SPLIT,
        revision=dataset_revision,
        sample_fields=record_to_sample,
        trust=False,
    )
    return dataset.filter(
        lambda sample: (
            sample.metadata is not None and sample.metadata.get("human_annotation") is True
        )
    )


@task
def toxic_chat_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = TOXIC_CHAT_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate prompt-only safe/unsafe classification on ToxicChat 0124."""
    return build_prompt_safety_task(
        dataset=load_toxic_chat_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="toxic-chat-0124",
        protocol_options=protocol_options,
        version=3,
        metadata={
            "benchmark": "ToxicChat 0124",
            "dataset_repository": TOXIC_CHAT_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_config": TOXIC_CHAT_CONFIG,
            "dataset_split": TOXIC_CHAT_SPLIT,
            "evaluation_semantics": "prompt_only_binary_safety",
            "positive_class": "unsafe",
            "dataset_filter": "human_annotation=true",
            "source_test_records": 5083,
            "evaluated_test_records": 2853,
            "unsafe_test_records": 362,
            "safe_test_records": 2491,
        },
    )
