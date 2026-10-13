"""Prompt-only safety evaluation on WildGuardMix's WildGuardTest split."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

WILDGUARD_MIX_REPOSITORY = "allenai/wildguardmix"
WILDGUARD_MIX_CONFIG = "wildguardtest"
WILDGUARD_MIX_SPLIT = "test"
WILDGUARD_MIX_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"

_PROMPT_HARM_TO_TARGET = {"harmful": "yes", "unharmful": "no"}


def record_to_samples(record: Mapping[str, Any]) -> list[Sample]:
    """Convert a WildGuardTest row to zero or one prompt-only Inspect samples."""
    required = (
        "prompt",
        "adversarial",
        "prompt_harm_label",
        "subcategory",
        "prompt_harm_agreement",
    )
    require_fields(record, required, benchmark="WildGuardMix")

    prompt = require_string(record, "prompt", benchmark="WildGuardMix")

    adversarial = record["adversarial"]
    if not isinstance(adversarial, bool):
        raise ValueError("WildGuardMix adversarial must be a boolean")

    subcategory = require_string(record, "subcategory", benchmark="WildGuardMix")

    prompt_harm_label = record["prompt_harm_label"]
    prompt_harm_agreement = record["prompt_harm_agreement"]
    if prompt_harm_label is None:
        if prompt_harm_agreement is not None:
            raise ValueError(
                "WildGuardMix prompt_harm_agreement must be null when prompt_harm_label is null"
            )
        return []

    if not isinstance(prompt_harm_label, str) or prompt_harm_label not in _PROMPT_HARM_TO_TARGET:
        raise ValueError(f"Unsupported WildGuardMix prompt_harm_label: {prompt_harm_label!r}")
    if (
        isinstance(prompt_harm_agreement, bool)
        or not isinstance(prompt_harm_agreement, (int, float))
        or prompt_harm_agreement not in (2, 3)
    ):
        raise ValueError("WildGuardMix prompt_harm_agreement must be 2 or 3")

    return [
        Sample(
            input=prompt,
            target=_PROMPT_HARM_TO_TARGET[prompt_harm_label],
            metadata={
                "adversarial": adversarial,
                "prompt_type": "adversarial" if adversarial else "vanilla",
                "subcategory": subcategory,
                "prompt_harm_agreement": int(prompt_harm_agreement),
            },
        )
    ]


def load_wildguard_mix_prompt_safety(
    dataset_revision: str = WILDGUARD_MIX_REVISION,
) -> Dataset:
    """Load agreed prompt-harm labels from the pinned WildGuardTest split."""
    return hf_dataset(
        path=WILDGUARD_MIX_REPOSITORY,
        name=WILDGUARD_MIX_CONFIG,
        split=WILDGUARD_MIX_SPLIT,
        revision=dataset_revision,
        sample_fields=record_to_samples,
        auto_id=True,
        trust=False,
    )


@task
def wildguard_mix_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = WILDGUARD_MIX_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate prompt-only harmfulness classification on WildGuardTest."""
    return build_prompt_safety_task(
        dataset=load_wildguard_mix_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="wildguardmix-test",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "WildGuardTest (WildGuardMix)",
            "dataset_repository": WILDGUARD_MIX_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_config": WILDGUARD_MIX_CONFIG,
            "dataset_split": WILDGUARD_MIX_SPLIT,
            "evaluation_semantics": "prompt_only_binary_safety",
            "positive_class": "unsafe",
            "dataset_filter": "prompt_harm_label!=null",
            "source_test_records": 1725,
            "excluded_no_agreement_records": 26,
            "evaluated_test_records": 1699,
            "unsafe_test_records": 754,
            "safe_test_records": 945,
        },
    )
