"""Policy-conditioned prompt-safety evaluation on DynaGuardrail."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval._resources import content_sha256, load_text_resource
from unsafe_eval.benchmarks._validation import require_fields, require_string
from unsafe_eval.protocols import ProtocolName
from unsafe_eval.task import build_prompt_safety_task

DYNA_GUARDRAIL_REPOSITORY = "dynamofl/DynaGuardrail"
DYNA_GUARDRAIL_CONFIG = "default"
DYNA_GUARDRAIL_SPLIT = "train"
DYNA_GUARDRAIL_REVISION = "3c21998805b692819e527f88702f761e33f6a013"
DYNA_GUARDRAIL_DATA_FILE = "data/train-00000-of-00001.parquet"
DYNA_GUARDRAIL_TOTAL_RECORDS = 2362
DYNA_GUARDRAIL_DOMAIN_COUNTS = {
    "finance": {"records": 500, "unsafe": 261, "safe": 239},
    "injection": {"records": 951, "unsafe": 449, "safe": 502},
    "safety": {"records": 546, "unsafe": 257, "safe": 289},
    "tax": {"records": 365, "unsafe": 162, "safe": 203},
}

DYNA_GUARDRAIL_POLICIES = {
    domain: load_text_resource("benchmarks", "policies", "dynaguardrail", f"{domain}.txt")
    for domain in DYNA_GUARDRAIL_DOMAIN_COUNTS
}
DYNA_GUARDRAIL_POLICY_RESOURCE_HASHES = {
    f"benchmark-policy/{domain}.txt": content_sha256(policy)
    for domain, policy in DYNA_GUARDRAIL_POLICIES.items()
}

_LABEL_TO_TARGET = {"unsafe": "yes", "safe": "no"}


def record_to_sample(record: Mapping[str, Any]) -> Sample:
    """Convert one DynaGuardrail row to a policy-conditioned Inspect sample."""
    required = (
        "prompt",
        "label",
        "label_detailed",
        "label_ai_predicted",
        "classes",
        "trickiness",
        "num_words",
        "abnormal_syntax",
        "source",
        "dataset_type",
    )
    require_fields(record, required, benchmark="DynaGuardrail")

    prompt = require_string(record, "prompt", benchmark="DynaGuardrail", non_empty=True)

    label = record["label"]
    if not isinstance(label, str) or label not in _LABEL_TO_TARGET:
        raise ValueError(f"Unsupported DynaGuardrail label: {label!r}")

    dataset_type = record["dataset_type"]
    if not isinstance(dataset_type, str) or dataset_type not in DYNA_GUARDRAIL_POLICIES:
        raise ValueError(f"Unsupported DynaGuardrail dataset_type: {dataset_type!r}")

    string_fields = (
        "label_detailed",
        "label_ai_predicted",
        "classes",
        "trickiness",
        "source",
    )
    for field in string_fields:
        require_string(record, field, benchmark="DynaGuardrail")

    num_words = record["num_words"]
    if isinstance(num_words, bool) or not isinstance(num_words, int) or num_words < 0:
        raise ValueError("DynaGuardrail num_words must be a non-negative integer")

    abnormal_syntax = record["abnormal_syntax"]
    if not isinstance(abnormal_syntax, bool):
        raise ValueError("DynaGuardrail abnormal_syntax must be a boolean")

    return Sample(
        input=prompt,
        target=_LABEL_TO_TARGET[label],
        metadata={
            "safety_policy": DYNA_GUARDRAIL_POLICIES[dataset_type],
            "policy_domain": dataset_type,
            "classes": record["classes"],
            "trickiness": record["trickiness"],
            "num_words": num_words,
            "abnormal_syntax": abnormal_syntax,
            "source": record["source"],
        },
    )


def load_dynaguardrail_prompt_safety(
    dataset_revision: str = DYNA_GUARDRAIL_REVISION,
) -> Dataset:
    """Load all four policy domains from the pinned DynaGuardrail aggregate split."""
    data_file = (
        f"hf://datasets/{DYNA_GUARDRAIL_REPOSITORY}@{dataset_revision}/{DYNA_GUARDRAIL_DATA_FILE}"
    )
    return hf_dataset(
        # The repository's legacy dataset_infos.json is incompatible with datasets 4.x.
        # Read the default config's aggregate shard directly with the generic builder.
        path="parquet",
        split=DYNA_GUARDRAIL_SPLIT,
        data_files={DYNA_GUARDRAIL_SPLIT: data_file},
        sample_fields=record_to_sample,
        auto_id=True,
        trust=False,
    )


@task
def dynaguardrail_prompt_safety(
    protocol: ProtocolName,
    dataset_revision: str = DYNA_GUARDRAIL_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate binary safety under DynaGuardrail's four domain policies."""
    return build_prompt_safety_task(
        dataset=load_dynaguardrail_prompt_safety(dataset_revision),
        protocol=protocol,
        benchmark="dynaguardrail",
        protocol_options=protocol_options,
        version=2,
        metadata={
            "benchmark": "DynaGuardrail",
            "dataset_repository": DYNA_GUARDRAIL_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_config": DYNA_GUARDRAIL_CONFIG,
            "dataset_split": DYNA_GUARDRAIL_SPLIT,
            "evaluation_semantics": "policy_conditioned_prompt_only_binary_safety",
            "positive_class": "unsafe",
            "label_mapping": dict(_LABEL_TO_TARGET),
            "source_records": DYNA_GUARDRAIL_TOTAL_RECORDS,
            "evaluated_records": DYNA_GUARDRAIL_TOTAL_RECORDS,
            "unsafe_records": 1129,
            "safe_records": 1233,
            "domain_counts": DYNA_GUARDRAIL_DOMAIN_COUNTS,
            "policy_resource_hashes": DYNA_GUARDRAIL_POLICY_RESOURCE_HASHES,
        },
    )
