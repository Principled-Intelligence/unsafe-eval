from importlib import import_module

import pytest
from inspect_ai.dataset import MemoryDataset, Sample


@pytest.mark.parametrize(
    ("name", "primary_metric"),
    [
        ("simple_safety_tests", "unsafe_recall"),
        ("harmbench_prompt_safety", "unsafe_recall"),
        ("aegis_prompt_safety", "unsafe_f1"),
        ("cosapien_prompt_safety", "unsafe_f1"),
        ("dynaguardrail_prompt_safety", "unsafe_f1"),
        ("openai_moderation_prompt_safety", "unsafe_f1"),
        ("toxic_chat_prompt_safety", "unsafe_f1"),
        ("wildguard_mix_prompt_safety", "unsafe_f1"),
    ],
)
def test_task_passes_the_revision_through_and_leads_with_its_primary_metric(
    monkeypatch: pytest.MonkeyPatch, name: str, primary_metric: str
) -> None:
    benchmark = import_module(f"unsafe_eval.benchmarks.{name}")
    dataset = MemoryDataset([Sample(input="Prompt", target="yes")])
    monkeypatch.setattr(benchmark, f"load_{name}", lambda revision: dataset)

    task = getattr(benchmark, name)(protocol="shieldstral-1.0-3b", dataset_revision="custom")

    assert task.metadata["dataset_revision"] == "custom"
    assert task.metrics is not None
    metric = next(iter(task.metrics["label"]))
    assert metric.__registry_info__.name == f"unsafe_eval/{primary_metric}"
