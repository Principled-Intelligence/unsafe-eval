from pathlib import Path
from typing import Any

import pytest
from inspect_ai.dataset import Sample

from tests.integration._support import (
    PROTOCOLS,
    configure_test_environment,
    run_task,
    task_with_dataset,
)


def test_grouped_metrics_flow_through_the_inspect_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise shared task assembly without repeating it for every benchmark."""
    configure_test_environment(tmp_path, monkeypatch, "grouped-metrics")
    task = task_with_dataset(
        module_name="harmbench_prompt_safety",
        loader_name="load_harmbench_prompt_safety",
        task_name="harmbench_prompt_safety",
        samples=[
            Sample(
                input="unsafe-example",
                target="yes",
                metadata={
                    "functional_category": "standard",
                    "semantic_category": "illegal",
                },
            )
        ],
        protocol=PROTOCOLS["shieldstral"],
        monkeypatch=monkeypatch,
    )
    observed: dict[str, Any] = {}

    results = run_task(
        task,
        profile="shieldstral",
        tmp_path=tmp_path,
        observed=observed,
    )

    scores = {score.name: score for score in results.scores}
    assert scores["label"].metrics["unsafe_eval/unsafe_recall"].value == 1.0
    probability_metrics = scores["unsafe_probability"].metrics
    assert probability_metrics["unsafe_eval/grouped_functional_category_standard"].value > 0.5
    assert probability_metrics["unsafe_eval/grouped_semantic_category_illegal"].value > 0.5
