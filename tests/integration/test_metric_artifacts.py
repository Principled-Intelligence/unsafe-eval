import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.log import list_eval_logs

from tests.integration._support import (
    PROTOCOLS,
    configure_test_environment,
    run_task,
    task_with_dataset,
)


def test_eval_automatically_writes_lightweight_metric_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_test_environment(tmp_path, monkeypatch, "metric-artifact")
    task = task_with_dataset(
        module_name="simple_safety_tests",
        loader_name="load_simple_safety_tests",
        task_name="simple_safety_tests",
        samples=[
            Sample(
                input="unsafe-example",
                target="yes",
                metadata={"harm_area": "test-area", "category": "test-category"},
            )
        ],
        protocol=PROTOCOLS["shieldstral"],
        monkeypatch=monkeypatch,
    )
    observed: dict[str, Any] = {}
    log_dir = tmp_path / "logs" / "shieldstral-1.0-3b"

    run_task(
        task,
        profile="shieldstral",
        tmp_path=log_dir,
        observed=observed,
    )

    artifacts = list(log_dir.glob("metrics-*.json"))
    assert len(artifacts) == 1
    assert not list((tmp_path / "logs").glob("metrics-*.json"))
    artifact_text = artifacts[0].read_text()
    artifact = json.loads(artifact_text)
    assert artifact["status"] == "success"
    assert artifact["model"]["name"] == "mockllm/model"
    assert artifact["metadata"]["benchmark_id"] == "simple-safety-tests"
    assert artifact["metadata"]["protocol"] == "shieldstral-1.0-3b"
    assert artifact["samples"] == {"completed": 1, "total": 1}
    assert artifact["scores"][0]["metrics"]["unsafe_eval/unsafe_recall"] == 1.0
    assert "unsafe-example" not in artifact_text

    native_logs = list_eval_logs(str(log_dir))
    assert len(native_logs) == 1
    assert native_logs[0].name.endswith(".eval")
