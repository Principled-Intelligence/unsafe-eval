import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.hooks import RunEnd, RunStart
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalError,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
    EvalStats,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage

from unsafe_eval import _metric_artifacts
from unsafe_eval._metric_artifacts import MetricsExportHooks, metric_artifact
from unsafe_eval.cli._run import INSPECT_ARGS_ENV


def make_log(
    *,
    status: str = "success",
    results: EvalResults | None = None,
    location: str = "/tmp/run.eval",
    benchmark_id: str | None = "aegis-2.0",
    eval_id: str = "eval-1",
) -> EvalLog:
    metadata = {
        "benchmark": "Aegis",
        "protocol": "scope-guard-2601",
        "dataset_revision": "dataset-commit",
        "protocol_options": {"include_evidence": False},
    }
    if benchmark_id is not None:
        metadata["benchmark_id"] = benchmark_id
    return EvalLog(
        status=status,  # type: ignore[arg-type]
        eval=EvalSpec(
            eval_id=eval_id,
            run_id="run-1",
            created="2026-08-17T12:00:00+00:00",
            task="aegis_prompt_safety",
            task_registry_name="unsafe_eval/aegis_prompt_safety",
            task_version=2,
            dataset=EvalDataset(name="aegis", samples=4),
            model="vllm/example/model",
            model_args={"revision": "model-commit", "api_key": "excluded"},
            config=EvalConfig(),
            metadata=metadata,
        ),
        results=results,
        stats=EvalStats(
            started_at="2026-08-17T12:00:01+00:00",
            completed_at="2026-08-17T12:01:00+00:00",
        ),
        location=location,
    )


def sample_results() -> EvalResults:
    return EvalResults(
        total_samples=4,
        completed_samples=4,
        scores=[
            EvalScore(
                name="label",
                scorer="unsafe_eval/safety_score",
                scored_samples=4,
                unscored_samples=0,
                metrics={
                    "unsafe_eval/balanced_accuracy": EvalMetric(
                        name="balanced_accuracy", value=0.75
                    ),
                    "unsafe_eval/grouped_category_unsafe": EvalMetric(
                        name="unsafe", group="category", value=1
                    ),
                    "unsafe_eval/unavailable": EvalMetric(name="unavailable", value=math.nan),
                },
            ),
            EvalScore(
                name="verdict",
                scorer="unsafe_eval/safety_score",
                reducer="mean",
                metrics={"mean": EvalMetric(name="mean", value=0.5)},
            ),
        ],
    )


def test_metric_artifact_is_strict_sample_free_summary() -> None:
    artifact = metric_artifact(make_log(results=sample_results()))

    assert artifact["schema_version"] == 5
    assert artifact["run_id"] == "run-1"
    assert artifact["eval_id"] == "eval-1"
    assert artifact["task"] == {
        "name": "aegis_prompt_safety",
        "registry_name": "unsafe_eval/aegis_prompt_safety",
        "version": 2,
    }
    assert artifact["model"] == {
        "name": "vllm/example/model",
        "revision": "model-commit",
    }
    assert artifact["metadata"]["benchmark_id"] == "aegis-2.0"
    assert artifact["samples"] == {"total": 4, "completed": 4}
    assert artifact["selection"] == {"limit": None, "sample_id": None}
    assert artifact["scores"][0]["metrics"] == {
        "unsafe_eval/balanced_accuracy": 0.75,
        "unsafe_eval/grouped_category_unsafe": 1,
        "unsafe_eval/unavailable": None,
    }
    assert artifact["scores"][1]["metrics"] == {"mean": 0.5}
    serialized = json.dumps(artifact, allow_nan=False)
    assert "api_key" not in serialized


def test_metric_artifact_records_provenance_and_forwarded_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _metric_artifacts, "provenance", lambda root: {"unsafe_eval_version": "1.0.0"}
    )
    monkeypatch.setenv(INSPECT_ARGS_ENV, '["--max-connections", "8"]')

    artifact = metric_artifact(make_log(results=sample_results()))

    assert artifact["provenance"] == {"unsafe_eval_version": "1.0.0"}
    assert artifact["inspect_args"] == ["--max-connections", "8"]
    monkeypatch.delenv(INSPECT_ARGS_ENV)
    assert metric_artifact(make_log(results=sample_results()))["inspect_args"] is None


START = datetime(2026, 1, 1, tzinfo=UTC)


def model_call(model: str, seconds: float, ends_at: float) -> ModelEvent:
    output = ModelOutput.from_content(model, "safe")
    output.time = seconds
    return ModelEvent(
        model=model,
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output,
        completed=START + timedelta(seconds=ends_at),
    )


def timed_sample(sample_id: int, *calls: ModelEvent, error: bool = False) -> EvalSample:
    return EvalSample(
        id=sample_id,
        epoch=1,
        input="prompt",
        target="safe",
        events=list(calls),
        error=EvalError(message="x", traceback="", traceback_ansi="") if error else None,
    )


def test_usage_and_latency_count_the_evaluated_model_from_the_log_file(tmp_path: Path) -> None:
    model, judge = "vllm/example/model", "openai/judge"
    log = make_log(results=sample_results(), location=str(tmp_path / "run.eval"))
    log.stats.model_usage = {
        model: ModelUsage(
            input_tokens=30,
            output_tokens=3,
            input_tokens_cache_write=20,
            input_tokens_cache_read=10,
            total_cost=0.5,
        ),
        judge: ModelUsage(input_tokens=99, output_tokens=9, total_cost=9.0),
    }
    # Calls overlap, as batched requests do; the first starts at 10 s, once any server is up.
    log.samples = [
        timed_sample(
            1,
            model_call(model, 1.0, ends_at=11.0),
            model_call(model, 0.5, ends_at=12.0),
            model_call(judge, 7.0, ends_at=20.0),
        ),
        timed_sample(2, model_call(model, 2.5, ends_at=13.0)),
        timed_sample(3, model_call(model, 60.0, ends_at=70.0), error=True),
    ]
    log.eval.dataset.sample_ids = [1, 2, 3]
    log.eval.model_generate_config = GenerateConfig(max_connections=1)
    write_eval_log(log, log.location)

    artifact = metric_artifact(read_eval_log(log.location, header_only=True))

    assert artifact["usage"] == {
        "input_tokens": 30,
        "input_tokens_cache_write": 20,
        "input_tokens_cache_read": 10,
        "output_tokens": 3,
        "reasoning_tokens": None,
        "total_cost": 0.5,
    }
    assert artifact["latency"] == {
        "seconds_per_sample": 1.5,
        "inference_seconds": 3.0,
        "timed_samples": 2,
        "median_seconds": 2.0,
        "p95_seconds": 2.5,
    }
    assert artifact["max_connections"] == 1


def test_usage_and_latency_are_null_without_usage_or_samples() -> None:
    artifact = metric_artifact(make_log(results=sample_results(), location="/missing/run.eval"))

    assert (artifact["usage"], artifact["latency"]) == (None, None)


def test_metric_artifact_records_how_samples_were_selected() -> None:
    log = make_log(results=sample_results())
    log.eval.config = EvalConfig(limit=(5, 15), sample_id=["a", 3])

    artifact = json.loads(json.dumps(metric_artifact(log), allow_nan=False))

    assert artifact["selection"] == {"limit": [5, 15], "sample_id": ["a", 3]}


def test_hook_exports_each_unsafe_eval_log_and_ignores_other_tasks(
    tmp_path: Path,
) -> None:
    first = make_log(
        results=sample_results(),
        location=str(tmp_path / "2026-first.eval"),
        eval_id="eval-first",
    )
    second = make_log(
        status="cancelled",
        location=str(tmp_path / "2026-second.eval"),
        eval_id="eval-second",
    )
    unrelated = make_log(
        results=sample_results(),
        location=str(tmp_path / "2026-other.eval"),
        benchmark_id=None,
        eval_id="eval-other",
    )

    asyncio.run(
        MetricsExportHooks().on_run_end(
            RunEnd(
                eval_set_id=None,
                run_id="run-1",
                exception=None,
                logs=[first, second, unrelated],  # type: ignore[arg-type]
            )
        )
    )

    artifacts = sorted(tmp_path.glob("metrics-*.json"))
    assert [json.loads(path.read_text())["eval_id"] for path in artifacts] == [
        "eval-first",
        "eval-second",
    ]


def test_hook_records_provenance_as_of_the_run_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks = MetricsExportHooks()
    monkeypatch.setattr(_metric_artifacts, "provenance", lambda root: {"git_dirty": False})
    asyncio.run(hooks.on_run_start(RunStart(eval_set_id=None, run_id="run-1", task_names=[])))
    monkeypatch.setattr(_metric_artifacts, "provenance", lambda root: {"git_dirty": True})
    log = make_log(results=sample_results(), location=str(tmp_path / "2026-run.eval"))

    asyncio.run(
        hooks.on_run_end(RunEnd(eval_set_id=None, run_id="run-1", exception=None, logs=[log]))  # type: ignore[arg-type]
    )

    [artifact] = tmp_path.glob("metrics-*.json")
    assert json.loads(artifact.read_text())["provenance"] == {"git_dirty": False}


def test_hook_warns_without_raising_when_an_export_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_log: EvalLog) -> str:
        raise OSError("storage unavailable")

    monkeypatch.setattr(_metric_artifacts, "write_metric_artifact", fail)
    log = make_log(results=sample_results())

    asyncio.run(
        MetricsExportHooks().on_run_end(
            RunEnd(
                eval_set_id=None,
                run_id="run-1",
                exception=None,
                logs=[log],  # type: ignore[arg-type]
            )
        )
    )

    assert "Unable to write metric artifact for eval eval-1" in caplog.text
