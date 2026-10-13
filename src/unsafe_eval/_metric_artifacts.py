"""Automatic lightweight metric artifacts for completed evaluations."""

import json
import logging
import math
import posixpath
import statistics
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from fsspec.core import url_to_fs
from inspect_ai.hooks import Hooks, RunEnd, RunStart, hooks
from inspect_ai.log import EvalLog, EvalSample, read_eval_log_samples

from unsafe_eval._configs import PACKAGED_CONFIG_ROOT
from unsafe_eval.cli._run import forwarded_inspect_args, provenance

logger = logging.getLogger(__name__)

# Taken when each run starts, by run id: the tree can change while a long run goes on.
_provenance_at_start: dict[str, dict[str, object]] = {}


def _metric_value(value: int | float) -> int | float | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _usage(log: EvalLog) -> dict[str, Any] | None:
    usage = log.stats.model_usage.get(log.eval.model)
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_cache_write": usage.input_tokens_cache_write,
        "input_tokens_cache_read": usage.input_tokens_cache_read,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_cost": usage.total_cost,
    }


def _samples(log: EvalLog) -> list[EvalSample]:
    if log.samples is not None:
        return log.samples
    # `inspect eval` hands hooks header-only logs, so the samples are read back from the file.
    try:
        return list(read_eval_log_samples(log.location, all_samples_required=False))
    except Exception as error:
        logger.debug("Unable to read samples of %s: %s", log.location, error)
        return []


def _latency(log: EvalLog) -> dict[str, Any] | None:
    starts: list[datetime] = []
    ends: list[datetime] = []
    per_sample: list[float] = []
    for sample in _samples(log):
        if sample.error is not None:
            continue
        calls = [
            (event.completed, event.output.time)
            for event in sample.events
            if event.event == "model"
            and event.model == log.eval.model
            and event.completed is not None
            and event.output.time is not None
        ]
        if not calls:
            continue
        per_sample.append(sum(seconds for _, seconds in calls))
        for completed, seconds in calls:
            ends.append(completed)
            # A call's own time leaves out what came before its request, such as starting a
            # local vLLM server inside the first call.
            starts.append(completed - timedelta(seconds=seconds))
    if not per_sample:
        return None
    inference_seconds = (max(ends) - min(starts)).total_seconds()
    ranked = sorted(per_sample)
    return {
        "seconds_per_sample": inference_seconds / len(per_sample),
        "inference_seconds": inference_seconds,
        "timed_samples": len(per_sample),
        "median_seconds": statistics.median(ranked),
        # Nearest rank, so the value is one a sample actually took.
        "p95_seconds": ranked[math.ceil(0.95 * len(ranked)) - 1],
    }


def metric_artifact(log: EvalLog) -> dict[str, Any]:
    """Create the strict, sample-free metric artifact for an Inspect log."""
    results = log.results
    revision = log.eval.model_args.get("revision")
    scores = [
        {
            "name": score.name,
            "scorer": score.scorer,
            "reducer": score.reducer,
            "scored_samples": score.scored_samples,
            "unscored_samples": score.unscored_samples,
            "metrics": {
                name: _metric_value(metric.value) for name, metric in score.metrics.items()
            },
        }
        for score in (results.scores if results is not None else [])
    ]
    return {
        "schema_version": 5,
        "run_id": log.eval.run_id,
        "eval_id": log.eval.eval_id,
        "status": log.status,
        "created_at": str(log.eval.created),
        "completed_at": str(log.stats.completed_at) if log.stats.completed_at else None,
        "task": {
            "name": log.eval.task,
            "registry_name": log.eval.task_registry_name,
            "version": log.eval.task_version,
        },
        "model": {
            "name": log.eval.model,
            "revision": revision if isinstance(revision, str) else None,
        },
        "metadata": dict(log.metadata),
        "samples": {
            "total": results.total_samples if results is not None else None,
            "completed": results.completed_samples if results is not None else None,
        },
        # Inspect's `--limit` and `--sample-id`, both `None` for a run over the whole dataset.
        "selection": {"limit": log.eval.config.limit, "sample_id": log.eval.config.sample_id},
        # `None` when the run left concurrency to Inspect's adaptive default.
        "max_connections": log.eval.model_generate_config.max_connections,
        "scores": scores,
        "usage": _usage(log),
        "latency": _latency(log),
        # The unsafe-eval source, whatever configuration file ran.
        "provenance": _provenance_at_start.get(log.eval.run_id) or provenance(PACKAGED_CONFIG_ROOT),
        "inspect_args": forwarded_inspect_args(),
    }


def metric_artifact_location(log_location: str) -> str:
    """Return a sidecar name that Inspect will not identify as a native log."""
    directory, filename = posixpath.split(log_location)
    stem, _ = posixpath.splitext(filename)
    return posixpath.join(directory, f"metrics-{stem}.json")


def write_metric_artifact(log: EvalLog) -> str:
    """Write one metric artifact beside its native Inspect log."""
    if not log.location:
        raise ValueError("Cannot write a metric artifact without an eval log location")

    location = metric_artifact_location(log.location)
    filesystem, path = url_to_fs(location)
    temporary_path = f"{path}.tmp-{uuid4().hex}"
    artifact = metric_artifact(log)
    content = (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    try:
        with filesystem.open(temporary_path, mode="wt", encoding="utf-8") as file:
            file.write(content)
        if filesystem.exists(path):
            filesystem.rm(path)
        filesystem.mv(temporary_path, path)
    except Exception:
        try:
            if filesystem.exists(temporary_path):
                filesystem.rm(temporary_path)
        except Exception:
            logger.debug("Unable to remove temporary metric artifact %s", temporary_path)
        raise

    return location


def _is_unsafe_eval(log: EvalLog) -> bool:
    benchmark_id = log.metadata.get("benchmark_id")
    return isinstance(benchmark_id, str) and bool(benchmark_id)


@hooks(
    name="metrics_export",
    description="Writes lightweight JSON metric artifacts for unsafe-eval tasks.",
)
class MetricsExportHooks(Hooks):
    """Export terminal unsafe-eval logs without affecting evaluation outcomes."""

    async def on_run_start(self, data: RunStart) -> None:
        _provenance_at_start[data.run_id] = provenance(PACKAGED_CONFIG_ROOT)

    async def on_run_end(self, data: RunEnd) -> None:
        for log in data.logs:
            if not _is_unsafe_eval(log):
                continue
            try:
                write_metric_artifact(log)
            except Exception as error:
                logger.warning(
                    "Unable to write metric artifact for eval %s: %s",
                    log.eval.eval_id,
                    error,
                )
        _provenance_at_start.pop(data.run_id, None)


__all__ = ["MetricsExportHooks"]
