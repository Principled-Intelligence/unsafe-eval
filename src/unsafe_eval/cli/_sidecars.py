"""Read the `metrics-*.json` sidecars that `MetricsExportHooks` writes beside each log."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EARLIEST = datetime.min.replace(tzinfo=UTC)

LATENCY_SCORE = "latency"
# Sidecar `latency` field to the metric name the report gives it.
LATENCY_METRICS = {"median_seconds": "latency_median_seconds", "p95_seconds": "latency_p95_seconds"}
USAGE_SCORE = "usage"
# Sidecar `usage` token totals to the per-sample metric name the report gives them.
USAGE_METRICS = {
    "input_tokens": "input_tokens_per_sample",
    "output_tokens": "output_tokens_per_sample",
    "reasoning_tokens": "reasoning_tokens_per_sample",
}
# The provenance fields that decide whether two runs are comparable. The git revision is left
# out: every commit changes it, documentation-only ones included.
ENVIRONMENT_FIELDS = ("unsafe_eval_version", "inspect_ai_version", "uv_lock_sha256", "git_dirty")


def provenance_value(provenance: Mapping[str, Any] | None, field: str) -> object:
    """One field of a sidecar's provenance, `unknown` for a sidecar from before schema 5."""
    return "unknown" if provenance is None else provenance.get(field)


def environment(provenance: Mapping[str, Any] | None) -> tuple[object, ...]:
    """The provenance fields that decide whether runs are comparable."""
    return tuple(provenance_value(provenance, field) for field in ENVIRONMENT_FIELDS)


@dataclass(frozen=True)
class Sidecar:
    """The parts of one metric sidecar that `report` and `run --resume` use."""

    path: Path
    status: str
    task: str
    model: str
    revision: str | None
    benchmark: str
    protocol: str | None
    protocol_options: dict[str, Any]
    dataset_revision: str | None
    partial: bool
    max_connections: int | None
    run_id: str
    eval_id: str
    created_at: str | None
    completed_at: str | None
    scores: list[dict[str, Any]]
    provenance: dict[str, Any] | None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def finished(self) -> datetime:
        """When the run completed, with a missing or unreadable time sorting first."""
        if self.completed_at is None:
            return _EARLIEST
        try:
            completed = datetime.fromisoformat(self.completed_at)
        except ValueError:
            return _EARLIEST
        return completed if completed.tzinfo else completed.replace(tzinfo=UTC)


def read_sidecar(path: Path) -> Sidecar | None:
    """Read one sidecar, or return `None` when it cannot be read or lacks a required field.

    A run counts as partial when it was limited to some of its samples with Inspect's
    `--limit` or `--sample-id`. Sidecars from before schema version 2 do not record that,
    so they count as complete.

    The run's per-sample latency and mean tokens per completed sample, when recorded, join
    `scores` as `latency` and `usage` scores, so they are reported like any metric.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        metadata = data["metadata"]
        selection = data.get("selection") or {}
        task = data.get("task") or {}
        latency = data.get("latency") or {}
        provenance = data.get("provenance")
        timing = {name: latency[key] for key, name in LATENCY_METRICS.items() if key in latency}
        usage = data.get("usage") or {}
        completed = (data.get("samples") or {}).get("completed")
        tokens = {
            name: usage[key] / completed
            for key, name in USAGE_METRICS.items()
            if usage.get(key) is not None and completed
        }
        scores = list(data.get("scores") or [])
        if timing:
            scores.append({"name": LATENCY_SCORE, "metrics": timing})
        if tokens:
            scores.append({"name": USAGE_SCORE, "metrics": tokens})
        return Sidecar(
            path=path,
            status=str(data["status"]),
            task=str(task.get("registry_name") or task.get("name") or ""),
            model=str(model["name"]),
            revision=model.get("revision"),
            benchmark=str(metadata["benchmark_id"]),
            protocol=metadata.get("protocol"),
            protocol_options=dict(metadata.get("protocol_options") or {}),
            dataset_revision=metadata.get("dataset_revision"),
            partial=(selection.get("limit") is not None or selection.get("sample_id") is not None),
            max_connections=data.get("max_connections"),
            run_id=str(data["run_id"]),
            eval_id=str(data["eval_id"]),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
            scores=scores,
            provenance=provenance if isinstance(provenance, dict) else None,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def find_sidecars(directory: Path) -> list[Path]:
    """Return every sidecar path below a directory, in sorted order."""
    return sorted(directory.rglob("metrics-*.json"))
