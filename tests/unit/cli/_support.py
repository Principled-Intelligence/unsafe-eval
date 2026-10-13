import json
from pathlib import Path
from typing import Any

MIXED_LABEL_SCORES = [
    {
        "name": "label",
        "metrics": {
            "grouped_scenario_film": 0.9,
            "unsafe_eval/balanced_accuracy": 0.8,
            "unsafe_eval/unsafe_f1": 0.75,
            "unsafe_eval/unsafe_recall": 0.7,
        },
    },
    {"name": "verdict", "metrics": {"frequency_yes": 3, "frequency_no": 5}},
]


PROVENANCE = {
    "unsafe_eval_version": "1.0.0",
    "inspect_ai_version": "0.3.273",
    "git_revision": "4be1c0d",
    "git_dirty": False,
    "uv_lock_sha256": "3f2a",
}


def add_config(config_root: Path, configuration: str, benchmark: str) -> Path:
    config = config_root / configuration / f"{benchmark}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"# {configuration}/{benchmark}\n", encoding="utf-8")
    return config


def write_sidecar(
    directory: Path,
    name: str,
    *,
    task: str = "task",
    benchmark: str = "aegis-2.0",
    model: str = "vllm/example/model",
    revision: str | None = "commit",
    status: str = "success",
    completed_at: str | None = "2026-09-01T12:00:00+00:00",
    scores: list[dict[str, Any]] | None = None,
    run_id: str = "run",
    protocol: str = "shieldstral-1.0-3b",
    protocol_options: dict[str, Any] | None = None,
    limit: int | None = None,
    dataset_revision: str | None = None,
    latency: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    max_connections: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    path = directory / f"metrics-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 5 if provenance else 2,
                "run_id": run_id,
                "eval_id": f"eval-{name}",
                "status": status,
                "created_at": "2026-09-01T11:00:00+00:00",
                "completed_at": completed_at,
                "task": {"name": task, "registry_name": None, "version": 1},
                "model": {"name": model, "revision": revision},
                "metadata": {
                    "benchmark_id": benchmark,
                    "protocol": protocol,
                    "protocol_options": protocol_options or {"threshold": 0.5},
                    "dataset_revision": dataset_revision,
                },
                "samples": {"total": 10, "completed": 10},
                "selection": {"limit": limit, "sample_id": None},
                "scores": MIXED_LABEL_SCORES if scores is None else scores,
                "latency": latency,
                "usage": usage,
                "max_connections": max_connections,
                **({"provenance": provenance} if provenance else {}),
            }
        ),
        encoding="utf-8",
    )
    return path
