import asyncio
import io
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _support import PROVENANCE, add_config, write_sidecar
from inspect_ai.hooks import SampleEnd, TaskStart

from unsafe_eval._configs import PACKAGED_CONFIG_ROOT
from unsafe_eval._progress import RunProgressHooks
from unsafe_eval.cli import _run


def test_build_command_keeps_the_config_and_appends_forwarded_arguments(tmp_path: Path) -> None:
    config = tmp_path / "vllm/model-a" / "benchmark-a.yaml"
    run_dir = tmp_path / "logs/vllm/model-a/benchmark-a/20260924T080000Z"

    command = _run.build_command(config, run_dir, ["--limit", "3", "-M", "key=value"])

    assert command == [
        sys.executable,
        "-m",
        "inspect_ai",
        "eval",
        "--json",
        "--run-config",
        str(config),
        "--log-dir",
        str(run_dir),
        "--limit",
        "3",
        "-M",
        "key=value",
    ]


def test_a_directory_taken_in_the_same_second_gets_a_suffix(tmp_path: Path) -> None:
    taken = tmp_path / "20260924T080000Z"
    taken.mkdir()

    assert _run.unused_path(tmp_path / "free") == tmp_path / "free"
    assert re.fullmatch(r"20260924T080000Z-[0-9a-f]{8}", _run.unused_path(taken).name)


def test_reserved_option_finds_flags_the_built_command_sets() -> None:
    assert _run.reserved_option(["--limit", "1", "--log-dir=elsewhere"]) == "--log-dir"


def fake_inspect(stdout: str, returncode: int = 0) -> object:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)

    return run


def done_event(*statuses: str) -> str:
    return "\n".join(
        [
            json.dumps({"event": "launch", "run_id": "run"}),
            "not json",
            json.dumps(
                {
                    "event": "done",
                    "logs": [
                        {"task": "task-a", "status": status, "location": "logs/result.eval"}
                        for status in statuses
                    ],
                }
            ),
        ]
    )


@pytest.mark.parametrize(
    ("status", "expected_success"),
    [("success", True), ("error", False)],
)
def test_evaluate_config_requires_successful_log_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_success: bool,
) -> None:
    monkeypatch.setattr(_run.subprocess, "run", fake_inspect(done_event(status)))

    result = _run.evaluate_config(tmp_path / "config.yaml", tmp_path / "logs", [])

    assert result.success is expected_success
    assert result.logs[0]["status"] == status


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        (done_event("success"), 1),
        (json.dumps({"event": "launch"}), 0),
    ],
)
def test_evaluate_config_fails_without_a_successful_done_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int
) -> None:
    monkeypatch.setattr(_run.subprocess, "run", fake_inspect(stdout, returncode))

    result = _run.evaluate_config(tmp_path / "config.yaml", tmp_path / "logs", [])

    assert result.success is False


def test_evaluate_config_hands_the_forwarded_arguments_to_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env: dict[str, str] = {}
    stderr: list[str] = []

    def run(*args: object, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        env.update(kwargs["env"])
        stderr.append(kwargs["stderr"].name)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=done_event("success"))

    monkeypatch.setattr(_run.subprocess, "run", run)

    _run.evaluate_config(tmp_path / "config.yaml", tmp_path / "logs", ["--limit", "1"])

    assert json.loads(env[_run.INSPECT_ARGS_ENV]) == ["--limit", "1"]
    assert stderr == [str(tmp_path / "logs" / _run.INSPECT_LOG)]


def test_the_progress_hook_reports_each_limited_sample_and_epoch_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks = RunProgressHooks()
    assert not hooks.enabled()
    spec = SimpleNamespace(
        dataset=SimpleNamespace(samples=100, sample_ids=["a", "b"]),
        config=SimpleNamespace(epochs=2),
    )

    def run(*args: object, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        monkeypatch.setenv(_run.PROGRESS_FD_ENV, kwargs["env"][_run.PROGRESS_FD_ENV])
        assert hooks.enabled()
        ids = {"eval_set_id": None, "run_id": "run", "eval_id": "eval"}
        start = TaskStart(**ids, spec=spec, plan=None)  # type: ignore[arg-type]
        asyncio.run(hooks.on_task_start(start))
        asyncio.run(hooks.on_sample_end(SampleEnd(**ids, sample_id="a", sample=None)))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=done_event("success"))

    monkeypatch.setattr(_run.subprocess, "run", run)
    updates: list[tuple[int | None, int]] = []

    _run.evaluate_config(
        tmp_path / "config.yaml", tmp_path / "logs", [], lambda *update: updates.append(update)
    )

    assert updates == [(4, 0), (4, 1)]


def test_provenance_outside_a_checkout_records_only_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(_run.subprocess, "run", no_git)

    provenance = _run.provenance(tmp_path)

    assert provenance["unsafe_eval_version"] is not None
    assert provenance["inspect_ai_version"] is not None
    assert provenance["git_revision"] is None
    assert provenance["git_dirty"] is None
    assert provenance["uv_lock_sha256"] is None


def test_provenance_in_a_checkout_records_revision_and_lock_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_text("lock\n", encoding="utf-8")
    config_root = tmp_path / "src" / "configs"
    config_root.mkdir(parents=True)
    outputs = {
        "ls-files": "vllm/model/benchmark.yaml\n",
        "--show-toplevel": f"{tmp_path}\n",
        "HEAD": "abc123\n",
        "--porcelain": " M file.py\n",
    }

    def fake_git(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["cwd"] == config_root
        stdout = next(value for flag, value in outputs.items() if flag in command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout)

    monkeypatch.setattr(_run.subprocess, "run", fake_git)

    provenance = _run.provenance(config_root)

    assert provenance["git_revision"] == "abc123"
    assert provenance["git_dirty"] is True
    assert provenance["uv_lock_sha256"] == _run.file_sha256(tmp_path / "uv.lock")


def test_provenance_ignores_a_repository_that_does_not_track_the_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wheel installed into a project's `.venv` sits inside that project's repository.
    def fake_git(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "ls-files" in command
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="")

    monkeypatch.setattr(_run.subprocess, "run", fake_git)

    provenance = _run.provenance(tmp_path)

    assert provenance["git_revision"] is None
    assert provenance["uv_lock_sha256"] is None


def test_provenance_records_no_path() -> None:
    recorded = _run.provenance(PACKAGED_CONFIG_ROOT)

    assert not [value for value in recorded.values() if isinstance(value, str) and "/" in value]


def test_a_missing_distribution_has_no_version() -> None:
    assert _run._installed_version("unsafe-eval-not-installed") is None


CURRENT = {
    "unsafe_eval_version": "1.0.1",
    "inspect_ai_version": "0.3.273",
    "git_revision": "81aa3f2c",
    "git_dirty": False,
    "uv_lock_sha256": "9c1e07b2",
}


class Terminal(io.StringIO):
    """Standard input that is a terminal; `run` reads its lines as answers."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    """Run a sweep of configurations under `tmp_path/configs` into `tmp_path/logs`.

    It answers yes to the setup questions unless a test passes `assume_yes=False`.
    """
    monkeypatch.setattr(_run, "provenance", lambda root: CURRENT)
    defaults: dict[str, Any] = {
        "log_dir": tmp_path / "logs",
        "inspect_args": [],
        "dry_run": False,
        "resume": False,
        "assume_yes": True,
    }

    def run(configs: list[Path], **options: Any) -> int:
        return _run.run_sweep(configs, **{**defaults, **options})

    return run


def succeed(
    config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
) -> _run.EvaluationResult:
    return _run.EvaluationResult(config=str(config), returncode=0, success=True, logs=[])


def test_every_configuration_gets_the_same_layout_and_a_failure_does_not_stop_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sweep: Callable[..., int]
) -> None:
    config_root = tmp_path / "configs"
    configs = [
        add_config(config_root, "vllm/model-a", "benchmark-a"),
        add_config(config_root, "vllm/model-a", "benchmark-b"),
    ]
    seen: list[tuple[Path, list[str]]] = []

    def fake_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        seen.append((log_dir, list(inspect_args)))
        success = config.stem == "benchmark-b"
        return _run.EvaluationResult(
            config=str(config),
            returncode=0,
            success=success,
            logs=[{"status": "success" if success else "error"}],
        )

    monkeypatch.setattr(_run, "evaluate_config", fake_evaluate)

    exit_code = sweep(configs, inspect_args=["--limit", "2"])

    logs = tmp_path / "logs"
    assert exit_code == 1
    assert [(run_dir.parent, args) for run_dir, args in seen] == [
        (logs / "vllm/model-a/benchmark-a", ["--limit", "2"]),
        (logs / "vllm/model-a/benchmark-b", ["--limit", "2"]),
    ]
    assert seen[0][0].name == seen[1][0].name


def test_a_configuration_that_fails_to_launch_stops_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config_root = tmp_path / "configs"
    configs = [
        add_config(config_root, "vllm/model-a", "benchmark-a"),
        add_config(config_root, "vllm/model-a", "benchmark-b"),
    ]
    seen: list[str] = []

    def fail_to_launch(
        config: Path, log_dir: Path, inspect_args: Sequence[str], on_progress: _run.OnProgress
    ) -> _run.EvaluationResult:
        seen.append(config.stem)
        on_progress(None, 0)
        log_dir.mkdir(parents=True)
        (log_dir / _run.INSPECT_LOG).write_text("banner\nMissing API key\n")
        return _run.EvaluationResult(config=str(config), returncode=1, success=False, logs=[])

    monkeypatch.setattr(_run, "evaluate_config", fail_to_launch)

    exit_code = sweep(configs)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert seen == ["benchmark-a"]
    assert "[1/2] vllm/model-a/benchmark-a failed" in captured.out
    assert "Missing API key" in captured.err
    assert "--resume" in captured.err
    assert "0/2 configurations succeeded" in captured.out


def test_an_interrupted_run_exits_130_without_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sweep: Callable[..., int]
) -> None:
    config = add_config(tmp_path / "configs", "vllm/model-a", "benchmark-a")

    def interrupt(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(_run, "evaluate_config", interrupt)

    exit_code = sweep([config])

    assert exit_code == 130
    assert not list(tmp_path.rglob("report.html"))


def test_dry_run_prints_commands_without_running_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config = add_config(tmp_path / "configs", "vllm/model-a", "benchmark-a")

    def unexpected_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        raise AssertionError((config, log_dir, inspect_args))

    monkeypatch.setattr(_run, "evaluate_config", unexpected_evaluate)

    exit_code = sweep([config], inspect_args=["--limit", "1"], dry_run=True)

    assert exit_code == 0
    assert not (tmp_path / "logs").exists()
    assert "--limit 1" in capsys.readouterr().out


RUN_CONFIG = """\
task:
  task: unsafe_eval/aegis_prompt_safety
  args:
    protocol: shieldstral-1.0-3b
    dataset_revision: dataset-commit
    protocol_options:
      threshold: 0.5
model:
  model: vllm/org/model
  args:
    revision: model-commit
"""


def write_run_config(config_root: Path, text: str = RUN_CONFIG) -> Path:
    config = config_root / "vllm" / "model" / "aegis-2.0.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(text, encoding="utf-8")
    return config


def write_run(log_dir: Path, name: str, **fields: Any) -> Path:
    """Write the sidecar of a successful, complete run of `RUN_CONFIG`, with `fields` changed."""
    run = {
        "task": "aegis_prompt_safety",
        "model": "vllm/org/model",
        "revision": "model-commit",
        "protocol_options": {"threshold": 0.5, "default": True},
        "dataset_revision": "dataset-commit",
    }
    return write_sidecar(log_dir / "vllm" / "model", name, **{**run, **fields})


def completed(config: Path, log_dir: Path, inspect_args: Sequence[str] = ()) -> Path | None:
    run = _run.completed_run(config, log_dir, inspect_args)
    return run.path if run else None


def test_completed_run_finds_the_latest_complete_success(tmp_path: Path) -> None:
    config = write_run_config(tmp_path / "configs")
    logs = tmp_path / "runs"
    write_run(logs, "older", completed_at="2026-09-01T12:00:00+00:00")
    # Runs are found wherever they are, including inside an earlier sweep.
    latest = write_run(
        logs / "sweeps/20260902T120000Z-model-vllm-model",
        "latest",
        completed_at="2026-09-02T12:00:00+00:00",
    )
    write_run(logs, "failed", status="error", completed_at="2026-09-03T12:00:00+00:00")
    write_run(logs, "smoke", limit=1, completed_at="2026-09-04T12:00:00+00:00")

    assert completed(config, logs) == latest


@pytest.mark.parametrize(
    "run",
    [
        {"revision": "older-commit"},
        {"protocol_options": {"threshold": 0.3}},
        {"max_connections": 256},
    ],
)
def test_completed_run_ignores_runs_that_do_not_count(
    tmp_path: Path, run: dict[str, object]
) -> None:
    config = write_run_config(tmp_path / "configs")
    logs = tmp_path / "runs"
    write_run(logs, "a", **run)

    assert _run.completed_run(config, logs) is None


def test_a_forwarded_max_connections_overrides_the_configs(tmp_path: Path) -> None:
    config = write_run_config(
        tmp_path / "configs", RUN_CONFIG + "generate_config:\n  max_connections: 4\n"
    )
    logs = tmp_path / "runs"
    configured = write_run(logs, "configured", max_connections=4)
    forwarded = write_run(logs, "forwarded", max_connections=256)

    assert completed(config, logs) == configured
    assert completed(config, logs, ["--limit", "5", "--max-connections", "256"]) == forwarded
    assert completed(config, logs, ["--max-connections=256"]) == forwarded


def test_a_config_that_cannot_be_read_has_no_completed_run(tmp_path: Path) -> None:
    config = write_run_config(tmp_path / "configs", "task: [unclosed\n")
    write_run(tmp_path / "runs", "a")

    assert _run.completed_run(config, tmp_path / "runs") is None


def test_resume_skips_completed_configurations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config_root = tmp_path / "configs"
    done = write_run_config(config_root)
    pending = add_config(config_root, "vllm/model", "cosapien")
    write_run(tmp_path / "logs", "a")
    ran: list[str] = []

    def fake_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        ran.append(config.stem)
        return succeed(config, log_dir, inspect_args)

    monkeypatch.setattr(_run, "evaluate_config", fake_evaluate)

    exit_code = sweep([done, pending], resume=True)

    assert exit_code == 0
    assert ran == ["cosapien"]
    assert "skipped: already completed" in capsys.readouterr().out


def test_a_run_ends_with_a_summary_of_the_whole_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config_root = tmp_path / "configs"
    done = write_run_config(config_root)
    pending = add_config(config_root, "vllm/model", "cosapien")
    failing = add_config(config_root, "vllm/model", "dynaguardrail")
    logs = tmp_path / "logs"
    write_run(logs, "previous")
    write_run(logs, "unrelated", revision="other-commit")

    def fake_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        if config == failing:
            location = str(log_dir / "vllm/model/failed.eval")
            return _run.EvaluationResult(
                config=str(config), returncode=1, success=False, logs=[{"location": location}]
            )
        write_run(log_dir, "smoke", limit=1)
        return _run.EvaluationResult(
            config=str(config),
            returncode=0,
            success=True,
            logs=[{"location": str(log_dir / "vllm/model/smoke.eval")}, {"status": "error"}],
        )

    monkeypatch.setattr(_run, "evaluate_config", fake_evaluate)

    sweep([done, pending, failing], resume=True)

    out = capsys.readouterr().out
    summary = out[out.index("## Overview") : out.index("Sweep complete")]
    assert "| vllm/org/model@model-c |" in summary
    assert "| vllm/org/model@model-c (partial) |" in summary
    assert "| vllm/org/model@other-c |" in summary
    page = logs / "report.html"
    assert f"Report: {page}" in out
    assert "vllm/org/model@model-c (partial)" in page.read_text(encoding="utf-8")


def test_a_sweep_with_no_successful_run_prints_no_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config = add_config(tmp_path / "configs", "vllm/model-a", "benchmark-a")

    def fail(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        return _run.EvaluationResult(config=str(config), returncode=1, success=False, logs=[])

    monkeypatch.setattr(_run, "evaluate_config", fail)

    exit_code = sweep([config])

    assert exit_code == 1
    assert "## Overview" not in capsys.readouterr().out


def test_resume_with_everything_completed_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config = write_run_config(tmp_path / "configs")
    write_run(tmp_path / "logs", "a")

    def unexpected_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        raise AssertionError(config)

    monkeypatch.setattr(_run, "evaluate_config", unexpected_evaluate)

    for dry_run in (True, False):
        exit_code = sweep([config], dry_run=dry_run, resume=True)
        assert exit_code == 0

    out = capsys.readouterr().out
    assert "Nothing to run" in out
    # Only the real run, not the dry run, summarizes the completed runs it skipped.
    assert out.count("## Overview") == 1
    assert "| vllm/org/model |" in out
    assert (tmp_path / "logs" / "report.html").is_file()


def test_a_run_writes_an_html_report_in_the_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config = add_config(tmp_path / "configs", "vllm/model", "aegis-2.0")

    def write_and_succeed(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        write_run(log_dir, "a")
        location = str(log_dir / "vllm/model/a.eval")
        return _run.EvaluationResult(
            config=str(config), returncode=0, success=True, logs=[{"location": location}]
        )

    monkeypatch.setattr(_run, "evaluate_config", write_and_succeed)

    sweep([config])

    logs = tmp_path / "logs"
    [page] = logs.rglob("report.html")
    assert page == logs / "report.html"
    assert f"Report: {page}" in capsys.readouterr().out
    assert '"model":"vllm/org/model"' in page.read_text(encoding="utf-8")


def test_a_report_page_that_cannot_be_written_only_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_run(tmp_path / "runs", "a")
    page = tmp_path / "report.html"
    page.mkdir()

    _run.print_summary(tmp_path / "runs", page)

    captured = capsys.readouterr()
    assert "## Overview" in captured.out
    assert "Report:" not in captured.out
    assert f"warning: could not write the HTML report {page}" in captured.err


@pytest.mark.parametrize(
    ("folder", "stdin", "options", "expected_code", "ran"),
    [
        ([], io.StringIO(""), {}, 0, True),
        ([PROVENANCE, CURRENT], io.StringIO(""), {}, 0, True),
        ([None], Terminal("y\n"), {}, 0, True),
        ([PROVENANCE], Terminal("n\n"), {}, 1, False),
        ([PROVENANCE], Terminal(""), {}, 1, False),
        ([PROVENANCE], io.StringIO(""), {}, 2, False),
        ([PROVENANCE], io.StringIO(""), {"assume_yes": True}, 0, True),
        ([PROVENANCE], io.StringIO(""), {"dry_run": True}, 0, False),
    ],
    ids=["empty", "holds-current", "unknown-yes", "no", "eof", "no-terminal", "yes", "dry"],
)
def test_run_asks_before_adding_to_a_log_directory_of_another_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep: Callable[..., int],
    folder: list[dict[str, object] | None],
    stdin: io.StringIO,
    options: dict[str, bool],
    expected_code: int,
    ran: bool,
) -> None:
    config = add_config(tmp_path / "configs", "vllm/model", "cosapien")
    for index, provenance in enumerate(folder):
        write_run(tmp_path / "logs", f"earlier-{index}", provenance=provenance)
    launched: list[Path] = []

    def fake_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        launched.append(config)
        return succeed(config, log_dir, inspect_args)

    monkeypatch.setattr(_run, "evaluate_config", fake_evaluate)
    monkeypatch.setattr(sys, "stdin", stdin)

    exit_code = sweep([config], **{"assume_yes": False, **options})

    assert exit_code == expected_code
    assert bool(launched) is ran


def test_the_setup_question_names_each_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
) -> None:
    config = add_config(tmp_path / "configs", "vllm/model", "cosapien")
    write_run(tmp_path / "logs", "earlier", provenance=PROVENANCE)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    sweep([config], assume_yes=False)

    err = capsys.readouterr().err
    assert "This log directory has runs from a different setup:" in err
    assert "  unsafe_eval_version 1.0.0 -> 1.0.1\n" in err
    assert "  uv_lock_sha256 3f2a -> 9c1e07b2\n" in err
    assert "  git_revision 4be1c0d -> 81aa3f2c (not compared)\n" in err
    assert "inspect_ai_version" not in err
    assert "--yes" in err


@pytest.mark.parametrize(
    ("previous", "stdin", "expected_code", "reran"),
    [
        (CURRENT, io.StringIO(""), 0, False),
        (PROVENANCE, Terminal("y\n"), 0, False),
        (PROVENANCE, Terminal("n\n"), 0, True),
        (PROVENANCE, io.StringIO(""), 2, False),
    ],
    ids=["same", "yes", "no", "no-terminal"],
)
def test_resume_asks_before_reusing_a_run_of_another_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sweep: Callable[..., int],
    previous: dict[str, object],
    stdin: io.StringIO,
    expected_code: int,
    reran: bool,
) -> None:
    config = write_run_config(tmp_path / "configs")
    write_run(tmp_path / "logs", "a", provenance=previous)
    launched: list[Path] = []

    def fake_evaluate(
        config: Path, log_dir: Path, inspect_args: Sequence[str], *_: object
    ) -> _run.EvaluationResult:
        launched.append(config)
        return succeed(config, log_dir, inspect_args)

    monkeypatch.setattr(_run, "evaluate_config", fake_evaluate)
    monkeypatch.setattr(sys, "stdin", stdin)

    # Terminal holds one answer, so a folder question after a "no" would read EOF and exit 1.
    exit_code = sweep([config], resume=True, assume_yes=False)

    assert exit_code == expected_code
    assert bool(launched) is reran
    if isinstance(stdin, Terminal):
        assert "N: rerun them with the current setup" in capsys.readouterr().err
