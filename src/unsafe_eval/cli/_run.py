"""Run one or many run configurations through `inspect eval`."""

import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from unsafe_eval._configs import PACKAGED_CONFIG_ROOT, config_id, full_config_id
from unsafe_eval.cli._sidecars import (
    ENVIRONMENT_FIELDS,
    Sidecar,
    environment,
    find_sidecars,
    provenance_value,
    read_sidecar,
)

# Flags the built command sets itself. Forwarding one would override a value `run`
# relies on: `--json` for the result, `--run-config` and `--log-dir` for what runs where.
RESERVED_INSPECT_OPTIONS = ("--json", "--run-config", "--log-dir")

# Set by `run` on each `inspect eval` it launches, as a JSON list.
INSPECT_ARGS_ENV = "UNSAFE_EVAL_INSPECT_ARGS"

# Set by `run` on each `inspect eval` it launches: the descriptor its progress hook writes to.
PROGRESS_FD_ENV = "UNSAFE_EVAL_PROGRESS_FD"

INSPECT_LOG = "inspect.log"

# Called with (total, done): the run's sample count, `None` until its task starts, and how many
# have finished.
OnProgress = Callable[[int | None, int], None]


def forwarded_inspect_args() -> list[str] | None:
    """Return the options `run` forwarded to this `inspect eval`, or `None` for a plain one."""
    value = os.environ.get(INSPECT_ARGS_ENV)
    return None if value is None else json.loads(value)


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of one run configuration."""

    config: str
    returncode: int
    success: bool
    logs: list[dict[str, object]]


def reserved_option(arguments: Sequence[str]) -> str | None:
    """Return the first forwarded argument that sets a flag the built command owns."""
    for argument in arguments:
        name = argument.split("=", 1)[0]
        if name in RESERVED_INSPECT_OPTIONS:
            return name
    return None


def forwarded_max_connections(arguments: Sequence[str]) -> int | None:
    """Return the last `--max-connections` among forwarded arguments, as Inspect reads it."""
    value: str | None = None
    for index, argument in enumerate(arguments):
        name, separator, inline = argument.partition("=")
        if name == "--max-connections":
            value = inline if separator else next(iter(arguments[index + 1 :]), None)
    return int(value) if value is not None else None


def build_command(config: Path, run_dir: Path, inspect_args: Sequence[str]) -> list[str]:
    """Build an Inspect invocation that logs to the run's own directory."""
    return [
        sys.executable,
        "-m",
        "inspect_ai",
        "eval",
        "--json",
        "--run-config",
        str(config),
        "--log-dir",
        str(run_dir),
        *inspect_args,
    ]


def _follow_progress(read_fd: int, on_progress: OnProgress) -> None:
    total: int | None = None
    done = 0
    with os.fdopen(read_fd, encoding="utf-8") as pipe:
        for line in pipe:
            if line.startswith("total "):
                total = int(line.removeprefix("total "))
            else:
                done += 1
            on_progress(total, done)


def evaluate_config(
    config: Path,
    run_dir: Path,
    inspect_args: Sequence[str],
    on_progress: OnProgress = lambda total, done: None,
) -> EvaluationResult:
    """Run one configuration and keep only Inspect's aggregate launch result.

    Inspect's stderr goes to `inspect.log` in the run directory, and `on_progress` is called
    each time the progress hook reports.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    read_fd, write_fd = os.pipe()
    reader = threading.Thread(target=_follow_progress, args=(read_fd, on_progress), daemon=True)
    reader.start()
    try:
        with (run_dir / INSPECT_LOG).open("w", encoding="utf-8") as log:
            completed = subprocess.run(  # noqa: S603
                build_command(config, run_dir, inspect_args),
                check=False,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                pass_fds=(write_fd,),
                env={
                    **os.environ,
                    INSPECT_ARGS_ENV: json.dumps(list(inspect_args)),
                    PROGRESS_FD_ENV: str(write_fd),
                },
            )
    finally:
        os.close(write_fd)
    # A process Inspect started, such as a model server, can keep the pipe open after Inspect exits.
    reader.join(timeout=1)
    done: dict[str, Any] = {}
    for line in completed.stdout.splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            event = json.loads(line)
            if isinstance(event, dict) and event.get("event") == "done":
                done = event
    logs: list[dict[str, object]] = done.get("logs") or []
    success = (
        completed.returncode == 0
        and bool(logs)
        and all(log.get("status") == "success" for log in logs)
    )
    return EvaluationResult(
        config=str(config.resolve()),
        returncode=completed.returncode,
        success=success,
        logs=logs,
    )


def _task_name(task: str) -> str:
    return task.rpartition("/")[2]


def _is_run_of(sidecar: Sidecar, config: dict[str, Any], max_connections: int | None) -> bool:
    """Whether a sidecar records a run of the task, model, and pins a configuration names, at
    `max_connections`.

    The configuration's protocol options must all appear in the sidecar with the same value.
    The sidecar holds more, because it records the protocol's defaults too.
    """
    task = config["task"]
    args = task.get("args") or {}
    model = config["model"]
    options = args.get("protocol_options") or {}
    return (
        _task_name(sidecar.task) == _task_name(str(task["task"]))
        and sidecar.model == model["model"]
        and sidecar.revision == (model.get("args") or {}).get("revision")
        and sidecar.protocol == args.get("protocol")
        and sidecar.dataset_revision == args.get("dataset_revision")
        and all(sidecar.protocol_options.get(key) == value for key, value in options.items())
        and sidecar.max_connections == max_connections
    )


def completed_run(config: Path, log_dir: Path, inspect_args: Sequence[str] = ()) -> Sidecar | None:
    """Return the latest successful, complete run of a configuration anywhere under `log_dir`.

    Runs are matched by what their sidecars record, not by where they are, so single runs,
    earlier sweeps and older layouts all count. A run must also have used the concurrency this
    one will: a forwarded `--max-connections`, or else the configuration's. A configuration
    that cannot be read has no completed run, so it is run again.
    """
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        max_connections = forwarded_max_connections(inspect_args)
        if max_connections is None:
            max_connections = (data.get("generate_config") or {}).get("max_connections")
        runs = [
            sidecar
            for path in find_sidecars(log_dir)
            if (sidecar := read_sidecar(path)) is not None
            and sidecar.succeeded
            and not sidecar.partial
            and _is_run_of(sidecar, data, max_connections)
        ]
    except (OSError, ValueError, yaml.YAMLError, KeyError, TypeError, AttributeError):
        return None
    return max(runs, key=lambda sidecar: sidecar.finished) if runs else None


def setup_differences(
    previous: Sequence[Mapping[str, Any] | None], current: Mapping[str, Any]
) -> list[str]:
    """Describe each provenance field where earlier runs differ from `current`."""
    lines = []
    for field in (*ENVIRONMENT_FIELDS, "git_revision"):
        now = str(current.get(field))[:12]
        before = {str(provenance_value(run, field))[:12] for run in previous} - {now}
        if before:
            note = " (not compared)" if field == "git_revision" else ""
            lines.append(f"{field} {', '.join(sorted(before))} -> {now}{note}")
    return lines


def _ask(
    heading: str, lines: Sequence[str], prompt: str, *, assume_yes: bool, dry_run: bool
) -> bool | None:
    """Show what differs and ask, or return `None` when there is no terminal to ask on.

    A dry run and `assume_yes` answer yes without asking.
    """
    print(heading, *(f"  {line}" for line in lines), sep="\n", file=sys.stderr)
    if assume_yes or dry_run:
        return True
    if not sys.stdin.isatty():
        print("error: no terminal to ask on; pass --yes to go ahead.", file=sys.stderr)
        return None
    # On stderr with the differences, so it shows when stdout is redirected to a log.
    print(prompt, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *arguments],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def provenance(config_root: Path) -> dict[str, object]:
    """Record the installed versions and, from a git checkout, the source revision.

    The git fields describe the checkout that tracks the configurations, which is the case for
    an editable install. They are `None` for an installed wheel, including one installed into
    a virtual environment inside some other repository, where the files are not tracked.
    """
    versions: dict[str, object] = {
        "unsafe_eval_version": _installed_version("unsafe-eval"),
        "inspect_ai_version": _installed_version("inspect-ai"),
    }
    unknown = {**versions, "git_revision": None, "git_dirty": None, "uv_lock_sha256": None}
    try:
        if not _git(config_root, "ls-files", "."):
            return unknown
        top_level = Path(_git(config_root, "rev-parse", "--show-toplevel").strip())
        revision = _git(config_root, "rev-parse", "HEAD").strip()
        status = _git(config_root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return unknown

    lock_path = top_level / "uv.lock"
    return {
        **versions,
        "git_revision": revision,
        "git_dirty": bool(status),
        "uv_lock_sha256": file_sha256(lock_path) if lock_path.is_file() else None,
    }


def unused_path(path: Path) -> Path:
    """Return `path`, or a suffixed sibling when a run started in the same second took it."""
    return path.with_name(f"{path.name}-{uuid.uuid4().hex[:8]}") if path.exists() else path


def run_sweep(
    configs: Sequence[Path],
    *,
    log_dir: Path,
    inspect_args: Sequence[str],
    dry_run: bool,
    resume: bool,
    assume_yes: bool,
) -> int:
    """Run or preview each configuration in turn and return the exit code.

    With `resume`, a configuration that already has a successful, complete run under the log
    directory is skipped. Before reusing a run from another setup, or adding runs to a log
    directory that holds none of the current setup, it asks.
    """
    # Every run of one invocation shares its start time.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dirs = {
        config: unused_path(log_dir / config_id(config) / config.stem / timestamp)
        for config in configs
    }
    current = provenance(PACKAGED_CONFIG_ROOT)
    skipped = {
        config: previous
        for config in configs
        if resume and (previous := completed_run(config, log_dir, inspect_args))
    }
    stale = {
        config: run
        for config, run in skipped.items()
        if environment(run.provenance) != environment(current)
    }
    if stale:
        answer = _ask(
            f"--resume would reuse {len(stale)} run(s) from a different setup:",
            [
                f"{full_config_id(config)}: "
                + "; ".join(setup_differences([run.provenance], current))
                for config, run in stale.items()
            ],
            "y: reuse them as they are\nN: rerun them with the current setup\nReuse? [y/N] ",
            assume_yes=assume_yes,
            dry_run=dry_run,
        )
        if answer is None:
            return 2
        if not answer:
            skipped = {config: run for config, run in skipped.items() if config not in stale}
    for config in configs:
        if config in skipped:
            print(f"{full_config_id(config)} skipped: already completed")
        elif dry_run:
            print(full_config_id(config))
            print(f"  {shlex.join(build_command(config, run_dirs[config], inspect_args))}")
    configs = [config for config in configs if config not in skipped]
    # After the reuse question the user has seen the setups differ, so this one is not asked.
    if configs and not stale:
        folder = [run for path in find_sidecars(log_dir) if (run := read_sidecar(path)) is not None]
        if folder and environment(current) not in {environment(run.provenance) for run in folder}:
            answer = _ask(
                "This log directory has runs from a different setup:",
                setup_differences([run.provenance for run in folder], current),
                "Results may not be comparable. Continue? [y/N] ",
                assume_yes=assume_yes,
                dry_run=dry_run,
            )
            if answer is None:
                return 2
            if not answer:
                return 1
    if dry_run:
        return 0
    if not configs:
        print("Nothing to run: every selected configuration has already completed.")
        print_summary(log_dir, log_dir / "report.html")
        return 0

    results: list[EvaluationResult] = []
    try:
        for index, config in enumerate(configs, start=1):
            name = f"[{index}/{len(configs)}] {full_config_id(config)}"
            with _progress_bar(name) as on_progress:
                result = evaluate_config(config, run_dirs[config], inspect_args, on_progress)
            print(name, end=" ")
            results.append(result)
            print("succeeded" if result.success else "failed", flush=True)
            if not result.success:
                log = run_dirs[config] / INSPECT_LOG
                with contextlib.suppress(OSError):
                    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
                    print(*tail, sep="\n", file=sys.stderr)
                print(f"Inspect's output: {log}", file=sys.stderr)
            # No log means Inspect never started the evaluation: a missing extra, API key or
            # file, which every later configuration would most likely fail on too.
            if not result.success and not result.logs:
                remaining = len(configs) - index
                if remaining:
                    print(
                        f"Stopping the sweep: {full_config_id(config)} failed before "
                        f"Inspect started, so {remaining} configuration(s) did not run. Fix the "
                        "error above and rerun with --resume.",
                        file=sys.stderr,
                    )
                break
    except KeyboardInterrupt:
        print("\nInterrupted; partial results were retained.", file=sys.stderr)
        return 130

    print_summary(log_dir, log_dir / "report.html")
    succeeded = sum(result.success for result in results)
    print(f"Sweep complete: {succeeded}/{len(configs)} configurations succeeded.")
    return 0 if all(result.success for result in results) else 1


@contextlib.contextmanager
def _progress_bar(description: str) -> Iterator[OnProgress]:
    """Show finished samples as a bar until the block ends. Off a terminal, rich draws nothing."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)
        yield lambda total, done: progress.update(task, total=total, completed=done)


def print_summary(log_dir: Path, page: Path | None = None) -> None:
    """Print the report's overview of every run under the log directory, partial ones included.

    With `page`, the same runs are also written there as an HTML report. A page that can't be
    written only gets a warning, since the runs themselves are done.
    """
    from unsafe_eval.cli import _html, _report
    from unsafe_eval.protocols import registered_benchmarks

    try:
        selection = _report.select_sidecars(log_dir, include_partial=True)
    except _report.ReportError:
        return
    specs = registered_benchmarks()
    print()
    _report.print_report(_report.report_tables(selection, specs)[:1])
    if page is None:
        return
    try:
        # Nothing may have launched, so the log directory may not exist yet.
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(_html.html_report(selection, specs, source=str(page.parent)), "utf-8")
    except OSError as error:
        print(f"warning: could not write the HTML report {page}: {error}", file=sys.stderr)
    else:
        print(f"Report: {page}")
