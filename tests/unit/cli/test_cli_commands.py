import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from _support import PROVENANCE, add_config, write_sidecar
from click.testing import CliRunner

from unsafe_eval import _configs, cli
from unsafe_eval.cli import _run
from unsafe_eval.cli._banner import print_banner


@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    add_config(root, "vllm/model-a", "benchmark-a")
    add_config(root, "vllm/model-a", "benchmark-b")
    add_config(root, "hf/model-b", "benchmark-a")
    return root


@pytest.fixture
def sweeps(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record each sweep the command starts instead of running it."""
    calls: list[dict[str, object]] = []

    def fake_run_sweep(configs: Sequence[Path], **kwargs: object) -> int:
        calls.append({"configs": list(configs), **kwargs})
        return 0

    monkeypatch.setattr(_run, "run_sweep", fake_run_sweep)
    return calls


def invoke(*arguments: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(cli.cli, list(arguments))
    return result.exit_code, result.stdout, result.stderr


def test_run_selects_a_configuration_and_forwards_the_rest(
    config_root: Path, sweeps: list[dict[str, object]]
) -> None:
    code, _, _ = invoke(
        "run",
        "--log-dir",
        "runs",
        "--config-dir",
        str(config_root),
        "vllm/model-a",
        "--limit",
        "1",
        "-M",
        "k=v",
    )

    [sweep] = sweeps
    assert code == 0
    assert [path.stem for path in sweep["configs"]] == ["benchmark-a", "benchmark-b"]
    assert sweep["inspect_args"] == ["--limit", "1", "-M", "k=v"]
    assert sweep["log_dir"] == Path("runs")
    assert sweep["assume_yes"] is False


def test_run_accepts_its_own_options_after_the_selector(
    config_root: Path, sweeps: list[dict[str, object]]
) -> None:
    code, _, _ = invoke(
        "run",
        "vllm/model-a/benchmark-a",
        "--limit",
        "1",
        "--config-dir",
        str(config_root),
        "--dry-run",
        "--log-dir",
        "out",
        "--resume",
        "--yes",
    )

    [sweep] = sweeps
    assert code == 0
    assert [path.stem for path in sweep["configs"]] == ["benchmark-a"]
    assert sweep["inspect_args"] == ["--limit", "1"]
    assert sweep["dry_run"] is True
    assert sweep["assume_yes"] is True
    assert sweep["resume"] is True
    assert sweep["log_dir"] == Path("out")


def test_run_forwards_everything_after_a_double_dash(
    config_root: Path, sweeps: list[dict[str, object]]
) -> None:
    code, _, _ = invoke(
        "run",
        "--log-dir",
        "runs",
        "--config-dir",
        str(config_root),
        "--benchmark",
        "benchmark-a",
        "--",
        "--dry-run",
        "--limit",
        "2",
    )

    [sweep] = sweeps
    assert code == 0
    assert [path.parent.name for path in sweep["configs"]] == ["model-b", "model-a"]
    assert sweep["dry_run"] is False
    assert sweep["resume"] is False
    assert sweep["inspect_args"] == ["--dry-run", "--limit", "2"]


def test_run_sweeps_several_selectors_once_each_in_order(
    config_root: Path, sweeps: list[dict[str, object]]
) -> None:
    code, _, _ = invoke(
        "run",
        "--log-dir",
        "runs",
        "--config-dir",
        str(config_root),
        "hf/model-b",
        "vllm/model-a",
        "vllm/model-a/benchmark-a",
        "--limit",
        "1",
    )

    [sweep] = sweeps
    assert code == 0
    assert [(path.parent.name, path.stem) for path in sweep["configs"]] == [
        ("model-b", "benchmark-a"),
        ("model-a", "benchmark-a"),
        ("model-a", "benchmark-b"),
    ]
    assert sweep["inspect_args"] == ["--limit", "1"]


def test_run_narrows_selectors_to_one_benchmark(
    config_root: Path, sweeps: list[dict[str, object]]
) -> None:
    code, _, _ = invoke(
        "run",
        "--log-dir",
        "runs",
        "vllm/model-a",
        "hf/model-b",
        "--benchmark",
        "benchmark-a",
        "--config-dir",
        str(config_root),
    )

    [sweep] = sweeps
    assert code == 0
    assert [path.parent.name for path in sweep["configs"]] == ["model-a", "model-b"]
    assert {path.stem for path in sweep["configs"]} == {"benchmark-a"}


def test_run_accepts_a_config_file_from_disk(
    tmp_path: Path, sweeps: list[dict[str, object]]
) -> None:
    config = add_config(tmp_path / "elsewhere", "team/model", "task")

    code, _, _ = invoke("run", "--log-dir", "runs", str(config))

    assert code == 0
    assert sweeps[0]["configs"] == [config]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["run", "--limit", "1", "vllm/model-a"], "Pass a SELECTOR before"),
        (["run", "vllm/model-a", "--", "--log-dir", "x"], "--log-dir is set by"),
    ],
)
def test_run_usage_errors_exit_2(
    config_root: Path,
    sweeps: list[dict[str, object]],
    arguments: list[str],
    message: str,
) -> None:
    code, _, stderr = invoke(
        arguments[0], "--log-dir", "runs", *arguments[1:], "--config-dir", str(config_root)
    )

    assert code == 2
    assert message in stderr
    assert not sweeps


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["run", "vllm/missing"], "Available identifiers: hf/model-b, vllm/model-a"),
        (["run", "missing.yaml"], "Run configuration not found: missing.yaml"),
        (
            ["run", "vllm/model-a", "--benchmark", "missing"],
            "Unknown benchmark identifier 'missing'",
        ),
        (
            ["run", "vllm/model-a", "hf/model-b", "--benchmark", "benchmark-b"],
            "hf/model-b has no benchmark-b configuration.",
        ),
    ],
)
def test_run_selection_errors_exit_2(
    config_root: Path,
    sweeps: list[dict[str, object]],
    arguments: list[str],
    message: str,
) -> None:
    code, _, stderr = invoke(
        arguments[0], "--log-dir", "runs", *arguments[1:], "--config-dir", str(config_root)
    )

    assert code == 2
    assert message in stderr


def test_run_requires_a_log_dir(config_root: Path, sweeps: list[dict[str, object]]) -> None:
    code, _, stderr = invoke("run", "--config-dir", str(config_root), "vllm/model-a")

    assert code == 2
    assert "--log-dir" in stderr
    assert sweeps == []


def test_run_help_appends_the_live_inspect_help(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_help(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert list(command[1:]) == ["-m", "inspect_ai", "eval", "--help"]
        return subprocess.CompletedProcess(command, 0, stdout="INSPECT HELP\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_help)

    code, stdout, _ = invoke("run", "--help")

    assert code == 0
    assert "--config-dir" in stdout
    assert stdout.index("--config-dir") < stdout.index("Options forwarded to `inspect eval`")
    assert stdout.endswith("INSPECT HELP\n")


def test_run_help_still_exits_0_when_inspect_help_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_help(command: Sequence[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli.subprocess, "run", failing_help)

    code, stdout, _ = invoke("run", "--help")

    assert code == 0
    assert "Could not show the options of `inspect eval`" in stdout


def test_list_prints_every_section_by_default(config_root: Path) -> None:
    code, stdout, _ = invoke("list", "--config-dir", str(config_root))

    assert code == 0
    assert "benchmarks:\n  aegis-2.0\n" in stdout
    assert "\nprotocols:\n" in stdout
    assert "  shieldstral-1.0-3b\n" in stdout
    assert stdout.endswith(
        "configs:\n  hf/model-b/benchmark-a\n  vllm/model-a/benchmark-a\n"
        "  vllm/model-a/benchmark-b\n"
    )


def test_list_one_section_prints_bare_identifiers() -> None:
    code, stdout, _ = invoke("list", "configs")

    assert code == 0
    assert "vllm/shieldstral-1.0-3b/aegis-2.0\n" in stdout
    assert not stdout.startswith(" ")


def test_config_path_prints_the_packaged_file() -> None:
    code, stdout, _ = invoke("config", "path", "vllm/shieldstral-1.0-3b/aegis-2.0")

    path = Path(stdout.strip())
    assert code == 0
    assert path.is_absolute()
    assert path == _configs.PACKAGED_CONFIG_ROOT / "vllm/shieldstral-1.0-3b/aegis-2.0.yaml"


def test_config_path_for_an_unknown_identifier_exits_2() -> None:
    code, _, stderr = invoke("config", "path", "vllm/shieldstral-1.0-3b/missing")

    assert code == 2
    assert "vllm/shieldstral-1.0-3b/aegis-2.0" in stderr


def test_report_prints_markdown_and_notes_what_it_skipped(tmp_path: Path) -> None:
    write_sidecar(tmp_path / "a", "one")
    write_sidecar(tmp_path / "b", "two", completed_at="2026-09-02")
    write_sidecar(tmp_path, "three", status="error")
    (tmp_path / "metrics-four.json").write_text("[]", encoding="utf-8")

    code, stdout, stderr = invoke("report", str(tmp_path))

    assert code == 0
    assert "| vllm/example/model | 0.750 |" in stdout
    assert "| model | unsafe_f1 | balanced_accuracy |" in stdout
    assert "skipped 1 older duplicate" in stderr
    assert (
        "skipped the error run of vllm/example/model on aegis-2.0; "
        "a successful run of it is reported." in stderr
    )
    assert "could not read metric sidecar" in stderr


def test_report_long_formats_honour_the_metric_filter(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "one")

    code, stdout, _ = invoke("report", str(tmp_path), "--format", "json", "--metric", "unsafe_f1")

    assert code == 0
    assert "unsafe_eval/unsafe_f1" in stdout
    assert "balanced_accuracy" not in stdout


def test_report_rejects_a_metric_no_run_has(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "one")

    code, stdout, stderr = invoke(
        "report", str(tmp_path), "--metric", "unsafe_f1", "--metric", "unsafe_fl"
    )

    assert code == 2
    assert stdout == ""
    assert (
        "No reported run has metric unsafe_fl. Available: balanced_accuracy, frequency_no, "
        "frequency_yes, grouped_*, unsafe_f1, unsafe_recall." in stderr
    )


def test_report_leaves_out_partial_runs_unless_asked(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "full")
    write_sidecar(tmp_path, "smoke", completed_at="2026-09-02", limit=1)

    _, default, stderr = invoke("report", str(tmp_path))
    _, included, _ = invoke("report", str(tmp_path), "--include-partial")

    assert "(partial)" not in default
    assert "skipped 1 partial run(s)" in stderr
    assert "| vllm/example/model (partial) | 0.750 |" in included


def test_report_notes_runs_from_several_setups_on_stderr_only(tmp_path: Path) -> None:
    write_sidecar(tmp_path / "a", "old")
    write_sidecar(tmp_path / "b", "new", benchmark="cosapien", provenance=PROVENANCE)

    code, stdout, stderr = invoke("report", str(tmp_path), "--format", "json")

    assert code == 0
    assert json.loads(stdout)
    assert "note: runs from 2 setups" in stderr


def test_report_without_sidecars_exits_1(tmp_path: Path) -> None:
    code, _, stderr = invoke("report", str(tmp_path))

    assert code == 1
    assert "No metrics-*.json sidecars found" in stderr


def test_report_html_goes_to_redirected_output(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "one")

    code, stdout, _ = invoke("report", str(tmp_path), "--format", "html")

    assert code == 0
    assert stdout.startswith("<!doctype html>")
    assert '"model":"vllm/example/model"' in stdout


def test_the_banner_shows_only_in_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    print_banner()
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    print_banner()
    assert "A framework for reproducible AI safety evaluation." in capsys.readouterr().out
