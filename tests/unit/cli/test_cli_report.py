import csv
import io
import json
import re
import sys
import webbrowser
from pathlib import Path

import pytest
from _support import PROVENANCE, write_sidecar

from unsafe_eval.cli import _html, _report
from unsafe_eval.cli._sidecars import environment, read_sidecar
from unsafe_eval.protocols import BenchmarkProtocolSpec

SPECS = {
    "aegis-2.0": BenchmarkProtocolSpec(mixed_labels=True, primary_metric="unsafe_f1"),
    "cosapien": BenchmarkProtocolSpec(
        mixed_labels=True, primary_metric="unsafe_f1", metric_group_by="scenario"
    ),
    "simple-safety-tests": BenchmarkProtocolSpec(
        mixed_labels=False,
        primary_metric="unsafe_recall",
        unsafe_group_by=("harm_area", "category"),
    ),
}

UNSAFE_ONLY_SCORES = [
    {
        "name": "label",
        "metrics": {
            "unsafe_eval/grouped_harm_area_Scams and Fraud": 1.0,
            "unsafe_eval/grouped_category_a/b": 0.5,
            "unsafe_eval/unsafe_recall": 0.95,
            "unsafe_eval/unsafe_recall_stderr": 0.02,
        },
    },
    {"name": "unsafe_probability", "metrics": {"unsafe_eval/mean": 0.6}},
]


def test_base_name_drops_only_a_package_namespace() -> None:
    assert _report.base_name("grouped_category_a/b") == "grouped_category_a/b"


def test_metrics_are_classified_by_score_and_group_prefix() -> None:
    assert _report.classify("unsafe_probability", "unsafe_eval/grouped_harm_area_x") == "other"


def test_the_latest_successful_sidecar_wins_per_model_revision_and_benchmark(
    tmp_path: Path,
) -> None:
    write_sidecar(tmp_path / "old", "a", completed_at="2026-09-01T12:00:00+00:00", run_id="old")
    write_sidecar(tmp_path / "new", "b", completed_at="2026-09-02T09:00:00+02:00", run_id="new")
    write_sidecar(tmp_path, "c", completed_at="2026-09-03T12:00:00+00:00", status="error")
    write_sidecar(tmp_path, "d", revision="other-commit")
    (tmp_path / "metrics-broken.json").write_text("{", encoding="utf-8")

    selection = _report.select_sidecars(tmp_path)

    assert [(item.revision, item.run_id) for item in selection.sidecars] == [
        ("commit", "new"),
        ("other-commit", "run"),
    ]
    assert selection.duplicates == 1
    [failed] = selection.unsuccessful
    assert failed.path == tmp_path / "metrics-c.json"
    assert _report.has_success(selection, failed)
    assert selection.unreadable == [tmp_path / "metrics-broken.json"]


@pytest.mark.parametrize("completed_at", [None, "not a time"])
def test_an_unusable_completion_time_still_orders(tmp_path: Path, completed_at: str | None) -> None:
    write_sidecar(tmp_path, "a", completed_at=completed_at, run_id="first")
    write_sidecar(tmp_path, "b", completed_at="2026-09-02T12:00:00+00:00", run_id="second")

    [sidecar] = _report.select_sidecars(tmp_path).sidecars

    assert sidecar.run_id == "second"


def test_long_format_records_every_metric_with_its_source(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "a")
    records = _report.metric_records(_report.select_sidecars(tmp_path))

    assert [(record.score, record.metric, record.metric_class) for record in records] == [
        ("label", "grouped_scenario_film", "grouped"),
        ("label", "unsafe_eval/balanced_accuracy", "headline"),
        ("label", "unsafe_eval/unsafe_f1", "headline"),
        ("label", "unsafe_eval/unsafe_recall", "headline"),
        ("verdict", "frequency_yes", "other"),
        ("verdict", "frequency_no", "other"),
    ]
    assert {record.eval_id for record in records} == {"eval-a"}
    assert {record.label for record in records} == {"vllm/example/model"}
    assert {record.protocol for record in records} == {"shieldstral-1.0-3b"}


def mixed_and_unsafe_only(tmp_path: Path) -> _report.Selection:
    write_sidecar(tmp_path, "a", model="model-b", revision=None)
    write_sidecar(tmp_path, "b", model="model-a")
    write_sidecar(tmp_path, "c", benchmark="simple-safety-tests", scores=UNSAFE_ONLY_SCORES)
    return _report.select_sidecars(tmp_path)


def test_markdown_overview_shows_each_benchmarks_primary_metric(tmp_path: Path) -> None:
    report = _report.markdown(_report.report_tables(mixed_and_unsafe_only(tmp_path), SPECS))

    overview = report.split("\n\n")[1].splitlines()
    assert overview[0] == (
        "| model | aegis-2.0 (unsafe_f1) | simple-safety-tests (unsafe_recall) |"
    )
    assert overview[2] == "| model-a | 0.750 |  |"
    assert overview[3] == "| model-b | 0.750 |  |"
    assert overview[4] == "| vllm/example/model |  | 0.950 |"


def test_markdown_all_includes_grouped_and_other_scores(tmp_path: Path) -> None:
    report = _report.markdown(
        _report.report_tables(mixed_and_unsafe_only(tmp_path), SPECS, include_all=True)
    )

    assert "### aegis-2.0: groups" in report
    assert "| model | verdict.frequency_yes | verdict.frequency_no |" in report
    assert "| model | unsafe_probability.mean |" in report


def test_markdown_metric_filter_limits_benchmark_tables(tmp_path: Path) -> None:
    report = _report.markdown(
        _report.report_tables(
            mixed_and_unsafe_only(tmp_path), SPECS, metrics=["balanced_accuracy"], groups=True
        )
    )

    assert "| model | balanced_accuracy |" in report
    # The overview still reports the primary metric, and a benchmark with no matching
    # metric gets no table.
    assert "aegis-2.0 (unsafe_f1)" in report
    assert "## simple-safety-tests" not in report
    assert "groups" not in report


def test_an_unregistered_benchmark_keeps_sidecar_order_and_an_empty_overview_cell(
    tmp_path: Path,
) -> None:
    write_sidecar(tmp_path, "a", benchmark="new-benchmark")

    report = _report.markdown(_report.report_tables(_report.select_sidecars(tmp_path), SPECS))

    assert "| model | new-benchmark |" in report
    assert "| vllm/example/model |  |" in report
    assert "| model | balanced_accuracy | unsafe_f1 | unsafe_recall |" in report


def test_csv_and_json_hold_the_same_long_format_rows(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "a", revision=None)
    records = _report.metric_records(_report.select_sidecars(tmp_path))

    rows = list(csv.DictReader(io.StringIO(_report.csv_report(records))))
    items = json.loads(_report.json_report(records))

    assert list(rows[0]) == list(_report.LONG_FORMAT_FIELDS)
    assert len(rows) == len(items) == len(records)
    assert items[0] == {
        "model": "vllm/example/model",
        "revision": None,
        "benchmark": "aegis-2.0",
        "protocol": "shieldstral-1.0-3b",
        "protocol_options": {"threshold": 0.5},
        "dataset_revision": None,
        "partial": False,
        "score": "label",
        "metric": "grouped_scenario_film",
        "value": 0.9,
        "run_id": "run",
        "eval_id": "eval-a",
    }
    assert rows[0]["value"] == "0.9"
    assert rows[0]["revision"] == ""
    assert rows[0]["protocol_options"] == '{"threshold":0.5}'
    assert rows[0]["partial"] == "False"


def test_runs_of_different_setups_get_their_own_rows(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "a", protocol_options={"threshold": 0.5}, run_id="default")
    write_sidecar(tmp_path, "b", protocol_options={"threshold": 0.3}, run_id="strict")
    write_sidecar(tmp_path, "c", protocol="other-protocol", run_id="other")
    write_sidecar(tmp_path, "d", model="model-z", run_id="alone")

    selection = _report.select_sidecars(tmp_path)

    assert selection.duplicates == 0
    assert sorted(selection.labels.values()) == [
        "model-z",
        "vllm/example/model [other-protocol]",
        'vllm/example/model [shieldstral-1.0-3b] {"threshold":0.3}',
        'vllm/example/model [shieldstral-1.0-3b] {"threshold":0.5}',
    ]
    report = _report.markdown(_report.report_tables(selection, SPECS))
    assert '| vllm/example/model [shieldstral-1.0-3b] {"threshold":0.3} |' in report


def test_runs_of_different_dataset_revisions_are_not_repeats(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "old", dataset_revision="data-1", run_id="old")
    write_sidecar(
        tmp_path,
        "new",
        dataset_revision="data-2",
        completed_at="2026-09-02T12:00:00+00:00",
        run_id="new",
    )
    write_sidecar(tmp_path, "sst", benchmark="simple-safety-tests", dataset_revision="data-3")

    selection = _report.select_sidecars(tmp_path)

    assert selection.duplicates == 0
    # Only the benchmark that ran under several dataset revisions names them.
    assert sorted(selection.labels.values()) == [
        "vllm/example/model",
        "vllm/example/model (dataset data-1)",
        "vllm/example/model (dataset data-2)",
    ]
    rows = json.loads(_report.json_report(_report.metric_records(selection)))
    assert {row["dataset_revision"] for row in rows} == {"data-1", "data-2", "data-3"}


def test_latency_is_a_headline_column_and_concurrency_splits_rows(tmp_path: Path) -> None:
    latency = {"seconds_per_sample": 0.01, "median_seconds": 0.05, "p95_seconds": 0.08}
    write_sidecar(tmp_path, "batched", latency=latency, run_id="batched")
    write_sidecar(tmp_path, "single", latency=latency, max_connections=1, run_id="single")

    selection = _report.select_sidecars(tmp_path)

    assert sorted(selection.labels.values()) == [
        "vllm/example/model (max_connections 1)",
        "vllm/example/model (max_connections default)",
    ]
    report = _report.markdown(_report.report_tables(selection, SPECS))
    assert (
        "| unsafe_f1 | balanced_accuracy | unsafe_recall"
        " | latency_median_seconds | latency_p95_seconds |"
    ) in report
    assert "| 0.050 | 0.080 |" in report


def test_tokens_per_completed_sample_are_headline_columns(tmp_path: Path) -> None:
    usage = {"input_tokens": 200, "output_tokens": 30, "reasoning_tokens": None}
    write_sidecar(tmp_path, "a", usage=usage)

    report = _report.markdown(_report.report_tables(_report.select_sidecars(tmp_path), SPECS))

    assert "| input_tokens_per_sample | output_tokens_per_sample |" in report
    assert "| 20.000 | 3.000 |" in report
    assert "reasoning_tokens_per_sample" not in report


def test_print_report_draws_tables_in_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_sidecar(tmp_path, "a", protocol="other-protocol")
    write_sidecar(tmp_path, "b")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("COLUMNS", "200")

    _report.print_report(_report.report_tables(_report.select_sidecars(tmp_path), SPECS))

    out = capsys.readouterr().out
    assert "╭" in out
    assert "| --- |" not in out
    assert "Overview" in out
    # A bracketed label is shown as written, not taken for rich markup.
    assert "vllm/example/model [other-protocol]" in out
    assert "0.750" in out


def test_a_partial_run_never_replaces_a_complete_one(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "full", completed_at="2026-09-01T12:00:00+00:00", run_id="full")
    write_sidecar(
        tmp_path, "smoke", completed_at="2026-09-02T12:00:00+00:00", run_id="smoke", limit=1
    )

    default = _report.select_sidecars(tmp_path)
    included = _report.select_sidecars(tmp_path, include_partial=True)

    assert [sidecar.run_id for sidecar in default.sidecars] == ["full"]
    assert default.partial == 1
    assert sorted(included.labels.values()) == [
        "vllm/example/model",
        "vllm/example/model (partial)",
    ]
    assert included.partial == 0


def test_only_partial_runs_suggests_including_them(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "smoke", limit=1)

    with pytest.raises(_report.ReportError, match="--include-partial"):
        _report.select_sidecars(tmp_path)


def test_html_payload_holds_the_long_format_rows_and_each_rows_setup(tmp_path: Path) -> None:
    write_sidecar(tmp_path, "a", benchmark="cosapien", dataset_revision="data-1")
    write_sidecar(
        tmp_path, "b", benchmark="simple-safety-tests", scores=UNSAFE_ONLY_SCORES, run_id="run-b"
    )
    selection = _report.select_sidecars(tmp_path)

    payload = _html.build_payload(selection, SPECS, metrics=["unsafe_f1"], source="runs")

    [row] = payload["rows"]
    assert row["label"] == "vllm/example/model"
    assert row["runs"] == {
        "cosapien": {
            "dataset_revision": "data-1",
            "run_id": "run",
            "eval_id": "eval-a",
            "completed_at": "2026-09-01T12:00:00+00:00",
        },
        "simple-safety-tests": {
            "dataset_revision": None,
            "run_id": "run-b",
            "eval_id": "eval-b",
            "completed_at": "2026-09-01T12:00:00+00:00",
        },
    }
    assert payload["benchmarks"] == [
        {
            "name": "cosapien",
            "primary_metric": "unsafe_f1",
            "headline_metrics": ["unsafe_f1", "balanced_accuracy", "unsafe_recall"],
        },
        {
            "name": "simple-safety-tests",
            "primary_metric": "unsafe_recall",
            "headline_metrics": ["unsafe_recall", "unsafe_recall_stderr"],
        },
    ]
    assert {
        (record["group_key"], record["group_value"])
        for record in payload["records"]
        if record["metric_class"] == "grouped"
    } == {("scenario", "film"), ("harm_area", "Scams and Fraud"), ("category", "a/b")}
    # --metric limits only the detail tables, so every record stays and is marked.
    assert {record["name"] for record in payload["records"] if record["in_detail"]} == {"unsafe_f1"}
    assert payload["notes"] == {"duplicates": 0, "unsuccessful": [], "partial": 0, "unreadable": []}


def test_html_notes_name_each_unsuccessful_run_and_whether_it_was_replaced(
    tmp_path: Path,
) -> None:
    write_sidecar(tmp_path, "aegis")
    write_sidecar(tmp_path, "aegis-failed", status="error")
    write_sidecar(tmp_path, "sst-failed", benchmark="simple-safety-tests", status="cancelled")

    payload = _html.build_payload(_report.select_sidecars(tmp_path), SPECS, source="runs")

    assert [
        (run["benchmark"], run["status"], run["has_success"], run["path"])
        for run in payload["notes"]["unsuccessful"]
    ] == [
        ("aegis-2.0", "error", True, str(tmp_path / "metrics-aegis-failed.json")),
        ("simple-safety-tests", "cancelled", False, str(tmp_path / "metrics-sst-failed.json")),
    ]


def test_the_html_page_is_self_contained_and_data_cannot_escape_its_script(tmp_path: Path) -> None:
    hostile = "</script><script>alert(1)</script> & \u2028 \u2029"
    write_sidecar(tmp_path, "a", model=hostile)

    page = _html.html_report(_report.select_sidecars(tmp_path), SPECS, source=str(tmp_path))

    assert page.startswith("<!doctype html>")
    assert page.count("</script>") == 2
    for placeholder in ("@DATA@", "/*@FONTS@*/", "<!--@MARK@-->"):
        assert placeholder not in page
    assert page.count("data:font/woff2;base64,") == len(_html.FONTS)
    assert not re.search(r"""(?:src|href)\s*=\s*["']?https?:|url\(\s*["']?https?:|@import""", page)
    data = page.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert not {"<", ">", "&", "\u2028", "\u2029"} & set(data)
    [row] = json.loads(data)["rows"]
    assert row["model"] == hostile


@pytest.mark.parametrize("display", [":0", None])
def test_a_page_opens_only_where_a_browser_can_show_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, display: str | None
) -> None:
    requested: list[str] = []

    def no_browser(url: str) -> bool:
        requested.append(url)
        raise webbrowser.Error("no display")

    monkeypatch.setattr(webbrowser, "open", no_browser)
    monkeypatch.setattr(sys, "platform", "linux")
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"):
        monkeypatch.delenv(name, raising=False)
    if display:
        monkeypatch.setenv("DISPLAY", display)

    path = tmp_path / "report.html"
    _html.open_in_browser(path)

    # Over SSH, `webbrowser` would start a text browser that takes over the terminal.
    assert requested == ([path.as_uri()] if display else [])


def test_a_sidecar_before_schema_5_has_an_unknown_environment(tmp_path: Path) -> None:
    old = read_sidecar(write_sidecar(tmp_path, "old"))
    new = read_sidecar(write_sidecar(tmp_path, "new", provenance=PROVENANCE))

    assert old is not None and new is not None
    assert old.provenance is None
    assert environment(old.provenance) == ("unknown",) * 4
    assert environment(new.provenance) == ("1.0.0", "0.3.273", "3f2a", False)
