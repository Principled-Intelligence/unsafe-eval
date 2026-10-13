"""Summarize the metric sidecars under a log directory.

The report reads only the `metrics-*.json` sidecars that `MetricsExportHooks` writes beside
each log, never a native log, so no prompt or model output is involved.
"""

import csv
import io
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from unsafe_eval.cli._sidecars import (
    LATENCY_SCORE,
    USAGE_SCORE,
    Sidecar,
    find_sidecars,
    read_sidecar,
)
from unsafe_eval.protocols import BenchmarkProtocolSpec

MetricClass = Literal["headline", "grouped", "other"]

# The score whose ungrouped metrics are the benchmark's headline numbers.
LABEL_SCORE = "label"
GROUPED_PREFIX = "grouped_"


class ReportError(ValueError):
    """Raised when a log directory holds nothing to report."""


@dataclass(frozen=True)
class MetricRecord:
    """One metric value from one sidecar, in long format."""

    label: str
    model: str
    revision: str | None
    benchmark: str
    protocol: str | None
    protocol_options: dict[str, Any]
    dataset_revision: str | None
    partial: bool
    score: str
    metric: str
    value: int | float | None
    metric_class: MetricClass
    run_id: str
    eval_id: str


@dataclass(frozen=True)
class Selection:
    """The sidecars a report covers, their row labels, and what it left out."""

    sidecars: list[Sidecar]
    labels: dict[Path, str]
    duplicates: int
    unsuccessful: list[Sidecar]
    partial: int
    unreadable: list[Path]


def base_name(metric: str) -> str:
    """Drop a registry namespace, so `unsafe_eval/unsafe_f1` becomes `unsafe_f1`.

    A group value may itself contain a slash, as in `grouped_category_a/b`, so only a
    leading package name counts as a namespace.
    """
    namespace, separator, name = metric.partition("/")
    if separator and not namespace.startswith(GROUPED_PREFIX):
        return name
    return metric


def classify(score: str, metric: str) -> MetricClass:
    """Put a metric into the headline, grouped, or other class."""
    if score in (LATENCY_SCORE, USAGE_SCORE):
        return "headline"
    if score != LABEL_SCORE:
        return "other"
    return "grouped" if base_name(metric).startswith(GROUPED_PREFIX) else "headline"


Setup = tuple[str, str | None, str | None, str, str | None, bool, int | None]


def setup(sidecar: Sidecar) -> Setup:
    """What a run measured apart from its benchmark: model, protocol, options, dataset
    revision, coverage, and concurrency.

    Two runs of one benchmark with the same setup are repeats, and the report keeps only the
    later one. Runs that differ in any of these get separate rows.
    """
    return (
        sidecar.model,
        sidecar.revision,
        sidecar.protocol,
        options_text(sidecar.protocol_options),
        sidecar.dataset_revision,
        sidecar.partial,
        sidecar.max_connections,
    )


def options_text(options: Mapping[str, Any]) -> str:
    """Return protocol options as compact JSON with sorted keys, so equal options compare equal."""
    return json.dumps(options, sort_keys=True, separators=(",", ":"))


def _model_label(sidecar: Sidecar) -> str:
    return f"{sidecar.model}@{sidecar.revision}" if sidecar.revision else sidecar.model


def row_labels(sidecars: Sequence[Sidecar]) -> dict[Path, str]:
    """Name each sidecar's row by its model, adding only as much setup as tells rows apart.

    A model gets its revision's first seven characters only when it ran at several revisions.
    The protocol is added when one model ran under several protocols, and the protocol
    options when it ran under several option sets of one protocol. The dataset revision is
    added when one benchmark ran under several of them in otherwise equal setups, and
    `max_connections` likewise, because concurrency changes latency. A partial run is always
    marked, so it cannot be mistaken for a complete one.
    """
    revisions: dict[str, set[str | None]] = defaultdict(set)
    protocols: dict[str, set[str | None]] = defaultdict(set)
    options: dict[tuple[str, str | None], set[str]] = defaultdict(set)
    datasets: dict[tuple[str, str | None, str, str], set[str | None]] = defaultdict(set)
    connections: dict[tuple[str, str | None, str, str, str | None], set[int | None]] = defaultdict(
        set
    )
    for sidecar in sidecars:
        model, protocol = _model_label(sidecar), sidecar.protocol
        option_text = options_text(sidecar.protocol_options)
        run = (model, protocol, option_text, sidecar.benchmark)
        revisions[sidecar.model].add(sidecar.revision)
        protocols[model].add(protocol)
        options[(model, protocol)].add(option_text)
        datasets[run].add(sidecar.dataset_revision)
        connections[(*run, sidecar.dataset_revision)].add(sidecar.max_connections)

    labels: dict[Path, str] = {}
    for sidecar in sidecars:
        model, protocol = _model_label(sidecar), sidecar.protocol
        option_text = options_text(sidecar.protocol_options)
        run = (model, protocol, option_text, sidecar.benchmark)
        label = sidecar.model
        if sidecar.revision and len(revisions[sidecar.model]) > 1:
            label += f"@{sidecar.revision[:7]}"
        if len(protocols[model]) > 1:
            label += f" [{protocol}]"
        if len(options[(model, protocol)]) > 1:
            label += f" {option_text}"
        if len(datasets[run]) > 1:
            label += f" (dataset {sidecar.dataset_revision})"
        if len(connections[(*run, sidecar.dataset_revision)]) > 1:
            label += f" (max_connections {sidecar.max_connections or 'default'})"
        if sidecar.partial:
            label += " (partial)"
        labels[sidecar.path] = label
    return labels


def select_sidecars(log_dir: Path, *, include_partial: bool = False) -> Selection:
    """Select from every sidecar under a log directory, as `select_paths` does."""
    paths = find_sidecars(log_dir)
    if not paths:
        raise ReportError(f"No metrics-*.json sidecars found under {log_dir}.")
    return select_paths(paths, include_partial=include_partial)


def select_paths(paths: Sequence[Path], *, include_partial: bool = False) -> Selection:
    """Keep the latest successful sidecar per benchmark and setup.

    Partial runs are left out unless `include_partial` is set, and then get rows of their
    own instead of replacing a complete run.
    """
    latest: dict[tuple[Setup, str], Sidecar] = {}
    kept = 0
    unsuccessful: list[Sidecar] = []
    partial = 0
    unreadable: list[Path] = []
    for path in paths:
        sidecar = read_sidecar(path)
        if sidecar is None:
            unreadable.append(path)
            continue
        if not sidecar.succeeded:
            unsuccessful.append(sidecar)
            continue
        if sidecar.partial and not include_partial:
            partial += 1
            continue
        kept += 1
        key = (setup(sidecar), sidecar.benchmark)
        current = latest.get(key)
        if current is None or sidecar.finished > current.finished:
            latest[key] = sidecar

    if not latest:
        raise ReportError(
            f"No successful complete evaluations among {len(paths)} sidecar(s)."
            + (" Pass --include-partial to report limited runs." if partial else "")
        )
    labels = row_labels(list(latest.values()))
    sidecars = sorted(latest.values(), key=lambda item: (labels[item.path], item.benchmark))
    return Selection(
        sidecars=sidecars,
        labels=labels,
        duplicates=kept - len(sidecars),
        unsuccessful=unsuccessful,
        partial=partial,
        unreadable=unreadable,
    )


def has_success(selection: Selection, sidecar: Sidecar) -> bool:
    """Whether the selection reports a successful run of this sidecar's setup and benchmark."""
    key = (setup(sidecar), sidecar.benchmark)
    return any((setup(kept), kept.benchmark) == key for kept in selection.sidecars)


def matches(metric: str, names: Sequence[str]) -> bool:
    """Whether a metric passes a `--metric` filter, with or without its namespace."""
    return not names or metric in names or base_name(metric) in names


def _metric_names(selection: Selection) -> set[str]:
    return {
        metric
        for sidecar in selection.sidecars
        for score in sidecar.scores
        for metric in score["metrics"]
    }


def unknown_metrics(selection: Selection, names: Sequence[str]) -> list[str]:
    """Return the requested names that match no metric of any selected sidecar, in order."""
    available = _metric_names(selection)
    return [name for name in names if not any(matches(m, [name]) for m in available)]


def available_metric_names(selection: Selection) -> list[str]:
    """Name the selected sidecars' metrics without namespaces, with groups as `grouped_*`."""
    names = {base_name(metric) for metric in _metric_names(selection)}
    return sorted(
        {GROUPED_PREFIX + "*" if name.startswith(GROUPED_PREFIX) else name for name in names}
    )


def metric_records(selection: Selection, names: Sequence[str] = ()) -> list[MetricRecord]:
    """Flatten the selected sidecars into one record per metric value, in sidecar order."""
    return [
        MetricRecord(
            label=selection.labels[sidecar.path],
            model=sidecar.model,
            revision=sidecar.revision,
            benchmark=sidecar.benchmark,
            protocol=sidecar.protocol,
            protocol_options=sidecar.protocol_options,
            dataset_revision=sidecar.dataset_revision,
            partial=sidecar.partial,
            score=str(score["name"]),
            metric=metric,
            value=value,
            metric_class=classify(str(score["name"]), metric),
            run_id=sidecar.run_id,
            eval_id=sidecar.eval_id,
        )
        for sidecar in selection.sidecars
        for score in sidecar.scores
        for metric, value in score["metrics"].items()
        if matches(metric, names)
    ]


def _format_value(value: int | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _escape(cell: str) -> str:
    return cell.replace("|", "\\|")


@dataclass(frozen=True)
class Table:
    """One titled table of the report: model rows against metric columns."""

    title: str
    level: int
    header: list[str]
    rows: list[list[str]]


def _markdown_table(table: Table) -> str:
    lines = [
        "#" * table.level + " " + table.title,
        "",
        "| " + " | ".join(_escape(cell) for cell in table.header) + " |",
        "| " + " | ".join("---" for _ in table.header) + " |",
    ]
    lines.extend("| " + " | ".join(_escape(cell) for cell in row) + " |" for row in table.rows)
    return "\n".join(lines)


def _pivot_table(
    title: str, level: int, records: Sequence[MetricRecord], first: str | None = None
) -> Table:
    """Tabulate models against metric names, in first-seen order with `first` leading."""
    models = sorted({record.label for record in records})
    columns: list[str] = []
    cells: dict[tuple[str, str], int | float | None] = {}
    for record in records:
        if record.metric not in columns:
            columns.append(record.metric)
        cells[(record.label, record.metric)] = record.value
    if first is not None and first in columns:
        columns.remove(first)
        columns.insert(0, first)
    rows = [
        [model, *(_format_value(cells.get((model, name))) for name in columns)] for model in models
    ]
    return Table(title, level, ["model", *columns], rows)


def markdown(tables: Sequence[Table]) -> str:
    """Join tables into one Markdown document."""
    return "\n\n".join(_markdown_table(table) for table in tables) + "\n"


def print_report(tables: Sequence[Table]) -> None:
    """Draw the tables in a terminal, or write them as Markdown when output is redirected."""
    if not sys.stdout.isatty():
        sys.stdout.write(markdown(tables))
        return

    from rich import box
    from rich.console import Console
    from rich.table import Table as RichTable
    from rich.text import Text

    # Cells go in as Text, so a label such as `model [protocol]` is not read as markup, and
    # every column wraps rather than cutting text off in a narrow terminal.
    console = Console()
    for table in tables:
        drawn = RichTable(
            title=Text(table.title, style="bold" if table.level == 2 else "italic"),
            title_justify="left",
            box=box.ROUNDED,
            header_style="bold",
        )
        drawn.add_column(Text(table.header[0]), overflow="fold")
        for name in table.header[1:]:
            drawn.add_column(Text(name), justify="right", overflow="fold")
        for row in table.rows:
            drawn.add_row(*(Text(cell) for cell in row))
        console.print(drawn)
        console.print()


def report_tables(
    selection: Selection,
    specs: Mapping[str, BenchmarkProtocolSpec],
    *,
    metrics: Sequence[str] = (),
    groups: bool = False,
    include_all: bool = False,
) -> list[Table]:
    """Build an overview of primary metrics, then one table per benchmark."""
    all_records = metric_records(selection)
    benchmarks = sorted({sidecar.benchmark for sidecar in selection.sidecars})
    models = sorted(set(selection.labels.values()))

    primary: dict[tuple[str, str], int | float | None] = {}
    for record in all_records:
        spec = specs.get(record.benchmark)
        if (
            spec is not None
            and record.metric_class == "headline"
            and base_name(record.metric) == spec.primary_metric
        ):
            primary[(record.label, record.benchmark)] = record.value

    header = ["model"]
    for benchmark in benchmarks:
        spec = specs.get(benchmark)
        header.append(f"{benchmark} ({spec.primary_metric})" if spec else benchmark)
    overview_rows = [
        [model, *(_format_value(primary.get((model, benchmark))) for benchmark in benchmarks)]
        for model in models
    ]
    tables = [Table("Overview", 2, header, overview_rows)]

    records = metric_records(selection, metrics)
    for benchmark in benchmarks:
        spec = specs.get(benchmark)
        benchmark_records = [record for record in records if record.benchmark == benchmark]
        headline = [record for record in benchmark_records if record.metric_class == "headline"]
        if headline:
            first = spec.primary_metric if spec else None
            headline = [replace(record, metric=base_name(record.metric)) for record in headline]
            tables.append(_pivot_table(benchmark, 2, headline, first))
        if groups or include_all:
            grouped = [
                replace(record, metric=base_name(record.metric).removeprefix(GROUPED_PREFIX))
                for record in benchmark_records
                if record.metric_class == "grouped"
            ]
            if grouped:
                tables.append(_pivot_table(f"{benchmark}: groups", 3, grouped))
        if include_all:
            other = [
                replace(record, metric=f"{record.score}.{base_name(record.metric)}")
                for record in benchmark_records
                if record.metric_class == "other"
            ]
            if other:
                tables.append(_pivot_table(f"{benchmark}: other scores", 3, other))
    return tables


LONG_FORMAT_FIELDS = (
    "model",
    "revision",
    "benchmark",
    "protocol",
    "protocol_options",
    "dataset_revision",
    "partial",
    "score",
    "metric",
    "value",
    "run_id",
    "eval_id",
)


def long_row(record: MetricRecord) -> dict[str, object]:
    """Pick a record's long-format fields, as `--format csv` and `json` write them."""
    return {field: getattr(record, field) for field in LONG_FORMAT_FIELDS}


def json_report(records: Sequence[MetricRecord]) -> str:
    """Render long-format records as a JSON array."""
    return json.dumps([long_row(record) for record in records], indent=2) + "\n"


def csv_report(records: Sequence[MetricRecord]) -> str:
    """Render long-format records as CSV with a header row."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LONG_FORMAT_FIELDS, lineterminator="\n")
    writer.writeheader()
    # CSV has no nested values, so the options mapping is written as compact JSON.
    writer.writerows(
        {**long_row(record), "protocol_options": options_text(record.protocol_options)}
        for record in records
    )
    return buffer.getvalue()
