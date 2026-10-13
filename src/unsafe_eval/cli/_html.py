"""Render a report as one self-contained HTML page.

The page embeds the selected runs' metric values as JSON and inlines its fonts and the
Principled Intelligence mark, so it opens offline and makes no network request. Like the
other formats it reads only `metrics-*.json` sidecars, so no prompt or model output reaches
it and the page is safe to share.
"""

import base64
import contextlib
import json
import os
import sys
import webbrowser
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from unsafe_eval.cli._report import (
    GROUPED_PREFIX,
    Selection,
    base_name,
    has_success,
    matches,
    metric_records,
    options_text,
)
from unsafe_eval.cli._sidecars import LATENCY_METRICS, USAGE_METRICS
from unsafe_eval.protocols import BenchmarkProtocolSpec

# Headline metrics with a direction. Only these are averaged, marked best, and colored as better
# or worse against a baseline, so a high `invalid_output_rate` never looks good.
HIGHER_IS_BETTER = frozenset(
    {"unsafe_f1", "balanced_accuracy", "unsafe_precision", "unsafe_recall", "safe_recall"}
)
LOWER_IS_BETTER = frozenset({*LATENCY_METRICS.values(), *USAGE_METRICS.values()})

# `json.dumps` already escapes every non-ASCII character, U+2028 and U+2029 among them. These
# three are what could otherwise close the <script> element that holds the data.
_SCRIPT_ESCAPES = str.maketrans({"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"})

# The fonts inlined into the page: family, CSS weight, and file under `report_html/fonts`.
# Latin subsets from Google Fonts, under the SIL Open Font License 1.1 (`fonts/OFL-*.txt`).
FONTS = (
    ("Archivo", "400 800", "archivo-latin.woff2"),
    ("JetBrains Mono", "400 500", "jetbrains-mono-latin.woff2"),
)


def _group_keys(spec: BenchmarkProtocolSpec | None) -> tuple[str, ...]:
    if spec is None:
        return ()
    keys = (spec.metric_group_by,) if spec.metric_group_by else ()
    return keys + spec.unsafe_group_by


def split_group(metric: str, keys: Sequence[str]) -> tuple[str | None, str]:
    """Split a grouped metric name into its group key and value.

    `grouped_harm_area_Scams and Fraud` with the key `harm_area` becomes
    `("harm_area", "Scams and Fraud")`. The longest matching key wins, and a name that
    matches none keeps its whole remainder as the value.
    """
    rest = base_name(metric).removeprefix(GROUPED_PREFIX)
    for key in sorted(keys, key=len, reverse=True):
        if rest.startswith(f"{key}_"):
            return key, rest[len(key) + 1 :]
    return None, rest


def build_payload(
    selection: Selection,
    specs: Mapping[str, BenchmarkProtocolSpec],
    *,
    metrics: Sequence[str] = (),
    source: str,
) -> dict[str, Any]:
    """Collect the page's rows, benchmarks, and every metric value.

    The page's script reads these keys:

    - `rows`: one per row label, with its model, revision, protocol, protocol options and
      their text as a label writes it, whether it is partial, and `runs`, which gives the
      dataset revision, run and eval IDs, and completion time of its run of each benchmark.
    - `benchmarks`: each benchmark's name, its primary metric (`None` when unregistered), and
      its headline metrics, the primary one first.
    - `records`: one per metric value, with the group key and value of a grouped metric.
    - `notes`: counts of what was left out, with each unsuccessful run's model, benchmark,
      status, start time, path, and whether a successful run of its setup is reported.
    - `source`, `generated_at`, `higher_is_better` and `lower_is_better` for the header and
      colors.

    `metrics` limits only the detail tables, as it does for Markdown. Every record stays in
    the payload and says whether it passed the filter, so the overview keeps its primary
    metrics.
    """
    records = metric_records(selection)

    rows: dict[str, dict[str, Any]] = {}
    for sidecar in selection.sidecars:
        label = selection.labels[sidecar.path]
        row = rows.setdefault(
            label,
            {
                "label": label,
                "model": sidecar.model,
                "revision": sidecar.revision,
                "protocol": sidecar.protocol,
                "protocol_options": sidecar.protocol_options,
                "options_text": options_text(sidecar.protocol_options),
                "partial": sidecar.partial,
                "runs": {},
            },
        )
        row["runs"][sidecar.benchmark] = {
            "dataset_revision": sidecar.dataset_revision,
            "run_id": sidecar.run_id,
            "eval_id": sidecar.eval_id,
            "completed_at": sidecar.completed_at,
        }

    benchmarks = sorted({sidecar.benchmark for sidecar in selection.sidecars})
    names: dict[str, list[str]] = {benchmark: [] for benchmark in benchmarks}
    for record in records:
        if record.metric_class != "headline":
            continue
        name = base_name(record.metric)
        if name not in names[record.benchmark]:
            names[record.benchmark].append(name)

    benchmark_payloads: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        spec = specs.get(benchmark)
        primary = spec.primary_metric if spec else None
        ordered = names[benchmark]
        if primary in ordered:
            ordered = [primary, *(name for name in ordered if name != primary)]
        benchmark_payloads.append(
            {"name": benchmark, "primary_metric": primary, "headline_metrics": ordered}
        )

    keys = {benchmark: _group_keys(specs.get(benchmark)) for benchmark in benchmarks}
    record_payloads: list[dict[str, object]] = []
    for record in records:
        group_key, group_value = (
            split_group(record.metric, keys[record.benchmark])
            if record.metric_class == "grouped"
            else (None, None)
        )
        record_payloads.append(
            {
                "label": record.label,
                "benchmark": record.benchmark,
                "score": record.score,
                "name": base_name(record.metric),
                "value": record.value,
                "metric_class": record.metric_class,
                "group_key": group_key,
                "group_value": group_value,
                "in_detail": matches(record.metric, metrics),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "notes": {
            "duplicates": selection.duplicates,
            "unsuccessful": [
                {
                    "model": sidecar.model,
                    "benchmark": sidecar.benchmark,
                    "status": sidecar.status,
                    "created_at": sidecar.created_at,
                    "path": str(sidecar.path),
                    "has_success": has_success(selection, sidecar),
                }
                for sidecar in selection.unsuccessful
            ],
            "partial": selection.partial,
            "unreadable": [str(path) for path in selection.unreadable],
        },
        "higher_is_better": sorted(HIGHER_IS_BETTER),
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "rows": list(rows.values()),
        "benchmarks": benchmark_payloads,
        "records": record_payloads,
    }


def embed_json(value: object) -> str:
    """Serialize `value` for a `<script type="application/json">` element.

    No string in the data can close the element or end the script early.
    """
    text = json.dumps(value, separators=(",", ":"), allow_nan=False)
    return text.translate(_SCRIPT_ESCAPES)


def html_report(
    selection: Selection,
    specs: Mapping[str, BenchmarkProtocolSpec],
    *,
    metrics: Sequence[str] = (),
    source: str,
) -> str:
    """Render the selected runs as one self-contained HTML page."""
    payload = build_payload(selection, specs, metrics=metrics, source=source)
    assets = resources.files("unsafe_eval.cli").joinpath("report_html")
    faces = "\n".join(
        f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
        "font-display:swap;src:url(data:font/woff2;base64,"
        + base64.b64encode(assets.joinpath("fonts", file).read_bytes()).decode("ascii")
        + ") format('woff2')}"
        for family, weight, file in FONTS
    )
    page = assets.joinpath("report.html").read_text(encoding="utf-8")
    page = page.replace("/*@FONTS@*/", faces)
    page = page.replace("<!--@MARK@-->", assets.joinpath("pi-mark.svg").read_text(encoding="utf-8"))
    # The data goes in last, so nothing in it can be taken for another placeholder.
    return page.replace("@DATA@", embed_json(payload))


def open_in_browser(path: Path) -> None:
    """Open a written page in a browser where one can show it, and ignore a failure to open.

    On Linux and other Unix systems without a display, over SSH for example, `webbrowser`
    would start a text browser such as lynx. It takes over the terminal and can't run the
    page's script, so nothing opens there unless `BROWSER` names a browser to use.
    """
    has_display = any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"))
    if sys.platform not in ("darwin", "win32") and not has_display:
        return
    with contextlib.suppress(webbrowser.Error):
        webbrowser.open(path.as_uri())
