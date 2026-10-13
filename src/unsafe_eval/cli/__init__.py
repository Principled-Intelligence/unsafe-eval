"""The `unsafe-eval` command: run, list, locate, and report evaluations.

`run` owns a few options and passes everything else to `inspect eval` unchanged, so the
command never re-declares an Inspect flag.
"""

import itertools
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import click

from unsafe_eval import _configs
from unsafe_eval.cli import _run
from unsafe_eval.cli._banner import print_banner

_EXIT_SELECTION = 2


def _config_root(config_dir: Path | None) -> Path:
    return config_dir if config_dir is not None else _configs.PACKAGED_CONFIG_ROOT


def _fail(message: str, code: int = _EXIT_SELECTION) -> NoReturn:
    click.echo(f"error: {message}", err=True)
    sys.exit(code)


def _resolve(selector: str, config_root: Path) -> list[Path]:
    """Resolve one selector to the files it names."""
    path = Path(selector)
    if path.suffix in {".yaml", ".yml"} or path.is_file():
        if not path.is_file():
            _fail(f"Run configuration not found: {selector}")
        return [path]
    return _configs.select_configs(selector, config_root)


def _select(selectors: Sequence[str], benchmark: str | None, config_root: Path) -> list[Path]:
    """Resolve the command's selection to the files it runs.

    With `benchmark`, each selector is narrowed to its configurations of that benchmark, and
    one that has none is an error rather than an empty part of the selection.
    """
    if not selectors:
        if benchmark is None:
            raise click.UsageError("Pass a SELECTOR before any forwarded option, or --benchmark.")
        return _configs.select_benchmark_configs(benchmark, config_root)

    groups = [_resolve(selector, config_root) for selector in selectors]
    if benchmark is not None:
        # Rejects an unknown benchmark with the list of known ones.
        _configs.select_benchmark_configs(benchmark, config_root)
        groups = [[config for config in configs if config.stem == benchmark] for configs in groups]
        for selector, configs in zip(selectors, groups, strict=True):
            if not configs:
                raise _configs.SweepSelectionError(f"{selector} has no {benchmark} configuration.")
    # A configuration named by two selectors runs once, where it first appears.
    return list(dict.fromkeys(config for configs in groups for config in configs))


def _inspect_eval_help() -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "inspect_ai", "eval", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"Could not show the options of `inspect eval`: {error}\n"
    return completed.stdout


def _print_run_help(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    print_banner()
    click.echo(ctx.get_help())
    click.echo("\nOptions forwarded to `inspect eval`:\n")
    click.echo(_inspect_eval_help(), nl=False)
    ctx.exit(0)


class _Group(click.Group):
    def get_help(self, ctx: click.Context) -> str:
        print_banner()
        return super().get_help(ctx)


@click.group(cls=_Group, context_settings={"help_option_names": ["--help"]})
@click.version_option(package_name="unsafe-eval")
def cli() -> None:
    """Run and summarize reproducible prompt-safety evaluations."""


@cli.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
    short_help="Run packaged or local run configurations.",
)
@click.option(
    "--help",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_print_run_help,
    help="Show this message, then the options of `inspect eval`, and exit.",
)
@click.option(
    "--benchmark",
    metavar="BENCHMARK_ID",
    help="Run every configuration for this benchmark instead of naming a selector.",
)
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Resolve identifiers against this directory instead of the packaged configurations.",
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help=(
        "The experiment's log directory, also where --resume looks. Each run logs to "
        "<log-dir>/<provider>/<configuration>/<benchmark>/<timestamp>/."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the selected configurations and commands without running them.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Skip configurations that already have a successful, complete run in --log-dir.",
)
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    help=(
        "Answer yes to the setup questions: reuse runs from another setup with --resume, and "
        "add runs to a log directory that holds another setup."
    ),
)
@click.argument(
    "arguments", nargs=-1, type=click.UNPROCESSED, metavar="[SELECTOR]... [INSPECT_OPTIONS]"
)
def run(
    benchmark: str | None,
    config_dir: Path | None,
    log_dir: Path,
    dry_run: bool,
    resume: bool,
    assume_yes: bool,
    arguments: tuple[str, ...],
) -> None:
    """Run one or more configurations through `inspect eval`.

    SELECTOR is one of:

    \b
      <provider>/<configuration>/<benchmark>  one packaged configuration
      <provider>/<configuration>              every benchmark for that configuration
      path/to/config.yaml                     one run configuration file

    Several selectors run together. --benchmark alone runs every configuration
    for one benchmark; with selectors, it narrows each of them to that benchmark. Selectors
    come first; every option this command does not define is forwarded to each
    `inspect eval` invocation, for example --limit, --max-connections, -M and -T. Use `--`
    to forward an option whose name this command also uses.
    """
    print_banner()
    selectors = list(itertools.takewhile(lambda argument: not argument.startswith("-"), arguments))
    inspect_args = list(arguments[len(selectors) :])
    reserved = _run.reserved_option(inspect_args)
    if reserved is not None:
        raise click.UsageError(f"{reserved} is set by unsafe-eval run and cannot be forwarded.")

    config_root = _config_root(config_dir)
    try:
        configs = _select(selectors, benchmark, config_root)
        exit_code = _run.run_sweep(
            configs,
            log_dir=log_dir,
            inspect_args=inspect_args,
            dry_run=dry_run,
            resume=resume,
            assume_yes=assume_yes,
        )
    except _configs.SweepSelectionError as error:
        _fail(str(error))
    sys.exit(exit_code)


LIST_SECTIONS = ("benchmarks", "protocols", "configs")


@cli.command("list")
@click.argument("section", required=False, type=click.Choice(LIST_SECTIONS))
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="List the configurations in this directory instead of the packaged ones.",
)
def list_command(section: str | None, config_dir: Path | None) -> None:
    """List registered benchmarks, registered protocols, and run configurations."""
    sections = (section,) if section is not None else LIST_SECTIONS
    for index, name in enumerate(sections):
        if len(sections) > 1:
            if index:
                click.echo()
            click.echo(f"{name}:")
        indent = "  " if len(sections) > 1 else ""
        for item in _list_items(name, config_dir):
            click.echo(f"{indent}{item}")


def _list_items(section: str, config_dir: Path | None) -> list[str]:
    if section == "configs":
        configs = _configs.discover_configs(_config_root(config_dir))
        return [_configs.full_config_id(config) for config in configs]
    # Importing the registry loads every built-in and installed protocol, which only
    # these two sections need.
    from unsafe_eval.protocols import registered_benchmarks, registered_protocols

    registry = registered_benchmarks() if section == "benchmarks" else registered_protocols()
    return sorted(registry)


@cli.group()
def config() -> None:
    """Locate packaged run configurations."""


@config.command("path")
@click.argument("identifier", metavar="<provider>/<configuration>/<benchmark>")
def config_path(identifier: str) -> None:
    """Print the absolute path of a packaged run configuration.

    Use it to run Inspect directly:

    \b
      inspect eval --run-config "$(unsafe-eval config path vllm/shieldstral-1.0-3b/aegis-2.0)"
    """
    try:
        path = _configs.config_path(identifier, _configs.PACKAGED_CONFIG_ROOT)
    except _configs.SweepSelectionError as error:
        _fail(str(error))
    click.echo(str(path))


@cli.command()
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "csv", "json", "html"]),
    default="markdown",
    show_default=True,
    help=(
        "Tables, drawn in a terminal and Markdown otherwise; one long-format record per "
        "metric value; or an HTML page, opened in a browser from a terminal."
    ),
)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    help="Report only this metric. Repeatable; `unsafe_f1` matches `unsafe_eval/unsafe_f1`.",
)
@click.option("--groups", is_flag=True, help="Add a table of per-group values per benchmark.")
@click.option("--all", "include_all", is_flag=True, help="Include grouped and other scores.")
@click.option(
    "--include-partial",
    is_flag=True,
    help="Also report runs limited with --limit or --sample-id, marked (partial).",
)
def report(
    log_dir: Path,
    output_format: str,
    metrics: tuple[str, ...],
    groups: bool,
    include_all: bool,
    include_partial: bool,
) -> None:
    """Summarize the metric sidecars under LOG_DIR.

    Reads only the metrics-*.json files written beside each log. For each benchmark it keeps
    the latest successful run of each setup: model, revision, protocol, protocol options, and
    dataset revision. It notes when the runs come from more than one setup of unsafe-eval and
    Inspect.

    `--format html` writes LOG_DIR/report.html, a page for comparing models across benchmarks.
    """
    from unsafe_eval.cli import _report
    from unsafe_eval.protocols import registered_benchmarks

    try:
        selection = _report.select_sidecars(log_dir, include_partial=include_partial)
    except _report.ReportError as error:
        _fail(str(error), code=1)

    unknown = _report.unknown_metrics(selection, metrics)
    if unknown:
        _fail(
            f"No reported run has metric {', '.join(unknown)}. Available: "
            f"{', '.join(_report.available_metric_names(selection))}."
        )

    if output_format == "markdown":
        _report.print_report(
            _report.report_tables(
                selection,
                registered_benchmarks(),
                metrics=metrics,
                groups=groups,
                include_all=include_all,
            )
        )
    elif output_format == "html":
        from unsafe_eval.cli import _html

        page = _html.html_report(
            selection, registered_benchmarks(), metrics=metrics, source=str(log_dir)
        )
        if sys.stdout.isatty():
            path = log_dir / "report.html"
            path.write_text(page, "utf-8")
            # The path comes first, so it is there even if no browser opens.
            click.echo(f"Report: {path}")
            _html.open_in_browser(path)
        else:
            click.echo(page, nl=False)
    else:
        # Long format is for downstream analysis, so it keeps every metric class.
        records = _report.metric_records(selection, metrics)
        render = _report.csv_report if output_format == "csv" else _report.json_report
        click.echo(render(records), nl=False)

    if selection.duplicates:
        click.echo(
            f"note: skipped {selection.duplicates} older duplicate sidecar(s); "
            "the latest run of each setup is reported.",
            err=True,
        )
    for sidecar in selection.unsuccessful:
        outcome = (
            "a successful run of it is reported"
            if _report.has_success(selection, sidecar)
            else "no successful run of it is reported"
        )
        click.echo(
            f"note: skipped the {sidecar.status} run of {sidecar.model} on {sidecar.benchmark}; "
            f"{outcome}. {sidecar.path}",
            err=True,
        )
    if selection.partial:
        click.echo(
            f"note: skipped {selection.partial} partial run(s) limited with --limit or "
            "--sample-id; pass --include-partial to report them.",
            err=True,
        )
    from unsafe_eval.cli._sidecars import environment

    setups = len({environment(sidecar.provenance) for sidecar in selection.sidecars})
    if setups > 1:
        click.echo(f"note: runs from {setups} setups; they may not be comparable.", err=True)
    for path in selection.unreadable:
        click.echo(f"warning: could not read metric sidecar {path}", err=True)
