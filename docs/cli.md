# Command line

`unsafe-eval` provides `run`, `list`, `config path`, and `report`. Each command accepts `--help`. In a terminal, `run` and the top-level help start with the `unsafe-eval` banner. `list`, `config path`, and `report` never print it, so their output can be piped or captured.

## `unsafe-eval run`

### Choosing what to run

`run` takes one or more selectors, each a [configuration](configuration.md) address or the path
of a YAML file:

```bash
unsafe-eval run vllm/shieldstral-1.0-3b/aegis-2.0 --log-dir runs  # one configuration
unsafe-eval run vllm/shieldstral-1.0-3b --log-dir runs            # every benchmark of that model
unsafe-eval run --log-dir runs --benchmark aegis-2.0              # every model on that benchmark
unsafe-eval run path/to/config.yaml --log-dir runs                # a configuration file of your own
```

Pass several selectors to run them together. Use `--benchmark` to select one benchmark for each:

```bash
unsafe-eval run vllm/shieldstral-1.0-3b vllm/qwen3-guard-gen-0.6b --log-dir runs --benchmark aegis-2.0
```

`unsafe-eval list configs` prints every configuration address. An unknown address or benchmark exits with code 2 and lists the available choices. A selector with no configuration for the requested `--benchmark` also exits with code 2. Configurations run in selector order; duplicates run once.

### Where the results go

Set `--log-dir` to the directory for your experiment. This option is required. You can add runs to the same directory later, including on different days. Each run has its own subdirectory containing an Inspect [`.eval` log](https://inspect.aisi.org.uk/eval-logs.html), a `metrics-*.json` file of aggregate scores, and `inspect.log` with everything Inspect wrote to stderr. Both `report` and `--resume` read the metrics files. Their schema is documented in [Reproducibility](reproducibility.md#metrics-file).

```text
runs/vllm/shieldstral-1.0-3b/aegis-2.0/20260924T075348Z/
```

`unsafe-eval run vllm/shieldstral-1.0-3b/aegis-2.0 --log-dir runs` writes to the directory above. Every configuration uses this layout, and all runs from one invocation share its UTC start timestamp. For your own YAML files, the path uses the two parent directories and the file name: `my-configs/vllm/my-model/aegis-2.0.yaml` writes to `<log-dir>/vllm/my-model/aegis-2.0/<timestamp>/`. Reruns and failed attempts get new directories, preserving earlier results.

While it runs, `run` prints one line per configuration and nothing from Inspect. In a terminal, that line shows a bar of finished samples until the configuration ends. When a configuration fails, it prints the last lines of its `inspect.log`. At the end, `run` prints the overview table, with each benchmark's primary metric for every run in the log directory, including earlier invocations. Runs limited by `--limit` or `--sample-id` are marked `(partial)`. It also saves the [HTML page](#the-html-page) as `report.html` in the log directory and prints its path without opening it. This happens even if `--resume` skips every configuration. Use [`report`](#unsafe-eval-report) for every metric.

### Picking up where you left off

`--resume` skips every configuration that already has a successful run over its whole dataset
anywhere under the log directory, from this invocation's layout or an older one:

```bash
unsafe-eval run vllm/shieldstral-1.0-3b vllm/qwen3-guard-gen-0.6b --log-dir runs --benchmark aegis-2.0 --resume
```

To reuse a run, its metrics file must match the configuration's task, model string, model revision, protocol, dataset revision, and every protocol option set in the file. Its `max_connections` must also match the forwarded `--max-connections` or, if none was given, the configuration's value. Concurrency affects latency. Runs limited by `--limit` or `--sample-id` never count as complete. Other forwarded options, including `-M` and `-T`, do not affect matching. Skipped runs still appear in the summary and in a [`report` of the log directory](#which-runs-it-picks).

### Mixing setups in one log directory

Each run records its setup: the `unsafe-eval` and `inspect-ai` versions, the `uv.lock` hash, and
whether unsafe-eval ran from a git tree with uncommitted changes. Runs from different setups may
not be comparable. The git commit is shown but not compared, since every commit changes it.
Runs whose metrics file is older than schema version 5 record no setup and count as "unknown".

Before it launches anything, `run` checks the setups already in the log directory. When the
directory holds runs and none of them has the current setup, it lists what differs and asks:

```text
This log directory has runs from a different setup:
  unsafe_eval_version 1.0.0 -> 1.0.1
  uv_lock_sha256 3f2a1c9e0b7d -> 9c1e07b2a4f1
  git_revision 4be1c0d -> 81aa3f2 (not compared)
Results may not be comparable. Continue? [y/N]
```

Once the directory holds a run of the current setup, it doesn't ask again.

With `--resume`, a completed run from another setup is not reused silently. `run` lists those
runs and asks once for all of them:

```text
--resume would reuse 2 run(s) from a different setup:
  openai/gpt-5.6-luna/aegis-2.0: unsafe_eval_version 1.0.0 -> 1.0.1
  openai/gpt-5.6-luna/harmbench-text-test: unsafe_eval_version 1.0.0 -> 1.0.1
y: reuse them as they are
N: rerun them with the current setup
Reuse? [y/N]
```

After this question, the first one isn't asked.

Without a terminal, as under `nohup` or in CI, `run` can't ask: it prints the differences and
exits with code 2. `--yes` answers yes to both questions. `--dry-run` prints the differences and
never asks.

### Options

`run` has these options of its own:

- `--benchmark ID` runs every configuration of one benchmark, or narrows the selectors to it.
- `--config-dir DIR` resolves addresses against `DIR`, laid out as
  `DIR/<provider>/<configuration>/<benchmark>.yaml`, instead of the packaged configurations.
- `--log-dir DIR` names the experiment's log directory. It is required, so that `--resume` looks
  in the same folder every time.
- `--dry-run` prints the selected configurations and their `inspect eval` commands, and runs
  and writes nothing.
- `--resume` skips configurations that [already completed](#picking-up-where-you-left-off).
- `--yes`, `-y` answers yes to the [setup questions](#mixing-setups-in-one-log-directory).

Every other option goes to `inspect eval` unchanged, once per configuration: `--limit`,
`--sample-id`, `--max-connections`, `-M`, `-T` and the rest of
[Inspect's options](https://inspect.aisi.org.uk/options.html). Put the selectors first, then
any Inspect options:

```bash
unsafe-eval run vllm/shieldstral-1.0-3b --log-dir runs --limit 1 --max-connections 16
unsafe-eval run --log-dir runs --benchmark aegis-2.0 --dry-run -- --limit 1
```

After `--`, every argument is forwarded, including one that shares a name with an option of
`run`. `run` sets `--json`, `--run-config` and `--log-dir` itself, so forwarding any of them is
an error. `unsafe-eval run --help` lists `run`'s options, then those of the installed
`inspect eval`.

### Failures and exit codes

Each configuration runs in its own `inspect eval` process, so a local runtime may load the model
again for each one. A [vLLM server you already run](configuration.md#local-models) avoids that.

If a configuration fails during evaluation, the remaining configurations still run. A failure before Inspect starts stops the rest because a missing extra, API key, or file is likely to affect them too. Such failures leave no log. Fix the error and rerun with `--resume`.

`run` exits with:

- 0 when every configuration succeeded, or none was left to run;
- 1 when any configuration failed, or you answered no to adding runs of another setup;
- 2 for a selection or usage error, such as a missing `--log-dir`, a forwarded reserved flag,
  or a setup question with no terminal to ask on;
- 130 when interrupted.

## `unsafe-eval list`

```bash
unsafe-eval list
unsafe-eval list protocols
unsafe-eval list configs --config-dir path/to/configs
```

With no argument it prints three sections: the registered benchmarks, the registered protocols,
and every configuration address. Name `benchmarks`, `protocols` or `configs` to print one
section, one identifier per line. Protocols that another package registers through the
[`unsafe_eval.protocols` entry point](extending.md#extension-api) appear too.

## `unsafe-eval config path`

This prints the absolute path of a packaged configuration, for running Inspect yourself:

```bash
inspect eval \
  --run-config "$(unsafe-eval config path vllm/scope-guard-4B-q-2601/aegis-2.0)" \
  --log-dir runs/vllm/scope-guard-4B-q-2601
```

Inspect doesn't read a log directory from a run configuration and
[defaults to `./logs`](https://inspect.aisi.org.uk/options.html#eval-logs), so pass `--log-dir`
yourself. `unsafe-eval run` does that for you.

## `unsafe-eval report`

Point `report` at a log directory for that experiment, or at a run directory for that run alone:

```bash
unsafe-eval report runs
unsafe-eval report runs/vllm/shieldstral-1.0-3b/aegis-2.0/20260924T075348Z
unsafe-eval report runs --format csv > results.csv
unsafe-eval report runs --format html > results.html
```

It reads only metrics files, never an Inspect log, so no prompt or model output reaches the
report. Each row is one setup: a model, its revision, protocol, protocol options, dataset
revision and `max_connections`.

### What it shows

With the default `--format markdown`, the tables are drawn in a terminal and written as Markdown
when redirected. An overview comes first, with each benchmark's
[primary metric](metrics.md) per row. One table per benchmark follows with all its headline
metrics, primary first, and then the median and 95th-percentile seconds the model took per
sample (`latency_median_seconds`, `latency_p95_seconds`) and the mean input, output and reasoning
tokens per completed sample (`input_tokens_per_sample`, `output_tokens_per_sample`,
`reasoning_tokens_per_sample`). See [measuring latency](#measuring-latency) for what those times
cover.

- `--groups` adds each benchmark's per-group values, such as per harm area.
- `--all` adds those and every other score, such as `verdict` frequencies.
- `--metric NAME` limits the benchmark tables to the named metrics. It can be repeated, and
  `unsafe_f1` matches `unsafe_eval/unsafe_f1`. A name that no run has exits with code 2 and
  lists the metrics that exist.
- `--format csv` and `--format json` write one record per metric value, with the fields
  `model`, `revision`, `benchmark`, `protocol`, `protocol_options`, `dataset_revision`,
  `partial`, `score`, `metric`, `value`, `run_id` and `eval_id`. They include every score, and
  `--metric` filters them too.
- `--format html` writes the [HTML page](#the-html-page). In a terminal it saves the page as
  `report.html` in the log directory, replacing the one `run` saved there, prints the path, and
  opens it in a browser when there is a display (over SSH, copy the path). Redirected, it writes
  the page to standard output.

### The HTML page

The page is a single file that needs no network. It holds only what the metrics files hold, so
it is safe to share.

- The page opens on a summary with balanced accuracy, unsafe F1, unsafe recall, safe recall and
  median latency per row, each the unweighted mean over the shown benchmarks that report it.
  Hiding a benchmark with the filter takes it out of every average. A row missing one of them
  gets no average: it shows how many it has, such as `5/6`, and sorts last. With a single
  benchmark shown, the summary lists that benchmark's own values.
- Pick one metric under Metric, including latency and tokens per sample, to see it on every
  benchmark next to an Average column with the same rules. A benchmark that doesn't report it
  shows its primary metric instead, in italics. With a single benchmark shown, there is no
  Average column.
- Click a column's header to sort by it, and a benchmark's name or cells to open its details.
  Hover over a row's name for its setup and the `run_id` and `eval_id` behind each value. The
  best value in each column is in mint, leaving out partial runs. For latency and tokens, lower
  is best.
- A baseline shows every other row as differences from it, marked ⚠ when the two runs used
  different dataset revisions or either was partial. Latency and token differences are colored
  by their size relative to the baseline's value.
- A benchmark's details list its headline metrics and chart its per-group values. Only the
  detection metrics, latency and tokens are colored, so rates such as `invalid_output_rate` are
  plain numbers.
- The subtitle counts the runs left out. Hover over it for each unsuccessful run's path and
  whether a successful run of the same setup is shown instead.
- `--metric` limits the detail tables. `--groups` and `--all` don't apply: the page has every
  score, with the other scores behind a toggle.

### Which runs it picks

`report` selects successful runs and keeps the latest run of each setup for each benchmark. It notes how many older repeats it skipped, and names each unsuccessful run it skipped and whether a successful run of the same setup is reported. Rows are named after the model, with the first seven characters of its revision only when it ran at several revisions. To distinguish runs, row names also include the protocol, option set, dataset revision, or `max_connections` when those values differ.

When the reported runs were made with more than one unsafe-eval setup, `report` adds a note on
standard error, since such runs may not be comparable.
[Mixing setups](#mixing-setups-in-one-log-directory) says what that setup is.

Runs limited with `--limit` or `--sample-id` are left out unless `--include-partial` adds them,
in rows marked `(partial)`. A benchmark that is not registered still gets its tables, with an
empty overview cell because it has no primary metric. A directory with no usable metrics file
exits with code 1.

### Measuring latency

A sample's latency is the time Inspect records for the evaluated model's requests, excluding waits for a free connection and failed retries. What that time covers depends on the provider:

- vLLM reports each HTTP request's own time, and its continuous batching admits a new request
  into the running batch at the next step, so a request does not wait for the GPU.
- The `laya`, `gliner2`, `clef`, `d1` and `clm` providers run the requests that queued while the GPU was busy as one
  batch, and report that batch's own time, leaving out the wait for the batch before it.
- The `kev` provider records the time for each HTTP request to `kev.serve`, including time in the server's batch queue.
- Hosted providers, including `typesafe` for Jev, record the successful request's own time:
  Inspect retries a failed request and leaves the failed attempt out.
- Inspect's `hf` provider records the whole call, including up to 2 seconds spent collecting a
  batch and up to 1 second polling for its result, so its latency is not the model's.

The local configurations (`vllm`, `hf`, `laya`, `gliner2`, `clef`, `kev`, `d1` and `clm`) set `max_connections: 4` to measure latency under a light load. A single request at a time gives an unrepresentative deployment load, while filling the GPU slows individual requests. Override concurrency with `--max-connections N`, for example to prioritize throughput when only detection metrics matter:

```bash
unsafe-eval run vllm/shieldstral-1.0-3b --log-dir runs --max-connections 256 --resume
```

`--resume` and `report` both tell runs at different concurrency apart, so such runs can share a
log directory with the default ones. A hosted model's provider batches however it chooses, so
its configurations leave concurrency to Inspect. Every time includes tokenization and, for a
hosted model, the network round trip, and depends on the hardware or the provider's load at the
time.
