# Working on `unsafe-eval`

## Purpose

This is a small Python package supporting Python 3.11 and newer for reproducible prompt-safety
evaluations built on Inspect AI. Keep changes narrow, typed, testable, and reproducible.
Preserve compatibility with Inspect's native tasks, solvers, scorers, model providers, and
`.eval` logs.

The benchmark data and generated logs may contain explicit or upsetting material. Do not
paste benchmark prompts or model outputs into source, tests, issues, or summaries unless the
task specifically requires it. Never commit credentials, `.env`, downloaded datasets, model
weights, caches, or files from `runs/` or `logs/`.

## Repository map

- `src/unsafe_eval/benchmarks/`: source-data loading, validation, `Sample` conversion, benchmark
  metadata, and public Inspect tasks. `_validation.py` contains only small shared record-shape
  helpers; benchmark-specific labels, filtering, and semantic validation stay in each adapter.
- `src/unsafe_eval/task.py`: public shared task composition. Benchmark tasks pass their dataset
  and metadata here; protocol resolution supplies setup, generation, scoring, and metrics.
- `src/unsafe_eval/profiles/`: model-native prompt formatting and response schemas.
- `src/unsafe_eval/providers/`: Inspect model providers this package owns, for models Inspect
  cannot reach through a stock provider. `typesafe.py` serves Jev, which answers structured
  questions instead of generating text. `laya.py` runs Laya, an open-weight model with the same
  interface, and `gliner2.py` runs GLiNER2 classifiers, both locally. A provider translates one
  conversation into one model call and holds no safety knowledge: it never decides which label
  means unsafe and never applies a threshold. Keep the vendor SDK out of the module's import
  path, because `protocols/_builtins.py` imports it to register the provider and the SDK is an
  optional extra.
- `src/unsafe_eval/protocols/`: versioned benchmark-to-model composition, benchmark semantics,
  protocol registration and extension discovery, generation settings, options, and prompt
  adaptations. `_benchmark_specs.py` is the source of truth for metric selection and grouping;
  `_resource_manifest.py` is the source of truth for packaged adaptation resources.
- `src/unsafe_eval/scorers/`: strict native-output parsing, common per-sample score shapes, and
  aggregate metrics.
- `src/unsafe_eval/_refusal.py`: recognition of provider-side content refusals for hosted
  families. `generate_or_refusal` records a recognized refusal on the sample instead of failing
  it; every other error is re-raised.
- `src/unsafe_eval/resources.py`: public packaged-text API. `_resources.py` is its implementation
  for loading, strict rendering, and content hashing.
- `src/unsafe_eval/_registry.py`: builds Inspect's entry-point inventory from benchmark exports
  and the public components owned by registered protocols.
- `tests/unit/`: focused behavior and boundary tests. `tests/integration/`: mocked pipelines,
  checked-config validation, registration, and public-API tests. The suite is offline.
- `src/unsafe_eval/configs/`: canonical benchmark/protocol run configurations with pinned input
  revisions, shipped in the wheel and laid out as `<provider>/<configuration>/<benchmark>.yaml`,
  where `<provider>` is the Inspect model provider the model string starts with. `_configs.py`
  lists and resolves them.
- `src/unsafe_eval/cli/`: the `unsafe-eval` console script. `__init__.py` holds the Click
  commands, `_run.py` builds and runs the `inspect eval` subprocesses and records provenance, which the metric hook imports inside `inspect eval`, `_report.py`
  summarizes metric sidecars, and `_sidecars.py` reads them for both. `_html.py` renders
  `_report`'s selection as the self-contained page in `report_html/`, with its fonts and mark. `run` forwards unknown options to Inspect; never re-declare an
  Inspect flag or import `inspect_ai._cli`.
- `docs/`: detailed behavior and extension guides. Treat the relevant topic page as a public
  contract alongside `README.md`.
- `pyproject.toml`: supported Python range, dependencies, entry points, and tool configuration.
- Consult Inspect AI's maintained documentation index at
  `https://inspect.aisi.org.uk/llms.txt` when an Inspect API contract is unclear.

## Setup and validation

Use `uv`; do not invoke tools from an unrelated global Python environment.

```bash
uv sync
uv run pre-commit install
```

Commits run ruff's import sorting and formatter through pre-commit. When a hook rewrites a
file, stage the result and commit again; do not bypass the hooks with `--no-verify`.

Run the narrowest relevant tests while iterating, for example:

```bash
uv run pytest --no-cov tests/unit/scorers/test_shieldstral_safety.py
```

Use `--no-cov` only for focused iteration; isolated tests otherwise fail the repository-wide
coverage threshold even when their behavior passes.

Before handing off a code or configuration change, run the full quality gate in this order:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

Pytest enforces at least 90% branch-aware coverage. The normal suite uses local samples and
Inspect mock models; it requires no network, GPU, benchmark download, or paid model call.
Do not run real `inspect eval` or `unsafe-eval run` jobs as validation unless the user
explicitly requests them: even checked-in configs can download large artifacts, require a GPU,
produce sensitive logs, or incur API costs. Validate config changes with
`tests/integration/test_configs.py` and the full suite.

If dependencies change, edit `pyproject.toml`, update `uv.lock` with `uv`, and rerun the full
quality gate. Do not hand-edit the lockfile.

## Implementation conventions

- Follow the existing module layout and nearby patterns; avoid speculative abstractions in
  this compact codebase.
- Keep all source fully typed and passing `ty check`. Use Python 3.11 syntax and a 100-column
  line limit. Ruff rules `B`, `E`, `F`, `I`, and `UP` are authoritative.
- Keep the test suite small. Add a test only when it covers code that no other test reaches or
  pins one of the evaluation invariants below. Give it one case per branch, or per distinct way
  an input can be invalid; several spellings of the same failure count as one case. Do not repeat
  a unit assertion in a CLI or integration test, re-test a shared helper in every family or
  adapter file, or assert a constant against a copy of itself. Check packaged data such as
  configs or prompt resources with one test that loops over the files and names every offender.
  Do not add source code that only tests call.
- Keep parsers strict and deterministic. Malformed, ambiguous, missing, or repeated model
  labels must remain invalid rather than being guessed from free text.
- Keep model-native prompt formatting and schemas in `profiles/`, versioned component
  composition and benchmark-to-model adaptations in `protocols/`, source-data behavior in
  `benchmarks/`, shared task assembly in `task.py`, and parsing/scoring/metrics in `scorers/`.
- Use Inspect decorators and native types for public components. Preserve serializable task,
  sample, score, and run metadata because logs are part of the reproducibility contract.
- Use public imports such as `unsafe_eval.resources`, `unsafe_eval.task`, and
  `unsafe_eval.protocols` in documentation, public-API tests, and extension code. Reserve
  underscore modules for package internals and work on their owning subsystem.
- Keep the relevant `README.md` and `docs/` pages, checked-in configs, registrations, exports,
  and tests synchronized with changes to public behavior or CLI examples.

## Registration and extension rules

- Benchmark tasks are Inspect entry-point components through `unsafe_eval.benchmarks.__all__`.
  When adding one, export it there and add its protocol semantics with `register_benchmark()`.
- Model support is registered as one versioned protocol with `register_protocol()`. Pass its
  public setup solvers and scorers through the registration's `components` argument; do not add
  separate profile or scorer inventories to `_registry.py`.
- A protocol names a model family, never an API. The API a hosted family is reached over is
  the required `route` option, resolved against `protocols/_routes.py`; add a gateway with
  `register_route()` and name it in the family's `SUPPORTED_ROUTES`. Do not add a protocol
  named after a route.
- Built-in protocol registration belongs in `protocols/_builtins.py`. Third-party integrations
  use zero-argument registration functions in the `unsafe_eval.protocols` entry-point group.
- Benchmark and protocol identifiers are intentionally open strings. Reject empty or duplicate
  registrations and unknown or ill-typed protocol options; do not introduce a closed enum that
  prevents extensions.
- Declare packaged benchmark-to-protocol adaptations with `register_prompt_resources()`. Keep
  their distribution package, relative path, empty-content rule, and SHA-256 digest accurate.
  Do not bypass the manifest or hash validation when loading adaptation text.
- Prefer the extension APIs exported by `unsafe_eval.protocols`; edit private registries only
  when changing the registration mechanism itself.

## Evaluation invariants

- Dataset and model revisions are intentionally immutable commit hashes. Do not update a pin
  as incidental cleanup. When a requested pin changes, update provenance metadata, configs,
  tests, and documentation together. Where a provider publishes no immutable identifier for the
  model, the configuration names the alias and is listed in `UNPINNED_CONFIGS` in
  `tests/integration/test_configs.py`. Such a tree is a runnable example, not a reproducible
  run; keep the exemption to those trees rather than weakening the assertion for all of them.
- `SimpleSafetyTests` and HarmBench Text Test contain only unsafe targets, so their detection
  metric is unsafe recall. Do not describe it as specificity, balanced accuracy, or classifier
  F1. HarmBench is prompt classification here, not its generative attack-success evaluation.
- Aegis is prompt-only and mixed-label. Never pass response text or response labels to the
  evaluated model, and keep redacted records excluded unless their separately distributed
  source data is explicitly brought into scope.
- Binary `yes` means unsafe and `no` means safe. Mixed-label tasks use unsafe F1 as their primary
  metric and always report balanced accuracy alongside it; unsafe is the positive class. Invalid
  outputs are misses for the target class and never count as predicted-unsafe positives.
- A provider content refusal counts as an unsafe prediction: a hit on an unsafe prompt and a
  false positive on a safe one. It keeps its own `refused` verdict and counts, so `refusal_rate`
  is always reported and mixed-label tasks also report `unsafe_refusal_rate` and
  `safe_refusal_rate`. Never drop refused samples. Keep refusal recognition narrow: rate limits,
  exhausted credit, timeouts, authentication errors, and ordinary bad requests must still fail
  the sample, because scoring them as refusals would silently raise unsafe recall.
- CoSApien and DynaGuardrail are policy-conditioned. Preserve per-sample policy metadata and the
  benchmark-specific metric grouping recorded in `protocols/_benchmark_specs.py`.
- ScopeGuard maps both `Restricted` and `Out of Scope` to unsafe. Nemotron scoring uses exactly
  one native `User Safety: safe|unsafe` line. Shieldstral prefers normalized first-token class
  log probabilities, predicts unsafe only above its threshold, and falls back to an exact
  completion only when class log probabilities are unavailable.
- Do not silently alter prompts, taxonomies, thresholds, metric denominators, invalid-output or
  refusal handling, metadata fields, or dataset filtering. Treat these as public evaluation
  semantics and test any intentional change directly.

## Working boundaries

- Read the relevant source and tests before editing. Prefer the smallest coherent patch.
- Preserve unrelated user changes and generated artifacts; do not reformat or rewrite files
  outside the task's scope.
- Do not add runtime dependencies, make network calls, launch real models, or regenerate
  evaluation logs without a task-specific reason.
- If upstream behavior matters, consult the pinned or official primary source and record any
  new revision or protocol assumption in code, tests, and user-facing documentation.
