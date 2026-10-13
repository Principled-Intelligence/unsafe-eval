# Extending unsafe-eval

Choose a guide for the component you want to add. Each starts from the current repository and can be followed on its own:

- [Add a benchmark](adding-benchmark.md): reproduce SimpleSafetyTests under a separate alias,
  then run one real Shieldstral evaluation.
- [Add a model](adding-model.md): add a model-native prompt, strict scorer, protocol builder,
  registration, and compatibility boundary.
- [Add a metric](adding-metric.md): implement, attach, test, and document a metric computed from
  normalized safety scores.

Install the development environment first:

```bash
uv sync
```

## Where a model's code lives

Each model has one module in each of three packages, named after the model:

| Concern | Location | ScopeGuard example |
| --- | --- | --- |
| Native messages and response schema | `src/unsafe_eval/profiles/` | `profiles/scope_guard.py` |
| Benchmark composition, options, generation settings, and resources | `src/unsafe_eval/protocols/` | `protocols/scope_guard.py` |
| Native output parsing and per-sample scores | `src/unsafe_eval/scorers/` | `scorers/scope_guard_safety.py` |

Each family has its own system message, user template, schema field descriptions, and benchmark adaptations. Keep separate prompt files even when two families use identical text. Shared helpers contain no family-specific prompt text. The hosted families use `protocols/_hosted.py` for option resolution and generation requests, `profiles/_structured_label.py` for field selection and template rendering, `scorers/_structured_label.py` for strict parsing and scoring, and `unsafe_eval._refusal` to record provider refusals. No family imports another family's module.

Benchmark adapters in `src/unsafe_eval/benchmarks/` load and validate source records, then pass
them to `unsafe_eval.task.build_prompt_safety_task()`, which resolves the protocol. Shared binary
metrics are in `scorers/metrics.py`, and `protocols/_benchmark_specs.py` decides which metrics
each benchmark reports.

## Extension API

Everything in this section is exported from `unsafe_eval.protocols`.

Benchmark and protocol identifiers are open strings, so an extension can add either without
touching the package. `register_benchmark()` and `register_protocol()` add a benchmark or a model
to normal task resolution and reject empty or duplicate identifiers. `registered_benchmarks()` and
`registered_protocols()` return read-only snapshots of what is registered.

`register_protocol()` takes a model's builder together with its public Inspect setup solvers and
scorers, and rejects a component name that is already taken. The package's
[Inspect entry point](https://inspect.aisi.org.uk/extensions-components.html) is built from these
registrations, so a registered component can be referenced by name.

To distribute an extension, add a zero-argument registration function to the `unsafe_eval.protocols` entry-point group. Importing `unsafe_eval.protocols` calls that function once. It can register benchmarks, protocols and their components, prompt resources, and routes through the same API. If loading or registration fails, the import raises and the run cannot start with an incomplete protocol. Check that your protocol appears in `unsafe-eval list protocols`.

`BenchmarkProtocolSpec` holds the evaluation semantics a benchmark shares across all protocols:
whether it has mixed labels, its primary metric, the metadata fields its metrics are grouped by,
and whether each sample carries its own policy. By default it chooses the built-in mixed-label or
unsafe-only metric set. A benchmark with other semantics can supply a `metric_builder`, and with it
a primary metric other than `unsafe_f1` or `unsafe_recall`.

`PromptResource` and `register_prompt_resources()` declare prompt adaptations by distribution package and relative path. Extensions can package their own text, subject to the same [hash validation and empty-content rule](protocols.md#prompt-resources) as built-in resources. The default layout is `protocols/prompts/<benchmark>/<protocol>/`.

`protocol_options()` validates a protocol's options. It rejects any name outside the set the
protocol passes it and always accepts `validate_prompt_resource_hashes`, which every protocol must
honour. A protocol that checks its own options by hand can reject that one by mistake.

If Inspect has no provider for a model, write one using [Inspect's model API guide](https://inspect.aisi.org.uk/extensions-model-api.html). This package's providers are `typesafe`, `laya`, `gliner2`, `clef`, `kev`, `d1`, and `clm`.

## Declaring routes

A protocol whose model is served in more than one place lists the [routes](protocols.md#routes)
it supports and resolves the `route` option against them:

```python
PROTOCOL: ProtocolName = "gpt"
SUPPORTED_ROUTES: tuple[str, ...] = ("openrouter", "openai")

route = route_option(values, SUPPORTED_ROUTES)
seed = GENERATION_SEED if route.honours_seed else None
```

`route_option()` requires a route name. It reports separate errors for an unregistered name, which may be a typo, and a registered route that the protocol doesn't support. Record `route.name` in the protocol options so the log identifies the service that answered.

`Route`, `register_route()`, `route_spec()` and `registered_routes()` cover the rest. A `Route`
has a `name`, the Inspect provider prefix `model_api`, and two capability flags, `honours_seed`
and `forwards_chat_template_kwargs`. Route names are open strings too, and `register_route()`
rejects empty and duplicate names, so an extension can add a gateway.

A flag describes a gateway together with the models behind it. `openrouter` honours a seed
because GPT, Nemotron and Gemini are served over chat completions there, but OpenRouter's
Anthropic endpoints take no seed, so `claude` sends none and does not consult the flag. A
protocol that reuses an existing route must check each flag against its own upstream. A protocol
that controls its model through the chat template checks `forwards_chat_template_kwargs` and
refuses a request the route cannot carry out, as Nemotron does on `openrouter`.
