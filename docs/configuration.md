# Configuration

An [Inspect run configuration](https://inspect.aisi.org.uk/tasks.html#run-config) defines one evaluation: its benchmark, model, protocol, and dataset and model revisions. The package includes a [configuration](#packaged-configurations) for every supported model and benchmark pairing. You can also [write your own](#writing-a-configuration).

## Packaged configurations

A packaged configuration is addressed as `<provider>/<configuration>/<benchmark>`, for example
`vllm/shieldstral-1.0-3b/aegis-2.0`. `<provider>` is the Inspect provider the model runs on,
which is the prefix of its model string: a local runtime such as `vllm` or a hosted API such as
`openrouter`. `<configuration>` names the model, and `<benchmark>` is a
[benchmark ID](benchmarks.md). A model served by two providers has a configuration of the same
name under each.

The [protocol table](protocols.md) lists the configurations of every protocol, and
`unsafe-eval list configs` prints every address. [Command line](cli.md) covers the commands that
run them.

## Writing a configuration

Start from the closest packaged configuration when adding a model or changing a setting. Find its file with `unsafe-eval config path <address>`, then copy and edit it. For example, `vllm/shieldstral-1.0-3b/simple-safety-tests` contains:

```yaml
task:
  task: unsafe_eval/simple_safety_tests
  args:
    protocol: shieldstral-1.0-3b
    dataset_revision: 98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5
    protocol_options:
      threshold: 0.5

model:
  model: vllm/mistralai/Shieldstral-1.0-3B
  args:
    revision: 003ec7e2b0bab5f0e6307edbaf186fa5822b76f5
    dtype: bfloat16
    max_model_len: 32768
```

`task` defines the benchmark, prompting, and scoring. `model` identifies the model to call.

### `task`

- `task.task` is the Inspect task, which picks the benchmark, such as
  `unsafe_eval/simple_safety_tests`. [Benchmarks](benchmarks.md) names the task of each one.
- `task.args.protocol` names the [protocol](protocols.md), which decides the prompt the model
  receives, how its output is scored, and its generation settings. Not every protocol runs every
  benchmark: Qwen3Guard-Gen and PolyGuard-Qwen, for example, reject the policy-conditioned
  CoSApien and DynaGuardrail, and the task fails to build.
- `task.args.protocol_options` sets that protocol's [options](protocols.md). An option you leave
  out takes its default, except `route`, which has none and must be the model string's prefix
  ([routes](protocols.md#routes)). Unknown options and values of the wrong type stop the run
  before it starts. While you edit a packaged prompt, `validate_prompt_resource_hashes: false`
  turns a hash mismatch into a warning ([prompt resources](protocols.md#prompt-resources)).
- `task.args.dataset_revision` is the commit of the benchmark's source dataset. Without it the
  task loads the [pinned commit](reproducibility.md#dataset-revisions). Set it anyway if you use
  `unsafe-eval run --resume`, which looks for an earlier run with the same value: a file without
  one never finds a match, so it always runs again. Another commit may fail to load if the data
  has changed, and its scores cannot be compared with runs on the pinned one.

### `model`

- `model.model` is the [Inspect model string](https://inspect.aisi.org.uk/models.html). Its
  prefix picks the [provider](#runtimes-and-providers) and the rest names the model on it, such
  as `vllm/mistralai/Shieldstral-1.0-3B` or `openrouter/anthropic/claude-sonnet-5-20260630`.
- `model.args` goes to the provider unchanged, like `-M` on the command line
  ([provider arguments](https://inspect.aisi.org.uk/providers.html)). On a local model,
  `revision` is the Hugging Face commit that pins the weights. The providers this package adds
  take these arguments:
  - `laya`: `revision`, a 40-character hub commit, and `max_len`, the most tokens Laya reads per
    question (512 to 8192), are required, and `device` is optional.
  - `d1`: `revision`, a 40-character hub commit, is required.
  - `gliner2` and `clef`: `revision` is required, and `device` defaults to `cuda`. `clef` also
    takes `quantization: fp8`.
  - `typesafe`: `revision` is optional and must repeat the model version in the model string.
  - `kev`: `revision` pins the checkpoint's hub commit, and `kev_revision` pins Kev's server code. Both are required. `quantization: fp8` stores a full-weight checkpoint's linear layers in FP8. The provider starts its own server and rejects `base_url`.
  - `clm`: `revision` pins the heads' hub commit, and `encoder_revision` pins Qwen3-8B, the encoder they score. Both are required.

  `laya`, `gliner2`, `clef`, `kev`, `d1` and `clm` reject any other argument.
- `model.base_url` points the provider at an endpoint of your own, as `--model-base-url` does,
  such as a vLLM server you already run.
- `model.config` holds generation settings for this model. A field also set in
  `generate_config` takes `generate_config`'s value.

### Other settings

Inspect rejects unknown top-level keys before the run starts. These accepted keys can change the evaluation:

- `generate_config`
  ([generation options](https://inspect.aisi.org.uk/options.html#model-generation)) replaces the
  protocol's own settings, such as `temperature`, `max_tokens`, `seed` or `reasoning_effort`, for
  the whole run, and so changes the evaluation. Limit it to operational fields: `timeout` or
  `max_retries` for a slow API, or `max_connections`, which is also how many prompts the `hf`
  provider [batches](https://inspect.aisi.org.uk/providers.html#batching).
  Packaged configurations set at most `max_connections`: 4 for every local model, so their
  [latency](cli.md#measuring-latency) is measured under a light load.
- `eval_config`: a `limit` or `sample_id`
  ([sample selection](https://inspect.aisi.org.uk/options.html#sample-selection)) makes the run
  partial, as the command-line flags do, so `--resume` never counts it.
- `solver`: leave it out. It replaces the protocol's solver, and the protocol's scorer then reads
  output it was not written for.

### Overriding a file from the command line

For a one-off change, such as different weights or an option value, pass a flag. `unsafe-eval run` forwards options it does not define to [`inspect eval`](https://inspect.aisi.org.uk/reference/inspect_eval.html). Use `-T key=value` to set a key under `task.args` or `-M key=value` to set one under `model.args`. The other keys in the file are preserved:

```bash
unsafe-eval run vllm/shieldstral-1.0-3b/simple-safety-tests --log-dir runs -M revision=<commit>
```

A flag replaces the key's whole value, so `-T protocol_options=...` discards every option the
file sets there. Repeat the ones you want to keep:

```bash
unsafe-eval run anthropic/claude-sonnet-5/aegis-2.0 --log-dir runs \
  -T 'protocol_options={route: anthropic, output_fields: [reasoning, user_safety], reasoning_effort: none}'
```

`--resume` compares earlier runs with the file and ignores the flags of the current command. An
earlier run made with other flags or generation settings still counts as a completed run of the
file when the [fields it compares](cli.md#picking-up-where-you-left-off) match.

### Running your file

`unsafe-eval run path/to/file.yaml` runs a file saved anywhere. Its log directory is named after
the file's two parent directories and its name, as
[Where the results go](cli.md#where-the-results-go) describes. To address your files by
identifier and sweep them like the packaged ones, lay them out as
`<dir>/<provider>/<configuration>/<benchmark>.yaml` and pass `--config-dir <dir>`.

## Runtimes and providers

For most providers, install the relevant extra and supply a credential. The package includes providers for models Inspect cannot otherwise reach. `typesafe` calls the TypeSafe System One API. `kev` runs Kev's open System One models on a local server. `orbitals` runs ScopeGuard through its own SDK on a local vLLM server. `clm` runs CLM's heads in-process over a local Qwen3-8B server. `laya`, `gliner2`, `clef`, and `d1` run in-process. To write another provider, see Inspect's [Model API guide](https://inspect.aisi.org.uk/extensions-model-api.html).

| Provider | Model prefix | Install extra | Credential | Revision pin<sup>1</sup> |
| --- | --- | --- | --- | --- |
| [OpenAI](https://inspect.aisi.org.uk/providers.html#openai) | `openai/<model>` | `openai` | `OPENAI_API_KEY` | Snapshot, where published |
| [Anthropic](https://inspect.aisi.org.uk/providers.html#anthropic) | `anthropic/<model>` | `anthropic` | `ANTHROPIC_API_KEY` | Snapshot |
| [Google](https://inspect.aisi.org.uk/providers.html#google-gemini) | `google/<model>`, or `google/vertex/<model>` for Vertex AI | `google` | `GOOGLE_API_KEY`; for Vertex, `gcloud auth application-default login` and `GOOGLE_CLOUD_PROJECT` | Snapshot, except previews |
| [OpenRouter](https://inspect.aisi.org.uk/providers.html#openrouter) | `openrouter/<org>/<model>` | `openrouter` | `OPENROUTER_API_KEY` | Snapshot, where published |
| TypeSafe | `typesafe/<model-version>` | `typesafe` | `TYPESAFE_API_KEY` | Exact version, verified |
| [vLLM](https://inspect.aisi.org.uk/providers.html#vllm) | `vllm/<hub-repo>` | `vllm`; `openai` for remote | `VLLM_BASE_URL`, `VLLM_API_KEY` | Yes |
| [Hugging Face](https://inspect.aisi.org.uk/providers.html#hugging-face) | `hf/<hub-repo>` | `hf` | `HF_TOKEN` (gated repos only) | Yes |
| Laya | `laya/<hub-repo>` | `laya` | None | Yes, required |
| GLiNER2 | `gliner2/<hub-repo>` | `gliner` | None | Yes, required |
| Clef | `clef/<hub-repo>` | `clef` | None | Yes, required |
| d1 | `d1/<hub-repo>` | `hf` | None | Yes, required |
| Kev | `kev/<hub-repo>` | `typesafe`, and `uv` on `PATH` | None | Yes, required, with Kev's code commit |
| Orbitals | `orbitals/<hub-repo>` | `orbitals` | None | Yes, required |
| CLM | `clm/<hub-repo>` | `clm` | None | Yes, required, with the encoder's commit |

<sup>1</sup> [Reproducibility](reproducibility.md#hosted-models-a-snapshot-where-one-exists)
explains what a snapshot pins and what an unpinned configuration still fixes.

Install only the extras your runtime needs:

```bash
# Local vLLM on Linux
pip install "unsafe-eval[vllm]"

# Local Hugging Face transformers
pip install "unsafe-eval[hf]"

# Hosted API providers, one or several
pip install "unsafe-eval[openai,anthropic,google,openrouter,typesafe]"

# Local GLiNER2 classifiers
pip install "unsafe-eval[gliner]"

# Local Laya
pip install "unsafe-eval[laya]"

# Local Clef
pip install "unsafe-eval[clef]"

# Local Kev, which also needs uv
pip install "unsafe-eval[typesafe]"

# Local ScopeGuard through the orbitals SDK, on Linux
pip install "unsafe-eval[orbitals]"

# Local CLM, on Linux
pip install "unsafe-eval[clm]"

# Every runtime and provider except gliner
pip install "unsafe-eval[all]"
```

From a checkout, `uv sync --extra <name>` installs the same extras into `.venv`, and
`--extra all --extra gliner` installs all of them. Prefix the commands on this page with
`uv run`, or activate the environment with `source .venv/bin/activate`.

### Local models

The guard models run on vLLM, normally on an NVIDIA GPU. To use a vLLM server you already run
instead of starting one, set `VLLM_BASE_URL` and `VLLM_API_KEY`
([vLLM Server](https://inspect.aisi.org.uk/providers.html#vllm-server)).

Qwen3Guard-Gen 0.6B and ScopeGuard v2 also have configurations under `hf/` for machines without
vLLM:

```bash
unsafe-eval run hf/qwen3-guard-gen-0.6b/simple-safety-tests --log-dir runs
```

ScopeGuard also runs through `orbitals`, its own SDK, which builds the prompt and parses the
answer in place of this package. Each `orbitals/` tree shares its name with the `vllm/` tree it
mirrors and starts its own `vllm serve` with the same flags, so the two report comparable
latency. `scope-guard-v2-4B-q-2609-classify` has no `vllm/` twin: it reads the class off the
first answer token's log-probabilities instead of generating it.

```bash
unsafe-eval run orbitals/scope-guard-v2-4B-q-2609/simple-safety-tests --log-dir runs
```

Laya, GLiNER2.5-Decide, GLiGuard, Clef-Flash and d1-3B run in-process through the `laya/`,
`gliner2/`, `clef/` and `d1/` providers. They need no server or API key:

```bash
unsafe-eval run laya/laya/simple-safety-tests --log-dir runs

unsafe-eval run gliner2/gliner2.5-decide/simple-safety-tests --log-dir runs

unsafe-eval run clef/clef-flash/simple-safety-tests --log-dir runs

unsafe-eval run d1/d1-3b/simple-safety-tests --log-dir runs
```

Kev requires torch below 2.9, which conflicts with this package's vLLM requirements. The `kev/` provider starts `kev.serve` with `uv run --isolated` in a separate environment built from the pinned Kev commit. It uses the torch, transformers, and peft versions in Kev's lockfile and the flash-linear-attention version in Kev's serving image.

The first run builds the environment and downloads the weights. Kev-27B's 51 GB of bf16 weights require an 80 GB GPU; `kev/kev-27b-fp8` stores them in FP8 instead and multiplies in bf16, because the FP8 matmul kernel needs a newer Triton than Kev's torch. It also turns off Kev's fused kernels and CUDA graphs. To run Kev-0.8B:

```bash
unsafe-eval run kev/kev-0.8b/simple-safety-tests --log-dir runs
```

CLM runs its projection heads in-process and starts a Qwen3-8B vLLM server for the embeddings, with CLM's reference settings: last-token pooling over at most 2,048 tokens. States longer than that are truncated, as in CLM's own server. The encoder's bf16 weights take about 16 GB:

```bash
unsafe-eval run clm/clm-v0.1-8b/simple-safety-tests --log-dir runs
```

### Hosted models

The GPT, Claude, Gemini and Jev configurations run on hosted APIs. They need no GPU, but every
sample is billed, so try a sweep with `--limit` first.

```bash
export OPENROUTER_API_KEY=your-key
unsafe-eval run openrouter/gpt-5.6-sol/simple-safety-tests --log-dir runs

export OPENAI_API_KEY=your-key
unsafe-eval run openai/gpt-5.6-luna/simple-safety-tests --log-dir runs

export ANTHROPIC_API_KEY=your-key
unsafe-eval run anthropic/claude-haiku-4.5/simple-safety-tests --log-dir runs

gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project
unsafe-eval run google/gemini-3.8-flash/simple-safety-tests --log-dir runs

export TYPESAFE_API_KEY=your-key
unsafe-eval run typesafe/jev-1.13/simple-safety-tests --log-dir runs
```

### Other models

To evaluate a model with no packaged configuration, name it and a protocol directly, with a
prefix from the [provider table](#runtimes-and-providers):

```bash
export OPENROUTER_API_KEY=your-key
inspect eval unsafe_eval/simple_safety_tests \
  --model openrouter/<org>/<model> \
  --log-dir runs/<compatible-versioned-protocol> \
  -T protocol=<compatible-versioned-protocol> \
  -T 'protocol_options={route: openrouter}' \
  -M 'provider={order: [<upstream>], allow_fallbacks: false, require_parameters: true}'
```

Choose a protocol for the model's family. A protocol expects that family's prompt and output format. Using it with a different family may complete a run, but the scores are meaningless. If no built-in protocol covers the model, register one through the [extension API](extending.md#extension-api).

OpenRouter can serve one model from several upstream services, and some of them ignore the JSON
response schema. The `provider` argument controls which one answers:

- `order` lists the upstreams to use, by OpenRouter's slug. The packaged configurations name the
  model's vendor: `openai`, `anthropic` or `google-vertex/global`.
- `allow_fallbacks: false` makes a request fail instead of moving to an upstream not in `order`.
- `require_parameters: true` skips any upstream that cannot honour every request parameter,
  including the response schema.

Without the argument, a request can reach an upstream that ignores the schema, and its answer is
scored as an invalid output.

## Direct task execution

The benchmarks are regular Inspect tasks, so
[`inspect eval`](https://inspect.aisi.org.uk/reference/inspect_eval.html) runs them without a
configuration file:

```bash
inspect eval unsafe_eval/simple_safety_tests \
  --model vllm/mistralai/Shieldstral-1.0-3B \
  --log-dir runs/vllm/shieldstral-1.0-3b \
  -T protocol=shieldstral-1.0-3b
```

So does Inspect's Python [`eval()`](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval):

```python
from inspect_ai import eval
from unsafe_eval.benchmarks import toxic_chat_prompt_safety

eval(
    toxic_chat_prompt_safety(protocol="shieldstral-1.0-3b"),
    model="vllm/mistralai/Shieldstral-1.0-3B",
    log_dir="runs/vllm/shieldstral-1.0-3b",
)
```

Either way the run writes the same `.eval` log and [metrics file](reproducibility.md#metrics-file)
as `unsafe-eval run`.
