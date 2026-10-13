# Reproducibility

Keep the configuration and Inspect `.eval` log together when publishing or comparing a result. Every packaged configuration pins its dataset to an immutable revision and pins the model where the provider allows it.

## Dataset revisions

| Benchmark | Revision |
| --- | --- |
| SimpleSafetyTests | `98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5` |
| HarmBench | `8e1604d1171fe8a48d8febecd22f600e462bdcdd` |
| Aegis 2.0 | `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c` |
| OpenAI Moderation | `f4ab51b5edd3bfbcb349a56324274235b674e0e4` |
| ToxicChat 0124 | `29df8e4dba60e1f4af4b4075c0705c5b313548a8` |
| WildGuardMix | `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` |
| CoSApien | `29bb85bd4f84319d825517519278c8f8959fa031` |
| DynaGuardrail | `3c21998805b692819e527f88702f761e33f6a013` |

Changing a pin changes what is evaluated, so the configurations, provenance metadata, tests and
documentation that name it change with it.

HarmBench and OpenAI Moderation download a file from GitHub rather than the Hugging Face Hub. At the pinned revision, task metadata records the expected SHA-256 as `dataset_file_sha256`. The download is not checked against that digest. For HarmBench's CSV, the expected digest is:

```text
75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29
```

OpenAI Moderation's gzipped JSONL:

```text
ef3f7de86ee394fe337b0a566ba4b9c7db058f3caa8d6a4a0006138b663ab9a8
```

With any other revision the task leaves the digest out, since it would describe a different file.

## Model revisions

| Model configuration | Model repository | Revision |
| --- | --- | --- |
| Shieldstral 1.0 3B | `mistralai/Shieldstral-1.0-3B` | `003ec7e2b0bab5f0e6307edbaf186fa5822b76f5` |
| `scope-guard-4B-q-2601` | `principled-intelligence/scope-guard-4B-q-2601` | `29fcf6a99b4487f5cbccfbf506d95d7c1724bc93` |
| `scope-guard-4B-g-2601` | `principled-intelligence/scope-guard-4B-g-2601` | `801b1a8a1f07a8606dbba067b0629b9f84f3a931` |
| `scope-guard-v2-4B-q-2609` | `principled-intelligence/scope-guard-v2-4B-q-2609` | `0110d5d0f50c537513096f597d46ff4a1ee18377` |
| Nemotron 3.5 Content Safety | `nvidia/Nemotron-3.5-Content-Safety` | `35645ed3543b7e7ffaed2e788699e57a5051497c` |
| PolyGuard-Qwen | `ToxicityPrompts/PolyGuard-Qwen` | `644bfe73ff498c9a14818b72a11187eaf23f0ff1` |
| Qwen3Guard-Gen 0.6B | `Qwen/Qwen3Guard-Gen-0.6B` | `fada3b2f655b89601929198343c94cd2f64d93cc` |
| Qwen3Guard-Gen 4B | `Qwen/Qwen3Guard-Gen-4B` | `6ec42827da0c1ff11e7a49dc269d2e810d27e108` |
| Qwen3Guard-Gen 8B | `Qwen/Qwen3Guard-Gen-8B` | `4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb` |
| Laya (English and multilingual, routed) | `convaiinnovations/laya` | `1c5edc17a7acd8701df6fc341c0d179f1c62c982` |
| GLiNER2.5-Decide | `fastino/GLiNER2.5-Decide` | `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` |
| GLiGuard 300M | `fastino/gliguard-LLMGuardrails-300M` | `0e9b87233990dd3ff657ef7a540ee2b62b07c7d6` |
| Clef-Flash | `Cloudflare/clef-flash` | `17f0b0ad64efb65d273590632833508766b2aae6` |
| Clef (27B, run in FP8) | `Cloudflare/clef` | `2f3de3dd85f379784083b0814d997ab627200f0c` |
| Kev-27B (bf16, or run in FP8) | `jaredpalmer/kev-27b` | `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf` |
| Kev-9B | `jaredpalmer/kev-9b` | `db029f08b290afd9fee4aa4bbcd9ae48602d1eb0` |
| Kev-4B | `jaredpalmer/kev-4b` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| Kev-0.8B | `jaredpalmer/kev-0.8b` | `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` |
| Parakh-4B | `YAlgoG/parakh-4b` | `f2992d283fd8eac93ea2b15712d23eb8fcc467df` |
| d1-3B | `LiquidAI/d1-3B` | `da1fe36a861f24690f27f622dca1d8688503d113` |
| CLM-v0.1-8B heads | `Contrastive-LM/CLM-v0.1-8B` | `e939398d4556fcd9400c76fa8c5a513202f42b0a` |
| CLM-v0.1-8B encoder | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

How exactly a run can be repeated depends on where the model is served.

### Self-hosted models: exact

`vllm/` and `hf/` configurations pass a hub commit as the `revision` model argument, so the table
identifies the exact weights. `laya/`, `gliner2/`, `clef/`, `d1/` and `orbitals/` require one, and each
provider downloads the checkpoint at that commit and loads it from there. GLiNER2 needs this
because the `gliner2` package's loader accepts a revision but still fetches the tokenizer from
`main`, and d1 because its code loads the tokenizer from the model's own path without a revision.

`kev/` requires both a hub commit and `kev_revision`, which pins Kev's server code. The server constructs the model input and loads the adapter onto the base revision named in the checkpoint. The provider fails the run unless `/v1/models` reports exactly the pinned `<repo>@<revision>`. This check is needed because `kev.serve` accepts requests for `kev-latest` regardless of which checkpoint it loaded.

The torch, transformers, peft, and flash-linear-attention versions are pinned to those Kev tests. The remaining dependencies are resolved when the environment is built.

`clm/` requires a hub commit for the heads and `encoder_revision` for Qwen3-8B. The `contrastive-lm` package downloads the heads from `main`, so the provider downloads the pinned commit itself. The `clm` extra pins `contrastive-lm` to exactly 0.1.0, because the package builds the text the heads score.

The `laya` and `gliner2` packages construct model inputs, and `laya` also selects the checkpoint. Those choices are part of the protocol, so the `laya` extra pins `laya` to 0.3.22 and the `gliner` extra pins `gliner2` to 2.0.0. Upgrading either changes the protocol. The `orbitals` SDK builds ScopeGuard's prompt for `orbitals/` configurations, so the `orbitals` extra pins it to 0.6.1 for the same reason. The Laya provider checks each question with the package's sequence builder before inference. Its required `max_len` argument is recorded in every log. Clef and d1 package their input encoding with the weights, so the hub commit pins both. All four providers batch samples by arrival, which can cause small differences in probabilities between runs (see [Laya](protocols.md#laya)).

`gliner2` 2.0.0 declares transformers older than 5. It works the same on transformers 5, so the
project overrides that requirement and runs it on the 5.15.0 that `uv.lock` pins for every other
runtime.

### Hosted models: a snapshot where one exists

Hosted APIs identify models by name. Except for TypeSafe, their configurations omit the `revision` model argument and logs record `revision: null`. When a provider publishes an immutable snapshot, its model ID is the pin and appears in the configuration's model string. OpenRouter configurations for [hosted general-purpose models](protocols.md#hosted-general-purpose-models) also pin the upstream service.

A moving alias can point to a different model on a later run. Configurations using one are runnable, but their results cannot be compared with later runs as if the model were unchanged. `tests/integration/test_configs.py` lists these exceptions explicitly in `UNPINNED_CONFIGS`:

- `openai/gpt-5.6-luna` and `openai/gpt-5.6-sol`. The OpenRouter trees of the same names pin a
  snapshot.
- `google/gemini-3.1-pro-preview`, a preview model, which Google does not document as one fixed
  build. The OpenRouter tree of the same name pins a snapshot.

Their dataset revisions and prompt hashes are pinned like any other configuration's.

A snapshot pins the model, but hosted inference can still vary. Protocols send a seed only on a [route](protocols.md#routes) that honours it. The seed requests repeatable sampling without guaranteeing identical outputs.

### TypeSafe: a checked version

TypeSafe model strings name an exact version, such as `typesafe/jev-1.13.0`. The moving `jev-latest` and `jev-preview` aliases are excluded. Every response reports the serving version, and the `typesafe` provider fails the run if it differs from the model string. That check is why `typesafe/jev-1.13` counts as pinned. Configurations also set the `revision` model argument to the same version, which records it in the log and metrics file. The provider rejects a different `revision`.

The pin fixes which model answers, and Jev can still answer differently on a second run. In this
package's runs, two runs of the same configuration differed by up to about 0.005 balanced
accuracy on benchmarks of 850 to 2,900 samples and about 0.01 on CoSApien's 200. Treat a smaller
difference between two Jev results as noise.

## What a run records

### Inspect log

Inspect's [`.eval` log](https://inspect.aisi.org.uk/eval-logs.html) holds every sample and score,
and the run's task metadata: the protocol, its options, the dataset revision, and the hash of
every prompt the run resolved, profile prompts included. It is the only record that can
reproduce, re-score or audit a result, so archive results in that format.
[Prompt resources](protocols.md#prompt-resources) describes how the prompt hashes are checked.

### Metrics file

Every unsafe-eval evaluation that reaches a final Inspect status, whether it succeeded, failed,
or was cancelled, also writes a metrics file of aggregate results next to its log. An Inspect
[hook](https://inspect.aisi.org.uk/extensions-hooks.html) the package registers writes it, so
`inspect eval` and `eval()` write one too. For a log such as:

```text
runs/vllm/shieldstral-1.0-3b/2026-08-17T12-00-00_simple-safety-tests_example.eval
```

the metrics file is:

```text
runs/vllm/shieldstral-1.0-3b/metrics-2026-08-17T12-00-00_simple-safety-tests_example.json
```

A run with several tasks or models writes one file per evaluation, and
`runs/**/metrics-*.json` collects them all. The schema is versioned:

```json
{
  "schema_version": 5,
  "run_id": "...",
  "eval_id": "...",
  "status": "success",
  "created_at": "...",
  "completed_at": "...",
  "task": {
    "name": "simple_safety_tests",
    "registry_name": "unsafe_eval/simple_safety_tests",
    "version": 2
  },
  "model": {
    "name": "vllm/example/model",
    "revision": "..."
  },
  "metadata": {
    "benchmark_id": "simple-safety-tests",
    "protocol": "shieldstral-1.0-3b",
    "dataset_revision": "..."
  },
  "samples": {
    "total": 100,
    "completed": 100
  },
  "selection": {
    "limit": null,
    "sample_id": null
  },
  "max_connections": 4,
  "provenance": {
    "unsafe_eval_version": "1.0.0",
    "inspect_ai_version": "0.3.273",
    "git_revision": "...",
    "git_dirty": false,
    "uv_lock_sha256": "..."
  },
  "inspect_args": ["--max-connections", "8"],
  "scores": [
    {
      "name": "label",
      "scorer": "unsafe_eval/shieldstral_safety",
      "reducer": null,
      "scored_samples": 100,
      "unscored_samples": 0,
      "metrics": {
        "unsafe_eval/unsafe_recall": 0.95
      }
    }
  ],
  "usage": {
    "input_tokens": 18515,
    "input_tokens_cache_write": null,
    "input_tokens_cache_read": null,
    "output_tokens": 1117,
    "reasoning_tokens": null,
    "total_cost": null
  },
  "latency": {
    "seconds_per_sample": 0.027,
    "inference_seconds": 2.7,
    "timed_samples": 100,
    "median_seconds": 0.025,
    "p95_seconds": 0.041
  }
}
```

`selection` records `--limit` and `--sample-id`; both are `null` for a full-dataset run. `max_connections` records concurrency, or `null` when Inspect uses its adaptive default. `usage` and `latency` include only the evaluated model, excluding judge-model calls. `input_tokens` counts only uncached input: tokens written to or read from a provider's prompt cache are in `input_tokens_cache_write` and `input_tokens_cache_read`, which are `null` when the provider reports no caching. They are billed at their own rates, so a cost estimate from `input_tokens` alone is too low. `total_cost` records the provider's reported cost and is `null` when none is reported, as with a local model.

`provenance` records the `unsafe-eval` and `inspect-ai` versions. For unsafe-eval running from a git clone, it also records the commit, whether the working tree had uncommitted changes, and the `uv.lock` hash. Those git and lock fields are `null` for a pip install. `inspect_args` records the options forwarded by `unsafe-eval run` to `inspect eval`, including flags such as `--limit` and `-M` that can affect results. It is `null` when the run starts directly with `inspect eval`.

`latency.seconds_per_sample` divides `inference_seconds` by the number of samples that finished without an error. `inference_seconds` spans the evaluated model's first request through its last response. Concurrent requests share that interval, so this measures inference time per sample across the dataset. For individual request times, use `median_seconds` and `p95_seconds` (nearest rank), computed from the timed samples. [Measuring latency](cli.md#measuring-latency) explains what each provider includes.

All `latency` values exclude model loading and local vLLM server startup. `latency` is `null` when no sample was timed; missing or non-finite values are also written as `null`.

`metadata` contains the full task metadata, including protocol options and prompt hashes. The metrics file omits samples, prompts, completions, and credentials, making it suitable for dashboards and experiment trackers. Keep the log to reproduce or audit the run. There is one exception to the credential exclusion: `inspect_args` is copied verbatim, so flags such as `-M api_key=...` can include secrets or local paths.

## Versioning

A change to a protocol's prompts, thresholds, taxonomy or scoring gets a new protocol identifier,
and the old identifier stays registered. `scope-guard-2601` and `scope-guard-2609` are both
registered for that reason: they differ in prompts and taxonomy, and a log that names
`scope-guard-2601` keeps the meaning it had when it was written.

A change to a benchmark's filtering or labels raises its Inspect task version, which the log and
the metrics file record as `task.version`. A benchmark's inputs are fixed by its pinned dataset
revision and the protocol that runs it.

The package version follows semantic versioning for the Python API and the `unsafe-eval`
command.
