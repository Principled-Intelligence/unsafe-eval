# Benchmarks

All benchmarks evaluate prompt classification. The model receives a user prompt and, for policy-conditioned benchmarks, a policy. It decides whether the prompt is unsafe. Each adapter loads a [pinned revision](reproducibility.md#dataset-revisions), validates the schema and labels, and records the source of each sample. Responses stored in a dataset are never passed to the evaluated model. [Add a benchmark](adding-benchmark.md) shows how to build an adapter by registering SimpleSafetyTests under another name.

The ID names a benchmark in configuration addresses and in `--benchmark`, and the Inspect task
is registered under the `unsafe_eval/` prefix. SimpleSafetyTests and HarmBench Text Test are
unsafe-only and report [different metrics](metrics.md) from the six mixed-label benchmarks,
where unsafe F1 depends on the [unsafe prevalence](metrics.md#why-unsafe-f1).

| Benchmark | ID | Inspect task | Prompts | Unsafe / safe | Unsafe prevalence | License |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [SimpleSafetyTests](#simplesafetytests) | `simple-safety-tests` | `simple_safety_tests` | 100 | 100 / 0 | 100% | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [HarmBench Text Test](#harmbench-text-test) | `harmbench-text-test` | `harmbench_prompt_safety` | 320 | 320 / 0 | 100% | [MIT](https://github.com/centerforaisafety/HarmBench) |
| [Aegis 2.0](#aegis-20) | `aegis-2.0` | `aegis_prompt_safety` | 1,928 | 1,039 / 889 | 53.9% | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [OpenAI Moderation](#openai-moderation) | `openai-moderation` | `openai_moderation_prompt_safety` | 859 | 522 / 337 | 60.8% | [MIT](https://github.com/openai/moderation-api-release) |
| [ToxicChat 0124](#toxicchat-0124) | `toxic-chat-0124` | `toxic_chat_prompt_safety` | 2,853 | 362 / 2,491 | 12.7% | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| [WildGuardMix / WildGuardTest](#wildguardmix--wildguardtest) | `wildguardmix-test` | `wildguard_mix_prompt_safety` | 1,699 | 754 / 945 | 44.4% | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| [CoSApien](#cosapien) | `cosapien` | `cosapien_prompt_safety` | 200 | 84 / 116 | 42.0% | [CDLA-Permissive 2.0](https://cdla.dev/permissive-2-0/) |
| [DynaGuardrail](#dynaguardrail) | `dynaguardrail` | `dynaguardrail_prompt_safety` | 2,362 | 1,129 / 1,233 | 47.8% | [MIT](https://huggingface.co/datasets/dynamofl/DynaGuardrail) |

## SimpleSafetyTests

Each of the 100 prompts carries a harm area and a request category.

Created by Bertie Vidgen, Nino Scherrer, Hannah Rose Kirk, Rebecca Qian, Anand Kannappan, Scott
A. Hale, and Paul Röttger.

## HarmBench Text Test

These are the original 320 text behaviors: 159 standard, 81 contextual, and 80 copyright.
Standard and copyright rows use the behavior as the prompt. Contextual rows are built the way
HarmBench's DirectRequest baseline builds them: the context, a `---` separator, then the
behavior.

Only the prompts are used. The adapter does not run HarmBench's own pipeline, which generates
red-team attacks, collects a target model's completions, and classifies them.

Created by Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham
Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks.

## Aegis 2.0

Each prompt takes its human `prompt_label` as the target, with `Safe` and `Needs Caution` both
mapped to safe. Responses and response labels are dropped: the model never sees them and sample
metadata does not keep them.

The pinned test split contains 1,964 records. Excluding the 36 marked `REDACTED` leaves 1,928 prompts. Restoring those records requires a separately distributed reconstruction dataset.

Aegis 2.0, also released as Nemotron Content Safety Dataset V2, was created by Shaona Ghosh,
Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan
Varghese, and Christopher Parisien.

## OpenAI Moderation

The official test release contains 1,680 records with up to eight category labels each. Any positive category makes a prompt unsafe. A prompt is safe only when all eight categories are present and negative. The adapter excludes 821 rows that have missing categories and no positive label, leaving 859 prompts. Sample metadata keeps the known category labels.

Created by Todor Markov, Chong Zhang, Sandhini Agarwal, Tyna Eloundou, Teddy Lee, Steven Adler,
Angela Jiang, and Lilian Weng.

## ToxicChat 0124

Only the human-annotated records of the pinned `toxicchat0124` test split are kept: 2,853 of
its 5,083 prompts. The binary `toxicity` label is the target, `1` for unsafe and `0` for safe.
Model responses and OpenAI moderation results are dropped. Sample metadata keeps
`human_annotation` and `jailbreaking`; all 91 jailbreaking examples in the split are labeled
toxic.

Created by Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, and Jingbo
Shang.

## WildGuardMix / WildGuardTest

The adapter uses the official `wildguardtest` split and its prompt-harmfulness label. It excludes rows where fewer than two of the three annotators agree on `prompt_harm_label`, leaving 1,699 of 1,725 prompts. Samples keep the prompt type (adversarial or vanilla), subcategory, and label agreement. Response text, response-harm labels, and refusal labels are dropped.

WildGuardTest is contrastive, so the package's WildGuardMix prompt adaptations judge the help a
prompt asks for. They allow benign discussion of sensitive topics, safe contexts, and benign
requests phrased adversarially.

Created by Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan
Lambert, Yejin Choi, and Nouha Dziri. Access is subject to AI2's Responsible Use Guidelines.

## CoSApien

CoSApien judges each prompt against its own policy. The adapter combines all five scenario splits. Each split contains application-specific policies, and none is held out. A sample's `scenario` supplies the trusted safety policy, and its `prompt` is the content to classify.

`safe` and `allowed` map to safe, `disallowed` and `partial` to unsafe. A `partial` prompt is
unsafe because at least one substantive part of it is prohibited. The label says nothing about
whether an assistant should refuse the whole request or fulfill only the permitted parts. The
dataset card documents only allowed, disallowed, and partial, but the pinned files also hold 52
records labeled `safe`, so the task validates and records all four.

All 200 records are evaluated. Partial prompts count toward unsafe recall, and sensitive content
the policy explicitly allows counts toward safe recall.

The paper's CoSA-Score rates generated responses for helpfulness and safety. This task
classifies prompts, so its results are not CoSA-Scores.

Created by Jingyu Zhang, Ahmed Elgohary, Ahmed Magooda, Daniel Khashabi, and Benjamin Van
Durme.

## DynaGuardrail

DynaGuardrail also judges each prompt against a policy. Its single split is named `train`, but
this package uses it only for evaluation. It combines four domains, and each row's
`dataset_type` selects the policy it is judged against: general safety, financial advice, tax
advice, or prompt injection.

All 2,362 rows are evaluated without deduplication: 500 finance, 951 injection, 546 safety, and
365 tax. The dataset card's component summary gives 546 tax rows, but the pinned shard has 365.
Diagnostic source fields are kept. The upstream auxiliary model predictions and detailed-label
columns are not copied into samples.

The adapter reads the pinned Parquet shard directly, because the repository's legacy
`dataset_infos.json` does not load with the Hugging Face `datasets` 4.x releases this package
supports.

Released by DynamoFL with the GuardFormer work by James O'Neill, Santhosh Subramanian, Eric Lin,
Abishek Satish, and Vaikkunth Mugunthan.

## Gated dataset access

WildGuardMix and DynaGuardrail are gated on Hugging Face. Accept the terms on the
[WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) or
[DynaGuardrail](https://huggingface.co/datasets/dynamofl/DynaGuardrail) page, then log in on the
machine that runs Inspect:

```bash
uv run hf auth login
uv run hf auth whoami
```

The loader uses standard Hugging Face authentication. On CI or ephemeral runners, supply a
read-only `HF_TOKEN` through the runner's secret store. Never put a token in a run config, shell
history, source file, or Inspect log.
