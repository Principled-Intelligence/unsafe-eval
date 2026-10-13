# Add a benchmark

To register a dataset, you need an adapter, a task export, and a run configuration. This example registers [SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) again as `simple-safety-tests-copy`, then evaluates one sample with Shieldstral. The built-in `simple_safety_tests` task stays unchanged. We'll run Shieldstral on vLLM, which requires an NVIDIA GPU.

## 1. Create the benchmark module

The benchmark module loads a pinned dataset revision, turns each record into an Inspect sample, and [registers](extending.md#extension-api) the benchmark's labels, primary metric, metric grouping, and prompt resources. Create `src/unsafe_eval/benchmarks/simple_safety_tests_copy.py`:

```python
"""Tutorial copy of the SimpleSafetyTests harmful-prompt suite."""

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from unsafe_eval.protocols import (
    BenchmarkProtocolSpec,
    PromptResource,
    ProtocolName,
    register_benchmark,
    register_prompt_resources,
)
from unsafe_eval.task import build_prompt_safety_task

BENCHMARK = "simple-safety-tests-copy"
DATASET_REPOSITORY = "Bertievidgen/SimpleSafetyTests"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_REVISION = "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5"
SHIELDSTRAL_PROTOCOL = "shieldstral-1.0-3b"

# Every target is unsafe, so detection is measured with unsafe recall.
register_benchmark(
    BENCHMARK,
    BenchmarkProtocolSpec(
        mixed_labels=False,
        primary_metric="unsafe_recall",
        unsafe_group_by=("harm_area", "category"),
    ),
)

# This alias has the same semantics as the built-in benchmark, so it can reuse the existing
# Shieldstral adaptation text. A genuinely new benchmark would normally provide its own text.
register_prompt_resources(
    BENCHMARK,
    SHIELDSTRAL_PROTOCOL,
    (
        PromptResource(
            filename="instruction.txt",
            sha256="2b0637733291f37bc93dbb7054e6ee2ecf2af0b4b0e2481d85b87170e5954f91",
            path=(
                "protocols",
                "prompts",
                "simple-safety-tests",
                SHIELDSTRAL_PROTOCOL,
                "instruction.txt",
            ),
        ),
        PromptResource(
            filename="query.txt",
            sha256="b8df2fc50b37e7522c8a2386c829a41f0167419cf215fb696b4f88ea59e78367",
            path=(
                "protocols",
                "prompts",
                "simple-safety-tests",
                SHIELDSTRAL_PROTOCOL,
                "query.txt",
            ),
        ),
    ),
)


def record_to_sample(record: Mapping[str, Any]) -> Sample:
    """Convert one upstream row into an unsafe-target Inspect sample."""
    required = ("id", "prompt", "harm_area", "category", "counter")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"SimpleSafetyTests copy is missing fields: {missing!r}")
    prompt = record["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("SimpleSafetyTests copy prompt must be a non-empty string")
    return Sample(
        id=str(record["id"]),
        input=prompt,
        target="yes",
        metadata={
            "harm_area": record["harm_area"],
            "category": record["category"],
            "counter": record["counter"],
        },
    )


def load_simple_safety_tests_copy(
    dataset_revision: str = DATASET_REVISION,
) -> Dataset:
    """Load the real test split from an immutable Hugging Face revision."""
    return hf_dataset(
        path=DATASET_REPOSITORY,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=dataset_revision,
        sample_fields=record_to_sample,
        trust=False,
    )


@task
def simple_safety_tests_copy(
    protocol: ProtocolName,
    dataset_revision: str = DATASET_REVISION,
    protocol_options: dict[str, object] | None = None,
) -> Task:
    """Evaluate unsafe-prompt detection on the tutorial benchmark copy."""
    return build_prompt_safety_task(
        dataset=load_simple_safety_tests_copy(dataset_revision),
        protocol=protocol,
        benchmark=BENCHMARK,
        protocol_options=protocol_options,
        version=1,
        metadata={
            "benchmark": "SimpleSafetyTests tutorial copy",
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": dataset_revision,
            "dataset_split": DATASET_SPLIT,
            "evaluation_semantics": "unsafe_prompt_detection",
            "all_samples_are_unsafe": True,
        },
    )
```

Keep `harm_area` and `category` in sample metadata so `unsafe_group_by` can group unsafe recall by those fields.

## 2. Register the Inspect task

Inspect finds `unsafe_eval/` tasks through the package's entry point, which exports every task
listed in `unsafe_eval.benchmarks.__all__`. Add this import to
`src/unsafe_eval/benchmarks/__init__.py`:

```python
from unsafe_eval.benchmarks.simple_safety_tests_copy import simple_safety_tests_copy
```

Append `"simple_safety_tests_copy"` to `__all__`. Run `inspect list tasks` from the repository root to check that Inspect finds the `@task` function in your source files. Step 4 checks that the task is also exported through the package entry point.

```bash
uv run inspect list tasks | rg simple_safety_tests_copy
```

The output should include:

```text
src/unsafe_eval/benchmarks/simple_safety_tests_copy.py@simple_safety_tests_copy
```

## 3. Create a runnable configuration

A [run configuration](configuration.md#writing-a-configuration) pairs a task with a protocol and pins the dataset and model revisions. Use the revisions from `vllm/shieldstral-1.0-3b/simple-safety-tests` for this example. Save the following as `tutorial-simple-safety-tests-copy.yaml` in the repository root, outside `src/unsafe_eval/configs/`:

```yaml
task:
  task: unsafe_eval/simple_safety_tests_copy
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

## 4. Run the new benchmark

Running one sample checks dataset loading, prompt adaptation, generation, strict scoring, metrics, and logging together. Install the vLLM runtime if you haven't already:

```bash
uv sync --extra vllm
```

Run one sample through the new task:

```bash
uv run inspect eval \
  --run-config tutorial-simple-safety-tests-copy.yaml \
  --log-dir runs/vllm/shieldstral-1.0-3b \
  --limit 1
```

The first run downloads the pinned dataset and model. Inspect writes the `.eval` log under
`runs/vllm/shieldstral-1.0-3b/`. To list it:

```bash
uv run inspect log list --log-dir runs
```

## Making it permanent

A benchmark that ships with the package also needs:

- a [packaged configuration](configuration.md#packaged-configurations) for each protocol that
  supports it;
- its dataset revision in `BENCHMARK_REVISIONS` in `tests/integration/test_configs.py`;
- its source, filtering, and attribution in `docs/benchmarks.md` and `docs/reproducibility.md`.

## Optional cleanup

To remove the copy, delete its module and the YAML file, and remove its import and `__all__` entry
from `benchmarks/__init__.py`.
