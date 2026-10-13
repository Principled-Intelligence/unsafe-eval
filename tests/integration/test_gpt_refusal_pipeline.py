"""End-to-end behaviour when a hosted provider refuses a prompt.

The failure this guards against is not hypothetical. In the first GPT-5.6 Luna
sweep a single refused prompt aborted four of eight benchmarks, because Inspect
fails a task on any sample error by default; and when those samples were merely
dropped, unsafe recall rose for free, since every refusal had landed on an unsafe
target.
"""

from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ModelOutput, get_model

from tests.integration._support import configure_test_environment, task_with_dataset

BIO_REFUSAL = (
    "Error 403 - This content was flagged for possible biological risk. If this "
    "seems wrong, try rephrasing your request."
)
OUT_OF_CREDITS = (
    "Error code: 402 - {'error': {'message': 'This request would exceed your "
    "available credits given your current inflight requests'}}"
)

_GROUPS = {"functional_category": "standard", "semantic_category": "chemical_biological"}

# HarmBench Text Test is all-unsafe, which is exactly where refusals concentrate:
# every refusal in the first live sweep landed on an unsafe target.
SAMPLES = [
    Sample(input="refused-example", target="yes", metadata=dict(_GROUPS)),
    Sample(input="unsafe-example", target="yes", metadata=dict(_GROUPS)),
]


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str) -> Any:
    task = task_with_dataset(
        module_name="harmbench_prompt_safety",
        loader_name="load_harmbench_prompt_safety",
        task_name="harmbench_prompt_safety",
        samples=SAMPLES,
        protocol="gpt",
        protocol_options={"route": "openrouter"},
        monkeypatch=monkeypatch,
    )

    def model_output(
        messages: list[ChatMessage],
        _tools: list[Any],
        _tool_choice: Any,
        _config: Any,
    ) -> ModelOutput:
        prompt = messages[-1].text
        if "refused-example" in prompt:
            raise RuntimeError(error)
        label = "unsafe" if "unsafe-example" in prompt else "safe"
        return ModelOutput.from_content(
            model="mockllm/model",
            content=f'{{"reasoning":"mock","user_safety":"{label}"}}',
        )

    return eval(
        task,
        model=get_model("mockllm/model", custom_outputs=model_output),
        log_dir=str(tmp_path),
        display="none",
        ctl_server=False,
    )[0]


def test_a_refusal_is_scored_as_unsafe_and_reported_as_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_test_environment(tmp_path, monkeypatch, "gpt-refusal")

    log = _run(tmp_path, monkeypatch, BIO_REFUSAL)

    # The whole point: one refused prompt no longer costs the benchmark.
    assert log.status == "success", log.error
    assert log.results is not None
    assert log.results.completed_samples == len(SAMPLES)
    metrics = {score.name: score for score in log.results.scores}["label"].metrics
    # Two unsafe targets, one refused: the refusal is an unsafe prediction, and it
    # is still reported as a refusal.
    assert metrics["unsafe_eval/unsafe_recall"].value == pytest.approx(1.0)
    assert metrics["unsafe_eval/refusal_rate"].value == pytest.approx(0.5)

    verdicts = [
        (sample.scores or {})["gpt_safety"].metadata["predicted_verdict"]
        for sample in (log.samples or [])
    ]
    assert sorted(verdicts) == ["refused", "yes"]


def test_an_infrastructure_failure_still_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_test_environment(tmp_path, monkeypatch, "gpt-credits")

    log = _run(tmp_path, monkeypatch, OUT_OF_CREDITS)

    # An exhausted balance must not be absorbed into the score. A visibly failed
    # run is the correct outcome; a quietly depressed one is not.
    assert log.status == "error"
