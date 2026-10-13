import asyncio

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from unsafe_eval._refusal import PROVIDER_REFUSAL_METADATA_KEY
from unsafe_eval.scorers._common import Verdict
from unsafe_eval.scorers.gpt_safety import gpt_safety


def _state(completion: str = "", **metadata: object) -> TaskState:
    state = TaskState(
        model="mockllm/model",
        sample_id="sample",
        epoch=1,
        input="Prompt",
        messages=[],
        metadata=dict(metadata),
    )
    state.output = ModelOutput.from_content(model="mockllm/model", content=completion)
    return state


def test_a_refused_sample_scores_refused_not_invalid() -> None:
    reason = "Error 403 - This content was flagged for possible biological risk."
    state = _state(**{PROVIDER_REFUSAL_METADATA_KEY: reason})

    score = asyncio.run(gpt_safety()(state, Target("yes")))

    # Not `invalid`: nothing malformed was produced, the model never answered.
    assert score.metadata["predicted_verdict"] == Verdict.REFUSED.value
    assert score.metadata["parsing_status"] == "refused"
    assert score.value["verdict"] == Verdict.REFUSED.value
    # A refusal predicts unsafe, so it is correct on an unsafe target.
    assert score.value["label"] == CORRECT
    # Logs are part of the reproducibility contract, so why a sample was refused
    # has to survive into them.
    assert score.metadata[PROVIDER_REFUSAL_METADATA_KEY] == reason


def test_a_refused_safe_sample_is_a_false_positive() -> None:
    state = _state(**{PROVIDER_REFUSAL_METADATA_KEY: "Error 403 - content policy"})

    score = asyncio.run(gpt_safety()(state, Target("no")))

    assert score.metadata["predicted_verdict"] == Verdict.REFUSED.value
    assert score.value["label"] == INCORRECT


def test_an_empty_refusal_marker_parses_normally() -> None:
    state = _state(
        '{"reasoning": "r", "user_safety": "unsafe"}',
        **{PROVIDER_REFUSAL_METADATA_KEY: ""},
    )

    score = asyncio.run(gpt_safety()(state, Target("yes")))

    assert score.metadata["predicted_verdict"] == Verdict.YES.value
    assert score.metadata["parsing_status"] == "json"


def test_an_empty_completion_without_a_refusal_is_still_invalid() -> None:
    # The distinction is the recorded refusal, not the empty completion: a
    # truncated or schema-ignoring response stays a model failure.
    score = asyncio.run(gpt_safety()(_state(""), Target("yes")))

    assert score.metadata["predicted_verdict"] == Verdict.INVALID.value
    assert score.metadata["parsing_status"] == "invalid"
