import asyncio

import pytest
from inspect_ai.model import ModelOutput, StopReason
from inspect_ai.model._model_output import StopDetails
from inspect_ai.solver import TaskState

from unsafe_eval._refusal import (
    PROVIDER_REFUSAL_METADATA_KEY,
    generate_or_refusal,
    provider_refusal_reason,
    relayed_block_reason,
)


def _state() -> TaskState:
    return TaskState(model="mockllm/model", sample_id="s", epoch=1, input="Prompt", messages=[])


def _generate(
    stop_reason: StopReason, content: str = "", details: StopDetails | None = None
) -> TaskState:
    async def generate(state: TaskState, **kwargs: object) -> TaskState:
        state.output = ModelOutput.from_content(
            model="mockllm/model", content=content, stop_reason=stop_reason
        )
        state.output.choices[0].stop_details = details
        return state

    return asyncio.run(generate_or_refusal(_state(), generate))


def test_a_content_filter_stop_is_recorded_as_a_refusal() -> None:
    # Anthropic's HTTP 200 with stop_reason "refusal", as Inspect reports it.
    details = StopDetails(type="refusal", category="bio", explanation="declined")

    state = _generate("content_filter", details=details)

    assert state.metadata[PROVIDER_REFUSAL_METADATA_KEY] == (
        "stop_reason=content_filter; type=refusal; category=bio; explanation=declined"
    )


def test_a_max_tokens_stop_is_left_for_the_scorer() -> None:
    # A truncated object is an invalid output, not a refusal.
    state = _generate("max_tokens", content='{"user_safety": "sa')

    assert PROVIDER_REFUSAL_METADATA_KEY not in state.metadata


def test_a_connection_failure_still_aborts_the_sample() -> None:
    # A failure with no status or refusal signature, such as a dropped connection or
    # a timeout, must fail the sample rather than score as a refusal.
    async def generate(state: TaskState, **kwargs: object) -> TaskState:
        raise ConnectionError("connection reset")

    with pytest.raises(ConnectionError):
        asyncio.run(generate_or_refusal(_state(), generate))


@pytest.mark.parametrize(
    "error",
    [
        # An exhausted balance is the failure that motivated the distinction: it
        # arrived mid-run and, scored as a miss, would have silently lowered every
        # affected benchmark instead of failing the run.
        "Error code: 402 - {'error': {'message': 'This request would exceed your "
        "available credits given your current inflight requests'}}",
        "Error code: 429 - rate limit exceeded",
        "Error code: 500 - internal server error",
        # 403 alone is not a refusal: it also covers authentication and permission.
        "Error 403 - Forbidden: invalid API key",
        # A content-policy signature without the refusal status is not a refusal.
        "Error code: 400 - content policy documentation link",
        "",
    ],
)
def test_other_failures_are_not_refusals(error: str) -> None:
    assert provider_refusal_reason(error) is None


# What OpenAI's own API returns when its input filter rejects a prompt. It carries
# the `invalid_prompt` code at status 400, where OpenRouter serves its refusals at
# 403, so recognising only 403 would abort a run on the first refused prompt.
OPENAI_INVALID_PROMPT = (
    "Error code: 400 - {'error': {'message': \"Invalid prompt: your prompt was "
    "flagged as potentially violating our usage policy. Please try again with a "
    "different prompt.\", 'type': 'invalid_request_error', 'param': None, 'code': "
    "'invalid_prompt'}}"
)


# What OpenAI's own API returns when its biological-risk filter rejects a prompt.
OPENAI_BIO_POLICY = (
    "Error code: 400 - {'error': {'message': 'This content was flagged for possible "
    "biological risk. If this seems wrong, try rephrasing your request.', 'type': "
    "'invalid_request_error', 'param': None, 'code': 'bio_policy'}}"
)


@pytest.mark.parametrize("error", [OPENAI_INVALID_PROMPT, OPENAI_BIO_POLICY])
def test_recognises_openai_refusal_codes_at_either_status(error: str) -> None:
    assert provider_refusal_reason(error) == error
    assert provider_refusal_reason(error.replace("Error code: 400", "Error code: 403")) is not None


@pytest.mark.parametrize(
    "signature",
    ["flagged for possible", "content policy", "content_policy_violation"],
)
def test_only_openai_refusal_codes_widen_to_400(signature: str) -> None:
    # The widening is deliberately narrow. 400 also covers an over-long context, a
    # malformed schema and a bad parameter, so admitting every signature there would
    # score ordinary request errors as refusals and quietly depress the run.
    assert provider_refusal_reason(f"Error code: 400 - {signature}") is None
    assert provider_refusal_reason(f"Error code: 403 - {signature}") is not None


def test_invalid_prompt_outside_the_refusal_statuses_is_not_a_refusal() -> None:
    assert provider_refusal_reason("Error code: 429 - invalid_prompt retry guidance") is None


def test_a_block_relayed_as_a_failed_choice_is_a_refusal() -> None:
    for native in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
        response = {"choices": [{"finish_reason": "error", "native_finish_reason": native}]}

        assert relayed_block_reason(response) == (
            f"finish_reason=error; native_finish_reason={native}"
        )


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"choices": []},
        # An upstream failure that is not a block stays an invalid output.
        {"choices": [{"finish_reason": "error", "native_finish_reason": "OTHER"}]},
        # A block reason on a choice that did not fail is not this shape.
        {"choices": [{"finish_reason": "stop", "native_finish_reason": "SAFETY"}]},
    ],
)
def test_anything_else_is_not_a_relayed_block(response: dict[str, object] | None) -> None:
    assert relayed_block_reason(response) is None
