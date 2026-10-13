"""Recognition of provider-side content refusals.

A hosted provider can decline to process a prompt at all. Evaluating prompt-safety
benchmarks makes this routine rather than exceptional: the benchmarks are built from
the prompts a provider's own input filter is designed to reject, so the refusals land
on exactly the samples the evaluation most needs.

A refusal is scored as an unsafe prediction. The evaluated model never saw the
prompt, but the provider blocked it, and blocking is how the deployed system treats
an unsafe prompt. It is not an invalid output either: nothing malformed was produced.
`Verdict.REFUSED` keeps refusals apart from model verdicts and parse failures so the
refusal rates can be reported next to the scores they affect.

A refusal arrives in one of three shapes. Most gateways raise an HTTP error, which is
matched below by signature. Anthropic and Google answer with HTTP 200 and a block
reason instead, and Inspect reports that as a `content_filter` stop, often with no
text. A `content_filter` stop means the same thing on any route and is recorded the
same way. The third shape is a gateway relaying an upstream block as a failed choice:
OpenRouter returns Vertex AI's `SAFETY` block as `finish_reason: "error"` with
`native_finish_reason: "SAFETY"`. Inspect keeps only the `unknown` stop reason and an
empty completion, so the block reason is read from the raw response on the transcript's
model event. `generate_or_refusal` handles all three shapes for every hosted family.

Recognition is deliberately narrow. An error counts as a refusal only when it carries
a known content-policy signature at a status that signature is served with: 403 for
the gateway signatures, 400 or 403 for OpenAI's `invalid_prompt` and `bio_policy` codes.
Every other failure,
such as an exhausted credit balance, a rate limit, a gateway timeout or an
authentication problem, stays an error and aborts the sample. Scoring an infrastructure
failure as a miss would silently turn an outage or an unpaid bill into a lower
benchmark score. A failed run is visible; a quietly lower score is not.
"""

import re
from collections.abc import Mapping
from typing import Any

from inspect_ai.event import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState

PROVIDER_REFUSAL_METADATA_KEY = "provider_refusal"

# The stop reason Inspect reports for a refusal delivered inside a response.
CONTENT_FILTER_STOP = "content_filter"

# What Inspect reports for a choice a gateway marked `finish_reason: "error"`.
UNKNOWN_STOP = "unknown"
GATEWAY_ERROR_FINISH = "error"

# Gemini's native finish reasons for a blocked response: the ones Inspect's google
# provider reports as a `content_filter` stop. Only these make a relayed error a refusal,
# so an upstream failure of any other kind stays an invalid output.
NATIVE_BLOCK_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}
)

# The status most providers serve a content refusal with. Required in addition to a
# signature so that a message merely discussing content policy cannot be read as a
# refusal, and because 403 also covers authentication and permission failures that
# must not be mistaken for a judgement about the prompt.
PROVIDER_REFUSAL_STATUS = "403"

# OpenAI's own API is the exception. It rejects a prompt its input filters dislike
# with a 400 carrying the `invalid_prompt` code, or `bio_policy` for possible
# biological risk, where OpenRouter serves the same judgement as a 403. Only these
# codes widen: 400 is also how an over-long context, a malformed schema and a bad
# parameter arrive, so admitting every signature there would score ordinary request
# errors as refusals.
OPENAI_REFUSAL_CODES: tuple[str, ...] = ("invalid_prompt", "bio_policy")
OPENAI_REFUSAL_STATUSES: tuple[str, ...] = ("400", PROVIDER_REFUSAL_STATUS)

# Substrings identifying a provider's input-side content refusal, matched
# case-insensitively, each against the statuses it is served with. Kept explicit
# rather than inferred from the status alone.
PROVIDER_CONTENT_REFUSAL_SIGNATURES: dict[str, tuple[str, ...]] = {
    "flagged for possible": (PROVIDER_REFUSAL_STATUS,),
    "content policy": (PROVIDER_REFUSAL_STATUS,),
    "content_policy_violation": (PROVIDER_REFUSAL_STATUS,),
    **dict.fromkeys(OPENAI_REFUSAL_CODES, OPENAI_REFUSAL_STATUSES),
}


def provider_refusal_reason(error: BaseException | str) -> str | None:
    """Return the refusal message when `error` is a provider content refusal.

    Returns `None` for every other failure, which leaves the caller to re-raise.
    """
    text = str(error)
    if not text:
        return None
    lowered = text.casefold()
    for signature, statuses in PROVIDER_CONTENT_REFUSAL_SIGNATURES.items():
        if signature in lowered and any(
            re.search(rf"\b{re.escape(status)}\b", text) for status in statuses
        ):
            return text
    return None


def content_filter_reason(output: ModelOutput) -> str:
    """Describe a `content_filter` stop, with the provider's refusal detail if any."""
    details = output.choices[0].stop_details if output.choices else None
    parts = [f"stop_reason={CONTENT_FILTER_STOP}"]
    if details is not None:
        if details.type:
            parts.append(f"type={details.type}")
        if details.category:
            parts.append(f"category={details.category}")
        if details.explanation:
            parts.append(f"explanation={details.explanation}")
    return "; ".join(parts)


def relayed_block_reason(response: Mapping[str, Any] | None) -> str | None:
    """Return a refusal description when a raw response relays an upstream block.

    Returns `None` unless the first choice has `finish_reason: "error"` and a
    `native_finish_reason` from `NATIVE_BLOCK_FINISH_REASONS`.
    """
    if not response:
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    choice = choices[0]
    native = choice.get("native_finish_reason")
    if choice.get("finish_reason") != GATEWAY_ERROR_FINISH:
        return None
    if native not in NATIVE_BLOCK_FINISH_REASONS:
        return None
    return f"finish_reason={GATEWAY_ERROR_FINISH}; native_finish_reason={native}"


def last_model_call_response() -> Mapping[str, Any] | None:
    """Return the raw API response of the sample's most recent model call, if recorded."""
    for event in reversed(transcript().events):
        if isinstance(event, ModelEvent):
            return event.call.response if event.call is not None else None
    return None


async def generate_or_refusal(state: TaskState, generate: Generate) -> TaskState:
    """Generate, recording any of the three shapes of provider refusal instead of failing.

    Inspect aborts a whole task on any sample error by default, so without this a
    single refused prompt discards an entire benchmark run. Only a recognised
    content refusal is absorbed: every other error is re-raised so that it still
    aborts the sample, and every other stop reason, `max_tokens` included, is left
    for the scorer, which keeps a truncated object invalid.
    """
    try:
        state = await generate(state, tool_calls="none")
    except Exception as error:  # noqa: BLE001 - re-raised unless it is a refusal
        reason = provider_refusal_reason(error)
        if reason is None:
            raise
        state.metadata[PROVIDER_REFUSAL_METADATA_KEY] = reason
        state.completed = True
        return state
    if state.output.stop_reason == CONTENT_FILTER_STOP:
        state.metadata[PROVIDER_REFUSAL_METADATA_KEY] = content_filter_reason(state.output)
    elif state.output.stop_reason == UNKNOWN_STOP:
        reason = relayed_block_reason(last_model_call_response())
        if reason is not None:
            state.metadata[PROVIDER_REFUSAL_METADATA_KEY] = reason
    return state
