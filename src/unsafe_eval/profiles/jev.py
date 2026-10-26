"""Jev's native System One Choice question format.

Jev does not generate text. It answers a set of named questions about a `state`,
and this protocol asks exactly one Choice question whose options are `safe` and
`unsafe`. The returned option is treated as the native label, the way Qwen3Guard's or
PolyGuard's emitted label is.

Two evaluation semantics follow from the API's shape rather than from a training format:

* **The questions are a per-benchmark protocol adaptation, not a profile resource.**
  Each benchmark's gold labels follow its own taxonomy, so one shared set would be
  scored against several disagreeing definitions of unsafe at once. The questions
  live in `protocols/prompts/<benchmark>/jev/` and arrive here as an argument.
* **A per-sample policy travels in the state, never in the questions.** TypeSafe's
  `state` accepts a structured object whose field names the caller chooses, so a
  policy-conditioned benchmark sends `{policy, message}` and the questions stay
  byte-identical across samples. The message is never templated into text, so it
  needs no escaping and cannot be confused with the policy.
"""

from typing import Any

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver

# The sample metadata field the policy-conditioned benchmarks populate.
SAFETY_POLICY_FIELD = "safety_policy"

# The key the provider reads the structured state from, on the last user message.
STATE_METADATA_FIELD = "state"


def _attach(state: TaskState, questions: str, payload: dict[str, Any]) -> TaskState:
    """Deliver the questions verbatim and attach the structured state.

    Inspect's `system_message` solver would render the JSON questions as a template,
    reading their braces as fields and changing the bytes the recorded SHA-256 pins.
    """
    state.messages.insert(0, ChatMessageSystem(content=questions))
    prompt = state.user_prompt
    prompt.metadata = {**(prompt.metadata or {}), STATE_METADATA_FIELD: payload}
    return state


@solver
def jev_prompt(questions: str) -> Solver:
    """Prepare the questions and message for a System One provider request.

    Insert `questions` verbatim as a system message and attach `{"message": prompt}`
    under `state` in the user message's metadata. The provider reads the questions as
    JSON; this solver checks only that the supplied string is non-empty.
    """
    if not questions:
        raise ValueError("jev_prompt requires non-empty questions")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return _attach(state, questions, {"message": state.user_prompt.text})

    return solve


@solver
def jev_policy_prompt(questions: str, policy_field: str = SAFETY_POLICY_FIELD) -> Solver:
    """Prepare a System One request with separate policy and message fields.

    Insert `questions` verbatim as a system message. Read the policy from the sample
    metadata key `policy_field` and attach `{"policy": policy, "message": prompt}`
    under `state` in the user message's metadata. Raise `ValueError` for empty questions
    or a missing, empty, or non-string policy.
    """
    if not questions:
        raise ValueError("jev_policy_prompt requires non-empty questions")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        policy = (state.metadata or {}).get(policy_field)
        if not isinstance(policy, str) or not policy:
            raise ValueError(
                f"Policy-conditioned Jev protocols require a non-empty "
                f"{policy_field!r} in sample metadata"
            )
        return _attach(
            state,
            questions,
            {"policy": policy, "message": state.user_prompt.text},
        )

    return solve
