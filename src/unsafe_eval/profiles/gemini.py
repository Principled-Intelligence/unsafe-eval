"""The prompt Gemini classifies content with, and the solvers that apply it.

Gemini is sent a system message and a user template from `profiles/prompts/gemini/`,
with the benchmark's policy from `protocols/prompts/<benchmark>/gemini/` filled into the
template. It answers in the JSON schema built from `_FIELD_DEFINITIONS`, and
`scorers/gemini_safety.py` parses the answer.

The guard models in this package reproduce the prompt their model was trained on.
Gemini has no such prompt, so this one is the package's own design, and editing it
changes what a Gemini result measures. The files belong to Gemini alone, so an edit
leaves the other families untouched. `profiles/gpt.py` explains the rules these prompts
follow and why each one matters; read it before editing.
"""

from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import Field

from unsafe_eval._refusal import generate_or_refusal
from unsafe_eval._resources import content_sha256, load_text_resource
from unsafe_eval.profiles._structured_label import (
    StructuredLabelFormat,
    UserSafety,
    apply_policy_prompt,
    sample_policy_instruction,
)

GEMINI_MODEL_RELEASE = "gemini"

GEMINI_SYSTEM_MESSAGE = load_text_resource(
    "profiles", "prompts", GEMINI_MODEL_RELEASE, "system.txt"
)
_USER_TEMPLATE = load_text_resource("profiles", "prompts", GEMINI_MODEL_RELEASE, "user.txt")

GEMINI_PROMPT_RESOURCE_HASHES = {
    "profile/system.txt": content_sha256(GEMINI_SYSTEM_MESSAGE),
    "profile/user.txt": content_sha256(_USER_TEMPLATE),
}

# `reasoning` comes before `user_safety`, so Gemini explains itself before it commits to a
# label. The descriptions are sent inside the schema, which makes them prompt text too.
_FIELD_DEFINITIONS: dict[str, tuple[Any, Any]] = {
    "reasoning": (
        str,
        Field(description="A short English explanation naming what decided the label."),
    ),
    "user_safety": (
        UserSafety,
        Field(description="The classification assigned to the request."),
    ),
}

GEMINI_FORMAT = StructuredLabelFormat(
    model_name="GeminiResponseModel", field_definitions=_FIELD_DEFINITIONS
)


@solver
def gemini_prompt(instruction: str) -> Solver:
    """Prepend Gemini's system message and wrap each prompt with a benchmark's policy."""
    if not instruction:
        raise ValueError("gemini_prompt requires a non-empty benchmark policy")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return apply_policy_prompt(
            state,
            system_message=GEMINI_SYSTEM_MESSAGE,
            user_template=_USER_TEMPLATE,
            instruction=instruction,
        )

    return solve


@solver
def gemini_metadata_policy_prompt(
    instruction_template: str,
    policy_field: str = "safety_policy",
) -> Solver:
    """Like `gemini_prompt`, for benchmarks whose policy changes from sample to sample.

    Each sample's `policy_field` metadata is rendered into `instruction_template`, and the
    result is the policy that sample is classified under.
    """
    if not instruction_template:
        raise ValueError("gemini_metadata_policy_prompt requires a non-empty template")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instruction = sample_policy_instruction(
            state, instruction_template, policy_field=policy_field, family="Gemini"
        )
        return apply_policy_prompt(
            state,
            system_message=GEMINI_SYSTEM_MESSAGE,
            user_template=_USER_TEMPLATE,
            instruction=instruction,
        )

    return solve


@solver
def gemini_generate_or_refusal() -> Solver:
    """Generate, and record a prompt Google blocks as a provider refusal.

    Google blocks a prompt inside a successful response rather than with an error, so a
    plain `generate()` would return an empty completion that scores invalid. A refusal
    is scored as an unsafe prediction instead. `unsafe_eval._refusal` recognises the
    block on both routes: Inspect reports it as a `content_filter` stop on `google`, and
    OpenRouter relays it as a failed choice.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate_or_refusal(state, generate)

    return solve
