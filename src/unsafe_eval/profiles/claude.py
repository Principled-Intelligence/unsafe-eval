"""The prompt Claude classifies content with, and the solvers that apply it.

Claude is sent a system message and a user template from `profiles/prompts/claude/`,
with the benchmark's policy from `protocols/prompts/<benchmark>/claude/` filled into the
template. It answers in the JSON schema built from `_FIELD_DEFINITIONS`, and
`scorers/claude_safety.py` parses the answer.

The guard models in this package reproduce the prompt their model was trained on.
Claude has no such prompt, so this one is the package's own design, and editing it
changes what a Claude result measures. The files belong to Claude alone, so an edit
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

CLAUDE_MODEL_RELEASE = "claude"

CLAUDE_SYSTEM_MESSAGE = load_text_resource(
    "profiles", "prompts", CLAUDE_MODEL_RELEASE, "system.txt"
)
_USER_TEMPLATE = load_text_resource("profiles", "prompts", CLAUDE_MODEL_RELEASE, "user.txt")

CLAUDE_PROMPT_RESOURCE_HASHES = {
    "profile/system.txt": content_sha256(CLAUDE_SYSTEM_MESSAGE),
    "profile/user.txt": content_sha256(_USER_TEMPLATE),
}

# `reasoning` comes before `user_safety`, so Claude explains itself before it commits to a
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

CLAUDE_FORMAT = StructuredLabelFormat(
    model_name="ClaudeResponseModel", field_definitions=_FIELD_DEFINITIONS
)


@solver
def claude_prompt(instruction: str) -> Solver:
    """Prepend Claude's system message and wrap each prompt with a benchmark's policy."""
    if not instruction:
        raise ValueError("claude_prompt requires a non-empty benchmark policy")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return apply_policy_prompt(
            state,
            system_message=CLAUDE_SYSTEM_MESSAGE,
            user_template=_USER_TEMPLATE,
            instruction=instruction,
        )

    return solve


@solver
def claude_metadata_policy_prompt(
    instruction_template: str,
    policy_field: str = "safety_policy",
) -> Solver:
    """Like `claude_prompt`, for benchmarks whose policy changes from sample to sample.

    Each sample's `policy_field` metadata is rendered into `instruction_template`, and the
    result is the policy that sample is classified under.
    """
    if not instruction_template:
        raise ValueError("claude_metadata_policy_prompt requires a non-empty template")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instruction = sample_policy_instruction(
            state, instruction_template, policy_field=policy_field, family="Claude"
        )
        return apply_policy_prompt(
            state,
            system_message=CLAUDE_SYSTEM_MESSAGE,
            user_template=_USER_TEMPLATE,
            instruction=instruction,
        )

    return solve


@solver
def claude_generate_or_refusal() -> Solver:
    """Generate, recording a provider refusal instead of failing the sample.

    Anthropic declines inside a successful response, which Inspect reports as a
    `content_filter` stop. `unsafe_eval._refusal.generate_or_refusal` records that
    stop, and an error-shaped refusal from a gateway, as a refusal.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate_or_refusal(state, generate)

    return solve
