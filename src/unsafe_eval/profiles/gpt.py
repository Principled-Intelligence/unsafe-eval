"""The prompt-safety classification format for GPT-5.6 models.

The guard-model profiles in this package each reproduce a format their model was
trained on. GPT-5.6 Sol is a general instruction-following model with no published
prompt-safety format, so the system message, user template and response schema here are
this package's own, versioned with the `gpt` protocol. Three choices in them affect what
a result means:

- The system message defines no taxonomy. It sets up the task and leaves the meaning of
  `unsafe` to the policy each benchmark supplies as `{instruction}`. Each benchmark's
  labels follow its own taxonomy, and a general definition of harm here would compete
  with it. The run is scored against the benchmark's categories.
- The frame says "content" throughout. Some benchmarks classify arbitrary text rather
  than requests to an assistant, and asking whether fulfilling a declarative passage
  would cause harm has no answer.
- A JSON schema constrains the label at decode time. Sol supports structured outputs,
  so the format is enforced by the provider instead of being requested in prose. The
  scorer still validates strictly, because the provider's guided decoding can fail, and
  a completion that breaks the schema stays invalid.

The per-benchmark policies in `protocols/prompts/<benchmark>/gpt/` name no benchmark,
dataset, annotation process or class balance. Sol has very likely seen these public
benchmarks in training. A named policy would invite it to recall the stored labels
instead of applying the policy, and the results could not tell the two apart. The
guard-model adaptations do name their benchmarks, which is safe for a small guard model
that has not memorised them.
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

GPT_MODEL_RELEASE = "gpt"

GPT_SYSTEM_MESSAGE = load_text_resource("profiles", "prompts", GPT_MODEL_RELEASE, "system.txt")
_USER_TEMPLATE = load_text_resource("profiles", "prompts", GPT_MODEL_RELEASE, "user.txt")

GPT_PROMPT_RESOURCE_HASHES = {
    "profile/system.txt": content_sha256(GPT_SYSTEM_MESSAGE),
    "profile/user.txt": content_sha256(_USER_TEMPLATE),
}

# `reasoning` comes before `user_safety`, so Sol explains itself before it commits to a
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

GPT_FORMAT = StructuredLabelFormat(
    model_name="GptResponseModel", field_definitions=_FIELD_DEFINITIONS
)


@solver
def gpt_prompt(instruction: str) -> Solver:
    """Prepend the GPT system message and wrap each prompt with a benchmark's policy."""
    if not instruction:
        raise ValueError("gpt_prompt requires a non-empty benchmark policy")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return apply_policy_prompt(
            state,
            system_message=GPT_SYSTEM_MESSAGE,
            user_template=_USER_TEMPLATE,
            instruction=instruction,
        )

    return solve


@solver
def gpt_metadata_policy_prompt(
    instruction_template: str,
    policy_field: str = "safety_policy",
) -> Solver:
    """Like `gpt_prompt`, for benchmarks whose policy changes from sample to sample.

    Each sample's `policy_field` metadata is rendered into `instruction_template`, and the
    result is the policy that sample is classified under.
    """
    if not instruction_template:
        raise ValueError("gpt_metadata_policy_prompt requires a non-empty template")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instruction = sample_policy_instruction(
            state, instruction_template, policy_field=policy_field, family="GPT-5.6 Sol"
        )
        return apply_policy_prompt(
            state,
            system_message=GPT_SYSTEM_MESSAGE,
            user_template=_USER_TEMPLATE,
            instruction=instruction,
        )

    return solve


@solver
def gpt_generate_or_refusal() -> Solver:
    """Generate, recording a provider refusal instead of failing the sample.

    `unsafe_eval._refusal.generate_or_refusal` records a refusal whether it arrives as
    an error or as a `content_filter` stop. Every other failure still aborts.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate_or_refusal(state, generate)

    return solve
