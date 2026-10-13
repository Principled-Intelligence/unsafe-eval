"""Shieldstral's documented policy-adaptive classification protocol."""

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    prompt_template,
    solver,
    system_message,
)

from unsafe_eval._resources import content_sha256, load_text_resource, render_template
from unsafe_eval.profiles._structured_label import sample_policy_instruction

SHIELDSTRAL_MODEL_RELEASE = "shieldstral-1.0-3b"

SHIELDSTRAL_SYSTEM_MESSAGE = load_text_resource(
    "profiles", "prompts", SHIELDSTRAL_MODEL_RELEASE, "system.txt"
)
_SHIELDSTRAL_USER_TEMPLATE = load_text_resource(
    "profiles", "prompts", SHIELDSTRAL_MODEL_RELEASE, "user.txt"
)
SHIELDSTRAL_PROMPT_RESOURCE_HASHES = {
    "profile/system.txt": content_sha256(SHIELDSTRAL_SYSTEM_MESSAGE),
    "profile/user.txt": content_sha256(_SHIELDSTRAL_USER_TEMPLATE),
}


def shieldstral_user_message(
    prompt: str,
    *,
    instruction: str,
    query: str,
) -> str:
    """Build the adaptive user message used by Shieldstral."""
    return render_template(
        _SHIELDSTRAL_USER_TEMPLATE,
        instruction=instruction,
        query=query,
        prompt=prompt,
    )


@solver
def shieldstral_prompt(
    instruction: str,
    query: str,
) -> Solver:
    """Apply Shieldstral's documented system and user prompt format."""
    template = shieldstral_user_message(
        "{prompt}",
        instruction=instruction,
        query=query,
    )
    return chain(
        system_message(SHIELDSTRAL_SYSTEM_MESSAGE),
        prompt_template(template),
    )


@solver
def shieldstral_metadata_policy_prompt(
    instruction_template: str,
    query: str,
    policy_field: str = "safety_policy",
) -> Solver:
    """Apply Shieldstral's format using a policy stored in sample metadata."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        instruction = sample_policy_instruction(
            state, instruction_template, policy_field=policy_field, family="Shieldstral"
        )
        state.user_prompt.text = shieldstral_user_message(
            state.user_prompt.text,
            instruction=instruction,
            query=query,
        )
        state.messages.insert(0, ChatMessageSystem(content=SHIELDSTRAL_SYSTEM_MESSAGE))
        return state

    return solve
