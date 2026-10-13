"""PolyGuard-Qwen's published prompt-classification format."""

from inspect_ai.solver import Solver, chain, prompt_template, solver, system_message

from unsafe_eval._resources import content_sha256, load_text_resource, render_template

POLYGUARD_MODEL_RELEASE = "polyguard-qwen"

POLYGUARD_SYSTEM_MESSAGE = load_text_resource(
    "profiles", "prompts", POLYGUARD_MODEL_RELEASE, "system.txt"
)
_POLYGUARD_USER_TEMPLATE = load_text_resource(
    "profiles", "prompts", POLYGUARD_MODEL_RELEASE, "user.txt"
)
POLYGUARD_PROMPT_RESOURCE_HASHES = {
    "profile/system.txt": content_sha256(POLYGUARD_SYSTEM_MESSAGE),
    "profile/user.txt": content_sha256(_POLYGUARD_USER_TEMPLATE),
}


def polyguard_user_message(prompt: str) -> str:
    """Build PolyGuard's prompt-only conversation input."""
    return render_template(_POLYGUARD_USER_TEMPLATE, prompt=prompt)


@solver
def polyguard_prompt() -> Solver:
    """Apply PolyGuard's published system and prompt-only user format."""
    return chain(
        system_message(POLYGUARD_SYSTEM_MESSAGE),
        prompt_template(polyguard_user_message("{prompt}")),
    )
