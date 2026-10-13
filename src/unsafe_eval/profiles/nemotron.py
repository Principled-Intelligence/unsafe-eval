"""Nemotron 3.5 Content Safety generation with per-sample custom policies."""

from inspect_ai.solver import Generate, Solver, TaskState, solver

from unsafe_eval.profiles._structured_label import sample_policy_instruction


@solver
def nemotron_metadata_policy_generate(
    custom_policy_template: str,
    policy_field: str = "safety_policy",
    request_categories: bool = True,
) -> Solver:
    """Generate with Nemotron's custom policy populated from sample metadata."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        custom_policy = sample_policy_instruction(
            state, custom_policy_template, policy_field=policy_field, family="Nemotron"
        )
        return await generate(
            state,
            tool_calls="none",
            extra_body={
                "chat_template_kwargs": {
                    "request_categories": (
                        "/categories" if request_categories else "/no_categories"
                    ),
                    "enable_thinking": False,
                    "custom_policy": custom_policy,
                }
            },
        )

    return solve


@solver
def nemotron_inline_policy_generate(
    custom_policy_template: str,
    policy_field: str = "safety_policy",
) -> Solver:
    """Generate with the per-sample policy written into the user turn.

    The `inline_policy` path, for routes such as OpenRouter that forward no chat-template
    arguments and so cannot reach the template's `custom_policy` slot. The template's
    default taxonomy still leads the prompt and the benchmark policy follows it, so these
    scores are not comparable with the `custom_policy` path's. It exists to put a
    realistic prompt on the wire.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        custom_policy = sample_policy_instruction(
            state, custom_policy_template, policy_field=policy_field, family="Nemotron"
        )
        prompt = state.user_prompt
        prompt.text = f"{custom_policy}\n\n### Prompt to classify\n{prompt.text}"
        return await generate(state, tool_calls="none")

    return solve
