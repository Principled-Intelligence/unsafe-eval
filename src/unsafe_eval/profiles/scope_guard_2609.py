"""ScopeGuard's selectable-output-field scope-classification protocol (2609).

Differences from `scope-guard-2601`, all of them consequences of the model release
rather than choices made here:

* **Seven classes, not five.** `Predefined Answer` and `Human Oversight` were added
  to the taxonomy, so a 2601 scorer cannot parse a 2609 completion.
* **Selectable output fields.** The requested keys are named in a
  `**REQUESTED OUTPUT FIELDS**` block at the end of the user turn, and the model
  emits exactly those keys in canonical order. 2601 always emits the same shape,
  with an opt-out instruction for evidences.
* **A fully materialised system prompt.** 2601 renders `{classes_manifest}` and
  `{response_schema}` into its template; the 2609 release ships one
  selection-agnostic prompt that already contains the class list and output rules,
  so `system.txt` has no placeholders. It is the `system_prompt.txt` published with
  `scope-guard-v2-4B-q-2609`, which the weights were trained on; do not edit it here.
"""

import json
from enum import StrEnum
from typing import Any

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import Field

from unsafe_eval._resources import (
    content_sha256,
    load_text_resource,
    render_template,
)
from unsafe_eval.profiles._structured_label import StructuredLabelFormat, sample_policy_instruction

SCOPE_GUARD_2609_MODEL_RELEASE = "scope-guard-2609"
SCOPE_GUARD_2609_REQUEST_METADATA = "scope_guard_2609_request"

_SYSTEM_TEMPLATE = load_text_resource(
    "profiles", "prompts", SCOPE_GUARD_2609_MODEL_RELEASE, "system.txt"
)
_USER_TEMPLATE = load_text_resource(
    "profiles", "prompts", SCOPE_GUARD_2609_MODEL_RELEASE, "user.txt"
)


class ScopeClass2609(StrEnum):
    """The seven native classification labels of the 2609 release."""

    DIRECTLY_SUPPORTED = "Directly Supported"
    POTENTIALLY_SUPPORTED = "Potentially Supported"
    PREDEFINED_ANSWER = "Predefined Answer"
    HUMAN_OVERSIGHT = "Human Oversight"
    OUT_OF_SCOPE = "Out of Scope"
    RESTRICTED = "Restricted"
    CHIT_CHAT = "Chit Chat"


SCOPE_CLASSES_2609 = tuple(scope_class.value for scope_class in ScopeClass2609)

# Key order mirrors CANONICAL_ORDER in the training repository: the model was trained to
# emit the requested keys in this order.
_FIELD_DEFINITIONS: dict[str, tuple[Any, Any]] = {
    "evidences": (
        list[str] | None,
        Field(
            default=None,
            description=(
                "Excerpts quoted verbatim from the AI Service Description that support "
                "the assigned class, or null when none apply."
            ),
        ),
    ),
    "reasoning": (
        str,
        Field(description=("A short English explanation naming the decisive evidence or rule.")),
    ),
    "scope_class": (
        ScopeClass2609,
        Field(description="The class assigned to the LAST MESSAGE."),
    ),
    "suggested_response": (
        str | None,
        Field(
            default=None,
            description=(
                "The shortest useful reply for the user when the class calls for one, "
                "otherwise null."
            ),
        ),
    ),
}


SCOPE_GUARD_2609_FORMAT = StructuredLabelFormat(
    model_name="ScopeGuard2609ResponseModel",
    field_definitions=_FIELD_DEFINITIONS,
    label_field="scope_class",
)


def scope_guard_2609_user_message(
    prompt: str,
    *,
    ai_service_description: str,
    selection: tuple[str, ...],
) -> str:
    """Build the 2609 AI-service description, conversation dump and selector."""
    return render_template(
        _USER_TEMPLATE,
        ai_service_description=ai_service_description,
        prompt=prompt,
        requested_output_fields=json.dumps(list(selection)),
    )


SCOPE_GUARD_2609_SYSTEM_MESSAGE = _SYSTEM_TEMPLATE
SCOPE_GUARD_2609_PROMPT_RESOURCE_HASHES = {
    "profile/system.txt": content_sha256(_SYSTEM_TEMPLATE),
    "profile/user.txt": content_sha256(_USER_TEMPLATE),
}


def _apply_prompt(
    state: TaskState, ai_service_description: str, selection: tuple[str, ...]
) -> TaskState:
    # The `orbitals` provider builds its own prompt from these raw inputs.
    state.user_prompt.metadata = {
        **(state.user_prompt.metadata or {}),
        SCOPE_GUARD_2609_REQUEST_METADATA: {
            "prompt": state.user_prompt.text,
            "ai_service_description": ai_service_description,
            "output_fields": list(selection),
        },
    }
    state.user_prompt.text = scope_guard_2609_user_message(
        state.user_prompt.text,
        ai_service_description=ai_service_description,
        selection=selection,
    )
    state.messages.insert(0, ChatMessageSystem(content=SCOPE_GUARD_2609_SYSTEM_MESSAGE))
    return state


@solver
def scope_guard_2609_prompt(
    ai_service_description: str,
    output_fields: tuple[str, ...] = ("reasoning", "scope_class"),
) -> Solver:
    """Apply the 2609 system and user prompt format."""
    selection = SCOPE_GUARD_2609_FORMAT.normalize_selection(list(output_fields))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return _apply_prompt(state, ai_service_description, selection)

    return solve


@solver
def scope_guard_2609_metadata_policy_prompt(
    ai_service_description_template: str,
    policy_field: str = "safety_policy",
    output_fields: tuple[str, ...] = ("reasoning", "scope_class"),
) -> Solver:
    """Apply the 2609 format using a policy stored in sample metadata."""
    selection = SCOPE_GUARD_2609_FORMAT.normalize_selection(list(output_fields))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return _apply_prompt(
            state,
            sample_policy_instruction(
                state,
                ai_service_description_template,
                policy_field=policy_field,
                family="ScopeGuard",
            ),
            selection,
        )

    return solve
