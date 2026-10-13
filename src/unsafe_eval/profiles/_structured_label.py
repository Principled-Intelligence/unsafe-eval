"""Family-neutral mechanics for a structured safe/unsafe classification against a policy.

The hosted families in this package have no published prompt-safety format, so each
profile writes its own system message, user template and schema field descriptions.
That text is prompt content. It belongs to the family and can diverge as each is tuned.
This module holds the parts that do not depend on the family: the two native labels,
how a field selection becomes a response model, and how a policy and the content are
placed into a family's template.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import TaskState
from inspect_ai.util import JSONSchema, json_schema
from pydantic import BaseModel, ConfigDict, create_model

from unsafe_eval._resources import render_template

# The field every selection contains, because it carries the verdict.
LABEL_FIELD = "user_safety"


class UserSafety(StrEnum):
    """The two native labels a structured classification chooses between."""

    SAFE = "safe"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class StructuredLabelFormat:
    """One family's response contract: its fields, in the order they are decoded.

    `field_definitions` maps each field to a pydantic `(type, Field(...))` pair. Its
    order is the canonical order, and the descriptions are sent to the model as part
    of the schema, so they are the family's prompt text rather than shared code.
    `model_name` names the pydantic model only: Inspect's schema carries no title,
    so the model never sees it. `label_field` carries the verdict, so every selection
    contains it.
    """

    model_name: str
    field_definitions: dict[str, tuple[Any, Any]]
    label_field: str = LABEL_FIELD

    def __post_init__(self) -> None:
        if self.label_field not in self.field_definitions:
            raise ValueError(f"A structured label format must define {self.label_field!r}")

    @property
    def canonical_order(self) -> tuple[str, ...]:
        return tuple(self.field_definitions)

    def normalize_selection(self, fields: object) -> tuple[str, ...]:
        """Canonicalise a requested-field selection.

        The label field is forced in rather than trusted to be present, and
        selections differing only in order normalise to the same tuple, so
        equivalent requests render a byte-identical schema.
        """
        if isinstance(fields, str) or not isinstance(fields, (list, tuple, set, frozenset)):
            raise ValueError(
                f"output_fields must be a list of field names, got {type(fields).__name__}"
            )
        requested = {str(field) for field in fields} | {self.label_field}
        unknown = sorted(requested - set(self.canonical_order))
        if unknown:
            raise ValueError(
                f"unknown output field(s) {unknown}; "
                f"expected a subset of {list(self.canonical_order)}"
            )
        return tuple(field for field in self.canonical_order if field in requested)

    def response_model_for(self, selection: tuple[str, ...]) -> type[BaseModel]:
        """Build the response model that emits exactly `selection`, in canonical order."""
        fields: dict[str, Any] = {field: self.field_definitions[field] for field in selection}
        model: type[BaseModel] = create_model(
            self.model_name,
            # extra="forbid" keeps a completion carrying an unrequested key invalid.
            # Pydantic's strict= governs type coercion only and would let it through.
            __config__=ConfigDict(title=self.model_name, strict=True, extra="forbid"),
            **fields,
        )
        return model

    def inspect_response_schema(self, selection: tuple[str, ...]) -> JSONSchema:
        """Inspect-side schema for guided decoding of exactly `selection`."""
        return json_schema(self.response_model_for(selection))


def apply_policy_prompt(
    state: TaskState,
    *,
    system_message: str,
    user_template: str,
    instruction: str,
) -> TaskState:
    """Rewrite the user turn around `instruction` and prepend the system message."""
    state.user_prompt.text = render_template(
        user_template, instruction=instruction, prompt=state.user_prompt.text
    )
    state.messages.insert(0, ChatMessageSystem(content=system_message))
    return state


def sample_policy_instruction(
    state: TaskState, instruction_template: str, *, policy_field: str, family: str
) -> str:
    """Render a per-sample policy into a policy-conditioned instruction template.

    The policy is substituted as a value, never as part of a format string, so
    braces inside a real policy stay literal.
    """
    policy = state.metadata.get(policy_field)
    if not isinstance(policy, str) or not policy:
        raise ValueError(
            f"{family} sample metadata field {policy_field!r} must be a non-empty string"
        )
    return render_template(instruction_template, safety_policy=policy)
