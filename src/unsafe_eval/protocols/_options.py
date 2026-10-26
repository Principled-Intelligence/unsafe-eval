"""Strict validation helpers for protocol-specific options."""

from collections.abc import Set
from typing import TypeVar

from unsafe_eval.protocols._routes import Route, route_spec
from unsafe_eval.protocols._types import ProtocolName

ROUTE = "route"
VALIDATE_PROMPT_RESOURCE_HASHES = "validate_prompt_resource_hashes"

_Choice = TypeVar("_Choice", bound=str)


def protocol_options(
    protocol: ProtocolName,
    supplied: dict[str, object] | None,
    allowed: Set[str],
) -> dict[str, object]:
    """Return options after rejecting keys outside `allowed`.

    `None` means no options. The shared `validate_prompt_resource_hashes` key is always
    accepted. This helper checks names only; callers must validate each option's value.
    """
    values = supplied or {}
    unexpected = sorted(set(values) - allowed - {VALIDATE_PROMPT_RESOURCE_HASHES})
    if unexpected:
        raise ValueError(f"Unsupported options for {protocol}: {unexpected!r}")
    return values


def boolean_option(values: dict[str, object], name: str, default: bool) -> bool:
    """Resolve a strictly boolean protocol option."""
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"Protocol option {name!r} must be a boolean")
    return value


def choice_option(
    values: dict[str, object],
    name: str,
    choices: tuple[_Choice, ...],
    default: _Choice,
) -> _Choice:
    """Resolve a protocol option that must be one of `choices`."""
    value = values.get(name, default)
    for choice in choices:
        if value == choice:
            return choice
    raise ValueError(f"Protocol option {name!r} must be one of {list(choices)}")


def validate_prompt_resource_hashes_option(values: dict[str, object]) -> bool:
    """Resolve the common prompt-resource hash-validation option."""
    return boolean_option(values, VALIDATE_PROMPT_RESOURCE_HASHES, True)


def threshold_option(values: dict[str, object]) -> float:
    """Read `threshold`, defaulting to 0.5, as a float between 0 and 1 inclusive.

    Raise `ValueError` for booleans, non-numeric values, or values outside that range.
    """
    value = values.get("threshold", 0.5)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Protocol option 'threshold' must be a number")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Protocol option 'threshold' must be between 0 and 1 inclusive")
    return threshold


def route_option(values: dict[str, object], supported: tuple[str, ...]) -> Route:
    """Resolve the required route option against the routes a protocol supports.

    An unregistered name fails before an unsupported one, so a typo and a route this
    family does not answer on raise different errors.
    """
    if ROUTE not in values:
        raise ValueError(
            f"Protocol option 'route' is required and must be one of {list(supported)}"
        )
    value = values[ROUTE]
    if not isinstance(value, str):
        raise ValueError("Protocol option 'route' must be a string")
    route = route_spec(value)
    if route.name not in supported:
        raise ValueError(f"Protocol option 'route' must be one of {list(supported)}")
    return route
