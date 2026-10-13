"""Small validation helpers shared by benchmark record adapters."""

from collections.abc import Mapping, Sequence
from typing import Any


def require_fields(
    record: Mapping[str, Any],
    fields: Sequence[str],
    *,
    benchmark: str,
) -> None:
    """Require a benchmark record to contain every declared field."""
    missing = [field for field in fields if field not in record]
    if missing:
        noun = "field" if len(fields) == 1 else "fields"
        raise ValueError(f"{benchmark} record is missing required {noun}: {', '.join(missing)}")


def require_string(
    record: Mapping[str, Any],
    field: str,
    *,
    benchmark: str,
    non_empty: bool = False,
) -> str:
    """Return one strictly typed string field, optionally rejecting emptiness."""
    value = record[field]
    if not isinstance(value, str) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise ValueError(f"{benchmark} {field} must be a {qualifier}string")
    return value
