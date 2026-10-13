"""Commit pins shared by the local providers."""

import re

_COMMIT = re.compile(r"[0-9a-f]{40}")


def commit(provider: str, name: str, value: object) -> str:
    """Return `value` if it is a 40-character commit hash, else raise `ValueError`."""
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{provider} requires `{name}`, a 40-character commit hash")
    return value
