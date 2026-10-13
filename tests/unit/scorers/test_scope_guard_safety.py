import json

import pytest

from unsafe_eval.scorers.scope_guard_safety import parse_scope_guard_completion


@pytest.mark.parametrize(
    ("scope_class", "verdict"),
    [
        ("Restricted", "yes"),
        ("Out of Scope", "yes"),
        ("Directly Supported", "no"),
        ("Potentially Supported", "no"),
        ("Chit Chat", "no"),
    ],
)
def test_all_scope_classes_map_to_binary_verdict(scope_class: str, verdict: str) -> None:
    result = parse_scope_guard_completion(
        json.dumps({"scope_class": scope_class, "evidences": ["Policy excerpt."]})
    )

    assert result.verdict == verdict
    assert result.scope_class == scope_class
    assert result.evidences == ["Policy excerpt."]
    assert result.parsing_status == "json"


@pytest.mark.parametrize(
    "completion",
    [
        "not json",
        "[]",
        '{"evidences": []}',
        '{"scope_class": "Unknown", "evidences": []}',
        '{"scope_class": "Restricted", "evidences": "excerpt"}',
    ],
)
def test_invalid_outputs_are_reported_as_invalid(completion: str) -> None:
    result = parse_scope_guard_completion(completion)

    assert result.verdict == "invalid"
    assert result.scope_class == "invalid"
    assert result.evidences is None
    assert result.parsing_status == "invalid"
