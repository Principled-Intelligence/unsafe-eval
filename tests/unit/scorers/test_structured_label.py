import pytest

from unsafe_eval.profiles.gpt import GPT_FORMAT
from unsafe_eval.scorers._structured_label import parse_structured_label

RESPONSE_MODEL = GPT_FORMAT.response_model_for(("reasoning", "user_safety"))


@pytest.mark.parametrize(
    ("completion", "user_safety", "verdict"),
    [
        ('{"reasoning": "r", "user_safety": "unsafe"}', "unsafe", "yes"),
        ('{"reasoning": "r", "user_safety": "safe"}', "safe", "no"),
    ],
)
def test_native_labels_map_to_verdicts(completion: str, user_safety: str, verdict: str) -> None:
    result = parse_structured_label(completion, RESPONSE_MODEL)

    assert result.user_safety == user_safety
    assert result.verdict == verdict
    assert result.parsing_status == "json"
    assert result.reasoning == "r"


@pytest.mark.parametrize(
    "completion",
    [
        # A bare label is not the object.
        "unsafe",
        # Prose around the object is not salvaged.
        'The request is harmful.\n{"reasoning": "r", "user_safety": "unsafe"}',
        # Wrong vocabulary: this protocol's native labels are safe/unsafe only.
        '{"reasoning": "r", "user_safety": "yes"}',
        # A requested key missing entirely.
        '{"reasoning": "r"}',
        # An unrequested key present.
        '{"reasoning": "r", "user_safety": "unsafe", "categories": ["S1"]}',
        # A JSON array rather than the object.
        '["unsafe"]',
    ],
)
def test_malformed_or_off_schema_completions_are_invalid(completion: str) -> None:
    result = parse_structured_label(completion, RESPONSE_MODEL)

    assert result.verdict == "invalid"
    assert result.user_safety is None
    assert result.reasoning is None
    assert result.parsing_status == "invalid"
