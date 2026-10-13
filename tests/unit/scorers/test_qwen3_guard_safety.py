import pytest

from unsafe_eval.scorers.qwen3_guard_safety import parse_qwen3_guard_completion


@pytest.mark.parametrize(
    ("completion", "safety_label", "verdict"),
    [
        ("Safety: Unsafe", "unsafe", "yes"),
        ("Safety: Safe", "safe", "no"),
        ("Safety: Controversial", "controversial", "yes"),
    ],
)
def test_native_safety_labels_map_to_verdicts(
    completion: str, safety_label: str, verdict: str
) -> None:
    result = parse_qwen3_guard_completion(completion)

    assert result.safety_label == safety_label
    assert result.verdict == verdict
    assert result.parsing_status == "native"


def test_controversial_can_map_to_safe() -> None:
    result = parse_qwen3_guard_completion(
        "Safety: Controversial",
        controversial_as_unsafe=False,
    )

    assert result.verdict == "no"


@pytest.mark.parametrize(
    "completion",
    [
        "Unsafe",
        "Safety: Maybe",
        "Safety: Unsafe extra",
        "Safety: Safe\nSafety: Safe",
        "Safety: Safe\nSafety: Maybe",
    ],
)
def test_missing_malformed_or_duplicated_labels_are_invalid(completion: str) -> None:
    result = parse_qwen3_guard_completion(completion)

    assert result.verdict == "invalid"
    assert result.safety_label is None
    assert result.categories == []
    assert result.parsing_status == "invalid"
