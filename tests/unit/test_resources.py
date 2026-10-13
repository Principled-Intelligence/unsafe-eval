import pytest

from unsafe_eval.resources import load_text_resource, render_template


def test_missing_prompt_template_fails_instead_of_becoming_literal_text() -> None:
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        load_text_resource("profiles", "prompts", "scope-guard-2601", "missing.txt")


@pytest.mark.parametrize(
    ("values", "detail"),
    [
        ({}, "missing=\\['prompt'\\]"),
        ({"prompt": "value", "extra": "value"}, "unexpected=\\['extra'\\]"),
    ],
)
def test_render_template_rejects_mismatched_values(values: dict[str, str], detail: str) -> None:
    with pytest.raises(ValueError, match=detail):
        render_template("Prompt: {prompt}", **values)
