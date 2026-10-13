"""Scoring for the Claude prompt-safety format.

Applies the hosted-family rules in `scorers/_structured_label.py` to the Claude response
model from `profiles/claude.py`.
"""

from inspect_ai.scorer import Scorer, scorer

from unsafe_eval.profiles.claude import CLAUDE_FORMAT
from unsafe_eval.scorers._structured_label import structured_label_scorer
from unsafe_eval.scorers.metrics import basic_metrics


@scorer(metrics=basic_metrics())
def claude_safety(output_fields: tuple[str, ...] = ("reasoning", "user_safety")) -> Scorer:
    """Score Claude's structured safe/unsafe classification."""
    return structured_label_scorer(CLAUDE_FORMAT, output_fields)
