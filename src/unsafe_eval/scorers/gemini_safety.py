"""Scoring for the Gemini prompt-safety format.

Applies the hosted-family rules in `scorers/_structured_label.py` to the Gemini response
model from `profiles/gemini.py`.
"""

from inspect_ai.scorer import Scorer, scorer

from unsafe_eval.profiles.gemini import GEMINI_FORMAT
from unsafe_eval.scorers._structured_label import structured_label_scorer
from unsafe_eval.scorers.metrics import basic_metrics


@scorer(metrics=basic_metrics())
def gemini_safety(output_fields: tuple[str, ...] = ("reasoning", "user_safety")) -> Scorer:
    """Score Gemini's structured safe/unsafe classification."""
    return structured_label_scorer(GEMINI_FORMAT, output_fields)
