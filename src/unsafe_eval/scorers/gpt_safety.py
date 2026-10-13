"""Scoring for the GPT prompt-safety format.

Applies the hosted-family rules in `scorers/_structured_label.py` to the GPT response
model from `profiles/gpt.py`.
"""

from inspect_ai.scorer import Scorer, scorer

from unsafe_eval.profiles.gpt import GPT_FORMAT
from unsafe_eval.scorers._structured_label import structured_label_scorer
from unsafe_eval.scorers.metrics import basic_metrics


@scorer(metrics=basic_metrics())
def gpt_safety(output_fields: tuple[str, ...] = ("reasoning", "user_safety")) -> Scorer:
    """Score a GPT model's structured safe/unsafe classification."""
    return structured_label_scorer(GPT_FORMAT, output_fields)
