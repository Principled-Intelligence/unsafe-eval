"""Prompt setup components for evaluated safety models."""

from unsafe_eval.profiles.polyguard import polyguard_prompt
from unsafe_eval.profiles.scope_guard import scope_guard_prompt
from unsafe_eval.profiles.shieldstral import shieldstral_prompt

__all__ = ["polyguard_prompt", "scope_guard_prompt", "shieldstral_prompt"]
