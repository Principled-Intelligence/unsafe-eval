"""Clef benchmark protocol construction.

Clef is an open-weight System One model: it takes the same state and questions as Jev and
returns a probability for every option of every question in one forward pass. The `clef`
model provider runs it locally, and this protocol reuses Jev's solvers and the
`choice_safety` scorer, so the two differ only in the provider and the questions each
benchmark asks. As for Jev, the verdict is the option with the higher probability, and the
per-sample unsafe probability is recorded.

Each benchmark supplies its own `protocols/prompts/<benchmark>/clef/questions.json` with
Jev's categories and guidance, written as the plain strings Clef's model card documents.
Clef sorts a question's options alphabetically, so `safe` precedes `unsafe`. The
instructions therefore carry the definition of unsafe, and each option refers only to them,
never to the other option.
"""

from functools import partial

from unsafe_eval.protocols._types import ProtocolName
from unsafe_eval.protocols.jev import build_system_one_protocol

PROTOCOL: ProtocolName = "clef"

build_clef_protocol = partial(build_system_one_protocol, PROTOCOL)
