"""d1 benchmark protocol construction.

d1 is an open-weight System One model: it takes the same state and questions as Jev and
returns a probability for every option of a question in one forward pass. The `d1` model
provider runs it locally, and this protocol reuses Jev's solvers and the `choice_safety`
scorer, so the two differ only in the provider and the questions each benchmark asks. As
for Jev, the verdict is the option with the higher probability, and the per-sample unsafe
probability is recorded.

d1's model card documents a question's instructions and each option as a plain string, and
d1 writes them into its prompt verbatim, so it cannot take Jev's structured questions. Each
benchmark therefore supplies its own `protocols/prompts/<benchmark>/d1/questions.json`, with
Clef's wording: Jev's question and unsafe definition in the instructions, and each option
describing its verdict, with any of Jev's exclusions after `Excludes:`. d1 shows each option
by its description alone, never by its name, so a description must read correctly without
the name.
"""

from functools import partial

from unsafe_eval.protocols._types import ProtocolName
from unsafe_eval.protocols.jev import build_system_one_protocol

PROTOCOL: ProtocolName = "d1"

build_d1_protocol = partial(build_system_one_protocol, PROTOCOL)
