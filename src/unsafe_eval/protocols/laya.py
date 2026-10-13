"""Laya benchmark protocol construction.

Laya is an open-weight System One model: it takes the same state and questions as Jev
and returns one calibrated answer per question. The `laya` model provider runs it
locally through Laya's `Router`, which picks the English or the multilingual checkpoint
for each sample, and this protocol reuses Jev's solvers and the `choice_safety` scorer, so the
two differ only in the provider and the questions each benchmark asks. As for Jev, the
verdict is the option with the higher probability, and the per-sample unsafe probability
is recorded.

Laya silently cuts each option to 48 tokens and the question text plus its options to
192 on the English checkpoint, and six of Jev's questions exceed that. Each benchmark
therefore supplies its own `protocols/prompts/<benchmark>/laya/questions.json`, which
keeps Jev's categories and wording but moves each category list into the question text
and keeps the options short. Their instructions are a plain string, because Laya
serializes an object to JSON whose keys and quotes spend the same budget. The provider
fails the run if a question would still be cut.

Laya scores options in one forward pass, so there is nothing to sample. Its context
length is the model argument `max_len`, set in each run configuration, so this protocol
takes no options.
"""

from functools import partial

from unsafe_eval.protocols._types import ProtocolName
from unsafe_eval.protocols.jev import build_system_one_protocol

PROTOCOL: ProtocolName = "laya"

build_laya_protocol = partial(build_system_one_protocol, PROTOCOL)
