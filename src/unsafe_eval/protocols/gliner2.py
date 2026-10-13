"""GLiNER2.5-Decide and GLiGuard benchmark protocol construction.

Both are GLiNER2 classifiers run locally by the `gliner2` model provider. Each answers a
single-label task whose labels are `safe` and `unsafe`, so the `choice_safety` scorer
reads the verdict and records the unsafe probability. The verdict is the more probable
label; there is no threshold, and there is nothing to sample.

They differ in where the question comes from. GLiNER2.5-Decide is a zero-shot decision
model with no safety training, so each benchmark supplies
`protocols/prompts/<benchmark>/gliner2.5-decide/schema.json`. It asks Laya's question as the
task instruction, with Laya's options as label descriptions, and a policy-conditioned
benchmark's policy is written into that instruction. GLiGuard answers its native
`prompt_safety` task the same way on every benchmark and reads a policy from the text.
"""

from unsafe_eval.profiles.gliner2 import (
    GLIGUARD_POLICY_TEXT,
    GLIGUARD_PROMPT_RESOURCE_HASHES,
    GLIGUARD_SCHEMA,
    gliner2_prompt,
)
from unsafe_eval.protocols._benchmark_specs import BenchmarkName, BenchmarkProtocolSpec
from unsafe_eval.protocols._options import (
    protocol_options,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.protocols.jev import choice_protocol

DECIDE: ProtocolName = "gliner2.5-decide"
GLIGUARD: ProtocolName = "gliguard"


def build_gliner2_decide_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build GLiNER2.5-Decide components for one benchmark."""
    values = protocol_options(DECIDE, options, set())
    validate_hashes = validate_prompt_resource_hashes_option(values)
    schema, hashes = adaptation_resource(
        benchmark, DECIDE, "schema.json", validate_hash=validate_hashes
    )
    return choice_protocol(
        DECIDE, spec, gliner2_prompt(schema), validate_hashes=validate_hashes, hashes=hashes
    )


def build_gliguard_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build GLiGuard components for one benchmark."""
    values = protocol_options(GLIGUARD, options, set())
    validate_hashes = validate_prompt_resource_hashes_option(values)
    text_template = GLIGUARD_POLICY_TEXT if spec.policy_conditioned else "{message}"
    return choice_protocol(
        GLIGUARD,
        spec,
        gliner2_prompt(GLIGUARD_SCHEMA, text_template),
        validate_hashes=validate_hashes,
        hashes=dict(GLIGUARD_PROMPT_RESOURCE_HASHES),
    )
