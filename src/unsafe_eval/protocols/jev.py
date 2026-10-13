"""Jev benchmark protocol construction.

Jev is a hosted System One model: it answers a set of structured questions instead of
generating text, so the model call lives behind a registered Inspect provider
(`providers/typesafe.py`) and every layer here keeps its usual job.

Jev emits calibrated probabilities, and a cut chosen by looking at benchmark results
would make the comparison with other protocols unfair. Jev's native label on a single
Choice question with `safe` and `unsafe` options is therefore the verdict, and the
protocol has no tunable parameter. The per-sample unsafe probability is still recorded
for later threshold or AUROC analysis.

The request is a state plus a set of questions and takes no sampling parameters, so the
`GenerateConfig` is empty and the log implies no controls that do not exist.

Hosted providers expose no immutable revision. The config pins an exact model version,
and the provider fails the run when the response reports a different one, so alias drift
is caught. The run is still not bit-reproducible; see `docs/reproducibility.md`.

Each benchmark supplies its own questions under `protocols/prompts/<benchmark>/jev/`,
because Jev publishes no prompt-safety taxonomy and each benchmark's gold labels follow
their own. On a policy-conditioned benchmark the per-sample policy travels as its own
field of the structured state, so the questions and their recorded digest are identical
on every sample.
"""

from functools import partial

from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Solver

from unsafe_eval.profiles.jev import jev_policy_prompt, jev_prompt
from unsafe_eval.protocols._benchmark_specs import (
    BenchmarkName,
    BenchmarkProtocolSpec,
    build_protocol_metrics,
)
from unsafe_eval.protocols._options import (
    protocol_options,
    validate_prompt_resource_hashes_option,
)
from unsafe_eval.protocols._resource_manifest import adaptation_resource
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.choice_safety import choice_safety

PROTOCOL: ProtocolName = "jev"


def build_system_one_protocol(
    protocol: ProtocolName,
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build components for a System One model asking `protocol`'s packaged questions."""
    values = protocol_options(protocol, options, set())
    validate_hashes = validate_prompt_resource_hashes_option(values)
    questions, questions_hashes = adaptation_resource(
        benchmark, protocol, "questions.json", validate_hash=validate_hashes
    )
    setup = jev_policy_prompt(questions) if spec.policy_conditioned else jev_prompt(questions)
    return choice_protocol(
        protocol, spec, setup, validate_hashes=validate_hashes, hashes=questions_hashes
    )


build_jev_protocol = partial(build_system_one_protocol, PROTOCOL)


def choice_protocol(
    protocol: ProtocolName,
    spec: BenchmarkProtocolSpec,
    setup: Solver,
    *,
    validate_hashes: bool,
    hashes: dict[str, str],
) -> SafetyProtocol:
    """Build components for a model whose verdict is the more probable of `safe` and `unsafe`."""
    return SafetyProtocol(
        setup=setup,
        scorer=choice_safety(),
        config=GenerateConfig(),
        # Only unsafe-only benchmarks aggregate the probability; mixed-label ones keep it
        # per sample.
        metrics=build_protocol_metrics(spec, probability=True),
        metadata={
            "protocol": protocol,
            "protocol_options": {"validate_prompt_resource_hashes": validate_hashes},
            "prompt_resource_hashes": hashes,
        },
    )
