"""The `gpt` protocol: GPT-5.6 models classifying prompts against a benchmark policy.

A run reaches the model over one of `SUPPORTED_ROUTES`, chosen by the required `route`
option. The route decides only which request parameters are sent. The prompt, response
schema, options and policies are the same on every route, so results from two routes
can be compared directly.

Each benchmark supplies its own policy in a packaged `instruction.txt`, because the
benchmarks disagree about which categories are unsafe, one of them classifies content
instead of requests, and their thresholds differ. On a policy-conditioned benchmark the
instruction is a template that each sample's own policy fills in, so the sample is
judged against the policy it was labelled under.

Two settings differ from the guard-model protocols, which serve open weights:

- GPT-5.6 models reject sampling parameters, so this protocol sends no temperature. It
  pins `seed` instead, on the routes that honour one, which makes a run repeatable on the
  same route and build but not across routes or builds.
- A hosted model has no commit hash to pin. A run records a null model revision, and the
  configuration can at best name a dated snapshot. See `docs/reproducibility.md`.
"""

from typing import Literal, get_args

from unsafe_eval.profiles.gpt import (
    GPT_FORMAT,
    GPT_PROMPT_RESOURCE_HASHES,
    gpt_generate_or_refusal,
    gpt_metadata_policy_prompt,
    gpt_prompt,
)
from unsafe_eval.protocols._benchmark_specs import BenchmarkName, BenchmarkProtocolSpec
from unsafe_eval.protocols._hosted import build_hosted_protocol
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.gpt_safety import gpt_safety

PROTOCOL: ProtocolName = "gpt"

# `reasoning` comes first so the model explains before it labels. Dropping it is
# cheaper: the field is billed output on every sample.
DEFAULT_OUTPUT_FIELDS: tuple[str, ...] = ("reasoning", "user_safety")

# Covers the hidden reasoning as well as the JSON object, and a completion cut off
# mid-object scores invalid.
DEFAULT_MAX_TOKENS = 16384

# `low` is the default because the task is one short classification and effort is billed.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low"

GENERATION_SEED = 0

SUPPORTED_ROUTES: tuple[str, ...] = ("openrouter", "openai")


def build_gpt_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build GPT components for one compatible benchmark, over the requested route."""
    return build_hosted_protocol(
        PROTOCOL,
        benchmark,
        spec,
        options,
        family="GPT",
        label_format=GPT_FORMAT,
        prompt=gpt_prompt,
        metadata_policy_prompt=gpt_metadata_policy_prompt,
        scorer=gpt_safety,
        solver=gpt_generate_or_refusal(),
        prompt_resource_hashes=GPT_PROMPT_RESOURCE_HASHES,
        supported_routes=SUPPORTED_ROUTES,
        default_output_fields=DEFAULT_OUTPUT_FIELDS,
        reasoning_efforts=REASONING_EFFORTS,
        default_reasoning_effort=DEFAULT_REASONING_EFFORT,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=GENERATION_SEED,
    )
