"""The `gemini` protocol: Gemini models classifying prompts against a benchmark policy.

The prompt, response schema and per-benchmark policies are Gemini's own, in
`profiles/gemini.py` and `protocols/prompts/<benchmark>/gemini/`. They started as copies
of the `claude` protocol's and can be tuned for Gemini without touching another family.
Each run records their digests, which show whether a Gemini result shared a prompt with
another family's.

A run reaches the model over one of `SUPPORTED_ROUTES`, chosen by the required `route`
option. The packaged configurations send both routes to Vertex AI: the google ones through
Google's SDK (`google/vertex/...`), the openrouter ones by pinning OpenRouter to its Vertex
endpoint. Google AI Studio blocks far more of these prompts than Vertex does, so a route
that reached it would report a different refusal rate for the same model. Both routes send
the same prompt, schema, thinking level and completion ceiling. Four settings need
explaining:

- The protocol sends no temperature. Google recommends leaving every Gemini 3 model at
  its default of 1.0 and warns that a lower value can cause looping.
- It pins `seed` on the routes that honour one. OpenRouter forwards it, but Inspect's
  `google/` provider never sends one, so the two routes differ in this one parameter.
  See `docs/reproducibility.md`.
- `reasoning_effort` becomes a Gemini thinking level. Inspect sends it to Gemini 3 as
  `thinking_level`, and OpenRouter maps it to the same level. Gemini 3 cannot turn
  thinking off, so `none` is rejected, and 3.7 and 3.8 Flash have no `minimal` level.
- Gemini counts thinking tokens against `max_tokens`, and a thinking level sets no fixed
  budget. The ceiling is therefore generous: a completion cut off mid-object scores
  invalid, and only the tokens Gemini generates are billed.

On the `google` route Inspect turns Gemini's adjustable safety filters off by default,
and the packaged configurations set the same thresholds explicitly so the run log records
them. Filters Google does not let a caller adjust still apply, and on OpenRouter the
protocol cannot set any. A blocked prompt is recorded as a refusal on either route.
"""

from typing import Literal, get_args

from unsafe_eval.profiles.gemini import (
    GEMINI_FORMAT,
    GEMINI_PROMPT_RESOURCE_HASHES,
    gemini_generate_or_refusal,
    gemini_metadata_policy_prompt,
    gemini_prompt,
)
from unsafe_eval.protocols._benchmark_specs import BenchmarkName, BenchmarkProtocolSpec
from unsafe_eval.protocols._hosted import build_hosted_protocol
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.gemini_safety import gemini_safety

PROTOCOL: ProtocolName = "gemini"

# `reasoning` comes first so the model explains before it labels. Matches `gpt`.
DEFAULT_OUTPUT_FIELDS: tuple[str, ...] = ("reasoning", "user_safety")

# The thinking levels every Gemini 3 model in the configurations accepts. `low` is the
# default because the task is one short classification and thinking is billed.
ReasoningEffort = Literal["low", "medium", "high"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low"

# Covers thinking and the JSON object together: half of Google's 65536-token ceiling,
# which leaves any thinking level room to finish.
DEFAULT_MAX_TOKENS = 32768

GENERATION_SEED = 0

SUPPORTED_ROUTES: tuple[str, ...] = ("google", "openrouter")


def build_gemini_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build Gemini components for one compatible benchmark, over the requested route."""
    return build_hosted_protocol(
        PROTOCOL,
        benchmark,
        spec,
        options,
        family="Gemini",
        label_format=GEMINI_FORMAT,
        prompt=gemini_prompt,
        metadata_policy_prompt=gemini_metadata_policy_prompt,
        scorer=gemini_safety,
        solver=gemini_generate_or_refusal(),
        prompt_resource_hashes=GEMINI_PROMPT_RESOURCE_HASHES,
        supported_routes=SUPPORTED_ROUTES,
        default_output_fields=DEFAULT_OUTPUT_FIELDS,
        reasoning_efforts=REASONING_EFFORTS,
        default_reasoning_effort=DEFAULT_REASONING_EFFORT,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=GENERATION_SEED,
    )
