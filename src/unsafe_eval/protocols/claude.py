"""The `claude` protocol: Claude models classifying prompts against a benchmark policy.

The prompt, response schema and per-benchmark policies are Claude's own, in
`profiles/claude.py` and `protocols/prompts/<benchmark>/claude/`. They started as copies
of the `gpt` protocol's and can be tuned for Claude without touching GPT. Each run
records their digests, which show whether a Claude result and a GPT result shared a
prompt.

A run reaches the model over one of `SUPPORTED_ROUTES`, chosen by the required `route`
option. Both routes build the same request. Three settings need explaining:

- No seed and no temperature are sent on either route. Anthropic's API has no seed
  parameter and OpenRouter lists none for its Anthropic endpoints, so this protocol
  ignores the `openrouter` route's `honours_seed`, which is true for other families.
  Current Claude models reject sampling parameters. A Claude run is therefore
  repeatable only in its inputs; see `docs/reproducibility.md`.
- `max_tokens` depends on the reasoning effort. Claude Haiku 4.5 predates adaptive
  thinking, so Inspect turns the effort into a fixed thinking budget, and Anthropic
  rejects a budget that is not below `max_tokens`. Sonnet 5 and Opus 5 think adaptively
  and treat the same number as a plain ceiling.
- Haiku 4.5 can think for a different number of tokens on each route, because
  OpenRouter turns the effort into a budget itself where Inspect does on `anthropic`.
"""

from typing import Literal, get_args

from unsafe_eval.profiles.claude import (
    CLAUDE_FORMAT,
    CLAUDE_PROMPT_RESOURCE_HASHES,
    claude_generate_or_refusal,
    claude_metadata_policy_prompt,
    claude_prompt,
)
from unsafe_eval.protocols._benchmark_specs import BenchmarkName, BenchmarkProtocolSpec
from unsafe_eval.protocols._hosted import build_hosted_protocol
from unsafe_eval.protocols._types import ProtocolName, SafetyProtocol
from unsafe_eval.scorers.claude_safety import claude_safety

PROTOCOL: ProtocolName = "claude"

# `reasoning` comes first so the model explains before it labels. Matches `gpt`.
DEFAULT_OUTPUT_FIELDS: tuple[str, ...] = ("reasoning", "user_safety")

# `none` turns thinking off. Inspect's `minimal` is left out because it sends `low`,
# and two option values for one request would make logs disagree.
ReasoningEffort = Literal["none", "low", "medium", "high"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low"

# The thinking budget Inspect gives Claude Haiku 4.5 at each effort, copied from
# Inspect's effort table. A test checks that the two still agree.
REASONING_BUDGETS: dict[str, int] = {"none": 0, "low": 4096, "medium": 10000, "high": 16000}

# Room for the JSON object on top of any thinking budget; a completion cut off
# mid-object scores invalid.
ANSWER_TOKENS = 4096

SUPPORTED_ROUTES: tuple[str, ...] = ("anthropic", "openrouter")


def build_claude_protocol(
    benchmark: BenchmarkName,
    spec: BenchmarkProtocolSpec,
    options: dict[str, object] | None,
) -> SafetyProtocol:
    """Build Claude components for one compatible benchmark, over the requested route."""
    return build_hosted_protocol(
        PROTOCOL,
        benchmark,
        spec,
        options,
        family="Claude",
        label_format=CLAUDE_FORMAT,
        prompt=claude_prompt,
        metadata_policy_prompt=claude_metadata_policy_prompt,
        scorer=claude_safety,
        solver=claude_generate_or_refusal(),
        prompt_resource_hashes=CLAUDE_PROMPT_RESOURCE_HASHES,
        supported_routes=SUPPORTED_ROUTES,
        default_output_fields=DEFAULT_OUTPUT_FIELDS,
        reasoning_efforts=REASONING_EFFORTS,
        default_reasoning_effort=DEFAULT_REASONING_EFFORT,
        max_tokens=ANSWER_TOKENS,
        reasoning_budgets=REASONING_BUDGETS,
    )
