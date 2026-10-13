"""Registration of model protocols shipped with unsafe-eval."""

import unsafe_eval.providers.clef  # noqa: F401  (registers the `clef/` model provider)
import unsafe_eval.providers.clm  # noqa: F401  (registers the `clm/` model provider)
import unsafe_eval.providers.d1  # noqa: F401  (registers the `d1/` model provider)
import unsafe_eval.providers.gliner2  # noqa: F401  (registers the `gliner2/` model provider)
import unsafe_eval.providers.kev  # noqa: F401  (registers the `kev/` model provider)
import unsafe_eval.providers.laya  # noqa: F401  (registers the `laya/` model provider)
import unsafe_eval.providers.orbitals  # noqa: F401  (registers the `orbitals/` model provider)
from unsafe_eval.profiles.claude import claude_prompt
from unsafe_eval.profiles.gemini import gemini_prompt
from unsafe_eval.profiles.gliner2 import gliner2_prompt
from unsafe_eval.profiles.gpt import gpt_prompt
from unsafe_eval.profiles.jev import jev_policy_prompt, jev_prompt
from unsafe_eval.profiles.polyguard import polyguard_prompt
from unsafe_eval.profiles.scope_guard import scope_guard_prompt
from unsafe_eval.profiles.scope_guard_2609 import scope_guard_2609_prompt
from unsafe_eval.profiles.shieldstral import shieldstral_prompt
from unsafe_eval.protocols._resolver import register_protocol
from unsafe_eval.protocols.claude import build_claude_protocol
from unsafe_eval.protocols.clef import build_clef_protocol
from unsafe_eval.protocols.d1 import build_d1_protocol
from unsafe_eval.protocols.gemini import build_gemini_protocol
from unsafe_eval.protocols.gliner2 import build_gliguard_protocol, build_gliner2_decide_protocol
from unsafe_eval.protocols.gpt import build_gpt_protocol
from unsafe_eval.protocols.jev import build_jev_protocol
from unsafe_eval.protocols.laya import build_laya_protocol
from unsafe_eval.protocols.nemotron import build_nemotron_protocol
from unsafe_eval.protocols.polyguard import build_polyguard_protocol
from unsafe_eval.protocols.qwen3_guard import build_qwen3_guard_protocol
from unsafe_eval.protocols.scope_guard import build_scope_guard_protocol
from unsafe_eval.protocols.scope_guard_2609 import build_scope_guard_2609_protocol
from unsafe_eval.protocols.shieldstral import build_shieldstral_protocol
from unsafe_eval.providers.typesafe import typesafe
from unsafe_eval.scorers.choice_safety import choice_safety
from unsafe_eval.scorers.claude_safety import claude_safety
from unsafe_eval.scorers.gemini_safety import gemini_safety
from unsafe_eval.scorers.gpt_safety import gpt_safety
from unsafe_eval.scorers.nemotron_safety import nemotron_safety
from unsafe_eval.scorers.polyguard_safety import polyguard_safety
from unsafe_eval.scorers.qwen3_guard_safety import qwen3_guard_safety
from unsafe_eval.scorers.scope_guard_2609_safety import scope_guard_2609_safety
from unsafe_eval.scorers.scope_guard_safety import scope_guard_safety
from unsafe_eval.scorers.shieldstral_safety import shieldstral_safety


def register_builtin_protocols() -> None:
    """Register the model protocols distributed in this package."""
    register_protocol(
        "shieldstral-1.0-3b",
        build_shieldstral_protocol,
        components=(shieldstral_prompt, shieldstral_safety),
    )
    register_protocol(
        "scope-guard-2601",
        build_scope_guard_protocol,
        components=(scope_guard_prompt, scope_guard_safety),
    )
    register_protocol(
        "scope-guard-2609",
        build_scope_guard_2609_protocol,
        components=(scope_guard_2609_prompt, scope_guard_2609_safety),
    )
    register_protocol(
        "nemotron-3.5-content-safety",
        build_nemotron_protocol,
        components=(nemotron_safety,),
    )
    register_protocol(
        "qwen3-guard-gen",
        build_qwen3_guard_protocol,
        components=(qwen3_guard_safety,),
    )
    register_protocol(
        "polyguard-qwen",
        build_polyguard_protocol,
        components=(polyguard_prompt, polyguard_safety),
    )
    register_protocol(
        "gpt",
        build_gpt_protocol,
        components=(gpt_prompt, gpt_safety),
    )
    register_protocol(
        "claude",
        build_claude_protocol,
        components=(claude_prompt, claude_safety),
    )
    register_protocol(
        "gemini",
        build_gemini_protocol,
        components=(gemini_prompt, gemini_safety),
    )
    register_protocol(
        "jev",
        build_jev_protocol,
        # Listing the `typesafe` provider here exposes it through Inspect's entry point.
        components=(typesafe, jev_prompt, jev_policy_prompt, choice_safety),
    )
    register_protocol(
        "laya",
        build_laya_protocol,
        # Laya reuses Jev's components. Its provider is imported instead of listed:
        # Inspect names every `@modelapi` wrapper `model_wrapper`, so it would collide
        # with `typesafe` in the component inventory.
    )
    # Clef and d1 reuse Jev's components too, and import their providers for the same reason.
    register_protocol("clef", build_clef_protocol)
    register_protocol("d1", build_d1_protocol)
    register_protocol(
        "gliner2.5-decide",
        build_gliner2_decide_protocol,
        # The `gliner2` provider is imported instead of listed, for the same
        # `model_wrapper` name collision as Laya's.
        components=(gliner2_prompt,),
    )
    register_protocol("gliguard", build_gliguard_protocol)
