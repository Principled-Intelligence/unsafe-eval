import re
from pathlib import Path

import yaml
from inspect_ai.model import ModelName

from unsafe_eval.protocols import resolve_protocol, route_spec

CONFIG_DIR = Path(__file__).parents[2] / "src" / "unsafe_eval" / "configs"

# A checked configuration pins its model wherever the provider can express one. `vllm/`, `hf/`,
# `laya/`, `gliner2/`, `clef/`, `kev/`, `d1/`, `clm/`, and `orbitals/` take a hub commit hash.
# `openrouter/` takes a dated snapshot slug, which is weaker -- it identifies a published build
# rather than its bytes -- but is still not a moving alias. `typesafe/` takes an exact model
# version, and its provider goes further than any other here by failing the run when the response
# reports a different served version, so the pin is verified rather than merely stated.
# `anthropic/` takes a snapshot ID, dated or not. `google/` takes a stable model code on Vertex AI,
# which Google documents as pointing to one specific model. `openai/` can express none of these for
# every model it serves, and `google/` none for a preview model, which is what UNPINNED_CONFIGS
# below is for.
PROVIDER_PREFIXES = (
    "vllm/",
    "hf/",
    "openrouter/",
    "typesafe/",
    "openai/",
    "anthropic/",
    "google/",
    "laya/",
    "gliner2/",
    "clef/",
    "kev/",
    "d1/",
    "orbitals/",
    "clm/",
)

# Configurations whose provider publishes no immutable identifier for the model
# being evaluated. They are runnable examples of a protocol, not reproducible pins:
# the dataset revision and every prompt digest are still fixed, but the model is
# whatever the alias resolves to on the day, so two runs are not comparable the way
# the pinned trees are. Listing them by name keeps the exemption to the trees that
# need it, where dropping the assertion would extend it to every future one.
# See docs/reproducibility.md.
UNPINNED_CONFIGS = {
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-6-astra",
    "openai/gpt-6-luna",
    "openai/gpt-6.1-sol",
    # A preview model code, which Google does not document as one fixed build. The
    # OpenRouter tree of the same name pins a dated slug.
    "google/gemini-3.1-pro-preview",
}

# The immutable identifier each configuration pins its model to: a hub commit hash
# for the local runtimes, a dated snapshot slug for the hosted one.
CONFIG_REVISIONS = {
    "openrouter/gpt-5.6-sol": "gpt-5.6-sol-20260709",
    "openrouter/gpt-5.6-luna": "gpt-5.6-luna-20260709",
    "openrouter/gpt-5.6-terra": "gpt-5.6-terra-20260709",
    "openrouter/gpt-6-astra": "gpt-6-astra-20260903",
    "openrouter/gpt-6-luna": "gpt-6-luna-20260922",
    "openrouter/gpt-6.1-sol": "gpt-6.1-sol-20260929",
    # Undated Claude IDs are snapshots too.
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-5": "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5": "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5.5": "anthropic/claude-sonnet-5-5",
    "anthropic/claude-opus-5.5": "anthropic/claude-opus-5-5",
    "anthropic/claude-haiku-5.5": "anthropic/claude-haiku-5-5",
    "openrouter/claude-haiku-4.5": "claude-4.5-haiku-20251001",
    "openrouter/claude-sonnet-5": "claude-sonnet-5-20260630",
    "openrouter/claude-opus-5": "claude-opus-5-20260723",
    "openrouter/claude-sonnet-5.5": "claude-sonnet-5.5-20260928",
    "openrouter/claude-opus-5.5": "claude-opus-5.5-20260921",
    "openrouter/claude-haiku-5.5": "claude-haiku-5.5-20261007",
    # Google documents a stable model code as pointing to one specific model, so the
    # code is the pin. OpenRouter publishes a dated slug for both.
    "google/gemini-3.8-flash": "google/vertex/gemini-3.8-flash",
    "openrouter/gemini-3.8-flash": "gemini-3.8-flash-20260902",
    "openrouter/gemini-3.1-pro-preview": "gemini-3.1-pro-preview-20260219",
    "openrouter/nemotron-3.5-content-safety-free": "nemotron-3.5-content-safety-20260604:free",
    "typesafe/jev-1.13": "jev-1.13.0",
    "laya/laya": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
    "clef/clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6",
    "clef/clef-fp8": "2f3de3dd85f379784083b0814d997ab627200f0c",
    "kev/kev-27b": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
    "kev/kev-27b-fp8": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
    "kev/kev-9b": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
    "kev/kev-4b": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
    "kev/kev-0.8b": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
    "d1/d1-3b": "da1fe36a861f24690f27f622dca1d8688503d113",
    "clm/clm-v0.1-8b": "e939398d4556fcd9400c76fa8c5a513202f42b0a",
    "kev/parakh-4b": "f2992d283fd8eac93ea2b15712d23eb8fcc467df",
    "gliner2/gliner2.5-decide": "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
    "gliner2/gliguard-300m": "0e9b87233990dd3ff657ef7a540ee2b62b07c7d6",
    "vllm/shieldstral-1.0-3b": "003ec7e2b0bab5f0e6307edbaf186fa5822b76f5",
    "vllm/scope-guard-4B-q-2601": "29fcf6a99b4487f5cbccfbf506d95d7c1724bc93",
    "vllm/scope-guard-4B-g-2601": "801b1a8a1f07a8606dbba067b0629b9f84f3a931",
    "vllm/scope-guard-v2-4B-q-2609": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "vllm/scope-guard-v2-4B-q-2609-class-only": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "vllm/scope-guard-v2-4B-q-2609-evidences": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "vllm/scope-guard-v2-4B-q-2609-all-fields": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "hf/scope-guard-v2-4B-q-2609": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "orbitals/scope-guard-4B-q-2601": "29fcf6a99b4487f5cbccfbf506d95d7c1724bc93",
    "orbitals/scope-guard-4B-g-2601": "801b1a8a1f07a8606dbba067b0629b9f84f3a931",
    "orbitals/scope-guard-v2-4B-q-2609": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "orbitals/scope-guard-v2-4B-q-2609-class-only": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "orbitals/scope-guard-v2-4B-q-2609-evidences": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "orbitals/scope-guard-v2-4B-q-2609-all-fields": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "orbitals/scope-guard-v2-4B-q-2609-classify": "0110d5d0f50c537513096f597d46ff4a1ee18377",
    "vllm/nemotron-3.5-content-safety": "35645ed3543b7e7ffaed2e788699e57a5051497c",
    "vllm/polyguard-qwen": "644bfe73ff498c9a14818b72a11187eaf23f0ff1",
    "vllm/qwen3-guard-gen-0.6b": "fada3b2f655b89601929198343c94cd2f64d93cc",
    "hf/qwen3-guard-gen-0.6b": "fada3b2f655b89601929198343c94cd2f64d93cc",
    "vllm/qwen3-guard-gen-4b": "6ec42827da0c1ff11e7a49dc269d2e810d27e108",
    "vllm/qwen3-guard-gen-8b": "4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb",
}
CONFIG_PROTOCOLS = {
    # All four GPT trees run one protocol. They differ in the model being evaluated
    # and in the route it is reached over, and the route is a protocol option rather
    # than a protocol of its own.
    "openrouter/gpt-5.6-sol": "gpt",
    "openrouter/gpt-5.6-luna": "gpt",
    "openai/gpt-5.6-luna": "gpt",
    "openai/gpt-5.6-sol": "gpt",
    "openrouter/gpt-5.6-terra": "gpt",
    "openrouter/gpt-6-astra": "gpt",
    "openrouter/gpt-6-luna": "gpt",
    "openrouter/gpt-6.1-sol": "gpt",
    "openai/gpt-5.6-terra": "gpt",
    "openai/gpt-6-astra": "gpt",
    "openai/gpt-6-luna": "gpt",
    "openai/gpt-6.1-sol": "gpt",
    # One claude protocol for every Claude tree, over either route.
    "anthropic/claude-haiku-4.5": "claude",
    "anthropic/claude-sonnet-5": "claude",
    "anthropic/claude-opus-5": "claude",
    "anthropic/claude-sonnet-5.5": "claude",
    "anthropic/claude-opus-5.5": "claude",
    "anthropic/claude-haiku-5.5": "claude",
    "openrouter/claude-haiku-4.5": "claude",
    "openrouter/claude-sonnet-5": "claude",
    "openrouter/claude-opus-5": "claude",
    "openrouter/claude-sonnet-5.5": "claude",
    "openrouter/claude-opus-5.5": "claude",
    "openrouter/claude-haiku-5.5": "claude",
    # One gemini protocol for every Gemini tree, over either route.
    "google/gemini-3.8-flash": "gemini",
    "google/gemini-3.1-pro-preview": "gemini",
    "openrouter/gemini-3.8-flash": "gemini",
    "openrouter/gemini-3.1-pro-preview": "gemini",
    # The same weights as the vllm/nemotron-3.5-content-safety tree, on the `openrouter`
    # route instead of `vllm`.
    "openrouter/nemotron-3.5-content-safety-free": "nemotron-3.5-content-safety",
    # The config tree keeps the model version; the protocol does not carry one, so
    # a later Jev release needs only a new tree here.
    "typesafe/jev-1.13": "jev",
    "clef/clef-flash": "clef",
    "clef/clef-fp8": "clef",
    "d1/d1-3b": "d1",
    # Kev serves Jev's API, so it answers Jev's questions.
    "kev/kev-27b": "jev",
    "kev/kev-27b-fp8": "jev",
    "kev/kev-9b": "jev",
    "kev/kev-4b": "jev",
    "kev/kev-0.8b": "jev",
    # CLM serves the same System One questions.
    "clm/clm-v0.1-8b": "jev",
    "kev/parakh-4b": "jev",
    "vllm/qwen3-guard-gen-0.6b": "qwen3-guard-gen",
    "hf/qwen3-guard-gen-0.6b": "qwen3-guard-gen",
    "vllm/qwen3-guard-gen-4b": "qwen3-guard-gen",
    "vllm/qwen3-guard-gen-8b": "qwen3-guard-gen",
    "vllm/scope-guard-4B-q-2601": "scope-guard-2601",
    "vllm/scope-guard-4B-g-2601": "scope-guard-2601",
    "vllm/scope-guard-v2-4B-q-2609": "scope-guard-2609",
    "vllm/scope-guard-v2-4B-q-2609-class-only": "scope-guard-2609",
    "vllm/scope-guard-v2-4B-q-2609-evidences": "scope-guard-2609",
    "vllm/scope-guard-v2-4B-q-2609-all-fields": "scope-guard-2609",
    "hf/scope-guard-v2-4B-q-2609": "scope-guard-2609",
    "orbitals/scope-guard-4B-q-2601": "scope-guard-2601",
    "orbitals/scope-guard-4B-g-2601": "scope-guard-2601",
    "orbitals/scope-guard-v2-4B-q-2609": "scope-guard-2609",
    "orbitals/scope-guard-v2-4B-q-2609-class-only": "scope-guard-2609",
    "orbitals/scope-guard-v2-4B-q-2609-evidences": "scope-guard-2609",
    "orbitals/scope-guard-v2-4B-q-2609-all-fields": "scope-guard-2609",
    "orbitals/scope-guard-v2-4B-q-2609-classify": "scope-guard-2609",
    "gliner2/gliguard-300m": "gliguard",
}
BENCHMARK_REVISIONS = {
    "harmbench-text-test": "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
    "simple-safety-tests": "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5",
    "aegis-2.0": "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c",
    "openai-moderation": "f4ab51b5edd3bfbcb349a56324274235b674e0e4",
    "toxic-chat-0124": "29df8e4dba60e1f4af4b4075c0705c5b313548a8",
    "wildguardmix-test": "d29c47f41c8b51348b5c8e8c81c039b3132b66d1",
    "cosapien": "29bb85bd4f84319d825517519278c8f8959fa031",
    "dynaguardrail": "3c21998805b692819e527f88702f761e33f6a013",
}
CONFIG_PATHS = sorted(CONFIG_DIR.rglob("*.yaml"))
CONFIGS = {path: yaml.safe_load(path.read_text()) for path in CONFIG_PATHS}

# Each check below walks every configuration and reports all the files that break it in
# one assertion, rather than collecting one test per file: there are hundreds of files,
# and a failure message that names each of them is as precise as a test per file.


def _config_id(config_path: Path) -> str:
    # A configuration is named by its provider directory and its tree, so the same
    # model reached two ways -- `vllm/` and `hf/`, or `openai/` and `openrouter/` --
    # keeps one tree name under each provider.
    return config_path.parent.relative_to(CONFIG_DIR).as_posix()


def _label(config_path: Path) -> str:
    return config_path.relative_to(CONFIG_DIR).as_posix()


def test_run_configs_are_grouped_by_the_provider_they_name() -> None:
    misplaced = [
        _label(path)
        for path, config in CONFIGS.items()
        if ModelName(config["model"]["model"]).api != path.relative_to(CONFIG_DIR).parts[0]
        or not config["model"]["model"].startswith(PROVIDER_PREFIXES)
    ]

    assert not misplaced


def _pin_problems(config_path: Path) -> list[str]:
    config_name = _config_id(config_path)
    protocol = CONFIG_PROTOCOLS.get(config_name, config_path.parent.name)
    config = config_path.read_text()
    problems = []
    if f"protocol: {protocol}" not in config:
        problems.append(f"does not run protocol {protocol}")
    if config_name not in UNPINNED_CONFIGS and CONFIG_REVISIONS.get(config_name, "") not in config:
        problems.append("does not pin its model revision")
    if BENCHMARK_REVISIONS[config_path.stem] not in config:
        problems.append("does not pin its dataset revision")
    # `max_connections` is how many prompts the `hf` provider batches, a hardware setting
    # like `max_model_len`, so it is the one generation field a configuration may set.
    generate_config = set(CONFIGS[config_path].get("generate_config") or {})
    if generate_config - {"max_connections"} or "eval_config" in CONFIGS[config_path]:
        problems.append("duplicates the task's generation config")
    return problems


def test_run_configs_keep_pins_without_duplicating_task_generation_config() -> None:
    problems = {_label(path): _pin_problems(path) for path in CONFIG_PATHS}

    assert not {label: found for label, found in problems.items() if found}


def test_unpinned_configs_say_so_rather_than_implying_a_pin() -> None:
    # An unpinned tree must not carry a `revision` argument or a dated model slug:
    # either would read as a pin the provider does not actually honour, which is
    # worse than no pin at all because nothing would flag the drift.
    unpinned = {
        path: config["model"]
        for path, config in CONFIGS.items()
        if _config_id(path) in UNPINNED_CONFIGS
    }
    implied_pins = [
        _label(path)
        for path, model in unpinned.items()
        if "revision" in model.get("args", {})
        or re.search(r"-\d{4}-?\d{2}-?\d{2}$", model["model"])
    ]

    assert unpinned
    assert not implied_pins


def test_run_config_tree_uses_known_identifiers() -> None:
    pairings = {(_config_id(path), path.stem) for path in CONFIG_PATHS}
    known_configs = CONFIG_REVISIONS.keys() | UNPINNED_CONFIGS

    assert pairings
    assert not CONFIG_REVISIONS.keys() & UNPINNED_CONFIGS
    assert {config for config, _benchmark in pairings} <= known_configs
    assert {benchmark for _protocol, benchmark in pairings} <= BENCHMARK_REVISIONS.keys()
    assert all(
        any(pairing_config == config for pairing_config, _benchmark in pairings)
        for config in known_configs
    )
    assert all(
        any(pairing_benchmark == benchmark for _protocol, pairing_benchmark in pairings)
        for benchmark in BENCHMARK_REVISIONS
    )


def _resolution_problem(config_path: Path) -> str | None:
    args = CONFIGS[config_path]["task"]["args"]
    options = args.get("protocol_options") or {}
    try:
        metadata = resolve_protocol(
            args["protocol"], benchmark=config_path.stem, options=options
        ).metadata
    except Exception as error:
        return repr(error)
    if metadata["protocol"] != args["protocol"]:
        return f"records protocol {metadata['protocol']!r}"
    recorded = {key: metadata["protocol_options"].get(key) for key in options}
    if recorded != options:
        return f"records options {recorded!r}"
    return None


def test_a_run_records_the_protocol_and_options_its_config_sets() -> None:
    # Resolving catches what the text checks above cannot: a routed protocol with no
    # route, or an option its route cannot deliver, such as Nemotron categories over
    # OpenRouter. Either would otherwise surface only when the run starts.
    # `unsafe-eval run --resume` then recognises a finished run of a configuration by
    # finding the configuration's protocol and options in the run's metadata, so every
    # option a configuration sets has to be recorded unchanged.
    problems = {_label(path): _resolution_problem(path) for path in CONFIG_PATHS}

    assert not {label: problem for label, problem in problems.items() if problem}


def test_a_routed_config_agrees_with_the_model_it_names() -> None:
    # A route that disagreed with the model string would send generation parameters
    # the API never accepts, and record them in run metadata as though it had.
    routes = {
        path: options["route"]
        for path, config in CONFIGS.items()
        if "route" in (options := config["task"]["args"].get("protocol_options") or {})
    }
    disagreeing = [
        _label(path)
        for path, route in routes.items()
        if ModelName(CONFIGS[path]["model"]["model"]).api != route_spec(route).model_api
    ]

    assert routes
    assert not disagreeing


GOOGLE_SAFETY_CATEGORIES = {
    "civic_integrity",
    "dangerous_content",
    "harassment",
    "hate_speech",
    "sexually_explicit",
}


def test_a_google_config_states_every_adjustable_safety_filter() -> None:
    # Inspect turns the filters off by default. Stating them keeps a change of that
    # default from changing a run, and puts the thresholds in the run log.
    google = {
        path: config["model"]
        for path, config in CONFIGS.items()
        if _config_id(path).startswith("google/")
    }
    unstated = [
        _label(path)
        for path, model in google.items()
        if model["args"]["safety_settings"] != dict.fromkeys(GOOGLE_SAFETY_CATEGORIES, "none")
    ]

    assert google
    assert not unstated


GEMINI_TREES = ("gemini-3.8-flash", "gemini-3.1-pro-preview")


def test_both_gemini_routes_reach_vertex_in_the_same_region() -> None:
    # Google AI Studio blocks far more of the unsafe-only prompts than Vertex AI does, so
    # a route that reached it would report a different refusal rate for the same model.
    for tree in GEMINI_TREES:
        for google_path in sorted((CONFIG_DIR / "google" / tree).glob("*.yaml")):
            google = CONFIGS[google_path]["model"]
            openrouter = CONFIGS[CONFIG_DIR / "openrouter" / tree / google_path.name]["model"]

            assert google["model"].startswith("google/vertex/")
            assert google["args"]["location"] == "global"
            assert openrouter["args"]["provider"]["order"] == ["google-vertex/global"]
            assert openrouter["args"]["provider"]["allow_fallbacks"] is False
