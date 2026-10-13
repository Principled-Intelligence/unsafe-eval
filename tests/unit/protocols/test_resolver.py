import pytest

from unsafe_eval._resources import content_sha256
from unsafe_eval.protocols import (
    BenchmarkProtocolSpec,
    PromptResource,
    _benchmark_specs,
    _resolver,
    _resource_manifest,
    load_protocol_extensions,
    register_benchmark,
    register_prompt_resources,
    register_protocol,
    registered_benchmarks,
    registered_protocol_components,
    registered_protocols,
    resolve_protocol,
)
from unsafe_eval.protocols.nemotron import build_nemotron_protocol


def test_protocol_rejects_options_owned_by_other_models() -> None:
    with pytest.raises(ValueError, match="Unsupported options"):
        resolve_protocol(
            "shieldstral-1.0-3b",
            benchmark="aegis-2.0",
            options={"include_evidence": True},
        )


@pytest.mark.parametrize("value", ["0.5", True])
def test_shieldstral_threshold_requires_a_number(value: object) -> None:
    with pytest.raises(ValueError, match="must be a number"):
        resolve_protocol(
            "shieldstral-1.0-3b",
            benchmark="aegis-2.0",
            options={"threshold": value},
        )


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_shieldstral_threshold_has_probability_bounds(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_protocol(
            "shieldstral-1.0-3b",
            benchmark="aegis-2.0",
            options={"threshold": value},
        )


def test_boolean_protocol_options_are_strict() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_protocol(
            "scope-guard-2601",
            benchmark="aegis-2.0",
            options={"validate_prompt_resource_hashes": 1},
        )


def test_protocol_metadata_records_resolved_options_and_resource_hashes() -> None:
    selected = resolve_protocol(
        "shieldstral-1.0-3b",
        benchmark="aegis-2.0",
        options={"threshold": 1},
    )

    assert selected.metadata["protocol_options"] == {
        "threshold": 1.0,
        "validate_prompt_resource_hashes": True,
    }
    hashes = selected.metadata["prompt_resource_hashes"]
    assert set(hashes) == {
        "profile/system.txt",
        "profile/user.txt",
        "protocol/instruction.txt",
        "protocol/query.txt",
    }
    assert all(len(digest) == 64 for digest in hashes.values())


def test_scope_guard_2609_defaults_are_the_released_model_settings() -> None:
    selected = resolve_protocol("scope-guard-2609", benchmark="aegis-2.0")

    assert selected.metadata["protocol_options"] == {
        "output_fields": ["reasoning", "scope_class"],
        "unsafe_axis": "pass",
        "validate_prompt_resource_hashes": True,
    }


def test_disabling_validation_applies_to_every_protocol_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_content = "changed adaptation"
    monkeypatch.setattr(
        "unsafe_eval.protocols._resource_manifest.load_text_resource",
        lambda *_parts, **_kwargs: changed_content,
    )

    with pytest.warns(UserWarning, match="hash validation is disabled") as warnings:
        selected = resolve_protocol(
            "shieldstral-1.0-3b",
            benchmark="aegis-2.0",
            options={"validate_prompt_resource_hashes": False},
        )

    assert len(warnings) == 2
    hashes = selected.metadata["prompt_resource_hashes"]
    assert hashes["protocol/instruction.txt"] == content_sha256(changed_content)
    assert hashes["protocol/query.txt"] == content_sha256(changed_content)


def test_qwen3_guard_records_default_mapping_and_has_no_prompt_resources() -> None:
    selected = resolve_protocol("qwen3-guard-gen", benchmark="aegis-2.0")

    assert selected.metadata["protocol_options"] == {
        "controversial_as_unsafe": True,
        "validate_prompt_resource_hashes": True,
    }
    assert selected.metadata["prompt_resource_hashes"] == {}


def test_polyguard_records_prompt_resource_hashes() -> None:
    selected = resolve_protocol("polyguard-qwen", benchmark="aegis-2.0")

    assert selected.metadata["protocol_options"] == {
        "validate_prompt_resource_hashes": True,
    }
    hashes = selected.metadata["prompt_resource_hashes"]
    assert set(hashes) == {"profile/system.txt", "profile/user.txt"}
    assert all(len(digest) == 64 for digest in hashes.values())


@pytest.mark.parametrize("protocol", ["polyguard-qwen", "qwen3-guard-gen"])
def test_fixed_policy_protocols_reject_policy_conditioned_benchmarks(protocol: str) -> None:
    with pytest.raises(ValueError, match="does not support policy-conditioned"):
        resolve_protocol(protocol, benchmark="cosapien")


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported safety protocol"):
        resolve_protocol("unknown", benchmark="aegis-2.0")


def test_benchmark_alias_can_reuse_a_hashed_builtin_prompt_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _benchmark_specs,
        "_BENCHMARK_PROTOCOL_SPECS",
        dict(registered_benchmarks()),
    )
    monkeypatch.setattr(
        _resource_manifest,
        "_PROMPT_RESOURCE_MANIFEST",
        dict(_resource_manifest._PROMPT_RESOURCE_MANIFEST),
    )
    benchmark = "test/simple-safety-tests-copy"
    protocol = "shieldstral-1.0-3b"
    register_benchmark(
        benchmark,
        BenchmarkProtocolSpec(
            mixed_labels=False,
            primary_metric="unsafe_recall",
            unsafe_group_by=("harm_area", "category"),
        ),
    )
    register_prompt_resources(
        benchmark,
        protocol,
        (
            PromptResource(
                filename="instruction.txt",
                sha256=("2b0637733291f37bc93dbb7054e6ee2ecf2af0b4b0e2481d85b87170e5954f91"),
                path=(
                    "protocols",
                    "prompts",
                    "simple-safety-tests",
                    protocol,
                    "instruction.txt",
                ),
            ),
            PromptResource(
                filename="query.txt",
                sha256=("b8df2fc50b37e7522c8a2386c829a41f0167419cf215fb696b4f88ea59e78367"),
                path=(
                    "protocols",
                    "prompts",
                    "simple-safety-tests",
                    protocol,
                    "query.txt",
                ),
            ),
        ),
    )

    selected = resolve_protocol(protocol, benchmark=benchmark)

    assert selected.metadata["protocol"] == protocol
    assert set(selected.metadata["prompt_resource_hashes"]) >= {
        "protocol/instruction.txt",
        "protocol/query.txt",
    }
    primary_metric = selected.metrics["label"][0]
    assert primary_metric.__registry_info__.name == "unsafe_eval/unsafe_recall"


def test_protocol_registration_owns_public_inspect_components() -> None:
    def custom_component() -> object:
        return object()

    register_protocol(
        "test/component-protocol",
        build_nemotron_protocol,
        components=(custom_component,),
    )

    assert custom_component in registered_protocol_components()


def test_protocol_registration_rejects_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        register_protocol("", build_nemotron_protocol)
    with pytest.raises(ValueError, match="already registered"):
        register_protocol("scope-guard-2601", build_nemotron_protocol)


def test_protocol_registration_rejects_duplicate_component_names() -> None:
    def shieldstral_prompt() -> object:
        return object()

    with pytest.raises(ValueError, match="component names must be unique"):
        register_protocol(
            "test/duplicate-component",
            build_nemotron_protocol,
            components=(shieldstral_prompt,),
        )


def test_installed_protocol_extensions_are_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extension() -> None:
        register_protocol("test/discovered-protocol", build_nemotron_protocol)

    class EntryPoint:
        name = "test-extension"

        @staticmethod
        def load() -> object:
            return extension

    monkeypatch.setattr(_resolver, "entry_points", lambda **_kwargs: [EntryPoint()])

    load_protocol_extensions()

    assert registered_protocols()["test/discovered-protocol"] is build_nemotron_protocol


def test_discovered_protocol_extension_must_be_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EntryPoint:
        name = "invalid-extension"

        @staticmethod
        def load() -> object:
            return object()

    monkeypatch.setattr(_resolver, "entry_points", lambda **_kwargs: [EntryPoint()])

    with pytest.raises(TypeError, match="not callable"):
        load_protocol_extensions()


def test_gpt_rejects_an_unsupported_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="'reasoning_effort' must be one of"):
        resolve_protocol(
            "gpt",
            benchmark="aegis-2.0",
            options={"route": "openrouter", "reasoning_effort": "LOW"},
        )


def test_gpt_rejects_a_route_it_does_not_answer_on() -> None:
    with pytest.raises(ValueError, match="'route' must be one of"):
        resolve_protocol("gpt", benchmark="aegis-2.0", options={"route": "anthropic"})


def test_gpt_records_the_policy_hash() -> None:
    protocol = resolve_protocol("gpt", benchmark="aegis-2.0", options={"route": "openrouter"})
    hashes = protocol.metadata["prompt_resource_hashes"]

    assert set(hashes) == {
        "profile/system.txt",
        "profile/user.txt",
        "protocol/instruction.txt",
    }
    assert all(len(digest) == 64 for digest in hashes.values())


def test_gpt_policy_conditioned_instructions_template_the_sample_policy() -> None:
    from unsafe_eval.protocols._resource_manifest import adaptation_resource

    # The per-sample policy is the whole definition of unsafe for these benchmarks,
    # so the packaged text must be a template around it rather than a finished one.
    offenders = [
        benchmark
        for benchmark in ("cosapien", "dynaguardrail")
        if "{safety_policy}" not in adaptation_resource(benchmark, "gpt", "instruction.txt")[0]
    ]

    assert not offenders
