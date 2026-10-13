from pathlib import Path

import pytest

from unsafe_eval._resources import content_sha256, load_text_resource
from unsafe_eval.protocols._resource_manifest import (
    _PROMPT_RESOURCE_MANIFEST,
    PromptResource,
    adaptation_resource,
    prompt_resources,
    register_prompt_resources,
)

PROMPT_ROOT = Path(__file__).parents[3] / "src" / "unsafe_eval" / "protocols" / "prompts"


def test_manifest_matches_the_complete_packaged_resource_tree() -> None:
    declared_paths = {
        Path(benchmark, protocol, resource.filename)
        for (benchmark, protocol), resources in _PROMPT_RESOURCE_MANIFEST.items()
        for resource in resources
    }
    packaged_paths = {
        path.relative_to(PROMPT_ROOT) for path in PROMPT_ROOT.rglob("*") if path.is_file()
    }

    assert declared_paths == packaged_paths


def test_manifest_hashes_and_empty_content_rules_match_resources() -> None:
    for (benchmark, protocol), resources in _PROMPT_RESOURCE_MANIFEST.items():
        for resource in resources:
            path = resource.path or (
                "protocols",
                "prompts",
                benchmark,
                protocol,
                resource.filename,
            )
            content = load_text_resource(*path, package=resource.package)

            assert content or resource.allow_empty
            assert content_sha256(content) == resource.sha256


def test_pairing_without_resources_is_rejected() -> None:
    with pytest.raises(ValueError, match="No prompt resources declared for pairing"):
        prompt_resources("aegis-2.0", "nemotron-3.5-content-safety")
    with pytest.raises(ValueError, match="No prompt resources declared for pairing"):
        prompt_resources("unknown", "shieldstral-1.0-3b")


def test_undeclared_resource_is_rejected() -> None:
    with pytest.raises(ValueError, match="Undeclared prompt resource"):
        adaptation_resource("aegis-2.0", "shieldstral-1.0-3b", "undeclared.txt")


def test_resource_hash_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unsafe_eval.protocols._resource_manifest.load_text_resource",
        lambda *_parts, **_kwargs: "changed content",
    )

    with pytest.raises(ValueError, match="Prompt resource hash mismatch"):
        adaptation_resource("aegis-2.0", "shieldstral-1.0-3b", "instruction.txt")


def test_unexpected_empty_resource_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unsafe_eval.protocols._resource_manifest.load_text_resource",
        lambda *_parts, **_kwargs: "",
    )

    with pytest.raises(ValueError, match="Prompt resource must not be empty"):
        adaptation_resource("aegis-2.0", "shieldstral-1.0-3b", "instruction.txt")


@pytest.mark.parametrize(
    "resource",
    [
        PromptResource(filename="", sha256="0" * 64),
        PromptResource(filename="prompt.txt", sha256="invalid"),
        PromptResource(filename="prompt.txt", sha256="0" * 64, package=""),
        PromptResource(filename="prompt.txt", sha256="0" * 64, path=("..",)),
    ],
)
def test_resource_registration_rejects_invalid_declarations(
    resource: PromptResource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unsafe_eval.protocols._resource_manifest._PROMPT_RESOURCE_MANIFEST",
        {},
    )

    with pytest.raises(ValueError, match="Prompt resource"):
        register_prompt_resources("test/benchmark", "test/protocol", (resource,))


def test_resource_registration_rejects_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unsafe_eval.protocols._resource_manifest._PROMPT_RESOURCE_MANIFEST",
        {},
    )
    resource = PromptResource(filename="prompt.txt", sha256="0" * 64)
    register_prompt_resources("test/benchmark", "test/protocol", (resource,))

    with pytest.raises(ValueError, match="already registered"):
        register_prompt_resources("test/benchmark", "test/protocol", (resource,))
