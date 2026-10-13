"""Tests for the CLM provider.

The encoder server, the heads download and the `contrastive-lm` engine are stubbed, so no
test loads a model or reaches the network.
"""

import asyncio
import json
import sys
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from unsafe_eval.providers.clm import CLMAPI, clm

MODEL = "Contrastive-LM/CLM-v0.1-8B"
REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QUESTIONS = {"safety": {"type": "choice", "criteria": {"safe": "Fine.", "unsafe": "Harmful."}}}


def choice_response() -> dict[str, Any]:
    return {
        "model": "clm-latest",
        "answers": {
            "safety": {
                "type": "choice",
                "choice": "unsafe",
                "confidence": 0.6,
                "probabilities": {"safe": 0.2, "unsafe": 0.8},
            }
        },
        "usage": {"billing_units": 1, "input_tokens": 37, "output_tokens": 0},
    }


class StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def answer(self, state: Any, questions: Any) -> dict[str, Any]:
        self.calls.append((state, questions))
        return choice_response()


def stub_load(monkeypatch: pytest.MonkeyPatch) -> tuple[StubEngine, list[tuple[str, ...]]]:
    engine = StubEngine()
    loads: list[tuple[str, ...]] = []

    def load(model_name: str, revision: str, encoder_revision: str) -> StubEngine:
        loads.append((model_name, revision, encoder_revision))
        return engine

    monkeypatch.setattr(CLMAPI, "_load", staticmethod(load))
    return engine, loads


def test_generate_asks_the_engine_and_records_both_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, loads = stub_load(monkeypatch)
    api = CLMAPI(MODEL, revision=REVISION, encoder_revision=ENCODER_REVISION)
    state = {"message": "prompt-sentinel"}
    messages = [
        ChatMessageSystem(content=json.dumps(QUESTIONS)),
        ChatMessageUser(content="prompt-sentinel", metadata={"state": state}),
    ]

    output, call = asyncio.run(api.generate(messages, [], "none", GenerateConfig()))

    assert loads == [(MODEL, REVISION, ENCODER_REVISION)]
    assert engine.calls == [(state, QUESTIONS)]
    assert output.completion == "unsafe"
    assert output.metadata is not None
    assert output.metadata["served_model"] == f"{MODEL}@{REVISION}"
    assert output.metadata["encoder"] == f"Qwen/Qwen3-8B@{ENCODER_REVISION}"
    assert call.time is not None


@pytest.mark.parametrize(
    "args",
    [
        {"encoder_revision": ENCODER_REVISION},
        {"revision": "main", "encoder_revision": ENCODER_REVISION},
        {"revision": REVISION},
    ],
)
def test_both_revisions_must_be_pinned(
    monkeypatch: pytest.MonkeyPatch, args: dict[str, str]
) -> None:
    stub_load(monkeypatch)

    with pytest.raises(ValueError, match="40-character commit hash"):
        CLMAPI(MODEL, **args)


def test_a_missing_package_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "clm", None)

    with pytest.raises(ImportError, match="uv sync --extra clm"):
        CLMAPI(MODEL, revision=REVISION, encoder_revision=ENCODER_REVISION)


def test_the_registered_factory_builds_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_load(monkeypatch)

    api = clm(MODEL, revision=REVISION, encoder_revision=ENCODER_REVISION)

    assert isinstance(api, CLMAPI)
    assert api.model_name == MODEL
