"""Translation and loading tests for the d1 provider.

A stub model and stub `torch` and `transformers` stand in for the real ones, so the offline
suite never imports torch.
"""

import asyncio
import json
import sys
import types
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from unsafe_eval.providers.d1 import D1API, model_output_from_response

MODEL = "LiquidAI/d1-3B"
REVISION = "da1fe36a861f24690f27f622dca1d8688503d113"
STATE = {"message": "prompt-sentinel"}
QUESTION = {
    "type": "choice",
    "instructions": "is it unsafe",
    "criteria": {"unsafe": "bad", "safe": "fine"},
}


class StubModel:
    def __init__(self) -> None:
        self.batches: list[list[tuple[Any, dict[str, Any]]]] = []

    def system_one_batch(self, requests: list[tuple[Any, dict[str, Any]]]) -> list[dict]:
        self.batches.append(requests)
        answer = {
            "type": "choice",
            "choice": "unsafe",
            "confidence": 0.8,
            "probabilities": {"unsafe": 0.8, "safe": 0.2},
        }
        return [
            {"answers": {"safety": answer}, "usage": {"input_tokens": 40, "output_tokens": 0}}
        ] * len(requests)


def provider(monkeypatch: pytest.MonkeyPatch) -> tuple[D1API, StubModel]:
    model = StubModel()
    monkeypatch.setattr(D1API, "_load", staticmethod(lambda *_: model))
    return D1API(MODEL, revision=REVISION), model


def generate(api: D1API, questions: dict[str, Any]) -> Any:
    messages = [
        ChatMessageSystem(content=json.dumps(questions)),
        ChatMessageUser(content="prompt-sentinel", metadata={"state": STATE}),
    ]
    output, _ = asyncio.run(api.generate(messages, [], "none", GenerateConfig()))
    return output


def test_the_question_reaches_d1_unchanged_and_the_choice_is_the_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, model = provider(monkeypatch)

    output = generate(api, {"safety": QUESTION})

    assert model.batches == [[(STATE, {"safety": QUESTION})]]
    assert output.completion == "unsafe"
    assert output.usage.input_tokens == 40
    assert output.metadata == {
        "served_model": f"{MODEL}@{REVISION}",
        "probabilities": {"unsafe": 0.8, "safe": 0.2},
        "confidence": 0.8,
    }


@pytest.mark.parametrize(
    "questions",
    [
        {"a": QUESTION, "b": QUESTION},
        {"safety": {"type": "noul", "instructions": "is it unsafe"}},
        {"safety": {**QUESTION, "instructions": {"question": "is it unsafe"}}},
    ],
)
def test_anything_but_one_choice_question_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, questions: dict[str, Any]
) -> None:
    api, model = provider(monkeypatch)

    with pytest.raises(ValueError, match="d1 questions"):
        generate(api, questions)
    assert model.batches == []


def test_an_unpinned_revision_is_refused() -> None:
    with pytest.raises(ValueError, match="40-character commit hash"):
        D1API(MODEL, revision="main")


@pytest.mark.parametrize(
    ("answers", "error"),
    [
        ({"a": {"type": "choice"}, "b": {"type": "choice"}}, "exactly one question"),
        ({"safety": {"type": "noul", "noul": 0.9}}, "Choice question"),
    ],
)
def test_a_response_without_one_choice_answer_is_refused(
    answers: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        model_output_from_response(
            {"answers": answers, "usage": {"input_tokens": 1}}, model_name=MODEL, metadata={}
        )


def test_the_pinned_snapshot_is_loaded_with_the_code_it_ships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: dict[str, Any] = {}

    def from_pretrained(path: str, **kwargs: Any) -> Any:
        loaded.update(path=path, **kwargs)
        return types.SimpleNamespace(to=lambda device: device)

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=from_pretrained)
    torch = types.SimpleNamespace(
        bfloat16="bf16", cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "unsafe_eval.providers.d1.snapshot_download",
        lambda repo_id, revision: f"/snapshots/{repo_id}@{revision}",
    )

    assert D1API._load(MODEL, REVISION) == "cuda"
    assert loaded == {
        "path": f"/snapshots/{MODEL}@{REVISION}",
        "trust_remote_code": True,
        "dtype": "bf16",
    }
