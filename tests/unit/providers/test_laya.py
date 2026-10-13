"""Translation and budget tests for the Laya provider.

The stub router, agent, tokenizer, and `laya.common` stand in for the `laya` package so the
offline suite never imports torch. The stub tokenizer counts whitespace-separated
words, which keeps the expected token counts readable in each test.
"""

import asyncio
import itertools
import json
import sys
import types
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from unsafe_eval.providers.laya import (
    LayaAPI,
    check_question,
    model_output_from_response,
)

MODEL = "convaiinnovations/laya"
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
STATE = {"message": "prompt-sentinel"}


class StubTokenizer:
    mask_token = "[MASK]"

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def question(
    instructions: Any = "is it unsafe",
    criteria: Any = None,
    type: str = "choice",
) -> dict[str, Any]:
    return {
        "safety": {
            "type": type,
            "instructions": instructions,
            "criteria": criteria if criteria is not None else {"unsafe": "bad", "safe": "fine"},
        }
    }


def words(tokenizer: StubTokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def render_options(q: dict[str, Any]) -> list[str]:
    def render(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value)

    return [
        key if value in (None, "") else f"{key}: {render(value)}"
        for key, value in q["crit"].items()
    ]


def build_sequence(
    tok: StubTokenizer, state: Any, q: dict[str, Any], max_len: int, head_max_len: int
) -> tuple[list[int], list[int]]:
    """`laya.common.build_sequence` without its option-shrinking rule; 0 marks a special."""
    options = [[0, *words(tok, " " + option)[:48]] for option in render_options(q)]
    head = words(tok, f"{q['t']} question: {q['ins']}")[: head_max_len - sum(map(len, options))]
    ids = [0, *head, 0, *itertools.chain(*options), 0]
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    ids += [*words(tok, text)[: max(0, max_len - len(ids) - 1)], 0]
    return ids[:max_len], []


@pytest.fixture(autouse=True)
def laya_common(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(build_sequence=build_sequence, render_options=render_options)
    monkeypatch.setitem(sys.modules, "laya.common", module)


class StubAgent:
    def __init__(self) -> None:
        self.tok = StubTokenizer()
        self.cfg: dict[str, Any] = {"max_len": 512, "head_max_len": 192}


def choice_response(**answer: Any) -> dict[str, Any]:
    return {
        "model": "laya-rl-agent",
        "answers": {
            "safety": {
                "type": "choice",
                "choice": "unsafe",
                "probabilities": {"unsafe": 0.8, "safe": 0.2},
                "confidence": 0.28,
                **answer,
            }
        },
        "usage": {"input_tokens": 40, "output_tokens": 0, "truncated": False},
    }


ROUTING = {"model": "multilingual", "repo": f"{MODEL}/multilingual", "reason": "stub"}


class StubRouter:
    """`laya.Router` routing every state to `multilingual` and choosing its message."""

    def __init__(self) -> None:
        self.agents = {"english": StubAgent(), "multilingual": StubAgent()}
        self.calls: list[list[dict[str, Any]]] = []

    def load(self, name: str) -> StubAgent:
        return self.agents[name]

    def predict_batch(self, requests: list[dict[str, Any]], batch_size: int) -> list[Any]:
        self.calls.append(requests)
        return [
            {**choice_response(choice=request["state"]["message"]), "routing": ROUTING}
            for request in requests
        ]


def api(router: StubRouter | None = None, **model_args: Any) -> LayaAPI:
    args: dict[str, Any] = {"revision": REVISION, "max_len": 1024, **model_args}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LayaAPI, "_load", staticmethod(lambda *_: router or StubRouter()))
        return LayaAPI(MODEL, **args)


def conversation(questions: dict[str, Any], state: Any = STATE) -> list[Any]:
    return [
        ChatMessageSystem(content=json.dumps(questions)),
        ChatMessageUser(content="prompt-sentinel", metadata={"state": state}),
    ]


def generate(model: LayaAPI, messages: list[Any]) -> Any:
    output, _ = asyncio.run(model.generate(messages, [], "none", GenerateConfig()))
    return output


# --- model arguments ---------------------------------------------------------


@pytest.mark.parametrize("revision", [None, "main", REVISION[:7]])
def test_a_missing_or_moving_revision_is_refused(revision: Any) -> None:
    with pytest.raises(ValueError, match="40-character commit hash"):
        api(revision=revision)


@pytest.mark.parametrize("max_len", [None, 511])
def test_max_len_must_be_an_integer_within_the_encoder_limits(max_len: Any) -> None:
    with pytest.raises(ValueError, match="max_len"):
        api(max_len=max_len)


def test_an_unknown_model_argument_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported Laya model arguments"):
        api(temperature=0.0)


def test_the_router_preloads_both_checkpoints_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routers: list[Any] = []

    class Router:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            routers.append(self)

        def preload(self, names: list[str]) -> "Router":
            self.preloaded = names
            return self

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Router=Router))

    LayaAPI(MODEL, revision=REVISION, max_len=1024, device="cuda")

    (router,) = routers
    assert router.kwargs == {
        "models": {"english": (MODEL, None), "multilingual": (MODEL, "multilingual")},
        "revision": REVISION,
        "device": "cuda",
    }
    assert router.preloaded == ["english", "multilingual"]


def test_a_missing_laya_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "laya", None)

    with pytest.raises(ImportError, match="uv sync --extra laya"):
        LayaAPI(MODEL, revision=REVISION, max_len=1024)


# --- question check ----------------------------------------------------------


def test_an_option_past_the_option_limit_fails_the_run() -> None:
    criteria = {"unsafe": " ".join(["word"] * 48), "safe": "fine"}

    with pytest.raises(ValueError, match="cut this question"):
        check_question(StubTokenizer(), 192, question(criteria=criteria))


def test_structured_instructions_reach_laya_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []

    def spy(tok: Any, state: Any, q: dict[str, Any], *limits: int) -> Any:
        built.append(q)
        return build_sequence(tok, state, q, *limits)

    monkeypatch.setattr(sys.modules["laya.common"], "build_sequence", spy)
    criteria = {"unsafe": {"what": "bad"}, "safe": None}
    questions = question(instructions={"question": "is it unsafe"}, criteria=criteria)

    check_question(StubTokenizer(), 192, questions)

    assert built[0] == {"t": "choice", "ins": '{"question": "is it unsafe"}', "crit": criteria}


def test_a_question_that_is_not_a_choice_is_refused() -> None:
    with pytest.raises(ValueError, match="Choice question"):
        check_question(StubTokenizer(), 192, question(type="noul"))


# --- response translation ----------------------------------------------------


def translate(response: dict[str, Any]) -> Any:
    return model_output_from_response(response, model_name=MODEL, revision=REVISION)


def test_the_chosen_option_becomes_the_completion_with_its_metadata() -> None:
    routing = {"model": "english", "repo": MODEL, "reason": "English Latin text"}
    response = choice_response()
    response["usage"]["truncated"] = True
    output = translate({**response, "routing": routing})

    assert output.completion == "unsafe"
    assert output.model == MODEL
    assert output.metadata == {
        "served_model": f"{MODEL}@{REVISION}",
        "probabilities": {"unsafe": 0.8, "safe": 0.2},
        "confidence": 0.28,
        "sequence_tokens": 40,
        "state_truncated": True,
        "routing": routing,
    }
    assert output.usage.input_tokens == 40
    assert output.usage.total_tokens == 40


def test_a_response_without_one_answer_is_refused() -> None:
    with pytest.raises(ValueError, match="no single answer"):
        translate({"answers": None})


def test_a_non_choice_answer_is_refused() -> None:
    with pytest.raises(ValueError, match="Choice question"):
        translate({"answers": {"safety": {"type": "noul", "noul": 0.7}}})


# --- generate ----------------------------------------------------------------


def test_generate_routes_the_questions_and_state_at_the_run_max_len() -> None:
    router = StubRouter()
    questions = question()

    output = generate(api(router), conversation(questions))

    assert router.calls == [[{"state": STATE, "questions": questions, "max_len": 1024}]]
    assert output.completion == "prompt-sentinel"
    assert output.metadata["served_model"] == f"{MODEL}/multilingual@{REVISION}"


def test_requests_that_queue_while_the_device_is_busy_share_a_batch() -> None:
    router = StubRouter()
    model = api(router)
    messages = [f"prompt-{i}" for i in range(4)]

    async def generate_all() -> list[Any]:
        return await asyncio.gather(
            *(
                model.generate(
                    conversation(question(), {"message": m}), [], "none", GenerateConfig()
                )
                for m in messages
            )
        )

    outputs = asyncio.run(generate_all())

    assert [len(call) for call in router.calls] == [4]
    assert [output.completion for output, _ in outputs] == messages


def test_generate_refuses_more_than_one_question() -> None:
    router = StubRouter()
    questions = {**question(), "other": question()["safety"]}

    with pytest.raises(ValueError, match="exactly one question"):
        generate(api(router), conversation(questions))
    assert router.calls == []


def test_a_question_laya_would_cut_never_reaches_the_model() -> None:
    router = StubRouter()

    with pytest.raises(ValueError, match="cut this question"):
        generate(api(router), conversation(question(instructions="word " * 200)))
    assert router.calls == []


def test_inspect_resolves_the_laya_prefix_to_this_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LayaAPI, "_load", staticmethod(lambda *_: StubRouter()))
    model = get_model(f"laya/{MODEL}", memoize=False, revision=REVISION, max_len=1024)

    assert isinstance(model.api, LayaAPI)
    assert model.api.model_name == MODEL
