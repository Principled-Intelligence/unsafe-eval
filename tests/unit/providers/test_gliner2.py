"""Translation tests for the GLiNER2 provider.

Stubs of `gliner2.classification` and `torch` stand in for the packages, so the
offline suite never imports torch. The stub's scores are already probabilities.
"""

import asyncio
import json
import sys
import types
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from unsafe_eval.providers.gliner2 import GLiNER2API

MODEL = "fastino/GLiNER2.5-Decide"
REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
SCHEMA = {"tasks": {"safety": {"labels": {"unsafe": "bad", "safe": "fine"}}}}


class StubScores:
    def __init__(self, tasks: dict[str, dict[str, float]]) -> None:
        self.tasks = tasks

    def probability(self, task: str, label: str) -> float:
        return self.tasks[task][label]


class OutOfMemory(Exception):
    pass


class StubClassifier:
    def __init__(self, tasks: dict[str, dict[str, float]] | None = None) -> None:
        self.tasks = tasks if tasks is not None else {"safety": {"unsafe": 0.8, "safe": 0.2}}
        self.calls: list[list[tuple[str, Any]]] = []
        self.scorer = types.SimpleNamespace(batch_score=self.batch_score)

    def compile_schema(self, schema: Any) -> Any:
        return schema

    def batch_score(self, texts: Any, schemas: Any, batch_size: int) -> list[StubScores]:
        self.calls.append(list(zip(texts, schemas, strict=True)))
        return [StubScores(self.tasks) for _ in texts]


@pytest.fixture(autouse=True)
def gliner2_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    schema_type = types.SimpleNamespace(from_dict=lambda data: data)
    module = types.SimpleNamespace(ClassificationSchema=schema_type)
    monkeypatch.setitem(sys.modules, "gliner2.classification", module)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(OutOfMemoryError=OutOfMemory))


def api(classifier: StubClassifier | None = None, **model_args: Any) -> GLiNER2API:
    args: dict[str, Any] = {"revision": REVISION, **model_args}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GLiNER2API, "_load", staticmethod(lambda *_: classifier or StubClassifier()))
        return GLiNER2API(MODEL, **args)


def conversation(schema: Any = SCHEMA, text: str = "prompt-sentinel") -> list[Any]:
    content = schema if isinstance(schema, str) else json.dumps(schema)
    return [ChatMessageSystem(content=content), ChatMessageUser(content=text)]


def generate(model: GLiNER2API, messages: list[Any]) -> Any:
    output, _ = asyncio.run(model.generate(messages, [], "none", GenerateConfig()))
    return output


@pytest.mark.parametrize("revision", [None, "main", REVISION[:7]])
def test_a_missing_or_moving_revision_is_refused(revision: Any) -> None:
    with pytest.raises(ValueError, match="40-character commit hash"):
        api(revision=revision)


def test_an_unknown_model_argument_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported GLiNER2 model arguments"):
        api(max_len=512)


def test_the_checkpoint_is_loaded_from_its_pinned_snapshot_onto_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[dict[str, Any]] = []
    loads: list[str] = []
    moves: list[str | None] = []

    def snapshot_download(repo_id: str, **kwargs: Any) -> str:
        downloads.append({"repo_id": repo_id, **kwargs})
        return "/snapshot"

    def to(device: str | None = None) -> StubClassifier:
        moves.append(device)
        return StubClassifier()

    def from_pretrained(path: str, **kwargs: Any) -> Any:
        loads.append(path)
        return types.SimpleNamespace(to=to)

    monkeypatch.setattr("unsafe_eval.providers.gliner2.snapshot_download", snapshot_download)
    classifier = types.SimpleNamespace(from_pretrained=from_pretrained)
    module = sys.modules["gliner2.classification"]
    monkeypatch.setattr(module, "Classifier", classifier, raising=False)

    GLiNER2API(MODEL, revision=REVISION)

    assert downloads == [{"repo_id": MODEL, "revision": REVISION}]
    assert loads == ["/snapshot"]
    assert moves == ["cuda"]


def test_a_missing_gliner_extra_fails_before_any_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[str] = []
    monkeypatch.setattr(
        "unsafe_eval.providers.gliner2.snapshot_download",
        lambda repo_id, **_: downloads.append(repo_id),
    )
    monkeypatch.setitem(sys.modules, "gliner2.classification", None)

    with pytest.raises(ImportError, match="uv sync --extra gliner"):
        GLiNER2API(MODEL, revision=REVISION)

    assert downloads == []


def test_the_most_probable_label_is_the_completion_with_every_probability() -> None:
    classifier = StubClassifier()

    output = generate(api(classifier), conversation())

    assert classifier.calls == [[("prompt-sentinel", SCHEMA)]]
    assert output.completion == "unsafe"
    assert output.model == MODEL
    assert output.metadata == {
        "served_model": f"{MODEL}@{REVISION}",
        "probabilities": {"unsafe": 0.8, "safe": 0.2},
    }


def test_requests_that_queue_while_the_device_is_busy_share_a_batch() -> None:
    classifier = StubClassifier()
    model = api(classifier)
    requests = [
        (f"prompt-{i}", {"tasks": {"safety": {**SCHEMA["tasks"]["safety"], "instruction": f"{i}"}}})
        for i in range(3)
    ]

    async def generate_all() -> list[Any]:
        return await asyncio.gather(
            *(
                model.generate(conversation(schema, text), [], "none", GenerateConfig())
                for text, schema in requests
            )
        )

    outputs = asyncio.run(generate_all())

    assert classifier.calls == [requests]
    assert [output.completion for output, _ in outputs] == ["unsafe"] * 3


class OneTextClassifier(StubClassifier):
    """Runs out of GPU memory on more than one text, and scores a text by its length."""

    def batch_score(self, texts: Any, schemas: Any, batch_size: int) -> list[StubScores]:
        if len(texts) > 1:
            raise OutOfMemory
        self.calls.append(list(zip(texts, schemas, strict=True)))
        return [StubScores({"safety": {"unsafe": len(t) / 10, "safe": 0.0}}) for t in texts]


def test_a_batch_runs_shortest_first_and_splits_until_it_fits_the_gpu() -> None:
    classifier = OneTextClassifier()
    model = api(classifier)
    texts = ["aaa", "a", "aa"]

    async def generate_all() -> list[Any]:
        return await asyncio.gather(
            *(model.generate(conversation(text=t), [], "none", GenerateConfig()) for t in texts)
        )

    outputs = asyncio.run(generate_all())

    assert classifier.calls == [[("a", SCHEMA)], [("aa", SCHEMA)], [("aaa", SCHEMA)]]
    assert [output.metadata["probabilities"]["unsafe"] for output, _ in outputs] == [0.3, 0.1, 0.2]


@pytest.mark.parametrize(
    "messages",
    [
        [ChatMessageUser(content="prompt-sentinel")],
        conversation("not json"),
        conversation({"tasks": {**SCHEMA["tasks"], "other": SCHEMA["tasks"]["safety"]}}),
        [ChatMessageSystem(content=json.dumps(SCHEMA))],
    ],
    ids=["no-schema", "unparsable", "two-tasks", "no-user-message"],
)
def test_a_malformed_conversation_never_reaches_the_model(messages: list[Any]) -> None:
    classifier = StubClassifier()

    with pytest.raises(ValueError, match="GLiNER2 requires"):
        generate(api(classifier), messages)
    assert classifier.calls == []


def test_scores_missing_the_task_fail_the_sample() -> None:
    # The processor's fallback for a record it cannot encode.
    classifier = StubClassifier(tasks={})

    with pytest.raises(ValueError, match="no scores"):
        generate(api(classifier), conversation())


def test_inspect_resolves_the_gliner2_prefix_to_this_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GLiNER2API, "_load", staticmethod(lambda *_: StubClassifier()))
    model = get_model(f"gliner2/{MODEL}", memoize=False, revision=REVISION)

    assert isinstance(model.api, GLiNER2API)
    assert model.api.model_name == MODEL
