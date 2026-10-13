"""Translation and batching tests for the Clef provider.

A stub `torch` and stub model code stand in for the real ones, so the offline suite never
imports torch. The stub model's logits are already probabilities.
"""

import asyncio
import contextlib
import json
import sys
import types
from typing import Any

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from unsafe_eval.providers.clef import BATCH_SIZE, MAX_LENGTH, ClefAPI

MODEL = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
QUESTIONS = {
    "safety": {
        "type": "choice",
        "instructions": "is it unsafe",
        "criteria": {"safe": "fine", "unsafe": "bad"},
    }
}


class StubLogits:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def float(self) -> "StubLogits":
        return self

    def softmax(self, dim: int) -> "StubLogits":
        return self

    def tolist(self) -> list[float]:
        return self.values


class StubRelease:
    """Encodes a request as one token per state word, and collates records into a list."""

    def __init__(self, tokens: int | None = None) -> None:
        self.tokens = tokens

    def encode_record(
        self, tokenizer: Any, record: dict[str, Any], max_length: int, processor: Any
    ) -> Any:
        assert max_length == MAX_LENGTH
        count = self.tokens or len(record["state"]["message"].split())
        options = tuple(record["questions"]["safety"]["criteria"])
        question = types.SimpleNamespace(option_ids=options)
        return types.SimpleNamespace(input_ids=tuple(range(count)), questions=(question,))

    def collate_records(self, records: list[Any], pad_token_id: int, device: Any) -> list[Any]:
        return records


class StubModel:
    def __init__(self) -> None:
        self.batches: list[int] = []
        self.head = types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(device="cpu")])
        )

    def __call__(self, batch: list[Any]) -> list[list[StubLogits]]:
        self.batches.append(len(batch))
        return [[StubLogits([0.2, 0.8])] for _ in batch]


@pytest.fixture(autouse=True)
def stub_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    )


def provider(
    monkeypatch: pytest.MonkeyPatch, release: StubRelease | None = None
) -> tuple[ClefAPI, StubModel]:
    model = StubModel()
    processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
    loaded = (release or StubRelease(), model, processor)
    monkeypatch.setattr(ClefAPI, "_load", staticmethod(lambda *_: loaded))
    return ClefAPI(model_name=MODEL, revision=REVISION), model


def messages(questions: dict[str, Any] = QUESTIONS, text: str = "one two three") -> list[Any]:
    return [
        ChatMessageSystem(content=json.dumps(questions)),
        ChatMessageUser(content=text, metadata={"state": {"message": text}}),
    ]


def test_the_most_probable_option_is_the_completion_with_every_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _ = provider(monkeypatch)

    output, call = asyncio.run(api.generate(messages(), [], "none", GenerateConfig()))

    assert output.completion == "unsafe"
    assert output.metadata == {
        "served_model": f"{MODEL}@{REVISION}",
        "probabilities": {"safe": 0.2, "unsafe": 0.8},
        "confidence": 0.8,
        "state_truncated": False,
    }
    assert output.usage is not None and output.usage.input_tokens == 3
    assert call.request == {"state": {"message": "one two three"}, "questions": QUESTIONS}


def test_requests_that_queue_while_the_device_is_busy_share_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, model = provider(monkeypatch)

    async def run() -> None:
        await asyncio.gather(
            *(api.generate(messages(), [], "none", GenerateConfig()) for _ in range(3))
        )

    asyncio.run(run())

    assert model.batches == [3]


def test_a_large_batch_is_split_into_forward_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    api, model = provider(monkeypatch)
    request = {"state": {"message": "text"}, "questions": QUESTIONS}

    answers = api._answer([request] * (BATCH_SIZE + 4))

    assert len(answers) == BATCH_SIZE + 4
    assert model.batches == [BATCH_SIZE, 4]


def test_a_state_cut_to_the_context_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    api, _ = provider(monkeypatch, StubRelease(tokens=MAX_LENGTH))

    output, _ = asyncio.run(api.generate(messages(), [], "none", GenerateConfig()))

    assert output.metadata is not None and output.metadata["state_truncated"] is True


@pytest.mark.parametrize(
    "questions",
    [
        {**QUESTIONS, "other": QUESTIONS["safety"]},
        {"safety": {"type": "noul", "instructions": "is it unsafe"}},
    ],
)
def test_anything_but_one_choice_question_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, questions: dict[str, Any]
) -> None:
    api, model = provider(monkeypatch)

    with pytest.raises(ValueError, match="Clef questions"):
        asyncio.run(api.generate(messages(questions), [], "none", GenerateConfig()))
    assert model.batches == []


@pytest.mark.parametrize("revision", [None, "main", REVISION[:12]])
def test_a_missing_or_moving_revision_is_refused(revision: Any) -> None:
    with pytest.raises(ValueError, match="40-character commit hash"):
        ClefAPI(model_name=MODEL, revision=revision)


@pytest.mark.parametrize("args", [{"max_len": 1024}, {"quantization": "int4"}])
def test_an_unknown_model_argument_is_refused(args: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="max_len|fp8"):
        ClefAPI(model_name=MODEL, revision=REVISION, **args)


SHIPPED_MODEL_CODE = """
class JointSchemaHead:
    def __init__(self, **config):
        self.config = config

    def load_state_dict(self, state, strict):
        self.state = state

    def to(self, device, dtype):
        self.device = device
        return self


class ClefModel:
    def __init__(self, backbone, head):
        self.backbone, self.head = backbone, head

    def eval(self):
        return self
"""


class StubFP8Config:
    def __init__(self, modules_to_not_convert: list[str]) -> None:
        self.modules_to_not_convert = modules_to_not_convert


class StubLinear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features, self.out_features = in_features, out_features


# Only a layer whose weight does not tile into 128-wide blocks stays out of FP8.
SKELETON = {"wide": StubLinear(256, 128), "narrow": StubLinear(256, 32), "norm": object()}


@pytest.mark.parametrize(
    ("device", "quantization", "device_map", "quantized"),
    [("cuda", None, {"": "cuda"}, False), ("cuda:1", "fp8", {"": "cuda:1"}, True)],
)
def test_the_pinned_snapshot_is_loaded_with_the_head_it_ships(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    device: str,
    quantization: str | None,
    device_map: Any,
    quantized: bool,
) -> None:
    (tmp_path / "joint_schema_model.py").write_text(SHIPPED_MODEL_CODE)
    (tmp_path / "joint_head_config.json").write_text('{"layers": 2}')
    loaded: dict[str, Any] = {}

    def from_pretrained(path: str, **kwargs: Any) -> Any:
        loaded.update(path=path, **kwargs)
        weight = types.SimpleNamespace(device="cuda:1")
        return types.SimpleNamespace(
            config=types.SimpleNamespace(),
            get_output_embeddings=lambda: types.SimpleNamespace(weight=weight),
        )

    transformers = types.ModuleType("transformers")
    transformers.Qwen3_5ForConditionalGeneration = types.SimpleNamespace(
        from_pretrained=from_pretrained,
        _from_config=lambda config: types.SimpleNamespace(named_modules=SKELETON.items),
    )
    transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda path: "config")
    transformers.FineGrainedFP8Config = StubFP8Config
    transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda path: "processor")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path: {"weights": str(path)}
    monkeypatch.setattr(
        "unsafe_eval.providers.clef.snapshot_download", lambda repo_id, revision: str(tmp_path)
    )
    torch = types.SimpleNamespace(
        bfloat16="bf16",
        device=lambda name: contextlib.nullcontext(),
        nn=types.SimpleNamespace(Linear=StubLinear),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torchvision", types.ModuleType("torchvision"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "joint_schema_model", raising=False)

    release, model, processor = ClefAPI._load(MODEL, REVISION, device, quantization)

    assert release.__file__ == str(tmp_path / "joint_schema_model.py")
    assert loaded["path"] == str(tmp_path)
    assert loaded["device_map"] == device_map
    fp8 = loaded["quantization_config"]
    assert (fp8.modules_to_not_convert if quantized else fp8) == (
        ["lm_head", "narrow"] if quantized else None
    )
    assert model.backbone.config.use_cache is False
    assert model.head.config == {"layers": 2}
    assert model.head.device == "cuda:1"
    assert processor == "processor"


def test_a_missing_clef_extra_fails_before_any_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "unsafe_eval.providers.clef.snapshot_download",
        lambda *_, **__: pytest.fail("downloaded without the clef extra"),
    )
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match="--extra clef"):
        ClefAPI._load(MODEL, REVISION, "cuda", None)
