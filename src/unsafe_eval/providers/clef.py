"""Inspect model provider for Clef, Cloudflare's open-weight System One model, run locally.

Clef takes the same request as Jev on TypeSafe's System One API: a `state` and named
questions about it. This provider reads both from the same messages the `typesafe`
provider reads, answers the single Choice question, and returns the option Clef picks
as the completion text, so the `jev` solvers and the `choice_safety` scorer work with
either model. Like `typesafe`, it knows nothing about safety.

* Clef's input encoding and decision head ship as `joint_schema_model.py` in the model
  repository rather than as a package. It is imported from the snapshot at the pinned hub
  commit, which runs that repository's code as `trust_remote_code` would and pins it
  together with the weights.
* The backbone is loaded here rather than by the shipped `load_release_model`, so that
  `quantization="fp8"` can store its linear layers in FP8 and fit the 27B Clef in about
  30 GB of one GPU. The output embeddings stay in bf16, because the decision head reads them.
* Requests that arrive while the GPU is busy are answered together, up to 16 per forward
  pass.
* Clef cuts the state so the whole sequence fits in 16,384 tokens. A cut state is recorded
  in the sample's `state_truncated` metadata, because it comes from the benchmark.
* `torch` and the model code are imported only when the model is constructed, because
  `protocols/_builtins` imports this module on every install.
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from unsafe_eval.providers._batching import Batcher
from unsafe_eval.providers._pins import commit
from unsafe_eval.providers.typesafe import questions_from_messages, state_from_messages

# Most states per forward pass.
BATCH_SIZE = 16

# The longest sequence Clef reads; `encode_record` cuts the state to fit.
MAX_LENGTH = 16384

# Side of the square weight blocks `FineGrainedFP8Config` scales by default.
FP8_BLOCK = 128


def _choice_question(questions: dict[str, Any]) -> None:
    """Raise `ValueError` unless `questions` declares exactly one Choice question."""
    if len(questions) != 1:
        raise ValueError(
            f"Clef questions must declare exactly one question; got {sorted(questions)!r}"
        )
    ((_, question),) = questions.items()
    if not isinstance(question, dict) or question.get("type") != "choice":
        raise ValueError("Clef questions must declare a Choice question")


class ClefAPI(ModelAPI):
    """Answer each conversation's question with a pinned local Clef checkpoint, in batches."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        revision: str | None = None,
        device: str = "cuda",
        quantization: str | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        if model_args:
            raise ValueError(f"Unsupported Clef model arguments: {sorted(model_args)!r}")
        revision = commit("Clef", "revision", revision)
        if quantization not in (None, "fp8"):
            raise ValueError(f"Clef `quantization` must be fp8 when set; got {quantization!r}")
        self.served_model = f"{model_name}@{revision}" + (
            f"+{quantization}" if quantization else ""
        )
        self._release, self._model, self._processor = self._load(
            model_name, revision, device, quantization
        )
        self._batcher = Batcher(self._answer)

    @staticmethod
    def _load(
        model_name: str, revision: str, device: str, quantization: str | None
    ) -> tuple[Any, Any, Any]:
        """Download the pinned checkpoint and load it with the code it ships."""
        try:
            import torch
            import torchvision  # noqa: F401
            from safetensors.torch import load_file
            from transformers import (
                AutoConfig,
                AutoProcessor,
                FineGrainedFP8Config,
                Qwen3_5ForConditionalGeneration,
            )
        except ImportError as error:
            raise ImportError(
                "The clef provider requires the `clef` extra: "
                "uv sync --extra clef (or pip install 'unsafe-eval[clef]')"
            ) from error

        path = snapshot_download(model_name, revision=revision)
        sys.path.insert(0, path)
        release = importlib.import_module("joint_schema_model")
        fp8 = None
        if quantization == "fp8":
            # transformers swaps in an FP8 layer even where the weight does not tile into
            # 128-wide blocks, then leaves its scale uninitialized, so those stay in bf16.
            with torch.device("meta"):
                skeleton = Qwen3_5ForConditionalGeneration._from_config(
                    AutoConfig.from_pretrained(path)
                )
            untileable = [
                name
                for name, module in skeleton.named_modules()
                if isinstance(module, torch.nn.Linear)
                and (module.in_features % FP8_BLOCK or module.out_features % FP8_BLOCK)
            ]
            fp8 = FineGrainedFP8Config(modules_to_not_convert=["lm_head", *untileable])
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map={"": device},
            quantization_config=fp8,
        )
        backbone.config.use_cache = False
        head = release.JointSchemaHead(
            **json.loads(Path(path, "joint_head_config.json").read_text())
        )
        head.load_state_dict(load_file(Path(path, "joint_head.safetensors")), strict=True)
        head = head.to(device=backbone.get_output_embeddings().weight.device, dtype=torch.bfloat16)
        model = release.ClefModel(backbone, head).eval()
        return release, model, AutoProcessor.from_pretrained(path)

    def _answer(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        answers: list[dict[str, Any]] = []
        for start in range(0, len(requests), BATCH_SIZE):
            answers.extend(self._answer_chunk(requests[start : start + BATCH_SIZE]))
        return answers

    def _answer_chunk(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import torch

        tokenizer = self._processor.tokenizer
        records = [
            self._release.encode_record(
                tokenizer, request, max_length=MAX_LENGTH, processor=self._processor
            )
            for request in requests
        ]
        # Inputs go where the head runs; the backbone's hooks move them to each layer's device.
        device = next(self._model.head.parameters()).device
        batch = self._release.collate_records(records, tokenizer.pad_token_id, device)
        with torch.inference_mode():
            logits = self._model(batch)
        return [
            {
                "probabilities": dict(
                    zip(
                        record.questions[0].option_ids,
                        record_logits[0].float().softmax(-1).tolist(),
                        strict=True,
                    )
                ),
                "input_tokens": len(record.input_ids),
            }
            for record, record_logits in zip(records, logits, strict=True)
        ]

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Answer the conversation's one question about its state."""
        questions = questions_from_messages(input)
        _choice_question(questions)
        request = {"state": state_from_messages(input), "questions": questions}
        result, seconds = await self._batcher.submit(request)
        probabilities: dict[str, float] = result["probabilities"]
        choice = max(probabilities, key=probabilities.__getitem__)
        tokens: int = result["input_tokens"]
        output = ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=choice, model=self.model_name),
                    stop_reason="stop",
                )
            ],
            usage=ModelUsage(input_tokens=tokens, output_tokens=0, total_tokens=tokens),
            metadata={
                "served_model": self.served_model,
                "probabilities": probabilities,
                "confidence": probabilities[choice],
                "state_truncated": tokens >= MAX_LENGTH,
            },
        )
        # Inspect times a call as its `ModelCall` says, so the wait for the device is left out.
        return output, ModelCall.create(request, result, time=seconds)


@modelapi(name="clef")
def clef() -> type[ModelAPI]:
    """Register the Clef provider with Inspect."""
    return ClefAPI
