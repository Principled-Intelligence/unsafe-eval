"""Inspect model provider for GLiNER2 classifiers run on the local machine.

A GLiNER2 checkpoint scores the labels of a classification task against a text instead
of generating one. This provider reads one task from the system message, as
`gliner2.classification.ClassificationSchema` JSON, scores the last user message's text
with a pinned checkpoint, and returns the most probable label as the completion, with
every label's probability in the output metadata. It knows nothing about safety.

* The checkpoint is downloaded at an exact hub commit and loaded from that directory,
  because `from_pretrained(revision=...)` still fetches the tokenizer from `main`.
* Requests that arrive while the GPU is busy are scored together, each with its own
  schema, up to 64 texts of similar length per forward pass. A pass that runs out of GPU
  memory is split in half until it fits.
* `gliner2` is imported only when the model is built or called, because
  `protocols/_builtins` imports this module on every install.
"""

import json
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
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from unsafe_eval.providers._batching import Batcher
from unsafe_eval.providers._pins import commit

# Most texts per forward pass.
BATCH_SIZE = 64


def _schema_from_messages(input: list[ChatMessage]) -> dict[str, Any]:
    """Read the single-task classification schema carried in the system message."""
    system = next((message for message in input if message.role == "system"), None)
    try:
        schema = json.loads(system.text) if system is not None else None
    except json.JSONDecodeError:
        schema = None
    tasks = schema.get("tasks") if isinstance(schema, dict) else None
    if not isinstance(schema, dict) or not isinstance(tasks, dict) or len(tasks) != 1:
        raise ValueError(
            "GLiNER2 requires a ClassificationSchema JSON with exactly one task "
            "in the system message"
        )
    return schema


class GLiNER2API(ModelAPI):
    """Classify each message with a pinned local GLiNER2 checkpoint, in batches."""

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
            raise ValueError(f"Unsupported GLiNER2 model arguments: {sorted(model_args)!r}")
        revision = commit("GLiNER2", "revision", revision)
        self.served_model = f"{model_name}@{revision}"
        self._classifier = self._load(model_name, revision, device)
        self._batcher = Batcher(self._score)

    @staticmethod
    def _load(model_name: str, revision: str, device: str) -> Any:
        """Download the pinned checkpoint and load it, importing the optional `gliner2` only now."""
        try:
            from gliner2.classification import Classifier
        except ImportError as error:
            raise ImportError(
                "The gliner2 provider requires the `gliner` extra: "
                "uv sync --extra gliner (or pip install 'unsafe-eval[gliner]')"
            ) from error

        path = snapshot_download(model_name, revision=revision)
        # `from_pretrained(device=...)` moves only the inputs, leaving the weights on the CPU.
        return Classifier.from_pretrained(path).to(device=device)

    def _score(self, requests: list[tuple[int, str, Any]]) -> list[Any]:
        # A forward pass pads to its longest input, so similar lengths share one.
        order = sorted(range(len(requests)), key=lambda index: requests[index][0])
        scores: list[Any] = [None] * len(requests)
        for start in range(0, len(order), BATCH_SIZE):
            chunk = order[start : start + BATCH_SIZE]
            results = self._score_chunk([requests[index] for index in chunk])
            for index, score in zip(chunk, results, strict=True):
                scores[index] = score
        return scores

    def _score_chunk(self, requests: list[tuple[int, str, Any]]) -> list[Any]:
        import torch

        _, texts, schemas = zip(*requests, strict=True)
        try:
            # `Classifier.batch_score` takes one schema for every text; its scorer takes one each.
            return self._classifier.scorer.batch_score(texts, schemas, batch_size=len(texts))
        except torch.OutOfMemoryError:
            if len(requests) == 1:
                raise
        # Split outside the handler, so the failed pass's tensors are freed first.
        half = len(requests) // 2
        return self._score_chunk(requests[:half]) + self._score_chunk(requests[half:])

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Classify the message under evaluation against the conversation's one task."""
        from gliner2.classification import ClassificationSchema

        schema = _schema_from_messages(input)
        text = next((message.text for message in reversed(input) if message.role == "user"), None)
        if text is None:
            raise ValueError("GLiNER2 requires a user message to classify")
        compiled = self._classifier.compile_schema(ClassificationSchema.from_dict(schema))
        size = len(text) + len(json.dumps(schema))
        scores, seconds = await self._batcher.submit((size, text, compiled))
        ((task, spec),) = schema["tasks"].items()
        labels = list(spec["labels"])
        # The processor replaces a record it cannot encode with a placeholder instead of raising.
        if set(scores.tasks.get(task, {})) != set(labels):
            raise ValueError(f"GLiNER2 returned no scores for the labels of task {task!r}")
        probabilities = {label: scores.probability(task, label) for label in labels}
        label = max(probabilities, key=probabilities.__getitem__)
        output = ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=label, model=self.model_name),
                    stop_reason="stop",
                )
            ],
            metadata={"served_model": self.served_model, "probabilities": probabilities},
        )
        # Inspect times a call as its `ModelCall` says, so the wait for the device is left out.
        call = ModelCall.create({"text": text, "schema": schema}, probabilities, time=seconds)
        return output, call


@modelapi(name="gliner2")
def gliner2() -> type[ModelAPI]:
    """Register the GLiNER2 provider with Inspect."""
    return GLiNER2API
