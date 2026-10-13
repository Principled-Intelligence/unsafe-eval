"""Inspect model provider for d1, LiquidAI's open-weight System One model, run locally.

d1 takes the same request as Jev on TypeSafe's System One API: a `state` and named
questions about it. This provider reads both from the same messages the `typesafe`
provider reads, answers the single Choice question, and returns the option d1 picks as
the completion text, so the `jev` solvers and the `choice_safety` scorer work with
either model. Like `typesafe`, it knows nothing about safety.

* d1's code ships in the model repository and loads its tokenizer from the model's own
  path without a revision, so the provider downloads the pinned hub commit and loads
  code, weights and tokenizer from that snapshot.
* d1 writes a question's instructions and options into its prompt verbatim, so anything
  but a plain string there fails the run.
* Requests that arrive while the GPU is busy are answered together by one
  `system_one_batch` call, which packs them with no padding.
* `torch` and `transformers` are imported only when the model is constructed, because
  `protocols/_builtins` imports this module on every install.
"""

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


def check_question(questions: dict[str, Any]) -> None:
    """Raise `ValueError` unless this is one Choice question written in plain strings."""
    if len(questions) != 1:
        raise ValueError(
            f"d1 questions must declare exactly one question; got {sorted(questions)!r}"
        )
    (question,) = questions.values()
    criteria = question.get("criteria")
    if question.get("type") != "choice" or not isinstance(criteria, dict) or not criteria:
        raise ValueError("d1 questions must declare a Choice question")
    texts = [question.get("instructions"), *criteria.values()]
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("d1 questions must give instructions and options as plain strings")


def model_output_from_response(
    response: dict[str, Any], *, model_name: str, metadata: dict[str, Any]
) -> ModelOutput:
    """Present a local System One model's single Choice answer as a completion."""
    answers = response["answers"]
    if len(answers) != 1:
        raise ValueError(
            f"System One questions must declare exactly one question; got {sorted(answers)!r}"
        )
    (answer,) = answers.values()
    if answer["type"] != "choice":
        raise ValueError(
            f"System One questions must declare a Choice question; got a {answer['type']} answer"
        )
    input_tokens = int(response["usage"]["input_tokens"])
    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=answer["choice"], model=model_name),
                stop_reason="stop",
            )
        ],
        usage=ModelUsage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens),
        metadata={
            **metadata,
            "probabilities": dict(answer["probabilities"]),
            "confidence": float(answer["confidence"]),
        },
    )


class D1API(ModelAPI):
    """Answer each message on the pinned d1 checkpoint, in batches."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        revision: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config or GenerateConfig(),
        )
        revision = commit("d1", "revision", revision)
        self.served_model = f"{model_name}@{revision}"
        model = self._load(model_name, revision)
        self._batcher = Batcher(
            lambda requests: model.system_one_batch(
                [(request["state"], request["questions"]) for request in requests]
            )
        )

    @staticmethod
    def _load(model_name: str, revision: str) -> Any:
        """Download the pinned checkpoint and load it with the code it ships."""
        try:
            import torch
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "The d1 provider requires the `hf` extra: "
                "uv sync --extra hf (or pip install 'unsafe-eval[hf]')"
            ) from error

        path = snapshot_download(model_name, revision=revision)
        model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)
        return model.to("cuda" if torch.cuda.is_available() else "cpu")

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Answer the conversation's one question about its state."""
        questions = questions_from_messages(input)
        check_question(questions)
        request = {"state": state_from_messages(input), "questions": questions}
        result, seconds = await self._batcher.submit(request)
        output = model_output_from_response(
            result, model_name=self.served_model, metadata={"served_model": self.served_model}
        )
        # Inspect times a call as its `ModelCall` says, so the wait for the device is left out.
        return output, ModelCall.create(request, result, time=seconds)


@modelapi(name="d1")
def d1() -> type[ModelAPI]:
    """Register the d1 provider with Inspect."""
    return D1API
