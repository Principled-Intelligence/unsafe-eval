"""Inspect model provider for Laya, an open-weight model run on the local machine.

Laya takes the same request as Jev on TypeSafe's System One API: a `state` and named
questions about it. This provider reads both from the same messages the `typesafe`
provider reads, answers the single Choice question, and returns the option Laya picks
as the completion text, so the `jev` solvers and the `choice_safety` scorer work with
either model. Like `typesafe`, it knows nothing about safety.

* Every request goes through Laya's `Router`, which picks the English or the
  multilingual checkpoint by the state's script and language, and loads both at one
  pinned hub commit.
* Requests that arrive while the GPU is busy are answered together by one
  `Router.predict_batch` call, which shares forward passes between states that route
  to the same checkpoint.
* `max_len`, the most tokens Laya reads per question, has no default, so every run's
  log records it.
* Laya silently cuts each option to 48 tokens, the question text to what the options
  leave of `head_max_len`, and the state to what the question leaves of `max_len`. A
  cut question fails the run, because the protocol wrote it; the provider checks each
  question once against both checkpoints with Laya's `build_sequence`. A cut state is
  recorded in the sample's `state_truncated` metadata, because it comes from the
  benchmark; Laya reports it.
* `laya` and `torch` are imported only when the model is constructed, because
  `protocols/_builtins` imports this module on every install.
"""

import json
from typing import Any

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

# Hub subfolder of each checkpoint `Router` picks by language. It picks `typed-decisions`
# only when a caller names it.
CHECKPOINTS = {"english": None, "multilingual": "multilingual"}

# The English checkpoint's trained context, and the encoders' position limit.
MIN_MAX_LEN = 512
MAX_MAX_LEN = 8192

# `build_sequence` keeps at most this many tokens of each option, a fixed limit rather
# than one of its arguments, so it is checked separately.
OPTION_TOKEN_LIMIT = 48

# Most states per forward pass.
BATCH_SIZE = 64

# Longer than any sequence Laya reads, so a sequence built to it is never cut.
_UNLIMITED = 1_000_000


def _laya_question(question: dict[str, Any]) -> dict[str, Any]:
    """Convert a Choice question to the form `build_sequence` takes, as `laya.Agent` does."""
    criteria = question.get("criteria")
    if question.get("type") != "choice" or not isinstance(criteria, dict) or not criteria:
        raise ValueError("Laya questions must declare a Choice question")
    instructions = question.get("instructions")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)
    return {"t": "choice", "ins": instructions, "crit": criteria}


def check_question(tokenizer: Any, head_max_len: int, questions: dict[str, Any]) -> None:
    """Raise `ValueError` if Laya would cut any part of this single question.

    A cut question is no longer the one whose digest the run records.
    """
    from laya.common import build_sequence, render_options

    (question,) = questions.values()
    laya_question = _laya_question(question)
    head, _ = build_sequence(tokenizer, "", laya_question, _UNLIMITED, head_max_len)
    whole_head, _ = build_sequence(tokenizer, "", laya_question, _UNLIMITED, _UNLIMITED)
    options = [
        len(tokenizer(" " + option, add_special_tokens=False)["input_ids"])
        for option in render_options(laya_question)
    ]
    if len(head) < len(whole_head) or max(options) > OPTION_TOKEN_LIMIT:
        raise ValueError(
            f"Laya would cut this question: its text and options must fit head_max_len="
            f"{head_max_len} and each option {OPTION_TOKEN_LIMIT} tokens; the options "
            f"use {options!r}"
        )


def model_output_from_response(
    response: dict[str, Any], *, model_name: str, revision: str
) -> ModelOutput:
    """Present Laya's single Choice answer, and the checkpoint it came from, as a completion."""
    answers = response.get("answers")
    if not isinstance(answers, dict) or len(answers) != 1:
        raise ValueError("Laya returned no single answer for the one question asked")
    (answer,) = answers.values()
    choice = answer.get("choice") if isinstance(answer, dict) else None
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(choice, str) or not isinstance(probabilities, dict):
        raise ValueError("Laya questions must declare a Choice question")
    confidence = answer.get("confidence")
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    routing = response["routing"]
    return ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=choice, model=model_name),
                stop_reason="stop",
            )
        ],
        usage=ModelUsage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens),
        metadata={
            "served_model": f"{routing['repo']}@{revision}",
            "probabilities": dict(probabilities),
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "sequence_tokens": input_tokens,
            "state_truncated": bool(usage.get("truncated")),
            "routing": routing,
        },
    )


class LayaAPI(ModelAPI):
    """Answer each message on the pinned Laya checkpoint its router picks, in batches."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        *,
        revision: str | None = None,
        max_len: int | None = None,
        device: str | None = None,
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
            raise ValueError(f"Unsupported Laya model arguments: {sorted(model_args)!r}")
        revision = commit("Laya", "revision", revision)
        if (
            not isinstance(max_len, int)
            or isinstance(max_len, bool)
            or not MIN_MAX_LEN <= max_len <= MAX_MAX_LEN
        ):
            raise ValueError(
                f"Laya requires `max_len`, an integer from {MIN_MAX_LEN} to {MAX_MAX_LEN}"
            )
        self.revision = revision
        self.max_len = max_len
        self._router = self._load(model_name, revision, device)
        self._checked: set[str] = set()
        self._batcher = Batcher(self._evaluate)

    @staticmethod
    def _load(model_name: str, revision: str, device: str | None) -> Any:
        """Load the pinned checkpoints into a router, importing the optional `laya` only now."""
        try:
            from laya import Router
        except ImportError as error:
            raise ImportError(
                "The laya provider requires the `laya` extra: "
                "uv sync --extra laya (or pip install 'unsafe-eval[laya]')"
            ) from error

        router = Router(
            # laya annotates `models` values as `str`, but its own defaults are these tuples.
            models={name: (model_name, subfolder) for name, subfolder in CHECKPOINTS.items()},  # ty: ignore[invalid-argument-type]
            revision=revision,
            device=device,
        )
        return router.preload(list(CHECKPOINTS))

    def _evaluate(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for request in requests:
            questions = request["questions"]
            if (key := json.dumps(questions)) not in self._checked:
                for name in CHECKPOINTS:
                    agent = self._router.load(name)
                    check_question(agent.tok, agent.cfg["head_max_len"], questions)
                self._checked.add(key)
        return self._router.predict_batch(requests, batch_size=BATCH_SIZE)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Answer the conversation's one question about its state."""
        questions = questions_from_messages(input)
        if len(questions) != 1:
            raise ValueError(
                f"Laya questions must declare exactly one question; got {sorted(questions)!r}"
            )
        request = {
            "state": state_from_messages(input),
            "questions": questions,
            "max_len": self.max_len,
        }
        result, seconds = await self._batcher.submit(request)
        output = model_output_from_response(
            result, model_name=self.model_name, revision=self.revision
        )
        # Inspect times a call as its `ModelCall` says, so the wait for the device is left out.
        return output, ModelCall.create(request, result, time=seconds)


@modelapi(name="laya")
def laya() -> type[ModelAPI]:
    """Register the Laya provider with Inspect."""
    return LayaAPI
