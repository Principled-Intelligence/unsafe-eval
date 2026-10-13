"""Inspect model provider for Jev, served by TypeSafe's System One API.

Jev does not generate text. A request gives it a `state`, the content to judge, and
named questions about that state, and Jev answers each question. This provider turns
one Inspect conversation into one `system_one` request that asks a single Choice
question, and returns the option Jev picks as the completion text, so Inspect's
solvers, scorers and logs treat Jev like any other chat model.

The provider knows nothing about safety. Which option means unsafe, and any threshold
on the returned probabilities, belong to the protocol and its scorer.

* Each input comes from exactly one place: the questions are JSON in the system
  message, and the state is an object under `state` in the last user message's
  metadata. A missing or malformed input raises, because falling back to the message
  text would make a protocol bug look like a model result.
* `typesafe_sdk` is imported only when the model is constructed, because
  `protocols/_builtins` imports this module on every install.
* A response from any model version other than the pinned one fails the run.
* The SDK's own retries are off and Inspect retries failed requests instead, so a sample's
  time is its successful request's alone, and Inspect's adaptive concurrency sees rate limits.
"""

import json
from typing import TYPE_CHECKING, Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    RetryDecision,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from typesafe_sdk import AsyncTypeSafeClient, SystemOneResponse

TYPESAFE_API_KEY = "TYPESAFE_API_KEY"


def questions_from_messages(input: list[ChatMessage]) -> dict[str, Any]:
    """Read the packaged questions carried in the system message."""
    for message in input:
        if message.role == "system":
            try:
                questions: dict[str, Any] = json.loads(message.text)
            except json.JSONDecodeError as error:
                raise ValueError("TypeSafe questions must be JSON in the system message") from error
            return questions
    raise ValueError("TypeSafe requires questions in a system message")


def state_from_messages(input: list[ChatMessage]) -> dict[str, Any]:
    """Read the structured state the protocol attached to the message under evaluation.

    The protocol decides the state's shape, such as which fields it has and what the
    per-sample policy is called, so this provider only checks that the state is a
    non-empty object.
    """
    for message in reversed(input):
        if message.role == "user":
            state = (message.metadata or {}).get("state")
            if not isinstance(state, dict) or not state:
                raise ValueError(
                    "TypeSafe requires a non-empty structured state in the user "
                    "message metadata under 'state'"
                )
            return state
    raise ValueError("TypeSafe requires a user message to evaluate")


def model_output_from_response(
    response: "SystemOneResponse",
    *,
    model_name: str,
) -> ModelOutput:
    """Present the single Choice answer as an Inspect completion."""
    if response.model != model_name:
        raise ValueError(
            f"TypeSafe served {response.model!r} for pinned model {model_name!r}; "
            "the evaluation would be attributed to the wrong model"
        )
    if len(response.answers) != 1:
        raise ValueError(
            "TypeSafe questions must declare exactly one question to produce a "
            f"completion; got {sorted(response.answers)!r}"
        )

    (answer,) = response.answers.values()
    # Noul and Score answers are continuous and have no option to return as text.
    if answer.type != "choice":
        raise ValueError(
            f"TypeSafe questions must declare a Choice question; got a {answer.type} answer"
        )
    input_tokens = response.usage.input_tokens or 0
    output_tokens = response.usage.output_tokens or 0
    return ModelOutput(
        model=response.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=answer.choice, model=response.model),
                stop_reason="stop",
            )
        ],
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        metadata={
            "served_model": response.model,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        },
    )


class TypeSafeAPI(ModelAPI):
    """Evaluate one message per request against a TypeSafe System One model."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[*(api_key_vars or []), TYPESAFE_API_KEY],
            config=config or GenerateConfig(),
        )
        # `revision` repeats the model name so the run config records provenance like
        # every other model's.
        revision = model_args.get("revision")
        if revision is not None and revision != model_name:
            raise ValueError(
                f"TypeSafe revision {revision!r} must match the pinned model {model_name!r}"
            )
        self._client = self._build_client(base_url)

    def _build_client(self, base_url: str | None) -> "AsyncTypeSafeClient":
        """Construct the SDK client, importing the optional SDK only now."""
        try:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
        except ImportError as error:
            raise ImportError(
                "The typesafe provider requires the `typesafe` extra: "
                "uv sync --extra typesafe (or pip install 'unsafe-eval[typesafe]')"
            ) from error

        return AsyncTypeSafeClient(
            api_key=self.api_key, base_url=base_url, retry=RetryPolicy(max_retries=0)
        )

    def should_retry(self, ex: Exception) -> bool | RetryDecision:
        """Retry what the SDK retries by default: 408, 429, 5xx, and lost connections."""
        from typesafe_sdk import (
            TypeSafeAPIConnectionError,
            TypeSafeAPIError,
            TypeSafeRateLimitError,
        )

        if isinstance(ex, TypeSafeRateLimitError):
            retry_after = ex.retry_after_ms / 1000 if ex.retry_after_ms is not None else None
            return RetryDecision.rate_limit(retry_after=retry_after)
        if isinstance(ex, TypeSafeAPIConnectionError) or (
            isinstance(ex, TypeSafeAPIError) and (ex.status == 408 or ex.status >= 500)
        ):
            return RetryDecision.transient()
        return RetryDecision.no()

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Issue one System One request for this conversation."""
        response = await self._client.system_one(
            state_from_messages(input),
            questions_from_messages(input),
            model=self.model_name,
        )
        return model_output_from_response(response, model_name=self.model_name)


@modelapi(name="typesafe")
def typesafe() -> type[ModelAPI]:
    """Register the TypeSafe provider with Inspect."""
    return TypeSafeAPI
