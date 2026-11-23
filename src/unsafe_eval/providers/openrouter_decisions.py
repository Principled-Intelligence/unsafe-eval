"""Inspect model provider for decision models on OpenRouter's Decisions API.

A decision model takes the same `state` and named questions as d1 run locally, so this
provider reads them from the messages the `d1` protocol builds and returns the chosen
option as the completion text, and the `choice_safety` scorer works unchanged.

* Name the dated model slug, such as `google/gemini-decision-1-20261111`, never the undated alias.
  A response from any other model fails the run.
* Inspect retries failed requests: 408, 429, 5xx and lost connections.
"""

import os
from typing import Any

import httpx
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    RetryDecision,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from unsafe_eval.providers.d1 import check_question, model_output_from_response
from unsafe_eval.providers.typesafe import questions_from_messages, state_from_messages

OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
TIMEOUT_SECONDS = 120


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Read a `Retry-After` given in seconds; the HTTP date form is ignored."""
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


class OpenRouterDecisionsAPI(ModelAPI):
    """Ask one Choice question per request of a decision model on OpenRouter."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key or os.environ.get(OPENROUTER_API_KEY),
            api_key_vars=[*(api_key_vars or []), OPENROUTER_API_KEY],
            config=config or GenerateConfig(),
        )
        if not self.api_key:
            raise ValueError(f"The openrouter-decisions provider requires {OPENROUTER_API_KEY}")
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=TIMEOUT_SECONDS
        )

    def should_retry(self, ex: Exception) -> bool | RetryDecision:
        """Retry rate limits, timeouts, server errors and lost connections."""
        if isinstance(ex, httpx.TransportError):
            return RetryDecision.transient()
        if isinstance(ex, httpx.HTTPStatusError):
            status = ex.response.status_code
            if status == 429:
                return RetryDecision.rate_limit(retry_after=retry_after_seconds(ex.response))
            if status == 408 or status >= 500:
                return RetryDecision.transient()
        return RetryDecision.no()

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        """Issue one Decisions request for this conversation."""
        questions = questions_from_messages(input)
        check_question(questions)
        request: dict[str, Any] = {
            "model": self.model_name,
            "state": state_from_messages(input),
            "questions": questions,
        }
        response = await self._client.post(self.base_url or DECISIONS_URL, json=request)
        if response.is_error:
            # OpenRouter explains a rejection only in the body, which raise_for_status drops.
            raise httpx.HTTPStatusError(
                f"OpenRouter returned {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )
        result = response.json()
        if result["model"] != self.model_name:
            raise ValueError(
                f"OpenRouter served {result['model']!r} for pinned model {self.model_name!r}"
            )
        output = model_output_from_response(
            result,
            model_name=self.model_name,
            metadata={"served_model": result["model"], "provider": result.get("provider")},
        )
        return output, ModelCall.create(request, result)


@modelapi(name="openrouter-decisions")
def openrouter_decisions() -> type[ModelAPI]:
    """Register the openrouter-decisions provider with Inspect."""
    return OpenRouterDecisionsAPI
