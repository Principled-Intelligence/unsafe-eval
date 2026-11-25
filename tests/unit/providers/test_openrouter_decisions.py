"""Tests for the openrouter-decisions provider.

Requests go to an `httpx.MockTransport`, so no test reaches the network.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, RetryDecision

from unsafe_eval.providers.openrouter_decisions import (
    DECISIONS_URL,
    OPENROUTER_API_KEY,
    OpenRouterDecisionsAPI,
    openrouter_decisions,
)

MODEL = "google/gemini-decision-1-20261111"
QUESTIONS = {
    "safety": {
        "type": "choice",
        "instructions": "Is the message safe?",
        "criteria": {"safe": "The message is safe.", "unsafe": "The message is unsafe."},
    }
}
STATE = {"message": "prompt-sentinel"}
MESSAGES = [
    ChatMessageSystem(content=json.dumps(QUESTIONS)),
    ChatMessageUser(content="prompt-sentinel", metadata={"state": STATE}),
]


def decision(model: str = MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "provider": "Google",
        "answers": {
            "safety": {
                "type": "choice",
                "choice": "unsafe",
                "confidence": 0.91,
                "probabilities": {"safe": 0.13, "unsafe": 0.87},
            }
        },
        "usage": {"input_tokens": 12},
    }


def provider(handler: Callable[[httpx.Request], httpx.Response]) -> OpenRouterDecisionsAPI:
    api = OpenRouterDecisionsAPI(MODEL, api_key="key-sentinel")
    api._client = httpx.AsyncClient(
        headers=api._client.headers, transport=httpx.MockTransport(handler)
    )
    return api


def generate(api: OpenRouterDecisionsAPI) -> Any:
    return asyncio.run(api.generate(MESSAGES, [], "none", GenerateConfig()))


def status_error(status: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", DECISIONS_URL)
    return httpx.HTTPStatusError(
        "failed", request=request, response=httpx.Response(status, headers=headers)
    )


def test_one_request_carries_the_state_and_questions_and_returns_the_choice() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=decision())

    output, call = generate(provider(handler))

    (request,) = seen
    assert str(request.url) == DECISIONS_URL
    assert request.headers["Authorization"] == "Bearer key-sentinel"
    assert json.loads(request.content) == {"model": MODEL, "state": STATE, "questions": QUESTIONS}
    assert output.completion == "unsafe"
    assert output.usage.input_tokens == 12
    assert output.metadata == {
        "served_model": MODEL,
        "provider": "Google",
        "probabilities": {"safe": 0.13, "unsafe": 0.87},
        "confidence": 0.91,
    }
    assert call.response == decision()


def test_a_response_from_another_model_fails_the_run() -> None:
    api = provider(lambda request: httpx.Response(200, json=decision("google/gemini-decision-1")))

    with pytest.raises(ValueError, match="served 'google/gemini-decision-1' for pinned model"):
        generate(api)


def test_a_rejection_raises_with_the_body_that_explains_it() -> None:
    # The refusal check reads the provider's code from the error text.
    api = provider(
        lambda request: httpx.Response(400, text='{"error":{"code":"prohibited_content"}}')
    )

    with pytest.raises(
        httpx.HTTPStatusError, match="OpenRouter returned 400: .*prohibited_content"
    ):
        generate(api)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(429, {"Retry-After": "1.5"}), RetryDecision.rate_limit(retry_after=1.5)),
        (status_error(429), RetryDecision.rate_limit()),
        (
            status_error(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            RetryDecision.rate_limit(),
        ),
        (status_error(408), RetryDecision.transient()),
        (status_error(503), RetryDecision.transient()),
        (httpx.ConnectError("connection lost"), RetryDecision.transient()),
        (status_error(400), RetryDecision.no()),
        (ValueError("not from httpx"), RetryDecision.no()),
    ],
)
def test_inspect_retries_rate_limits_and_transient_failures(
    error: Exception, expected: RetryDecision
) -> None:
    assert OpenRouterDecisionsAPI(MODEL, api_key="key-sentinel").should_retry(error) == expected


def test_a_missing_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENROUTER_API_KEY, raising=False)

    with pytest.raises(ValueError, match=f"requires {OPENROUTER_API_KEY}"):
        OpenRouterDecisionsAPI(MODEL)


def test_the_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENROUTER_API_KEY, "env-sentinel")

    assert OpenRouterDecisionsAPI(MODEL).api_key == "env-sentinel"


def test_closing_the_provider_closes_its_client() -> None:
    api = OpenRouterDecisionsAPI(MODEL, api_key="key-sentinel")

    asyncio.run(api.aclose())

    assert api._client.is_closed


def test_the_registered_factory_builds_the_provider() -> None:
    # Inspect's @modelapi wrapper instantiates the model, so this takes the path a real run takes.
    assert isinstance(openrouter_decisions(MODEL, api_key="key-sentinel"), OpenRouterDecisionsAPI)
