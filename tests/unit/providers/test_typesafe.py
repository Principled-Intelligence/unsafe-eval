"""Translation tests for the TypeSafe provider.

Responses are the SDK's own types; only the client is stubbed, so no test reaches
the network.
"""

import asyncio
import json
import sys
from typing import Any

import httpx2
import pytest
import typesafe_sdk
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, RetryDecision
from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    SystemOneResponse,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
    Usage,
)

from unsafe_eval.providers.typesafe import (
    TypeSafeAPI,
    model_output_from_response,
    questions_from_messages,
    state_from_messages,
    typesafe,
)

MODEL = "jev-1.13.0"
QUESTIONS = json.dumps({"safety": {"type": "choice", "criteria": {"safe": None}}})


def choice_answer() -> ChoiceAnswer:
    return ChoiceAnswer(
        choice="unsafe", confidence=0.91, probabilities={"safe": 0.13, "unsafe": 0.87}
    )


def sdk_response(
    model: str = MODEL,
    answers: dict[str, Any] | None = None,
) -> SystemOneResponse:
    return SystemOneResponse(
        model=model,
        answers=answers if answers is not None else {"safety": choice_answer()},
        usage=Usage(input_tokens=12, output_tokens=3),
    )


class StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, state: Any, questions: Any, *, model: str | None = None) -> Any:
        self.calls.append({"state": state, "questions": questions, "model": model})
        return sdk_response()


def conversation(
    *, questions: str = QUESTIONS, state: Any = None, include_system: bool = True
) -> list[Any]:
    metadata = {} if state is None else {"state": state}
    messages: list[Any] = []
    if include_system:
        messages.append(ChatMessageSystem(content=questions))
    messages.append(ChatMessageUser(content="prompt-sentinel", metadata=metadata))
    return messages


# --- questions channel -------------------------------------------------------


def test_a_conversation_without_a_system_message_is_refused() -> None:
    with pytest.raises(ValueError, match="requires questions"):
        questions_from_messages(conversation(include_system=False))


def test_a_system_message_that_is_not_json_is_refused() -> None:
    with pytest.raises(ValueError, match="must be JSON"):
        questions_from_messages(conversation(questions="not json"))


# --- state channel ---------------------------------------------------------


def test_a_missing_state_is_refused() -> None:
    # The single channel is a contract. The prior provider fell back to the bare
    # message text here, which is exactly what this provider must not do.
    with pytest.raises(ValueError, match="structured state"):
        state_from_messages(conversation(state=None))


# --- response translation --------------------------------------------------


def test_the_chosen_option_becomes_the_completion_with_metadata_and_usage() -> None:
    output = model_output_from_response(sdk_response(), model_name=MODEL)

    assert output.completion == "unsafe"
    assert output.model == MODEL
    assert output.metadata is not None
    assert output.metadata["probabilities"] == {"safe": 0.13, "unsafe": 0.87}
    assert output.metadata["confidence"] == 0.91
    assert output.metadata["served_model"] == MODEL
    assert output.usage is not None
    assert output.usage.input_tokens == 12
    assert output.usage.output_tokens == 3
    assert output.usage.total_tokens == 15


def test_a_served_version_other_than_the_pin_fails_the_run() -> None:
    # Continuing would attribute the numbers to the wrong model.
    response = sdk_response(model="jev-1.14.0")

    with pytest.raises(ValueError, match="wrong model"):
        model_output_from_response(response, model_name=MODEL)


def test_a_non_choice_answer_is_a_configuration_error_not_a_result() -> None:
    response = sdk_response(answers={"safety": NoulAnswer(noul=0.72)})

    with pytest.raises(ValueError, match="Choice"):
        model_output_from_response(response, model_name=MODEL)


def test_a_response_carrying_no_answer_is_refused() -> None:
    response = sdk_response(answers={})

    with pytest.raises(ValueError, match="exactly one question"):
        model_output_from_response(response, model_name=MODEL)


# --- generate --------------------------------------------------------------


def generate(api: TypeSafeAPI, messages: list[Any]) -> Any:
    return asyncio.run(api.generate(messages, [], "none", GenerateConfig()))


def stub_client(monkeypatch: pytest.MonkeyPatch) -> StubClient:
    client = StubClient()
    monkeypatch.setattr(TypeSafeAPI, "_build_client", lambda self, base_url: client)
    return client


def test_generate_sends_the_questions_and_state_and_pins_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = stub_client(monkeypatch)
    api = TypeSafeAPI(MODEL)
    payload = {"message": "prompt-sentinel"}

    output = generate(api, conversation(state=payload))

    assert client.calls == [{"state": payload, "questions": json.loads(QUESTIONS), "model": MODEL}]
    assert output.completion == "unsafe"


def test_a_missing_sdk_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    with pytest.raises(ImportError, match="uv sync --extra typesafe"):
        TypeSafeAPI(MODEL)


def test_the_sdk_client_leaves_retries_to_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}
    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", lambda **kwargs: built.update(kwargs))

    TypeSafeAPI(MODEL, api_key="key")

    assert built["retry"].max_retries == 0


HEADERS = httpx2.Headers({"retry-after-ms": "1500"})


@pytest.mark.parametrize(
    ("error", "decision"),
    [
        (TypeSafeRateLimitError(429, None, HEADERS), RetryDecision.rate_limit(retry_after=1.5)),
        (TypeSafeInternalServerError(503, None, httpx2.Headers()), RetryDecision.transient()),
        (TypeSafeAPIError(408, None, httpx2.Headers()), RetryDecision.transient()),
        (TypeSafeAPITimeoutError("timed out"), RetryDecision.transient()),
        (TypeSafeBadRequestError(400, None, httpx2.Headers()), RetryDecision.no()),
        (ValueError("not from the SDK"), RetryDecision.no()),
    ],
)
def test_inspect_retries_what_the_sdk_retried_by_default(
    monkeypatch: pytest.MonkeyPatch, error: Exception, decision: RetryDecision
) -> None:
    stub_client(monkeypatch)

    assert TypeSafeAPI(MODEL).should_retry(error) == decision


def test_a_revision_argument_disagreeing_with_the_pin_is_refused() -> None:
    with pytest.raises(ValueError, match="must match the pinned model"):
        TypeSafeAPI(MODEL, revision="jev-1.14.0")


def test_the_registered_factory_builds_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inspect's @modelapi wrapper instantiates the model rather than handing back
    # the class, so this exercises the registration path a real run takes.
    stub_client(monkeypatch)
    api = typesafe(MODEL)

    assert isinstance(api, TypeSafeAPI)
    assert api.model_name == MODEL
