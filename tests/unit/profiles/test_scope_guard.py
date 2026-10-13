import json

from unsafe_eval.profiles.scope_guard import (
    SCOPE_GUARD_RESPONSE_SCHEMA,
    SCOPE_GUARD_SYSTEM_MESSAGE,
    scope_guard_user_message,
)


def test_system_prompt_embeds_response_schema() -> None:
    schema = json.loads(SCOPE_GUARD_SYSTEM_MESSAGE.rsplit("\n", 1)[-1])
    assert schema == SCOPE_GUARD_RESPONSE_SCHEMA


def test_user_message_can_skip_evidence() -> None:
    with_evidence = scope_guard_user_message(
        "prompt-sentinel",
        ai_service_description="description-sentinel",
    )
    without_evidence = scope_guard_user_message(
        "prompt-sentinel",
        ai_service_description="description-sentinel",
        include_evidence=False,
    )

    assert without_evidence != with_evidence
    assert "prompt-sentinel" in without_evidence
    assert "description-sentinel" in without_evidence
