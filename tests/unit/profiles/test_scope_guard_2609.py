from unsafe_eval._resources import content_sha256
from unsafe_eval.profiles.scope_guard_2609 import (
    SCOPE_GUARD_2609_FORMAT,
    SCOPE_GUARD_2609_SYSTEM_MESSAGE,
    scope_guard_2609_user_message,
)


def test_system_prompt_is_the_one_published_with_the_model() -> None:
    # The sha256 the scope-guard-v2-4B-q-2609 model card gives for its
    # system_prompt.txt, which ends in a newline that resource loading strips.
    # A different prompt would raise no error; the model would just do worse.
    published = "f0f68e48096938b93c2c497386295ff7db2929b128e9dddea944ebb6578960b5"

    assert content_sha256(SCOPE_GUARD_2609_SYSTEM_MESSAGE + "\n") == published


def test_user_message_renders_description_prompt_and_selector() -> None:
    message = scope_guard_2609_user_message(
        "prompt-sentinel",
        ai_service_description="description-sentinel",
        selection=SCOPE_GUARD_2609_FORMAT.normalize_selection(["reasoning"]),
    )

    assert message.count("description-sentinel") == 1
    assert message.count("prompt-sentinel") == 1
    assert '["reasoning", "scope_class"]' in message
    assert "{ai_service_description}" not in message
    assert "{prompt}" not in message
    assert "{requested_output_fields}" not in message


def test_user_message_marks_the_last_message_the_way_training_did() -> None:
    """The 2609 corpus uses 'LAST MESSAGE (USER):'; 2601 used 'LAST USER MESSAGE'.

    The models key on their own marker, so this is a train/inference contract and
    not cosmetic.
    """
    message = scope_guard_2609_user_message(
        "prompt-sentinel",
        ai_service_description="description-sentinel",
        selection=SCOPE_GUARD_2609_FORMAT.normalize_selection(["scope_class"]),
    )
    assert "LAST MESSAGE (USER):" in message
    assert "**REQUESTED OUTPUT FIELDS**" in message
