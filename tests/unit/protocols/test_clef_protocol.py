from unsafe_eval.protocols import resolve_protocol


def test_metadata_records_the_protocol_its_options_and_the_questions_digest() -> None:
    selected = resolve_protocol("clef", benchmark="aegis-2.0")

    assert selected.metadata["protocol"] == "clef"
    assert set(selected.metadata["prompt_resource_hashes"]) == {"protocol/questions.json"}
    assert selected.metadata["protocol_options"] == {"validate_prompt_resource_hashes": True}
