from importlib.metadata import entry_points


def test_inspect_entry_point_is_installed() -> None:
    inspect_entries = entry_points(group="inspect_ai")
    unsafe_eval_entries = [entry for entry in inspect_entries if entry.name == "unsafe_eval"]

    assert len(unsafe_eval_entries) == 1
    assert unsafe_eval_entries[0].value == "unsafe_eval._registry"
