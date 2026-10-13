from pathlib import Path

import pytest
from _support import add_config

from unsafe_eval import _configs


def test_a_file_outside_the_layout_gets_no_empty_or_absolute_name() -> None:
    assert _configs.config_id(Path("/config.yaml")) == ""


def test_discovery_ignores_files_outside_the_three_level_layout(tmp_path: Path) -> None:
    expected = add_config(tmp_path, "vllm/model-a", "benchmark-a")
    (tmp_path / "stray.yaml").write_text("", encoding="utf-8")
    add_config(tmp_path, "vllm/model-a/nested", "benchmark-b")

    assert _configs.discover_configs(tmp_path) == [expected]


def test_config_path_lists_configurations_for_an_unknown_tree(tmp_path: Path) -> None:
    add_config(tmp_path, "vllm/model-a", "benchmark-a")

    with pytest.raises(_configs.SweepSelectionError, match="Available identifiers: vllm/model-a"):
        _configs.config_path("vllm/missing/benchmark-a", tmp_path)
