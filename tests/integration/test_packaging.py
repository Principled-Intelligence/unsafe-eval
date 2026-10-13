"""The built wheel carries the packaged configurations, the report page, and the console script."""

import zipfile
from pathlib import Path

from hatchling.builders.wheel import WheelBuilder

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = PROJECT_ROOT / "src" / "unsafe_eval" / "configs"


def test_the_wheel_holds_every_config_the_report_page_and_the_unsafe_eval_command(
    tmp_path: Path,
) -> None:
    [wheel_path] = WheelBuilder(str(PROJECT_ROOT)).build(
        directory=str(tmp_path), versions=["standard"]
    )

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        [entry_points] = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        entry_points_text = wheel.read(entry_points).decode()

    expected_configs = {
        f"unsafe_eval/configs/{path.relative_to(CONFIG_ROOT).as_posix()}"
        for path in CONFIG_ROOT.rglob("*.yaml")
    }
    assert expected_configs
    assert expected_configs <= names
    assert {
        "unsafe_eval/cli/report_html/report.html",
        "unsafe_eval/cli/report_html/pi-mark.svg",
        "unsafe_eval/cli/report_html/fonts/archivo-latin.woff2",
        "unsafe_eval/cli/report_html/fonts/jetbrains-mono-latin.woff2",
        "unsafe_eval/cli/report_html/fonts/OFL-Archivo.txt",
        "unsafe_eval/cli/report_html/fonts/OFL-JetBrainsMono.txt",
    } <= names
    assert "[console_scripts]\nunsafe-eval = unsafe_eval.cli:cli" in entry_points_text
