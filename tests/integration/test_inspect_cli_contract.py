"""The flags `unsafe-eval run` builds every command from must still exist in Inspect."""

import subprocess
import sys

from unsafe_eval.cli._run import RESERVED_INSPECT_OPTIONS


def test_inspect_eval_still_accepts_the_flags_run_builds_on() -> None:
    # An early warning: if Inspect drops or renames one of these, every sweep breaks.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "inspect_ai", "eval", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in RESERVED_INSPECT_OPTIONS:
        assert flag in completed.stdout
