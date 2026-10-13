"""Report sample progress to the `unsafe-eval run` that launched this `inspect eval`."""

import contextlib
import os

from inspect_ai.hooks import Hooks, SampleEnd, TaskStart, hooks

from unsafe_eval.cli._run import PROGRESS_FD_ENV


def _report(line: str) -> None:
    # Progress is cosmetic, so a pipe `run` has closed must not fail the evaluation.
    with contextlib.suppress(OSError):
        os.write(int(os.environ[PROGRESS_FD_ENV]), f"{line}\n".encode())


@hooks(name="run_progress", description="Reports sample progress to unsafe-eval run.")
class RunProgressHooks(Hooks):
    """Tell `unsafe-eval run` how many samples a task runs and when each one ends."""

    def enabled(self) -> bool:
        return PROGRESS_FD_ENV in os.environ

    async def on_task_start(self, data: TaskStart) -> None:
        # `samples` is the whole dataset; `sample_ids` reflects `--limit` and `--sample-id`.
        samples = len(data.spec.dataset.sample_ids or [])
        _report(f"total {samples * (data.spec.config.epochs or 1)}")

    async def on_sample_end(self, data: SampleEnd) -> None:
        _report("done")


__all__ = ["RunProgressHooks"]
