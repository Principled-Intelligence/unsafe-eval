import threading
from types import SimpleNamespace

import anyio
import anyio.to_thread
import pytest

from unsafe_eval.providers import _batching
from unsafe_eval.providers._batching import Batcher


def test_a_queued_request_reports_its_batch_time_without_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each request costs one tick on a fake clock, so a batch of two takes two.
    clock = [0.0]
    monkeypatch.setattr(_batching, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    running, release = threading.Event(), threading.Event()

    def run(batch: list[str]) -> list[str]:
        if batch == ["a"]:
            running.set()
            release.wait(timeout=5)
        clock[0] += len(batch)
        return batch

    async def main() -> dict[str, tuple[str, float]]:
        batcher = Batcher(run)
        results: dict[str, tuple[str, float]] = {}

        async def submit(request: str) -> None:
            results[request] = await batcher.submit(request)

        async with anyio.create_task_group() as group:
            group.start_soon(submit, "a")
            await anyio.to_thread.run_sync(running.wait, 5)
            group.start_soon(submit, "b")
            group.start_soon(submit, "c")
            await anyio.sleep(0.05)
            release.set()
        return results

    # b and c queued while a's batch ran, then ran together: two ticks, not three.
    assert anyio.run(main) == {"a": ("a", 1.0), "b": ("b", 2.0), "c": ("c", 2.0)}
