"""Batching for providers that run one local model on one device."""

import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import anyio
import anyio.to_thread

Request = TypeVar("Request")
Result = TypeVar("Result")


class Batcher(Generic[Request, Result]):
    """Answer the requests that queue while the device is busy with one call of `run`.

    `run` runs in a worker thread and returns one result per request, in order. If it
    raises, every request in that batch fails with the error.
    """

    def __init__(self, run: Callable[[list[Request]], list[Result]]) -> None:
        self._run = run
        self._pending: list[tuple[Request, dict[str, Any], anyio.Event]] = []
        # Held by whichever request is running a batch on the device.
        self._lock = anyio.Lock()

    async def submit(self, request: Request) -> tuple[Result, float]:
        """Return the request's result and the seconds its batch ran on the device.

        The seconds leave out waiting for earlier batches to free the device.
        """
        slot: dict[str, Any] = {}
        done = anyio.Event()
        self._pending.append((request, slot, done))
        async with self._lock:
            # A request the previous batch already answered skips; otherwise this one runs
            # every request that queued while the device was busy.
            if not done.is_set():
                batch, self._pending = self._pending, []
                start = time.monotonic()
                try:
                    results: list[Any] = await anyio.to_thread.run_sync(
                        self._run, [queued for queued, _, _ in batch]
                    )
                except Exception as error:
                    results = [error] * len(batch)
                except BaseException:
                    # Cancelled: the next request to take the lock runs the others.
                    self._pending[:0] = [queued for queued in batch if queued[1] is not slot]
                    raise
                seconds = time.monotonic() - start
                for (_, queued_slot, queued_done), result in zip(batch, results, strict=True):
                    queued_slot["result"] = result
                    queued_slot["seconds"] = seconds
                    queued_done.set()
        if isinstance(result := slot["result"], Exception):
            raise result
        return result, slot["seconds"]
