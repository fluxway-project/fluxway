"""
ArrivalBarrier — shared primitive: wait until N items have arrived.

Two modes:
  one-shot (default):  the barrier fires exactly once when total arrivals >= n.
  periodic:            the barrier fires every n arrivals; a remainder batch
                       also fires if total is not a multiple of n.

Used by both the transport layer (TransportReceiver) and the SPMD mailbox
(StepBarrier) so that the "count arrivals, then fire" logic is implemented
exactly once.
"""

import asyncio


class ArrivalBarrier:
    """
    Fires when arrivals reach the configured threshold.

    One-shot mode (periodic=False, default):
        The barrier fires once when arrive() has been called at least `n` times.
        await wait()                — blocks until that first firing.
        is_done() -> bool           — non-blocking check.

    Periodic mode (periodic=True):
        The barrier fires every `n` arrivals and also fires a final time for
        any remainder (total % n != 0).  Example: n=3, total=10 fires at
        counts 3, 6, 9, **and** 10 (remainder=1).

        await wait_batch() -> int   — blocks until the next batch fires;
                                      returns the cumulative arrival count
                                      at the moment of firing.
        close()                     — signal no more arrivals (flushes remainder
                                      if periodic and remainder > 0).

    Args:
        n:        Batch size (one-shot: minimum arrivals; periodic: interval).
        periodic: Enable repeating / chunked mode.
    """

    def __init__(self, n: int, periodic: bool = False) -> None:
        self._n = max(n, 0)
        self._periodic = periodic
        self._count = 0

        if not periodic:
            # One-shot: use asyncio.Event
            self._event = asyncio.Event()
            if self._n == 0:
                self._event.set()
        else:
            # Periodic: use asyncio.Queue — each entry is the count at firing
            self._queue: asyncio.Queue[int] = asyncio.Queue()
            self._closed = False

    # ------------------------------------------------------------------
    # Arrival notification
    # ------------------------------------------------------------------

    def arrive(self) -> bool:
        """
        Record one arrival.

        One-shot mode:
            Returns True on the transition from not-done to done.
        Periodic mode:
            Returns True every time a batch boundary is crossed (every n
            arrivals).  Does NOT fire for the remainder — call close() to
            flush the final partial batch.
        """
        self._count += 1

        if not self._periodic:
            if not self._event.is_set() and self._count >= self._n:
                self._event.set()
                return True
            return False
        else:
            if self._n > 0 and self._count % self._n == 0:
                self._queue.put_nowait(self._count)
                return True
            return False

    def close(self) -> bool:
        """
        Periodic mode only: signal that no more arrivals will come.

        If a partial batch remains (count % n != 0), it is flushed as a
        final batch entry.  Must be called after the last arrive() so that
        consumers of wait_batch() can detect the end of the stream.

        Returns True if a remainder batch was flushed.
        """
        if not self._periodic or self._closed:
            return False
        self._closed = True
        remainder = self._count % self._n if self._n > 0 else self._count
        if remainder > 0:
            self._queue.put_nowait(self._count)
            return True
        return False

    # ------------------------------------------------------------------
    # One-shot interface
    # ------------------------------------------------------------------

    async def wait(self) -> None:
        """One-shot mode: block until the barrier has fired."""
        if self._periodic:
            raise TypeError("Use wait_batch() in periodic mode")
        await self._event.wait()

    def is_done(self) -> bool:
        """One-shot mode: True if the barrier has fired."""
        if self._periodic:
            return self._closed and self._queue.empty()
        return self._event.is_set()

    # ------------------------------------------------------------------
    # Periodic interface
    # ------------------------------------------------------------------

    async def wait_batch(self) -> int:
        """
        Periodic mode: block until the next batch fires.

        Returns the cumulative arrival count at the moment of firing.
        Raises StopAsyncIteration when close() has been called and all
        batches (including the remainder) have been consumed.
        """
        if not self._periodic:
            raise TypeError("Use wait() in one-shot mode")
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        count = await self._queue.get()
        return count

    def __aiter__(self):
        """Iterate over batch completions in periodic mode."""
        if not self._periodic:
            raise TypeError("async iteration requires periodic=True")
        return self

    async def __anext__(self) -> int:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=None)
        except asyncio.CancelledError as error:
            raise StopAsyncIteration from error

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the barrier to its initial state (one-shot mode only)."""
        if self._periodic:
            raise TypeError("reset() is not supported in periodic mode")
        self._count = 0
        self._event.clear()

    @property
    def count(self) -> int:
        return self._count

    @property
    def target(self) -> int:
        return self._n
