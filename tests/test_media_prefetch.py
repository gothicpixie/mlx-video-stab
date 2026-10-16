"""``media.prefetch`` concurrency/failure tests (Stage 7 media reliability), deterministic and ffmpeg-free.

The worker must deliver frames in order through one bounded queue, re-raise a producer failure only after every
frame queued before it has been yielded, and -- when the consumer closes early while the queue is full -- signal
cancellation so the parked ``put`` exits, close the upstream source in the producer thread (so ``media.decode``
can terminate/reap ffmpeg) and join the worker. These tests use plain in-process iterators, so they cannot hang
on a real subprocess or flake on process scheduling.
"""
import threading
import time

import pytest

from mlxstab import media


def _wait(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class _FailingCloseSource:
    """Iterator whose exhausted ``__next__`` raises a preset error and whose ``close`` can also fail.

    A plain class (not a generator) so the failure and the close failure are independent and both observable.
    """

    def __init__(self, frames, error=None, close_error=None):
        self._frames = iter(frames)
        self.error = error
        self.close_error = close_error
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.error is None:
            return next(self._frames)
        try:
            return next(self._frames)
        except StopIteration:
            raise self.error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def test_prefetch_normal_delivers_frames_in_order_and_closes_source():
    """The happy path yields every frame in order and closes the upstream source from the worker."""
    closed = []

    def source():
        try:
            for i in range(11):                                  # more frames than the bounded queue
                yield i
        finally:
            closed.append(True)

    assert list(media.prefetch(source(), n=3)) == list(range(11))
    assert closed == [True]                                      # upstream closed after natural exhaustion


def test_prefetch_read_ahead_is_bounded_by_n():
    """The worker reads ahead by at most ``n`` queued frames plus the one in flight; no unbounded buffer."""
    n = 3
    produced = []

    def source():
        i = 0
        while True:
            produced.append(i)
            yield i
            i += 1

    it = media.prefetch(source(), n=n)
    try:
        assert next(it) == 0                                     # primes the worker, consumes one frame
        assert _wait(lambda: len(produced) >= n + 2)             # n queued, one parked mid-put
        time.sleep(0.05)
        assert len(produced) == n + 2                            # a full queue stops further read-ahead
    finally:
        it.close()


@pytest.mark.parametrize("bad_n", [0, -1, -5])
def test_prefetch_refuses_non_positive_capacity_before_starting(bad_n):
    """``queue.Queue(0)`` is unbounded, so n<=0 is refused eagerly, before any worker touches the source."""
    consumed = []

    def source():
        consumed.append(True)
        yield 0

    with pytest.raises(ValueError, match="capacity must be positive"):
        media.prefetch(source(), n=bad_n)
    assert consumed == []                                        # refused before the source was ever iterated


def test_prefetch_smallest_positive_capacity_still_starts_worker():
    """n=1 remains valid and delivers normally: the refusal must not narrow the accepted range above zero."""
    assert list(media.prefetch(iter([0, 1, 2]), n=1)) == [0, 1, 2]


def test_prefetch_propagates_producer_exception_after_queued_frames():
    """A producer failure is raised only once every frame queued before it has been delivered, and verbatim."""
    boom = ValueError("producer exploded")
    release = threading.Event()

    def source():
        yield 0
        yield 1                                                  # already queued when the failure happens
        release.wait(5.0)
        raise boom

    it = media.prefetch(source(), n=4)
    assert next(it) == 0
    release.set()
    assert next(it) == 1                                         # queued-before-failure frame is not dropped
    with pytest.raises(ValueError) as ei:
        next(it)
    assert ei.value is boom                                      # original exception object, not a wrapper
    with pytest.raises(StopIteration):                           # the wrapped generator is exhausted after it
        next(it)


def test_prefetch_failed_producer_with_full_queue_still_delivers_every_queued_frame():
    """When the producer fails with the queue already full, the trailer waits for the consumer to drain it."""
    boom = IndexError("producer failed with a full queue")
    n = 2

    def source():
        yield 0
        yield 1                                                  # both slots of the bounded queue are used
        raise boom

    it = media.prefetch(source(), n=n)
    got = [next(it), next(it)]                                   # draining frees the slot the trailer waits on
    with pytest.raises(IndexError) as ei:
        next(it)
    assert got == [0, 1] and ei.value is boom


def test_prefetch_early_close_unblocks_full_queue_and_closes_source():
    """Closing while the worker is parked on a full queue cancels the put, closes the source and joins."""
    n = 3
    produced = []
    closed = threading.Event()

    def source():
        try:
            i = 0
            while True:
                produced.append(i)
                yield i
                i += 1
        finally:
            closed.set()                                         # media.decode's terminate/reap lives here

    it = media.prefetch(source(), n=n)
    assert next(it) == 0
    assert _wait(lambda: len(produced) >= n + 2)                 # queue full, worker blocked in put

    done = threading.Event()

    def close_it():
        it.close()
        done.set()

    threading.Thread(target=close_it, daemon=True).start()
    assert done.wait(5.0), "prefetch.close() must not deadlock on a full queue"
    assert closed.wait(5.0), "the worker must close the upstream source so decode can reap ffmpeg"
    assert len(produced) == n + 2                                # cancellation stopped further reads


def test_prefetch_consumer_error_cancels_and_joins_the_worker():
    """An exception thrown into the consumer cancels the parked worker and is propagated verbatim."""
    n = 3
    produced = []
    closed = threading.Event()

    def source():
        try:
            i = 0
            while True:
                produced.append(i)
                yield i
                i += 1
        finally:
            closed.set()

    it = media.prefetch(source(), n=n)
    assert next(it) == 0
    assert _wait(lambda: len(produced) >= n + 2)
    boom = RuntimeError("consumer exploded")
    with pytest.raises(RuntimeError) as ei:
        it.throw(boom)
    assert ei.value is boom
    assert closed.wait(5.0)                                      # the worker still closed the source


def test_prefetch_consumer_throw_survives_upstream_generator_close_failure():
    """An exception thrown into the consumer must survive a failure while the worker closes the upstream gen.

    Cancellation lands while the worker is parked on a full queue, so the upstream source's ``finally`` runs
    during ``close()`` and raises. The consumer's primary error must win -- a close-time failure can never
    replace it -- while the close is still attempted and further reads stop.
    """
    n = 2
    produced = []
    closed = threading.Event()

    def source():
        try:
            i = 0
            while True:
                produced.append(i)
                yield i
                i += 1
        finally:
            closed.set()
            raise OSError("upstream close failed")                # reached only via close(), i.e. cancellation

    it = media.prefetch(source(), n=n)
    assert next(it) == 0
    assert _wait(lambda: len(produced) >= n + 2)                 # queue full, worker parked in put
    boom = RuntimeError("consumer exploded")
    with pytest.raises(RuntimeError) as ei:
        it.throw(boom)
    assert ei.value is boom                                      # consumer error wins, not the OSError
    assert closed.is_set()                                       # the generator's finally did run
    assert len(produced) == n + 2                                # cancellation stopped further reads


def test_prefetch_consumer_throw_survives_upstream_close_failure_after_exhaustion():
    """The consumer error also wins when the upstream close failure happens after natural exhaustion."""
    src = _FailingCloseSource([0, 1], close_error=OSError("close failed"))
    it = media.prefetch(src, n=4)
    assert next(it) == 0
    assert next(it) == 1
    boom = RuntimeError("consumer exploded")
    with pytest.raises(RuntimeError) as ei:
        it.throw(boom)
    assert ei.value is boom
    assert src.close_calls == 1


def test_prefetch_primary_producer_error_is_not_masked_by_close_failure():
    """A failing upstream close must not replace the real producer failure."""
    boom = LookupError("producer")
    src = _FailingCloseSource([0, 1], error=boom, close_error=OSError("close failed"))
    got = []
    it = media.prefetch(src, n=4)
    with pytest.raises(LookupError) as ei:
        for x in it:
            got.append(x)
    assert got == [0, 1]                                         # queued frames were still delivered first
    assert ei.value is boom
    assert src.close_calls == 1                                  # close was attempted, just not allowed to win


class _FakeStream:
    """Serves queued raw reads then EOF; enough for ``media.decode`` to build frames."""

    def __init__(self, reads):
        self._reads = list(reads)

    def read(self, n):
        return self._reads.pop(0) if self._reads else b""

    def close(self):
        pass


class _FakeProc:
    """Minimal ffmpeg child double recording terminate/wait, always still running until signalled."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.args = ("ffmpeg",)
        self.returncode = 0
        self._running = True
        self.terminate_calls = 0
        self.waited = False

    def poll(self):
        return None if self._running else self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self._running = False

    def wait(self, *a, **k):
        self.waited = True
        self._running = False
        return self.returncode


def test_prefetch_early_close_closes_decode_and_reaps_ffmpeg(monkeypatch):
    """Closing the prefetched consumer terminates and waits for the decode child, in the producer thread."""
    w, h = 3, 2
    proc = _FakeProc(_FakeStream([bytes([i]) * (w * h * 3) for i in range(6)]))
    monkeypatch.setattr(media.subprocess, "Popen", lambda *a, **k: proc)
    info = {"width": w, "height": h, "fps": 30.0, "interlaced": False}
    gen, _, _ = media.decode("in.mp4", info)
    it = media.prefetch(gen, n=2)
    assert next(it) is not None
    it.close()                                                   # cancels the worker; it closes decode itself
    assert proc.terminate_calls == 1
    assert proc.waited


def test_prefetch_upstream_close_failure_surfaces_when_there_is_no_primary():
    """With no producer failure, a failing upstream close is itself the error the consumer sees."""
    src = _FailingCloseSource([0], close_error=OSError("close failed"))
    it = media.prefetch(src, n=2)
    assert next(it) == 0
    with pytest.raises(OSError, match="close failed"):
        next(it)
    assert src.close_calls == 1
