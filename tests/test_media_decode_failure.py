"""Decode worker-failure handling (Stage 7 media reliability), deterministic with fake subprocesses.

``mlxstab.media.decode`` must reap its ffmpeg child and close its stdout on every path: a full natural end of
stream waits for ffmpeg and raises on a nonzero exit, a trailing partial raw RGB frame is reported instead of
being silently truncated, and an early close, consumer throw or read failure terminates a still-running child
while preserving the primary exception. These tests never touch real ffmpeg; the real byte/cadence behavior
lives in ``tests/test_media_cadence.py``.
"""
import subprocess

import numpy as np
import pytest

from mlxstab import media

W, H = 3, 2
FRAME_BYTES = W * H * 3
INFO = {"width": W, "height": H, "fps": 30.0, "interlaced": False}


def _frame(value):
    return np.full((H, W, 3), value % 255, np.uint8)


class FakeStream:
    """Serves queued read results (bytes or an exception) then EOF; records close calls."""

    def __init__(self, reads, close_error=None):
        self._reads = list(reads)
        self.close_error = close_error
        self.closed = False
        self.close_calls = 0

    def read(self, n):
        if self._reads:
            item = self._reads.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return b""

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakePopen:
    """Minimal child double: controllable exit status, liveness and teardown failures."""

    def __init__(self, stdout, returncode=0, args=("ffmpeg",), running=False,
                 terminate_error=None, wait_error=None):
        self.stdout = stdout
        self.returncode = returncode
        self.args = args
        self._running = running
        self.terminate_error = terminate_error
        self.wait_error = wait_error
        self.terminate_calls = 0
        self.wait_calls = 0
        self.waited = False
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        return None if self._running else self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_error is not None:
            raise self.terminate_error
        self._running = False

    def wait(self, *a, **k):
        self.wait_calls += 1
        self.waited = True
        if self.wait_error is not None:
            raise self.wait_error
        self._running = False
        return self.returncode


def _patch(monkeypatch, proc):
    monkeypatch.setattr(media.subprocess, "Popen", lambda *a, **k: proc)


def test_decode_complete_stream_yields_frames_and_reaps(monkeypatch):
    """Normal output is byte/shape identical and the child is waited without being signalled."""
    frames = [_frame(1), _frame(2), _frame(3)]
    stream = FakeStream([f.tobytes() for f in frames])
    proc = FakePopen(stream, returncode=0, running=False)
    _patch(monkeypatch, proc)
    gen, fps, (w, h) = media.decode("in.mp4", INFO)
    assert fps == 30.0 and (w, h) == (W, H)
    got = list(gen)
    assert len(got) == len(frames)
    for g, f in zip(got, frames):
        assert g.shape == (H, W, 3) and g.dtype == np.uint8
        assert np.array_equal(g, f)
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 0 and proc.poll_calls == 0   # natural EOF never signals the child


def test_decode_nonzero_natural_exit_raises_called_process_error(monkeypatch):
    """A full stream followed by a nonzero ffmpeg exit must surface CalledProcessError, not end quietly."""
    stream = FakeStream([_frame(4).tobytes()])
    proc = FakePopen(stream, returncode=7, args=("ffmpeg", "-i", "in.mp4"), running=False)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    with pytest.raises(subprocess.CalledProcessError) as ei:
        list(gen)
    assert ei.value.returncode == 7
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 0


def test_decode_partial_trailing_frame_raises_instead_of_truncating(monkeypatch):
    """A short trailing read is a clear error; the yielded frames before it stay intact."""
    good = _frame(5).tobytes()
    partial = b"\x00" * (FRAME_BYTES - 1)
    stream = FakeStream([good, partial])
    proc = FakePopen(stream, returncode=1, running=True)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    it = iter(gen)
    assert np.array_equal(next(it), _frame(5))
    with pytest.raises(RuntimeError, match="partial"):
        next(it)
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 1                            # the child had not finished cleanly


def test_decode_early_close_terminates_and_reaps(monkeypatch):
    """Closing the generator early stops the still-running child and waits for it."""
    frames = [_frame(1), _frame(2)]
    stream = FakeStream([f.tobytes() for f in frames])
    proc = FakePopen(stream, returncode=0, running=True)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    assert np.array_equal(next(gen), frames[0])
    gen.close()
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 1


def test_decode_early_close_does_not_signal_an_already_exited_child(monkeypatch):
    """A child that already exited is reaped but not signalled."""
    stream = FakeStream([_frame(1).tobytes()])
    proc = FakePopen(stream, returncode=0, running=False)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    next(gen)
    gen.close()
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 0 and proc.poll_calls == 1


def test_decode_early_close_preserves_generator_exit_when_cleanup_fails(monkeypatch):
    """GeneratorExit must survive a failing terminate/wait; ``close`` still attempts both."""
    stream = FakeStream([_frame(1).tobytes()])
    proc = FakePopen(stream, returncode=0, running=True, terminate_error=OSError("terminate failed"),
                     wait_error=OSError("wait failed"))
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    next(gen)
    gen.close()                                                 # must not raise the cleanup OSError
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 1


def test_decode_consumer_throw_terminates_child_and_preserves_exception(monkeypatch):
    """An exception thrown into the generator is primary: it is re-raised after cleanup."""
    stream = FakeStream([_frame(1).tobytes(), _frame(2).tobytes()])
    proc = FakePopen(stream, returncode=0, running=True)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    next(gen)
    boom = LookupError("consumer exploded")
    with pytest.raises(LookupError) as ei:
        gen.throw(boom)
    assert ei.value is boom
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 1


def test_decode_read_error_is_not_masked_by_cleanup_failure(monkeypatch):
    """The primary read error must win over a failing terminate/wait."""
    read_error = ValueError("read exploded")
    cleanup_error = KeyError("cleanup exploded")
    stream = FakeStream([_frame(1).tobytes(), read_error])
    proc = FakePopen(stream, returncode=0, running=True, terminate_error=cleanup_error,
                     wait_error=cleanup_error)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    it = iter(gen)
    next(it)
    with pytest.raises(ValueError) as ei:
        next(it)
    assert ei.value is read_error
    assert stream.closed and proc.waited
    assert proc.terminate_calls == 1


def test_decode_stdout_close_failure_is_surfaced_on_natural_eof(monkeypatch):
    """A stdout close failure is a real failure and is raised once the child has been reaped."""
    close_error = OSError("stdout close failed")
    stream = FakeStream([_frame(1).tobytes()], close_error=close_error)
    proc = FakePopen(stream, returncode=0, running=False)
    _patch(monkeypatch, proc)
    gen, _, _ = media.decode("in.mp4", INFO)
    with pytest.raises(OSError) as ei:
        list(gen)
    assert ei.value is close_error
    assert stream.closed and proc.waited                          # reaping is not skipped
