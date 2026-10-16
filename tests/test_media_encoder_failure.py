"""Encoder worker-failure handling (Stage 7 media reliability), deterministic with fake subprocesses.

A stdin failure (ffmpeg exited early -> ``BrokenPipeError``) must not wedge the bounded queue, must not be lost,
and a nonzero ffmpeg exit status must be surfaced. These tests never touch real ffmpeg so they cannot hang or
flake on process scheduling; the real byte/cadence behavior lives in ``tests/test_media_cadence.py``.
"""
import subprocess
import threading
import time

import numpy as np
import pytest

from mlxstab import media


def _frame(value):
    return np.full((4, 4, 3), value % 255, np.uint8)


def _wait(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class FakeStdin:
    """Accepts every write; records exact bytes, whether it was closed, and how many times."""

    def __init__(self):
        self.writes = []
        self.closed = False
        self.close_calls = 0

    def write(self, b):
        self.writes.append(b)

    def flush(self):
        pass

    def close(self):
        self.close_calls += 1
        self.closed = True


class FakeProc:
    def __init__(self, stdin, returncode=0, args=("ffmpeg",)):
        self.stdin = stdin
        self.returncode = returncode
        self.args = args
        self.waited = False

    def wait(self, *a, **k):
        self.waited = True
        return self.returncode


class BrokenStdin(FakeStdin):
    """Every write fails, as when ffmpeg has already exited and closed its stdin."""

    def write(self, b):
        self.writes.append(b)
        raise BrokenPipeError("ffmpeg exited")


class GatedFailingStdin(FakeStdin):
    """First write parks until released, then every write fails; lets a test fill the bounded queue first."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.gate = threading.Event()

    def write(self, b):
        self.started.set()
        self.gate.wait(5.0)
        self.writes.append(b)
        raise BrokenPipeError("ffmpeg exited")


def _patch_proc(monkeypatch, proc):
    monkeypatch.setattr(media.subprocess, "Popen", lambda *a, **k: proc)


def test_encoder_success_keeps_bounded_pipeline_and_close_returns(monkeypatch):
    """Successful close is transparent: every frame reaches the sink in order and close returns None."""
    stdin = FakeStdin()
    proc = FakeProc(stdin, returncode=0)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder("out.mp4", 4, 4, 30)
    count = media.Encoder.QUEUE_SIZE * 3 + 2                    # more frames than the queue bound
    for i in range(count):
        enc.write(_frame(i))
    assert enc.close() is None
    assert len(stdin.writes) == count                           # nothing dropped by the bounded queue
    for i, blob in enumerate(stdin.writes):
        assert blob == _frame(i).tobytes()                      # raw bytes preserved, in order
    assert stdin.closed and proc.waited
    assert enc.q.unfinished_tasks == 0


def test_encoder_close_is_idempotent_after_success(monkeypatch):
    stdin = FakeStdin()
    _patch_proc(monkeypatch, FakeProc(stdin, returncode=0))
    enc = media.Encoder("out.mp4", 4, 4, 30)
    enc.write(_frame(1))
    enc.close()
    enc.close()                                                 # second close must not re-send or re-close
    assert stdin.close_calls == 1
    assert len(stdin.writes) == 1


def test_encoder_broken_pipe_full_queue_does_not_deadlock(monkeypatch):
    """With the writer parked and the bounded queue full, a stdin failure must still drain, not wedge."""
    stdin = GatedFailingStdin()
    proc = FakeProc(stdin, returncode=1)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder("bad.mp4", 4, 4, 30)
    for i in range(media.Encoder.QUEUE_SIZE):
        enc.write(_frame(i))
    assert stdin.started.wait(5.0)                              # writer is parked on the first frame
    assert enc.q.unfinished_tasks == media.Encoder.QUEUE_SIZE   # queue is full

    stdin.gate.set()                                            # first write fails; writer must keep consuming
    assert _wait(lambda: enc._writer_error is not None)
    assert _wait(lambda: enc.q.unfinished_tasks == 0)           # nothing left unjoined/abandoned

    with pytest.raises(BrokenPipeError):
        enc.write(_frame(99))                                   # write surfaces it instead of blocking
    with pytest.raises(BrokenPipeError):
        enc.close()                                             # close still cleans up, then raises
    assert stdin.closed and proc.waited
    assert not enc.t.is_alive()


def test_encoder_write_after_failure_raises_instead_of_hanging(monkeypatch):
    stdin = BrokenStdin()
    _patch_proc(monkeypatch, FakeProc(stdin, returncode=1))
    enc = media.Encoder("bad.mp4", 4, 4, 30)
    enc.write(_frame(1))
    with pytest.raises(BrokenPipeError):
        enc.drain()                                             # barrier makes the writer failure observable
    with pytest.raises(BrokenPipeError):
        enc.write(_frame(2))                                    # subsequent writes raise immediately
    with pytest.raises(BrokenPipeError):
        enc.close()


def test_encoder_close_propagates_writer_error_without_drain(monkeypatch):
    """A writer failure discovered only during close must still be raised, not swallowed."""
    stdin = BrokenStdin()
    proc = FakeProc(stdin, returncode=1)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder("bad.mp4", 4, 4, 30)
    enc.write(_frame(1))
    with pytest.raises(BrokenPipeError):
        enc.close()
    assert stdin.closed and proc.waited
    with pytest.raises(BrokenPipeError):
        enc.close()                                             # idempotent: re-raises the same failure
    assert stdin.close_calls == 1


def test_encoder_close_surfaces_nonzero_child_exit(monkeypatch):
    stdin = FakeStdin()
    proc = FakeProc(stdin, returncode=3, args=("ffmpeg", "-i", "-"))
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder("bad.mp4", 4, 4, 30)
    for i in range(3):
        enc.write(_frame(i))
    with pytest.raises(subprocess.CalledProcessError) as ei:
        enc.close()
    assert ei.value.returncode == 3
    assert len(stdin.writes) == 3                               # stdin was fully written before waiting
    assert stdin.closed and proc.waited


def test_encoder_close_waits_for_child_even_if_stdin_close_fails(monkeypatch):
    """close must attempt the full teardown (close + wait) before raising the first failure."""
    class CloseBoomStdin(FakeStdin):
        def close(self):
            self.close_calls += 1
            raise OSError("stdin close failed")

    stdin = CloseBoomStdin()
    proc = FakeProc(stdin, returncode=0)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder("bad.mp4", 4, 4, 30)
    enc.write(_frame(1))
    with pytest.raises(OSError):
        enc.close()
    assert proc.waited                                          # a close failure must not skip waiting on ffmpeg


class _RecordingEncoder:
    """Minimal close-only encoder recording close order and raising a preset failure."""

    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.error is not None:
            raise self.error


def test_close_all_closes_every_sibling_then_raises_the_first_failure():
    """A failing close must not skip its siblings; the first failure is raised only after all are attempted."""
    first = ValueError("first close failure")
    second = OSError("later close failure")
    a, b, c = _RecordingEncoder("a"), _RecordingEncoder("b", first), _RecordingEncoder("c", second)
    with pytest.raises(ValueError) as ei:
        media.close_all([a, b, c])
    assert ei.value is first                                   # the first failure, not the later one
    assert [e.close_calls for e in (a, b, c)] == [1, 1, 1]     # every sibling was closed


def test_close_all_skips_none_and_returns_none_on_success():
    a, b = _RecordingEncoder("a"), _RecordingEncoder("b")
    assert media.close_all([a, None, b]) is None
    assert [e.close_calls for e in (a, b)] == [1, 1]
    assert media.close_all([]) is None
    assert media.close_all([None, None]) is None


def test_close_all_does_not_retry_a_failed_encoder():
    """close_all itself never retries: exactly one close per non-None encoder, even after a failure."""
    failed = _RecordingEncoder("failed", BrokenPipeError("worker died"))
    survivors = [_RecordingEncoder(f"s{i}") for i in range(3)]
    with pytest.raises(BrokenPipeError):
        media.close_all([failed, *survivors])
    assert failed.close_calls == 1
    assert [e.close_calls for e in survivors] == [1, 1, 1]
