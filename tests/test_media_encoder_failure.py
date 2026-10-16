"""Encoder worker-failure handling (Stage 7 media reliability), deterministic with fake subprocesses.

A stdin failure (ffmpeg exited early -> ``BrokenPipeError``) must not wedge the bounded queue, must not be lost,
and a nonzero ffmpeg exit status must be surfaced. These tests never touch real ffmpeg so they cannot hang or
flake on process scheduling; the real byte/cadence behavior lives in ``tests/test_media_cadence.py``.
"""
import os
import stat
import subprocess
import threading
import time
from pathlib import Path

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


class _WritingStdin:
    """Real-file stdin double: every accepted buffer is appended to ffmpeg's temporary output."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "wb")
        self.writes = []
        self.closed = False
        self.close_calls = 0

    def write(self, b):
        self.writes.append(b)
        self._fh.write(b)
        self._fh.flush()

    def flush(self):
        self._fh.flush()

    def close(self):
        self.close_calls += 1
        self.closed = True
        self._fh.close()


class _BrokenWritingStdin(_WritingStdin):
    """Writes one buffer through to the temp, then fails like ffmpeg closing its stdin early."""

    def write(self, b):
        self.writes.append(b)
        self._fh.write(b)
        self._fh.flush()
        raise BrokenPipeError("ffmpeg exited")


class _CloseErrorWritingStdin(_WritingStdin):
    """Streams buffers through to the temp, then fails on close after releasing the real file handle.

    The handle is closed before raising so cleanup can still remove the temp and callers can observe the
    stream was really closed -- mirroring a stdin close failure that follows buffered output.
    """

    def close(self):
        self.close_calls += 1
        self.closed = True
        self._fh.close()
        raise OSError("stdin close failed")


class _WritingPopen:
    """Popen double that opens the encoder's temporary output and returns itself as the child handle."""

    def __init__(self, returncode=0, stdin_cls=_WritingStdin, fail=False):
        self.returncode = returncode
        self.stdin_cls = stdin_cls
        self.fail = fail
        self.stdin = None
        self.args = None
        self.waited = False
        self.tmp_mode = None

    def __call__(self, cmd, **kwargs):
        if self.fail:
            raise OSError("ffmpeg could not be started")
        self.args = list(cmd)
        self.tmp_mode = stat.S_IMODE(os.stat(cmd[-1]).st_mode)   # mode ffmpeg would find on the temp
        self.stdin = self.stdin_cls(cmd[-1])
        return self

    def wait(self, *a, **k):
        self.waited = True
        return self.returncode


def _hidden_temps(directory):
    """Encoder temp siblings are hidden and .mp4-suffixed, so this never counts a real output."""
    return sorted(p.name for p in Path(directory).iterdir()
                  if p.name.startswith(".") and p.name.endswith(".mp4"))


def test_encoder_close_atomically_replaces_existing_target(monkeypatch, tmp_path):
    """A zero-exit close publishes the temp over an existing target, leaving no hidden temp behind."""
    target = tmp_path / "out.mp4"
    target.write_bytes(b"PRE-EXISTING TARGET")
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(5)]
    enc = media.Encoder(str(target), 4, 4, 30)
    for frame in frames:
        enc.write(frame)
    assert enc.close() is None
    assert target.read_bytes() == b"".join(f.tobytes() for f in frames)   # atomically replaced
    assert _hidden_temps(tmp_path) == []                                  # temp consumed by os.replace
    assert proc.stdin.closed and proc.waited


def test_encoder_close_atomically_creates_missing_target(monkeypatch, tmp_path):
    """A zero-exit close creates a target that did not exist, again without a temp leak."""
    target = tmp_path / "new.mp4"
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(target), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 7, np.uint8))
    assert enc.close() is None
    assert target.read_bytes() == np.full((4, 4, 3), 7, np.uint8).tobytes()
    assert _hidden_temps(tmp_path) == []


def test_encoder_new_target_mode_honors_process_umask(monkeypatch, tmp_path):
    """A brand-new output gets the mode a fresh 0o666 file would get, matched against an OS-created reference."""
    reference = tmp_path / "reference.mode"
    fd = os.open(str(reference), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    os.close(fd)
    expected = stat.S_IMODE(os.stat(reference).st_mode)          # whatever this process's umask yields

    target = tmp_path / "new.mp4"
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(target), 4, 4, 30)
    assert proc.tmp_mode == expected                             # temp created umask-correct, no chmod guesswork
    enc.write(np.full((4, 4, 3), 7, np.uint8))
    assert enc.close() is None
    assert stat.S_IMODE(os.stat(target).st_mode) == expected     # publishing keeps the umask-derived mode


def test_encoder_existing_target_mode_is_preserved(monkeypatch, tmp_path):
    """os.replace swaps inodes, so the temp must carry the destination's mode or publishing would reset it."""
    target = tmp_path / "out.mp4"
    target.write_bytes(b"ORIGINAL")
    os.chmod(target, 0o640)
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(target), 4, 4, 30)
    assert proc.tmp_mode == 0o640                                # the temp already matches the destination
    enc.write(np.full((4, 4, 3), 5, np.uint8))
    assert enc.close() is None
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o640        # and the published file keeps it


def test_encoder_construction_never_reads_process_umask(monkeypatch, tmp_path):
    """The temp must be created without touching the process-global umask read/restore dance."""
    def boom(*a, **k):
        raise AssertionError("Encoder construction must not call os.umask")

    monkeypatch.setattr(media.os, "umask", boom)
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(tmp_path / "new.mp4"), 4, 4, 30)
    enc.write(_frame(1))
    assert enc.close() is None


def test_encoder_success_replaces_symlink_with_regular_file(monkeypatch, tmp_path):
    """On success the symlink entry (not its referent) becomes a regular file carrying the referent's mode."""
    referent = tmp_path / "real.mp4"
    referent.write_bytes(b"REFERENT")
    os.chmod(referent, 0o640)
    link = tmp_path / "link.mp4"
    link.symlink_to(referent.name)
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    enc = media.Encoder(str(link), 4, 4, 30)
    for frame in frames:
        enc.write(frame)
    assert enc.close() is None
    assert not link.is_symlink()                                 # os.replace swapped the link entry itself
    assert link.read_bytes() == b"".join(f.tobytes() for f in frames)
    assert stat.S_IMODE(os.stat(link).st_mode) == 0o640          # mode read through the old link
    assert referent.read_bytes() == b"REFERENT"                  # referent is never written through
    assert _hidden_temps(tmp_path) == []


def test_encoder_failure_preserves_symlink_and_referent(monkeypatch, tmp_path):
    """A failed encode leaves the symlink pointing at an unchanged referent, with no temp behind."""
    referent = tmp_path / "real.mp4"
    referent.write_bytes(b"REFERENT")
    link = tmp_path / "link.mp4"
    link.symlink_to(referent.name)
    proc = _WritingPopen(returncode=3)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(link), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 2, np.uint8))
    with pytest.raises(subprocess.CalledProcessError):
        enc.close()
    assert link.is_symlink() and os.readlink(link) == referent.name
    assert referent.read_bytes() == b"REFERENT"
    assert _hidden_temps(tmp_path) == []


def test_encoder_broken_pipe_preserves_target_and_removes_temp(monkeypatch, tmp_path):
    """A broken stdin pipe leaves an existing target byte-for-byte unchanged and drops the partial temp."""
    target = tmp_path / "out.mp4"
    target.write_bytes(b"ORIGINAL")
    proc = _WritingPopen(returncode=1, stdin_cls=_BrokenWritingStdin)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(target), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 3, np.uint8))
    with pytest.raises(BrokenPipeError):
        enc.close()
    assert target.read_bytes() == b"ORIGINAL"
    assert _hidden_temps(tmp_path) == []
    assert proc.stdin.closed and proc.waited


def test_encoder_stdin_close_failure_preserves_target_and_removes_temp(monkeypatch, tmp_path):
    """A stdin close failure after buffered output must not publish: close raises it, waits, and drops the temp.

    Covers the atomic-publication gap where every write succeeded and the child exits zero, but stdin's close
    fails. The stream handle is released before the raise, so the hidden temp is removable and the pre-existing
    target must survive byte-for-byte instead of being overwritten.
    """
    target = tmp_path / "out.mp4"
    target.write_bytes(b"ORIGINAL")
    proc = _WritingPopen(returncode=0, stdin_cls=_CloseErrorWritingStdin)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(target), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 6, np.uint8))
    with pytest.raises(OSError, match="stdin close failed"):
        enc.close()
    assert target.read_bytes() == b"ORIGINAL"     # a zero exit must not publish when stdin's close failed
    assert proc.stdin.closed                      # the real stream handle was released before the raise
    assert proc.waited                            # the child wait was still attempted during teardown
    assert _hidden_temps(tmp_path) == []          # the partial temp is removed, not leaked


def test_encoder_nonzero_exit_preserves_target_and_removes_temp(monkeypatch, tmp_path):
    """A nonzero child exit after a clean stdin must not publish: target survives, temp removed."""
    target = tmp_path / "out.mp4"
    target.write_bytes(b"ORIGINAL")
    proc = _WritingPopen(returncode=3)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    enc = media.Encoder(str(target), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 9, np.uint8))
    with pytest.raises(subprocess.CalledProcessError) as ei:
        enc.close()
    assert ei.value.returncode == 3
    assert target.read_bytes() == b"ORIGINAL"
    assert _hidden_temps(tmp_path) == []


def test_encoder_replace_failure_removes_temp_and_leaves_target(monkeypatch, tmp_path):
    """A failed os.replace must not publish: the pre-existing target (bytes and mode) survives, temp cleaned."""
    target = tmp_path / "out.mp4"
    target.write_bytes(b"ORIGINAL")
    os.chmod(target, 0o600)
    proc = _WritingPopen(returncode=0)
    monkeypatch.setattr(media.subprocess, "Popen", proc)

    def boom(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(media.os, "replace", boom)
    enc = media.Encoder(str(target), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 4, np.uint8))
    with pytest.raises(OSError, match="replace refused"):
        enc.close()
    assert target.read_bytes() == b"ORIGINAL"
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600       # a failed publish must not change the mode
    assert _hidden_temps(tmp_path) == []
    with pytest.raises(OSError, match="replace refused"):
        enc.close()                                          # idempotent: the same replace failure is re-raised


def test_encoder_popen_failure_removes_temp_and_leaves_target(monkeypatch, tmp_path):
    """A Popen failure during construction cleans its temp and leaves a pre-existing target untouched."""
    target = tmp_path / "out.mp4"
    target.write_bytes(b"ORIGINAL")
    proc = _WritingPopen(fail=True)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    with pytest.raises(OSError):
        media.Encoder(str(target), 4, 4, 30)
    assert target.read_bytes() == b"ORIGINAL"
    assert _hidden_temps(tmp_path) == []


def test_encoder_popen_failure_does_not_create_target(monkeypatch, tmp_path):
    """A Popen failure with no pre-existing target leaves no output and no temp at all."""
    target = tmp_path / "absent.mp4"
    proc = _WritingPopen(fail=True)
    monkeypatch.setattr(media.subprocess, "Popen", proc)
    with pytest.raises(OSError):
        media.Encoder(str(target), 4, 4, 30)
    assert not target.exists()
    assert _hidden_temps(tmp_path) == []


def test_encoder_success_keeps_bounded_pipeline_and_close_returns(monkeypatch, tmp_path):
    """Successful close is transparent: every frame reaches the sink in order and close returns None."""
    stdin = FakeStdin()
    proc = FakeProc(stdin, returncode=0)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder(str(tmp_path / "out.mp4"), 4, 4, 30)
    count = media.Encoder.QUEUE_SIZE * 3 + 2                    # more frames than the queue bound
    for i in range(count):
        enc.write(_frame(i))
    assert enc.close() is None
    assert len(stdin.writes) == count                           # nothing dropped by the bounded queue
    for i, blob in enumerate(stdin.writes):
        assert blob == _frame(i).tobytes()                      # raw bytes preserved, in order
    assert stdin.closed and proc.waited
    assert enc.q.unfinished_tasks == 0


def test_encoder_close_is_idempotent_after_success(monkeypatch, tmp_path):
    stdin = FakeStdin()
    _patch_proc(monkeypatch, FakeProc(stdin, returncode=0))
    enc = media.Encoder(str(tmp_path / "out.mp4"), 4, 4, 30)
    enc.write(_frame(1))
    enc.close()
    enc.close()                                                 # second close must not re-send or re-close
    assert stdin.close_calls == 1
    assert len(stdin.writes) == 1


def test_encoder_broken_pipe_full_queue_does_not_deadlock(monkeypatch, tmp_path):
    """With the writer parked and the bounded queue full, a stdin failure must still drain, not wedge."""
    stdin = GatedFailingStdin()
    proc = FakeProc(stdin, returncode=1)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder(str(tmp_path / "bad.mp4"), 4, 4, 30)
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


def test_encoder_write_after_failure_raises_instead_of_hanging(monkeypatch, tmp_path):
    stdin = BrokenStdin()
    _patch_proc(monkeypatch, FakeProc(stdin, returncode=1))
    enc = media.Encoder(str(tmp_path / "bad.mp4"), 4, 4, 30)
    enc.write(_frame(1))
    with pytest.raises(BrokenPipeError):
        enc.drain()                                             # barrier makes the writer failure observable
    with pytest.raises(BrokenPipeError):
        enc.write(_frame(2))                                    # subsequent writes raise immediately
    with pytest.raises(BrokenPipeError):
        enc.close()


def test_encoder_close_propagates_writer_error_without_drain(monkeypatch, tmp_path):
    """A writer failure discovered only during close must still be raised, not swallowed."""
    stdin = BrokenStdin()
    proc = FakeProc(stdin, returncode=1)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder(str(tmp_path / "bad.mp4"), 4, 4, 30)
    enc.write(_frame(1))
    with pytest.raises(BrokenPipeError):
        enc.close()
    assert stdin.closed and proc.waited
    with pytest.raises(BrokenPipeError):
        enc.close()                                             # idempotent: re-raises the same failure
    assert stdin.close_calls == 1


def test_encoder_close_surfaces_nonzero_child_exit(monkeypatch, tmp_path):
    stdin = FakeStdin()
    proc = FakeProc(stdin, returncode=3, args=("ffmpeg", "-i", "-"))
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder(str(tmp_path / "bad.mp4"), 4, 4, 30)
    for i in range(3):
        enc.write(_frame(i))
    with pytest.raises(subprocess.CalledProcessError) as ei:
        enc.close()
    assert ei.value.returncode == 3
    assert len(stdin.writes) == 3                               # stdin was fully written before waiting
    assert stdin.closed and proc.waited


def test_encoder_close_waits_for_child_even_if_stdin_close_fails(monkeypatch, tmp_path):
    """close must attempt the full teardown (close + wait) before raising the first failure."""
    class CloseBoomStdin(FakeStdin):
        def close(self):
            self.close_calls += 1
            raise OSError("stdin close failed")

    stdin = CloseBoomStdin()
    proc = FakeProc(stdin, returncode=0)
    _patch_proc(monkeypatch, proc)
    enc = media.Encoder(str(tmp_path / "bad.mp4"), 4, 4, 30)
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
