"""Real (cheap, CPU-only) ffmpeg regression for output cadence conversion.

The encoder must convert frame rate exactly. Relying on output ``-r`` alone produced 32 frames for a 1 s
60 fps clip at 30 fps; these tests pin the explicit ``fps`` filter path and mask/video alignment.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from mlxstab import media

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                               reason="ffmpeg/ffprobe required")


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames,avg_frame_rate,r_frame_rate,duration,start_time", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def _write_frames(path, W, H, T, in_fps, out_fps=None, gray=False):
    enc = media.Encoder(str(path), W, H, in_fps, gray=gray, out_fps=out_fps, crf=28, preset="ultrafast")
    for i in range(T):
        frame = np.full((H, W), i * 3 % 255, np.uint8) if gray else np.full((H, W, 3), i * 3 % 255, np.uint8)
        enc.write(frame)
    enc.close()


def test_encoder_60_to_30_is_exact(tmp_path):
    p = tmp_path / "a.mp4"
    _write_frames(p, 64, 48, 60, 60, out_fps=30)
    s = _probe(p)
    assert int(s["nb_read_frames"]) == 30
    assert s["avg_frame_rate"] == "30/1"
    assert abs(float(s["duration"]) - 1.0) < 0.05


def test_encoder_ntsc_60000_to_30000_is_exact(tmp_path):
    p = tmp_path / "b.mp4"
    _write_frames(p, 64, 48, 60, "60000/1001", out_fps="30000/1001")
    s = _probe(p)
    assert int(s["nb_read_frames"]) == 30
    assert s["avg_frame_rate"] == "30000/1001"
    assert abs(float(s["duration"]) - 60 / (60000 / 1001)) < 0.05


def test_encoder_native_rate_keeps_all_frames(tmp_path):
    p = tmp_path / "c.mp4"
    _write_frames(p, 64, 48, 60, 60)
    assert int(_probe(p)["nb_read_frames"]) == 60


def test_mask_matches_video_cadence_under_conversion(tmp_path):
    v = tmp_path / "v.mp4"; m = tmp_path / "m.mp4"
    _write_frames(v, 64, 48, 60, 60, out_fps=30)
    _write_frames(m, 64, 48, 60, 60, out_fps=30, gray=True)
    assert int(_probe(v)["nb_read_frames"]) == int(_probe(m)["nb_read_frames"]) == 30


def test_stabilize_renders_60_and_encodes_30_without_raft(tmp_path):
    """End-to-end: fake zero flow + real ffmpeg must render 60 frames but encode 30 (no RAFT)."""
    import mlx.core as mx
    from mlxstab.stabilize import Config, stabilize

    W, H, T = 32, 24, 60
    src = tmp_path / "in60.mp4"
    _write_frames(src, W, H, T, 60)

    class FakeFlow:
        def reset(self):
            pass

        def bidir(self, a, b):
            return mx.zeros((H, W, 2)), mx.zeros((H, W, 2))

    out = tmp_path / "out30.mp4"
    rep = stabilize(str(src), str(out), None, None, Config(mesh=(0, 0), rs_rows=0, flow_iters=2, flow_downscale=1),
                    flow_fn=FakeFlow(), fps_out="30000/1001", log=lambda m: None)
    assert rep["rendered_frames"] == 60
    assert "encoded_frames" not in rep                      # stabilize must not claim a false encoded count
    assert int(_probe(out)["nb_read_frames"]) == 30
    assert _probe(out)["avg_frame_rate"] == "30000/1001"


def test_encoder_drain_blocks_until_queued_writes_are_consumed(tmp_path, monkeypatch):
    """``Encoder.drain`` must not return while frames are still queued for the writer thread.

    A gated fake stdin holds the writer thread inside ``write`` so the queue provably stays non-empty; the
    barrier thread must not complete until the gate opens and every queued frame has been consumed.
    """
    import threading
    import time

    class GatedStdin:
        def __init__(self):
            self.writes = []
            self.gate = threading.Event()

        def write(self, b):
            self.gate.wait(5.0)
            self.writes.append(b)

        def flush(self):
            pass

        def close(self):
            pass

    class FakeProc:
        def __init__(self, *a, **k):
            self.stdin = GatedStdin()

        def wait(self, *a, **k):
            return 0

    proc = FakeProc()
    monkeypatch.setattr(media.subprocess, "Popen", lambda *a, **k: proc)
    enc = media.Encoder(str(tmp_path / "q.mp4"), 4, 4, 30)
    for i in range(4):
        enc.write(np.full((4, 4, 3), i, np.uint8))

    drained = threading.Event()
    threading.Thread(target=lambda: (enc.drain(), drained.set()), daemon=True).start()

    time.sleep(0.05)
    assert not drained.is_set()                 # gate closed: the writer is blocked, frames still queued
    assert enc.q.unfinished_tasks == 4
    proc.stdin.gate.set()                       # release the writer
    assert drained.wait(5.0)                    # drain returns only after the queue is consumed
    assert enc.q.unfinished_tasks == 0
    assert len(proc.stdin.writes) == 4          # every queued frame reached the sink, none dropped
    enc.close()


def test_encoder_drain_raises_writer_error_instead_of_hanging(tmp_path, monkeypatch):
    """If the writer thread failed, ``drain`` surfaces that error rather than blocking forever on the queue."""
    import threading

    class BrokenStdin:
        def write(self, b):
            raise BrokenPipeError("ffmpeg exited")

        def flush(self):
            pass

        def close(self):
            pass

    class FakeProc:
        def __init__(self, *a, **k):
            self.stdin = BrokenStdin()

        def wait(self, *a, **k):
            return 1

    monkeypatch.setattr(media.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(threading, "excepthook", lambda args: None)   # swallow the daemon-thread traceback
    enc = media.Encoder(str(tmp_path / "bad.mp4"), 4, 4, 30)
    enc.write(np.full((4, 4, 3), 1, np.uint8))
    with pytest.raises(BrokenPipeError):
        enc.drain()


def test_encoder_drain_midstream_keeps_cadence_exact(tmp_path):
    """A mid-stream chunk barrier must be transparent to the encoded stream (cadence unchanged)."""
    p = tmp_path / "d.mp4"
    enc = media.Encoder(str(p), 64, 48, 60, out_fps=30, crf=28, preset="ultrafast")
    for i in range(60):
        enc.write(np.full((48, 64, 3), i * 3 % 255, np.uint8))
        if (i + 1) % 15 == 0:
            enc.drain()                         # barrier at frames 15/30/45
    enc.close()
    s = _probe(p)
    assert int(s["nb_read_frames"]) == 30
    assert s["avg_frame_rate"] == "30/1"
    assert abs(float(s["duration"]) - 1.0) < 0.05
