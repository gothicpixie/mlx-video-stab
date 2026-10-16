"""Focused tests for the offline KLT-vs-captured-RAFT agreement audit.

All inputs are synthetic: a textured canvas with a known translation path plus a motion payload built from the
same translations. No RAFT, no media decode, no private footage.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from bench import track_agreement_audit as TA
from mlxstab import stabilize as S
from mlxstab.motion import Fit

cv2 = pytest.importorskip("cv2")


def _frames(W, H, steps, seed=0, pad=64):
    """Textured frames whose content shifts by ``steps[t]`` pixels (positive = right)."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (H + 2 * pad, W + 2 * pad, 3), dtype=np.uint8), (3, 3), 0.6)
    y0, x0 = pad, pad
    return np.ascontiguousarray([base[y0:y0 + H, x0 - int(s):x0 - int(s) + W] for s in steps])


def _frames_sha256(frames):
    h = hashlib.sha256()
    for frame in frames:
        h.update(np.ascontiguousarray(frame).tobytes())
    return h.hexdigest()


def _motion(W, H, fps, step):
    fits = [Fit(np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.zeros((2, 2, 2)), np.zeros((1, 2)), 1.0, 0.0, False) for dx in step]
    return S.Motion(W, H, fps, fits=fits, stride=1, n_frames=len(step) + 1)


def test_matching_transform_tracks_have_near_zero_gap():
    W = H = 120
    steps = [0, 2, 1, 4, 3, 6]                      # a jittery but exactly-known integer path
    frames = _frames(W, H, steps)
    mot = _motion(W, H, 30.0, np.diff(steps))
    rep = TA.run_audit(frames, mot, {"seed": 0})
    assert rep["schema"] == TA.SCHEMA
    assert rep["track"]["fallbacks"] == 0 and rep["track"]["pairs"] == 5
    assert rep["agreement"]["point_displacement_epe_px"]["mean"] < 0.5
    assert abs(rep["agreement"]["jitter_relative_gap"]) < 0.2
    assert rep["agreement"]["scale_delta_abs"]["max"] < 0.01


def test_pure_pan_reports_undefined_jitter_gap():
    """A constant-velocity pan has no residual jitter, so the ratio is undefined and must be null, not huge."""
    W = H = 120
    frames = _frames(W, H, [0, 1, 2, 3, 4, 5])
    rep = TA.run_audit(frames, _motion(W, H, 30.0, [1, 1, 1, 1, 1]), {"seed": 0})
    assert rep["agreement"]["jitter_relative_gap"] is None
    assert rep["raft"]["jitter_px"] < 0.05


def test_run_audit_refuses_frame_count_and_size_mismatches():
    frames = _frames(100, 80, [0, 1, 2])
    mot = _motion(100, 80, 30.0, [1, 1])
    with pytest.raises(ValueError, match="frame count"):
        TA.run_audit(frames[:2], mot, {"seed": 0})
    with pytest.raises(ValueError, match="frame size"):
        TA.run_audit(_frames(96, 80, [0, 1, 2]), mot, {"seed": 0})
    with pytest.raises(ValueError, match="no fitted pairs"):
        TA.run_audit(frames[:1], _motion(100, 80, 30.0, []), {"seed": 0})


def test_cli_writes_atomic_path_free_finite_report(tmp_path):
    W = H = 120
    steps = [0, 1, 2, 3]
    frames = _frames(W, H, steps)
    fp = tmp_path / "frames.npz"
    frame_meta = {"schema": "test.frames/1", "count": len(frames), "width": W, "height": H, "fps": 30.0,
                  "frames_sha256": _frames_sha256(frames), "source_sha256": "b" * 64,
                  "window": {"start_s": 12.0, "duration_s": len(frames) / 30.0}}
    np.savez_compressed(fp, frames=frames, meta=np.array(json.dumps(frame_meta)))
    mp = tmp_path / "motion_r0.npz"
    S.save_motion_npz(_motion(W, H, 30.0, np.diff(steps)), mp)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "success",
        "source": {"sha256": frame_meta["source_sha256"], "window": frame_meta["window"],
                   "decoded": {"frames": len(frames), "width": W, "height": H, "fps": 30.0}},
        "outputs": {"motion": "motion_r0.npz"}}))
    out = tmp_path / "report.json"

    assert TA.main(["--frames-npz", str(fp), "--motion-npz", str(mp), "--manifest", str(manifest),
                    "--out", str(out)]) == 0
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert str(tmp_path) not in text                                  # path-free
    rep = json.loads(text)
    assert rep["status"] == "ok" and rep["inputs"]["motion_schema"] == S.MOTION_SCHEMA
    assert rep["inputs"]["verified_frame_window"]["source_sha256"] == frame_meta["source_sha256"]
    assert rep["inputs"]["manifest_sha256"]
    assert rep["agreement"]["point_displacement_epe_px"]["count"] > 0


def test_manifest_rejects_frame_cache_from_a_different_source(tmp_path, capsys):
    W = H = 120
    frames = _frames(W, H, [0, 1, 2, 3])
    frame_meta = {"schema": "test.frames/1", "count": len(frames), "width": W, "height": H, "fps": 30.0,
                  "frames_sha256": _frames_sha256(frames), "source_sha256": "b" * 64,
                  "window": {"start_s": 12.0, "duration_s": len(frames) / 30.0}}
    fp = tmp_path / "frames.npz"; np.savez_compressed(fp, frames=frames, meta=np.array(json.dumps(frame_meta)))
    mp = tmp_path / "motion_r0.npz"; S.save_motion_npz(_motion(W, H, 30.0, [1, 1, 1]), mp)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "success",
        "source": {"sha256": "c" * 64, "window": frame_meta["window"],
                   "decoded": {"frames": len(frames), "width": W, "height": H, "fps": 30.0}},
        "outputs": {"motion": "motion_r0.npz"}}))
    out = tmp_path / "report.json"
    assert TA.main(["--frames-npz", str(fp), "--motion-npz", str(mp), "--manifest", str(manifest),
                    "--out", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "mismatch"
    assert not out.exists()


def test_cli_rejects_frame_cache_with_wrong_content_digest(tmp_path, capsys):
    W = H = 100
    frames = _frames(W, H, [0, 1, 2])
    bad_meta = {"frames_sha256": "0" * 64}
    fp = tmp_path / "frames.npz"; np.savez_compressed(fp, frames=frames, meta=np.array(json.dumps(bad_meta)))
    mp = tmp_path / "motion_r0.npz"; S.save_motion_npz(_motion(W, H, 30.0, [1, 1]), mp)
    out = tmp_path / "report.json"
    assert TA.main(["--frames-npz", str(fp), "--motion-npz", str(mp), "--out", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "frames_digest_mismatch"
    assert not out.exists()


def test_cli_reports_stable_codes_for_bad_inputs(tmp_path, capsys):
    W = H = 100
    frames = _frames(W, H, [0, 1, 2])
    fp = tmp_path / "frames.npz"; np.savez(fp, frames=frames)
    mp = tmp_path / "motion_r0.npz"; S.save_motion_npz(_motion(W, H, 30.0, [1, 1]), mp)
    out = tmp_path / "r.json"

    assert TA.main(["--frames-npz", str(tmp_path / "nope.npz"), "--motion-npz", str(mp), "--out", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "frames_unreadable"

    fp2 = tmp_path / "wrong.npz"; np.savez(fp2, frames=frames[:1])
    assert TA.main(["--frames-npz", str(fp2), "--motion-npz", str(mp), "--out", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "mismatch"

    bad = tmp_path / "bad.npz"; np.savez(bad, not_frames=frames)
    assert TA.main(["--frames-npz", str(bad), "--motion-npz", str(mp), "--out", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "frames_missing"
