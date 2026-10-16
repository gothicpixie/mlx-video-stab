"""Focused tests for the research-only blinded A/B real-footage review builder.

Coverage: ``frame_center`` / ``composed_translation_path`` sign on pure translations built from real
``mlxstab.motion.Fit`` objects, ``scale_path``, ``correction_metrics`` and ``support_summary``,
``resize_bilinear`` / ``warp_panel`` / the A/B bitmap labels, the deterministic ``blind_assignment`` and the
per-run ``derive_blinding_seed`` provenance derivation, ``manifest_provenance`` / ``load_review_inputs``
refusals for missing and mismatched provenance, and a fully mocked ``run_review`` that checks four outputs,
the derived per-run side mapping and the absence of any quality score.
Pure NumPy plus the motion npz writer: no MLX, no FFmpeg, no production default is changed.
"""
import json
import math

import numpy as np
import pytest

from mlxstab import media
from mlxstab import stabilize as S
from mlxstab.motion import Fit

from bench import multiscale_motion_audit as A
from bench import multiscale_real_review as R
from bench.manifest import sha256_file

BWDIF = R.BWDIF_VF


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _all_keys(value)


def _assert_json_native(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float in report: {obj!r}"
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert isinstance(key, str), f"non-string key {key!r}"
            _assert_json_native(value)
        return
    if isinstance(obj, list):
        for value in obj:
            _assert_json_native(value)
        return
    raise AssertionError(f"non-JSON-native type {type(obj)!r}: {obj!r}")


def _translation_fit(tx, ty):
    """A real Fit whose global H is a pure (tx, ty) translation and zero residual motion."""
    H = np.eye(3)
    H[0, 2], H[1, 2] = float(tx), float(ty)
    return Fit(H, np.zeros((3, 3, 2)), np.zeros((1, 2)), 0.9, 0.0, False)


def _rotation_about_center(width, height, degrees):
    theta = math.radians(degrees)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    to_o = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0], [0, 0, 1]], np.float64)
    from_o = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    return from_o @ rot @ to_o


def _build_run(tmp_path, *, W=64, H=48, T=8, fps=30.0, motion_mutate=None, manifest_mutate=None):
    """Write a synthetic run directory with a valid motion payload and manifest; return its pieces."""
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.bin"
    source.write_bytes(b"fake-interlaced-source")
    fits = [_translation_fit(0.6 * math.sin(0.5 * k) + 0.5, 0.3 * math.cos(0.7 * k)) for k in range(T - 1)]
    mot = S.Motion(W, H, fps, fits=fits, stride=1, n_frames=T)
    motion_path = run / R.DEFAULT_MOTION_NAME
    S.save_motion_npz(mot, str(motion_path), S.Config(mesh=(2, 2), rs_rows=1))
    if motion_mutate is not None:
        motion_mutate(run, motion_path)

    man = {
        "schema": "mlxstab.bench.manifest/1",
        "status": "success",
        "run_id": "synthetic-review",
        "case": {"name": "baseline"},
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "window": {"start_s": 0.0, "duration_s": T / fps},
            "preprocessing": {"interlaced": True, "deinterlace": BWDIF, "field_order": "tt"},
            "decoded": {"width": W, "height": H, "fps": fps, "frames": T},
        },
        "outputs": {"motion": str(motion_path), "plan": None},
        "experiment_identity": {"capture": {"dump_motion": True, "dump_plan": False}},
    }
    if manifest_mutate is not None:
        manifest_mutate(man)
    (run / R.MANIFEST_NAME).write_text(json.dumps(man), encoding="utf-8")
    return run, motion_path, man, source


def _fake_frames(T, H, W):
    frames = []
    for i in range(T):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :, 0] = (i * 17) % 256
        frame[:, :, 1] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
        frames.append(frame)
    return frames


class _FakeEncoder:
    instances = []

    def __init__(self, path, width, height, fps, **kwargs):
        self.path = str(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
        self.closed = False
        _FakeEncoder.instances.append(self)

    def write(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------------------
# frame centre and composed translation path (real Fit objects)
# --------------------------------------------------------------------------------------

def test_frame_center_is_the_pixel_centre_grid():
    assert np.array_equal(R.frame_center(1, 1), [0.0, 0.0])
    assert np.allclose(R.frame_center(64, 48), [(64 - 1) / 2.0, (48 - 1) / 2.0])
    with pytest.raises(ValueError):
        R.frame_center(0, 4)
    with pytest.raises(ValueError):
        R.frame_center(4.0, 4)


def test_composed_translation_path_sign_on_pure_translation():
    W, H, T = 64, 48, 5
    tx, ty = 2.0, 1.0
    fits = [_translation_fit(tx, ty) for _ in range(T - 1)]
    path = R.composed_translation_path(fits, W, H, T)
    assert path.shape == (T, 2)
    assert np.allclose(path[0], 0.0)                      # frame 0 is identity
    for t in range(T):
        # Psi_t(center) = center + t*(tx, ty), so C_t = center - Psi_t(center) = -t*(tx, ty).
        assert np.allclose(path[t], -t * np.array([tx, ty]))
    # Negative shifts follow the same sign convention.
    neg = R.composed_translation_path([_translation_fit(-3.0, 0.5) for _ in range(T - 1)], W, H, T)
    assert np.allclose(neg[1], [3.0, -0.5])
    with pytest.raises(ValueError):
        R.composed_translation_path(fits, W, H, T + 1)     # n_frames must be len(fits) + 1


def test_composed_translation_path_is_translation_only_for_rotation():
    W, H, T = 64, 48, 4
    rot = _rotation_about_center(W, H, 10.0)
    fits = [Fit(rot, np.zeros((3, 3, 2)), np.zeros((1, 2)), 0.9, 0.0, False) for _ in range(T - 1)]
    # The rotation fixes the frame centre, so the translation-only reduction reports no correction.
    assert np.allclose(R.composed_translation_path(fits, W, H, T), 0.0)


def test_scale_path_scales_each_channel_independently():
    path = np.array([[10.0, 20.0], [-4.0, 8.0]])
    scaled = R.scale_path(path, 100, 50, 480, 270)
    assert np.allclose(scaled, [[48.0, 108.0], [-19.2, 43.2]])
    with pytest.raises(ValueError):
        R.scale_path(np.zeros(3), 100, 50, 480, 270)


# --------------------------------------------------------------------------------------
# correction metrics and support summary
# --------------------------------------------------------------------------------------

def test_correction_metrics_reports_crop_and_jitter_proxies():
    delta = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    metrics = R.correction_metrics(delta)
    assert metrics["n_samples"] == 4
    assert metrics["rms_correction_px"] == pytest.approx(math.sqrt(25 / 8))
    assert metrics["max_abs_correction_px"] == pytest.approx(4.0)
    assert metrics["max_abs_correction_per_channel_px"] == pytest.approx([3.0, 4.0])
    assert metrics["rms_second_difference_correction_px"] is not None
    # Fewer than three samples cannot form a second difference.
    assert R.correction_metrics(delta[:2])["rms_second_difference_correction_px"] is None
    with pytest.raises(ValueError):
        R.correction_metrics(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        R.correction_metrics(np.array([[np.nan, 0.0]]))


def test_support_summary_reports_mean_min_max_and_min_frame():
    summary = R.support_summary([1.0, 0.5, 0.75])
    assert summary == {"frames": 3, "mean": pytest.approx(0.75), "min": pytest.approx(0.5),
                       "max": pytest.approx(1.0), "min_frame": 1}
    with pytest.raises(ValueError):
        R.support_summary([])
    with pytest.raises(ValueError):
        R.support_summary([0.0, np.inf])


# --------------------------------------------------------------------------------------
# resize, warp and A/B labels
# --------------------------------------------------------------------------------------

def test_resize_bilinear_identity_and_constant():
    constant = np.full((10, 20, 3), 77, np.uint8)
    same = R.resize_bilinear(constant, 20, 10)
    assert same.shape == (10, 20, 3) and np.all(same == 77)
    down = R.resize_bilinear(constant, 10, 5)
    assert down.shape == (5, 10, 3) and np.all(down == 77)
    with pytest.raises(ValueError):
        R.resize_bilinear(np.zeros((4, 4)), 4, 4)


def test_warp_panel_zero_delta_is_identity_and_shift_moves_content():
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    panel, valid = R.warp_panel(frame, np.array([0.0, 0.0]))
    assert valid.all()
    assert np.array_equal(panel, frame)
    shifted, valid = R.warp_panel(frame, np.array([1.0, 0.0]))
    assert np.array_equal(shifted[:, :7], frame[:, 1:])
    assert not valid[:, 7].any()


def test_assemble_ab_panel_draws_distinct_a_b_labels():
    blank = np.zeros((40, 60, 3), np.uint8)
    panel = R.assemble_ab_panel(blank, blank)
    assert panel.shape == (40, 120, 3)
    left_label = panel[:24, :24, 0]
    right_label = panel[:24, 60:84, 0]
    assert left_label.max() == 255 and right_label.max() == 255
    assert not np.array_equal(left_label, right_label)
    with pytest.raises(ValueError):
        R.draw_bitmap_label(np.zeros((10, 10, 3), np.uint8), "Z")


# --------------------------------------------------------------------------------------
# blinding
# --------------------------------------------------------------------------------------

def test_blind_assignment_is_deterministic_and_seed_bound():
    first = R.blind_assignment(R.BLINDING_SEED, 4)
    assert first == R.blind_assignment(R.BLINDING_SEED, 4)
    assert all(isinstance(v, bool) for v in first)
    assert len(first) == 4
    assert R.blind_assignment(R.BLINDING_SEED + 1, 8) != R.blind_assignment(R.BLINDING_SEED, 8)
    with pytest.raises(ValueError):
        R.blind_assignment(R.BLINDING_SEED, -1)


def test_derive_blinding_seed_is_stable_and_provenance_bound():
    base = R.BLINDING_SEED
    derived = R.derive_blinding_seed(base, "abc", 0.0, 2.0, "run-1")
    assert isinstance(derived, int) and derived >= 0
    # Same provenance replays bit-for-bit.
    assert derived == R.derive_blinding_seed(base, "abc", 0.0, 2.0, "run-1")
    # Every stable provenance field feeds the seed, so a different window/run/source/base generally differs.
    assert derived != R.derive_blinding_seed(base, "abc", 0.0, 2.0, "run-2")
    assert derived != R.derive_blinding_seed(base, "abc", 30.0, 2.0, "run-1")
    assert derived != R.derive_blinding_seed(base, "abc", 0.0, 4.0, "run-1")
    assert derived != R.derive_blinding_seed(base, "def", 0.0, 2.0, "run-1")
    assert derived != R.derive_blinding_seed(base + 1, "abc", 0.0, 2.0, "run-1")
    # The derived seed is a valid blind_assignment seed.
    assert R.blind_assignment(derived, 4) == R.blind_assignment(derived, 4)


# --------------------------------------------------------------------------------------
# provenance: missing and mismatched
# --------------------------------------------------------------------------------------

def test_manifest_provenance_refuses_missing_manifest(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(R.ReviewError) as exc:
        R.manifest_provenance(empty, empty / R.DEFAULT_MOTION_NAME)
    assert exc.value.code == "manifest_missing"


def test_load_review_inputs_refuses_missing_run_and_motion(tmp_path):
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(tmp_path / "absent")
    assert exc.value.code == "run_dir_missing"
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(run)
    assert exc.value.code == "motion_missing"


def test_manifest_provenance_refusals_use_stable_codes(tmp_path):
    run, motion_path, _, _ = _build_run(
        tmp_path / "status", manifest_mutate=lambda m: m.update(status="failed"))
    with pytest.raises(R.ReviewError) as exc:
        R.manifest_provenance(run, motion_path)
    assert exc.value.code == "run_not_successful"

    run, motion_path, _, _ = _build_run(
        tmp_path / "mismatch", manifest_mutate=lambda m: m["outputs"].update(motion="other_r0.npz"))
    with pytest.raises(R.ReviewError) as exc:
        R.manifest_provenance(run, motion_path)
    assert exc.value.code == "mismatched_motion_output"

    run, motion_path, _, _ = _build_run(
        tmp_path / "capture",
        manifest_mutate=lambda m: m["experiment_identity"]["capture"].update(dump_motion=False))
    with pytest.raises(R.ReviewError) as exc:
        R.manifest_provenance(run, motion_path)
    assert exc.value.code == "manifest_dump_motion_not_captured"

    run, motion_path, _, _ = _build_run(
        tmp_path / "deint",
        manifest_mutate=lambda m: m["source"]["preprocessing"].update(deinterlace="yadif"))
    with pytest.raises(R.ReviewError) as exc:
        R.manifest_provenance(run, motion_path)
    assert exc.value.code == "unsupported_deinterlace"


def test_load_review_inputs_refuses_hash_and_motion_mismatches(tmp_path):
    run, _, _, _ = _build_run(tmp_path / "hash", manifest_mutate=lambda m: m["source"].update(sha256="0" * 64))
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(run)
    assert exc.value.code == "source_hash_mismatch"

    run, _, _, _ = _build_run(
        tmp_path / "width", manifest_mutate=lambda m: m["source"]["decoded"].update(width=32))
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(run)
    assert exc.value.code == "dimension_mismatch"

    run, _, _, _ = _build_run(
        tmp_path / "fps", manifest_mutate=lambda m: m["source"]["decoded"].update(fps=29.0))
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(run)
    assert exc.value.code == "fps_mismatch"

    run, _, _, _ = _build_run(
        tmp_path / "frames", manifest_mutate=lambda m: m["source"]["decoded"].update(frames=7))
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(run)
    assert exc.value.code == "frame_count_mismatch"


def test_load_review_inputs_accepts_valid_run_and_verifies_hash(tmp_path, monkeypatch):
    run, _, _, source = _build_run(tmp_path)
    # Exercise the hash-verification branch explicitly: a wrong hash must be refused even for a valid payload.
    monkeypatch.setattr(R, "sha256_file", lambda path: "deadbeef")
    with pytest.raises(R.ReviewError) as exc:
        R.load_review_inputs(run)
    assert exc.value.code == "source_hash_mismatch"
    monkeypatch.undo()
    mot, meta, prov = R.load_review_inputs(run)
    assert (meta["width"], meta["height"], meta["n_frames"]) == (64, 48, 8)
    assert prov["source_sha256_verified"] is True
    assert prov["source_sha256"] == sha256_file(source)


# --------------------------------------------------------------------------------------
# fully mocked end-to-end run
# --------------------------------------------------------------------------------------

def _mock_media(monkeypatch, W, H, T, fps):
    def fake_probe(path):
        return {"width": W, "height": H, "fps": fps / 2.0, "field_order": "tt", "interlaced": True,
                "duration": T / fps}

    def fake_decode(path, info=None, start=None, duration=None, deinterlace=False, vf=None):
        return iter(_fake_frames(T, H, W)), fps, (W, H)

    _FakeEncoder.instances = []
    monkeypatch.setattr(media, "probe", fake_probe)
    monkeypatch.setattr(media, "decode", fake_decode)
    monkeypatch.setattr(media, "Encoder", _FakeEncoder)


def test_run_review_mocked_writes_four_outputs_with_fixed_mapping_and_no_score(tmp_path, monkeypatch):
    W, H, T, fps = 64, 48, 8, 30.0
    run, _, _, _ = _build_run(tmp_path, W=W, H=H, T=T, fps=fps)
    _mock_media(monkeypatch, W, H, T, fps)

    out_dir = tmp_path / "review"
    report = R.run_review(run, out_dir=out_dir, out_width=48, out_height=36, write=True)

    # Four comparisons, four clips, four encoders, every frame written and closed.
    assert report["output"]["comparisons"] == 4
    assert len(report["pairs"]) == 4
    assert len(report["output"]["clips"]) == 4
    assert len(_FakeEncoder.instances) == 4
    for encoder in _FakeEncoder.instances:
        assert encoder.closed
        assert (encoder.width, encoder.height) == (2 * 48, 36)
        assert len(encoder.frames) == T
        assert all(frame.shape == (36, 96, 3) for frame in encoder.frames)

    # The blind mapping is the deterministic per-run draw, hidden per pair and labelled A/B only.
    _, _, prov = R.load_review_inputs(run)
    derived = R.derive_blinding_seed(
        R.BLINDING_SEED, prov["source_sha256"], prov["source_window"]["start_s"],
        prov["source_window"]["duration_s"], prov["run_id"])
    assignment = R.blind_assignment(derived, 4)
    assert report["blinding"]["seed"] == derived
    assert report["blinding"]["seed"] != R.BLINDING_SEED
    assert report["blinding"]["base_seed"] == R.BLINDING_SEED
    assert report["blinding"]["assignment"] == ["multiscale" if v else "l1" for v in assignment]
    for record, left_is_multiscale in zip(report["pairs"], assignment):
        assert record["blind_mapping"]["left_panel"] == ("multiscale" if left_is_multiscale else "l1")
        assert record["blind_mapping"]["right_panel"] == ("l1" if left_is_multiscale else "multiscale")
        assert record["blind_mapping"]["video_labels"] == {"left": "A", "right": "B"}
        for side in ("multiscale", "l1"):
            assert "crop_proxy_px" in record[side]
            assert 0.0 <= record[side]["geometric_support"]["min"] <= 1.0

    # No image-quality score, ranking or winner anywhere; report is JSON-safe and path-free.
    keys = set(_all_keys(report))
    assert not (keys & {"mae", "rms", "psnr", "ssim", "score", "winner", "rank", "quality", "aggregate"})
    _assert_json_native(report)
    payload = (out_dir / "review.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in payload
    assert json.loads(payload)["schema"] == R.SCHEMA


def test_run_review_write_false_skips_clips(tmp_path, monkeypatch):
    W, H, T, fps = 64, 48, 8, 30.0
    run, _, _, _ = _build_run(tmp_path, W=W, H=H, T=T, fps=fps)
    _mock_media(monkeypatch, W, H, T, fps)

    report = R.run_review(run, out_dir=tmp_path / "review", out_width=48, out_height=36, write=False)
    assert _FakeEncoder.instances == []
    assert not (tmp_path / "review" / "review.json").exists()
    assert report["output"]["comparisons"] == 4


def test_run_review_derives_independent_seed_per_window(tmp_path, monkeypatch):
    W, H, T, fps = 64, 48, 8, 30.0
    run_a, _, _, _ = _build_run(tmp_path / "a", W=W, H=H, T=T, fps=fps)
    run_b, _, _, _ = _build_run(
        tmp_path / "b", W=W, H=H, T=T, fps=fps,
        manifest_mutate=lambda m: m["source"]["window"].update(start_s=30.0))
    _mock_media(monkeypatch, W, H, T, fps)

    first = R.run_review(run_a, out_dir=tmp_path / "ra", out_width=48, out_height=36, write=False)
    second = R.run_review(run_b, out_dir=tmp_path / "rb", out_width=48, out_height=36, write=False)
    # Different source windows generally get different derived seeds and assignment patterns.
    assert first["blinding"]["seed"] != second["blinding"]["seed"]
    assert first["blinding"]["base_seed"] == R.BLINDING_SEED
    # The same run replays to the same derived seed and mapping.
    replay = R.run_review(run_a, out_dir=tmp_path / "ra2", out_width=48, out_height=36, write=False)
    assert replay["blinding"]["seed"] == first["blinding"]["seed"]
    assert replay["blinding"]["assignment"] == first["blinding"]["assignment"]


def test_run_review_closes_every_clip_and_propagates_first_close_failure(tmp_path, monkeypatch):
    """A failing clip close must still close the other three, and the first failure is the one raised."""
    W, H, T, fps = 64, 48, 8, 30.0
    run, _, _, _ = _build_run(tmp_path, W=W, H=H, T=T, fps=fps)
    _mock_media(monkeypatch, W, H, T, fps)

    class CloseBoomEncoder(_FakeEncoder):
        def close(self):
            self.closed = True
            raise RuntimeError(f"close {self.path}")

    monkeypatch.setattr(media, "Encoder", CloseBoomEncoder)
    with pytest.raises(RuntimeError) as ei:
        R.run_review(run, out_dir=tmp_path / "review", out_width=48, out_height=36, write=True)
    assert len(_FakeEncoder.instances) == 4
    assert all(encoder.closed for encoder in _FakeEncoder.instances)   # every sibling was closed
    assert str(ei.value) == f"close {_FakeEncoder.instances[0].path}"  # the first failure, not a later one


def test_run_review_preserves_build_failure_while_closing_every_clip(tmp_path, monkeypatch):
    """A frame-write failure must not be masked by a later close failure; all clips still close."""
    W, H, T, fps = 64, 48, 8, 30.0
    run, _, _, _ = _build_run(tmp_path, W=W, H=H, T=T, fps=fps)
    _mock_media(monkeypatch, W, H, T, fps)

    class BuildBoom(Exception):
        pass

    class WriteAndCloseBoomEncoder(_FakeEncoder):
        def write(self, frame):
            raise BuildBoom("write failed")
        def close(self):
            self.closed = True
            raise RuntimeError(f"close {self.path}")

    monkeypatch.setattr(media, "Encoder", WriteAndCloseBoomEncoder)
    with pytest.raises(BuildBoom, match="write failed"):
        R.run_review(run, out_dir=tmp_path / "review", out_width=48, out_height=36, write=True)
    assert len(_FakeEncoder.instances) == 4
    assert all(encoder.closed for encoder in _FakeEncoder.instances)   # closes attempted despite the failure
