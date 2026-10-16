"""Focused, media-free tests for the read-only ``bench/border_backoff_screen.py`` audit.

The screen reconstructs a baseline Plan, verifies the NumPy invalid mask against the decoded
``mask_r0.mp4`` and screens the opt-in sampled border backoff. These tests drive the real
:func:`bench.border_backoff_screen.run_screen` with synthetic plans and monkeypatched ``media`` /
``load_motion_npz``, so no footage, FFmpeg encoding, MLX/Metal graph or render lock is touched; only the
NumPy production helpers in ``mlxstab.stabilize`` run. Coverage: successful exact verification and
candidate selection, the pure counting/verification helpers, the strict gray threshold, report
invariants, and the refusal paths (bad cadence, non-finite/zero cadence, mask frame/pixel/shape
mismatch, bad and empty limits).

The reported geometry is deliberately not treated as a quality result: the tests assert the screen's
coverage/coverage-delta accounting and its refusals, not that the selected scale is perceptually good.
"""
import copy
import json

import numpy as np
import pytest

from bench import border_backoff_screen as P

W, H, T = 8, 6, 4
ANALYSIS_FPS = 60.0
OUTPUT_FPS = 30.0
PLAN_INDICES = [0, 2]


# --------------------------------------------------------------------------------------
# synthetic fixtures / fakes
# --------------------------------------------------------------------------------------

class _FakeMotion:
    def __init__(self, w=W, h=H, n_frames=T):
        self.W, self.H, self.n_frames = w, h, n_frames


def _make_plan(tx=0.0, t=T):
    """A plan whose global part is a constant ``tx``-pixel horizontal shift on every frame."""
    G = []
    for _ in range(t):
        g = np.eye(3)
        g[0, 2] = tx
        G.append(g)
    return P.S.Plan(G=G, mesh_corr=np.zeros((t, 1, 1, 2)), row_corr=np.zeros((t, 1, 2)),
                    raw_quads=np.zeros((t, 4, 2)), path_quads=np.zeros((t, 4, 2)))


def _grid(w=W, h=H):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def _gray(mask):
    """Encode a boolean invalid mask (True = out of source) as the gray ``mask_r0.mp4`` convention."""
    g = np.where(np.asarray(mask, bool)[..., None], 255, 0).astype(np.uint8)
    return np.repeat(g, 3, axis=2)


def _encoded(plan, indices=PLAN_INDICES):
    return [_gray(m) for m in P._reconstructed_masks(plan, indices, W, H, _grid())]


def _meta(fps=ANALYSIS_FPS):
    return ({"schema": "mlxstab.motion/4", "width": W, "height": H, "n_frames": T, "fps": fps,
             "stride": 1, "n_fits": T - 1, "mesh_cells": [1, 1], "row_bands": 1, "fallback_fits": 0})


def _install(monkeypatch, *, plan, encoded, width=W, height=H, motion_fps=ANALYSIS_FPS, decode_fps=OUTPUT_FPS):
    """Patch the module's only I/O seams: the NPZ loader, the Plan and ``mlxstab.media``."""
    monkeypatch.setattr(P, "load_motion_npz", lambda path: (_FakeMotion(), _meta(motion_fps)))
    monkeypatch.setattr(P.S, "plan", lambda mot, cfg, n_frames=None: plan)
    monkeypatch.setattr(P.media, "probe", lambda path: {"width": width, "height": height, "fps": decode_fps})
    monkeypatch.setattr(P.media, "decode",
                        lambda path, info: (iter([np.array(f, copy=True) for f in encoded]),
                                            decode_fps, (width, height)))


def _write_manifest(tmp_path, *, analysis_fps=ANALYSIS_FPS, output_fps=OUTPUT_FPS, status="success",
                    source_sha="a" * 64, run_id="synthetic_run"):
    man = {
        "schema": "mlxstab.bench.manifest/1",
        "status": status,
        "run_id": run_id,
        "source": {"sha256": source_sha, "window": {"start_s": 0.0, "duration_s": 1.0},
                   "decoded": {"fps": analysis_fps, "frames": 60}},
        "weights": {"sha256": "b" * 64},
        "config": {"mesh": [1, 1], "rs_rows": 1},
        "outputs": {"plan": None, "decoded": {"fps": output_fps, "frames": len(PLAN_INDICES)}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    return path


def _success_report(monkeypatch, tmp_path, *, plan=None, limits=(0.2,), bottom_rows=2):
    plan = _make_plan(tx=4.0) if plan is None else plan
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    return P.run_screen(_write_manifest(tmp_path), limits=limits, bottom_rows=bottom_rows,
                        log=lambda *a: None)


# --------------------------------------------------------------------------------------
# successful validation + math
# --------------------------------------------------------------------------------------

def test_run_screen_success_verifies_exactly_and_reports_scaled_candidates(monkeypatch, tmp_path):
    """End-to-end success: exact mask agreement, identity baseline, and a feasible scaled candidate."""
    plan = _make_plan(tx=4.0)
    masks = P._reconstructed_masks(plan, PLAN_INDICES, W, H, _grid())
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    report = P.run_screen(_write_manifest(tmp_path), limits=(0.2, 0.05), bottom_rows=2, log=lambda *a: None)

    assert report["schema"] == P.SCHEMA and report["status"] == "success"
    ver = report["verification"]
    assert ver["exact"] is True and ver["differing_pixels_total"] == 0
    assert ver["frames_checked"] == len(PLAN_INDICES)
    assert [d["plan_frame"] for d in ver["per_frame"]] == PLAN_INDICES
    # Cadence mapping: 60p analysis -> 30p output decimates by 2 over the whole plan.
    assert report["input"]["output_frame_step"] == 2
    assert report["input"]["plan_frames"] == T

    base = report["baseline"]
    assert base["selected_scale"] == 1.0 and base["sampled_border_measured"] is False
    expected_pooled = sum(int(m.sum()) for m in masks) / (len(masks) * W * H)
    assert base["full_resolution"]["pooled_fraction"] == pytest.approx(expected_pooled)

    assert len(report["candidates"]) == 2
    for limit, cand in zip((0.2, 0.05), report["candidates"]):
        assert cand["limit"] == limit
        assert cand["sampled_border_measured"] is True
        assert 0.0 <= cand["selected_scale"] <= 1.0
        assert cand["sampled_worst_frac"] <= limit + 1e-12
        assert cand["full_resolution"]["pooled_fraction"] <= base["full_resolution"]["pooled_fraction"]
    # A half-frame shift at scale 1 must be pulled back below identity for a tight cap.
    tight = report["candidates"][-1]
    assert tight["selected_scale"] < 1.0
    assert tight["full_resolution"]["pooled_fraction"] < base["full_resolution"]["pooled_fraction"]
    # Report is JSON-safe with no NaN even on the success branch.
    json.dumps(report, allow_nan=False)


def test_candidate_measurement_matches_the_scaled_plan_reconstruction(monkeypatch, tmp_path):
    """The candidate's full-resolution fraction is exactly the scaled-plan NumPy reconstruction."""
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    report = P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    cand = report["candidates"][0]

    ref = P.S.sampled_border_backoff(plan, W, H, 0.2)
    assert cand["selected_scale"] == ref.sampled_border_scale
    assert cand["sampled_worst_frac"] == pytest.approx(ref.sampled_border_worst_frac)

    scaled = P.S._scaled_plan(plan, cand["selected_scale"])
    masks = P._reconstructed_masks(scaled, PLAN_INDICES, W, H, _grid())
    expected = sum(int(m.sum()) for m in masks) / (len(masks) * W * H)
    assert cand["full_resolution"]["pooled_fraction"] == pytest.approx(expected)


def test_run_screen_default_limits_produce_one_candidate_each(monkeypatch, tmp_path):
    """With no explicit limits the report screens exactly the four module defaults, in order."""
    report = _success_report(monkeypatch, tmp_path, limits=P.DEFAULT_LIMITS)
    assert [c["limit"] for c in report["candidates"]] == list(P.DEFAULT_LIMITS)


# --------------------------------------------------------------------------------------
# pure helpers: counting, verification, decode threshold
# --------------------------------------------------------------------------------------

def test_measure_masks_pooled_mean_max_and_bottom_rows():
    masks = [np.array([[False, True], [True, True]]), np.zeros((2, 2), bool)]
    measured = P._measure_masks(masks, [0, 2], bottom_rows=1)
    assert measured["full_resolution"] == {
        "frames": 2, "frame_pixels": 4, "total_pixels": 8, "total_invalid_pixels": 3,
        "pooled_fraction": 3 / 8, "mean_fraction": (0.75 + 0.0) / 2, "max_fraction": 0.75,
    }
    bottom = measured["bottom_rows"]
    assert bottom["rows"] == 1 and bottom["per_frame_pixels"] == 2
    assert bottom["total_invalid_pixels"] == 2 and bottom["pooled_fraction"] == 0.5
    assert [r["output_frame"] for r in measured["per_frame"]] == [0, 1]
    assert [r["plan_frame"] for r in measured["per_frame"]] == [0, 2]
    assert measured["per_frame"][0]["invalid_fraction"] == 0.75
    assert measured["per_frame"][0]["bottom_invalid_fraction"] == 1.0


def test_measure_masks_clamps_bottom_rows_to_frame_height():
    masks = [np.ones((2, 2), bool)]
    measured = P._measure_masks(masks, [0], bottom_rows=99)
    assert measured["bottom_rows"]["rows"] == 2
    assert measured["bottom_rows"]["total_invalid_pixels"] == 4


def test_verify_masks_counts_only_differing_pixels():
    reconstructed = np.zeros((2, 2), bool); reconstructed[0, 0] = True
    encoded = np.zeros((2, 2), bool); encoded[0, 1] = True
    ver = P._verify_masks([reconstructed], [encoded], [0])
    assert ver["exact"] is False and ver["differing_pixels_total"] == 2
    assert ver["per_frame"][0]["reconstructed_invalid_pixels"] == 1
    assert ver["per_frame"][0]["encoded_invalid_pixels"] == 1


def test_decode_mask_uses_strict_greater_than_127(monkeypatch, tmp_path):
    """Gray 127 is valid and gray 128 is invalid, matching the ``f[..., 0] > 127`` convention."""
    frame = np.array([[[127, 0, 0], [128, 0, 0]]], np.uint8)     # shape (1, 2, 3)
    monkeypatch.setattr(P.media, "probe", lambda path: {"width": 2, "height": 1, "fps": OUTPUT_FPS})
    monkeypatch.setattr(P.media, "decode", lambda path, info: (iter([frame]), OUTPUT_FPS, (2, 1)))
    decoded, info = P._decode_mask_bool(tmp_path / "mask_r0.mp4")
    assert decoded[0].dtype == bool and decoded[0].tolist() == [[False, True]]
    assert info == {"width": 2, "height": 1, "fps": OUTPUT_FPS, "frames": 1}


# --------------------------------------------------------------------------------------
# report invariants
# --------------------------------------------------------------------------------------

def test_validate_report_accepts_success_and_is_path_free(monkeypatch, tmp_path):
    report = _success_report(monkeypatch, tmp_path)
    assert P._validate_report(report) is report
    text = json.dumps(report, allow_nan=False)
    for fragment in ("/Users/", "/home/", "\\", "out/"):
        assert fragment not in text


def test_validate_report_rejects_inconsistent_pooled_fraction(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["baseline"]["full_resolution"]["pooled_fraction"] = 0.9
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def test_validate_report_rejects_out_of_range_per_frame_fraction(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["candidates"][0]["per_frame"][0]["invalid_fraction"] = 1.5
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def test_validate_report_rejects_private_path(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["input"]["run_id"] = "/Users/secret/run"
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "not_path_free"


def test_validate_report_rejects_success_without_baseline_or_candidates(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["baseline"] = None
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["candidates"] = []
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


# --------------------------------------------------------------------------------------
# refusal paths
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("analysis_fps,output_fps", [
    (45.0, 30.0),             # ratio 1.5, not an integer decimation
    (60.0, 25.0),             # ratio 2.4
    (0.0, 30.0),              # analysis fps must be > 0
    (60.0, 0.0),              # output fps must be > 0 (previously a bare ZeroDivisionError)
    (float("nan"), 30.0),
    (60.0, float("nan")),     # previously a bare ``int(round(nan))`` ValueError
    (float("inf"), 30.0),
    (60.0, float("inf")),
])
def test_run_screen_refuses_bad_cadence_with_a_stable_code(monkeypatch, tmp_path, analysis_fps, output_fps):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_fps=analysis_fps, output_fps=output_fps)
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(manifest, limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "bad_cadence"


def test_run_screen_refuses_non_numeric_cadence_as_malformed(monkeypatch, tmp_path):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_fps="fast")
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(manifest, limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "manifest_malformed"


def test_run_screen_refuses_motion_fps_mismatch(monkeypatch, tmp_path):
    """A loaded Motion whose fps disagrees with the manifest analysis fps is refused, not index-mapped."""
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), motion_fps=OUTPUT_FPS)
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "motion_cadence_mismatch"


@pytest.mark.parametrize("motion_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_run_screen_refuses_non_positive_or_non_finite_motion_fps(monkeypatch, tmp_path, motion_fps):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), motion_fps=motion_fps)
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "bad_cadence"


def test_run_screen_refuses_mask_fps_mismatch_despite_aligned_frames(monkeypatch, tmp_path):
    """Same frame count and pixel-exact masks still refuse when decoded mask fps != manifest output fps."""
    plan = _make_plan(tx=4.0)
    encoded = _encoded(plan)                          # 2 frames, pixel-exact: would verify exactly
    _install(monkeypatch, plan=plan, encoded=encoded, decode_fps=ANALYSIS_FPS)   # 60 vs manifest output 30
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "mask_cadence_mismatch"


@pytest.mark.parametrize("mask_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_run_screen_refuses_non_positive_or_non_finite_mask_fps(monkeypatch, tmp_path, mask_fps):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), decode_fps=mask_fps)
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "bad_cadence"


def test_run_screen_refuses_mask_frame_count_mismatch(monkeypatch, tmp_path):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=[_gray(np.zeros((H, W), bool))])   # 1 frame vs 2 plan indices
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "cadence_mismatch"


def test_run_screen_refuses_mask_shape_mismatch(monkeypatch, tmp_path):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), width=W, height=H + 1)
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "shape_mismatch"


def test_run_screen_stops_with_verification_failed_on_mask_pixel_mismatch(monkeypatch, tmp_path):
    """A pixel-level mask disagreement is reported and stops the candidate comparison (no exception)."""
    plan = _make_plan(tx=4.0)
    encoded = _encoded(plan)
    encoded[0] = np.zeros((H, W, 3), np.uint8)                   # frame 0 claims all-valid, reconstructed isn't
    _install(monkeypatch, plan=plan, encoded=encoded)
    report = P.run_screen(_write_manifest(tmp_path), limits=(0.2, 0.05), log=lambda *a: None)
    assert report["status"] == "verification_failed"
    assert report["verification"]["exact"] is False
    assert report["verification"]["differing_pixels_total"] > 0
    assert report["candidates"] == [] and report["baseline"] is None
    assert "differs" in report["stop_reason"]
    json.dumps(report, allow_nan=False)                          # refusal report is still JSON-safe


@pytest.mark.parametrize("limits", [(), (-0.1,), (1.0,), (float("nan"),), (float("inf"),)])
def test_run_screen_refuses_bad_or_empty_limits(monkeypatch, tmp_path, limits):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=limits, log=lambda *a: None)
    assert excinfo.value.code == "bad_limit"


def test_run_screen_refuses_bad_bottom_rows(monkeypatch, tmp_path):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), bottom_rows=0, log=lambda *a: None)
    assert excinfo.value.code == "bad_bottom_rows"


def test_run_screen_refuses_expected_source_mismatch(monkeypatch, tmp_path):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path), limits=(0.2,), expect_source_sha256="c" * 64,
                     log=lambda *a: None)
    assert excinfo.value.code == "source_mismatch"


def test_run_screen_refuses_non_success_manifest(monkeypatch, tmp_path):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderBackoffScreenError) as excinfo:
        P.run_screen(_write_manifest(tmp_path, status="failure"), limits=(0.2,), log=lambda *a: None)
    assert excinfo.value.code == "not_success"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_success_writes_report(monkeypatch, tmp_path, capsys):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "report.json"
    rc = P.main([str(manifest), "--out", str(out), "--limits", "0.2,0.05", "--bottom-rows", "2"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["status"] == "success" and summary["verification_exact"] is True
    assert summary["candidates"] == 2
    assert json.loads(out.read_text())["schema"] == P.SCHEMA


def test_cli_failure_prints_json_error_and_exits_one(monkeypatch, tmp_path, capsys):
    plan = _make_plan(tx=4.0)
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_fps=45.0)
    rc = P.main([str(manifest), "--out", str(tmp_path / "report.json")])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "failure"
    assert payload["error"]["type"] == "BorderBackoffScreenError"
