"""Focused, media-free tests for the read-only ``bench/real_border_component_audit.py`` audit.

The audit reconstructs a production Plan, verifies the NumPy invalid mask against the decoded ``mask_r0.mp4``
and then ablates the mesh/row residual components. These tests drive the real
:func:`bench.real_border_component_audit.run_audit` with synthetic plans and monkeypatched ``media`` /
``load_motion_npz``, so no footage, FFmpeg encoding, MLX/Metal graph or render lock is touched; only the NumPy
production helpers in ``mlxstab.stabilize`` run.

Coverage: successful exact verification and component/delta math, the pure plan/delta helpers, report
invariants and path privacy, and the refusal paths (bad source hash, bad cadence, motion/mask cadence, frame
count, shape, bottom rows, expected-source mismatch), plus the CLI.

The reported geometry is a coverage diagnostic, not a quality result: the tests assert the audit's coverage
accounting, its non-additivity identity and its refusals, not that any component is perceptually better.
"""
import copy
import json

import numpy as np
import pytest

from bench import real_border_component_audit as P

W, H, T = 8, 6, 4
ANALYSIS_FPS = 60.0
OUTPUT_FPS = 30.0
PLAN_INDICES = [0, 2]
# With an identity global map and constant residual shifts (mesh +1 px in x, rows +0.6 px in y) the synthetic
# plan's invalid counts are exactly: full 13, global_only 0, global_mesh 6 (right column), global_rows 8
# (bottom row); the mesh/row components overlap on one pixel, so the interaction is -1 per frame.
EXPECTED_FULL = 13
EXPECTED_GLOBAL_MESH = 6
EXPECTED_GLOBAL_ROWS = 8


# --------------------------------------------------------------------------------------
# synthetic fixtures / fakes
# --------------------------------------------------------------------------------------

class _FakeMotion:
    def __init__(self, w=W, h=H, n_frames=T):
        self.W, self.H, self.n_frames = w, h, n_frames


def _make_plan(t=T, *, tx=0.0, mesh_x=1.0, row_y=0.6):
    """A plan with an identity (or shifted) global map and constant mesh/row residual corrections."""
    G = []
    for _ in range(t):
        g = np.eye(3)
        g[0, 2] = tx
        G.append(g)
    mesh = np.zeros((t, 2, 2, 2)); mesh[..., 0] = mesh_x
    row = np.zeros((t, 2, 2)); row[..., 1] = row_y
    return P.S.Plan(G=[np.array(g, np.float64) for g in G], mesh_corr=mesh, row_corr=row,
                    raw_quads=np.zeros((t, 4, 2)), path_quads=np.zeros((t, 4, 2)))


def _grid(w=W, h=H):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def _gray(mask):
    """Encode a boolean invalid mask (True = out of source) as the gray ``mask_r0.mp4`` convention."""
    g = np.where(np.asarray(mask, bool)[..., None], 255, 0).astype(np.uint8)
    return np.repeat(g, 3, axis=2)


def _encoded(plan, indices=PLAN_INDICES):
    return [_gray(m) for m in P._reconstructed_masks(plan, indices, W, H, _grid())]


def _meta(fps=ANALYSIS_FPS, n_frames=T, width=W, height=H):
    return ({"schema": "mlxstab.motion/4", "width": width, "height": height, "n_frames": n_frames,
             "fps": fps, "stride": 1, "n_fits": n_frames - 1, "mesh_cells": [1, 1], "row_bands": 2,
             "fallback_fits": 0})


def _install(monkeypatch, *, plan, encoded, width=W, height=H, motion_fps=ANALYSIS_FPS, decode_fps=OUTPUT_FPS,
             n_frames=T, motion_width=W, motion_height=H):
    """Patch the module's only I/O seams: the NPZ loader, the Plan and ``mlxstab.media``."""
    monkeypatch.setattr(P, "load_motion_npz",
                        lambda path: (_FakeMotion(w=W, h=H, n_frames=n_frames),
                                      _meta(motion_fps, n_frames, motion_width, motion_height)))
    monkeypatch.setattr(P.S, "plan", lambda mot, cfg, n_frames=None: plan)
    monkeypatch.setattr(P.media, "probe", lambda path: {"width": width, "height": height, "fps": decode_fps})
    monkeypatch.setattr(P.media, "decode",
                        lambda path, info: (iter([np.array(f, copy=True) for f in encoded]),
                                            decode_fps, (width, height)))


def _write_manifest(tmp_path, *, analysis_fps=ANALYSIS_FPS, output_fps=OUTPUT_FPS, status="success",
                    source_sha="a" * 64, run_id="synthetic_run", analysis_frames=T,
                    output_frames=len(PLAN_INDICES), source_width=W, source_height=H):
    man = {
        "schema": "mlxstab.bench.manifest/1",
        "status": status,
        "run_id": run_id,
        "source": {"sha256": source_sha, "window": {"start_s": 0.0, "duration_s": 1.0},
                   "decoded": {"fps": analysis_fps, "frames": analysis_frames,
                               "width": source_width, "height": source_height}},
        "weights": {"sha256": "b" * 64},
        "config": {"mesh": [1, 1], "rs_rows": 1},
        "outputs": {"plan": None, "decoded": {"fps": output_fps, "frames": output_frames}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    return path


def _success_report(monkeypatch, tmp_path, *, plan=None, bottom_rows=2):
    plan = _make_plan() if plan is None else plan
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    return P.run_audit(_write_manifest(tmp_path), bottom_rows=bottom_rows, log=lambda *a: None)


# --------------------------------------------------------------------------------------
# successful validation + component/delta math
# --------------------------------------------------------------------------------------

def test_run_audit_success_verifies_exactly_and_reports_component_math(monkeypatch, tmp_path):
    """End-to-end success: exact mask agreement and the expected component/delta attribution."""
    report = _success_report(monkeypatch, tmp_path, bottom_rows=2)

    assert report["schema"] == P.SCHEMA and report["status"] == "success"
    ver = report["verification"]
    assert ver["exact"] is True and ver["differing_pixels_total"] == 0
    assert ver["frames_checked"] == len(PLAN_INDICES)
    assert [d["plan_frame"] for d in ver["per_frame"]] == PLAN_INDICES
    assert ver["mask_dimensions"] == [W, H] and ver["output_frames_expected"] == len(PLAN_INDICES)
    assert report["input"]["output_frame_step"] == 2 and report["input"]["plan_frames"] == T
    assert report["input"]["source_decoded_width"] == W
    assert report["input"]["source_decoded_height"] == H

    agg = report["components"]["aggregate"]
    assert agg["full"]["full_resolution"]["total_invalid_pixels"] == EXPECTED_FULL * len(PLAN_INDICES)
    assert agg["global_only"]["full_resolution"]["total_invalid_pixels"] == 0
    assert agg["global_mesh"]["full_resolution"]["total_invalid_pixels"] == EXPECTED_GLOBAL_MESH * len(PLAN_INDICES)
    assert agg["global_rows"]["full_resolution"]["total_invalid_pixels"] == EXPECTED_GLOBAL_ROWS * len(PLAN_INDICES)

    deltas = report["components"]["deltas"]["full_image"]
    assert deltas["mesh"]["total_pixels"] == (EXPECTED_GLOBAL_MESH - 0) * len(PLAN_INDICES)
    assert deltas["rows"]["total_pixels"] == (EXPECTED_GLOBAL_ROWS - 0) * len(PLAN_INDICES)
    assert deltas["combined"]["total_pixels"] == EXPECTED_FULL * len(PLAN_INDICES)
    # One overlapping pixel per frame makes the sequential composition sub-additive.
    assert deltas["interaction"]["total_pixels"] == (
        EXPECTED_FULL - EXPECTED_GLOBAL_MESH - EXPECTED_GLOBAL_ROWS) * len(PLAN_INDICES)
    assert set(report["components"]["deltas"]["definitions"]) == set(P.DELTA_NAMES)

    for name in ("mesh", "rows", "combined"):
        assert deltas[name]["sign_counts"]["positive"] == len(PLAN_INDICES)
        assert deltas[name]["sign_counts"]["negative"] == 0
    assert deltas["interaction"]["sign_counts"]["negative"] == len(PLAN_INDICES)
    json.dumps(report, allow_nan=False)


def test_run_audit_bottom_row_attribution_matches_counts(monkeypatch, tmp_path):
    """The bottom strip attribution is the same difference identity restricted to the bottom rows."""
    report = _success_report(monkeypatch, tmp_path, bottom_rows=2)
    agg = report["components"]["aggregate"]
    for name in P.COMPONENTS:
        assert agg[name]["bottom_rows"]["rows"] == 2
        assert agg[name]["bottom_rows"]["per_frame_pixels"] == W * 2
    deltas = report["components"]["deltas"]["bottom_rows"]
    # full bottom = right column (2 px) + bottom row (8 px) - overlap (1 px) = 9; mesh = 2, rows = 8.
    assert deltas["combined"]["total_pixels"] == 9 * len(PLAN_INDICES)
    assert deltas["mesh"]["total_pixels"] == 2 * len(PLAN_INDICES)
    assert deltas["rows"]["total_pixels"] == 8 * len(PLAN_INDICES)
    assert deltas["interaction"]["total_pixels"] == (9 - 2 - 8) * len(PLAN_INDICES)


def test_run_audit_reuses_full_verification_masks_for_the_full_component(monkeypatch, tmp_path):
    """The full component's counts equal the verified reconstruction, not a second sampling pass."""
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    report = P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    reference = P._measure_masks(P._reconstructed_masks(plan, PLAN_INDICES, W, H, _grid()), PLAN_INDICES, 2)
    assert report["components"]["aggregate"]["full"] == {
        "full_resolution": reference["full_resolution"], "bottom_rows": reference["bottom_rows"]}


# --------------------------------------------------------------------------------------
# pure helpers: component plans, delta stats
# --------------------------------------------------------------------------------------

def test_component_plans_share_the_global_map_and_zero_the_right_residuals():
    pl = _make_plan()
    plans = P._component_plans(pl)
    assert plans["full"] is pl
    for name in P.COMPONENTS:
        assert plans[name].G is pl.G
        assert plans[name].raw_quads is pl.raw_quads and plans[name].path_quads is pl.path_quads
        assert plans[name].mesh_corr.shape == pl.mesh_corr.shape
        assert plans[name].row_corr.shape == pl.row_corr.shape
    assert not np.any(plans["global_only"].mesh_corr) and not np.any(plans["global_only"].row_corr)
    assert np.array_equal(plans["global_mesh"].mesh_corr, pl.mesh_corr)
    assert not np.any(plans["global_mesh"].row_corr)
    assert not np.any(plans["global_rows"].mesh_corr)
    assert np.array_equal(plans["global_rows"].row_corr, pl.row_corr)


def test_delta_values_and_stats_identity_and_sign_counts():
    counts = {"full": [13], "global_only": [0], "global_mesh": [6], "global_rows": [8]}
    deltas = P._delta_values(counts)
    assert deltas == {"mesh": [6], "rows": [8], "interaction": [-1], "combined": [13]}
    stats = P._delta_stats([-1, 0, 3], region_pixels=10)
    assert stats["total_pixels"] == 2 and stats["mean_pixels"] == pytest.approx(2 / 3)
    assert stats["min_pixels"] == -1 and stats["max_pixels"] == 3
    assert stats["mean_fraction"] == pytest.approx(((-0.1) + 0.0 + 0.3) / 3)
    assert stats["sign_counts"] == {"positive": 1, "negative": 1, "zero": 1}


def test_measure_masks_pooled_mean_max_and_bottom_rows():
    masks = [np.array([[False, True], [True, True]]), np.zeros((2, 2), bool)]
    measured = P._measure_masks(masks, [0, 2], bottom_rows=1)
    assert measured["full_resolution"] == {
        "frames": 2, "frame_pixels": 4, "total_pixels": 8, "total_invalid_pixels": 3,
        "pooled_fraction": 3 / 8, "mean_fraction": (0.75 + 0.0) / 2, "max_fraction": 0.75,
    }
    assert measured["bottom_rows"]["total_invalid_pixels"] == 2
    assert [r["output_frame"] for r in measured["per_frame"]] == [0, 1]
    assert [r["plan_frame"] for r in measured["per_frame"]] == [0, 2]


def test_verify_masks_counts_only_differing_pixels():
    reconstructed = np.zeros((2, 2), bool); reconstructed[0, 0] = True
    encoded = np.zeros((2, 2), bool); encoded[0, 1] = True
    ver = P._verify_masks([reconstructed], [encoded], [0])
    assert ver["exact"] is False and ver["differing_pixels_total"] == 2
    assert ver["per_frame"][0]["reconstructed_invalid_pixels"] == 1


def test_decode_mask_uses_strict_greater_than_127(monkeypatch, tmp_path):
    """Gray 127 is valid and gray 128 is invalid, matching the ``f[..., 0] > 127`` convention."""
    frame = np.array([[[127, 0, 0], [128, 0, 0]]], np.uint8)     # shape (1, 2, 3)
    monkeypatch.setattr(P.media, "probe", lambda path: {"width": 2, "height": 1, "fps": OUTPUT_FPS})
    monkeypatch.setattr(P.media, "decode", lambda path, info: (iter([frame]), OUTPUT_FPS, (2, 1)))
    decoded, info = P._decode_mask_bool(tmp_path / "mask_r0.mp4")
    assert decoded[0].dtype == bool and decoded[0].tolist() == [[False, True]]
    assert info == {"width": 2, "height": 1, "fps": OUTPUT_FPS, "frames": 1}


# --------------------------------------------------------------------------------------
# bounded streaming: chunked reconstruction, close-on-error, EOF/extra refusal
# --------------------------------------------------------------------------------------

def test_iter_reconstructed_masks_matches_materialized_across_chunk_sizes():
    """The lazy per-frame iterator yields the same masks regardless of the bounded chunk size."""
    plan = _make_plan()
    reference = P._reconstructed_masks(plan, PLAN_INDICES, W, H, _grid())
    for chunk in (1, 2, 3, 100):
        got = list(P._iter_reconstructed_masks(plan, PLAN_INDICES, W, H, _grid(), chunk=chunk))
        assert len(got) == len(reference)
        assert all(np.array_equal(a, b) for a, b in zip(got, reference))


def test_iter_decode_mask_bool_closes_ffmpeg_generator_on_partial_close(monkeypatch, tmp_path):
    """Closing the streaming decoder early must close the underlying FFmpeg generator (no leak)."""
    closed = {"v": False}

    def source():
        try:
            yield np.array([[[0, 0, 0], [255, 0, 0]]], np.uint8)
            yield np.array([[[0, 0, 0], [0, 0, 0]]], np.uint8)
        finally:
            closed["v"] = True

    monkeypatch.setattr(P.media, "probe", lambda path: {"width": 2, "height": 1, "fps": OUTPUT_FPS})
    monkeypatch.setattr(P.media, "decode", lambda path, info: (source(), OUTPUT_FPS, (2, 1)))
    it, meta = P._iter_decode_mask_bool(tmp_path / "mask_r0.mp4")
    assert next(it).tolist() == [[False, True]]
    assert meta["frames"] == 1
    it.close()
    assert closed["v"] is True


def test_verify_and_measure_full_matches_materialized_path():
    """One streaming pass produces the same verification and full measurement as the materialized path."""
    plan = _make_plan()
    reference_masks = P._reconstructed_masks(plan, PLAN_INDICES, W, H, _grid())
    encoded = [f[..., 0] > 127 for f in _encoded(plan)]
    ver_reference = P._verify_masks(reference_masks, encoded, PLAN_INDICES)
    measure_reference = P._measure_masks(reference_masks, PLAN_INDICES, 2)
    verification, measured = P._verify_and_measure_full(
        plan, PLAN_INDICES, iter(encoded), W, H, _grid(), 2)
    assert verification == ver_reference
    assert measured == measure_reference


def test_run_audit_refuses_extra_mask_frames(monkeypatch, tmp_path):
    """A decoded mask with more frames than the plan maps is refused, not silently truncated."""
    plan = _make_plan()
    encoded = _encoded(plan) + [_gray(np.zeros((H, W), bool))]     # 3 frames vs 2 plan indices
    _install(monkeypatch, plan=plan, encoded=encoded)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "frame_count_mismatch"


def test_run_audit_refuses_early_eof_and_closes_decode_generator(monkeypatch, tmp_path):
    """A short decoded stream is refused and the FFmpeg generator is closed (no partial report/leak)."""
    plan = _make_plan()
    closed = {"v": False}

    def short_stream():
        try:
            yield _gray(P._reconstructed_masks(plan, PLAN_INDICES[:1], W, H, _grid())[0])
        finally:
            closed["v"] = True

    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    monkeypatch.setattr(P.media, "decode",
                        lambda path, info: (short_stream(), OUTPUT_FPS, (W, H)))
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "frame_count_mismatch"
    assert closed["v"] is True


def test_cli_propagates_decode_error_without_writing_report(monkeypatch, tmp_path, capsys):
    """A decode failure mid-stream aborts with no partial output report and closes the generator."""
    plan = _make_plan()
    closed = {"v": False}

    def failing_stream():
        try:
            yield _gray(P._reconstructed_masks(plan, PLAN_INDICES[:1], W, H, _grid())[0])
            raise RuntimeError("decoder failed")
        finally:
            closed["v"] = True

    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    monkeypatch.setattr(P.media, "decode",
                        lambda path, info: (failing_stream(), OUTPUT_FPS, (W, H)))
    out = tmp_path / "report.json"
    rc = P.main([str(_write_manifest(tmp_path)), "--out", str(out), "--bottom-rows", "2"])
    assert rc == 1
    assert not out.exists()
    assert closed["v"] is True
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "failure" and payload["error"]["type"] == "RuntimeError"


def test_measure_masks_streams_chunked_iterator_like_a_list():
    """``_measure_masks`` over a chunk=1 generator equals the materialized list result exactly."""
    plan = _make_plan()
    reference = P._measure_masks(P._reconstructed_masks(plan, PLAN_INDICES, W, H, _grid()), PLAN_INDICES, 2)
    streamed = P._measure_masks(
        P._iter_reconstructed_masks(plan, PLAN_INDICES, W, H, _grid(), chunk=1), PLAN_INDICES, 2)
    assert streamed == reference


def test_measure_masks_refuses_early_eof_and_closes_iterator():
    """A mask stream shorter than ``plan_indices`` is refused with ``frame_count_mismatch`` and closed."""
    closed = {"v": False}

    def short_stream():
        try:
            yield np.zeros((2, 2), bool)
        finally:
            closed["v"] = True

    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._measure_masks(short_stream(), [0, 2], bottom_rows=1)
    assert excinfo.value.code == "frame_count_mismatch"
    assert closed["v"] is True


def test_measure_masks_refuses_extra_frames_and_closes_iterator():
    """A mask stream longer than ``plan_indices`` is refused, not silently truncated, and closed."""
    closed = {"v": False}

    def long_stream():
        try:
            for _ in range(3):
                yield np.zeros((2, 2), bool)
        finally:
            closed["v"] = True

    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._measure_masks(long_stream(), [0, 2], bottom_rows=1)
    assert excinfo.value.code == "frame_count_mismatch"
    assert closed["v"] is True


def test_measure_masks_closes_iterator_on_success():
    """Consuming exactly ``plan_indices`` masks still closes the iterator (no leak on the success path)."""
    closed = {"v": False}

    def exact_stream():
        try:
            yield np.zeros((2, 2), bool)
            yield np.zeros((2, 2), bool)
        finally:
            closed["v"] = True

    measured = P._measure_masks(exact_stream(), [0, 2], bottom_rows=1)
    assert measured["full_resolution"]["frames"] == 2
    assert measured["full_resolution"]["total_invalid_pixels"] == 0
    assert closed["v"] is True


def test_measure_masks_closes_iterator_when_source_raises():
    """An exception raised by the mask source propagates but the iterator is still closed on the way out."""
    closed = {"v": False}

    def failing_stream():
        try:
            yield np.zeros((2, 2), bool)
            raise RuntimeError("mask source failed")
        finally:
            closed["v"] = True

    with pytest.raises(RuntimeError, match="mask source failed"):
        P._measure_masks(failing_stream(), [0, 2], bottom_rows=1)
    assert closed["v"] is True


# --------------------------------------------------------------------------------------
# verification failure and report invariants
# --------------------------------------------------------------------------------------

def test_run_audit_reports_verification_failed_without_ablations_on_mask_mismatch(monkeypatch, tmp_path):
    plan = _make_plan()
    encoded = _encoded(plan)
    encoded[0] = np.zeros((H, W, 3), np.uint8)                   # frame 0 claims all-valid, reconstructed isn't
    _install(monkeypatch, plan=plan, encoded=encoded)
    report = P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert report["status"] == "verification_failed"
    assert report["verification"]["exact"] is False
    assert report["verification"]["differing_pixels_total"] > 0
    assert report["components"] is None
    assert "differs" in report["stop_reason"]
    json.dumps(report, allow_nan=False)


def test_validate_report_accepts_success_and_is_path_free(monkeypatch, tmp_path):
    report = _success_report(monkeypatch, tmp_path)
    assert P._validate_report(report) is report
    text = json.dumps(report, allow_nan=False)
    for fragment in ("/Users/", "/home/", "\\", "out/"):
        assert fragment not in text


def test_validate_report_rejects_inconsistent_pooled_fraction(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["components"]["aggregate"]["full"]["full_resolution"]["pooled_fraction"] = 0.9
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def test_validate_report_rejects_inconsistent_interaction(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["components"]["per_frame"][0]["deltas"]["interaction_full_pixels"] = 12345
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def _corrupt_component_aggregate_total(report):
    blk = report["components"]["aggregate"]["global_mesh"]["bottom_rows"]
    blk["total_invalid_pixels"] += 1
    blk["pooled_fraction"] = blk["total_invalid_pixels"] / blk["total_pixels"]   # keep the fraction self-consistent


def _corrupt_component_aggregate_frames(report):
    report["components"]["aggregate"]["full"]["full_resolution"]["frames"] += 1


def _corrupt_full_image_delta_total(report):
    report["components"]["deltas"]["full_image"]["mesh"]["total_pixels"] += 1


def _corrupt_bottom_delta_total(report):
    report["components"]["deltas"]["bottom_rows"]["rows"]["total_pixels"] += 1


def _corrupt_delta_sign_counts(report):
    report["components"]["deltas"]["full_image"]["interaction"]["sign_counts"]["zero"] += 1


def _corrupt_per_frame_bottom_delta(report):
    report["components"]["per_frame"][0]["deltas"]["mesh_bottom_pixels"] += 1


def _corrupt_per_frame_bottom_identity(report):
    report["components"]["per_frame"][0]["deltas"]["combined_bottom_pixels"] += 1


def _corrupt_per_frame_count_type(report):
    report["components"]["per_frame"][0]["global_only"]["full_invalid_pixels"] = 1.0


def _corrupt_aggregate_interaction_identity(report):
    report["components"]["deltas"]["full_image"]["interaction"]["total_pixels"] += 1


def _corrupt_component_mean_fraction(report):
    report["components"]["aggregate"]["global_mesh"]["full_resolution"]["mean_fraction"] += 0.01


def _corrupt_component_bottom_mean_fraction(report):
    report["components"]["aggregate"]["full"]["bottom_rows"]["mean_fraction"] += 0.01


def _corrupt_component_max_fraction(report):
    report["components"]["aggregate"]["global_rows"]["bottom_rows"]["max_fraction"] += 0.01


def _corrupt_component_frame_pixels(report):
    report["components"]["aggregate"]["full"]["full_resolution"]["frame_pixels"] += 1


def _corrupt_component_bottom_per_frame_pixels(report):
    report["components"]["aggregate"]["global_mesh"]["bottom_rows"]["per_frame_pixels"] += 1


def _corrupt_denominator_cross_component(report):
    """A self-consistent single-component denominator change must still be refused: the four components
    measure the same output pixels, so their per-frame denominators must agree."""
    n = len(report["components"]["per_frame"])
    blk = report["components"]["aggregate"]["full"]["full_resolution"]
    blk["frame_pixels"] += 1
    blk["total_pixels"] = blk["frame_pixels"] * n
    blk["pooled_fraction"] = blk["total_invalid_pixels"] / blk["total_pixels"]


def _corrupt_per_frame_fraction(report):
    """A per-frame fraction that disagrees with its own count/denominator is refused (denominator hinge)."""
    report["components"]["per_frame"][0]["global_rows"]["bottom_invalid_fraction"] += 0.01


def _corrupt_delta_mean_pixels(report):
    report["components"]["deltas"]["full_image"]["mesh"]["mean_pixels"] += 0.5


def _corrupt_delta_mean_fraction(report):
    report["components"]["deltas"]["bottom_rows"]["rows"]["mean_fraction"] += 0.01


def _corrupt_delta_min_pixels(report):
    report["components"]["deltas"]["full_image"]["interaction"]["min_pixels"] -= 1


def _corrupt_delta_min_fraction(report):
    report["components"]["deltas"]["bottom_rows"]["combined"]["min_fraction"] -= 0.01


def _corrupt_delta_max_pixels(report):
    report["components"]["deltas"]["full_image"]["rows"]["max_pixels"] += 1


def _corrupt_delta_max_fraction(report):
    report["components"]["deltas"]["bottom_rows"]["interaction"]["max_fraction"] += 0.01


def _corrupt_delta_pooled_fraction(report):
    report["components"]["deltas"]["full_image"]["combined"]["pooled_fraction"] += 0.01


@pytest.mark.parametrize("mutate", [
    pytest.param(_corrupt_component_aggregate_total, id="component_aggregate_total"),
    pytest.param(_corrupt_component_aggregate_frames, id="component_aggregate_frames"),
    pytest.param(_corrupt_full_image_delta_total, id="full_image_delta_total"),
    pytest.param(_corrupt_bottom_delta_total, id="bottom_delta_total"),
    pytest.param(_corrupt_delta_sign_counts, id="delta_sign_counts"),
    pytest.param(_corrupt_per_frame_bottom_delta, id="per_frame_bottom_delta"),
    pytest.param(_corrupt_per_frame_bottom_identity, id="per_frame_bottom_identity"),
    pytest.param(_corrupt_per_frame_count_type, id="per_frame_count_type"),
    pytest.param(_corrupt_aggregate_interaction_identity, id="aggregate_interaction_identity"),
    pytest.param(_corrupt_component_mean_fraction, id="component_mean_fraction"),
    pytest.param(_corrupt_component_bottom_mean_fraction, id="component_bottom_mean_fraction"),
    pytest.param(_corrupt_component_max_fraction, id="component_max_fraction"),
    pytest.param(_corrupt_component_frame_pixels, id="component_frame_pixels"),
    pytest.param(_corrupt_component_bottom_per_frame_pixels, id="component_bottom_per_frame_pixels"),
    pytest.param(_corrupt_denominator_cross_component, id="denominator_cross_component"),
    pytest.param(_corrupt_per_frame_fraction, id="per_frame_fraction"),
    pytest.param(_corrupt_delta_mean_pixels, id="delta_mean_pixels"),
    pytest.param(_corrupt_delta_mean_fraction, id="delta_mean_fraction"),
    pytest.param(_corrupt_delta_min_pixels, id="delta_min_pixels"),
    pytest.param(_corrupt_delta_min_fraction, id="delta_min_fraction"),
    pytest.param(_corrupt_delta_max_pixels, id="delta_max_pixels"),
    pytest.param(_corrupt_delta_max_fraction, id="delta_max_fraction"),
    pytest.param(_corrupt_delta_pooled_fraction, id="delta_pooled_fraction"),
])
def test_validate_report_rejects_aggregate_corruption(monkeypatch, tmp_path, mutate):
    """The aggregate validator must refuse per-frame/aggregate count, denominator, delta and statistic
    corruption, including tampered pooled/mean/min/max fractions and sign counts."""
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    mutate(report)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def test_validate_report_rejects_self_consistent_component_denominator(monkeypatch, tmp_path):
    """A denominator corrupted together with total_pixels and pooled_fraction (internally consistent) is still
    refused, because the four components must share the same per-frame output pixel count."""
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    _corrupt_denominator_cross_component(report)
    blk = report["components"]["aggregate"]["full"]["full_resolution"]
    assert abs(blk["pooled_fraction"] - blk["total_invalid_pixels"] / blk["total_pixels"]) < 1e-12
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def test_validate_report_rejects_aggregate_per_frame_sum_mismatch(monkeypatch, tmp_path):
    """A component aggregate total that disagrees with the per-frame sum is refused even when its fraction
    is self-consistent, i.e. the reconciliation (not the pooled-fraction check) is what rejects it."""
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    blk = report["components"]["aggregate"]["full"]["full_resolution"]
    blk["total_invalid_pixels"] += 1
    blk["pooled_fraction"] = blk["total_invalid_pixels"] / blk["total_pixels"]
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "bad_report"


def test_validate_report_rejects_private_path(monkeypatch, tmp_path):
    report = copy.deepcopy(_success_report(monkeypatch, tmp_path))
    report["input"]["run_id"] = "/Users/secret/run"
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P._validate_report(report)
    assert excinfo.value.code == "not_path_free"


# --------------------------------------------------------------------------------------
# refusal paths
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("analysis_fps,output_fps", [
    (45.0, 30.0),             # ratio 1.5, not an integer decimation
    (60.0, 25.0),             # ratio 2.4
    (0.0, 30.0),              # analysis fps must be > 0
    (60.0, 0.0),              # output fps must be > 0
    (float("nan"), 30.0),
    (60.0, float("nan")),
    (float("inf"), 30.0),
    (60.0, float("inf")),
])
def test_run_audit_refuses_bad_cadence_with_a_stable_code(monkeypatch, tmp_path, analysis_fps, output_fps):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_fps=analysis_fps, output_fps=output_fps)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "bad_cadence"


def test_run_audit_refuses_non_numeric_cadence_as_malformed(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_fps="fast")
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "manifest_malformed"


def test_run_audit_refuses_motion_fps_mismatch(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), motion_fps=OUTPUT_FPS)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "motion_cadence_mismatch"


def test_run_audit_refuses_motion_frame_count_mismatch(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), n_frames=T)
    manifest = _write_manifest(tmp_path, analysis_frames=T + 1)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "motion_frame_mismatch"


@pytest.mark.parametrize("motion_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_run_audit_refuses_non_positive_or_non_finite_motion_fps(monkeypatch, tmp_path, motion_fps):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), motion_fps=motion_fps)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "bad_cadence"


def test_run_audit_refuses_mask_fps_mismatch_despite_aligned_frames(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), decode_fps=ANALYSIS_FPS)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "mask_cadence_mismatch"


def test_run_audit_refuses_mask_frame_count_mismatch(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=[_gray(np.zeros((H, W), bool))])   # 1 frame vs 2 plan indices
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "frame_count_mismatch"


def test_run_audit_refuses_manifest_output_frame_count_mismatch(monkeypatch, tmp_path):
    """``outputs.decoded.frames`` must match the plan's output mapping even when the mask agrees with the plan."""
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))                    # 2 encoded frames == 2 plan indices
    manifest = _write_manifest(tmp_path, output_frames=len(PLAN_INDICES) + 1)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "frame_count_mismatch"


@pytest.mark.parametrize("value", [True, 2.5, 0, -1, "4"])
def test_run_audit_refuses_non_integer_analysis_frame_count(monkeypatch, tmp_path, value):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_frames=value)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "bad_frame_count"


@pytest.mark.parametrize("value", [True, 2.5, 0, -1, "4"])
def test_run_audit_refuses_non_integer_output_frame_count(monkeypatch, tmp_path, value):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, output_frames=value)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "bad_frame_count"


@pytest.mark.parametrize("field", ["source_width", "source_height"])
@pytest.mark.parametrize("value", [True, 2.5, 0, -3, "1920"])
def test_run_audit_refuses_non_integer_source_dimensions(monkeypatch, tmp_path, field, value):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, **{field: value})
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "bad_dimensions"


@pytest.mark.parametrize("field,expected", [("source_width", W + 2), ("source_height", H + 2)])
def test_run_audit_refuses_source_dimension_mismatch(monkeypatch, tmp_path, field, expected):
    """source.decoded dimensions must equal the strict motion artifact's dimensions exactly."""
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, **{field: expected})
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "shape_mismatch"


def test_run_audit_refuses_motion_metadata_dimension_mismatch(monkeypatch, tmp_path):
    """Motion NPZ metadata that disagrees with the reconstructed Motion.W/H is refused."""
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), motion_width=W + 2)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "shape_mismatch"


def test_run_audit_refuses_mask_shape_mismatch(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan), width=W, height=H + 1)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "shape_mismatch"


@pytest.mark.parametrize("source_sha", ["abc", "z" * 64, "A" * 64])
def test_run_audit_refuses_bad_source_sha_shape(monkeypatch, tmp_path, source_sha):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, source_sha=source_sha)
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(manifest, bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "bad_source_sha"


def test_run_audit_refuses_empty_source_sha_as_malformed(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path, source_sha=""), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "manifest_malformed"


def test_run_audit_refuses_expected_source_mismatch(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=2, expect_source_sha256="c" * 64,
                    log=lambda *a: None)
    assert excinfo.value.code == "source_mismatch"


def test_run_audit_refuses_non_success_manifest(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path, status="failure"), bottom_rows=2, log=lambda *a: None)
    assert excinfo.value.code == "not_success"


def test_run_audit_refuses_bad_bottom_rows(monkeypatch, tmp_path):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    with pytest.raises(P.BorderComponentAuditError) as excinfo:
        P.run_audit(_write_manifest(tmp_path), bottom_rows=0, log=lambda *a: None)
    assert excinfo.value.code == "bad_bottom_rows"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_success_writes_report(monkeypatch, tmp_path, capsys):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "report.json"
    rc = P.main([str(manifest), "--out", str(out), "--bottom-rows", "2"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["status"] == "success" and summary["verification_exact"] is True
    assert summary["output_frames"] == len(PLAN_INDICES)
    assert summary["full_invalid_pixels"] == EXPECTED_FULL * len(PLAN_INDICES)
    assert json.loads(out.read_text())["schema"] == P.SCHEMA


def test_cli_failure_prints_json_error_and_exits_one(monkeypatch, tmp_path, capsys):
    plan = _make_plan()
    _install(monkeypatch, plan=plan, encoded=_encoded(plan))
    manifest = _write_manifest(tmp_path, analysis_fps=45.0)
    rc = P.main([str(manifest), "--out", str(tmp_path / "report.json")])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "failure"
    assert payload["error"]["type"] == "BorderComponentAuditError"
