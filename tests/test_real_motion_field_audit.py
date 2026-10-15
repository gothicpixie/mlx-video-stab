"""Focused tests for the captured real motion-field audit (``bench/real_motion_field_audit.py``).

Covers: the image-y-down similarity rotation/scale extraction from analytic ``H`` fixtures, the non-rotating
identity case, residual magnitude statistics (RMS/median/p95/max), the x-independent row-band line profile
(exact linear shear, nonlinear, constant, one-band/empty and legacy-motion JSON behavior), the per-cell spatial
grid and its cautious dispersion/coherence proxy, per-pair frame indexing under analysis stride, fallback
metadata, strict manifest
validation (success only, ``outputs.motion`` present and resolvable), manifest/motion identity cross-checks,
matching source identity across several manifests (including that nested windows are allowed and recorded),
malformed/missing manifest cases, the no-pickle guarantee of the reused loader, JSON-safe finite output, and
the CLI. No media, FFmpeg, RAFT or changed production defaults.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mlxstab import stabilize as S
from mlxstab.motion import Fit

from bench import real_composition_audit as RCA
from bench import real_motion_field_audit as R

W, H = 64, 48


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def similarity_h(W, H, deg_cw, scale):
    """Similarity about the image centre: scale then visual-clockwise rotation (image y down)."""
    r = math.radians(deg_cw)
    A = scale * np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]], np.float64)
    c = np.array([W / 2 - .5, H / 2 - .5])
    t = c - A @ c
    Hm = np.eye(3)
    Hm[:2, :2] = A
    Hm[:2, 2] = t
    return Hm


def make_fit(Hm=None, mesh=None, rows=None, *, gc=4, gr=3, nr=2, inlier=0.95, mover=0.01, fallback=False,
             n_valid_samples=None, n_ransac_inliers=None, fallback_reason=None,
             mesh_support_mass=None, row_support_mass=None, cycle_err_rms=None,
             cycle_inconsistency_frac=None, cycle_sample_mass=None):
    if Hm is None:
        Hm = np.eye(3)
    if mesh is None:
        mesh = np.zeros((gr + 1, gc + 1, 2))
    if rows is None:
        rows = np.zeros((nr, 2))
    return Fit(np.asarray(Hm, np.float64), np.asarray(mesh, np.float64), np.asarray(rows, np.float64),
               inlier, mover, fallback, n_valid_samples=n_valid_samples,
               n_ransac_inliers=n_ransac_inliers, fallback_reason=fallback_reason,
               mesh_support_mass=mesh_support_mass, row_support_mass=row_support_mass,
               cycle_err_rms=cycle_err_rms, cycle_inconsistency_frac=cycle_inconsistency_frac,
               cycle_sample_mass=cycle_sample_mass)


def write_manifest(run, motion_path, *, case="baseline", status="success", dump_motion=True,
                   outputs_motion=None, source_sha="s" * 64, weights_sha="w" * 64, decoded=None,
                   window=None, preprocessing=None):
    man = {
        "schema": "mlxstab.bench.manifest/1",
        "run_id": run.name,
        "case": {"name": case},
        "status": status,
        "error": None,
        "source": {"sha256": source_sha,
                   "window": window if window is not None else {"start_s": 0.0, "duration_s": 1.0},
                   "preprocessing": preprocessing if preprocessing is not None else {"interlaced": False},
                   "decoded": decoded if decoded is not None else {"width": W, "height": H, "fps": 30.0}},
        "weights": {"sha256": weights_sha},
        "outputs": {"motion": outputs_motion if outputs_motion is not None else str(motion_path)},
        "experiment_identity": {"capture": {"dump_motion": dump_motion}},
    }
    (run / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return man


def save_case(root, *, name="run", case="baseline", fits=None, stride=1, motion_name="motion_r0.npz",
              gc=4, gr=3, nr=2, status="success", dump_motion=True, outputs_motion=None, source_sha="s" * 64,
              weights_sha="w" * 64, decoded=None, window=None, preprocessing=None, write=True, W=W, H=H,
              fps=30.0):
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    if fits is None:
        fits = [make_fit(similarity_h(W, H, 2.0, 1.05), gc=gc, gr=gr, nr=nr)]
    mot = S.Motion(W, H, fps, fits=fits, stride=stride, n_frames=len(fits) * stride + 1)
    motion_path = run / motion_name
    S.save_motion_npz(mot, str(motion_path), S.Config(mesh=(gc, gr), rs_rows=nr))
    manifest_path = run / "manifest.json"
    if write:
        write_manifest(run, motion_path, case=case, status=status, dump_motion=dump_motion,
                       outputs_motion=outputs_motion, source_sha=source_sha, weights_sha=weights_sha,
                       decoded=decoded if decoded is not None else {"width": W, "height": H, "fps": fps},
                       window=window, preprocessing=preprocessing)
    return manifest_path, motion_path, run, mot


# --------------------------------------------------------------------------------------
# similarity extraction
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("deg,scale", [(3.0, 1.25), (-2.5, 0.9), (0.0, 1.0), (10.0, 2.0)])
def test_similarity_from_h_roundtrip(deg, scale):
    angle, got = R.similarity_from_h(similarity_h(W, H, deg, scale), W, H)
    assert got == pytest.approx(scale, abs=1e-9)
    assert angle == pytest.approx(deg, abs=1e-9)


def test_similarity_from_h_identity_is_zero_rotation_unit_scale():
    angle, scale = R.similarity_from_h(np.eye(3), W, H)
    assert angle == pytest.approx(0.0, abs=1e-12)
    assert scale == pytest.approx(1.0, abs=1e-12)


def test_similarity_from_h_projective_is_finite():
    from mlxstab import geometry as G
    P = np.eye(3)
    P[0, 0] = 1.01
    P[0, 2] = 2.0
    P[2, 0] = 1e-4
    angle, scale = R.similarity_from_h(P, W, H)
    assert math.isfinite(angle) and math.isfinite(scale) and scale > 0


# --------------------------------------------------------------------------------------
# magnitude statistics and spatial grid
# --------------------------------------------------------------------------------------

def test_stats_exact():
    st = R._stats([3.0, 4.0, 5.0])
    assert st["n"] == 3 and st["max"] == pytest.approx(5.0) and st["min"] == pytest.approx(3.0)
    assert st["rms"] == pytest.approx(math.sqrt((9 + 16 + 25) / 3))
    assert st["median"] == pytest.approx(4.0)
    assert st["p95"] == pytest.approx(4.9)
    empty = R._stats([])
    assert empty["n"] == 0 and all(empty[k] is None for k in ("rms", "median", "p95", "max"))


def test_magnitude_stats_on_vectors():
    st = R._magnitude_stats(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert st["n"] == 2 and st["max"] == pytest.approx(5.0) and st["rms"] == pytest.approx(math.sqrt(12.5))


def test_mesh_cell_vectors_shape_and_value():
    mesh = np.zeros((4, 5, 2))
    mesh[:, :, 0] = 2.0
    mesh[:, :, 1] = -1.0
    cells = R.mesh_cell_vectors(mesh)
    assert cells.shape == (3, 4, 2)
    assert cells[1, 2].tolist() == [2.0, -1.0]


def test_spatial_proxy_uniform_field():
    mesh = np.empty((4, 5, 2))
    mesh[:] = [1.0, 2.0]
    proxy = R.spatial_proxy(R.mesh_cell_vectors(mesh))
    assert proxy["n_cells"] == 12
    assert proxy["dispersion_rms_px"] == pytest.approx(0.0)
    assert proxy["neighbor_cosine_weighted_mean"] is None
    assert proxy["neighbor_pairs"] == 0


def test_spatial_proxy_gradient_field():
    mesh = np.zeros((4, 5, 2))
    for j in range(5):
        mesh[:, j, 0] = j                       # x residual grows with the column
    proxy = R.spatial_proxy(R.mesh_cell_vectors(mesh))
    assert proxy["dispersion_rms_px"] == pytest.approx(math.sqrt(1.25))
    assert -1.0 <= proxy["neighbor_cosine_weighted_mean"] <= 1.0
    assert proxy["neighbor_pairs"] > 0


def test_summarize_pair_reports_all_fields():
    mesh = np.empty((4, 5, 2))
    mesh[:] = [3.0, 4.0]                        # every vertex magnitude 5
    rows = np.empty((2, 2))
    rows[:] = [3.0, 4.0]
    pair = R.summarize_pair(make_fit(similarity_h(W, H, 2.0, 1.05), mesh=mesh, rows=rows),
                            pair=1, frame_index=3, width=W, height=H)
    assert pair["pair"] == 1 and pair["frame_index"] == 3
    assert pair["rotation_deg_y_down"] == pytest.approx(2.0, abs=1e-9)
    assert pair["scale"] == pytest.approx(1.05, abs=1e-9)
    assert pair["mesh_residual_magnitude_px"]["rms"] == pytest.approx(5.0)
    assert pair["mesh_residual_magnitude_px"]["max"] == pytest.approx(5.0)
    assert pair["row_residual_magnitude_px"]["rms"] == pytest.approx(5.0)
    assert pair["row_residual_magnitude_px"]["median"] == pytest.approx(5.0)
    assert pair["spatial_grid"]["proxy"]["n_cells"] == 12


def test_summarize_pair_raw_vertex_and_cell_grid_shapes():
    gc, gr = 4, 3
    mesh = np.zeros((gr + 1, gc + 1, 2))
    pair = R.summarize_pair(make_fit(mesh=mesh, gc=gc, gr=gr), pair=0, frame_index=0, width=W, height=H)
    raw_vec = np.asarray(pair["mesh_vertex_residual_vectors_px"])
    raw_mag = np.asarray(pair["mesh_vertex_residual_magnitudes_px"])
    cell_vec = np.asarray(pair["spatial_grid"]["cell_vectors_px"])
    cell_mag = np.asarray(pair["spatial_grid"]["cell_magnitudes_px"])
    # Raw mesh residuals are the per-VERTEX field [gr+1, gc+1, 2]; the averaged spatial grid is per-CELL.
    assert raw_vec.shape == (gr + 1, gc + 1, 2)
    assert raw_mag.shape == (gr + 1, gc + 1)
    assert cell_vec.shape == (gr, gc, 2)
    assert cell_mag.shape == (gr, gc)
    # The raw vertex arrays must not masquerade under the old cell names; true cells keep their keys.
    assert "mesh_cell_vectors_px" not in pair
    assert "mesh_cell_magnitudes_px" not in pair
    assert "cell_vectors_px" in pair["spatial_grid"]
    assert "cell_magnitudes_px" in pair["spatial_grid"]


def test_summarize_pair_fallback_metadata():
    pair = R.summarize_pair(make_fit(fallback=True), pair=0, frame_index=0, width=W, height=H)
    assert pair["fallback"] is True
    assert pair["scale"] == pytest.approx(1.0, abs=1e-9)


def test_summarize_pair_reports_fit_gate_diagnostics():
    pair = R.summarize_pair(
        make_fit(fallback=True, n_valid_samples=42, n_ransac_inliers=0,
                 fallback_reason="insufficient_valid_samples"),
        pair=3, frame_index=3, width=W, height=H)
    assert pair["n_valid_samples"] == 42 and pair["n_ransac_inliers"] == 0
    assert pair["fallback_reason"] == "insufficient_valid_samples"
    assert pair["fit_metadata_available"] is True
    json.dumps(R.json_safe(pair), allow_nan=False, sort_keys=True)


def test_summarize_pair_fit_gate_unavailable_for_manual_fit():
    pair = R.summarize_pair(make_fit(), pair=0, frame_index=0, width=W, height=H)
    assert pair["n_valid_samples"] is None and pair["n_ransac_inliers"] is None
    assert pair["fallback_reason"] is None and pair["fit_metadata_available"] is False


# --------------------------------------------------------------------------------------
# row-band line profile (x-independent Fit.rows vs band centre y)
# --------------------------------------------------------------------------------------

def _band_centers(nr, height=H):
    return (np.arange(nr, dtype=np.float64) + 0.5) * height / nr - 0.5


def test_row_line_profile_exact_linear_shear():
    nr = 4
    cy = _band_centers(nr)
    slope = np.array([0.04, -0.01])
    rows = np.stack([slope[0] * cy, slope[1] * cy], axis=1)
    prof = R.row_line_profile(rows, H)
    assert prof["available"] is True and prof["n_bands"] == nr
    assert prof["slope_px_per_row"] == pytest.approx(slope.tolist(), abs=1e-12)
    assert prof["delta_first_to_last_px"] == pytest.approx((slope * (cy[-1] - cy[0])).tolist(), abs=1e-12)
    assert prof["nonlinear_residual_rms_px"] == pytest.approx(0.0, abs=1e-9)
    assert prof["linear_explained_fraction"] == pytest.approx(1.0, abs=1e-12)
    json.dumps(prof, allow_nan=False)


def test_row_line_profile_nonlinear_field_has_residual_and_partial_fraction():
    nr = 8
    cy = _band_centers(nr)
    rows = np.zeros((nr, 2))
    rows[:, 0] = 0.05 * cy                      # exactly linear component
    rows[:, 1] = 0.05 * cy + 0.5 * np.sin(cy)   # linear + nonlinear component
    prof = R.row_line_profile(rows, H)
    assert prof["available"] is True
    assert prof["nonlinear_residual_rms_px"] > 0.0
    assert 0.0 < prof["linear_explained_fraction"] < 1.0
    json.dumps(prof, allow_nan=False)


def test_row_line_profile_constant_field_is_finite_and_null_fraction():
    rows = np.tile([2.5, -1.25], (5, 1))
    prof = R.row_line_profile(rows, H)
    assert prof["available"] is True and prof["n_bands"] == 5
    assert prof["slope_px_per_row"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert prof["delta_first_to_last_px"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert prof["nonlinear_residual_rms_px"] == pytest.approx(0.0, abs=1e-9)
    assert prof["linear_explained_fraction"] is None    # TSS = 0 -> not identifiable
    json.dumps(prof, allow_nan=False)


@pytest.mark.parametrize("nr", [0, 1])
def test_row_line_profile_one_band_and_empty_are_disabled_and_finite(nr):
    prof = R.row_line_profile(np.zeros((nr, 2)), H)
    assert prof["available"] is False and prof["n_bands"] == nr
    assert prof["slope_px_per_row"] is None and prof["delta_first_to_last_px"] is None
    assert prof["linear_explained_fraction"] is None
    assert prof["nonlinear_residual_rms_px"] == 0.0     # finite-safe, not nan/inf
    json.dumps(prof, allow_nan=False)


def test_row_line_profile_rejects_bad_shape():
    with pytest.raises(ValueError):
        R.row_line_profile(np.zeros((3,)), H)
    with pytest.raises(ValueError):
        R.row_line_profile(np.zeros((3, 3)), H)


def test_summarize_pair_reports_row_line_profile():
    nr = 4
    cy = _band_centers(nr)
    rows = np.stack([0.01 * cy, -0.02 * cy], axis=1)
    pair = R.summarize_pair(make_fit(rows=rows, nr=nr), pair=0, frame_index=0, width=W, height=H)
    prof = pair["row_line_profile"]
    assert prof["available"] is True and prof["n_bands"] == nr
    assert prof["slope_px_per_row"] == pytest.approx([0.01, -0.02], abs=1e-9)
    assert prof["nonlinear_residual_rms_px"] == pytest.approx(0.0, abs=1e-9)
    assert prof["linear_explained_fraction"] == pytest.approx(1.0, abs=1e-9)


def test_report_aggregates_row_line_profile(tmp_path):
    nr = 4
    cy = _band_centers(nr)
    f1 = make_fit(rows=np.stack([0.01 * cy, np.zeros(nr)], axis=1), nr=nr)
    f2 = make_fit(rows=np.stack([np.zeros(nr), 0.02 * cy], axis=1), nr=nr)
    manifest, *_ = save_case(tmp_path, name="rowagg", fits=[f1, f2], nr=nr)
    rep = R.run_real_motion_field_audit([manifest])
    summary = rep["cases"][0]["summary"]
    rlp = summary["row_line_profile"]
    assert rlp["available_fits"] == 2 and rlp["n_fits"] == 2 and rlp["all_available"] is True
    assert summary["per_pair_row_line_available"] == [True, True]
    assert summary["per_pair_row_slope_magnitude_px_per_row"] == [pytest.approx(0.01), pytest.approx(0.02)]
    assert summary["per_pair_row_delta_magnitude_px"] == [
        pytest.approx(0.01 * (cy[-1] - cy[0])), pytest.approx(0.02 * (cy[-1] - cy[0]))]
    assert rlp["slope_magnitude_px_per_row"]["max"] == pytest.approx(0.02)
    assert rlp["nonlinear_residual_rms_px"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert rlp["linear_explained_fraction"]["min"] == pytest.approx(1.0, abs=1e-9)
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_one_band_disabled_row_field_is_finite_safe(tmp_path):
    # rs_rows <= 0 yields a single zero band; the row line is then not identifiable but must stay finite-safe.
    fit = make_fit(rows=np.zeros((1, 2)), nr=1)
    manifest, *_ = save_case(tmp_path, name="rowone", fits=[fit], nr=1)
    rep = R.run_real_motion_field_audit([manifest])
    prof = rep["cases"][0]["pairs"][0]["row_line_profile"]
    assert prof["available"] is False and prof["n_bands"] == 1
    assert prof["slope_px_per_row"] is None and prof["delta_first_to_last_px"] is None
    assert prof["nonlinear_residual_rms_px"] == 0.0 and prof["linear_explained_fraction"] is None
    rlp = rep["cases"][0]["summary"]["row_line_profile"]
    assert rlp["available_fits"] == 0 and rlp["all_available"] is False
    assert rlp["slope_magnitude_px_per_row"]["n"] == 0
    assert rep["cases"][0]["summary"]["per_pair_row_slope_magnitude_px_per_row"] == [None]
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_row_line_profile_is_available_for_legacy_motion(tmp_path):
    # Fit.rows exists in every motion schema, so the unweighted line profile stays available with null support
    # and cycle maps on legacy /1 motion, and the report keeps its /4 schema.
    manifest, motion_path, run, mot = save_case(tmp_path, name="rowlegacy")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.asarray("mlxstab.motion/1")
    for key in RCA.MOTION_SUPPORT_KEYS + RCA.MOTION_CYCLE_KEYS:
        del payload[key]
    np.savez_compressed(motion_path, **payload)
    rep = R.run_real_motion_field_audit([manifest])
    assert rep["schema"] == "mlxstab.real_motion_field_audit/5"
    pair = rep["cases"][0]["pairs"][0]
    assert pair["row_line_profile"]["available"] is True
    assert pair["row_line_profile"]["n_bands"] == 2
    assert pair["mesh_vertex_support_mass"] is None
    assert pair["cycle_consistency"]["available"] is False
    assert rep["cases"][0]["summary"]["per_pair_row_line_available"] == [True]
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_row_line_definition_and_limitations(tmp_path):
    manifest, *_ = save_case(tmp_path, name="rowdef")
    rep = R.run_real_motion_field_audit([manifest])
    defn = rep["row_line_definition"]
    assert "rolling-shutter" in defn["is_not"] and "readout" in defn["is_not"]
    assert "unweighted" in defn and "support masses" in defn["unweighted"]
    assert "band_centers" in defn
    assert rep["motion_payload_safety"]["row_line_note"]
    assert any("row-band line profile" in lim for lim in rep["limitations"])
    assert "cycle maps" in " ".join(rep["limitations"])


# --------------------------------------------------------------------------------------
# support mass (observation coverage) and supported-vs-filled residuals
# --------------------------------------------------------------------------------------

def test_support_split_groups_by_threshold():
    s = R.support_split(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 2.9, 3.0, 10.0]))
    assert s["threshold"] == 3.0
    assert s["n_supported"] == 2 and s["n_filled"] == 2
    assert s["supported_fraction"] == pytest.approx(0.5)
    assert s["supported_residual_magnitude_px"]["mean"] == pytest.approx(3.5)
    assert s["filled_residual_magnitude_px"]["mean"] == pytest.approx(1.5)
    assert R.support_split(np.array([1.0]), None) is None
    with pytest.raises(ValueError, match="size"):
        R.support_split(np.array([1.0, 2.0]), np.array([1.0]))


def test_summarize_pair_reports_support_maps_and_split():
    gc, gr, nr = 4, 3, 2
    mesh = np.zeros((gr + 1, gc + 1, 2)); mesh[..., 0] = 5.0
    rows = np.zeros((nr, 2)); rows[..., 1] = 4.0
    msm = np.full((gr + 1, gc + 1), 3.0); msm[0, 0] = 1.0            # one filled vertex
    rsm = np.array([3.0, 1.0])                                      # one filled band
    pair = R.summarize_pair(
        make_fit(mesh=mesh, rows=rows, gc=gc, gr=gr, nr=nr, mesh_support_mass=msm, row_support_mass=rsm),
        pair=0, frame_index=0, width=W, height=H)
    assert pair["support_threshold"] == 3.0
    assert np.asarray(pair["mesh_vertex_support_mass"]).shape == (gr + 1, gc + 1)
    assert pair["row_band_support_mass"] == [3.0, 1.0]
    ms = pair["mesh_supported_vs_filled"]
    assert ms["n_supported"] == gr * (gc + 1) + gc and ms["n_filled"] == 1
    assert ms["filled_residual_magnitude_px"]["rms"] == pytest.approx(5.0)
    rs = pair["row_supported_vs_filled"]
    assert rs["n_supported"] == 1 and rs["n_filled"] == 1
    assert rs["filled_residual_magnitude_px"]["rms"] == pytest.approx(4.0)


def test_summarize_pair_support_is_none_for_legacy_fit():
    pair = R.summarize_pair(make_fit(), pair=0, frame_index=0, width=W, height=H)
    assert pair["mesh_vertex_support_mass"] is None and pair["row_band_support_mass"] is None
    assert pair["mesh_supported_vs_filled"] is None and pair["row_supported_vs_filled"] is None


# --------------------------------------------------------------------------------------
# forward-backward cycle-consistency maps
# --------------------------------------------------------------------------------------

def test_cycle_pair_summary_none_when_unavailable():
    s = R.cycle_pair_summary(None, None, None)
    assert s["available"] is False and s["n_vertices_with_samples"] == 0
    assert s["error_rms_px"]["n"] == 0 and s["inconsistency_fraction"]["n"] == 0
    assert s["inconsistency_fraction_mass_weighted"] is None and s["sample_mass_total"] is None


def test_summarize_pair_reports_cycle_maps_and_stats():
    gc, gr = 4, 3
    rms = np.full((gr + 1, gc + 1), 0.8); rms[0, 0] = 0.2
    frac = np.full((gr + 1, gc + 1), 0.25); frac[0, 0] = 0.0
    mass = np.full((gr + 1, gc + 1), 4.0); mass[0, 0] = 0.0        # zero-sample vertex excluded
    pair = R.summarize_pair(
        make_fit(gc=gc, gr=gr, cycle_err_rms=rms, cycle_inconsistency_frac=frac, cycle_sample_mass=mass),
        pair=0, frame_index=0, width=W, height=H)
    assert np.asarray(pair["mesh_cycle_error_rms_px"]).shape == (gr + 1, gc + 1)
    cc = pair["cycle_consistency"]
    assert cc["available"] is True
    assert cc["n_vertices_with_samples"] == (gr + 1) * (gc + 1) - 1
    assert cc["error_rms_px"]["rms"] == pytest.approx(0.8)        # the zero-mass vertex is excluded
    assert cc["inconsistency_fraction"]["mean"] == pytest.approx(0.25)
    assert cc["inconsistency_fraction_mass_weighted"] == pytest.approx(0.25)
    assert cc["sample_mass_total"] == pytest.approx(4.0 * ((gr + 1) * (gc + 1) - 1))


def test_summarize_pair_cycle_is_unavailable_for_manual_fit():
    pair = R.summarize_pair(make_fit(), pair=0, frame_index=0, width=W, height=H)
    assert pair["mesh_cycle_error_rms_px"] is None
    assert pair["mesh_cycle_inconsistency_fraction"] is None
    assert pair["mesh_cycle_sample_mass"] is None
    assert pair["cycle_consistency"]["available"] is False


def test_report_cycle_maps_and_pooled_stats(tmp_path):
    gc, gr, nr = 4, 3, 2
    rms = np.full((gr + 1, gc + 1), 0.5)
    frac = np.full((gr + 1, gc + 1), 0.5)
    mass = np.full((gr + 1, gc + 1), 2.0)
    fit = make_fit(gc=gc, gr=gr, nr=nr, cycle_err_rms=rms, cycle_inconsistency_frac=frac,
                   cycle_sample_mass=mass)
    manifest, *_ = save_case(tmp_path, name="cycle", fits=[fit], gc=gc, gr=gr, nr=nr)
    rep = R.run_real_motion_field_audit([manifest])
    pair = rep["cases"][0]["pairs"][0]
    assert np.asarray(pair["mesh_cycle_error_rms_px"]).shape == (gr + 1, gc + 1)
    assert pair["cycle_consistency"]["available"] is True
    assert pair["cycle_consistency"]["error_rms_px"]["rms"] == pytest.approx(0.5)
    summary = rep["cases"][0]["summary"]
    cc = summary["cycle_consistency"]
    assert cc["available_fits"] == 1 and cc["n_fits"] == 1 and cc["all_available"] is True
    assert cc["error_rms_px"]["mean"] == pytest.approx(0.5)
    assert cc["inconsistency_fraction"]["mean"] == pytest.approx(0.5)
    assert cc["inconsistency_fraction_mass_weighted"] == pytest.approx(0.5)
    assert cc["sample_mass_total"] == pytest.approx(2.0 * (gr + 1) * (gc + 1))
    assert summary["per_pair_cycle_available"] == [True]
    assert summary["per_pair_cycle_error_rms_px"] == [pytest.approx(0.5)]
    assert rep["cycle_consistency_definition"]["is_not"]
    assert "correlation" in rep["cycle_consistency_definition"]["is_not"]
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_cycle_is_null_for_manual_fit(tmp_path):
    manifest, *_ = save_case(tmp_path, name="manualcycle")            # manual Fit -> no cycle maps
    rep = R.run_real_motion_field_audit([manifest])
    pair = rep["cases"][0]["pairs"][0]
    assert pair["mesh_cycle_error_rms_px"] is None and pair["mesh_cycle_inconsistency_fraction"] is None
    assert pair["mesh_cycle_sample_mass"] is None
    assert pair["cycle_consistency"]["available"] is False
    summary = rep["cases"][0]["summary"]
    assert summary["per_pair_cycle_available"] == [False]
    assert summary["per_pair_cycle_error_rms_px"] == [None]
    assert summary["cycle_consistency"]["available_fits"] == 0
    assert summary["cycle_consistency"]["all_available"] is False
    assert summary["cycle_consistency"]["error_rms_px"]["n"] == 0
    assert summary["cycle_consistency"]["inconsistency_fraction_mass_weighted"] is None
    assert summary["cycle_consistency"]["sample_mass_total"] is None
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_cycle_is_null_for_legacy_v2(tmp_path):
    manifest, motion_path, run, mot = save_case(tmp_path, name="v2cycle")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.asarray("mlxstab.motion/2")
    for key in RCA.MOTION_CYCLE_KEYS:
        del payload[key]
    np.savez_compressed(motion_path, **payload)
    rep = R.run_real_motion_field_audit([manifest])
    pair = rep["cases"][0]["pairs"][0]
    assert pair["mesh_cycle_error_rms_px"] is None
    assert rep["motion_payload_safety"]["legacy_cycle"]
    # Support mass (a /2 feature) is still present.
    assert pair["mesh_vertex_support_mass"] is not None
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_cycle_is_null_for_legacy_v1(tmp_path):
    manifest, motion_path, run, mot = save_case(tmp_path, name="v1cycle")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.asarray("mlxstab.motion/1")
    for key in RCA.MOTION_SUPPORT_KEYS + RCA.MOTION_CYCLE_KEYS + RCA.MOTION_FIT_META_KEYS:
        del payload[key]
    np.savez_compressed(motion_path, **payload)
    rep = R.run_real_motion_field_audit([manifest])
    pair = rep["cases"][0]["pairs"][0]
    assert pair["mesh_cycle_error_rms_px"] is None and pair["mesh_vertex_support_mass"] is None
    assert pair["fallback_reason"] is None and pair["fit_metadata_available"] is False
    assert rep["motion_payload_safety"]["legacy_cycle"]
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_fallback_reason_counts_and_per_pair_series(tmp_path):
    fits = [
        make_fit(n_valid_samples=150, n_ransac_inliers=140, fallback_reason="none"),
        make_fit(fallback=True, n_valid_samples=12, n_ransac_inliers=0,
                 fallback_reason="insufficient_valid_samples"),
        make_fit(fallback=True, n_valid_samples=200, n_ransac_inliers=20,
                 fallback_reason="insufficient_ransac_inliers"),
        make_fit(fallback=True, n_valid_samples=500, n_ransac_inliers=60,
                 fallback_reason="low_ransac_inlier_fraction"),
    ]
    manifest, *_ = save_case(tmp_path, name="reasons", fits=fits)
    rep = R.run_real_motion_field_audit([manifest])
    summary = rep["cases"][0]["summary"]
    counts = summary["fallback_reason_counts"]
    assert counts["none"] == 1
    assert counts["insufficient_valid_samples"] == 1
    assert counts["insufficient_ransac_inliers"] == 1
    assert counts["low_ransac_inlier_fraction"] == 1
    assert counts["unavailable"] == 0
    assert summary["fallback_reason_unavailable_pairs"] == []
    assert summary["n_fallback"] == 3
    assert summary["per_pair_fallback_reason"] == [
        "none", "insufficient_valid_samples", "insufficient_ransac_inliers", "low_ransac_inlier_fraction"]
    assert summary["per_pair_n_valid_samples"] == [150, 12, 200, 500]
    assert summary["per_pair_n_ransac_inliers"] == [140, 0, 20, 60]
    assert summary["per_pair_fit_metadata_available"] == [True, True, True, True]
    pair_reasons = {p["pair"]: p["fallback_reason"] for p in rep["cases"][0]["pairs"]}
    assert pair_reasons == {0: "none", 1: "insufficient_valid_samples",
                            2: "insufficient_ransac_inliers", 3: "low_ransac_inlier_fraction"}
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_fallback_reason_unavailable_for_legacy(tmp_path):
    manifest, motion_path, run, mot = save_case(tmp_path, name="reasonlegacy")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.asarray("mlxstab.motion/3")
    for key in RCA.MOTION_FIT_META_KEYS:
        del payload[key]
    np.savez_compressed(motion_path, **payload)
    rep = R.run_real_motion_field_audit([manifest])
    summary = rep["cases"][0]["summary"]
    assert summary["fallback_reason_counts"]["unavailable"] == 1
    assert summary["fallback_reason_counts"]["none"] == 0
    assert summary["fallback_reason_unavailable_pairs"] == [0]
    assert summary["per_pair_fallback_reason"] == [None]
    assert summary["per_pair_fit_metadata_available"] == [False]
    pair = rep["cases"][0]["pairs"][0]
    assert pair["n_valid_samples"] is None and pair["n_ransac_inliers"] is None
    assert pair["fallback_reason"] is None
    assert rep["motion_payload_safety"]["legacy_fit_metadata"]
    assert rep["fallback_reason_definition"]["availability"]
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


# --------------------------------------------------------------------------------------
# manifest + motion loading
# --------------------------------------------------------------------------------------

def test_load_manifest_motion_roundtrip(tmp_path):
    manifest, motion_path, *_ = save_case(tmp_path, name="rt")
    ident, mot, meta = R.load_manifest_motion(manifest)
    assert ident["status"] == "success" and ident["capture_dump_motion"] is True
    assert ident["motion_path"] == str(motion_path)
    assert meta["width"] == W and meta["height"] == H and meta["fps"] == 30.0
    assert len(mot.fits) == 1


def test_resolve_motion_path_absolute_and_sibling(tmp_path):
    manifest, motion_path, run, *_ = save_case(tmp_path, name="res")
    assert R.resolve_motion_path(manifest, str(motion_path)) == motion_path
    assert R.resolve_motion_path(manifest, "motion_r0.npz") == run / "motion_r0.npz"
    assert R.resolve_motion_path(manifest, "does_not_exist.npz") is None


def test_frame_indexing_under_stride(tmp_path):
    fits = [make_fit() for _ in range(3)]
    manifest, *_ = save_case(tmp_path, name="stride", fits=fits, stride=3)
    rep = R.run_real_motion_field_audit([manifest])
    idx = [p["pair"] for p in rep["cases"][0]["pairs"]]
    frames = [p["frame_index"] for p in rep["cases"][0]["pairs"]]
    assert idx == [0, 1, 2] and frames == [0, 3, 6]


# --------------------------------------------------------------------------------------
# report shape, JSON safety and nested windows
# --------------------------------------------------------------------------------------

def test_report_is_json_safe_and_has_no_depth_claim(tmp_path):
    manifest, *_ = save_case(tmp_path, name="json")
    rep = R.run_real_motion_field_audit([manifest])
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)
    assert rep["schema"] == R.SCHEMA and rep["ok"] is True
    assert len(rep["cases"]) == 1 and rep["cases"][0]["summary"]["n_pairs"] == 1
    for bad in ("quality_score", "winner", "ranking", "metric_depth"):
        assert bad not in rep
    assert "winner" in rep["scope"] and "no composite" in rep["scope"]
    assert "NOT metric depth" in rep["scope"]
    assert any("parallax" in lim for lim in rep["limitations"])
    assert "is_not_depth" in rep["spatial_proxy_definition"]
    assert rep["support_definition"]["threshold"] == 3.0
    assert "NOT a probability" in rep["support_definition"]["is_not"]
    assert any("support mass" in lim for lim in rep["limitations"])


def test_report_includes_support_maps_and_supported_vs_filled(tmp_path):
    gc, gr, nr = 4, 3, 2
    msm = np.full((gr + 1, gc + 1), 5.0); msm[0, 0] = 0.0           # one filled vertex
    rsm = np.array([0.0, 9.0])                                     # one filled band
    fit = make_fit(mesh=np.zeros((gr + 1, gc + 1, 2)), rows=np.zeros((nr, 2)), gc=gc, gr=gr, nr=nr,
                   mesh_support_mass=msm, row_support_mass=rsm)
    manifest, motion_path, run, mot = save_case(tmp_path, name="support", fits=[fit], gc=gc, gr=gr, nr=nr)
    rep = R.run_real_motion_field_audit([manifest])
    pair = rep["cases"][0]["pairs"][0]
    assert np.asarray(pair["mesh_vertex_support_mass"]).shape == (gr + 1, gc + 1)
    assert pair["mesh_supported_vs_filled"]["n_filled"] == 1
    assert pair["row_supported_vs_filled"]["n_supported"] == 1
    summary = rep["cases"][0]["summary"]
    assert summary["per_pair_mesh_supported_fraction"] == [pytest.approx(19 / 20)]
    assert summary["mesh_supported_vs_filled"]["threshold"] == 3.0
    assert summary["mesh_supported_vs_filled"]["supported_residual_magnitude_px"]["n"] == 19
    assert summary["mesh_supported_vs_filled"]["filled_residual_magnitude_px"]["n"] == 1
    assert summary["row_supported_vs_filled"]["filled_residual_magnitude_px"]["n"] == 1
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_report_is_legacy_safe_when_support_absent(tmp_path):
    manifest, motion_path, run, mot = save_case(tmp_path, name="legacy")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.asarray("mlxstab.motion/1")
    for key in ("mesh_support_mass", "row_support_mass"):
        del payload[key]
    np.savez_compressed(motion_path, **payload)
    rep = R.run_real_motion_field_audit([manifest])
    pair = rep["cases"][0]["pairs"][0]
    assert pair["mesh_vertex_support_mass"] is None and pair["row_band_support_mass"] is None
    assert pair["mesh_supported_vs_filled"] is None and pair["row_supported_vs_filled"] is None
    summary = rep["cases"][0]["summary"]
    assert summary["per_pair_mesh_supported_fraction"] == [None]
    assert summary["mesh_supported_vs_filled"]["supported_residual_magnitude_px"]["n"] == 0
    assert rep["motion_payload_safety"]["legacy_support"]
    json.dumps(R.json_safe(rep), allow_nan=False, sort_keys=True)


def test_nested_windows_are_allowed_and_recorded(tmp_path):
    m1, *_ = save_case(tmp_path, name="n1", case="t060", window={"start_s": 60.0, "duration_s": 1.0})
    m2, *_ = save_case(tmp_path, name="n2", case="t800", window={"start_s": 800.0, "duration_s": 1.0})
    rep = R.run_real_motion_field_audit([m1, m2])
    assert {c["case"] for c in rep["cases"]} == {"t060", "t800"}
    windows = {c["case"]: c["source_window"] for c in rep["cases"]}
    assert windows["t060"]["start_s"] == 60.0 and windows["t800"]["start_s"] == 800.0
    assert rep["identity"]["source_window_is_not_matched"] is True


# --------------------------------------------------------------------------------------
# refusals: manifest shape, status, provenance and source identity
# --------------------------------------------------------------------------------------

def test_refuses_missing_manifest(tmp_path):
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([tmp_path / "nope" / "manifest.json"])
    assert ei.value.code == "manifest_missing"


def test_refuses_unreadable_manifest(tmp_path):
    run = tmp_path / "bad"; run.mkdir()
    (run / "manifest.json").write_text("{not json")
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([run / "manifest.json"])
    assert ei.value.code == "manifest_unreadable"


def test_refuses_non_object_manifest(tmp_path):
    run = tmp_path / "listman"; run.mkdir()
    (run / "manifest.json").write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([run / "manifest.json"])
    assert ei.value.code == "manifest_malformed" and "top level" in str(ei.value)


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("status"),
    lambda m: m.pop("outputs"),
    lambda m: m["outputs"].pop("motion"),
    lambda m: m["outputs"].__setitem__("motion", ""),
    lambda m: m["outputs"].__setitem__("motion", 7),
    lambda m: m.pop("source"),
    lambda m: m["source"].pop("sha256"),
    lambda m: m["source"].pop("window"),
    lambda m: m["source"].pop("preprocessing"),
    lambda m: m["source"].pop("decoded"),
    lambda m: m["source"]["decoded"].pop("fps"),
    lambda m: m.pop("weights"),
    lambda m: m.pop("case"),
])
def test_refuses_malformed_manifest(tmp_path, mutate):
    manifest, *_ = save_case(tmp_path, name="mut")
    man = json.loads(manifest.read_text())
    mutate(man)
    manifest.write_text(json.dumps(man))
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "manifest_malformed"


def test_refuses_non_success_manifest(tmp_path):
    manifest, *_ = save_case(tmp_path, name="fail", status="failure")
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "manifest_not_success" and "failure" in str(ei.value)


def test_refuses_missing_motion_file(tmp_path):
    manifest, motion_path, *_ = save_case(tmp_path, name="nomotion")
    motion_path.unlink()
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "motion_missing"


def test_refuses_motion_dimension_mismatch(tmp_path):
    manifest, *_ = save_case(tmp_path, name="dim", decoded={"width": 32, "height": 24, "fps": 30.0})
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "motion_identity_mismatch"


def test_refuses_motion_fps_mismatch(tmp_path):
    manifest, *_ = save_case(tmp_path, name="fps", decoded={"width": W, "height": H, "fps": 24.0})
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "motion_identity_mismatch" and "fps" in str(ei.value)


def test_refuses_mismatched_source_and_weights(tmp_path):
    m1, *_ = save_case(tmp_path, name="h1", source_sha="a" * 64)
    m2, *_ = save_case(tmp_path, name="h2", source_sha="b" * 64)
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([m1, m2])
    assert ei.value.code == "mismatched_source"

    m3, *_ = save_case(tmp_path, name="w3", source_sha="a" * 64, weights_sha="c" * 64)
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([m1, m3])
    assert ei.value.code == "mismatched_weights"


def test_refuses_mismatched_preprocessing(tmp_path):
    m1, *_ = save_case(tmp_path, name="p1")
    m2, *_ = save_case(tmp_path, name="p2", preprocessing={"interlaced": True})
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([m1, m2])
    assert ei.value.code == "mismatched_preprocessing"


def test_refuses_mismatched_decode_between_manifests(tmp_path):
    m1, *_ = save_case(tmp_path, name="d1")
    m2, *_ = save_case(tmp_path, name="d2", W=32, H=24, decoded={"width": 32, "height": 24, "fps": 30.0},
                       fits=[make_fit()])
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([m1, m2])
    assert ei.value.code == "mismatched_decode"


@pytest.mark.parametrize("case2", [
    dict(W=32, H=24, decoded={"width": 32, "height": 24, "fps": 30.0}),  # differing width+height
    dict(fps=24.0, decoded={"width": W, "height": H, "fps": 24.0}),      # differing fps
], ids=["width-height", "fps"])
def test_refuses_mismatched_decoded_identity_fields(tmp_path, case2):
    m1, *_ = save_case(tmp_path, name="id1")
    m2, *_ = save_case(tmp_path, name="id2", fits=[make_fit()], **case2)
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([m1, m2])
    assert ei.value.code == "mismatched_decode"


def test_same_source_distinct_windows_and_decoded_counts_are_audited_together(tmp_path):
    # Same source (sha256), preprocessing, weights and decoded width/height/fps, but a longer capture window
    # makes decoded.frames/duration_s larger. Those follow the window length and are provenance, not cross-run
    # source identity: the two runs must be audited together rather than refused as a decode mismatch.
    m1, *_ = save_case(
        tmp_path, name="win1s", source_sha="a" * 64,
        window={"start_s": 60.0, "duration_s": 1.0},
        decoded={"width": W, "height": H, "fps": 30.0, "frames": 60, "duration_s": 1.0})
    m2, *_ = save_case(
        tmp_path, name="win10s", source_sha="a" * 64,
        window={"start_s": 60.0, "duration_s": 10.0},
        decoded={"width": W, "height": H, "fps": 30.0, "frames": 600, "duration_s": 10.0})
    rep = R.run_real_motion_field_audit([m1, m2])
    assert rep["ok"] is True
    assert len(rep["cases"]) == 2
    assert rep["identity"]["matched_source_identity"] is True
    assert rep["identity"]["decoded_identity"] == {"width": W, "height": H, "fps": 30.0}
    # Full decoded provenance is retained per manifest, including the window-dependent frames/duration.
    manifests = {m["manifest_path"]: m for m in rep["identity"]["manifests"]}
    assert manifests[str(m1)]["decoded"]["frames"] == 60
    assert manifests[str(m2)]["decoded"]["frames"] == 600
    assert manifests[str(m2)]["decoded"]["duration_s"] == pytest.approx(10.0)
    # The windows are recorded per case and remain distinct.
    assert [c["source_window"]["duration_s"] for c in rep["cases"]] == [1.0, 10.0]


def test_duplicate_case_names_are_retained_and_distinguished(tmp_path):
    # The t060/t800 baseline captures share the case name `baseline`; both must be retained and told apart by
    # run_id, manifest_path and source_window rather than refused.
    m1, *_ = save_case(tmp_path, name="dup060", case="baseline", window={"start_s": 60.0, "duration_s": 1.0})
    m2, *_ = save_case(tmp_path, name="dup800", case="baseline", window={"start_s": 800.0, "duration_s": 1.0})
    rep = R.run_real_motion_field_audit([m1, m2])
    assert rep["ok"] is True
    cases = rep["cases"]
    assert len(cases) == 2
    assert [c["case"] for c in cases] == ["baseline", "baseline"]
    assert [c["run_id"] for c in cases] == ["dup060", "dup800"]
    assert [c["manifest_path"] for c in cases] == [str(m1), str(m2)]
    assert [c["source_window"]["start_s"] for c in cases] == [60.0, 800.0]
    assert len(set(c["run_id"] for c in cases)) == 2
    assert len(set(c["manifest_path"] for c in cases)) == 2
    assert len(set(c["source_window"]["start_s"] for c in cases)) == 2


# --------------------------------------------------------------------------------------
# payload safety (reused non-pickle loader)
# --------------------------------------------------------------------------------------

def test_bad_motion_payload_surfaces_wrapped_loader_error(tmp_path):
    manifest, motion_path, *_ = save_case(tmp_path, name="badpayload")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema"] = np.asarray("mlxstab.motion/999")
    np.savez_compressed(motion_path, **payload)
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "motion_bad_schema"


def test_loader_never_unpickles(tmp_path, monkeypatch):
    manifest, *_ = save_case(tmp_path, name="nopickle")
    calls = []
    real_load = np.load

    def spy(path, *a, **k):
        calls.append(dict(k))
        return real_load(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    R.run_real_motion_field_audit([manifest])
    assert calls and all(c.get("allow_pickle") is False for c in calls)


def test_object_array_is_rejected_without_unpickling(tmp_path):
    manifest, motion_path, *_ = save_case(tmp_path, name="objectarray")
    with np.load(motion_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    obj = np.empty((1, 3, 3), dtype=object)
    obj[:] = np.eye(3)
    payload["H"] = obj
    np.savez_compressed(motion_path, **payload)
    with pytest.raises(R.MotionFieldAuditError) as ei:
        R.run_real_motion_field_audit([manifest])
    assert ei.value.code == "motion_unreadable"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_writes_json(tmp_path):
    manifest, *_ = save_case(tmp_path, name="cli")
    out = tmp_path / "audit.json"
    assert R.main([str(manifest), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == R.SCHEMA and rep["cases"][0]["summary"]["n_pairs"] == 1


def test_cli_prints_json_to_stdout(tmp_path, capsys):
    manifest, *_ = save_case(tmp_path, name="clistd")
    assert R.main([str(manifest)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["ok"] is True


def test_cli_emits_json_safe_error(tmp_path, capsys):
    manifest, *_ = save_case(tmp_path, name="clierr", status="failure")
    assert R.main([str(manifest)]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False and err["error"]["code"] == "manifest_not_success"
    json.dumps(err, allow_nan=False)


def test_cli_error_written_to_out_file(tmp_path):
    manifest, motion_path, *_ = save_case(tmp_path, name="clierrout")
    motion_path.unlink()
    out = tmp_path / "err.json"
    assert R.main([str(manifest), "--out", str(out)]) == 2
    err = json.loads(out.read_text())
    assert err["ok"] is False and err["error"]["code"] == "motion_missing"


def test_cli_requires_a_manifest():
    with pytest.raises(SystemExit):
        R.main([])
