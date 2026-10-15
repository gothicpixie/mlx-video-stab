"""Analytic behavioral tests for complete-warp shape and coverage diagnostics.

These build explicit source-coordinate fields with known Jacobians, plus a couple of integration cases that
go through the real ``stabilize.source_coords`` / ``stabilize.render_frame`` path so the diagnostics are not
disconnected from the rendered warp.
"""
import json

import numpy as np
import pytest

from mlxstab import stabilize as S, warpmetrics as WM


def grid(H, W):
    ys, xs = np.mgrid[0:H, 0:W]
    return xs.astype(np.float64), ys.astype(np.float64)


def field(H, W, sx, sy):
    return np.stack([sx, sy], -1).astype(np.float64)


def assert_json_safe(obj):
    json.dumps(obj, allow_nan=False)


# --------------------------------------------------------------------------------------
# shape: identity, similarity, anisotropy, shear
# --------------------------------------------------------------------------------------

def test_identity_is_perfect_and_fully_supported():
    H, W = 48, 64
    xs, ys = grid(H, W)
    d = WM.diagnose(field(H, W, xs, ys), H, W)
    s = d["shape"]
    assert s["det_mean"] == pytest.approx(1.0)
    assert s["anisotropy_mean"] == pytest.approx(1.0)
    assert s["shear_deg_mean"] == pytest.approx(0.0)
    assert s["det_negative_frac"] == 0.0 and s["det_collapse_frac"] == 0.0
    c = d["coverage"]
    assert c["center_valid_frac"] == 1.0
    # exact identity edge pixels have zero-weight outside taps and must count as full support
    for k in WM.KERNELS:
        assert c[f"full_support_{k}_frac"] == 1.0


def test_similarity_rotation_scale_has_no_shear():
    H, W = 48, 64
    xs, ys = grid(H, W)
    s, th = 1.3, 0.3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    x0, y0 = W / 2, H / 2
    dx, dy = xs - x0, ys - y0
    sx = s * (R[0, 0] * dx + R[0, 1] * dy) + x0
    sy = s * (R[1, 0] * dx + R[1, 1] * dy) + y0
    d = WM.diagnose(field(H, W, sx, sy), H, W)["shape"]
    assert d["anisotropy_mean"] == pytest.approx(1.0, abs=1e-6)
    assert d["shear_deg_mean"] == pytest.approx(0.0, abs=1e-3)
    assert d["det_mean"] == pytest.approx(s * s, rel=1e-9)
    assert d["det_negative_frac"] == 0.0


def test_anisotropic_scale_and_shear_are_separated():
    H, W = 48, 64
    xs, ys = grid(H, W)
    aniso = WM.diagnose(field(H, W, 2.0 * xs, 0.5 * ys), H, W)["shape"]
    assert aniso["anisotropy_mean"] == pytest.approx(4.0, rel=1e-9)      # s1/s2
    assert aniso["shear_deg_mean"] == pytest.approx(0.0, abs=1e-9)       # anisotropic scale stays orthogonal
    assert aniso["det_mean"] == pytest.approx(1.0, rel=1e-9)
    shear = WM.diagnose(field(H, W, xs + 0.5 * ys, ys), H, W)["shape"]
    assert shear["shear_deg_p95"] == pytest.approx(np.degrees(np.arctan(0.5)), rel=1e-6)
    assert shear["anisotropy_p95"] > 1.0


# --------------------------------------------------------------------------------------
# reflection, collapse, alternating fold
# --------------------------------------------------------------------------------------

def test_reflection_and_collapse_are_not_perfect():
    H, W = 48, 64
    xs, ys = grid(H, W)
    refl = WM.diagnose(field(H, W, W - 1 - xs, ys), H, W)["shape"]       # mirror: det = -1
    assert refl["det_negative_frac"] == 1.0
    assert refl["anisotropy_mean"] == pytest.approx(1.0)                 # singular values alone look "perfect"
    # half-collapse: the left half maps normally, the right half maps x to a constant
    sx = np.where(xs < W // 2, xs, float(W // 2 - 1))
    col = WM.diagnose(field(H, W, sx, ys), H, W)["shape"]
    assert col["det_collapse_frac"] > 0.4
    assert col["det_abs_min"] == 0.0


def test_alternating_local_fold_is_not_cancelled():
    """Centred differences would cancel an alternating fold; forward differences must keep it."""
    H, W = 48, 64
    xs, ys = grid(H, W)
    alt = field(H, W, xs + ((-1.0) ** xs) * 2.0, ys)
    d = WM.diagnose(alt, H, W)["shape"]
    assert d["det_negative_frac"] > 0.4                                  # ~half the cells reverse orientation
    ident = WM.diagnose(field(H, W, xs, ys), H, W)["shape"]
    assert ident["det_negative_frac"] == 0.0


def test_opposite_corner_fold_and_concave_cell_are_detected():
    """A top-left-only Jacobian reports identity here; the four-corner check must not."""
    fold = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [-1., -1.]]], np.float64)
    df = WM.diagnose(fold, 2, 2)["shape"]
    assert df["cells"] == 1 and df["usable_cells"] == 1
    assert df["det_negative_frac"] == 1.0
    assert df["det_min"] == pytest.approx(-3.0)                         # bottom-right corner determinant

    concave = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [0.2, 0.2]]], np.float64)
    dc = WM.diagnose(concave, 2, 2)["shape"]
    assert dc["det_negative_frac"] == 1.0
    assert dc["det_min"] < 0.0                                           # negative near the far corner only


def test_cell_with_missing_finite_corner_is_nonfinite():
    bad = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [np.nan, np.nan]]], np.float64)
    d = WM.diagnose(bad, 2, 2)
    assert d["shape"]["cells"] == 1 and d["shape"]["nonfinite_cells"] == 1
    assert d["shape"]["nonfinite_cell_frac"] == 1.0
    assert d["shape"]["usable_cells"] == 0 and d["shape"]["det_min"] is None
    assert d["shape"]["anisotropy_mean"] is None
    assert_json_safe(d)


def test_stable_singular_values_match_numpy_svd():
    rng = np.random.default_rng(0)
    for scale in (1.0, 1e4, 1e150):
        J = rng.normal(0.0, scale, (200, 2, 2))
        s1, s2, det, ovf = WM._stable_svd2(J[..., 0, 0], J[..., 0, 1], J[..., 1, 0], J[..., 1, 1])
        ref = np.linalg.svd(J, compute_uv=False)
        assert not ovf.any()
        assert np.allclose(np.sort(np.stack([s1, s2], -1), -1), np.sort(ref, -1), rtol=1e-9, atol=0.0)
    # the reviewed high-shear matrix [[1, 1e4], [0, 1]]: det = 1, anisotropy ~1e8, no cancellation
    a, b, c, d = (np.array([1.0]), np.array([1e4]), np.array([0.0]), np.array([1.0]))
    s1, s2, det, ovf = WM._stable_svd2(a, b, c, d)
    ref = np.linalg.svd(np.array([[1.0, 1e4], [0.0, 1.0]]), compute_uv=False)
    assert not ovf[0] and det[0] == pytest.approx(1.0)
    assert s2[0] == pytest.approx(ref[1], rel=1e-9)
    assert s1[0] / s2[0] == pytest.approx(ref[0] / ref[1], rel=1e-6)


def test_overflow_is_accounted_not_reported_as_perfect():
    huge = np.array([[[0., 0.], [1e200, 0.]], [[0., 1e200], [1e200, 1e200]]], np.float64)
    d = WM.diagnose(huge, 2, 2)
    s = d["shape"]
    assert s["cells"] == 1 and s["nonfinite_cells"] == 0
    assert s["overflow_cells"] == 1 and s["usable_cells"] == 0
    assert s["det_mean"] is None and s["anisotropy_mean"] is None       # unusable, not a perfect score
    assert_json_safe(d)
    # aggregation must not reintroduce infinities from a frame like this
    agg = WM.aggregate([d, WM.diagnose(np.zeros((3, 3, 2)), 3, 3)])
    assert_json_safe(agg)


def test_representable_large_determinant_mean_stays_finite():
    """Four finite corner determinants of 1e308 must average to 1e308, not inf (review counterexample)."""
    scale = 1e154
    src = np.array([[[0., 0.], [scale, 0.]], [[0., scale], [scale, scale]]], np.float64)
    d = WM.diagnose(src, 2, 2)
    s = d["shape"]
    assert s["usable_cells"] == 1 and s["overflow_cells"] == 0
    assert s["det_mean"] == pytest.approx(1e308, rel=1e-12)
    assert s["anisotropy_mean"] == pytest.approx(1.0)
    assert_json_safe(d)
    agg = WM.aggregate([d])
    assert agg["shape.det_mean"]["mean"] == pytest.approx(1e308, rel=1e-12)
    assert_json_safe(agg)


def test_stats_handle_large_equal_and_opposite_sign_values():
    big = WM._stats(np.array([1e308, 1e308, 1e308]))
    assert big["mean"] == pytest.approx(1e308, rel=1e-12)
    assert big["p50"] == pytest.approx(1e308, rel=1e-12) and big["p95"] == pytest.approx(1e308, rel=1e-12)
    opp = WM._stats(np.array([1e308, -1e308, 0.0]))
    assert opp["mean"] == pytest.approx(0.0)
    assert opp["min"] == -1e308 and opp["max"] == 1e308
    assert np.isfinite(opp["p50"]) and np.isfinite(opp["p95"])
    ordinary = WM._stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert ordinary["mean"] == pytest.approx(2.5) and ordinary["p50"] == pytest.approx(2.5)
    assert_json_safe({"big": big, "opp": opp, "ordinary": ordinary})


# --------------------------------------------------------------------------------------
# nonfinite / degenerate behaviour and JSON safety
# --------------------------------------------------------------------------------------

def test_nonfinite_and_degenerate_are_explicit_and_json_safe():
    H, W = 12, 16
    xs, ys = grid(H, W)
    bad = field(H, W, xs, ys)
    bad[0, 0] = np.nan
    bad[3, 4] = np.inf
    d = WM.diagnose(bad, H, W)
    assert d["shape"]["nonfinite_frac"] > 0 and d["shape"]["nonfinite_cell_frac"] > 0
    assert d["coverage"]["nonfinite_frac"] == pytest.approx(2 / (H * W))
    # no NaN/Inf leaks into results
    def check(node):
        if isinstance(node, dict):
            for v in node.values(): check(v)
        elif isinstance(node, float):
            assert np.isfinite(node)
    check(d); assert_json_safe(d)

    for hh, ww, empty in ((1, 8, False), (8, 1, False), (0, 8, True)):
        deg = WM.diagnose(np.zeros((hh, ww, 2)), hh, ww)
        assert deg["shape"]["degenerate"] is True and deg["shape"]["cells"] == 0
        assert deg["shape"]["anisotropy_mean"] is None
        # coverage is still explicit for a single row/column; only an empty image is undefined
        assert (deg["coverage"]["center_valid_frac"] is None) is empty
        assert_json_safe(deg)

    with pytest.raises(ValueError):
        WM.diagnose(np.zeros((4, 4, 3)))


# --------------------------------------------------------------------------------------
# translated borders and filter support
# --------------------------------------------------------------------------------------

def test_translated_borders_and_filter_support():
    H, W = 48, 64
    xs, ys = grid(H, W)

    ident = WM.diagnose(field(H, W, xs, ys), H, W)["coverage"]
    assert ident["center_valid_frac"] == 1.0 and ident["full_support_lanczos3_frac"] == 1.0

    # exact integer shift: pixels that leave the frame lose center validity
    ishift = WM.diagnose(field(H, W, xs + 1.0, ys), H, W)["coverage"]
    assert ishift["center_valid_frac"] == pytest.approx((W - 1) / W)

    # fractional half-pixel shift: every point stays centre-valid, but the right edge loses full bilinear support
    fshift = WM.diagnose(field(H, W, xs + 0.5, ys), H, W)["coverage"]
    assert fshift["center_valid_frac"] == 1.0
    assert fshift["full_support_bilinear_frac"] == pytest.approx((W - 1) / W)
    assert fshift["full_support_bicubic_frac"] < fshift["full_support_bilinear_frac"]
    assert fshift["full_support_lanczos3_frac"] <= fshift["full_support_bicubic_frac"]
    assert fshift["partial_support_bilinear_frac"] == pytest.approx(1 / W)


# --------------------------------------------------------------------------------------
# integration through the real source_coords / render_frame path
# --------------------------------------------------------------------------------------

def _bent_plan(mesh=(0, 0), rs_rows=0, row_amp=1.5, mesh_amp=2.0, period=16.0):
    H, W = 48, 64
    gc, gr = mesh
    mesh_corr = np.zeros((1, gr + 1, gc + 1, 2))
    row_corr = np.zeros((1, max(rs_rows, 1), 2))
    if rs_rows:
        yy = (np.arange(rs_rows) + 0.5) * H / rs_rows
        row_corr[0, :, 1] = row_amp * np.sin(2 * np.pi * yy / period)             # vertical displacement per row
    if gc:
        yv = (np.arange(gr + 1) + 0.5) * H / gr
        for i in range(gr + 1):
            mesh_corr[0, i, :, 0] = mesh_amp * np.sin(2 * np.pi * yv[i] / period)  # x displacement across mesh rows
    return S.Plan(G=[np.eye(3)], mesh_corr=mesh_corr, row_corr=row_corr,
                  raw_quads=np.zeros((1, 4, 2)), path_quads=np.zeros((1, 4, 2)))


def test_identity_global_with_bent_row_and_mesh_is_detected_by_render_path():
    H, W = 48, 64
    frame = np.zeros((H, W, 3), np.uint8)
    flat = _bent_plan()
    cfg0 = S.Config(mesh=(0, 0), rs_rows=0, kernel="lanczos3")
    _, _, global_dist = S.render_frame(frame, flat, 0, cfg0)
    assert global_dist == pytest.approx(1.0)                                 # old global-only metric says perfect

    cfg_row = S.Config(mesh=(0, 0), rs_rows=8, kernel="lanczos3")
    _, _, row_dist, row_diag = S.render_frame(frame, _bent_plan(rs_rows=8), 0, cfg_row, diagnose=True)
    assert row_dist == pytest.approx(1.0)                                    # still perfect globally
    assert row_diag["shape"]["anisotropy_p95"] > 1.2                         # complete warp detects the bend
    assert row_diag["shape"]["det_min"] < 0.8

    cfg_mesh = S.Config(mesh=(8, 6), rs_rows=0, kernel="lanczos3")
    _, _, mesh_dist, mesh_diag = S.render_frame(frame, _bent_plan(mesh=(8, 6)), 0, cfg_mesh, diagnose=True)
    assert mesh_dist == pytest.approx(1.0)
    assert mesh_diag["shape"]["shear_deg_p95"] > 10.0                        # cross-term shear from the mesh
    assert mesh_diag["coverage"]["center_valid_frac"] > 0.95                 # a little displacement leaves the frame


def test_warp_diagnostics_api_aggregates_and_is_json_safe():
    H, W = 48, 64
    cfg = S.Config(mesh=(8, 6), rs_rows=8, kernel="bicubic")
    agg = S.warp_diagnostics(_bent_plan(mesh=(8, 6), rs_rows=8), cfg, W, H, frames=[0])
    assert "shape.anisotropy_p95" in agg and "coverage.full_support_bicubic_frac" in agg
    assert agg["shape.anisotropy_p95"]["max"] > 1.2
    assert agg["coverage.center_valid_frac"]["mean"] > 0.95
    assert_json_safe(agg)
