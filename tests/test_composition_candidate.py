"""Focused tests for the Stage 1 composition candidate (``bench/composition_candidate.py``).

These validate the candidate against independently derived ground truth and the existing audit oracle, using
the pinned analytic fixtures from ``bench/composition_audit.py``. They are NumPy-only in their assertions but
import ``mlxstab.stabilize`` (and therefore MLX) transitively; run them under the shared heavy lock alongside
the rest of the suite, never while another GPU job holds it.
"""
import warnings

import numpy as np
import pytest

from bench import composition_audit as CA
from bench import composition_candidate as CC
from mlxstab import geometry as G
from mlxstab import stabilize as S
from mlxstab.motion import Fit
from mlxstab.stabilize import Motion


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _motion(fx):
    """Build the same Motion ``composition_audit.build_plan`` builds for a fixture."""
    nr = max(fx.rs_rows, 1)
    fits = [Fit(p.H.copy(), np.asarray(p.mesh, np.float64).copy(),
                np.asarray(p.rows, np.float64).copy() if p.rows is not None else np.zeros((nr, 2)),
                1.0, 0.0, False) for p in fx.pairs]
    return Motion(fx.W, fx.H, fx.fps, fits=fits, stride=1, n_frames=fx.T)


def _plan(fx):
    pl, cfg = CA.build_plan(fx)
    return pl, cfg, _motion(fx)


@pytest.fixture(scope="module")
def fixtures():
    return CA.build_fixtures()


# --------------------------------------------------------------------------------------
# global path preservation and oracle equivalence
# --------------------------------------------------------------------------------------

def test_global_only_exactly_matches_plan_G(fixtures):
    """Zero residual: the candidate must reproduce Plan.G bit-for-bit as its global component."""
    fx = fixtures["rigid_rotation_control"]
    pl, cfg, mot = _plan(fx)
    pts = CC.output_grid(fx.W, fx.H)
    for t in range(fx.T):
        base = CC.global_component(pl, t, pts)
        np.testing.assert_allclose(base, G.apply_h(pl.G[t], pts), atol=0.0)
        cand = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=False)
        np.testing.assert_allclose(cand, base, atol=1e-9, err_msg=f"frame {t}")


def test_residual_composition_matches_oracle_all_fixtures(fixtures):
    """Raw candidate composition (highpass=False) equals the exact composed oracle on every fixture."""
    for name, fx in fixtures.items():
        pl, cfg, mot = _plan(fx)
        pts = CC.output_grid(fx.W, fx.H)
        for t in range(fx.T):
            direct = CC.compose_reference(list(mot.fits), t, pts, fx.W, fx.H)
            oracle = CA.compose_oracle(fx.pairs, t, pts, fx.W, fx.H)
            np.testing.assert_allclose(direct, oracle, atol=1e-12, err_msg=f"{name} direct frame {t}")
            cand = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=False)
            np.testing.assert_allclose(cand, oracle, atol=1e-8, err_msg=f"{name} candidate frame {t}")


def test_global_component_is_independent_of_residual(fixtures):
    """The smoothed global path must not depend on the residual channel at all."""
    a = fixtures["rigid_rotation_control"]        # H path (R, R), zero residual
    b = fixtures["constant_residual_rotation"]    # same H path, nonzero residual
    pla, _ = CA.build_plan(a)
    plb, _ = CA.build_plan(b)
    pts = CC.output_grid(a.W, a.H)
    np.testing.assert_allclose(CC.global_component(pla, 2, pts), CC.global_component(plb, 2, pts), atol=0.0)


# --------------------------------------------------------------------------------------
# central closed-form rotation transport
# --------------------------------------------------------------------------------------

def test_constant_residual_rotation_transport_closed_form(fixtures):
    """delta_2 = R_linear u and the naive-vs-composed gap is the independently derived ||u - R_linear u||."""
    fx = fixtures["constant_residual_rotation"]
    pl, cfg, mot = _plan(fx)
    W, H = fx.W, fx.H
    pts = CC.output_grid(W, H)
    R = CA.rotation_about_clockwise(W, H, 10.0)
    u = np.array([8.0, 0.0])

    # delta_2 is the transported residual, constant over the reference grid
    delta2 = CC.delta_reference(list(mot.fits), 2, pts, W, H)
    np.testing.assert_allclose(delta2, np.broadcast_to(R[:2, :2] @ u, pts.shape), atol=1e-9)

    # independent expected value, not copied from the implementation
    expected = float(np.linalg.norm(u - R[:2, :2] @ u))
    assert expected == pytest.approx(1.3944918839625307, rel=1e-9)

    naive = CA.naive_from_plan(pl, 2, pts, W, H)
    cand = CC.candidate_source_coords(mot, pl, cfg, 2, pts, highpass=False)
    np.testing.assert_allclose(np.linalg.norm(naive - cand, axis=1), expected, atol=1e-6)

    # and the candidate is the closed-form composition R(R(x)+u), not the fixed-grid sum
    closed = G.apply_h(R, G.apply_h(R, pts) + u)
    np.testing.assert_allclose(cand, closed, atol=1e-8)


# --------------------------------------------------------------------------------------
# spatial location error and parallax localisation
# --------------------------------------------------------------------------------------

def test_spatial_ramp_location_error_is_vector_exact(fixtures):
    """A linear residual must be evaluated at its source: delta_1 = L z and the gap is exactly L T."""
    fx = fixtures["spatial_ramp_translation"]
    pl, cfg, mot = _plan(fx)
    W, H = fx.W, fx.H
    pts = CC.output_grid(W, H)
    L = np.diag([0.02, 0.01]); T = np.array([6.0, 0.0])
    interior = (pts[:, 0] <= W - 14) & (pts[:, 1] > 1) & (pts[:, 1] < H - 2)

    delta1 = CC.delta_reference(list(mot.fits), 1, pts, W, H)
    np.testing.assert_allclose(delta1[interior], (pts @ L.T)[interior], atol=1e-9)

    naive = CA.naive_from_plan(pl, 1, pts, W, H)
    cand = CC.candidate_source_coords(mot, pl, cfg, 1, pts, highpass=False)
    np.testing.assert_allclose((naive - cand)[interior],
                               np.broadcast_to(L @ T, pts.shape)[interior], atol=1e-9)


def test_parallax_error_is_localised_to_the_foreground(fixtures):
    fx = fixtures["parallax_two_layer"]
    pl, cfg, mot = _plan(fx)
    W, H = fx.W, fx.H
    pts = CC.output_grid(W, H)
    reg = fx.region(pts)
    naive = CA.naive_from_plan(pl, 2, pts, W, H)
    cand = CC.candidate_source_coords(mot, pl, cfg, 2, pts, highpass=False)
    d = np.linalg.norm(naive - cand, axis=1)
    fg, bg = d[reg == 1], d[reg == 0]
    assert fg.mean() > bg.mean()
    assert fg.mean() > 0.1
    assert bg.mean() < 0.1


# --------------------------------------------------------------------------------------
# row/mesh sampling order
# --------------------------------------------------------------------------------------

def test_rows_mesh_order_uses_source_position(fixtures):
    """Rows are sampled at the source row (not after the mesh shift); production's order differs by 0.06 px."""
    fx = fixtures["rows_mesh_order"]
    pl, cfg, mot = _plan(fx)
    W, H = fx.W, fx.H
    pts = CC.output_grid(W, H)
    naive = CA.naive_from_plan(pl, 1, pts, W, H)      # mesh first, then rows sampled at the shifted y
    cand = CC.candidate_source_coords(mot, pl, cfg, 1, pts, highpass=False)
    diff = naive - cand
    # ``sample_rows_bilinear`` is linear in py only between the first and last band centres, i.e. while
    # v = (py+0.5)*nr/H - 0.5 lies in [0, nr-1]; outside that it clamps to the edge band (constant).
    # rows(y+shift) - rows(y) is therefore exactly slope*shift only where BOTH arguments are inside the
    # linear window: y >= row_lo and y + shift <= row_hi. Derive those bounds from the sampler instead of
    # guessing margins, so the clamped taper (0.05/0.01) is excluded rather than averaged in.
    nr = fx.rs_rows
    row_vals = np.asarray(fx.pairs[0].rows, np.float64)
    shift = float(np.abs(fx.pairs[0].mesh[..., 1]).max())              # constant mesh y-shift
    slope = float((row_vals[1, 1] - row_vals[0, 1]) / (H / nr))        # row band values are linear in y
    row_lo = 0.5 * H / nr - 0.5
    row_hi = (nr - 0.5) * H / nr - 0.5
    assert shift > 0.0 and row_hi - shift > row_lo                    # the exact window is non-empty
    interior = (pts[:, 1] >= row_lo) & (pts[:, 1] <= row_hi - shift)
    assert interior.sum() > 0
    assert np.abs(diff[interior, 0]).max() < 1e-9
    np.testing.assert_allclose(diff[interior, 1], slope * shift, atol=1e-6)


# --------------------------------------------------------------------------------------
# high-pass / smoothing behaviour
# --------------------------------------------------------------------------------------

def test_l1_short_series_highpass_is_zero_and_gauss_is_budget_capped(fixtures):
    """L1 is a no-op for T<5 (documented); with gauss the correction is nonzero and bounded by mesh_budget*W."""
    fx = fixtures["constant_residual_rotation"]
    pl, cfg, mot = _plan(fx)
    W, H = fx.W, fx.H
    pts = CC.output_grid(W, H)
    base = CC.global_component(pl, 2, pts)

    hp_l1 = CC.candidate_source_coords(mot, pl, cfg, 2, pts, highpass=True) - base
    assert np.abs(hp_l1).max() < 1e-9

    cfg_gauss = S.Config(mesh=fx.mesh_cells, rs_rows=fx.rs_rows, smooth="gauss", sigma=1.0)
    hp_g = CC.candidate_source_coords(mot, pl, cfg_gauss, 2, pts, highpass=True) - base
    assert np.abs(hp_g).max() > 0.0
    assert np.abs(hp_g).max() <= cfg_gauss.mesh_budget * W + 1e-9


def test_no_residual_leakage_into_smoothed_global_path(fixtures):
    """candidate - delta == Plan.G, so the residual correction never shifts the global path."""
    fx = fixtures["constant_residual_rotation"]
    pl, cfg, mot = _plan(fx)
    W, H = fx.W, fx.H
    pts = CC.output_grid(W, H)
    base = CC.global_component(pl, 2, pts)
    cand = CC.candidate_source_coords(mot, pl, cfg, 2, pts, highpass=False)
    V = CC.virtual_map(list(mot.fits), pl, 2)
    z = G.apply_h(V, pts)
    delta = CC.delta_reference(list(mot.fits), 2, z, W, H)
    np.testing.assert_allclose(cand - delta, base, atol=1e-9)


# --------------------------------------------------------------------------------------
# support / fold reporting
# --------------------------------------------------------------------------------------

def test_border_fixture_reports_partial_support(fixtures):
    fx = fixtures["rotation_translation_border"]
    pl, cfg, mot = _plan(fx)
    rep = CC.candidate_report(mot, pl, cfg, 2, highpass=False)
    assert rep["in_bounds_frac"] < 0.5
    assert rep["valid_cells"] > 0


def test_fold_over_is_reported_and_strict_mode_rejects():
    W, H, gc, gr = 48, 36, 4, 3
    cx = W / 2 - .5
    fold = CA.field_mesh(W, H, gc, gr, lambda p: np.stack([-2.0 * (p[:, 0] - cx), np.zeros(len(p))], 1))
    fx = CA.Fixture(name="fold_fixture", description="orientation-reversing residual", W=W, H=H, T=2,
                    pairs=[CA.PairMotion(np.eye(3), fold)], mesh_cells=(gc, gr))
    pl, cfg, mot = _plan(fx)
    field = CC.candidate_field(mot, pl, cfg, 1, highpass=False)
    rep = CC.fold_report(field, W, H)
    assert rep["fold_frac"] is not None and rep["fold_frac"] > 0.5
    with pytest.raises(CC.WarpFoldError):
        CC.candidate_field(mot, pl, cfg, 1, highpass=False, strict=True)


# --------------------------------------------------------------------------------------
# numerical helpers and scope guards
# --------------------------------------------------------------------------------------

def test_samplers_and_reference_grid_match_audit():
    W, H, gc, gr = 64, 48, 5, 4
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(-3, W + 3, 300), rng.uniform(-3, H + 3, 300)], 1)
    mesh = rng.standard_normal((gr + 1, gc + 1, 2))
    np.testing.assert_allclose(CC.sample_mesh_bilinear(mesh, pts, W, H),
                               CA.sample_mesh_bilinear(mesh, pts, W, H), atol=1e-12)
    rows = rng.standard_normal((6, 2))
    np.testing.assert_allclose(CC.sample_rows_bilinear(rows, pts[:, 1], H),
                               CA.sample_rows_bilinear(rows, pts[:, 1], H), atol=1e-12)
    np.testing.assert_allclose(CC.reference_points(W, H, (gc, gr)), CA.vertex_grid(W, H, gc, gr), atol=0.0)


def test_scope_guards_reject_stride_and_inconsistent_plan():
    W, H, gc, gr = 64, 48, 4, 3
    mesh = np.zeros((gr + 1, gc + 1, 2))
    fits = [Fit(np.eye(3), mesh, np.zeros((1, 2)), 1.0, 0.0, False)]
    with pytest.raises(CC.CompositionScopeError):
        CC.validate_scope(S.Motion(W, H, 30.0, fits=fits, stride=2, n_frames=5))

    class _PlanLike:
        G = [np.eye(3)] * 3

    with pytest.raises(CC.CompositionScopeError):
        CC.validate_scope(S.Motion(W, H, 30.0, fits=fits, stride=1, n_frames=2), _PlanLike())


def test_reference_cells_fallback_when_mesh_disabled():
    class _Cfg:
        mesh = (0, 0)

    assert CC.reference_cells(_Cfg()) == CC.FALLBACK_CELLS

    class _Cfg2:
        mesh = (5, 4)

    assert CC.reference_cells(_Cfg2()) == (5, 4)


# --------------------------------------------------------------------------------------
# four-corner fold / non-finite detection (review repair)
# --------------------------------------------------------------------------------------

def test_fold_report_checks_all_four_corners():
    """The warpmetrics opposite-corner cell: a top-left-only Jacobian reports identity (+1) here."""
    field = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [-1., -1.]]], np.float64)
    rep = CC.fold_report(field, 2, 2)
    assert rep["cells"] == 1 and rep["valid_cells"] == 1
    assert rep["nonfinite_cells"] == 0 and rep["nonfinite_frac"] == 0.0
    assert rep["fold_frac"] == 1.0
    assert rep["collapse_frac"] == 0.0
    assert rep["det_min"] == pytest.approx(-3.0)          # bottom-right corner determinant
    assert rep["det_max"] == pytest.approx(1.0)           # top-left corner determinant only
    assert rep["det_abs_min"] == pytest.approx(1.0)

    # a constant-positive-Jacobian cell must stay clean
    ident = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [1., 1.]]], np.float64)
    clean = CC.fold_report(ident, 2, 2)
    assert clean["fold_frac"] == 0.0 and clean["collapse_frac"] == 0.0


def test_nonfinite_field_is_reported_and_strict_rejects():
    bad = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [np.nan, np.nan]]], np.float64)
    rep = CC.fold_report(bad, 2, 2)
    assert rep["cells"] == 1 and rep["valid_cells"] == 0
    assert rep["nonfinite_cells"] == 1 and rep["nonfinite_frac"] == 1.0
    assert rep["fold_frac"] is None and rep["det_min"] is None

    # candidate strict mode must reject a non-finite composed field, not report it clean
    W, H, gc, gr = 16, 12, 2, 1
    bad_mesh = np.full((gr + 1, gc + 1, 2), np.inf)
    fits = [Fit(np.eye(3), bad_mesh, np.zeros((1, 2)), 1.0, 0.0, False)]
    mot = Motion(W, H, 30.0, fits=fits, stride=1, n_frames=2)

    class _PlanLike:
        G = [np.eye(3), np.eye(3)]

    cfg = S.Config(mesh=(gc, gr), rs_rows=0)
    field = CC.candidate_field(mot, _PlanLike(), cfg, 1, highpass=False)
    assert not np.isfinite(field).all()
    with pytest.raises(CC.WarpFoldError):
        CC.candidate_field(mot, _PlanLike(), cfg, 1, highpass=False, strict=True)


def test_fold_report_suppresses_expected_nonfinite_warnings():
    """Infs subtract to NaN with a RuntimeWarning; fold_report ignores the warning but still reports the cell."""
    field = np.full((2, 2, 2), np.inf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = CC.fold_report(field, 2, 2)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert rep["nonfinite_cells"] == 1 and rep["valid_cells"] == 0
    assert rep["fold_frac"] is None and rep["nonfinite_frac"] == 1.0


def test_fold_report_exposes_overflow_and_strict_rejects_it():
    """Finite coordinates can still overflow the corner determinant; strict mode must reject that too."""
    # (a) analytic, warpmetrics-style: a finite 2x2 cell with ~1e200 differences has det = 1e400 = inf
    huge = np.array([[[0., 0.], [1e200, 0.]], [[0., 1e200], [1e200, 1e200]]], np.float64)
    rep = CC.fold_report(huge, 2, 2)
    assert np.isfinite(huge).all()                       # source coordinates are finite
    assert rep["cells"] == 1 and rep["valid_cells"] == 1
    assert rep["nonfinite_cells"] == 0 and rep["nonfinite_frac"] == 0.0
    assert rep["overflow_cells"] == 1 and rep["overflow_frac"] == 1.0
    assert rep["usable_cells"] == 0
    assert rep["fold_frac"] is None and rep["collapse_frac"] is None
    assert rep["det_min"] is None and rep["det_max"] is None and rep["det_abs_min"] is None
    for key in ("det_min", "det_max", "det_abs_min", "fold_frac", "collapse_frac", "nonfinite_frac", "overflow_frac"):
        assert rep[key] is None or np.isfinite(rep[key])   # JSON-safe summaries (no inf/NaN)

    # (b) a finite huge-scale candidate field: strict mode cannot accept det=inf as clean
    W, H = 8, 8
    Hk = np.diag([1e200, 1e200, 1.0])
    fits = [Fit(Hk.copy(), np.zeros((2, 2, 2)), np.zeros((1, 2)), 1.0, 0.0, False)]
    mot = Motion(W, H, 30.0, fits=fits, stride=1, n_frames=2)

    class _PlanLike:
        G = [np.eye(3), Hk.copy()]

    cfg = S.Config(mesh=(1, 1), rs_rows=0)
    field = CC.candidate_field(mot, _PlanLike(), cfg, 1, highpass=False)
    assert np.isfinite(field).all()                      # finite coordinates, finite differences
    crep = CC.fold_report(field, W, H)
    assert crep["nonfinite_cells"] == 0
    assert crep["overflow_cells"] == (H - 1) * (W - 1)
    assert crep["usable_cells"] == 0 and crep["fold_frac"] is None
    with pytest.raises(CC.WarpFoldError):
        CC.candidate_field(mot, _PlanLike(), cfg, 1, highpass=False, strict=True)


# --------------------------------------------------------------------------------------
# delta_grid / delta_reference consistency
# --------------------------------------------------------------------------------------

def test_delta_grid_agrees_with_delta_reference(fixtures):
    for name, fx in fixtures.items():
        pl, cfg, mot = _plan(fx)
        W, H = fx.W, fx.H
        cells = CC.reference_cells(cfg)
        ref_pts = CC.reference_points(W, H, cells)
        T = len(pl.G)
        grid = CC.delta_grid(list(mot.fits), ref_pts, W, H, T)
        for t in range(T):
            point = CC.delta_reference(list(mot.fits), t, ref_pts, W, H)
            np.testing.assert_allclose(grid[t], point, atol=1e-10, err_msg=f"{name} frame {t}")


# --------------------------------------------------------------------------------------
# high-pass channel on a >=6-frame nonconstant residual (default L1)
# --------------------------------------------------------------------------------------

def _long_motion(W=64, H=48, gc=4, gr=3, T=7, A=3.0):
    """Identity globals with an alternating (nonconstant) constant-in-space residual series."""
    fits = []
    for k in range(T - 1):
        m = np.zeros((gr + 1, gc + 1, 2))
        m[..., 1] = A * ((-1.0) ** k)
        fits.append(Fit(np.eye(3), m, np.zeros((1, 2)), 1.0, 0.0, False))
    return Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T)


def test_highpass_channel_is_active_for_long_nonconstant_residual():
    """T=7 (L1 solver active) with an oscillating residual: the correction must be nonzero and non-identity."""
    mot = _long_motion()
    W, H = int(mot.W), int(mot.H)
    cfg = S.Config(mesh=(4, 3), rs_rows=0)                 # default smooth="l1"
    with CA.pin_first_sample():
        pl = S.plan(mot, cfg, n_frames=mot.n_frames)
    pts = CC.output_grid(W, H)
    base = CC.global_component(pl, 5, pts)
    raw = CC.candidate_source_coords(mot, pl, cfg, 5, pts, highpass=False)
    hp = CC.candidate_source_coords(mot, pl, cfg, 5, pts, highpass=True)
    corr = hp - base
    assert np.abs(corr).max() > 1e-6                       # high-pass is active
    assert np.abs(hp - raw).max() > 1e-6                   # and differs from the raw composition
    assert np.abs(corr).max() <= cfg.mesh_budget * W + 1e-9

    sm = CC.smoothed_delta(mot, pl, cfg)                   # precompute-once helper
    hp2 = CC.candidate_source_coords(mot, pl, cfg, 5, pts, highpass=True, smoothed=sm)
    np.testing.assert_allclose(hp2, hp, atol=0.0)


def test_candidate_rejects_mismatched_smoothed_grid(fixtures):
    fx = fixtures["constant_residual_rotation"]
    pl, cfg, mot = _plan(fx)
    pts = CC.output_grid(fx.W, fx.H)
    bad = np.zeros((len(pl.G), 3, 2))                      # wrong Nref
    with pytest.raises(ValueError):
        CC.candidate_source_coords(mot, pl, cfg, 2, pts, highpass=True, smoothed=bad)


# --------------------------------------------------------------------------------------
# temporal-smoothing chart decision experiment (source vs reference)
# --------------------------------------------------------------------------------------

def _rotating_motion(W=64, H=48, gc=4, gr=3, T=7, deg=10.0):
    """Identical rotating globals with a nonconstant constant-in-space residual series."""
    R = CA.rotation_about_clockwise(W, H, deg)
    fits, vs = [], []
    for k in range(T - 1):
        v = np.array([2.0 * (k + 1), 1.0 * ((-1.0) ** k)])
        m = np.zeros((gr + 1, gc + 1, 2)); m[..., 0] = v[0]; m[..., 1] = v[1]
        fits.append(Fit(R.copy(), m, np.zeros((1, 2)), 1.0, 0.0, False))
        vs.append(v)
    return Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T), vs, R


def _analytic_delta_eta(vs, Rlin, t):
    """Independent closed forms for the two charts.

    With ``H_k = h`` (affine rotation, linear part ``Rlin``) and constant-in-space residuals ``v_k``:
    ``Phi_t = h^t + sum_k Rlin^{t-1-k} v_k`` and ``Psi_t = h^t``, so
      delta_t = sum_{k<t} Rlin^{t-1-k} v_k
      eta_t   = inv(Psi_t)(Phi_t) - z = Rlin^{-t} delta_t = sum_{k<t} Rlin^{-(k+1)} v_k.
    """
    delta = np.zeros(2); eta = np.zeros(2)
    for k in range(t):
        delta = delta + np.linalg.matrix_power(Rlin, t - 1 - k) @ vs[k]
        eta = eta + np.linalg.matrix_power(Rlin, -(k + 1)) @ vs[k]
    return delta, eta


def test_smoothing_basis_decision_experiment_source_vs_reference():
    """Both charts are implemented; neither is selected. Source keeps its old result; reference is exact."""
    mot, vs, R = _rotating_motion()
    W, H, T = int(mot.W), int(mot.H), int(mot.n_frames)
    cfg = S.Config(mesh=(4, 3), rs_rows=0)                 # default smooth="l1", T=7 -> solver active
    with CA.pin_first_sample():
        pl = S.plan(mot, cfg, n_frames=T)
    pts = CC.output_grid(W, H)
    fits = list(mot.fits)
    t = 5
    Rlin = R[:2, :2]

    # 1) both charts match their independent closed forms at every frame
    for tt in range(T):
        d, e = _analytic_delta_eta(vs, Rlin, tt)
        np.testing.assert_allclose(CC.delta_reference(fits, tt, pts, W, H),
                                   np.broadcast_to(d, pts.shape), atol=1e-9)
        np.testing.assert_allclose(CC.eta_reference(fits, tt, pts, W, H),
                                   np.broadcast_to(e, pts.shape), atol=1e-9)

    # 2) highpass=False stays exact Phi_t(V_t(x)) in both charts
    raw_s = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=False, smoothing_basis="source")
    raw_r = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=False, smoothing_basis="reference")
    np.testing.assert_allclose(raw_s, raw_r, atol=0.0)
    np.testing.assert_allclose(raw_s, CC.compose_reference(fits, t, pts, W, H), atol=1e-8)

    # 3) default is source and preserves the previous behavior
    hp_default = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=True)
    hp_src = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=True, smoothing_basis="source")
    np.testing.assert_allclose(hp_default, hp_src, atol=0.0)

    base = CC.global_component(pl, t, pts)
    ref_pts = CC.reference_points(W, H, (4, 3))
    d_t, e_t = _analytic_delta_eta(vs, Rlin, t)
    d_grid = np.stack([np.broadcast_to(_analytic_delta_eta(vs, Rlin, tt)[0], (len(ref_pts), 2))
                       for tt in range(T)])
    e_grid = np.stack([np.broadcast_to(_analytic_delta_eta(vs, Rlin, tt)[1], (len(ref_pts), 2))
                       for tt in range(T)])

    # 4) source chart: delta - smooth(delta) from the closed form and its independent smoothing
    sm_src = CC.smooth_delta_grid(d_grid, cfg, W)
    np.testing.assert_allclose(CC.smoothed_delta(mot, pl, cfg, smoothing_basis="source"), sm_src, atol=1e-12)
    corr_src = d_t - sm_src[t][0]
    np.testing.assert_allclose(hp_src - base, np.broadcast_to(corr_src, pts.shape), atol=1e-8)

    # 5) reference chart: exact projective push-forward of the high-pass eta (no linear block)
    sm_ref = CC.smooth_delta_grid(e_grid, cfg, W)
    np.testing.assert_allclose(CC.smoothed_delta(mot, pl, cfg, smoothing_basis="reference"), sm_ref, atol=1e-12)
    eta_hp = e_t - sm_ref[t][0]
    psi = CC.global_homography(fits, t)
    z = G.apply_h(CC.virtual_map(fits, pl, t), pts)
    expected_ref = G.apply_h(psi, z + eta_hp) - G.apply_h(psi, z)
    hp_ref = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=True, smoothing_basis="reference")
    np.testing.assert_allclose(hp_ref - base, expected_ref, atol=1e-8)

    # 6) the two charts disagree on this rotating fixture (decision experiment, no selected winner)
    diff = np.linalg.norm((hp_src - base) - (hp_ref - base), axis=1)
    assert diff.max() > 1e-6


def test_invalid_smoothing_basis_rejected():
    mot, _vs, _R = _rotating_motion()
    cfg = S.Config(mesh=(4, 3), rs_rows=0)
    with CA.pin_first_sample():
        pl = S.plan(mot, cfg, n_frames=mot.n_frames)
    pts = CC.output_grid(int(mot.W), int(mot.H))
    with pytest.raises(ValueError):
        CC.candidate_source_coords(mot, pl, cfg, 1, pts, smoothing_basis="bogus")
    with pytest.raises(ValueError):
        CC.smoothed_delta(mot, pl, cfg, smoothing_basis="bogus")
    with pytest.raises(ValueError):
        CC.candidate_field(mot, pl, cfg, 1, smoothing_basis="bogus")


# --------------------------------------------------------------------------------------
# projective (non-affine) globals: exact push-forward, not a Jacobian linearisation
# --------------------------------------------------------------------------------------

def _projective_motion(W=64, H=48, gc=4, gr=3, nr=6, T=7):
    """Well-conditioned *non-affine* projective globals with nonzero, time-varying mesh and row residuals."""
    pairs, fits = [], []
    for k in range(T - 1):
        Hk = (CA.rotation_about_clockwise(W, H, 5.0 * (k + 1)) @ CA.translation(2.0 * (k + 1), -1.5 * k)).copy()
        Hk[2, 0] = 2.0e-3 * (1.0 + 0.2 * k)          # perspective terms -> genuinely projective (non-affine)
        Hk[2, 1] = -1.6e-3 * (1.0 + 0.2 * k)
        tf = 1.0 + 0.9 * ((-1.0) ** k)               # strongly nonconstant over k
        mesh = CA.field_mesh(W, H, gc, gr, lambda p, tf=tf: np.stack(
            [tf * 2.5 * np.sin(p[:, 1] / (0.35 * H)), tf * 1.8 * np.cos(p[:, 0] / (0.30 * W))], 1))
        rows = CA.row_field(W, H, nr, lambda y, tf=tf: np.stack(
            [np.zeros_like(y), tf * 1.6 * np.sin(y / (0.28 * H))], 1))
        pairs.append(CA.PairMotion(Hk.copy(), mesh, rows))
        fits.append(Fit(Hk.copy(), mesh, rows, 1.0, 0.0, False))
    return pairs, fits


def _fd_jacobian(Hm, z, eps=1e-3):
    """Finite-difference Jacobian columns of ``apply_h(Hm, .)`` at ``z`` (for the first-order comparison)."""
    ex = np.array([eps, 0.0]); ey = np.array([0.0, eps])
    jx = (G.apply_h(Hm, z + ex) - G.apply_h(Hm, z - ex)) / (2 * eps)
    jy = (G.apply_h(Hm, z + ey) - G.apply_h(Hm, z - ey)) / (2 * eps)
    return jx, jy


def test_reference_chart_is_exact_for_projective_global():
    """eta/eta_grid match the CA oracle, highpass=False is exact in both charts, and the reference high-pass is
    pushed forward by the full projective Psi_t -- a first-order Jacobian block measurably differs."""
    W, H, gc, gr, nr, T = 64, 48, 4, 3, 6, 7
    pairs, fits = _projective_motion(W, H, gc, gr, nr, T)
    mot = Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T)
    cfg = S.Config(mesh=(gc, gr), rs_rows=nr, mesh_budget=0.20)    # T=7 -> L1 active, large correction cap
    with CA.pin_first_sample():
        pl = S.plan(mot, cfg, n_frames=T)
    pts = CC.output_grid(W, H)
    cells = (gc, gr)
    ref_pts = CC.reference_points(W, H, cells)
    t = 5

    # well-conditioned, genuinely non-affine globals
    psi = CA.compose_globals(pairs, t)
    w = psi[2, 0] * pts[:, 0] + psi[2, 1] * pts[:, 1] + psi[2, 2]
    assert np.abs(w).min() > 0.25
    assert np.abs(psi[2, :2]).max() > 1e-4

    # eta_grid == eta_reference == independent CA oracle, every frame
    eta_c = CC.eta_grid(fits, ref_pts, W, H, T)
    for tt in range(T):
        np.testing.assert_allclose(eta_c[tt], CC.eta_reference(fits, tt, ref_pts, W, H), atol=1e-9)
        phi = CA.compose_oracle(pairs, tt, ref_pts, W, H)
        eta_o = G.apply_h(np.linalg.inv(CA.compose_globals(pairs, tt)), phi) - ref_pts
        np.testing.assert_allclose(eta_c[tt], eta_o, atol=1e-9)

    # highpass=False returns the exact composed map Phi_t(V_t(x)) in both charts
    V = CC.virtual_map(fits, pl, t)
    z = G.apply_h(V, pts)
    base = CC.global_component(pl, t, pts)
    exact = CA.compose_oracle(pairs, t, z, W, H)
    for basis in ("source", "reference"):
        raw = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=False, smoothing_basis=basis)
        np.testing.assert_allclose(raw, exact, atol=1e-8)

    # highpass=True reference: correction == exact projective push-forward Psi(z+eta_hp) - Psi(z)
    eta_grid_o = np.stack([G.apply_h(np.linalg.inv(CA.compose_globals(pairs, tt)),
                                     CA.compose_oracle(pairs, tt, ref_pts, W, H)) - ref_pts
                           for tt in range(T)])
    sm_o = CC.smooth_delta_grid(eta_grid_o, cfg, W)
    np.testing.assert_allclose(CC.smoothed_delta(mot, pl, cfg, smoothing_basis="reference"), sm_o, atol=1e-6)
    eta_hp = CC.eta_reference(fits, t, z, W, H) - CC.sample_grid_bilinear(sm_o[t], z, W, H, cells)
    assert np.linalg.norm(eta_hp, axis=1).max() > 0.5
    expected = G.apply_h(psi, z + eta_hp) - G.apply_h(psi, z)
    hp_ref = CC.candidate_source_coords(mot, pl, cfg, t, pts, highpass=True, smoothing_basis="reference")
    corr = hp_ref - base
    np.testing.assert_allclose(corr, expected, atol=1e-6)

    # a first-order Jacobian block is not equivalent: the actual correction and a fixed probe both differ
    jx, jy = _fd_jacobian(psi, z)
    approx_actual = jx * eta_hp[:, 0:1] + jy * eta_hp[:, 1:2]
    assert np.linalg.norm(expected - approx_actual, axis=1).max() > 1e-3
    probe = np.broadcast_to(np.array([4.0, 3.0]), z.shape)
    exact_probe = G.apply_h(psi, z + probe) - G.apply_h(psi, z)
    approx_probe = jx * probe[:, 0:1] + jy * probe[:, 1:2]
    assert np.linalg.norm(exact_probe - approx_probe, axis=1).max() > 1e-3
