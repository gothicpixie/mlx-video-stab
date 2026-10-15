"""Focused tests for the forward-backward cycle-consistency diagnostic.

``observe`` keeps the per-sample cycle diagnostics it uses to decide ``valid`` -- ``fb_err = ||f + sampled_b||``,
the tolerance ``fb_abs + fb_rel*||f||`` and whether the forward endpoint is in-bounds -- for *every* sampled
point, including points the tolerance rejects. ``fit_motion`` aggregates them into Gaussian spatial maps on the
mesh grid (``cycle_err_rms`` px, ``cycle_inconsistency_frac`` in [0,1], ``cycle_sample_mass``) built from every
finite, in-bounds forward endpoint, independent of the fit's ``valid``/non-mover mask.

They are a forward-backward **cycle-consistency** diagnostic only: not a true correspondence error, not a
probability and not calibrated uncertainty, and they never feed stabilization or fitting. This file covers the
per-sample fields, the analytic spatial aggregation (including a rejected-but-in-bounds sample and excluded
out-of-bounds / non-finite samples), the fallback path preserving diagnostics, and ``None`` maps for
hand-built observations and legacy Fits. No media, FFmpeg or RAFT.
"""
import numpy as np
import pytest

import mlx.core as mx

from mlxstab import motion as MO


def _obs(pts, flow, *, err, tol, inside, valid=None, W=64, H=48):
    pts = np.asarray(pts, np.float64); flow = np.asarray(flow, np.float64)
    if valid is None:
        valid = np.asarray(err, np.float64) < np.asarray(tol, np.float64)
        valid = valid & np.asarray(inside, bool)
    return MO.Obs(pts, flow, np.asarray(valid, bool), W, H,
                  fb_err=np.asarray(err, np.float64), fb_tol=np.asarray(tol, np.float64),
                  fb_inside=np.asarray(inside, bool))


def _expected_cycle(pts, flow, err, tol, inside, W, H, gc, gr):
    """Independent recomputation of the Gaussian spatial cycle maps ``_cycle_maps`` produces."""
    pts = np.asarray(pts, np.float64); flow = np.asarray(flow, np.float64)
    err = np.asarray(err, np.float64); tol = np.asarray(tol, np.float64)
    sel = np.isfinite(pts + flow).all(1) & np.asarray(inside, bool) & np.isfinite(err) & np.isfinite(tol)
    vx = -.5 + np.arange(gc + 1) * W / gc; vy = -.5 + np.arange(gr + 1) * H / gr
    cx, cy = np.meshgrid(vx, vy)
    centers = np.stack([cx.ravel(), cy.ravel()], 1)
    sig = np.array([W / gc, H / gr])
    rms = np.zeros((gr + 1, gc + 1)); frac = np.zeros((gr + 1, gc + 1)); mass = np.zeros((gr + 1, gc + 1))
    idx = np.flatnonzero(sel)
    if idx.size:
        d2 = (((pts[idx][:, None, :] - centers[None]) / sig) ** 2).sum(-1)
        K = np.exp(-0.5 * d2) * (d2 < 9.0)
        w = K.sum(0)
        e = err[idx]
        num = (K * (e ** 2)[:, None]).sum(0)
        bad = (K * (e >= tol[idx]).astype(float)[:, None]).sum(0)
        nz = w > 0
        rms.ravel()[nz] = np.sqrt(num[nz] / w[nz])
        frac.ravel()[nz] = bad[nz] / w[nz]
        mass.ravel()[nz] = w[nz]
    return rms, frac, mass


# --------------------------------------------------------------------------------------
# observe(): per-sample diagnostics are retained, including rejected samples
# --------------------------------------------------------------------------------------

def test_observe_stores_per_sample_cycle_diagnostics():
    W = H = 32
    F = mx.zeros((H, W, 2))
    F = F.at[:, :, 0].add(2.0)                      # constant forward flow (2 px right)
    B = mx.zeros((H, W, 2))
    B = B.at[:, :, 0].add(-1.0)                     # backward flow does not cancel -> cycle error 1 px
    obs = MO.observe(F, B, step=4, fb_abs=1.0, fb_rel=0.05)
    assert obs.fb_err.shape == obs.pts.shape[:1] and obs.fb_tol.shape == obs.fb_err.shape
    assert obs.fb_inside.shape == obs.fb_err.shape and obs.fb_inside.dtype == bool
    # fb_err = ||f + sampled_b|| = |2 + (-1)| = 1 px and tol = 1.0 + 0.05*2 = 1.1 px at interior samples.
    np.testing.assert_allclose(obs.fb_err, 1.0, atol=1e-6)
    np.testing.assert_allclose(obs.fb_tol, 1.1, atol=1e-6)
    # valid is exactly the stored threshold predicate AND-ed with in-bounds; no diagnostic is dropped.
    np.testing.assert_array_equal(obs.valid, (obs.fb_err < obs.fb_tol) & obs.fb_inside)
    assert obs.valid.any() and not obs.valid.all()          # the right-edge endpoints leave the frame


def test_observe_keeps_rejected_but_in_bounds_samples():
    W = H = 32
    F = mx.zeros((H, W, 2)); F = F.at[:, :, 0].add(0.0)
    B = mx.zeros((H, W, 2)); B = B.at[:, :, 0].add(3.0)      # cycle error 3 px, well past the 1 px tolerance
    obs = MO.observe(F, B, step=8, fb_abs=1.0, fb_rel=0.05)
    assert (~obs.valid).all()                                 # every sample rejected by the tolerance
    assert obs.fb_inside.all()                               # ... yet all are in-bounds
    # The rejected samples are still present with their diagnostics for the spatial map.
    assert np.all(np.isfinite(obs.fb_err)) and obs.fb_err.size == obs.pts.shape[0]
    assert np.all(obs.fb_err > obs.fb_tol)


def test_observe_marks_out_of_bounds_forward_endpoints():
    W = H = 16
    F = mx.zeros((H, W, 2)); F = F.at[:, :, 0].add(100.0)    # shove every endpoint far outside the frame
    B = mx.zeros((H, W, 2))
    obs = MO.observe(F, B, step=4, fb_abs=1.0, fb_rel=0.05)
    assert not obs.fb_inside.any()
    assert not obs.valid.any()


def test_observe_does_not_change_valid_decision_for_manually_built_obs():
    """Adding the diagnostic fields must not perturb the derived ``valid`` boolean for existing callers."""
    F = mx.zeros((8, 8, 2)); B = mx.zeros((8, 8, 2))
    obs = MO.observe(F, B, step=4)
    np.testing.assert_array_equal(obs.valid, (obs.fb_err < obs.fb_tol) & obs.fb_inside)
    assert obs.valid.all()


# --------------------------------------------------------------------------------------
# _cycle_maps: analytic spatial aggregation
# --------------------------------------------------------------------------------------

def test_cycle_maps_analytic_constant_error_includes_rejected_samples():
    gc, gr = 4, 3
    pts = np.array([[4.0, 4.0], [20.0, 4.0], [4.0, 20.0], [20.0, 20.0], [32.0, 24.0]], np.float64)
    flow = np.zeros_like(pts)
    # Sample 1 is in-bounds but fails the tolerance; it must still be counted (constant 2 px error).
    err = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    tol = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    inside = np.array([True, True, True, True, True])
    obs = _obs(pts, flow, err=err, tol=tol, inside=inside)
    assert not obs.valid[1]                                   # rejected by tolerance, still in-bounds
    rms, frac, mass = MO._cycle_maps(obs, gc, gr)
    e_rms, e_frac, e_mass = _expected_cycle(pts, flow, err, tol, inside, obs.W, obs.H, gc, gr)
    np.testing.assert_allclose(rms, e_rms, rtol=1e-12)
    np.testing.assert_allclose(frac, e_frac, rtol=1e-12)
    np.testing.assert_allclose(mass, e_mass, rtol=1e-12)
    # A constant error means the weighted RMS is exactly that error and every sampled vertex is fully failing.
    nz = mass > 0
    np.testing.assert_allclose(rms[nz], 2.0, rtol=1e-12)
    np.testing.assert_allclose(frac[nz], 1.0, rtol=1e-12)
    # Conditioning only on Obs.valid would have dropped the rejected sample and changed the map.
    keep = [0, 2, 3, 4]
    subset = _obs(pts[keep], flow[keep], err=err[keep], tol=tol[keep], inside=inside[keep])
    _, _, mass_subset = MO._cycle_maps(subset, gc, gr)
    assert not np.allclose(mass, mass_subset)


def test_cycle_maps_exclude_out_of_bounds_and_nonfinite_samples():
    gc, gr = 4, 3
    pts = np.array([[8.0, 8.0], [24.0, 8.0], [8.0, 24.0], [24.0, 24.0]], np.float64)
    # Sample 1 endpoint is outside the frame; sample 2's forward endpoint is non-finite; sample 3 has a
    # non-finite cycle error. All three must be excluded, leaving only sample 0.
    flow = np.array([[0.0, 0.0], [200.0, 200.0], [np.inf, 0.0], [0.0, 0.0]], np.float64)
    err = np.array([0.5, 0.5, 0.5, np.inf])
    tol = np.array([1.0, 1.0, 1.0, 1.0])
    inside = np.array([True, False, True, True])
    obs = _obs(pts, flow, err=err, tol=tol, inside=inside, valid=[True, False, False, False])
    rms, frac, mass = MO._cycle_maps(obs, gc, gr)
    e_rms, e_frac, e_mass = _expected_cycle(pts, flow, err, tol, inside, obs.W, obs.H, gc, gr)
    np.testing.assert_allclose(rms, e_rms, rtol=1e-12)
    np.testing.assert_allclose(frac, e_frac, rtol=1e-12)
    np.testing.assert_allclose(mass, e_mass, rtol=1e-12)
    # Only the one finite, in-bounds sample contributes, so the mass is a single sample's weight sum.
    assert mass.sum() > 0.0


def test_cycle_maps_zero_denominator_vertices_are_zero():
    gc, gr = 4, 3
    obs = _obs([[2.0, 2.0]], [[0.0, 0.0]], err=[0.5], tol=[1.0], inside=[True])
    rms, frac, mass = MO._cycle_maps(obs, gc, gr)
    assert mass.shape == (gr + 1, gc + 1)
    zero = mass == 0.0
    assert zero.any()                                        # samples cannot reach every vertex
    assert np.all(rms[zero] == 0.0) and np.all(frac[zero] == 0.0)


def test_cycle_maps_none_for_manual_obs_without_diagnostics():
    obs = MO.Obs(np.zeros((3, 2)), np.zeros((3, 2)), np.ones(3, bool), 64, 48)
    assert MO._cycle_maps(obs, 4, 3) == (None, None, None)


# --------------------------------------------------------------------------------------
# fit_motion: maps on the fit, including the fallback path
# --------------------------------------------------------------------------------------

def _observed_obs(W=64, H=48, step=4, err=0.25, tol=1.0, inside=True, valid=None):
    ys = np.arange(step // 2, H, step); xs = np.arange(step // 2, W, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    flow = np.zeros_like(pts)
    n = len(pts)
    return _obs(pts, flow, err=np.full(n, float(err)), tol=np.full(n, float(tol)),
                inside=np.full(n, bool(inside)), valid=valid)


def test_fit_motion_attaches_cycle_maps_from_observed_samples():
    gc, gr = 4, 3
    obs = _observed_obs()
    fit = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=2)
    assert fit.fallback is False
    rms, frac, mass = MO._cycle_maps(obs, gc, gr)
    np.testing.assert_allclose(fit.cycle_err_rms, rms, rtol=1e-12)
    np.testing.assert_allclose(fit.cycle_inconsistency_frac, frac, rtol=1e-12)
    np.testing.assert_allclose(fit.cycle_sample_mass, mass, rtol=1e-12)
    assert fit.cycle_err_rms.shape == (gr + 1, gc + 1)
    np.testing.assert_allclose(fit.cycle_err_rms, 0.25, rtol=1e-12)   # constant error, every sampled vertex


def test_fit_motion_fallback_still_carries_cycle_maps():
    """A geometric fallback (too few valid samples) must not discard the raw cycle diagnostics."""
    gc, gr = 4, 3
    obs = _observed_obs(valid=np.zeros(len(_observed_obs().pts), bool))   # zero valid -> fallback
    fit = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=2, min_valid=60)
    assert fit.fallback is True
    assert fit.cycle_err_rms is not None and fit.cycle_inconsistency_frac is not None
    assert fit.cycle_sample_mass is not None and fit.cycle_sample_mass.sum() > 0.0
    rms, frac, mass = MO._cycle_maps(obs, gc, gr)
    np.testing.assert_allclose(fit.cycle_err_rms, rms, rtol=1e-12)
    np.testing.assert_allclose(fit.cycle_inconsistency_frac, frac, rtol=1e-12)
    np.testing.assert_allclose(fit.cycle_sample_mass, mass, rtol=1e-12)


def test_fit_motion_manual_obs_has_none_cycle_maps():
    gc, gr = 4, 3
    ys = np.arange(2, 48, 4); xs = np.arange(2, 64, 4)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    obs = MO.Obs(pts, np.zeros_like(pts), np.ones(len(pts), bool), 64, 48)   # no cycle diagnostics
    fit = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=2)
    assert fit.cycle_err_rms is None and fit.cycle_inconsistency_frac is None and fit.cycle_sample_mass is None


# --------------------------------------------------------------------------------------
# Fit / Motion defaults and serialization availability
# --------------------------------------------------------------------------------------

def test_manual_fit_defaults_cycle_maps_to_none():
    fit = MO.Fit(np.eye(3), np.zeros((3, 5, 2)), np.zeros((2, 2)), 0.9, 0.0, False)
    assert fit.cycle_err_rms is None and fit.cycle_inconsistency_frac is None and fit.cycle_sample_mass is None


def test_cycle_maps_do_not_change_h_mesh_rows():
    """The diagnostic maps must not perturb the fitted H/mesh/rows an existing caller sees."""
    gc, gr = 4, 3
    obs = _observed_obs()
    fit = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=2)
    np.testing.assert_array_equal(fit.H, np.eye(3))
    np.testing.assert_allclose(fit.mesh, 0.0, atol=1e-12)
    np.testing.assert_allclose(fit.rows, 0.0, atol=1e-12)


def test_cycle_map_float32_endpoint_is_excluded_when_nonfinite():
    """A non-finite forward endpoint is excluded even when ``fb_inside``/``fb_err`` are otherwise finite."""
    obs = _obs([[8.0, 8.0]], [[np.inf, 0.0]], err=[0.1], tol=[1.0], inside=[True])
    rms, frac, mass = MO._cycle_maps(obs, 2, 2)
    assert rms.sum() == 0.0 and frac.sum() == 0.0 and mass.sum() == 0.0
