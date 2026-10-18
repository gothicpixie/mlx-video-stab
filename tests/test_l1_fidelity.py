"""Focused tests for the opt-in low-frequency path-fidelity term in the L1 planner.

The default L1 path minimizes derivative sparsity and can erase a sustained pan. The opt-in term adds a
per-sample soft pull towards a separately low-passed raw path. These tests pin the contract:

* the default and historical scalar-fidelity behaviour are unchanged;
* a sustained pan is retained while high-frequency jitter stays suppressed;
* the low-pass target is derived independently per shot;
* per-sample targets/weights work through the windowed solver; and
* the opt-in Config control is validated and disabled by default.
"""
import numpy as np
import pytest

from bench.path_smoother_gain_audit import exact_motion, _path_metrics
from mlxstab import smooth as SM, stabilize as S
from mlxstab import __main__ as cli
from mlxstab.motion import Fit


def _motion_from_dx(width, height, dx, cut=None):
    """Exact translation Fits (camera +x makes image content move -x)."""
    fits = []
    for value in np.asarray(dx, np.float64):
        H = np.eye(3)
        H[0, 2] = -value
        fits.append(Fit(H=H, mesh=np.zeros((1, 1, 2)), rows=np.zeros((0, 2)),
                        inlier_frac=1.0, mover_frac=0.0, fallback=False))
    cuts = () if cut is None else (cut,)
    return S.Motion(width, height, 30.0, fits=fits, n_frames=len(dx) + 1, cuts=cuts)


def _linear_plan(width, height, frames, pan_px_per_frame, jitter_px=0.0, seed=17,
                 low_freq_fidelity=0.0, low_freq_sigma=15.0):
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter1d
    noise = gaussian_filter1d(rng.standard_normal(frames), 0.8)
    noise = noise / (noise.std() + 1e-12) * jitter_px
    t = np.arange(frames, dtype=np.float64)
    intended = pan_px_per_frame * t
    actual = intended + noise
    cfg = S.Config(mesh=(0, 0), rs_rows=0, smooth="l1", budget=0.04, fidelity=0.1,
                   l1_low_freq_fidelity=low_freq_fidelity, l1_low_freq_sigma=low_freq_sigma)
    plan = exact_motion(width, height, 30.0, intended, actual, cfg)
    metrics = _path_metrics(plan, intended, actual, 30.0, "linear", 20.0, 120.0)
    return plan, metrics


def test_default_and_scalar_fidelity_are_unchanged():
    rng = np.random.default_rng(3)
    c = np.cumsum(rng.normal(0.0, 0.4, 64))
    budget = 2.5
    legacy = SM.l1_path(c, budget, fidelity=0.1)
    explicit = SM.l1_path(c, budget, fidelity=0.1,
                          fidelity_target=c, fidelity_weight=np.full(len(c), 0.1))
    np.testing.assert_allclose(legacy, explicit, atol=1e-12, rtol=0)

    # Zero per-sample weight disables the term even when a target is supplied.
    off = SM.l1_path(c, budget, fidelity=0.0,
                     fidelity_target=np.full(len(c), 1e3), fidelity_weight=np.zeros(len(c)))
    np.testing.assert_allclose(off, SM.l1_path(c, budget, fidelity=0.0), atol=1e-12, rtol=0)

    # The new production control defaults to disabled and an unused sigma cannot change the plan.
    assert S.Config().l1_low_freq_fidelity == 0.0
    base, _ = _linear_plan(320, 180, 40, 0.5)
    inert, _ = _linear_plan(320, 180, 40, 0.5, low_freq_fidelity=0.0, low_freq_sigma=3.0)
    np.testing.assert_array_equal(base.path_quads, inert.path_quads)
    np.testing.assert_array_equal(base.G, inert.G)


def test_sustained_pan_is_retained_versus_current_l1_behaviour():
    baseline, before = _linear_plan(1920, 1080, 120, 0.5)
    retained, after = _linear_plan(1920, 1080, 120, 0.5, low_freq_fidelity=5.0, low_freq_sigma=12.0)

    # Current behaviour: the exact linear pan is removed.
    assert abs(before["path_gain"]) < 1e-6
    assert abs(before["output_endpoint_x_px"]) < 1e-6
    # Opt-in: the sustained pan is retained, still under the hard camera-deviation budget. The Gaussian
    # target has a small boundary lag, so the endpoint is close to, but not exactly at, the intended 59.5 px.
    assert after["path_gain"] > 0.95
    assert abs(after["output_endpoint_x_px"] - 59.5) < 6.0
    assert abs(after["output_endpoint_x_px"]) > abs(before["output_endpoint_x_px"]) + 50.0

    base = np.asarray(retained.path_quads[0, 0], np.float64)
    raw = np.asarray(retained.raw_quads[:, 0], np.float64)
    out = np.asarray(retained.path_quads[:, 0], np.float64) - base[0]
    raw_rel = raw - raw[0, 0]
    # Camera deviation budget (fraction of width) is still enforced sample by sample.
    assert np.max(np.abs(out - raw_rel)) <= 0.04 * 1920 + 1e-6


def test_opt_in_term_keeps_high_frequency_jitter_suppressed():
    baseline, before = _linear_plan(1920, 1080, 120, 0.5, jitter_px=2.0, seed=17)
    retained, after = _linear_plan(1920, 1080, 120, 0.5, jitter_px=2.0, seed=17,
                                   low_freq_fidelity=3.0, low_freq_sigma=15.0)

    assert abs(before["path_gain"]) < 1e-6
    assert after["path_gain"] > 0.95
    assert after["output_path_rmse_px"] < 0.25 * before["output_path_rmse_px"]
    # Relative to the raw path, the enabled term still attenuates high-frequency velocity error by >10x.
    assert after["highpass_velocity_error_rms_px_per_frame"] < 0.1 * after[
        "raw_highpass_velocity_error_rms_px_per_frame"]
    # It must not reintroduce the raw disturbance: it stays far below the raw jitter level.
    assert after["highpass_velocity_error_rms_px_per_frame"] < 0.1


def test_low_frequency_target_is_derived_independently_per_shot():
    width, height, frames, cut = 640, 360, 60, 30
    t = np.arange(frames - 1, dtype=np.float64)
    noise = np.where(np.arange(frames - 1) % 2, -0.7, 0.7)
    shot1 = 0.5 + noise
    shot2 = 2.0 + noise
    dx_a = np.concatenate([shot1[:cut - 1], shot2[cut - 1:]])
    dx_b = dx_a.copy()
    dx_b[cut - 1:] += 0.75                       # alter only the second shot

    cfg = S.Config(mesh=(0, 0), rs_rows=0, budget=0.04, fidelity=0.1,
                   l1_low_freq_fidelity=3.0, l1_low_freq_sigma=8.0)
    a = S.plan(_motion_from_dx(width, height, dx_a, cut=cut), cfg, n_frames=frames)
    b = S.plan(_motion_from_dx(width, height, dx_b, cut=cut), cfg, n_frames=frames)

    np.testing.assert_allclose(a.path_quads[:cut], b.path_quads[:cut], atol=1e-12, rtol=0)
    assert np.abs(np.asarray(a.path_quads[cut:]) - np.asarray(b.path_quads[cut:])).max() > 1.0
    # The raw path (and therefore the shot reference) still restarts exactly at the cut.
    np.testing.assert_allclose(a.raw_quads[cut], S.G.corners(width, height), atol=1e-12, rtol=0)


def test_windowed_solver_accepts_per_sample_target_and_weight():
    frames = 40
    c = np.zeros(frames)
    target = np.linspace(0.0, 20.0, frames)
    budget = 5.0
    plain = SM.l1_path_windowed(c, budget, window=16, overlap=4, fidelity=0.1)
    pulled = SM.l1_path_windowed(c, budget, window=16, overlap=4, fidelity=0.1,
                                 fidelity_target=target, fidelity_weight=np.full(frames, 5.0))

    assert np.abs(plain).max() < 1e-9                   # pure L1: constant path
    assert pulled[-1] > 0.9 * budget                    # target is followed up to the hard budget
    assert np.max(np.abs(pulled - c)) <= budget + 1e-9  # budget/pinning semantics intact
    assert np.all(np.diff(pulled) >= -1e-9)             # overlapping windows chain monotonically

    # Historical scalar-fidelity call is exactly the explicit full-length target/weight call.
    legacy = SM.l1_path_windowed(c, budget, window=16, overlap=4, fidelity=0.1)
    explicit = SM.l1_path_windowed(c, budget, window=16, overlap=4, fidelity=0.1,
                                   fidelity_target=c, fidelity_weight=np.full(frames, 0.1))
    np.testing.assert_allclose(legacy, explicit, atol=1e-12, rtol=0)


def test_per_sample_zero_weight_makes_target_inert_at_those_samples():
    frames = 50
    c = np.zeros(frames)
    weight = np.zeros(frames)
    weight[20:30] = 5.0
    base_target = np.zeros(frames)
    inside = base_target.copy()
    inside[20:30] = np.linspace(1.0, 10.0, 10)
    outside = base_target.copy()
    outside[40:] = 100.0

    a = SM.l1_path(c, 100.0, fidelity=0.0, fidelity_target=base_target, fidelity_weight=weight)
    b = SM.l1_path(c, 100.0, fidelity=0.0, fidelity_target=inside, fidelity_weight=weight)
    d = SM.l1_path(c, 100.0, fidelity=0.0, fidelity_target=outside, fidelity_weight=weight)

    # Target values at zero-weight samples do not enter the objective/feasible region for p: inert.
    np.testing.assert_allclose(a, d, atol=1e-12, rtol=0)
    # The same target change at a positively weighted sample does move the solution.
    assert np.abs(a[20:30] - b[20:30]).max() > 0.5


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"l1_low_freq_fidelity": -0.1}, "l1_low_freq_fidelity"),
    ({"l1_low_freq_fidelity": float("nan")}, "l1_low_freq_fidelity"),
    ({"l1_low_freq_fidelity": float("inf")}, "l1_low_freq_fidelity"),
    ({"l1_low_freq_sigma": 0.0}, "l1_low_freq_sigma"),
    ({"l1_low_freq_sigma": -1.0}, "l1_low_freq_sigma"),
    ({"l1_low_freq_sigma": float("inf")}, "l1_low_freq_sigma"),
])
def test_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        S.Config(**kwargs)


def test_low_level_fidelity_terms_are_validated():
    c = np.zeros(10)
    with pytest.raises(ValueError, match="length 10"):
        SM.l1_path(c, 1.0, fidelity_target=np.zeros(9))
    with pytest.raises(ValueError, match="finite and >= 0"):
        SM.l1_path(c, 1.0, fidelity_weight=np.full(10, -1.0))
    with pytest.raises(ValueError, match="finite"):
        SM.l1_path(c, 1.0, fidelity_target=np.full(10, np.nan))


def test_cli_passes_opt_in_controls_into_production_config(monkeypatch, capsys):
    captured = {}

    def fake_stabilize(*args, **kwargs):
        captured["cfg"] = args[4]
        return {"frames": 0}

    monkeypatch.setattr(cli, "stabilize", fake_stabilize)
    cli.main(["in.mp4", "out.mp4", "--l1-low-freq-fidelity", "4.5", "--l1-low-freq-sigma", "9"])
    assert captured["cfg"].l1_low_freq_fidelity == 4.5
    assert captured["cfg"].l1_low_freq_sigma == 9.0
    assert '"frames": 0' in capsys.readouterr().out
