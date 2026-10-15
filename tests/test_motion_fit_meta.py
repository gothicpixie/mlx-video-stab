"""Focused tests for the ``fit_motion`` fallback-gate diagnostics.

``Fit.n_valid_samples`` / ``Fit.n_ransac_inliers`` / ``Fit.fallback_reason`` record which fit gate produced an
identity fallback and with which observed counts. They are diagnostics only: they label the existing
``fit_motion`` refusal (``inl.sum() < min_valid or inlier_frac < 0.15``) and never change the fit, plan or
pixels. Legacy / hand-built Fits keep the dataclass defaults (``None``), meaning "not recorded", not zero.

Covers: the explicit reason precedence, every fallback branch (insufficient valid samples, insufficient RANSAC
inliers, low RANSAC inlier fraction) plus a normal successful fit, and that a hand-built Fit leaves the new
fields unset. No media, FFmpeg, RAFT or changed production defaults.
"""
import numpy as np

from mlxstab import motion as MO


def _obs(n, *, flow=(3.0, 2.0), valid=True, W=640, H=480, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(5.0, W - 5.0, n), rng.uniform(5.0, H - 5.0, n)], 1)
    f = np.tile(np.asarray(flow, np.float64), (n, 1))
    if valid is True:
        v = np.ones(n, bool)
    elif valid is False:
        v = np.zeros(n, bool)
    else:
        v = np.asarray(valid, bool)
    return MO.Obs(pts, f, v, W, H)


# --------------------------------------------------------------------------------------
# the stable reason helper and its precedence
# --------------------------------------------------------------------------------------

def test_fallback_reason_none_when_every_gate_passes():
    assert MO.fallback_reason(500, 400, min_valid=60) == MO.FALLBACK_REASON_NONE
    assert MO.fallback_reason(60, 60, min_valid=60) == MO.FALLBACK_REASON_NONE


def test_fallback_reason_each_gate_in_order():
    assert MO.fallback_reason(10, 0, min_valid=60) == MO.FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES
    assert MO.fallback_reason(100, 40, min_valid=60) == MO.FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS
    assert MO.fallback_reason(500, 60, min_valid=60) == MO.FALLBACK_REASON_LOW_RANSAC_INLIER_FRACTION


def test_fallback_reason_precedence_when_several_gates_fail():
    # Both the absolute-count gate and the fraction gate fail; the earlier count gate names the reason.
    assert MO.fallback_reason(500, 40, min_valid=60) == MO.FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS
    # Too few valid samples dominates everything else.
    assert MO.fallback_reason(5, 5, min_valid=60) == MO.FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES


def test_fallback_reason_matches_the_existing_refusal_expression():
    for n_valid, n_inliers in [(0, 0), (100, 10), (500, 60), (500, 75), (960, 960), (60, 60)]:
        frac = (n_inliers / n_valid) if n_valid else 0.0
        old_refuses = n_valid < 60 or n_inliers < 60 or frac < 0.15
        assert (MO.fallback_reason(n_valid, n_inliers, 60) != MO.FALLBACK_REASON_NONE) == old_refuses


# --------------------------------------------------------------------------------------
# fit_motion end to end
# --------------------------------------------------------------------------------------

def test_fit_motion_insufficient_valid_samples_branch():
    obs = _obs(10, valid=False)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4)
    assert fit.fallback is True
    assert fit.fallback_reason == MO.FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES
    assert fit.n_valid_samples == 0 and fit.n_ransac_inliers == 0
    assert np.array_equal(fit.H, np.eye(3))


def test_fit_motion_insufficient_ransac_inliers_branch(monkeypatch):
    obs = _obs(100)

    def fake(src, dst, **kw):
        return np.eye(3), np.zeros(len(src), bool)          # too few inliers in absolute count

    monkeypatch.setattr(MO, "ransac_homography", fake)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4, min_valid=60)
    assert fit.fallback is True
    assert fit.fallback_reason == MO.FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS
    assert fit.n_valid_samples == 100 and fit.n_ransac_inliers == 0


def test_fit_motion_low_ransac_inlier_fraction_branch(monkeypatch):
    obs = _obs(500)

    def fake(src, dst, **kw):
        m = np.zeros(len(src), bool); m[:60] = True         # enough in absolute count, below the fraction gate
        return np.eye(3), m

    monkeypatch.setattr(MO, "ransac_homography", fake)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4, min_valid=60)
    assert fit.fallback is True
    assert fit.fallback_reason == MO.FALLBACK_REASON_LOW_RANSAC_INLIER_FRACTION
    assert fit.n_valid_samples == 500 and fit.n_ransac_inliers == 60


def test_fit_motion_prefers_count_reason_when_both_gates_fail(monkeypatch):
    obs = _obs(500)

    def fake(src, dst, **kw):
        m = np.zeros(len(src), bool); m[:40] = True         # 40 < 60 and 40/500 = 0.08 < 0.15
        return np.eye(3), m

    monkeypatch.setattr(MO, "ransac_homography", fake)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4, min_valid=60)
    assert fit.fallback_reason == MO.FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS


def test_fit_motion_successful_fit_records_counts_and_reason_none():
    obs = _obs(960)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4)
    assert fit.fallback is False
    assert fit.fallback_reason == MO.FALLBACK_REASON_NONE
    assert fit.n_valid_samples == 960 and fit.n_ransac_inliers == 960
    assert fit.inlier_frac == 1.0
    assert np.isfinite(fit.mesh).all() and np.isfinite(fit.rows).all()


def test_hand_built_fit_leaves_gate_metadata_unset():
    fit = MO.Fit(np.eye(3), np.zeros((4, 5, 2)), np.zeros((2, 2)), 0.0, 0.0, True)
    assert fit.n_valid_samples is None and fit.n_ransac_inliers is None and fit.fallback_reason is None
