import numpy as np
from mlxstab import geometry as G, smooth as S


def test_dlt_and_ransac():
    rng = np.random.default_rng(0)
    Ht = np.array([[1.02, 0.03, 4.0], [-0.02, 0.99, -3.0], [2e-5, -1e-5, 1.0]])
    src = rng.uniform(0, 1000, (800, 2)); dst = G.apply_h(Ht, src)
    assert np.abs(G.dlt(src, dst) - Ht).max() < 1e-6
    bad = dst.copy(); bad[:300] += rng.uniform(-60, 60, (300, 2)); bad += rng.normal(0, .2, bad.shape)
    H, inl = G.ransac_homography(src, bad, thr=1.5)
    assert np.abs(G.apply_h(H, src[300:]) - dst[300:]).max() < 2.0 and inl[300:].mean() > .9


def test_similarity_and_roll_sign():
    q = G.corners(640, 360); a, t, res = G.similarity_fit(q, G.rotate_quad(q, 10))
    assert np.abs(res).max() < 1e-9 and abs(np.angle(a) + np.deg2rad(10)) < 1e-9     # visual CCW = negative angle with y down


def test_l1_path_is_piecewise_and_within_budget():
    rng = np.random.default_rng(1); T = 200
    c = np.cumsum(np.r_[np.zeros(100), np.full(100, 1.0)]) + rng.normal(0, 3, T)    # a pan with shake
    p = S.l1_path(c, 12.0)
    assert np.abs(p - c).max() <= 12.0 + 1e-6
    d3 = np.abs(np.diff(p, 3)); assert (d3 < 1e-6).mean() > .8                           # constant-acceleration segments, jerk only at few knots
    assert np.abs(np.diff(p)).mean() < np.abs(np.diff(c)).mean() / 3
    assert np.abs(S.l1_path_windowed(c, 12.0, window=80, overlap=20) - c).max() <= 12.0 + 1e-6
