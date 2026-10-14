"""Homography helpers (numpy, float64). Coordinates are pixel centres on integers; a W x H frame spans [-0.5, W-0.5]."""
import numpy as np


def corners(W, H):
    return np.array([[-.5, -.5], [W - .5, -.5], [W - .5, H - .5], [-.5, H - .5]], np.float64)


def apply_h(Hm, p):
    p = np.asarray(p, np.float64); q = p @ Hm[:2, :2].T + Hm[:2, 2]
    w = p @ Hm[2, :2] + Hm[2, 2]
    return q / w[..., None]


def dlt(src, dst, w=None):
    """Least-squares homography src->dst (Hartley-normalised DLT). src,dst [N,2], N>=4."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    def norm(p):
        c = p.mean(0); s = np.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1)).mean(), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.]])
    Ts, Td = norm(src), norm(dst)
    a = apply_h(Ts, src); b = apply_h(Td, dst)
    z = np.zeros(len(a)); o = np.ones(len(a))
    A = np.concatenate([np.stack([-a[:, 0], -a[:, 1], -o, z, z, z, b[:, 0] * a[:, 0], b[:, 0] * a[:, 1], b[:, 0]], 1),
                        np.stack([z, z, z, -a[:, 0], -a[:, 1], -o, b[:, 1] * a[:, 0], b[:, 1] * a[:, 1], b[:, 1]], 1)])
    if w is not None: sw = np.sqrt(np.asarray(w, np.float64)); A = A * np.concatenate([sw, sw])[:, None]
    h = np.linalg.eigh(A.T @ A)[1][:, 0].reshape(3, 3)          # smallest right singular vector via the 9x9 normal matrix
    Hm = np.linalg.inv(Td) @ h @ Ts
    return Hm / Hm[2, 2]


def ransac_homography(src, dst, iters=300, thr=1.5, rng=None, max_score_pts=3000):
    """Robust homography src->dst. Returns (H, inlier_mask). Batched 4-point hypotheses, then weighted refinement."""
    rng = rng or np.random.default_rng(0)
    n = len(src)
    if n < 8: return np.eye(3), np.zeros(n, bool)
    c = src.mean(0); s = max(src.std(), 1e-6)                                # condition the minimal solves
    a = (src - c) / s; b = (dst - c) / s
    idx = rng.integers(0, n, (iters, 4))
    A = np.zeros((iters, 8, 8)); rhs = np.zeros((iters, 8))
    x, y = a[idx, 0], a[idx, 1]; u, v = b[idx, 0], b[idx, 1]
    for k in range(4):
        A[:, 2 * k] = np.stack([x[:, k], y[:, k], np.ones(iters), 0 * x[:, k], 0 * x[:, k], 0 * x[:, k], -u[:, k] * x[:, k], -u[:, k] * y[:, k]], 1); rhs[:, 2 * k] = u[:, k]
        A[:, 2 * k + 1] = np.stack([0 * x[:, k], 0 * x[:, k], 0 * x[:, k], x[:, k], y[:, k], np.ones(iters), -v[:, k] * x[:, k], -v[:, k] * y[:, k]], 1); rhs[:, 2 * k + 1] = v[:, k]
    ok = np.abs(np.linalg.det(A)) > 1e-9
    A[~ok] = np.eye(8); rhs[~ok] = 0
    h = np.linalg.solve(A, rhs[..., None])[..., 0]
    Hs = np.concatenate([h, np.ones((iters, 1))], 1).reshape(iters, 3, 3)
    pa = np.concatenate([a, np.ones((n, 1))], 1)
    sub = rng.choice(n, min(n, max_score_pts), replace=False) if n > max_score_pts else np.arange(n)      # score hypotheses on a subset
    proj = (Hs.reshape(iters * 3, 3) @ pa[sub].T).reshape(iters, 3, -1)
    pr = proj[:, :2] / np.where(np.abs(proj[:, 2:3]) < 1e-9, 1e-9, proj[:, 2:3])
    err = np.sqrt(((pr - b[sub].T[None]) ** 2).sum(1)) * s
    score = np.where(ok[:, None], (err < thr), False).sum(1)
    best = int(score.argmax())
    pb = Hs[best] @ pa.T; pb = pb[:2] / np.where(np.abs(pb[2]) < 1e-9, 1e-9, pb[2])
    inl = np.sqrt(((pb.T - b) ** 2).sum(1)) * s < thr
    Hm = None
    for _ in range(3):                                                        # refine on inliers (IRLS-ish)
        if inl.sum() < 8: break
        Hm = dlt(src[inl], dst[inl])
        e = np.sqrt(((apply_h(Hm, src) - dst) ** 2).sum(1)); inl = e < thr
    if Hm is None: return np.eye(3), np.zeros(n, bool)
    return Hm, inl


def similarity_fit(base, quad):
    """Best similarity (complex a = s*e^{i*theta}, translation t) mapping base [4,2] -> quad [4,2], plus the 4x2 residual."""
    zb = base[:, 0] + 1j * base[:, 1]; zc = quad[:, 0] + 1j * quad[:, 1]
    mb, mc = zb.mean(), zc.mean()
    a = ((zc - mc) * np.conj(zb - mb)).sum() / (np.abs(zb - mb) ** 2).sum()
    t = mc - a * mb
    fit = a * zb + t
    res = np.stack([(zc - fit).real, (zc - fit).imag], 1)
    return a, t, res


def similarity_apply(base, a, t):
    z = a * (base[:, 0] + 1j * base[:, 1]) + t
    return np.stack([z.real, z.imag], 1)


def rotate_quad(quad, deg_ccw):
    """Rotate a quad about its centroid by a VISUAL counter-clockwise angle (image y points down)."""
    c = quad.mean(0); r = np.deg2rad(deg_ccw); cs, sn = np.cos(r), np.sin(r)
    d = quad - c
    return c + np.stack([d[:, 0] * cs + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * cs], 1)


def fit_model(src, dst, model="similarity", w=None):
    """Least-squares global model src->dst as a 3x3 matrix: 'similarity' (4 dof), 'affine' (6) or 'homography' (8)."""
    if model == "homography": return dlt(src, dst, w)
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if w is None else np.asarray(w, np.float64)
    if model == "affine":
        A = np.concatenate([np.c_[src, np.ones(len(src))], ]); sw = np.sqrt(w)[:, None]
        sol = np.linalg.lstsq(A * sw, dst * sw, rcond=None)[0]
        Hm = np.eye(3); Hm[:2] = sol.T; return Hm
    ms, md = (w[:, None] * src).sum(0) / w.sum(), (w[:, None] * dst).sum(0) / w.sum()
    zs = (src[:, 0] - ms[0]) + 1j * (src[:, 1] - ms[1]); zd = (dst[:, 0] - md[0]) + 1j * (dst[:, 1] - md[1])
    a = (w * zd * np.conj(zs)).sum() / (w * np.abs(zs) ** 2).sum()
    t = (md[0] + 1j * md[1]) - a * (ms[0] + 1j * ms[1])
    return np.array([[a.real, -a.imag, t.real], [a.imag, a.real, t.imag], [0, 0, 1.]])
