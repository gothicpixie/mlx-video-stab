"""Per-frame motion from dense optical flow: robust global homography + spatially varying (mesh / per-row) residual.

Everything here consumes flow *samples* (a coarse grid of the dense flow), so it is cheap and testable without a flow net."""
from dataclasses import dataclass
import numpy as np
import mlx.core as mx
from .geometry import apply_h, ransac_homography, dlt, fit_model
from .warp import sample


@dataclass
class Obs:                       # flow samples for one frame pair (frame t -> t+1)
    pts: np.ndarray              # [N,2] pixel positions in frame t
    flow: np.ndarray             # [N,2]
    valid: np.ndarray            # [N] bool, forward-backward consistent and inside the frame
    W: int
    H: int


@dataclass
class Fit:
    H: np.ndarray                # 3x3, frame t -> t+1 (global)
    mesh: np.ndarray             # [gr+1, gc+1, 2] residual displacement at mesh vertices (content motion beyond H)
    rows: np.ndarray             # [nr, 2] x-independent residual per row band (rolling shutter), content motion
    inlier_frac: float
    mover_frac: float
    fallback: bool


def observe(F, B, step=8, fb_abs=1.0, fb_rel=0.05):
    """Sample dense flows F (t->t+1) and B (t+1->t), [H,W,2] mx arrays, on a grid; forward-backward consistency gives `valid`."""
    H, W = F.shape[:2]
    ys = np.arange(step // 2, H, step); xs = np.arange(step // 2, W, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    f = F[mx.array(ys)][:, mx.array(xs)].reshape(-1, 2)
    tgt = mx.array(pts) + f
    b = sample(B[None], tgt.reshape(1, 1, -1, 2), "bilinear", "border").reshape(-1, 2)
    err = mx.sqrt(mx.sum((f + b) ** 2, -1)); mag = mx.sqrt(mx.sum(f ** 2, -1))
    inside = (tgt[:, 0] >= 0) & (tgt[:, 0] <= W - 1) & (tgt[:, 1] >= 0) & (tgt[:, 1] <= H - 1)
    valid = (err < fb_abs + fb_rel * mag) & inside
    mx.eval(f, valid)
    return Obs(pts.astype(np.float64), np.array(f, np.float64), np.array(valid), W, H)


def _kernel_field(pts, vals, w, centers, sig, iters=3, huber=1.5):
    """Robust Gaussian-kernel regression of vals [N,k] at centers [V,d] (anisotropic sig [d]). Returns (est [V,k], mass [V])."""
    d2 = (((pts[:, None, :] - centers[None]) / sig) ** 2).sum(-1)
    K = np.exp(-0.5 * d2) * (d2 < 9.0)
    rw = np.ones(len(pts))
    for _ in range(iters):
        Kw = K * (w * rw)[:, None]
        mass = Kw.sum(0)
        est = (Kw.T @ vals) / np.maximum(mass, 1e-9)[:, None]
        at = (K @ est) / np.maximum(K.sum(1), 1e-9)[:, None]
        r = np.linalg.norm(vals - at, axis=1)
        rw = np.minimum(1.0, huber / np.maximum(r, 1e-9))
    return est, (K * w[:, None]).sum(0)


def _fill(field, mass, thr):
    """Replace under-observed vertices (mass<thr) by diffusion from observed neighbours (field [R,C,2])."""
    ok = mass >= thr
    f = np.where(ok[..., None], field, 0.0)
    if ok.all(): return f
    if not ok.any(): return np.zeros_like(f)
    cur = ok.copy()
    for _ in range(field.shape[0] + field.shape[1]):
        if cur.all(): break
        pad = np.pad(np.where(cur[..., None], f, 0.0), ((1, 1), (1, 1), (0, 0))); pc = np.pad(cur.astype(float), 1)
        nsum = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        ncnt = pc[:-2, 1:-1] + pc[2:, 1:-1] + pc[1:-1, :-2] + pc[1:-1, 2:]
        new = (~cur) & (ncnt > 0)
        f[new] = nsum[new] / ncnt[new][:, None]; cur = cur | new
    return f


def _movers(obs, resid, inl, gate=6.0, cell=32):
    """Mask of points on independently moving things: coherent blobs whose flow disagrees with the dominant motion."""
    s = 1.4826 * np.median(resid[inl]) if inl.any() else 1.0
    out = obs.valid & (resid > max(gate, 4 * s))
    gy, gx = obs.H // cell + 1, obs.W // cell + 1
    cy = (obs.pts[:, 1] // cell).astype(int); cx = (obs.pts[:, 0] // cell).astype(int)
    tot = np.zeros((gy, gx)); bad = np.zeros((gy, gx))
    np.add.at(tot, (cy, cx), obs.valid); np.add.at(bad, (cy, cx), out)
    cellbad = (tot > 0) & (bad / np.maximum(tot, 1) >= 0.5)
    # keep blobs of >= 3 cells, dilate one cell
    from scipy.ndimage import label, binary_dilation
    lab, n = label(cellbad)
    keep = np.zeros_like(cellbad)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= 3: keep |= m
    keep = binary_dilation(keep, iterations=1)
    return keep[cy, cx], keep


def fit_motion(obs, mesh=(16, 9), rs_rows=18, thr=1.5, rng=None, min_valid=60, model="similarity"):
    """Global homography by RANSAC on valid flow samples, movers excluded, then row-wise and mesh residual fields."""
    gc, gr = mesh
    W, H = obs.W, obs.H
    v = obs.valid
    zero = Fit(np.eye(3), np.zeros((gr + 1, gc + 1, 2)), np.zeros((max(rs_rows, 1), 2)), 0.0, 0.0, True)
    if v.sum() < min_valid: return zero
    src = obs.pts[v]; dst = src + obs.flow[v]
    Hm, inl = ransac_homography(src, dst, thr=thr, rng=rng)
    inlier_frac = inl.mean()
    if inl.sum() < min_valid or inlier_frac < 0.15: return zero
    mv_last = np.zeros(len(v), bool)
    for _ in range(2):                                                  # re-fit with moving subjects removed
        resid_all = np.linalg.norm(obs.pts + obs.flow - apply_h(Hm, obs.pts), axis=1)
        mv, _cells = _movers(obs, resid_all, (resid_all < thr) & v)
        use = v & ~mv; mv_last = mv
        if use.sum() < min_valid: break
        Hm, inl = ransac_homography(obs.pts[use], obs.pts[use] + obs.flow[use], thr=thr, rng=rng)
    if model != "homography":                                           # homography RANSAC picks inliers; the kept model is lower-dof
        e = np.linalg.norm(obs.pts + obs.flow - apply_h(Hm, obs.pts), axis=1)
        k = v & ~mv_last & (e < 2 * thr)
        if k.sum() >= min_valid: Hm = fit_model(obs.pts[k], obs.pts[k] + obs.flow[k], model)
    resid_vec = obs.flow - (apply_h(Hm, obs.pts) - obs.pts)
    resid_all = np.linalg.norm(resid_vec, axis=1)
    mv, _ = _movers(obs, resid_all, (resid_all < thr) & v)
    use = v & ~mv
    w = use.astype(float)
    # rolling shutter: x-independent residual per row band
    rows = np.zeros((max(rs_rows, 1), 2))
    if rs_rows > 0:
        cy = (np.arange(rs_rows) + .5) * H / rs_rows - .5
        est, mass = _kernel_field(obs.pts[:, 1:2], resid_vec, w, cy[:, None], np.array([H / rs_rows]))
        rows = _fill(est[:, None], mass[:, None], 3.0)[:, 0]
        row_at = np.stack([np.interp(obs.pts[:, 1], cy, rows[:, 0]), np.interp(obs.pts[:, 1], cy, rows[:, 1])], 1)
        resid_vec = resid_vec - row_at
    vx = -.5 + np.arange(gc + 1) * W / gc; vy = -.5 + np.arange(gr + 1) * H / gr
    cx, cy2 = np.meshgrid(vx, vy)
    centers = np.stack([cx.ravel(), cy2.ravel()], 1)
    est, mass = _kernel_field(obs.pts, resid_vec, w, centers, np.array([W / gc, H / gr]))
    mesh_f = _fill(est.reshape(gr + 1, gc + 1, 2), mass.reshape(gr + 1, gc + 1), 3.0)
    p = np.pad(mesh_f, ((1, 1), (1, 1), (0, 0)), mode="edge")                # light spatial regularisation [1 2 1]^2/16
    mesh_f = (p[:-2, 1:-1] + 2 * p[1:-1, 1:-1] + p[2:, 1:-1]) / 4
    p = np.pad(mesh_f, ((0, 0), (1, 1), (0, 0)), mode="edge"); mesh_f = (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4
    return Fit(Hm, mesh_f, rows, float(inlier_frac), float(mv[v].mean()), False)


def exclude_masked(obs, m0, m1, pad=12):
    """Drop samples near invalid (no-source / border) pixels of either frame. m0, m1: bool [H,W], True = invalid."""
    for m in (m0, m1):
        if m is None: continue
        mm = mx.array(m.astype(np.float32))[None, :, :, None]
        H, W = m.shape
        k = 2 * pad + 1
        # separable max filter (cheap): rows then columns
        a = mx.pad(mm, [(0, 0), (pad, pad), (0, 0), (0, 0)]); r = a[:, :H]
        for d in range(1, k): r = mx.maximum(r, a[:, d:d + H])
        b = mx.pad(r, [(0, 0), (0, 0), (pad, pad), (0, 0)]); c = b[:, :, :W]
        for d in range(1, k): c = mx.maximum(c, b[:, :, d:d + W])
        pts = obs.pts.astype(int)
        bad = np.array(c[0, :, :, 0])[pts[:, 1], pts[:, 0]] > 0
        obs.valid &= ~bad
