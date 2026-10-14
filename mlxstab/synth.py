"""Deterministic synthetic test clip with ground truth (CC0 by construction): three parallax layers (far texture, mid
vertical 'trunks', near blobs), an independently moving subject, a smooth intended camera path plus jitter, and a rolling
shutter (row y is captured rho*y/H of a frame later)."""
import numpy as np
import mlx.core as mx
from scipy.ndimage import gaussian_filter
from .warp import sample


def _noise(rng, h, w, s):
    n = gaussian_filter(rng.standard_normal((h, w, 3)), (s, s, 0)); n = (n - n.mean()) / (n.std() + 1e-9)
    return n


def make_layers(W, H, seed=0, margin=160):
    rng = np.random.default_rng(seed); h, w = H + 2 * margin, W + 2 * margin
    g = _noise(rng, h, w, 6)[..., :1] * 40 + _noise(rng, h, w, 1.5)[..., :1] * 22      # mostly luminance texture, mild colour
    far = np.clip(125 + g + 10 * _noise(rng, h, w, 8), 0, 255); far[..., 2] += 20
    mid = np.zeros((h, w, 4), np.float32)
    for x in rng.integers(0, w, 14):
        bw = min(int(rng.integers(8, 22)), w - x); col = 30 + 40 * rng.random()
        mid[:, x:x + bw, :3] = col + 15 * _noise(rng, h, bw, 2)[:, :, :3]; mid[:, x:x + bw, 3] = 1
    near = np.zeros((h, w, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(6):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(30, 70)
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        near[m, :3] = np.array([200, 60, 40]) + 30 * rng.standard_normal(3); near[m, 3] = 1
    return [np.clip(far, 0, 255).astype(np.float32), mid, near], margin


def camera(T, seed=1, jitter=6.0, rot_jit=0.6):
    """(intended [T,3], actual(tau) function) in px/px/deg. tau is in frames (continuous, for rolling shutter)."""
    rng = np.random.default_rng(seed)
    knots = np.arange(-4, T + 6)
    jx, jy, jr = (gaussian_filter(rng.standard_normal(len(knots)), 0.8) for _ in range(3))
    jx, jy, jr = (v / (v.std() + 1e-9) for v in (jx, jy, jr))
    def intended(tau):
        tau = np.asarray(tau, np.float64)
        return np.stack([40 * np.sin(tau / 70.0) + 0.5 * tau, 10 * np.sin(tau / 55.0), 1.2 * np.sin(tau / 90.0)], -1)
    def actual(tau):
        tau = np.asarray(tau, np.float64)
        j = np.stack([np.interp(tau, knots, jx) * jitter, np.interp(tau, knots, jy) * jitter, np.interp(tau, knots, jr) * rot_jit], -1)
        return intended(tau) + j
    return intended, actual


def render(T=90, W=320, H=180, seed=0, rho=0.8, jitter=6.0, rot_jit=0.6, depths=(1.0, 0.55, 0.25), subject=True):
    layers, m = make_layers(W, H, seed)
    intended, actual = camera(T, seed + 1, jitter, rot_jit)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    out = np.zeros((T, H, W, 3), np.uint8)
    for t in range(T):
        tau = t + rho * ys[:, 0] / H                                            # per-row capture time
        pose = actual(tau)                                                       # [H,3]
        th = np.deg2rad(pose[:, 2])[:, None]
        rx, ry = xs - cx, ys - cy
        ux = rx * np.cos(th) - ry * np.sin(th) + cx; uy = rx * np.sin(th) + ry * np.cos(th) + cy
        acc = None
        for d, L in zip(depths, layers):
            c = np.stack([ux + m + d * pose[:, 0:1], uy + m + d * pose[:, 1:2]], -1)
            s = np.array(sample(mx.array(L)[None], mx.array(c.astype(np.float32))[None], "bilinear", "border")[0])
            if L.shape[-1] == 3: acc = s
            else: a = s[..., 3:4]; acc = acc * (1 - a) + s[..., :3] * a
        if subject:                                                              # independently moving disk, fixed in the world
            sxw = 60 + 1.6 * t; syw = H * 0.55
            ex = sxw - (ux + 0 * pose[:, 0:1]) + 0; ey = syw - uy
            disk = (ex ** 2 + ey ** 2) < 22 ** 2
            acc[disk] = np.array([240, 220, 60]) * (0.8 + 0.2 * np.sin(ex[disk] / 3))[:, None]
        out[t] = np.clip(acc, 0, 255).astype(np.uint8)
    return out, intended(np.arange(T))
