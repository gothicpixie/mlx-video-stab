"""Objective stabilization metrics, measured on the video itself (input and output measured the same way).

stability   Liu et al. 2013: fraction of camera-path energy in frequency bins 2..6 over 2..N/2 (DC excluded) for x, y and
            rotation; the score is the minimum. Higher is steadier. Depends on clip length: compare input vs output only.
jitter_px   mean |residual frame-to-frame motion| in px: per-frame velocity of 5 image points minus its Gaussian-smoothed trend
            (intentional pans are kept in the trend).
wobble_px   rolling-shutter wobble: RMS over row bands of the per-row inter-frame motion after removing its linear trend in y.
cropping    1 - mean fraction of pixels with no source (needs the mask).
distortion  Liu et al. 2013 anisotropy of the global warp; reported by stabilize() (min over frames).
vline_deg   optional (needs OpenCV): std of the high-passed lean angle of long near-vertical line segments.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import geometry as G
from .stabilize import analyze, Config


def path_series(mot):
    cs = G.corners(mot.W, mot.H); Cs = [np.eye(3)]
    for f in mot.fits: Cs.append(Cs[-1] @ np.linalg.inv(f.H))
    centre = np.array([mot.W / 2 - .5, mot.H / 2 - .5]); base = cs - centre
    a, t = zip(*[G.similarity_fit(base, G.apply_h(C, cs))[:2] for C in Cs])
    a = np.array(a); t = np.array(t)
    return np.stack([t.real, t.imag, np.rad2deg(np.unwrap(np.angle(a)))], 1)


def stability(series):
    out = []
    for k in range(series.shape[1]):
        P = np.abs(np.fft.rfft(series[:, k])) ** 2; n = len(series)
        tot = P[1:n // 2 + 1].sum()
        out.append(float(P[1:6].sum() / tot) if tot > 0 else 1.0)
    return out


def jitter(mot, sigma=15.0):
    W, H = mot.W, mot.H
    pts = np.array([[W / 2, H / 2], [0, 0], [W, 0], [W, H], [0, H]], np.float64)
    v = np.array([(G.apply_h(f.H, pts) - pts) for f in mot.fits])                  # [T-1,5,2]
    v = v.reshape(len(v), -1)
    hp = v - gaussian_filter1d(v, sigma, axis=0, mode="nearest")
    return float(np.abs(hp).mean()), float(np.abs(v).mean())


def wobble(mot):
    w = []
    for f in mot.fits:
        if f.fallback: continue
        r = f.rows; y = np.arange(len(r))
        res = [r[:, k] - np.polyval(np.polyfit(y, r[:, k], 1), y) for k in (0, 1)]
        w.append(np.sqrt(np.mean(np.square(res))))
    return float(np.mean(w)) if w else float("nan")


def vline_wobble(frames, every=2, sigma=6.0):
    try: import cv2
    except Exception: return None
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD); ang = []
    for i, fr in enumerate(frames):
        if i % every: continue
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), (3, 3), 0)
        ls = lsd.detect(g)[0]; a = []; w = []
        if ls is not None:
            for x1, y1, x2, y2 in ls[:, 0]:
                dx, dy = float(x2 - x1), float(y1 - y2)
                if dy < 0: dx, dy = -dx, -dy
                d = np.degrees(np.arctan2(-dx, dy)); ln = np.hypot(dx, dy)
                if abs(d) <= 15 and ln >= 0.12 * g.shape[0]: a.append(d); w.append(ln)
        if a:
            o = np.argsort(a); c = np.cumsum(np.array(w)[o]); ang.append(float(np.array(a)[o][np.searchsorted(c, c[-1] / 2)]))
        else: ang.append(np.nan)
    ang = np.array(ang); ok = ~np.isnan(ang)
    if ok.sum() < 8: return None
    ang = np.interp(np.arange(len(ang)), np.flatnonzero(ok), ang[ok])
    return float(np.std(ang - gaussian_filter1d(ang, sigma, mode="nearest")))


def measure(frames, flow_fn, W, H, fps, masks=None, keep_frames=False):
    """frames: iterable of uint8 [H,W,3]. Returns dict of metrics (see module docstring)."""
    cfg = Config(mesh=(0, 0), rs_rows=18)
    frames = list(frames) if keep_frames else frames
    mot = analyze(iter(frames), flow_fn, cfg, W, H, fps, masks=masks)
    ser = path_series(mot); st = stability(ser); jt, fm = jitter(mot)
    r = {"stability": min(st), "stability_xyr": st, "jitter_px": jt, "frame_motion_px": fm, "wobble_px": wobble(mot),
         "fallback_frames": int(sum(f.fallback for f in mot.fits))}
    if keep_frames:
        v = vline_wobble(frames)
        if v is not None: r["vline_wobble_deg"] = v
    return r
