import numpy as np, pytest
cv2 = pytest.importorskip("cv2")
import mlx.core as mx
from mlxstab import synth, stabilize as S, motion as MO


def farneback(a, b, key=None):
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    f = cv2.calcOpticalFlowFarneback(g(a), g(b), None, 0.5, 4, 21, 5, 7, 1.5, 0)
    return mx.array(f.astype(np.float32))


def jitter_px(vid, cfg):
    """mean |frame-to-frame residual motion| of the global translation (high-pass of per-frame velocity)."""
    from scipy.ndimage import gaussian_filter1d
    mot = S.analyze(iter(vid), farneback, cfg, vid.shape[2], vid.shape[1], 30.0)
    v = np.array([f.H[:2, 2] for f in mot.fits])
    return float(np.abs(v - gaussian_filter1d(v, 8, axis=0, mode="nearest")).mean())


def test_stabilizer_reduces_jitter_and_masks_border():
    vid, _ = synth.render(T=80, W=256, H=144, jitter=6.0)
    cfg = S.Config(budget=0.06, step=8, mesh=(8, 4), rs_rows=8)
    mot = S.analyze(iter(vid), farneback, cfg, 256, 144, 30.0)
    assert sum(f.fallback for f in mot.fits) < 10
    pl = S.plan(mot, cfg)
    out = []; masks = []
    for t in range(len(pl.G)):
        im, bad, d = S.render_frame(vid[t], pl, t, cfg); out.append(im); masks.append(bad)
        assert d > 0.9
    out = np.stack(out)
    jin, jout = jitter_px(vid[:len(out)], cfg), jitter_px(out, cfg)
    print("jitter in/out", jin, jout, "border", np.mean(masks))
    assert jout < 0.5 * jin
    assert 0 < np.mean(masks) < 0.2                                           # a border exists, and it is bounded by the budget
    assert np.all(out[np.array(masks)] == 0)                                  # masked pixels carry no content


def test_roll_levelling_sign():
    from mlxstab import metrics
    vid, _ = synth.render(T=30, W=256, H=144, jitter=0.0, rot_jit=0.0, subject=False)
    a = np.deg2rad(5.0); H_, W_ = 144, 256
    ys, xs = np.mgrid[0:H_, 0:W_].astype(np.float32); dx, dy = xs - W_ / 2, ys - H_ / 2
    c = np.stack([dx * np.cos(a) - dy * np.sin(a) + W_ / 2, dx * np.sin(a) + dy * np.cos(a) + H_ / 2], -1)      # content rotated visually CCW by 5 deg
    from mlxstab.warp import sample
    tilted = np.stack([np.array(sample(mx.array(f.astype(np.float32))[None], mx.array(c)[None], "bilinear", "border")[0]).astype(np.uint8) for f in vid])
    lean_in = metrics.vline_wobble  # noqa: just to make sure cv2 path imports
    def lean(frames):
        import cv2
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD); r = []
        for fr in frames[::5]:
            ls = lsd.detect(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY))[0]; a_ = []
            for x1, y1, x2, y2 in ls[:, 0]:
                ddx, ddy = float(x2 - x1), float(y1 - y2)
                if ddy < 0: ddx, ddy = -ddx, -ddy
                if np.hypot(ddx, ddy) > 35 and abs(np.degrees(np.arctan2(-ddx, ddy))) < 20: a_.append(np.degrees(np.arctan2(-ddx, ddy)))
            r.append(np.median(a_))
        return float(np.median(r))
    cfg = S.Config(budget=0.12, mesh=(0, 0), rs_rows=0)
    mot = S.analyze(iter(tilted), farneback, cfg, W_, H_, 30.0)
    pl = S.plan(mot, cfg, ("frames", np.full(30, 5.0)))
    out = np.stack([S.render_frame(tilted[t], pl, t, cfg)[0] for t in range(len(pl.G))])
    li, lo = lean(tilted), lean(out)
    print("lean in/out", li, lo)
    assert li > 3.5 and abs(lo) < 1.5
