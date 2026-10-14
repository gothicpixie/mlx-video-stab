"""No-crop video stabilizer: RAFT flow -> robust global homography + mesh/row residuals -> L1-optimal camera paths -> one
Lanczos resample to the full original frame, plus a per-frame mask of pixels that have no source."""
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import mlx.core as mx
from . import geometry as G, motion as MO, smooth as SM, media
from .warp import sample, valid_mask
from .raft.model import load_raft, raft_flow


@dataclass
class Config:
    smooth: str = "l1"             # "l1" (Grundmann) or "gauss"
    sigma: float = 15.0            # gaussian sigma in frames (smooth="gauss")
    l1_weights: tuple = (10.0, 1.0, 100.0)
    budget: float = 0.04           # max path deviation as a fraction of frame width = the border budget
    mesh: tuple = (16, 9)          # cells (cols, rows) of the residual mesh; (0, 0) disables
    mesh_budget: float = 0.012     # residual correction cap, fraction of width
    rs_rows: int = 18              # row bands for rolling-shutter correction; 0 disables
    flow_iters: int = 24
    flow_downscale: int = 1        # 1 = full resolution flow
    warm_start: bool = True
    step: int = 8                  # flow sampling grid for motion fitting
    kernel: str = "lanczos3"       # final resampling kernel
    zoom: float = 1.0              # optional output zoom composed into the same warp
    seed: int = 0
    model: str = "similarity"      # global camera model: similarity | affine | homography (residuals go to the mesh)
    fidelity: float = 0.1          # L1 pull of the path towards the raw path (keeps the border small; 1.0 leaked jitter, 0 -> big border)


def load_roll(path):
    """Roll series (degrees, positive = counter-clockwise lean). JSON: [deg per frame] | [[t, deg(, conf)], ...] | {"series": ...}."""
    d = json.loads(Path(path).read_text()); d = d.get("series", d) if isinstance(d, dict) else d
    a = np.array(d, np.float64)
    return ("frames", a) if a.ndim == 1 else ("timed", a[:, :2])


def roll_per_frame(roll, T, fps):
    if roll is None: return np.zeros(T)
    kind, a = roll
    if kind == "frames":
        return np.interp(np.arange(T), np.arange(len(a)), a)
    return np.interp(np.arange(T) / fps, a[:, 0], a[:, 1])


class RaftFlow:
    """Forward/backward RAFT flow with optional warm start from the previous pair's low-res flow."""
    def __init__(self, weights, iters=24, downscale=1, warm=True, corr_dtype=mx.float32):
        self.m = load_raft(weights); self.iters, self.ds, self.warm, self.cd = iters, downscale, warm, corr_dtype
        self.prev = {}

    def _prep(self, im):
        x = mx.array(im).astype(mx.float32)
        if self.ds > 1:
            H, W = x.shape[:2]; H2, W2 = H // self.ds, W // self.ds
            x = x[:H2 * self.ds, :W2 * self.ds].reshape(H2, self.ds, W2, self.ds, 3).mean(axis=(1, 3))
        return x

    def __call__(self, a, b, key="f"):
        from .raft.model import pad8, unpad
        H, W = a.shape[:2]
        A, p = pad8(self._prep(a)[None]); B, _ = pad8(self._prep(b)[None])
        init = self.prev.get(key) if self.warm else None
        low, up = self.m(A, B, self.iters, init, corr_dtype=self.cd)
        self.prev[key] = low; mx.eval(low, up)
        f = unpad(up, p)[0]
        if self.ds > 1:
            h, w = f.shape[:2]
            ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
            c = mx.stack([(xs + .5) / self.ds - .5, (ys + .5) / self.ds - .5], -1)[None].astype(mx.float32)
            f = sample(f[None], c, "bilinear", "border")[0] * self.ds
        return f


@dataclass
class Motion:
    W: int
    H: int
    fps: float
    fits: list = field(default_factory=list)


def analyze(frames, flow_fn, cfg, W, H, fps, log=None, masks=None, on_step=None):
    """Pass 1: motion fit for each consecutive pair. `frames` yields uint8 [H,W,3]; flow_fn(a, b, key) -> [H,W,2] mx."""
    mot = Motion(W, H, fps); prev = None; rng = np.random.default_rng(cfg.seed)
    gc, gr = cfg.mesh
    pm = None
    for i, fr in enumerate(frames):
        cm = next(masks) if masks is not None else None
        if on_step: on_step(i)                 # hook for callers that share the GPU (e.g. release/re-acquire a lock between chunks)
        if prev is not None:
            Ff = flow_fn(prev, fr, "f"); Bf = flow_fn(fr, prev, "b")
            obs = MO.observe(Ff, Bf, cfg.step)
            if cm is not None: MO.exclude_masked(obs, pm, cm)
            mot.fits.append(MO.fit_motion(obs, (max(gc, 1), max(gr, 1)), cfg.rs_rows, rng=rng, model=cfg.model))
            if log and i % 20 == 0: log(f"flow {i}")
        prev = fr; pm = cm
    return mot


@dataclass
class Plan:
    G: list                    # per frame: 3x3 virtual-frame pixel -> source-frame pixel (global part)
    mesh_corr: np.ndarray      # [T, gr+1, gc+1, 2] source displacement correction
    row_corr: np.ndarray       # [T, nr, 2]
    raw_quads: np.ndarray      # [T,4,2] in reference coords
    path_quads: np.ndarray     # smoothed
    fallbacks: int = 0


def plan(mot, cfg, roll=None):
    W, H, T = mot.W, mot.H, len(mot.fits) + 1
    cs = G.corners(W, H)
    Cs = [np.eye(3)]
    for f in mot.fits: Cs.append(Cs[-1] @ np.linalg.inv(f.H))                 # frame t+1 -> reference
    quads = np.stack([G.apply_h(C, cs) for C in Cs])                           # camera path in corner space
    rolls = roll_per_frame(roll, T, mot.fps) if roll is not None else None
    target = np.stack([G.rotate_quad(q, r) for q, r in zip(quads, rolls)]) if rolls is not None else quads.copy()
    centre = np.array([W / 2 - .5, H / 2 - .5]); base = cs - centre
    a_l, t_l, res_l = [], [], []
    for q in target:
        a, t, r = G.similarity_fit(base, q); a_l.append(a); t_l.append(t); res_l.append(r)
    a = np.array(a_l); t = np.array(t_l); res = np.stack(res_l)
    th = np.unwrap(np.angle(a)); ls = np.log(np.abs(a))
    R = 0.5 * np.hypot(W, H); b = cfg.budget * W
    P = np.stack([t.real, t.imag], 1)
    P = SM.smooth_columns(P, cfg.smooth, cfg.sigma, b, cfg.l1_weights, cfg.fidelity)
    ang = SM.smooth_columns(np.stack([th, ls], 1) * R, cfg.smooth, cfg.sigma, b, cfg.l1_weights, cfg.fidelity) / R      # in corner-pixel units
    rs = SM.smooth_columns(res.reshape(T, 8), cfg.smooth, cfg.sigma, b, cfg.l1_weights, cfg.fidelity).reshape(T, 4, 2)
    path = np.zeros_like(quads)
    for i in range(T):
        path[i] = G.similarity_apply(base, np.exp(ang[i, 1] + 1j * ang[i, 0]), P[i, 0] + 1j * P[i, 1]) + rs[i]
        if cfg.zoom != 1.0: c = path[i].mean(0); path[i] = c + (path[i] - c) / cfg.zoom
    Gs = [np.linalg.inv(Cs[i]) @ G.dlt(cs, path[i]) for i in range(T)]
    # mesh / rolling-shutter residual paths (content motion beyond the global homography), accumulated and smoothed
    gc, gr = cfg.mesh
    mesh_corr = np.zeros((T, gr + 1, gc + 1, 2)); row_corr = np.zeros((T, max(cfg.rs_rows, 1), 2))
    mb = cfg.mesh_budget * W
    if gc > 0 and mot.fits:
        M = np.concatenate([np.zeros((1, gr + 1, gc + 1, 2)), np.cumsum([f.mesh for f in mot.fits], 0)])
        Ms = SM.smooth_columns(M.reshape(T, -1), cfg.smooth, cfg.sigma, mb, cfg.l1_weights, cfg.fidelity).reshape(M.shape)
        mesh_corr = M - Ms
    if cfg.rs_rows > 0 and mot.fits:
        Rw = np.concatenate([np.zeros((1, cfg.rs_rows, 2)), np.cumsum([f.rows for f in mot.fits], 0)])
        Rs = SM.smooth_columns(Rw.reshape(T, -1), cfg.smooth, cfg.sigma, mb, cfg.l1_weights, cfg.fidelity).reshape(Rw.shape)
        row_corr = Rw - Rs
    return Plan(Gs, mesh_corr, row_corr, quads, path, sum(f.fallback for f in mot.fits))


def source_coords(pl, t, W, H, grid=None):
    """Virtual-frame pixel -> source pixel field [H,W,2] (global homography + mesh + row corrections)."""
    if grid is None:
        ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij"); grid = (xs.astype(mx.float32), ys.astype(mx.float32))
    x, y = grid
    Gm = [[float(v) for v in r] for r in pl.G[t]]
    # reference matrix entries as float64 python floats; mx computes in float32 (|coords| ~ 1e3 -> ~1e-4 px)
    w = Gm[2][0] * x + Gm[2][1] * y + Gm[2][2]
    sx = (Gm[0][0] * x + Gm[0][1] * y + Gm[0][2]) / w; sy = (Gm[1][0] * x + Gm[1][1] * y + Gm[1][2]) / w
    src = mx.stack([sx, sy], -1)[None]
    gr, gc = pl.mesh_corr.shape[1] - 1, pl.mesh_corr.shape[2] - 1
    if gc > 0:
        u = (src[..., 0] + .5) / (W / gc); v = (src[..., 1] + .5) / (H / gr)
        mesh = mx.array(pl.mesh_corr[t].astype(np.float32))[None]
        src = src + sample(mesh, mx.stack([u, v], -1), "bilinear", "border")
    nr = pl.row_corr.shape[1]
    if np.abs(pl.row_corr[t]).max() > 0:
        v = (src[..., 1] + .5) / (H / nr) - .5
        rows = mx.array(pl.row_corr[t].astype(np.float32)).reshape(1, nr, 1, 2)
        src = src + sample(rows, mx.stack([mx.zeros_like(v), v], -1), "bilinear", "border")
    return src


def render_frame(frame, pl, t, cfg, grid=None):
    """-> (stabilized uint8 [H,W,3], border mask bool [H,W] (True = no source), distortion)."""
    H, W = frame.shape[:2]
    src = source_coords(pl, t, W, H, grid)
    img = sample(mx.array(frame).astype(mx.float32)[None], src, cfg.kernel, "zeros")
    bad = ~valid_mask(src, H, W)
    img = mx.where(bad[..., None], 0.0, mx.clip(img, 0, 255))
    mx.eval(img, bad, src)
    return np.array(img[0]).round().astype(np.uint8), np.array(bad[0]), _distortion(pl.G[t], W, H)


def _distortion(Gm, W, H):
    """Liu et al. 2013 distortion: ratio of the two singular values of the affine part of the global output->input homography
    (1 = none). The residual mesh is local detail and is not counted."""
    from .geometry import apply_h
    ys, xs = np.mgrid[0:H:H // 8 or 1, 0:W:W // 8 or 1]; q = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    A = np.c_[q, np.ones(len(q))]
    sol = np.linalg.lstsq(A, apply_h(Gm, q), rcond=None)[0]
    sv = np.linalg.svd(sol[:2].T, compute_uv=False)
    return float(sv[1] / sv[0])


def stabilize(src, out, mask=None, weights=None, cfg=None, roll=None, start=None, duration=None, fps_out=None,
              flow_fn=None, log=print, report=None, on_step=None, dump_motion=None):
    cfg = cfg or Config()
    info = media.probe(src)
    if flow_fn is None:
        flow_fn = RaftFlow(weights, cfg.flow_iters, cfg.flow_downscale, cfg.warm_start)
    frames, fps, (W, H) = media.decode(src, info, start, duration)
    if info["interlaced"]: log(f"interlaced ({info['field_order']}): bwdif send_field -> {fps:.3f} fps")
    import time; t0 = time.perf_counter()
    mot = analyze(frames, flow_fn, cfg, W, H, fps, log, on_step=on_step); t1 = time.perf_counter()
    if dump_motion:
        import pickle; Path(dump_motion).write_bytes(pickle.dumps(mot))
    pl = plan(mot, cfg, load_roll(roll) if isinstance(roll, (str, Path)) else roll)
    t2 = time.perf_counter()
    frames, fps, _ = media.decode(src, info, start, duration)
    enc = media.Encoder(out, W, H, fps, out_fps=fps_out)
    menc = media.Encoder(mask, W, H, fps, gray=True, out_fps=fps_out) if mask else None
    grid = None; borders = []; dist = []
    ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij"); grid = (xs.astype(mx.float32), ys.astype(mx.float32))
    for t, fr in enumerate(frames):
        if t >= len(pl.G): break
        if on_step: on_step(t)
        img, bad, d = render_frame(fr, pl, t, cfg, grid)
        enc.write(img); borders.append(float(bad.mean())); dist.append(d)
        if menc: menc.write(bad.astype(np.uint8) * 255)
        if t % 20 == 0: log(f"render {t}/{len(pl.G)}")
    enc.close()
    if menc: menc.close()
    t3 = time.perf_counter(); n = len(pl.G)
    rep = {"frames": n, "timing_s": {"analysis": t1 - t0, "plan": t2 - t1, "render": t3 - t2},
           "fps_analysis": n / (t1 - t0), "fps_render": n / (t3 - t2), "fps_total": n / (t3 - t0), "fps": fps, "interlaced": info["interlaced"], "fallback_frames": pl.fallbacks,
           "mean_border_frac": float(np.mean(borders)), "max_border_frac": float(np.max(borders)),
           "cropping_ratio": float(1 - np.mean(borders)), "distortion_min": float(np.min(dist)), "distortion_mean": float(np.mean(dist))}
    if report: Path(report).write_text(json.dumps(rep, indent=1))
    return rep
