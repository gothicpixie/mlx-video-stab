"""Independent camera-motion evaluation from sparse KLT features (RAFT-free).

Why this exists
---------------
``mlxstab.metrics`` measures input and output jitter/wobble from the *same* dense RAFT estimator the
stabilizer uses. A single estimator's systematic errors can therefore reward themselves, and the plan calls
for "independent tracks or another estimator for at least part of evaluation". This module estimates the same
inter-frame global motion from sparse corners tracked with pyramidal Lucas-Kanade and robustly fitted as a
similarity, then reports the *same* jitter/path metrics so the two estimators can be compared.

Boundaries (explicit)
---------------------
- This is a *different* estimator, not ground truth. KLT fails on low texture, motion blur, large
  displacement and rolling shutter, and LK is a local linear approximation. Agreement bounds gross
  disagreement between two estimators; it does not establish correctness.
- Nothing here is imported by ``stabilize``/``render``/``plan``. It is evaluation-only and changes no
  production default.
- ``fallback`` pairs (too few corners, too few forward-backward-consistent tracks, too few RANSAC inliers)
  are reported explicitly. ``jitter_px_supported`` recomputes the metric over the supported sub-sequence
  instead of scoring a failed fit as perfectly stable static motion.

OpenCV is an *optional* dependency (like ``metrics.vline_wobble``); the functions raise a clear error when it
is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

cv2 = None

from . import geometry as G
from . import motion as MO


@dataclass
class _Motion:
    """Minimal Motion-shaped record (W, H, fps, fits) so ``metrics.jitter``/``path_series`` can consume the
    track path without importing ``mlxstab.stabilize`` (keeps this estimator independent of the flow stack)."""
    W: int
    H: int
    fps: float
    fits: list = field(default_factory=list)
    stride: int = 1
    n_frames: int = 0

# Defaults are deliberately conservative (small displacement, low texture resilience is not attempted).
DEFAULTS = dict(max_corners=500, quality=0.01, min_distance=8.0, block_size=7,
                win=21, levels=3, fb_tol=1.0, ransac_thr=1.5, min_tracks=8, ransac_iters=300, seed=0)

#: Smallest input jitter (px) below which an output/input ratio is not interpretable.
MIN_RATIO_JITTER_PX = 0.05


def _require_cv2():
    global cv2
    if cv2 is None:
        try:  # lazy: default RAFT-only evaluation must not pay OpenCV import cost
            import cv2 as cv2_module
        except Exception as exc:  # pragma: no cover - exercised only without OpenCV
            raise RuntimeError("mlxstab.tracks needs OpenCV (cv2) for sparse KLT tracking; it is an optional dependency") from exc
        cv2 = cv2_module
    return cv2


def has_cv2():
    """Load-check OpenCV only when a caller explicitly enables sparse KLT tracking."""
    try:
        _require_cv2()
        return True
    except RuntimeError:
        return False


def _gray(frame):
    """uint8 RGB [H,W,3] -> contiguous uint8 grayscale [H,W]."""
    _require_cv2()
    f = np.asarray(frame)
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)
    f = np.ascontiguousarray(f)
    if f.ndim == 2:
        return f
    if f.ndim == 3 and f.shape[2] == 3:
        return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    if f.ndim == 3 and f.shape[2] == 4:
        return cv2.cvtColor(f, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"expected a grayscale or RGB frame, got shape {f.shape}")


def _ransac_similarity(src, dst, thr, rng, iters=300, min_inliers=8):
    """Deterministic robust similarity ``src -> dst`` ([N,2] float64), returning ``(H, inlier_mask)``.

    A minimal sample is two points (a similarity has four degrees of freedom). Unlike ``geometry.ransac_homography``
    this fits a similarity, which is what a shake/short-pan camera pair is dominated by and which is much less
    prone to overfitting a handful of tracks than a full homography. The RNG is caller-supplied, so the result is
    reproducible.
    """
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    n = len(src)
    if n < 2:
        return np.eye(3), np.zeros(n, bool)
    best_inl = np.zeros(n, bool); best_H = None
    for _ in range(int(iters)):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(src[i] - src[j]) < 1e-6:
            continue
        H = G.fit_model(src[[i, j]], dst[[i, j]], "similarity")
        if not np.all(np.isfinite(H)):
            continue
        e = np.linalg.norm(G.apply_h(H, src) - dst, axis=1)
        inl = e < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl; best_H = H
    if best_H is None:
        return np.eye(3), np.zeros(n, bool)
    for _ in range(3):                                     # IRLS-ish refinement on the current inliers
        if best_inl.sum() < 2:
            break
        best_H = G.fit_model(src[best_inl], dst[best_inl], "similarity")
        e = np.linalg.norm(G.apply_h(best_H, src) - dst, axis=1)
        best_inl = e < thr
    if best_inl.sum() < min_inliers:
        return np.eye(3), np.zeros(n, bool)
    return best_H, best_inl


def pair_similarity(prev_gray, next_gray, *, prev_invalid=None, next_invalid=None,
                    max_corners=500, quality=0.01, min_distance=8.0, block_size=7,
                    win=21, levels=3, fb_tol=1.0, ransac_thr=1.5, min_tracks=8, ransac_iters=300, rng=None):
    """Independent ``prev -> next`` inter-frame similarity from sparse KLT tracks (grayscale uint8 inputs).

    Returns ``(H, fallback, inlier_frac, n_tracks)``. ``H`` is identity when ``fallback`` is True. Tracks
    failing the forward-backward consistency check (or the image bounds) are dropped before fitting. This is a
    motion *estimator*, not a flow field: it does not produce per-pixel correspondences.
    """
    _require_cv2()
    rng = rng if rng is not None else np.random.default_rng(0)
    Hh, Ww = prev_gray.shape[:2]
    def checked_mask(mask, name):
        if mask is None:
            return None
        mask = np.asarray(mask, bool)
        if mask.shape != (Hh, Ww):
            raise ValueError(f"{name} must have shape {(Hh, Ww)}, got {mask.shape}")
        return mask
    prev_invalid = checked_mask(prev_invalid, "prev_invalid")
    next_invalid = checked_mask(next_invalid, "next_invalid")
    detect_mask = None if prev_invalid is None else np.where(prev_invalid, 0, 255).astype(np.uint8)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=int(max_corners), qualityLevel=float(quality),
                                 mask=detect_mask,
                                 minDistance=float(min_distance), blockSize=int(block_size))
    if p0 is None or len(p0) < min_tracks:
        return np.eye(3), True, 0.0, 0
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None,
                                          winSize=(int(win), int(win)), maxLevel=int(levels), criteria=crit)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, p1, None,
                                           winSize=(int(win), int(win)), maxLevel=int(levels), criteria=crit)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1)
    if int(ok.sum()) < min_tracks:
        return np.eye(3), True, 0.0, int(ok.sum())
    a = p0[ok, 0].astype(np.float64); b = p1[ok, 0].astype(np.float64); ab = p0b[ok, 0].astype(np.float64)
    fb = np.linalg.norm(a - ab, axis=1)
    ok = fb < fb_tol
    # Respect both sampler support and the validity masks at the correct correspondence endpoints.
    in_bounds = ((a[:, 0] >= 0) & (a[:, 0] <= Ww - 1) & (a[:, 1] >= 0) & (a[:, 1] <= Hh - 1) &
                 (b[:, 0] >= 0) & (b[:, 0] <= Ww - 1) & (b[:, 1] >= 0) & (b[:, 1] <= Hh - 1) &
                 (ab[:, 0] >= 0) & (ab[:, 0] <= Ww - 1) & (ab[:, 1] >= 0) & (ab[:, 1] <= Hh - 1))
    ok &= in_bounds
    if prev_invalid is not None or next_invalid is not None:
        # Reuse the RAFT evaluator's conservative mask semantics: reject a feature if either its source
        # support or its destination support touches a dilated no-source region.
        obs = MO.Obs(a, b - a, np.ones(len(a), bool), Ww, Hh)
        MO.exclude_masked(obs, prev_invalid, next_invalid)
        ok &= obs.valid
    if int(ok.sum()) < min_tracks:
        return np.eye(3), True, 0.0, int(ok.sum())
    a = a[ok]; b = b[ok]
    H, inl = _ransac_similarity(a, b, ransac_thr, rng, iters=ransac_iters, min_inliers=min_tracks)
    if not np.all(np.isfinite(H)):
        return np.eye(3), True, 0.0, len(a)
    if int(inl.sum()) < min_tracks:
        # RANSAC deliberately returns an empty inlier mask when refinement loses support. Do not pass empty
        # arrays to fit_model: its weighted-mean divisions would emit NaNs/warnings before the fallback.
        return np.eye(3), True, 0.0, len(a)
    H = G.fit_model(a[inl], b[inl], "similarity")
    if not np.all(np.isfinite(H)) or H[2, 2] == 0:
        return np.eye(3), True, 0.0, len(a)
    return H / H[2, 2], False, float(inl.mean()), int(len(a))


class TrackAccumulator:
    """Single-pass KLT tracker: ``push`` each RGB frame in order, then read ``motion``/``metrics``.

    Computes one similarity per consecutive pair. Storing only the 3x3 transforms keeps memory O(frames);
    no dense flow is produced. Use ``wrap`` to tee an existing frame iterator through the accumulator while
    ``metrics.measure`` consumes it, so the clip is decoded and visited once.
    """

    def __init__(self, W, H, fps, **kwargs):
        _require_cv2()
        self.W, self.H, self.fps = int(W), int(H), float(fps)
        cfg = {**DEFAULTS, **kwargs}
        self.rng = np.random.default_rng(int(cfg.pop("seed")))
        self.cfg = cfg
        self.transforms = []                                   # list of (H, fallback, inlier_frac, n_tracks)
        self.n_frames = 0
        self._prev_gray = None
        self._prev_invalid = None

    def push(self, frame, invalid_mask=None):
        g = _gray(frame)
        invalid = None if invalid_mask is None else np.asarray(invalid_mask, bool)
        if invalid is not None and invalid.shape != g.shape:
            raise ValueError(f"invalid_mask must have shape {g.shape}, got {invalid.shape}")
        if self._prev_gray is not None:
            self.transforms.append(pair_similarity(self._prev_gray, g, prev_invalid=self._prev_invalid,
                                                   next_invalid=invalid, rng=self.rng, **self.cfg))
        self._prev_gray = g
        self._prev_invalid = invalid
        self.n_frames += 1

    def wrap(self, frames, invalid_masks=None):
        """Yield frames unchanged while tracking them; optional masks are True for pixels with no source.

        When masks are supplied they must have exactly one entry per frame. Mask iterators can be tee'd by the
        caller so the flow evaluator receives the same sequence in lockstep.
        """
        if invalid_masks is None:
            for f in frames:
                self.push(f)
                yield f
        else:
            for f, mask in zip(frames, invalid_masks, strict=True):
                self.push(f, mask)
                yield f

    def _fits(self):
        fits = []
        for H, fb, inl, n in self.transforms:
            fits.append(MO.Fit(H, np.zeros((2, 2, 2)), np.zeros((1, 2)), float(inl), 0.0, bool(fb)))
        return fits

    def motion(self):
        return _Motion(self.W, self.H, self.fps, fits=self._fits(), stride=1, n_frames=self.n_frames)

    def metrics(self, sigma=15.0):
        """Same keys as ``metrics.measure`` where they are defined, plus support/reliability accounting.

        Path metrics are only reported when every pair is supported. Replacing failed fits with identity would
        make a weak estimator look artificially stable, while deleting failed pairs would join non-adjacent
        observations as if they were consecutive. Partial tracking support is therefore reported as unavailable
        for path jitter/stability, with the failed-pair count preserved explicitly.
        """
        from . import metrics as M
        n = len(self.transforms)
        fallback = int(sum(1 for _, fb, _, _ in self.transforms if fb))
        base = {"method": "klt_similarity", "fallback_frames": fallback, "support_frames": n,
                "fallback_frac": (fallback / n) if n else None,
                "inlier_frac_mean": float(np.mean([inl for _, _, inl, _ in self.transforms])) if n else None,
                "inlier_frac_min": float(np.min([inl for _, _, inl, _ in self.transforms])) if n else None,
                "tracks_mean": float(np.mean([nt for _, _, _, nt in self.transforms])) if n else None,
                "n_frames": self.n_frames}
        base["supported_frames"] = n - fallback
        if n < 2:                                              # too short to high-pass a path
            base.update({"stability": None, "jitter_px": None, "frame_motion_px": None,
                         "jitter_px_supported": None,
                         "unavailable_reason": "fewer than two tracked pairs"})
            return base
        if fallback:
            base["jitter_px_supported"] = None
            base["stability"] = None
            base["jitter_px"] = None
            base["frame_motion_px"] = None
            base["unavailable_reason"] = (
                f"{fallback} failed pair(s); path metrics require contiguous support and do not treat failures as identity"
            )
        else:
            mot = self.motion()
            jt, fm = M.jitter(mot, sigma=sigma)
            ser = M.path_series(mot)
            base.update({"stability": min(M.stability(ser)), "jitter_px": jt,
                         "frame_motion_px": fm, "jitter_px_supported": jt})
        return base


def track_metrics(frames, W, H, fps, **kwargs):
    """Convenience: consume an iterable of RGB frames and return the track metrics dict."""
    acc = TrackAccumulator(W, H, fps, **kwargs)
    for f in frames:
        acc.push(f)
    return acc.metrics()


def _direction(ratio, tol):
    if ratio is None or not np.isfinite(ratio):
        return "unknown"
    if ratio <= 1.0 - tol:
        return "improves"
    if ratio >= 1.0 + tol:
        return "worsens"
    return "unchanged"


def agreement(raft_input, raft_output, track_input, track_output, *, tol=0.05, min_input_jitter_px=MIN_RATIO_JITTER_PX):
    """Compare the RAFT- and KLT-derived input->output jitter change.

    ``*_input``/``*_output`` are metric dicts (as returned by ``metrics.measure`` / ``TrackAccumulator.metrics``)
    for the same input and stabilized output clips. A ratio ``output/input < 1`` means the output is steadier.
    Returns both ratios, their agreement on direction, the relative gap, and machine-readable reliability
    reasons. This is a *cross-check between two estimators*, not ground truth: agreement means only that the
    single-estimator story is not contradicted.
    """
    reasons = []
    out = {"raft_jitter_ratio": None, "track_jitter_ratio": None,
           "raft_direction": "unknown", "track_direction": "unknown",
           "agree_on_direction": None, "ratio_relative_gap": None, "reliable": False, "reasons": reasons}

    def _ratio(inp, outp, tag):
        if not isinstance(inp, dict) or not isinstance(outp, dict):
            reasons.append(f"{tag} metrics missing"); return None
        for m, side in ((inp, "input"), (outp, "output")):
            failed = int(m.get("fallback_frames") or 0)
            if failed > 0:
                reasons.append(f"{tag} {side} has {failed} failed fit(s); path metrics are unavailable")
            jv = m.get("jitter_px")
            if jv is None or not np.isfinite(jv):
                reasons.append(f"{tag} {side} jitter_px missing/non-finite"); return None
        if inp["jitter_px"] < min_input_jitter_px:
            reasons.append(f"{tag} input jitter {inp['jitter_px']:.4g} px below the {min_input_jitter_px} px "
                           "ratio floor")
            return None
        return float(outp["jitter_px"] / inp["jitter_px"])

    r_raft = _ratio(raft_input, raft_output, "raft")
    r_track = _ratio(track_input, track_output, "track")
    out["raft_jitter_ratio"] = r_raft
    out["track_jitter_ratio"] = r_track
    out["raft_direction"] = _direction(r_raft, tol)
    out["track_direction"] = _direction(r_track, tol)
    if r_raft is not None and r_track is not None:
        out["agree_on_direction"] = out["raft_direction"] == out["track_direction"]
        denom = max(abs(r_raft), abs(r_track), 1e-12)
        out["ratio_relative_gap"] = float(abs(r_raft - r_track) / denom)
    out["reliable"] = not reasons and out["agree_on_direction"] is not None
    return out
