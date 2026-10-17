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
    for blob_index in range(6):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(30, 70)
        if blob_index == 0:
            # Guarantee one near object intersects the nominal camera window. Remaining blobs stay random,
            # preserving scene variation without allowing a seed to silently omit the near layer entirely.
            cy, cx = H // 2 + margin, W // 2 + margin
            r = max(12, min(32, int(round(0.25 * min(W, H)))))
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        near[m, :3] = np.array([200, 60, 40]) + 30 * rng.standard_normal(3); near[m, 3] = 1
    return [np.clip(far, 0, 255).astype(np.float32), mid, near], margin


def camera(T, seed=1, jitter=6.0, rot_jit=0.6, intent_mode="smooth", intent_profile="historical"):
    """Return known intent and actual camera pose in px/px/degrees.

    ``intent_mode='smooth'`` with ``intent_profile='historical'`` preserves the original path exactly. With
    ``intent_profile='matched-spectrum'``, smooth and stochastic paths share the same zero-mean periodic component
    spectra, so its displacement, velocity, acceleration and band energy match while phase structure changes. The
    common horizontal drift is separate and identical in both paths.
    ``intent_mode='static'`` supplies a zero intended path for calibration controls. Camera shake uses a separate,
    unchanged RNG stream, so paired intent modes receive identical shake for the same seed. ``tau`` is continuous
    frame time for rolling-shutter row sampling.
    """
    rng = np.random.default_rng(seed)
    knots = np.arange(-4, T + 6)
    jx, jy, jr = (gaussian_filter(rng.standard_normal(len(knots)), 0.8) for _ in range(3))
    jx, jy, jr = (v / (v.std() + 1e-9) for v in (jx, jy, jr))

    if intent_mode not in ("smooth", "stochastic", "static"):
        raise ValueError("intent_mode must be smooth|stochastic|static")
    if intent_profile not in ("historical", "matched-spectrum"):
        raise ValueError("intent_profile must be historical|matched-spectrum")
    if intent_profile == "matched-spectrum" and T < 16:
        raise ValueError("matched-spectrum intent requires at least 16 frames")

    if intent_mode == "static":
        def intended(tau):
            tau = np.asarray(tau, np.float64)
            return np.zeros(tau.shape + (3,), dtype=np.float64)
    elif intent_profile == "matched-spectrum":
        # Randomize only Fourier phase. Per-axis amplitudes and bins are fixed, preserving the
        # complete discrete power spectrum and therefore the finite-difference velocity/acceleration
        # energies. The shared horizontal drift is kept outside this phase-randomized component.
        t = np.arange(T, dtype=np.float64)
        bins = np.arange(1, min(6, (T - 1) // 2) + 1, dtype=np.int64)
        target_rms = (16.0, 6.0, 0.7)
        waves = np.zeros((T, 3), dtype=np.float64)
        phase_rng = np.random.default_rng(int(seed) ^ 0x5EED5EED)
        phases = (np.zeros(bins.size, dtype=np.float64) if intent_mode == "smooth"
                  else phase_rng.uniform(0.0, 2.0 * np.pi, size=bins.size))
        if bins.size:
            weights = 1.0 / bins.astype(np.float64)
            weights /= np.sqrt(np.sum(weights ** 2) / 2.0)
            for axis, rms in enumerate(target_rms):
                phase = 2.0 * np.pi * np.outer(t, bins) / max(T, 1) + phases[None, :]
                waves[:, axis] = rms * np.sum(weights[None, :] * np.cos(phase), axis=1)

        def intended(tau):
            tau = np.asarray(tau, np.float64)
            values = np.stack([np.interp(tau, t, waves[:, axis]) if T > 1 else np.zeros_like(tau)
                               for axis in range(3)], axis=-1)
            values[..., 0] += 0.5 * tau
            return values
    elif intent_mode == "smooth":
        def intended(tau):
            tau = np.asarray(tau, np.float64)
            return np.stack([40 * np.sin(tau / 70.0) + 0.5 * tau,
                             10 * np.sin(tau / 55.0), 1.2 * np.sin(tau / 90.0)], -1)
    elif intent_mode == "stochastic":
        intent_rng = np.random.default_rng(int(seed) ^ 0x5EED5EED)
        profiles = [gaussian_filter(intent_rng.standard_normal(len(knots)), 12.0) for _ in range(3)]
        target_stds = (32.0, 10.0, 1.2)
        profiles = [p / (p.std() + 1e-12) * std for p, std in zip(profiles, target_stds)]
        profiles[0] = profiles[0] + 0.5 * knots

        def intended(tau):
            tau = np.asarray(tau, np.float64)
            return np.stack([np.interp(tau, knots, p) for p in profiles], axis=-1)
    else:  # ``static`` already returned above; keep an explicit guard for future modes.
        raise ValueError("unsupported intent mode")

    def actual(tau):
        tau = np.asarray(tau, np.float64)
        j = np.stack([np.interp(tau, knots, jx) * jitter, np.interp(tau, knots, jy) * jitter, np.interp(tau, knots, jr) * rot_jit], -1)
        return intended(tau) + j
    return intended, actual


def render(T=90, W=320, H=180, seed=0, rho=0.8, jitter=6.0, rot_jit=0.6,
           depths=(1.0, 0.55, 0.25), subject=True, subject_speed=1.6, intent_mode="smooth",
           intent_profile="historical", return_truth=False):
    layers, m = make_layers(W, H, seed)
    if len(depths) != 3 or not np.all(np.isfinite(depths)) or any(d <= 0 for d in depths):
        raise ValueError("depths must contain three finite positive layer motion factors")
    if not np.isfinite(subject_speed):
        raise ValueError("subject_speed must be finite")
    intended, actual = camera(T, seed + 1, jitter, rot_jit, intent_mode=intent_mode,
                              intent_profile=intent_profile)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    out = np.zeros((T, H, W, 3), np.uint8)
    visible_layer_id = np.zeros((T, H, W), np.uint8) if return_truth else None
    for t in range(T):
        tau = t + rho * ys[:, 0] / H                                            # per-row capture time
        pose = actual(tau)                                                       # [H,3]
        th = np.deg2rad(pose[:, 2])[:, None]
        rx, ry = xs - cx, ys - cy
        ux = rx * np.cos(th) - ry * np.sin(th) + cx; uy = rx * np.sin(th) + ry * np.cos(th) + cy
        acc = None
        for layer_index, (d, L) in enumerate(zip(depths, layers)):
            c = np.stack([ux + m + d * pose[:, 0:1], uy + m + d * pose[:, 1:2]], -1)
            s = np.array(sample(mx.array(L)[None], mx.array(c.astype(np.float32))[None], "bilinear", "border")[0])
            if L.shape[-1] == 3: acc = s
            else:
                a = s[..., 3:4]
                acc = acc * (1 - a) + s[..., :3] * a
                if return_truth:
                    visible_layer_id[t, a[..., 0] >= 0.5] = np.uint8(layer_index)
        if subject:                                                              # independently moving disk, fixed in the world
            sxw = 60 + subject_speed * t; syw = H * 0.55
            # Sample the subject in world coordinates with the same inverse camera transform as
            # the near layer. This makes ``subject_speed=0`` world-static: camera motion still moves
            # it in the image, while nonzero speed adds independent world motion.
            ex = sxw - (ux + pose[:, 0:1]); ey = syw - (uy + pose[:, 1:2])
            disk = (ex ** 2 + ey ** 2) < 22 ** 2
            acc[disk] = np.array([240, 220, 60]) * (0.8 + 0.2 * np.sin(ex[disk] / 3))[:, None]
            if return_truth:
                visible_layer_id[t, disk] = 3
        out[t] = np.clip(acc, 0, 255).astype(np.uint8)
    if return_truth:
        return out, intended(np.arange(T)), {"visible_layer_id": visible_layer_id}
    return out, intended(np.arange(T))
