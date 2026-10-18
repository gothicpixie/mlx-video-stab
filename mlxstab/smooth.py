"""Camera-path smoothing: L1-optimal (Grundmann et al. 2011) and Gaussian, with a box budget around the raw path."""
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linprog
from scipy.signal import butter, sosfiltfilt


def _diff_mat(T, k):
    D = sparse.eye(T, format="csr")
    for _ in range(k): D = (D[1:] - D[:-1])
    return D


def _fidelity_terms(T, c, fidelity, target, weight):
    """Resolve the soft fidelity pull into a per-sample target and nonnegative weight.

    ``weight=None`` keeps the historical scalar ``fidelity`` (and the raw path as target). ``target`` must
    already match ``c`` sample-for-sample; ``weight`` may be scalar or per-sample."""
    tw = np.asarray(fidelity if weight is None else weight, np.float64)
    try: tw = np.broadcast_to(tw, (T,)).copy()
    except ValueError:
        raise ValueError(f"fidelity weight must be scalar or have length {T}, got shape {tw.shape}") from None
    if not np.all(np.isfinite(tw)) or np.any(tw < 0):
        raise ValueError("fidelity weight must be finite and >= 0")
    tt = np.asarray(c if target is None else target, np.float64)
    if tt.shape != (T,):
        raise ValueError(f"fidelity target must have length {T}, got shape {tt.shape}")
    if not np.all(np.isfinite(tt)):
        raise ValueError("fidelity target must contain only finite values")
    return tt, tw


def l1_path(c, budget, w=(10.0, 1.0, 100.0), fixed=None, fidelity=0.0, *, fidelity_target=None,
            fidelity_weight=None):
    """argmin_p  w1|Dp|_1 + w2|D^2 p|_1 + w3|D^3 p|_1 + sum_t fw_t |p_t - ft_t|
    s.t. |p - c| <= budget  (c: [T]; budget scalar or [T]).

    The optimum is piecewise static / constant-velocity / constant-acceleration (tripod, dolly, ease in/out).
    ``fidelity``/``fidelity_target``/``fidelity_weight`` add an optional soft pull: scalar ``fidelity`` (with
    target ``c``) is the historical behaviour; a per-sample ``fidelity_weight`` may instead target a
    per-sample ``fidelity_target``. The hard budget around ``c`` is unchanged.
    `fixed` = (values array, count): pin the first `count` samples (used to chain windows)."""
    c = np.asarray(c, np.float64); T = len(c)
    if T < 5: return c.copy()
    b = np.broadcast_to(np.asarray(budget, np.float64), (T,)).copy()
    ft, fw = _fidelity_terms(T, c, fidelity, fidelity_target, fidelity_weight)
    use_fidelity = bool(np.any(fw > 0))
    Ds = [_diff_mat(T, k) for k in (1, 2, 3)]
    n_e = [T - k for k in (1, 2, 3)]; off = np.cumsum([T] + n_e)             # variable layout: d (T), e1, e2, e3 (+ |d| slack f)
    nv = int(off[-1]) + (T if use_fidelity else 0)
    cost = np.zeros(nv)
    for i, wi in enumerate(w): cost[off[i]:off[i + 1]] = wi
    rows, rhs = [], []
    for i, D in enumerate(Ds):
        pad = sparse.hstack([D, sparse.csr_matrix((D.shape[0], nv - T))], format="csr")
        eye_e = sparse.csr_matrix((np.ones(D.shape[0]), (np.arange(D.shape[0]), off[i] + np.arange(D.shape[0]))), shape=(D.shape[0], nv))
        dc = D @ c
        rows += [pad - eye_e, -pad - eye_e]; rhs += [-dc, dc]
    if use_fidelity:                                                         # soft pull towards target: fw * |p - ft|
        cost[off[-1]:] = fw
        I = sparse.csr_matrix((np.ones(T), (np.arange(T), np.arange(T))), shape=(T, nv))
        F = sparse.csr_matrix((np.ones(T), (np.arange(T), off[-1] + np.arange(T))), shape=(T, nv))
        dt = ft - c                                                          # LP variable is d = p - c
        rows += [I - F, -I - F]; rhs += [dt, -dt]
    A = sparse.vstack(rows, format="csr"); bvec = np.concatenate(rhs)
    lo = np.concatenate([-b, np.zeros(nv - T)]); hi = np.concatenate([b, np.full(nv - T, np.inf)])
    if fixed is not None:
        vals, cnt = fixed; lo[:cnt] = hi[:cnt] = np.asarray(vals)[:cnt] - c[:cnt]
    res = linprog(cost, A_ub=A, b_ub=bvec, bounds=np.stack([lo, hi], 1), method="highs")
    if res.status != 0: return gaussian_path(c, 5.0, b)
    return c + res.x[:T]


def l1_path_windowed(c, budget, w=(10.0, 1.0, 100.0), window=600, overlap=60, fidelity=0.0, *,
                     fidelity_target=None, fidelity_weight=None):
    """Long clips: solve overlapping windows, pinning the first 3 samples of each to the previous solution.

    Per-sample fidelity targets/weights are sliced with each window, so pinning and the hard budget keep
    their historical semantics."""
    c = np.asarray(c, np.float64); T = len(c)
    ft, fw = _fidelity_terms(T, c, fidelity, fidelity_target, fidelity_weight)
    if T <= window: return l1_path(c, budget, w, fidelity=fidelity, fidelity_target=ft, fidelity_weight=fw)
    b = np.broadcast_to(np.asarray(budget, np.float64), (T,))
    out = np.zeros(T); s = 0; first = True
    while s < T:
        e = min(T, s + window)
        if first:
            p = l1_path(c[s:e], b[s:e], w, fidelity=fidelity, fidelity_target=ft[s:e], fidelity_weight=fw[s:e])
            out[s:e] = p; first = False
        else:
            p = l1_path(c[s:e], b[s:e], w, fixed=(out[s:s + 3], 3), fidelity=fidelity,
                        fidelity_target=ft[s:e], fidelity_weight=fw[s:e]); out[s + 3:e] = p[3:]
        if e == T: break
        s = e - overlap
    return out


def gaussian_path(c, sigma, budget=None):
    c = np.asarray(c, np.float64)
    p = gaussian_filter1d(c, sigma, axis=0, mode="nearest") if sigma > 0 else c.copy()
    if budget is not None: p = c + np.clip(p - c, -budget, budget)
    return p


def lowpass_path(c, sigma):
    """Zero-phase Gaussian low-pass of ``c`` along time; used only to derive an optional fidelity target.

    This does not replace the L1 smoother: the production planner still solves its L1 objective and hard
    budget, with this path acting as a per-sample soft-fidelity target."""
    c = np.asarray(c, np.float64)
    s = float(sigma)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("low-pass sigma must be finite and > 0")
    if c.ndim != 1:
        raise ValueError("c must be a 1-D path")
    if not np.all(np.isfinite(c)):
        raise ValueError("c must contain only finite values")
    return gaussian_filter1d(c, s, axis=0, mode="nearest")


def butterworth_path(c, cutoff_period_frames, budget=None, order=4):
    """Offline zero-phase low-pass candidate with a specified frequency corner.

    ``cutoff_period_frames`` is the period (in samples) at the single-pass Butterworth -3 dB corner.
    Forward/backward filtering squares the magnitude response (an effective eighth-order magnitude rolloff
    for the default fourth-order one-pass design), so the realized corner is -6 dB. This experimental method
    makes the frequency tradeoff explicit; it does not infer which motion is intended.
    """
    c = np.asarray(c, np.float64)
    period = float(cutoff_period_frames)
    if not np.isfinite(period) or period <= 2.0:
        raise ValueError("cutoff_period_frames must be finite and greater than the 2-frame Nyquist period")
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    if c.ndim != 1:
        raise ValueError("c must be a 1-D path")
    if not np.all(np.isfinite(c)):
        raise ValueError("c must contain only finite values")
    if len(c) < 4:
        p = c.copy()
    else:
        sos = butter(int(order), 1.0 / period, btype="lowpass", fs=1.0, output="sos")
        # scipy's default pad length is short for this order; choose the minimum valid symmetric extension.
        padlen = min(3 * (2 * len(sos) + 1), len(c) - 1)
        p = sosfiltfilt(sos, c, padlen=padlen)
    if budget is not None:
        p = c + np.clip(p - c, -budget, budget)
    return p


def _column_terms(values, k):
    """Select column ``k`` from an optional scalar/1-D/2-D fidelity target or weight array."""
    if values is None: return None
    a = np.asarray(values, np.float64)
    if a.ndim == 0: return float(a)
    if a.ndim == 1: return a
    if a.ndim == 2 and k < a.shape[1]: return a[:, k]
    raise ValueError(f"fidelity column {k} is unavailable in an array of shape {a.shape}")


def smooth_columns(C, mode, sigma, budget, w=(10.0, 1.0, 100.0), fidelity=0.0, *, fidelity_target=None,
                   fidelity_weight=None):
    """Smooth every column of C [T,K]. budget: scalar or [K].

    Modes are ``l1``, ``gauss`` and the experimental offline ``butter`` low-pass. For ``butter``,
    ``sigma`` is interpreted as the single-pass -3 dB cutoff period in frames (the forward/backward response
    is -6 dB there, with an effective eighth-order magnitude rolloff). The frequency-only method cannot
    distinguish intended motion from disturbance. Optional per-sample ``fidelity_target``/``fidelity_weight``
    arrays may be scalar, ``[T]`` (shared by columns) or ``[T,K]``; they apply to the ``l1`` mode only.
    """
    C = np.asarray(C, np.float64); out = np.empty_like(C)
    bud = np.broadcast_to(np.asarray(budget, np.float64), (C.shape[1],))
    for k in range(C.shape[1]):
        if mode == "l1":
            out[:, k] = l1_path_windowed(C[:, k], bud[k], w, fidelity=fidelity,
                                         fidelity_target=_column_terms(fidelity_target, k),
                                         fidelity_weight=_column_terms(fidelity_weight, k))
        elif mode == "gauss":
            out[:, k] = gaussian_path(C[:, k], sigma, bud[k])
        elif mode == "butter":
            out[:, k] = butterworth_path(C[:, k], sigma, bud[k])
        else:
            raise ValueError(f"unknown smoothing mode: {mode}")
    return out
