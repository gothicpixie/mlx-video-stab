"""Camera-path smoothing: L1-optimal (Grundmann et al. 2011) and Gaussian, with a box budget around the raw path."""
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linprog


def _diff_mat(T, k):
    D = sparse.eye(T, format="csr")
    for _ in range(k): D = (D[1:] - D[:-1])
    return D


def l1_path(c, budget, w=(10.0, 1.0, 100.0), fixed=None, fidelity=0.0):
    """argmin_p  w1|Dp|_1 + w2|D^2 p|_1 + w3|D^3 p|_1   s.t. |p - c| <= budget  (c: [T]; budget scalar or [T]).

    The optimum is piecewise static / constant-velocity / constant-acceleration (tripod, dolly, ease in/out).
    `fixed` = (values array, count): pin the first `count` samples (used to chain windows)."""
    c = np.asarray(c, np.float64); T = len(c)
    if T < 5: return c.copy()
    b = np.broadcast_to(np.asarray(budget, np.float64), (T,)).copy()
    Ds = [_diff_mat(T, k) for k in (1, 2, 3)]
    n_e = [T - k for k in (1, 2, 3)]; off = np.cumsum([T] + n_e)             # variable layout: d (T), e1, e2, e3 (+ |d| slack f)
    nv = int(off[-1]) + (T if fidelity > 0 else 0)
    cost = np.zeros(nv)
    for i, wi in enumerate(w): cost[off[i]:off[i + 1]] = wi
    rows, rhs = [], []
    for i, D in enumerate(Ds):
        pad = sparse.hstack([D, sparse.csr_matrix((D.shape[0], nv - T))], format="csr")
        eye_e = sparse.csr_matrix((np.ones(D.shape[0]), (np.arange(D.shape[0]), off[i] + np.arange(D.shape[0]))), shape=(D.shape[0], nv))
        dc = D @ c
        rows += [pad - eye_e, -pad - eye_e]; rhs += [-dc, dc]
    if fidelity > 0:                                                         # soft pull towards the raw path: w0 * |p - c|
        cost[off[-1]:] = fidelity
        I = sparse.csr_matrix((np.ones(T), (np.arange(T), np.arange(T))), shape=(T, nv))
        F = sparse.csr_matrix((np.ones(T), (np.arange(T), off[-1] + np.arange(T))), shape=(T, nv))
        rows += [I - F, -I - F]; rhs += [np.zeros(T), np.zeros(T)]
    A = sparse.vstack(rows, format="csr"); bvec = np.concatenate(rhs)
    lo = np.concatenate([-b, np.zeros(nv - T)]); hi = np.concatenate([b, np.full(nv - T, np.inf)])
    if fixed is not None:
        vals, cnt = fixed; lo[:cnt] = hi[:cnt] = np.asarray(vals)[:cnt] - c[:cnt]
    res = linprog(cost, A_ub=A, b_ub=bvec, bounds=np.stack([lo, hi], 1), method="highs")
    if res.status != 0: return gaussian_path(c, 5.0, b)
    return c + res.x[:T]


def l1_path_windowed(c, budget, w=(10.0, 1.0, 100.0), window=600, overlap=60, fidelity=0.0):
    """Long clips: solve overlapping windows, pinning the first 3 samples of each to the previous solution."""
    c = np.asarray(c, np.float64); T = len(c)
    if T <= window: return l1_path(c, budget, w, fidelity=fidelity)
    b = np.broadcast_to(np.asarray(budget, np.float64), (T,))
    out = np.zeros(T); s = 0; first = True
    while s < T:
        e = min(T, s + window)
        if first: p = l1_path(c[s:e], b[s:e], w, fidelity=fidelity); out[s:e] = p; first = False
        else:
            p = l1_path(c[s:e], b[s:e], w, fixed=(out[s:s + 3], 3), fidelity=fidelity); out[s + 3:e] = p[3:]
        if e == T: break
        s = e - overlap
    return out


def gaussian_path(c, sigma, budget=None):
    c = np.asarray(c, np.float64)
    p = gaussian_filter1d(c, sigma, axis=0, mode="nearest") if sigma > 0 else c.copy()
    if budget is not None: p = c + np.clip(p - c, -budget, budget)
    return p


def smooth_columns(C, mode, sigma, budget, w=(10.0, 1.0, 100.0), fidelity=0.0):
    """Smooth every column of C [T,K]. budget: scalar or [K]."""
    C = np.asarray(C, np.float64); out = np.empty_like(C)
    bud = np.broadcast_to(np.asarray(budget, np.float64), (C.shape[1],))
    for k in range(C.shape[1]):
        out[:, k] = l1_path_windowed(C[:, k], bud[k], w, fidelity=fidelity) if mode == "l1" else gaussian_path(C[:, k], sigma, bud[k])
    return out
