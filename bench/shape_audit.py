"""CPU-only E02 rigid-line shape audit (NumPy; analytic scope only).

Compares the current fixed-grid residual map (``composition_audit.naive_from_plan``) with the Stage 1
composed candidate (``composition_candidate.candidate_source_coords(..., highpass=False)``) on analytic
fixtures, using an output-space straightness metric. A projective homography maps lines to lines, so a
fixture whose true per-pair motions are projective homographies must stay straight; the fixed-grid residual
sum can bend it.

Scope: analytic geometry, pure NumPy/CPU. No media, FFmpeg, MLX kernels, production wiring, or changed
defaults. Nothing here is a real-footage quality claim. Run with ``python -m bench.shape_audit``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlxstab import geometry as G

from . import composition_audit as CA
from . import composition_candidate as CC
from .manifest import json_safe

SCHEMA = "mlxstab.shape_audit/1"
_SPAN_TOL = 1e-9  # px; a mapped line with less extent than this is treated as collapsed
_DEV_KEYS = ("span_px", "dev_max_px", "dev_p95_px", "dev_rms_px", "dev_max_over_span", "dev_p95_over_span")


# --------------------------------------------------------------------------------------
# straightness metric
# --------------------------------------------------------------------------------------

def sample_line(p0, p1, n):
    """``n`` points on the segment ``p0 -> p1``, inclusive of both endpoints, shape ``[n, 2]``."""
    p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
    t = np.linspace(0.0, 1.0, int(n))[:, None]
    return p0[None, :] * (1.0 - t) + p1[None, :] * t


def _dev_none(out):
    for key in _DEV_KEYS:
        out.setdefault(key, None)
    return out


def best_fit_line(pts):
    """Perpendicular deviation of ``pts [N,2]`` from their best-fit line (SVD/PCA), in pixels.

    Centres the points, takes the dominant singular direction as the line, and measures the absolute
    perpendicular residual. Reports max / p95 / RMS deviation, the along-line span, and the deviation
    normalized by that span. Bad inputs are rejected, not hidden: non-finite coordinates, fewer than two
    points, a failed SVD and a collapsed span (``<= _SPAN_TOL`` px) all set ``ok=False`` and a reason, and the
    deviation summaries become ``None`` so a collapsed line cannot look perfectly straight.
    """
    pts = np.asarray(pts, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be [N,2]")
    n = int(len(pts))
    out = {"n": n, "ok": False, "reasons": [], "nonfinite_frac": None}
    if n < 2:
        out["reasons"].append("fewer_than_two_points")
        return _dev_none(out)
    finite = np.isfinite(pts).all(axis=1)
    out["nonfinite_frac"] = float((~finite).mean())
    if not finite.all():
        out["reasons"].append("nonfinite_coordinates")
        return _dev_none(out)
    cen = pts - pts.mean(0)
    try:
        _, _s, vt = np.linalg.svd(cen, full_matrices=False)
    except np.linalg.LinAlgError:
        out["reasons"].append("svd_failed")
        return _dev_none(out)
    direction, normal = vt[0], vt[1]
    proj = cen @ direction
    perp = np.abs(cen @ normal)
    span = float(proj.max() - proj.min())
    out["span_px"] = span
    if span <= _SPAN_TOL:
        out["reasons"].append("degenerate_span")
        return _dev_none(out)
    out.update({
        "dev_max_px": float(perp.max()),
        "dev_p95_px": float(np.percentile(perp, 95)),
        "dev_rms_px": float(np.sqrt((perp ** 2).mean())),
        "dev_max_over_span": float(perp.max() / span),
        "dev_p95_over_span": float(np.percentile(perp, 95) / span),
        "ok": True,
    })
    return out


def map_line_deviation(map_fn, p0, p1, n):
    """Sample the output-space line ``p0 -> p1``, map it with ``map_fn``, and measure straightness."""
    res = best_fit_line(np.asarray(map_fn(sample_line(p0, p1, n)), np.float64))
    res["input_span_px"] = float(np.linalg.norm(np.asarray(p1, np.float64) - np.asarray(p0, np.float64)))
    return res


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def projective(a, b):
    """Homography ``p -> p / (1 + a*x + b*y)``: genuinely projective (non-affine) when ``a`` or ``b`` != 0."""
    Hm = np.eye(3); Hm[2, 0] = float(a); Hm[2, 1] = float(b)
    return Hm


def _residual_mesh(Hm, W, H, gc, gr):
    """Vertex samples of ``Hm(p) - p`` (the mesh the fit interface would report for a true homography)."""
    return CA.field_mesh(W, H, gc, gr, lambda p: G.apply_h(Hm, p) - p)


@dataclass
class _FitLike:
    H: np.ndarray
    mesh: np.ndarray
    rows: object = None


@dataclass
class _MotionLike:
    W: int
    H: int
    fits: list
    stride: int = 1
    n_frames: int = 0


def _plan_and_motion(pairs, W, H, T, gc, gr, name, description):
    """Production pinned plan (``CA.build_plan``) plus a duck-typed Motion over the same pairs."""
    fx = CA.Fixture(name, description, W, H, T, pairs, (gc, gr))
    plan, cfg = CA.build_plan(fx)
    mot = _MotionLike(W, H, [_FitLike(np.asarray(p.H, np.float64).copy(),
                                      np.asarray(p.mesh, np.float64).copy(), None) for p in pairs], 1, T)
    return fx, plan, cfg, mot


def _default_line(W, H):
    return np.array([5.0, 5.0]), np.array([W - 6.0, H - 6.0])


def _compose_matrices(homs):
    """Matrix product ``H_{t-1} @ ... @ H_0`` (apply ``H_0`` first); identity for an empty list."""
    M = np.eye(3)
    for Hm in homs:
        M = np.asarray(Hm, np.float64) @ M
    return M


def _max_step(values):
    """Max absolute step-to-step change in a scalar series; ``None`` if any entry is ``None``."""
    vals = list(values)
    if any(v is None for v in vals):
        return None
    if len(vals) < 2:
        return 0.0
    return float(max(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))))


def metric_controls(W=64, H=48, n=201):
    """Identity and single-projective-homography controls (straight) and a bowed-field sensitivity control."""
    p0, p1 = _default_line(W, H)
    Hp = projective(2.5e-3, -1.5e-3)

    def bowed(p):
        s = (p[:, 0] - p0[0]) / (p1[0] - p0[0])
        return p + np.stack([np.zeros(len(p)), 6.0 * np.sin(np.pi * s)], 1)

    return {
        "identity": {"expect_straight": True, "deviation": map_line_deviation(lambda p: p, p0, p1, n)},
        "projective_homography": {"expect_straight": True,
                                  "deviation": map_line_deviation(lambda p: G.apply_h(Hp, p), p0, p1, n)},
        "bowed_field": {"expect_straight": False, "amplitude_px": 6.0,
                        "deviation": map_line_deviation(bowed, p0, p1, n)},
        "line": {"p0": p0.tolist(), "p1": p1.tolist(), "n": int(n)},
    }


def pipeline_projective_control(W=64, H=48, gc=16, gr=12, n=201):
    """Projective globals with zero residuals: both fixed-grid and candidate maps are the same homography."""
    P0, P1 = projective(1.5e-3, -1.0e-3), projective(-1.2e-3, 1.4e-3)
    pairs = [CA.PairMotion(P0, np.zeros((gr + 1, gc + 1, 2))),
             CA.PairMotion(P1, np.zeros((gr + 1, gc + 1, 2)))]
    _, plan, cfg, mot = _plan_and_motion(pairs, W, H, 3, gc, gr, "pipeline_projective_control",
                                         "projective globals, zero residuals")
    p0, p1 = _default_line(W, H)
    return {
        "fixed_grid": map_line_deviation(lambda p: CA.naive_from_plan(plan, 2, p, W, H), p0, p1, n),
        "candidate": map_line_deviation(
            lambda p: CC.candidate_source_coords(mot, plan, cfg, 2, p, highpass=False), p0, p1, n),
    }


def two_step_projective_fixture(W=64, H=48, gc=48, gr=36, n=401):
    """Two-step analytic fixture with an independent matrix-product ground truth and a frame time series.

    Each true per-pair motion is a full projective homography ``P_k``, reported through the fit interface as
    ``Fit.H = I`` plus the vertex samples of ``P_k(p) - p``. The oracle applies the **matrix product**
    ``P_{t-1} @ ... @ P_0`` once (not nested application); the true composition is a homography and maps lines
    to lines. The candidate composes the pair maps at their source positions; the fixed grid adds the
    cumulative residual at the current global coordinate and bends the line. The line is horizontal through
    mid-frame so it samples the perspective gradient along ``x`` maximally.

    The time series records ``dev_max_px`` for ground truth, candidate and fixed grid at frames 0, 1, 2, plus
    the max step-to-step change, so the audit measures temporal shape variation and not only final-frame
    bending.
    """
    Ps = [projective(7.5e-3, -4.5e-3), projective(-6.0e-3, 6.0e-3)]
    pairs = [CA.PairMotion(np.eye(3), _residual_mesh(P, W, H, gc, gr)) for P in Ps]
    _, plan, cfg, mot = _plan_and_motion(pairs, W, H, len(Ps) + 1, gc, gr, "two_step_projective",
                                         "two projective pair motions represented as I + residual mesh")
    p0, p1 = np.array([4.0, H / 2.0]), np.array([W - 5.0, H / 2.0])
    frames = list(range(len(Ps) + 1))
    pts = sample_line(p0, p1, n)
    gt_series, cand_series, naive_series = {}, {}, {}
    cand_vs, naive_vs = [], []
    for t in frames:
        M = _compose_matrices(Ps[:t])
        gt_map = lambda p, M=M: G.apply_h(M, p)
        cand_map = lambda p, t=t: CC.candidate_source_coords(mot, plan, cfg, t, p, highpass=False)
        naive_map = lambda p, t=t: CA.naive_from_plan(plan, t, p, W, H)
        gt_series[t] = map_line_deviation(gt_map, p0, p1, n)
        cand_series[t] = map_line_deviation(cand_map, p0, p1, n)
        naive_series[t] = map_line_deviation(naive_map, p0, p1, n)
        gt_pts = gt_map(pts)
        cand_vs.append(float(np.linalg.norm(cand_map(pts) - gt_pts, axis=1).max()))
        naive_vs.append(float(np.linalg.norm(naive_map(pts) - gt_pts, axis=1).max()))

    def _devs(series):
        return [series[t]["dev_max_px"] for t in frames]

    gt_dev, cand_dev, naive_dev = _devs(gt_series), _devs(cand_series), _devs(naive_series)
    mesh_points = [v for v in cand_vs if v is not None]
    return {
        "id": "two_step_projective",
        "description": "P1 @ P0 (matrix product) is a homography; candidate composes, fixed grid sums.",
        "config": {"W": W, "H": H, "mesh_cells": [gc, gr], "n_samples": int(n), "pairs": len(Ps),
                   "frames": frames},
        "globals": {"P0": Ps[0].tolist(), "P1": Ps[1].tolist()},
        "line": {"p0": p0.tolist(), "p1": p1.tolist()},
        "time_series": {
            "frames": frames,
            "ground_truth_dev_max_px": gt_dev,
            "candidate_dev_max_px": cand_dev,
            "fixed_grid_dev_max_px": naive_dev,
            "ground_truth_max_step_px": _max_step(gt_dev),
            "candidate_max_step_px": _max_step(cand_dev),
            "fixed_grid_max_step_px": _max_step(naive_dev),
            "candidate_vs_ground_truth_max_px": cand_vs,
            "fixed_grid_vs_ground_truth_max_px": naive_vs,
            "candidate_mesh_discretization_max_px": max(mesh_points) if mesh_points else None,
        },
        "ground_truth": gt_series[frames[-1]],
        "candidate": cand_series[frames[-1]],
        "fixed_grid": naive_series[frames[-1]],
        "candidate_vs_ground_truth_max_px": cand_vs[-1],
        "fixed_grid_vs_ground_truth_max_px": naive_vs[-1],
    }


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

METRIC_DOC = {
    "name": "best-fit-line straightness",
    "units": "px",
    "method": "centre the mapped output-line samples, take the dominant SVD direction as the line, and "
              "measure the absolute perpendicular residual; span is the along-line extent",
    "reports": list(_DEV_KEYS),
    "rejects": ["nonfinite_coordinates", "degenerate_span", "fewer_than_two_points", "svd_failed"],
}

SCOPE_NOTE = (
    "Analytic-only, CPU NumPy. No media, FFmpeg, MLX kernels, production wiring or changed defaults. "
    "Does not establish real-footage quality; the two-step fixture is a controlled mechanism check."
)


def run_shape_audit(out_dir=None, *, n=401):
    """Run the metric controls and the two-step fixture; return a JSON-safe report (optionally write it).

    Invalid measured lines are recorded (``ok=False``/``None`` values) and never compared: the conclusion then
    reports an ``unknown`` separation with reasons instead of raising or silently treating ``None`` as a value.
    """
    controls = metric_controls(n=n)
    pipeline = pipeline_projective_control(n=n)
    fixture = two_step_projective_fixture(n=n)
    ts = fixture["time_series"]
    gt_f, cand_f, naive_f = fixture["ground_truth"], fixture["candidate"], fixture["fixed_grid"]

    reasons = []
    final_ok = all(isinstance(d.get("dev_max_px"), (int, float)) and d.get("ok") for d in (gt_f, cand_f, naive_f))
    if not final_ok:
        reasons.append("final_frame_line_invalid")
    series_ok = all(v is not None for v in
                    ts["candidate_dev_max_px"] + ts["fixed_grid_dev_max_px"] + ts["ground_truth_dev_max_px"])
    if not series_ok:
        reasons.append("time_series_line_invalid")

    separation = bool(cand_f["dev_max_px"] < naive_f["dev_max_px"]) if final_ok else None
    cand_step, naive_step = ts["candidate_max_step_px"], ts["fixed_grid_max_step_px"]
    mesh_err = ts["candidate_mesh_discretization_max_px"]
    cand_bounded = bool(cand_step <= mesh_err + 1e-12) if (series_ok and mesh_err is not None) else None
    fixed_exceeds = bool(naive_step > 0.5 and naive_step > 100.0 * (cand_step or 0.0)) if series_ok else None

    conclusion = {
        "valid": bool(final_ok and series_ok),
        "separation": ("candidate" if separation else "fixed_grid") if separation is not None else "unknown",
        "candidate_straighter_than_fixed_grid": separation,
        "candidate_temporal_variation_bounded": cand_bounded,
        "fixed_grid_temporal_variation_exceeds_control": fixed_exceeds,
        "reasons": reasons,
        "candidate_dev_max_px": cand_f.get("dev_max_px"),
        "fixed_grid_dev_max_px": naive_f.get("dev_max_px"),
        "ground_truth_dev_max_px": gt_f.get("dev_max_px"),
        "candidate_max_step_px": cand_step,
        "fixed_grid_max_step_px": naive_step,
        "ground_truth_max_step_px": ts["ground_truth_max_step_px"],
        "candidate_mesh_discretization_max_px": mesh_err,
        "candidate_vs_ground_truth_max_px": fixture["candidate_vs_ground_truth_max_px"],
        "fixed_grid_vs_ground_truth_max_px": fixture["fixed_grid_vs_ground_truth_max_px"],
        "note": ("analytic fixture only; a positive result is a mechanism check, not a real-footage quality "
                 "claim, and no production change is proposed"),
    }
    report = {
        "schema": SCHEMA,
        "scope": SCOPE_NOTE,
        "metric": METRIC_DOC,
        "controls": controls,
        "pipeline_control": pipeline,
        "fixture": fixture,
        "conclusion": conclusion,
    }
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(json_safe(report), indent=1, sort_keys=True, allow_nan=False)
        (out / "shape_audit.json").write_text(payload, encoding="utf-8")
        report["out_dir"] = str(out)
    return report


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="CPU-only E02 rigid-line shape audit (analytic; no media).")
    ap.add_argument("--out-dir", default=None, help="write shape_audit.json here")
    ap.add_argument("--samples", type=int, default=401, help="output-line samples per map")
    args = ap.parse_args(argv)
    rep = run_shape_audit(args.out_dir, n=args.samples)
    c = rep["conclusion"]
    print(json.dumps(json_safe({"schema": rep["schema"], "scope": rep["scope"], "conclusion": c}),
                     indent=1, sort_keys=True, allow_nan=False))
    fi = rep["fixture"]
    ts = fi["time_series"]
    print(f"frames={ts['frames']}")
    print(f"ground_truth dev_max px={ts['ground_truth_dev_max_px']}")
    print(f"candidate    dev_max px={ts['candidate_dev_max_px']} step={ts['candidate_max_step_px']:.6g}")
    print(f"fixed_grid   dev_max px={ts['fixed_grid_dev_max_px']} step={ts['fixed_grid_max_step_px']:.6g}")
    print(f"candidate vs ground truth max={fi['candidate_vs_ground_truth_max_px']:.6g} px; "
          f"fixed grid vs ground truth max={fi['fixed_grid_vs_ground_truth_max_px']:.6g} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
