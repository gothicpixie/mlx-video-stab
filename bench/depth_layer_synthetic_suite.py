"""Isolated, exploratory **multi-seed / foreground-geometry** suite over the known-truth two-layer synthetic
diagnostic in :mod:`bench.depth_layer_synthetic`.

This module is a thin, deterministic *harness* around :func:`bench.depth_layer_synthetic.run_experiment`. It
does **not** add a new model, a new metric, a threshold, a winner or any production wiring. Its only job is to
run the existing E09/E10 experiment across a fixed, predeclared grid of **seeds** and **foreground
geometries** and to report the resulting per-cell numbers without pooling across geometries.

Scope caution (read this)
-------------------------
These results are **EXPLORATORY and NOT INFERENTIAL**. The grid is a small, predeclared, deterministic
convenience sample of seeds and foreground rectangles -- not a random sample of any population. The scene is a
coordinate-model construction with known ground truth (see :mod:`bench.depth_layer_synthetic`); its raw depth
values are arbitrary labels and no physical near/far or semantic foreground/background meaning is attached to
any rank. Reporting a median or a min/max across a few seeds is a descriptive summary of those fixed
realisations only. There is **no winner, no significance claim, no threshold sweep, and no production default
change** implied or performed here.

Fixed grid
----------
* Seeds (fixed): ``(0, 1, 7, 42)``.
* Foreground geometries (predeclared, normalized fractions ``(cx, cy, half_w, half_h)``): the default centered
  rectangle ``(0.56, 0.52, 0.20, 0.20)``; an off-center tall rectangle ``(0.32, 0.48, 0.12, 0.35)``; and an
  upper-wide rectangle ``(0.64, 0.35, 0.30, 0.14)``.
* Conditions (fixed, run **separately**, never mixed): ``ideal`` with all E10 controls at zero, and
  ``stress`` with ``flow_noise_px=0.5``, ``depth_error_frac=0.10`` and ``fg_nonaffine_px=1.0``.

For every ``(seed, geometry, condition)`` the module calls :func:`run_experiment` exactly once and keeps that
individual report verbatim under ``cases``. It then summarises, for each ``(geometry, condition, policy)``
cell **separately** (never pooling across geometries, conditions or seeds), the median and min/max of the
held-out target-flow error, the target layer's residual jitter, and the target layer's mean inverse-warp
coverage, plus available/total counts and material-win counts. ``median``/``min``/``max`` are ``None`` when no
finite value is available for that cell, never fabricated.

Determinism, JSON and scope
---------------------------
Everything is float64 NumPy with fixed seeds; :func:`run_suite` is a pure function of its arguments. The
report is path-free and finite-safe (written atomically through :mod:`bench.manifest`). No media is decoded,
no RAFT/MLX/production code is imported, no weights are read, the heavy lock is untouched, and no production
default is changed. Use ``python -m bench.depth_layer_synthetic_suite --out <path>``.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

import numpy as np

from . import manifest as MF
from . import depth_layer_synthetic as D

SCHEMA = "mlxstab.depth_layer_synthetic_suite/1"

# Fixed, predeclared seed list. A small deterministic convenience sample, not a random sample of anything.
SEEDS = (0, 1, 7, 42)

# Predeclared foreground geometries as normalized fractions (cx, cy, half_w, half_h). The first entry is the
# E09/E10 default, so geometry "default_center" is bit-for-bit the existing fixture geometry.
GEOMETRIES = (
    ("default_center", (0.56, 0.52, 0.20, 0.20)),
    ("offcenter_vertical", (0.32, 0.48, 0.12, 0.35)),
    ("upper_wide", (0.64, 0.35, 0.30, 0.14)),
)

# Two fixed conditions, run separately and never pooled. "ideal" is all-zero E10 controls; "stress" turns on
# the three E10 robustness controls at fixed values.
CONDITIONS = (
    ("ideal", {"flow_noise_px": 0.0, "depth_error_frac": 0.0, "fg_nonaffine_px": 0.0}),
    ("stress", {"flow_noise_px": 0.5, "depth_error_frac": 0.10, "fg_nonaffine_px": 1.0}),
)

POLICY_NAMES = D.POLICY_NAMES

LIMITATIONS = (
    "EXPLORATORY, NOT INFERENTIAL: the seed list and geometries are a small predeclared deterministic "
    "convenience sample, not a random sample; medians and min/max describe those fixed realisations only",
    "results are SYNTHETIC: a deterministic coordinate model with known ground-truth layer transforms; they "
    "say nothing about real footage, RAFT accuracy, or metric depth",
    "no geometry, condition or policy is declared a winner; no threshold is selected or swept and no "
    "significance test is performed",
    "summaries are reported per (geometry, condition, policy) cell and are NEVER pooled across geometries, "
    "conditions or policies",
    "the stress condition is a synthetic corruption of the coordinate model, not a calibrated model of real "
    "sensor noise, depth errors or non-rigid motion",
    "no production wiring or default change is implied; this harness only re-runs the existing synthetic "
    "diagnostic across a fixed grid",
)


def _finite(values: Sequence[Optional[float]]) -> list:
    """Finite floats from ``values`` (dropping ``None``/non-finite); empty list when none survive."""
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return out


def _median_range(values: Sequence[Optional[float]]) -> dict:
    """Finite-safe ``{median, min, max, n}`` over ``values``; metrics are ``None`` when no finite value exists.

    This is a *descriptive* summary of the given realisations only -- no inference, no threshold, no winner.
    """
    fin = _finite(values)
    if not fin:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": int(len(fin)), "median": float(np.median(fin)),
            "min": float(np.min(fin)), "max": float(np.max(fin))}


def _policy_target_metrics(report: dict, policy_name: str) -> Optional[dict]:
    """Target-layer metrics for one policy from one report, or ``None`` when unavailable.

    Reads the policy's held-out target-flow error (``target_error.frame_median_median``) and, when the policy's
    downstream block is available, the **target layer's** residual jitter and mean inverse-warp coverage. Every
    field is passed through finite-safe aggregation downstream, so a ``None`` metric is never fabricated.
    """
    for p in report.get("policies", []):
        if p.get("policy") != policy_name:
            continue
        if not p.get("available"):
            return {"available": False, "target_error_median_px": None,
                    "target_layer_jitter_px": None, "target_layer_coverage_frac": None,
                    "material_win_frames": 0, "comparable_frames": 0}
        te = p.get("target_error") or {}
        ds = p.get("downstream")
        target_layer = p.get("target_layer")
        jitter = None
        coverage = None
        if ds is not None and target_layer is not None and 0 <= int(target_layer) < len(ds.get("layers", [])):
            layer = ds["layers"][int(target_layer)]
            jitter = layer.get("residual_jitter_px")
            coverage = layer.get("coverage_frac_mean")
        return {
            "available": True,
            "target_error_median_px": te.get("frame_median_median"),
            "target_layer_jitter_px": jitter,
            "target_layer_coverage_frac": coverage,
            "material_win_frames": int(te.get("material_win_frames") or 0),
            "comparable_frames": int(te.get("comparable_frames") or 0),
        }
    return None


def _summarize_cell(per_seed_metrics: Sequence[Optional[dict]]) -> dict:
    """Summarise one ``(geometry, condition, policy)`` cell across its seeds, without pooling anything else.

    Reports the median and min/max of the held-out target error, target-layer residual jitter and target-layer
    mean inverse-warp coverage, plus available/total counts and the count of seeds with at least one material
    win. Only this cell's seeds are aggregated.
    """
    total = len(per_seed_metrics)
    available = [m for m in per_seed_metrics if m is not None and m.get("available")]
    return {
        "total_cases": int(total),
        "available_cases": int(len(available)),
        "target_error_median_px": _median_range([m["target_error_median_px"] for m in available]),
        "target_layer_jitter_px": _median_range([m["target_layer_jitter_px"] for m in available]),
        "target_layer_coverage_frac": _median_range([m["target_layer_coverage_frac"] for m in available]),
        "material_win_count": int(sum(1 for m in available if m.get("material_win_frames", 0) > 0)),
        "material_win_frames_total": int(sum(m.get("material_win_frames", 0) for m in available)),
        "comparable_frames_total": int(sum(m.get("comparable_frames", 0) for m in available)),
    }


def run_suite(seed_list: Sequence[int] = SEEDS,
              geometries: Sequence = GEOMETRIES,
              conditions: Sequence = CONDITIONS,
              *, base: Optional[dict] = None) -> dict:
    """Run the fixed grid of ``(seed, geometry, condition)`` cases and summarise per cell. Pure and deterministic.

    ``base`` supplies extra :class:`~bench.depth_layer_synthetic.SyntheticSpec` keyword arguments shared by every
    case (e.g. ``T``/``W``/``H`` for a cheaper test); it defaults to the module's own defaults. Individual
    per-seed reports are preserved under ``cases``; summaries are keyed by ``geometry|condition|policy`` and
    never pool across geometries, conditions or policies. No threshold is selected and no winner is declared.
    """
    base = dict(base or {})
    cases = []
    # per-cell key -> list of per-seed metric dicts (one per seed, in seed order)
    cell_metrics: dict = {}
    for gname, fg_rect in geometries:
        for cname, controls in conditions:
            for seed in seed_list:
                spec = D.SyntheticSpec(seed=int(seed), fg_rect=tuple(fg_rect), **controls, **base)
                report = D.run_experiment(spec)
                cases.append({
                    "geometry": gname,
                    "fg_rect": list(spec.fg_rect),
                    "condition": cname,
                    "controls": dict(controls),
                    "seed": int(seed),
                    "schema": report["schema"],
                    "spec": report["spec"],
                    "policies": report["policies"],
                    "baseline": report["baseline"],
                    "corruption": report["corruption"],
                })
                for pname in POLICY_NAMES:
                    key = f"{gname}|{cname}|{pname}"
                    cell_metrics.setdefault(key, []).append(_policy_target_metrics(report, pname))

    summaries = {}
    for gname, _ in geometries:
        for cname, _ in conditions:
            for pname in POLICY_NAMES:
                key = f"{gname}|{cname}|{pname}"
                summaries[key] = {
                    "geometry": gname, "condition": cname, "policy": pname,
                    **_summarize_cell(cell_metrics.get(key, [])),
                }

    return {
        "schema": SCHEMA,
        "seeds": [int(s) for s in seed_list],
        "geometries": [{"name": n, "fg_rect": list(r)} for n, r in geometries],
        "conditions": [{"name": n, "controls": dict(c)} for n, c in conditions],
        "grid": {
            "n_seeds": len(seed_list),
            "n_geometries": len(geometries),
            "n_conditions": len(conditions),
            "n_policies": len(POLICY_NAMES),
            "n_cases": len(cases),
        },
        "cases": cases,
        "summaries": summaries,
        "notes": {
            "synthetic_only": True,
            "exploratory_only": True,
            "not_inferential": True,
            "no_media_decoded": True,
            "no_production_code_imported": True,
            "winner_declared": False,
            "threshold_selected": False,
            "pooled_across_geometries": False,
        },
        "limitations": list(LIMITATIONS),
    }


def dumps_report(report: dict) -> str:
    """Deterministic JSON serialization of a suite report, finite-safe and path-free."""
    return MF.dumps_json(report)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.depth_layer_synthetic_suite",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", type=int, default=10, help="frames per case (>= 3)")
    p.add_argument("--W", type=int, default=48, help="frame width per case (>= 8)")
    p.add_argument("--H", type=int, default=36, help="frame height per case (>= 8)")
    p.add_argument("--out", default=None, help="output JSON path (atomic write)")
    p.add_argument("--dry-run", action="store_true", help="print resolved grid and exit")
    return p


def run(opts: argparse.Namespace, log=print) -> dict:
    """Run the fixed grid (optionally with a cheaper T/W/H) and optionally write the atomic JSON report."""
    report = run_suite(base={"T": opts.T, "W": opts.W, "H": opts.H})
    if opts.out:
        MF.write_manifest(opts.out, report)
    return report


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.dry_run:
        print(json.dumps({"seeds": list(SEEDS),
                          "geometries": [{"name": n, "fg_rect": list(r)} for n, r in GEOMETRIES],
                          "conditions": [{"name": n, "controls": dict(c)} for n, c in CONDITIONS],
                          "grid": {"n_seeds": len(SEEDS), "n_geometries": len(GEOMETRIES),
                                   "n_conditions": len(CONDITIONS), "n_policies": len(POLICY_NAMES),
                                   "n_cases": len(SEEDS) * len(GEOMETRIES) * len(CONDITIONS)},
                          "T": a.T, "W": a.W, "H": a.H, "out": a.out},
                         indent=1, sort_keys=True))
        return 0
    try:
        report = run(a, log=lambda m: print(m, file=sys.stderr))
    except Exception as e:
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    print(json.dumps({"status": "ok", "out": a.out, "schema": report["schema"],
                      "grid": report["grid"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
