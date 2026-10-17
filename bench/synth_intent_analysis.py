"""Summarize the paired-seed synthetic intent factorial without treating cells as replicates.

The seed is the experimental unit. Main-cell measurements are averaged within each intent mode and seed, then
stochastic-minus-smooth paired contrasts are summarized across complete seeds. Calibration anchors are excluded.
For shake residuals, the source metric is 20*log10(output residual RMS / input shake RMS): negative values mean
attenuation, and lower values mean stronger shake rejection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median, stdev

import numpy as np
from scipy.stats import t as student_t


METRICS = {
    "intent_x_rmse_px": ("intent_retention", "translation_x_px", "rmse"),
    "intent_y_rmse_px": ("intent_retention", "translation_y_px", "rmse"),
    "intent_rotation_rmse_deg": ("intent_retention", "rotation_deg", "rmse"),
    "shake_x_residual_db": ("shake_rejection", "translation_x_px", "attenuation_db"),
    "shake_y_residual_db": ("shake_rejection", "translation_y_px", "attenuation_db"),
    "shake_rotation_residual_db": ("shake_rejection", "rotation_deg", "attenuation_db"),
}
EXPECTED_CELL_SUFFIXES = {
    f"{layer}__{subject}"
    for layer in ("far_only", "parallax")
    for subject in ("absent", "static", "moving")
}


def _metric(run: dict, spec: tuple[str, str, str]) -> float | None:
    section, axis, key = spec
    value = run.get("far_camera_metrics")
    if not isinstance(value, dict):
        return None
    value = value.get(section)
    if not isinstance(value, dict):
        return None
    value = value.get(axis)
    if not isinstance(value, dict):
        return None
    result = value.get(key)
    if result is None:
        return None
    result = float(result)
    return result if math.isfinite(result) else None


def _bootstrap_mean_ci(values: list[float], *, draws: int = 20_000, seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    samples = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(samples), size=(draws, len(samples)))
    means = np.mean(samples[indices], axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _t_mean_ci(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    mean = fmean(values)
    half_width = float(student_t.ppf(0.975, df=len(values) - 1)) * stdev(values) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def _summarize(rows: list[dict]) -> dict:
    deltas = [row["delta_stochastic_minus_smooth"] for row in rows]
    return {
        "n_seeds": len(deltas),
        "mean_delta_stochastic_minus_smooth": fmean(deltas) if deltas else None,
        "median_delta_stochastic_minus_smooth": median(deltas) if deltas else None,
        "sd_between_seeds": stdev(deltas) if len(deltas) > 1 else None,
        "t_95pct_ci_for_mean": _t_mean_ci(deltas),
        "bootstrap_95pct_ci_for_mean": _bootstrap_mean_ci(deltas, seed=0) if deltas else None,
        "negative_seed_deltas": sum(delta < 0 for delta in deltas),
        "positive_seed_deltas": sum(delta > 0 for delta in deltas),
        "per_seed": rows,
    }


def analyze(screen: dict) -> dict:
    """Return per-seed contrasts and seed-level summaries for a factorial checkpoint."""
    runs = screen.get("runs")
    if not isinstance(runs, list):
        raise ValueError("screen must contain a runs list")
    design = screen.get("design") or {}
    configured_cells = design.get("cells_per_seed") or []
    expected_cell_suffixes = {
        str(cell["id"]).split("__", 1)[1]
        for cell in configured_cells
        if cell.get("kind") == "main" and "__" in str(cell.get("id", ""))
    } or EXPECTED_CELL_SUFFIXES

    main: dict[int, dict[str, list[dict]]] = {}
    for run in runs:
        if run.get("kind") != "main":
            continue
        seed = int(run["seed"])
        cell_id = str(run["cell_id"])
        mode = cell_id.split("__", 1)[0]
        if mode not in {"smooth", "stochastic"}:
            continue
        main.setdefault(seed, {"smooth": [], "stochastic": []})[mode].append(run)

    per_seed: dict[str, list[dict]] = {name: [] for name in METRICS}
    strata_per_seed: dict[str, dict[str, dict[str, list[dict]]]] = {
        name: {"by_layer": {}, "by_subject": {}, "by_cell": {}} for name in METRICS
    }
    incomplete_seeds: list[dict] = []
    for seed, modes in sorted(main.items()):
        design_mismatch = any(
            {str(run["cell_id"]).split("__", 1)[1] for run in rows} != expected_cell_suffixes
            or len(rows) != 6
            for rows in modes.values()
        )
        if design_mismatch:
            incomplete_seeds.append({"seed": seed, "smooth_cells": len(modes["smooth"]),
                                     "stochastic_cells": len(modes["stochastic"]),
                                     "reason": "main_cell_design_mismatch"})
            continue
        if any(run.get("status") != "success" for rows in modes.values() for run in rows):
            incomplete_seeds.append({"seed": seed, "smooth_cells": len(modes["smooth"]),
                                     "stochastic_cells": len(modes["stochastic"]),
                                     "reason": "missing_or_failed_main_cells"})
            continue
        seed_results = {}
        for name, spec in METRICS.items():
            by_mode: dict[str, float] = {}
            for mode in ("smooth", "stochastic"):
                values = [_metric(run, spec) for run in modes[mode]]
                if any(value is None for value in values):
                    break
                by_mode[mode] = fmean(value for value in values if value is not None)
            if len(by_mode) == 2:
                seed_results[name] = {
                    "seed": seed,
                    "smooth_mean": by_mode["smooth"],
                    "stochastic_mean": by_mode["stochastic"],
                    "delta_stochastic_minus_smooth": by_mode["stochastic"] - by_mode["smooth"],
                }
        if len(seed_results) != len(METRICS):
            incomplete_seeds.append({"seed": seed, "smooth_cells": 6, "stochastic_cells": 6,
                                     "reason": "missing_truth_metrics"})
            continue
        for name, row in seed_results.items():
            per_seed[name].append(row)
            spec = METRICS[name]
            for facet, values in (("by_layer", ("far_only", "parallax")),
                                  ("by_subject", ("absent", "static", "moving"))):
                for level in values:
                    smooth_values = []
                    stochastic_values = []
                    for mode, destination in (("smooth", smooth_values), ("stochastic", stochastic_values)):
                        for run in modes[mode]:
                            parts = str(run["cell_id"]).split("__")
                            observed = parts[1] if facet == "by_layer" else parts[2]
                            value = _metric(run, spec)
                            if observed == level and value is not None:
                                destination.append(value)
                    if smooth_values and stochastic_values:
                        strata_per_seed[name][facet].setdefault(level, []).append({
                            "seed": seed,
                            "smooth_mean": fmean(smooth_values),
                            "stochastic_mean": fmean(stochastic_values),
                            "delta_stochastic_minus_smooth": fmean(stochastic_values) - fmean(smooth_values),
                        })
            for run_smooth in modes["smooth"]:
                cell_suffix = "__".join(str(run_smooth["cell_id"]).split("__")[1:])
                run_stochastic = next((run for run in modes["stochastic"]
                                       if str(run["cell_id"]).split("__", 1)[1] == cell_suffix), None)
                smooth_value = _metric(run_smooth, spec)
                stochastic_value = _metric(run_stochastic, spec) if run_stochastic else None
                if smooth_value is not None and stochastic_value is not None:
                    strata_per_seed[name]["by_cell"].setdefault(cell_suffix, []).append({
                        "seed": seed, "smooth_mean": smooth_value, "stochastic_mean": stochastic_value,
                        "delta_stochastic_minus_smooth": stochastic_value - smooth_value,
                    })

    summaries = {}
    for name, rows in per_seed.items():
        summaries[name] = _summarize(rows)
        summaries[name]["stratified"] = {
            facet: {level: _summarize(level_rows) for level, level_rows in levels.items()}
            for facet, levels in strata_per_seed[name].items()
        }
        cell_rows = strata_per_seed[name]["by_cell"]
        by_seed = {}
        for suffix, cell_values in cell_rows.items():
            layer, subject = suffix.split("__", 1)
            for row in cell_values:
                by_seed.setdefault(row["seed"], {}).setdefault((layer, subject),
                    row["delta_stochastic_minus_smooth"])

        def mean_cell_delta(seed: int, layer: str, subjects: tuple[str, ...]) -> float:
            cells = by_seed[seed]
            return fmean(cells[(layer, subject)] for subject in subjects)

        layers = ("far_only", "parallax")
        subjects = ("absent", "static", "moving")
        interaction_rows = {
            "phase_by_layer": [],
            "phase_by_subject_static_vs_absent": [],
            "phase_by_subject_moving_vs_absent": [],
            "phase_by_layer_by_subject_static_vs_absent": [],
            "phase_by_layer_by_subject_moving_vs_absent": [],
        }
        required_cells = {(layer, subject) for layer in layers for subject in subjects}
        for seed, cells in sorted(by_seed.items()):
            if not required_cells.issubset(cells):
                continue
            layer_effect = {
                layer: mean_cell_delta(seed, layer, subjects) for layer in layers
            }
            subject_effect = {
                subject: fmean(cells[(layer, subject)] for layer in layers) for subject in subjects
            }
            layer_subject_effect = {
                subject: cells[("parallax", subject)] - cells[("far_only", subject)]
                for subject in subjects
            }
            estimates = {
                "phase_by_layer": layer_effect["parallax"] - layer_effect["far_only"],
                "phase_by_subject_static_vs_absent": subject_effect["static"] - subject_effect["absent"],
                "phase_by_subject_moving_vs_absent": subject_effect["moving"] - subject_effect["absent"],
                "phase_by_layer_by_subject_static_vs_absent": (
                    layer_subject_effect["static"] - layer_subject_effect["absent"]
                ),
                "phase_by_layer_by_subject_moving_vs_absent": (
                    layer_subject_effect["moving"] - layer_subject_effect["absent"]
                ),
            }
            for contrast, estimate in estimates.items():
                interaction_rows[contrast].append({
                    "seed": seed, "delta_stochastic_minus_smooth": estimate,
                })
        summaries[name]["factorial_interactions"] = {
            contrast: _summarize(contrast_rows)
            for contrast, contrast_rows in interaction_rows.items()
        }

    configured_seeds = {int(seed) for seed in design.get("seeds", [])}
    for seed in sorted(configured_seeds - set(main)):
        incomplete_seeds.append({"seed": seed, "smooth_cells": 0, "stochastic_cells": 0,
                                 "reason": "no_main_cells"})
    incomplete_seeds.sort(key=lambda row: row["seed"])

    return {
        "schema": "synth-intent-factorial-analysis-v1",
        "experimental_unit": "seed",
        "within_seed_aggregation": "unweighted mean over six main cells per intent mode",
        "contrast": "stochastic minus smooth; anchors excluded",
        "primary_interval": "two-sided Student-t interval over independent seed-level contrasts",
        "bootstrap_interval": "percentile seed bootstrap is a sensitivity check; coarse and unstable at small n",
        "interaction_contrasts": (
            "per-seed differences-in-differences for phase x layer, phase x subject, and phase x layer x subject"
        ),
        "interpretation": {
            "intent_rmse": "negative delta means lower intent-band error for stochastic",
            "shake_residual_db": "20*log10(output residual / input shake); negative values mean attenuation; negative delta means stronger rejection for stochastic",
        },
        "complete_seeds": sorted(row["seed"] for row in per_seed["intent_x_rmse_px"]),
        "incomplete_seeds": incomplete_seeds,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path, help="screen.json written by synth_intent_factorial")
    parser.add_argument("--out", type=Path, help="write JSON report here; otherwise print to stdout")
    args = parser.parse_args()
    report = analyze(json.loads(args.screen.read_text()))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
