"""Sequential, resumable paired-seed screen for synthetic camera intent.

The main design is intent phase (smooth/stochastic) x layer motion (far-only/parallax) x subject
(absent/world-static/world-moving), with three calibration anchors per seed. Every cell reuses its seed's scene,
intent, and camera-shake streams. The runner writes one manifest, production plan and truth sidecar per cell;
``synth_intent_metrics`` then evaluates each output against visible-layer camera truth. No aggregate winner is
computed: the report preserves every cell and failure for later preregistered contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_CASE = ROOT / "bench" / "run_case.py"
METRICS = ROOT / "bench" / "synth_intent_metrics.py"
SCHEMA = "mlxstab.synth_intent_factorial/1"
SOURCE_FILES = (ROOT / "mlxstab" / "synth.py", ROOT / "bench" / "runner.py",
                ROOT / "bench" / "run_case.py", ROOT / "bench" / "synth_intent_metrics.py",
                Path(__file__).resolve())


def build_cells():
    """Return the predeclared 12 main cells followed by intent-only, shake-only and neither anchors."""
    cells = []
    layer_modes = {"far_only": (1.0, 1.0, 1.0), "parallax": (1.0, 0.85, 0.7)}
    for intent in ("smooth", "stochastic"):
        for layer_name, factors in layer_modes.items():
            for subject in ("absent", "static", "moving"):
                cells.append({"kind": "main", "id": f"{intent}__{layer_name}__{subject}",
                              "intent_mode": intent, "layer_mode": layer_name,
                              "layer_motion_factors": list(factors), "subject_mode": subject,
                              "subject_speed": 1.6, "jitter_px": 3.0, "rot_jitter_deg": 0.2})
    cells.extend([
        {"kind": "anchor", "id": "intent_only", "intent_mode": "smooth", "layer_mode": "far_only",
         "layer_motion_factors": [1.0, 1.0, 1.0], "subject_mode": "absent", "subject_speed": 0.0,
         "jitter_px": 0.0, "rot_jitter_deg": 0.0},
        {"kind": "anchor", "id": "shake_only", "intent_mode": "static", "layer_mode": "far_only",
         "layer_motion_factors": [1.0, 1.0, 1.0], "subject_mode": "absent", "subject_speed": 0.0,
         "jitter_px": 3.0, "rot_jitter_deg": 0.2},
        {"kind": "anchor", "id": "neither", "intent_mode": "static", "layer_mode": "far_only",
         "layer_motion_factors": [1.0, 1.0, 1.0], "subject_mode": "absent", "subject_speed": 0.0,
         "jitter_px": 0.0, "rot_jitter_deg": 0.0},
    ])
    return cells


def _source_hashes():
    hashes = {}
    for path in SOURCE_FILES:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def build_design(*, seeds=tuple(range(8)), frames=96, width=320, height=192, fps=30.0,
                 eval_iters=12, eval_downscale=2, rho=0.4):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique integers")
    if frames < 64 or width < 320 or height < 192 or fps <= 0:
        raise ValueError("factorial requires >=64 frames, >=320x192, and positive fps")
    if eval_iters < 1 or eval_downscale < 1 or not 0 <= rho <= 1:
        raise ValueError("invalid evaluation or rolling-shutter settings")
    return {
        "schema": SCHEMA,
        "design": "2x2x3 paired-seed phase-coherence x layer-motion x subject factorial",
        "seeds": list(seeds), "frames": int(frames), "width": int(width), "height": int(height),
        "fps": float(fps), "intent_profile": "matched-spectrum", "eval_iters": int(eval_iters),
        "eval_downscale": int(eval_downscale), "rho": float(rho), "cells_per_seed": build_cells(),
        "anchor_policy": "intent+shake is present in the main cells; add intent-only, shake-only, neither per seed",
        "primary_outputs": ["far-layer intent gain/correlation/RMSE", "shake-band attenuation",
                            "per-layer fit residual and visibility", "input-valid crop fraction", "fallbacks/unavailable frames"],
        "reporting": "retain all seed/cell rows; no composite score, aggregate winner, or per-cell tuning",
        "code_sha256": _source_hashes(),
    }


def _json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp, path)


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _cell_dir(out_root, seed, cell):
    return Path(out_root) / f"seed-{int(seed):04d}" / cell["id"]


def _expected_synthetic(seed, cell, design):
    return {
        "intent_mode": cell["intent_mode"], "intent_profile": "matched-spectrum",
        "layer_motion_factors": cell["layer_motion_factors"], "subject_mode": cell["subject_mode"],
        "jitter_px": cell["jitter_px"], "rot_jitter_deg": cell["rot_jitter_deg"],
        "rolling_shutter_rho": design["rho"], "frames_requested": design["frames"],
        "width": design["width"], "height": design["height"], "fps": design["fps"],
        "subject_speed_px_per_frame": (None if cell["subject_mode"] == "absent" else
                                        0.0 if cell["subject_mode"] == "static" else cell["subject_speed"]),
    }


def _run_case_command(seed, cell, design, run_dir, overwrite=False):
    factors = ",".join(str(v) for v in cell["layer_motion_factors"])
    cmd = [sys.executable, str(RUN_CASE), "--case", "baseline", "--source", "synthetic",
           "--seed", str(seed), "--synth-frames", str(design["frames"]), "--synth-w", str(design["width"]),
           "--synth-h", str(design["height"]), "--synth-fps", str(design["fps"]),
           "--synth-intent-mode", cell["intent_mode"], "--synth-intent-profile", "matched-spectrum",
           "--synth-layer-motion-factors", factors, "--synth-subject-mode", cell["subject_mode"],
           "--synth-subject-speed", str(cell["subject_speed"]), "--synth-jitter", str(cell["jitter_px"]),
           "--synth-rot-jitter", str(cell["rot_jitter_deg"]), "--synth-rho", str(design["rho"]),
           "--eval-iters", str(design["eval_iters"]), "--eval-downscale", str(design["eval_downscale"]),
           "--dump-plan", "--lock", "off", "--out-dir", str(run_dir)]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _manifest_matches(manifest, expected):
    if not manifest or manifest.get("status") != "success":
        return False
    actual = manifest.get("source", {}).get("synthetic", {})
    for key, value in expected.items():
        got = actual.get(key)
        if isinstance(value, float):
            if got is None or abs(float(got) - value) > 1e-9:
                return False
        elif got != value:
            return False
    return bool(manifest.get("outputs", {}).get("plan"))


def _run_one(seed, cell, design, out_root, *, resume=True):
    run_dir = _cell_dir(out_root, seed, cell)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "intent_metrics.json"
    expected = _expected_synthetic(seed, cell, design)
    manifest = _load_json(manifest_path)
    if resume and _manifest_matches(manifest, expected):
        metric = _load_json(metrics_path) if metrics_path.exists() else None
        if metric and metric.get("run_id") == manifest.get("run_id"):
            return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "success",
                    "manifest": str(manifest_path), "metrics": str(metrics_path), "resumed": True,
                    "far_camera_metrics": metric.get("far_camera_metrics"),
                    "layer_support_and_distortion": metric.get("layer_support_and_distortion")}
        # A successful stabilization can be reused when only the truth-aware metric step was interrupted.
        metrics_command = [sys.executable, str(METRICS), str(manifest_path), "--out", str(metrics_path)]
        measured = subprocess.run(metrics_command, cwd=ROOT, capture_output=True, text=True, check=False)
        if measured.returncode == 0:
            metric = _load_json(metrics_path)
            return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "success",
                    "manifest": str(manifest_path), "metrics": str(metrics_path), "resumed": True,
                    "far_camera_metrics": metric.get("far_camera_metrics") if metric else None,
                    "layer_support_and_distortion": metric.get("layer_support_and_distortion") if metric else None}

    # Every run directory is nested beneath the design hash and this seed/cell identity. It is owned by this
    # screen; an interrupted or failed cell can be safely replaced on resume without touching another experiment.
    overwrite = bool(manifest_path.exists() or run_dir.exists())
    command = _run_case_command(seed, cell, design, run_dir, overwrite=overwrite)
    try:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        (run_dir / "run.log").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.log").write_text((run.stdout or "") + ("\nSTDERR:\n" + run.stderr if run.stderr else ""))
        result_line = next((line for line in reversed((run.stdout or "").splitlines()) if line.startswith("{")), None)
        status_payload = json.loads(result_line) if result_line else None
        manifest = _load_json(manifest_path)
        if run.returncode != 0 or not _manifest_matches(manifest, expected):
            return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "failure",
                    "manifest": str(manifest_path) if manifest_path.exists() else None,
                    "error": {"returncode": int(run.returncode), "runner": status_payload,
                              "tail": (run.stderr or run.stdout or "")[-3000:]}}
        metrics_command = [sys.executable, str(METRICS), str(manifest_path), "--out", str(metrics_path)]
        measured = subprocess.run(metrics_command, cwd=ROOT, capture_output=True, text=True, check=False)
        if measured.returncode != 0:
            return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "failure",
                    "manifest": str(manifest_path), "error": {"stage": "intent_metrics",
                        "returncode": int(measured.returncode), "tail": (measured.stderr or measured.stdout or "")[-3000:]}}
        metric = _load_json(metrics_path)
        return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "success",
                "manifest": str(manifest_path), "metrics": str(metrics_path), "resumed": False,
                "far_camera_metrics": metric.get("far_camera_metrics") if metric else None,
                "layer_support_and_distortion": metric.get("layer_support_and_distortion") if metric else None}
    except Exception as exc:
        return {"seed": seed, "cell_id": cell["id"], "kind": cell["kind"], "status": "failure",
                "manifest": str(manifest_path) if manifest_path.exists() else None,
                "error": {"type": type(exc).__name__, "message": str(exc)}}


def run_screen(design, out_root, *, limit_cells=None, resume=True, log=print):
    out_root = Path(out_root)
    screen_path = out_root / "screen.json"
    existing = _load_json(screen_path)
    if existing and existing.get("design") != design:
        raise ValueError("existing screen.json has a different design fingerprint; refusing to mix experiments")
    rows_by_key = {(int(r["seed"]), r["cell_id"]): r for r in (existing or {}).get("runs", [])}
    cells = design["cells_per_seed"]
    if limit_cells is not None:
        if limit_cells < 1:
            raise ValueError("limit_cells must be >= 1")
        cells = cells[:limit_cells]
    planned = [(seed, cell) for seed in design["seeds"] for cell in cells]
    total = len(design["seeds"]) * len(design["cells_per_seed"])
    for index, (seed, cell) in enumerate(planned, start=1):
        log(f"[{index}/{len(planned)}; design {total} cells] seed={seed} cell={cell['id']}")
        row = _run_one(seed, cell, design, out_root, resume=resume)
        rows_by_key[(seed, cell["id"])] = row
        rows = sorted(rows_by_key.values(), key=lambda r: (int(r["seed"]), r["cell_id"]))
        done = {(int(r["seed"]), r["cell_id"]) for r in rows}
        full_keys = {(seed, cell["id"]) for seed in design["seeds"] for cell in design["cells_per_seed"]}
        _json_atomic(screen_path, {
            "schema": SCHEMA, "design": design, "runs": rows,
            "counts": {"planned": len(full_keys), "recorded": len(done),
                       "succeeded": sum(r["status"] == "success" for r in rows),
                       "failed": sum(r["status"] == "failure" for r in rows),
                       "pending": len(full_keys - done)},
        })
    return _load_json(screen_path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(x) for x in range(8)),
                        help="comma-separated paired seed IDs (default 0..7)")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--eval-iters", type=int, default=12)
    parser.add_argument("--eval-downscale", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--out-root", help="explicit root; otherwise a code/design-hashed directory under out/bench")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-cells", type=int, help="execute only the first N cells per seed (plumbing smoke)")
    parser.add_argument("--no-resume", action="store_true", help="re-run each selected cell even if valid output exists")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        design = build_design(seeds=seeds, frames=args.frames, width=args.width, height=args.height,
                              fps=args.fps, eval_iters=args.eval_iters,
                              eval_downscale=args.eval_downscale, rho=args.rho)
    except ValueError as exc:
        print(json.dumps({"status": "failure", "error": str(exc)}))
        return 2
    encoded = json.dumps(design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:12]
    out_root = Path(args.out_root) if args.out_root else ROOT / "out" / "bench" / f"synth_intent_factorial_{key}"
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "design_key": key, "out_root": str(out_root),
                          "seeds": list(seeds), "cells_per_seed": len(design["cells_per_seed"]),
                          "total_runs": len(seeds) * len(design["cells_per_seed"]),
                          "cells": design["cells_per_seed"]}, indent=2, sort_keys=True))
        return 0
    try:
        result = run_screen(design, out_root, limit_cells=args.limit_cells, resume=not args.no_resume)
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 1
    counts = result["counts"]
    status = ("success" if counts["pending"] == 0 and counts["failed"] == 0 else
              "partial" if args.limit_cells is not None and counts["failed"] == 0 else "incomplete")
    print(json.dumps({"status": status,
                      "screen": str(out_root / "screen.json"), "counts": counts}, sort_keys=True))
    return 0 if status in ("success", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
