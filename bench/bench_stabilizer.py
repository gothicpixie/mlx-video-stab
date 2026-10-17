"""Compare MLX and PyTorch MPS RAFT inside the same stabilization pipeline.

Example:
    python -I bench/bench_stabilizer.py --source synthetic --synth-frames 30 \
        --synth-w 960 --synth-h 544 --flow-iters 12 --flow-downscale 1

The planner, renderer, media path and MLX-based evaluator are shared. This is a
flow-backend comparison, not a comparison with an independently implemented
PyTorch stabilizer.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import runner  # noqa: E402


def _summary(manifest):
    reps = manifest.get("repeats") or []
    warm = [float(r["wall_s"]) for r in reps[1:] if isinstance(r.get("wall_s"), (int, float))]
    rep0 = reps[0] if reps else {}
    return {
        "backend": manifest.get("flow_backend", {}).get("name"),
        "run_dir": manifest.get("outputs", {}).get("manifest", "").rsplit("/", 1)[0],
        "manifest": manifest.get("outputs", {}).get("manifest"),
        "flow_weights_sha256": manifest.get("flow_backend", {}).get("weights_sha256"),
        "weight_equivalence": manifest.get("flow_backend", {}).get("weight_equivalence"),
        "source_sha256": manifest.get("source", {}).get("sha256"),
        "config_digest": manifest.get("config_digest"),
        "status": manifest.get("status"),
        "frames": manifest.get("outputs", {}).get("encoded_frames"),
        "measured_fps": manifest.get("outputs", {}).get("cadence", {}).get("measured_fps"),
        "timeline_aligned": manifest.get("outputs", {}).get("timeline_alignment", {}).get("aligned"),
        "cold_model_load_s": manifest.get("timing", {}).get("model_construct_s"),
        "cold_repeat_s": rep0.get("wall_s"),
        "warm_repeat_s": warm,
        "warm_repeat_median_s": statistics.median(warm) if warm else None,
        "end_to_end_stabilization_s": manifest.get("timing", {}).get("end_to_end_stabilization_s"),
        "motion": manifest.get("outputs", {}).get("motion"),
        "input_jitter_px": (manifest.get("metrics", {}).get("input") or {}).get("jitter_px"),
        "output_jitter_px": (manifest.get("metrics", {}).get("output") or {}).get("jitter_px"),
        "output_wobble_px": (manifest.get("metrics", {}).get("output") or {}).get("wobble_px"),
        "mean_border_frac": ((rep0.get("stabilize_report") or {}).get("mean_border_frac")),
        "fallback_frames": ((rep0.get("stabilize_report") or {}).get("fallback_frames")),
        "video": manifest.get("outputs", {}).get("video"),
        "mask": manifest.get("outputs", {}).get("mask"),
        "plan": manifest.get("outputs", {}).get("plan"),
        "error": manifest.get("error"),
    }


def _backend_order(order):
    if order == "mlx-first":
        return ("mlx", "torch-mps")
    if order == "torch-first":
        return ("torch-mps", "mlx")
    raise ValueError("backend_order must be 'mlx-first' or 'torch-first'")


def _plan_delta(first, second):
    if not first or not second:
        return {"available": False, "reason": "both runs must save a plan"}
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if int(a["frames"]) != int(b["frames"]):
            return {"available": False, "reason": "plan frame counts differ"}
        fields = ("G", "mesh_corr", "row_corr")
        result = {"available": True, "frames": int(a["frames"])}
        for key in fields:
            if a[key].shape != b[key].shape:
                result[key] = {"shape_match": False, "left": list(a[key].shape), "right": list(b[key].shape)}
                continue
            delta = np.abs(np.asarray(a[key], np.float64) - np.asarray(b[key], np.float64))
            result[key] = {"shape_match": True, "max_abs": float(delta.max()) if delta.size else 0.0,
                           "mean_abs": float(delta.mean()) if delta.size else 0.0}
        return result


def _cadence_matches(left, right):
    """Compare numeric encoded PTS samples and measured rates from runner manifests."""
    left_pts = (left.get("encoded_pts") or {}).get("pts_s")
    right_pts = (right.get("encoded_pts") or {}).get("pts_s")
    left_rate = (left.get("cadence") or {}).get("measured_fps")
    right_rate = (right.get("cadence") or {}).get("measured_fps")
    if not isinstance(left_pts, list) or not isinstance(right_pts, list) or len(left_pts) != len(right_pts):
        return False
    if not isinstance(left_rate, (int, float)) or not isinstance(right_rate, (int, float)):
        return False
    try:
        a, b = np.asarray(left_pts, np.float64), np.asarray(right_pts, np.float64)
        return (np.isfinite(a).all() and np.isfinite(b).all()
                and abs(float(left_rate) - float(right_rate)) <= 1e-9
                and np.allclose(a, b, rtol=0.0, atol=1e-9))
    except (TypeError, ValueError):
        return False


def _runner_command(source, out_dir, backend, weights, torch_weights, start, duration,
                    flow_iters, flow_downscale, eval_iters, eval_downscale, warm_repeats,
                    seed, eval_tracks, output_fps, lock, dump_motion=False, dump_flow_samples=False):
    """Build the equivalent single-backend runner invocation stored in its manifest."""
    command = [sys.executable, "-I", str(ROOT / "bench" / "run_case.py"),
               "--case", "baseline", "--source", str(source), "--backend", backend,
               "--weights", str(weights), "--torch-weights", str(torch_weights),
               "--out-dir", str(out_dir), "--repeats", str(warm_repeats + 1),
               "--eval-iters", str(eval_iters), "--eval-downscale", str(eval_downscale),
               "--seed", str(seed), "--lock", lock, "--dump-plan",
               "--set", f"flow_iters={flow_iters}", "--set", f"flow_downscale={flow_downscale}"]
    if start is not None:
        command.extend(["--start", str(start)])
    if duration is not None:
        command.extend(["--duration", str(duration)])
    if eval_tracks:
        command.append("--eval-tracks")
    if dump_motion:
        command.append("--dump-motion")
    if dump_flow_samples:
        command.append("--dump-flow-samples")
    if output_fps is not None:
        command.extend(["--output-fps", str(output_fps)])
    return command


def compare(source="synthetic", out_dir=None, weights=str(runner.DEFAULT_WEIGHTS),
            torch_weights=str(runner.DEFAULT_TORCH_WEIGHTS), start=None, duration=None,
            flow_iters=12, flow_downscale=1, eval_iters=12, eval_downscale=2,
            warm_repeats=3, seed=0, synth_frames=30, synth_w=960, synth_h=544,
            synth_fps=30.0, eval_tracks=False, output_fps=None, lock="auto", backend_order="mlx-first",
            dump_motion=False, dump_flow_samples=False):
    if warm_repeats < 1:
        raise ValueError("warm_repeats must be >= 1")
    run_order = _backend_order(backend_order)
    root = Path(out_dir) if out_dir else runner.DEFAULT_OUT / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_stabilizer_backend_comparison")
    root.mkdir(parents=True, exist_ok=True)
    for backend in run_order:
        child = root / backend
        if child.exists() and any(child.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty backend run directory: {child}")
    source_origin = "provided file"
    if source == "synthetic":
        source_path = root / "comparison_input.mp4"
        if source_path.exists():
            raise FileExistsError(f"refusing to replace existing comparison source: {source_path}")
        runner.make_synthetic(source_path, synth_w, synth_h, synth_frames, seed, synth_fps)
        source = str(source_path)
        source_origin = "generated once and shared by both backends"

    manifests = {}
    for backend in run_order:
        opts = runner.RunOptions(
            case="baseline", source=source, weights=weights, backend=backend, torch_weights=torch_weights,
            start=start, duration=duration, out_dir=str(root / backend), seed=seed,
            repeats=warm_repeats + 1, eval_iters=eval_iters, eval_downscale=eval_downscale,
            eval_tracks=eval_tracks, output_fps=output_fps, lock=lock, dump_plan=True, dump_motion=dump_motion,
            dump_flow_samples=dump_flow_samples,
            overrides={"flow_iters": flow_iters, "flow_downscale": flow_downscale, "seed": seed},
            command=_runner_command(source, root / backend, backend, weights, torch_weights, start, duration,
                                   flow_iters, flow_downscale, eval_iters, eval_downscale, warm_repeats,
                                   seed, eval_tracks, output_fps, lock, dump_motion, dump_flow_samples),
            log=print,
        )
        man = runner.run_case(opts)
        manifests[backend] = man
        if man.get("status") != "success":
            result = {"schema": "mlxstab.stabilizer_backend_comparison/1", "status": "failure",
                      "failed_backend": backend, "error": man.get("error"),
                      "runs": {k: _summary(v) for k, v in manifests.items()}}
            path = root / "comparison.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return result

    left, right = manifests["mlx"], manifests["torch-mps"]
    summary = {k: _summary(v) for k, v in manifests.items()}
    cadence_matches = _cadence_matches(left.get("outputs", {}), right.get("outputs", {}))
    comparable = (summary["mlx"]["source_sha256"] == summary["torch-mps"]["source_sha256"]
                  and summary["mlx"]["config_digest"] == summary["torch-mps"]["config_digest"]
                  and summary["mlx"]["frames"] == summary["torch-mps"]["frames"]
                  and cadence_matches
                  and summary["mlx"]["timeline_aligned"] is True
                  and summary["torch-mps"]["timeline_aligned"] is True)
    result = {
        "schema": "mlxstab.stabilizer_backend_comparison/1", "status": "success" if comparable else "incomparable",
        "claim_scope": "RAFT flow backend inside the shared mlxstab planner, renderer, media path and MLX evaluator; not a full independent PyTorch stabilizer comparison",
        "source_origin": source_origin,
        "config": {"flow_iters": flow_iters, "flow_downscale": flow_downscale,
                   "eval_iters": eval_iters, "eval_downscale": eval_downscale,
                   "warm_repeats": warm_repeats, "seed": seed, "backend_order": backend_order,
                   "dump_motion": dump_motion, "dump_flow_samples": dump_flow_samples},
        "runs": summary,
        "exact_checkpoint_tensor_match": (summary["torch-mps"].get("weight_equivalence") or {}).get("exact_tensor_match") is True,
        "same_config_and_cadence": comparable,
        "encoded_cadence_matches": cadence_matches,
        "plan_delta": _plan_delta(left.get("outputs", {}).get("plan"), right.get("outputs", {}).get("plan")),
    }
    path = root / "comparison.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if comparable:
        rows = []
        for backend in ("mlx", "torch-mps"):
            item = summary[backend]
            rows.append(f"| {backend} | {item['warm_repeat_median_s']:.3f} | {item['end_to_end_stabilization_s']:.3f} | "
                        f"{item['output_jitter_px']} | {item['output_wobble_px']} | {item['mean_border_frac']} |")
        markdown = "\n".join([
            "# Shared-pipeline flow-backend comparison", "",
            "This compares RAFT flow backends inside the same mlxstab planner, renderer, media path and MLX evaluator. "
            "It does not compare against an independently implemented PyTorch stabilizer.", "",
            "| Flow backend | Warm stabilization repeat median (s) | End-to-end incl. model load (s) | Output jitter (px) | Output wobble (px) | Mean border fraction |",
            "|---|---:|---:|---:|---:|---:|", *rows, "",
            f"Same source hash, resolved configuration, frame count and encoded cadence: **{comparable}**.",
            f"Exact checkpoint tensor equality after conversion: **{result['exact_checkpoint_tensor_match']}**.",
            f"Per-frame plan differences: `{json.dumps(result['plan_delta'], sort_keys=True)}`.",
            "Metrics are one run setup; use independent clips and repeated comparisons before making a quality claim.",
            f"Detailed machine-readable result: `{path}`.", "",
        ])
        (root / "comparison.md").write_text(markdown)
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="synthetic", help="synthetic or a local source video")
    p.add_argument("--out-dir")
    p.add_argument("--weights", default=str(runner.DEFAULT_WEIGHTS), help="MLX safetensors weights")
    p.add_argument("--torch-weights", default=str(runner.DEFAULT_TORCH_WEIGHTS), help="PyTorch RAFT .pth weights")
    p.add_argument("--start", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--flow-iters", type=int, default=12)
    p.add_argument("--flow-downscale", type=int, default=1)
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--eval-downscale", type=int, default=2)
    p.add_argument("--warm-repeats", type=int, default=3, help="timed cached-model runs after one cold run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synth-frames", type=int, default=30)
    p.add_argument("--synth-w", type=int, default=960)
    p.add_argument("--synth-h", type=int, default=544)
    p.add_argument("--synth-fps", type=float, default=30.0)
    p.add_argument("--eval-tracks", action="store_true")
    p.add_argument("--output-fps")
    p.add_argument("--lock", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--backend-order", choices=["mlx-first", "torch-first"], default="mlx-first",
                   help="paired backend run order; alternate across windows to reduce order effects")
    p.add_argument("--dump-motion", action="store_true",
                   help="capture the retained repeat fitted-motion NPZ for offline diagnostics")
    p.add_argument("--dump-flow-samples", action="store_true",
                   help="capture raw forward/backward vectors on the exact grid used by the motion fit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = compare(source=args.source, out_dir=args.out_dir, weights=args.weights,
                    torch_weights=args.torch_weights, start=args.start, duration=args.duration,
                    flow_iters=args.flow_iters, flow_downscale=args.flow_downscale,
                    eval_iters=args.eval_iters, eval_downscale=args.eval_downscale,
                    warm_repeats=args.warm_repeats, seed=args.seed, synth_frames=args.synth_frames,
                    synth_w=args.synth_w, synth_h=args.synth_h, synth_fps=args.synth_fps,
                    eval_tracks=args.eval_tracks, output_fps=args.output_fps, lock=args.lock,
                    backend_order=args.backend_order, dump_motion=args.dump_motion,
                    dump_flow_samples=args.dump_flow_samples)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
