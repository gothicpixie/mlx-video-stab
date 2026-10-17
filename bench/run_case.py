#!/usr/bin/env python
"""CLI for the Stage 0 benchmark runner.

Examples
--------
List cases::

    python -I bench/run_case.py --list-cases

Public synthetic smoke (no private footage, no heavy lock)::

    python -I bench/run_case.py --case baseline --source synthetic --duration 1 --synth-frames 30

Cheap real-input smoke (1 s of a local window; uses the shared heavy lock)::

    python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS --start 60 --duration 1 \
        --diag-every 15 --out-dir out/bench/smoke_t060

Full 10 s baseline window (bounded; run in the background and keep the printed handle)::

    python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS --start 60 --duration 10 \
        --lock on --log out/bench/t060.log

Every run writes an atomic JSON manifest under the run directory (default ``out/bench/<time>_<case>_<tag>``)
and prints the manifest path on the last line.

Residual-channel ablation (mesh and/or rows disabled; the remaining motion fields are refit per case, so this
is a practical configuration ablation, not fixed-plan causal isolation)::

    python -I bench/run_case.py --case global-only --source /path/to/local/clip.MTS --start 60 --duration 1 \
        --diag-every 15 --lock on --out-dir out/bench/ablation_t060_global_1s
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import cases, runner  # noqa: E402


def _parse_set(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_parser():
    p = argparse.ArgumentParser(prog="bench/run_case.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-cases", action="store_true", help="list case names and exit")
    p.add_argument("--case", default="baseline", choices=list(cases.case_names))
    p.add_argument("--source", default="synthetic", help="'synthetic' or a local video path")
    p.add_argument("--weights", default=str(runner.DEFAULT_WEIGHTS))
    p.add_argument("--backend", choices=["mlx", "torch-mps"], default="mlx",
                   help="flow backend; the planner, renderer, media path and evaluator stay shared mlxstab code")
    p.add_argument("--torch-weights", default=str(runner.DEFAULT_TORCH_WEIGHTS),
                   help="upstream RAFT .pth weights used by --backend torch-mps")
    p.add_argument("--start", type=float, help="source window start (seconds)")
    p.add_argument("--duration", type=float, help="source window duration (seconds)")
    p.add_argument("--out-dir", help="explicit run directory (default out/bench/<time>_<case>_<tag>)")
    p.add_argument("--overwrite", action="store_true",
                   help="allow reusing an existing run directory; by default existing evidence is protected")
    p.add_argument("--output-fps", help="output frame rate override (rational, or 'native'); default from case")
    p.add_argument("--eval-vf", help="ffmpeg -vf filter for evaluation cadence; default from case")
    p.add_argument("--selection-note", help="provenance note (e.g. the window was inferred from historical filenames)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1, help="run 1 (cold) + N-1 warm repeats")
    p.add_argument("--eval-iters", type=int, default=12, help="iterations for the evaluation flow")
    p.add_argument("--eval-downscale", type=int, default=2, help="downscale for the evaluation flow")
    p.add_argument("--eval-flow-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--eval-corr-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--eval-tracks", action="store_true",
                   help="also measure the independent sparse-KLT estimator and report RAFT/track agreement (opt-in)")
    p.add_argument("--diag-every", type=int, default=0,
                   help="sample complete-warp diagnostics every K rendered frames from the actual plan (0=off)")
    p.add_argument("--dump-plan", action="store_true",
                   help="write the retained repeat-0 Plan to <out-dir>/plan_r0.npz (compressed, non-pickle; private motion data)")
    p.add_argument("--dump-motion", action="store_true",
                   help="write the retained repeat-0 fitted Motion to <out-dir>/motion_r0.npz (compressed, non-pickle; private motion data)")
    p.add_argument("--dump-flow-samples", action="store_true",
                   help="write repeat-0 raw forward/backward flow vectors on the fitting grid to <out-dir>/flow_samples_r0.npz")
    p.add_argument("--lock", choices=["auto", "on", "off"], default="auto",
                   help="shared heavy lock: auto=on for file sources, off for synthetic")
    p.add_argument("--lock-chunk", type=int, default=0, help="release/re-acquire the lock every N on_step calls")
    p.add_argument("--lock-pause", type=float, default=0.0, help="seconds to yield between lock chunks (etiquette ~12 s)")
    p.add_argument("--synth-frames", type=int, default=30)
    p.add_argument("--synth-w", type=int, default=320)
    p.add_argument("--synth-h", type=int, default=180)
    p.add_argument("--synth-fps", type=float, default=30.0)
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one Config field (JSON value)")
    p.add_argument("--config-json", help="JSON object of Config overrides")
    p.add_argument("--dry-run", action="store_true", help="print the resolved case and exit without running")
    p.add_argument("--log", help="also append stdout/stderr of the run to this file")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.list_cases:
        for name in cases.case_names:
            c = cases.get_case(name)
            print(f"{name}\tchanged={c.changed or 'none'}\t{c.description}")
        return 0

    overrides = _parse_set(a.set)
    if a.config_json:
        overrides.update(json.loads(a.config_json))
    # One seed drives synthetic clip generation and the per-clip fitting RNG; --set seed= can override it.
    overrides.setdefault("seed", a.seed)
    case = cases.resolve_case(a.case, overrides)
    resolved = case.resolved_config()
    output_fps = a.output_fps if a.output_fps is not None else case.output_fps
    if output_fps == "native":
        output_fps = None
    eval_vf = a.eval_vf if a.eval_vf is not None else case.eval_vf

    if a.dry_run:
        print(json.dumps({"case": case.name, "changed": case.changed, "eval_vf": eval_vf,
                          "output_fps": output_fps,
                          "preset_overrides": case.config, "resolved_config": resolved,
                          "canonical": cases.canonical_config(resolved),
                          "config_digest": cases.config_digest(resolved)}, indent=1, sort_keys=True))
        return 0

    logfile = None
    log = print
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        logfile = open(a.log, "a", buffering=1)
        log = lambda m: (print(m), logfile.write(str(m) + "\n"))
    opts = runner.RunOptions(
        case=a.case, source=a.source, weights=a.weights, backend=a.backend, torch_weights=a.torch_weights,
        start=a.start, duration=a.duration, out_dir=a.out_dir,
        overwrite=a.overwrite, seed=a.seed, repeats=a.repeats, eval_iters=a.eval_iters, eval_downscale=a.eval_downscale,
        eval_flow_dtype=a.eval_flow_dtype, eval_corr_dtype=a.eval_corr_dtype, eval_vf=a.eval_vf, output_fps=a.output_fps,
        eval_tracks=a.eval_tracks,
        diag_every=a.diag_every, dump_plan=a.dump_plan, dump_motion=a.dump_motion,
        dump_flow_samples=a.dump_flow_samples, lock=a.lock,
        lock_chunk=a.lock_chunk, lock_pause_s=a.lock_pause,
        synth_frames=a.synth_frames, synth_w=a.synth_w, synth_h=a.synth_h, synth_fps=a.synth_fps,
        selection_note=a.selection_note, overrides=overrides, command=[sys.executable, *sys.argv], log=log)
    try:
        man = runner.run_case(opts)
    except Exception as e:          # overwrite refusal, initial-write failure, etc.: structured, nonzero
        if logfile:
            logfile.close()
        print(json.dumps({"status": "failure", "case": a.case, "manifest": None,
                          "error": {"type": type(e).__name__, "message": str(e)}}, sort_keys=True))
        return 1
    if logfile:
        logfile.close()
    case_name = (man.get("case") or {}).get("name") or a.case
    print(json.dumps({"status": man["status"], "case": case_name,
                      "manifest": man["outputs"]["manifest"], "error": man.get("error")}, sort_keys=True))
    return 0 if man["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
