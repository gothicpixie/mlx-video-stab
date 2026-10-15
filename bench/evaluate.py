"""Measure input vs stabilized output with the objective metrics (mlxstab.metrics), using RAFT flow.

    python -I bench/evaluate.py in.mp4 out.mp4 [mask.mp4] [--weights W] [--json out.json] [--start S --duration D]"""
import argparse, json, sys
import numpy as np
from mlxstab import media, metrics
from mlxstab.stabilize import RaftFlow
from mlxstab.__main__ import DEFAULT_WEIGHTS

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("mask", nargs="?")
ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS)); ap.add_argument("--json"); ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--downscale", type=int, default=1); ap.add_argument("--vf", help="ffmpeg filter applied after decoding (e.g. fps=30000/1001) to put clips at a common rate"); ap.add_argument("--start", type=float); ap.add_argument("--duration", type=float)
a = ap.parse_args()
flow = RaftFlow(a.weights, a.iters, a.downscale)
res = {}
for name, path in (("input", a.inp), ("output", a.out)):
    info = media.probe(path)
    fr, fps, (W, H) = media.decode(path, info, a.start if name == "input" else None, a.duration if name == "input" else None, vf=a.vf)
    masks = None
    if name == "output" and a.mask:
        mi = media.probe(a.mask); mf, _, _ = media.decode(a.mask, mi, vf=a.vf); masks = (m[..., 0] > 127 for m in mf)
    flow.reset()                        # each clip is an independent sequence: no warm start or feature carry-over
    res[name] = metrics.measure(fr, flow, W, H, fps, masks, keep_frames=False)
    print(name, json.dumps(res[name]), file=sys.stderr)
if a.json: open(a.json, "w").write(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
