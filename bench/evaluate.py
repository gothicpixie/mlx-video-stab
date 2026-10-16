"""Measure input vs stabilized output with the objective metrics (mlxstab.metrics), using RAFT flow.

    python -I bench/evaluate.py in.mp4 out.mp4 [mask.mp4] [--weights W] [--json out.json] [--start S --duration D]

Add ``--tracks`` to also measure the same input/output pair with the independent sparse-KLT estimator
(``mlxstab.tracks``) and report whether the two estimators agree on the jitter change. KLT is a different,
RAFT-free estimator, not ground truth; it cross-checks the single-estimator story instead of replacing it.
"""
import argparse, json, sys
from itertools import tee
import numpy as np
from mlxstab import media, metrics, tracks
from mlxstab.stabilize import RaftFlow
from mlxstab.__main__ import DEFAULT_WEIGHTS

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("mask", nargs="?")
ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS)); ap.add_argument("--json"); ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--downscale", type=int, default=1); ap.add_argument("--vf", help="ffmpeg filter applied after decoding (e.g. fps=30000/1001) to put clips at a common rate"); ap.add_argument("--start", type=float); ap.add_argument("--duration", type=float)
ap.add_argument("--tracks", action="store_true", help="also measure with the independent KLT estimator and report agreement")
ap.add_argument("--track-seed", type=int, default=0, help="deterministic RANSAC seed for the KLT estimator")
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
    acc = tracks.TrackAccumulator(W, H, fps, seed=a.track_seed) if a.tracks else None
    track_masks = None
    if acc is not None and masks is not None:
        track_masks, masks = tee(masks)
    evaluated_frames = acc.wrap(fr, track_masks) if acc else fr
    res[name] = metrics.measure(evaluated_frames, flow, W, H, fps, masks, keep_frames=False)
    if acc:
        res[name]["tracks"] = acc.metrics()
    print(name, json.dumps(res[name]), file=sys.stderr)
if a.tracks and isinstance(res.get("input"), dict) and isinstance(res.get("output"), dict):
    res["agreement"] = tracks.agreement(res["input"], res["output"], res["input"].get("tracks"), res["output"].get("tracks"))
if a.json: open(a.json, "w").write(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
