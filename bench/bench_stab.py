"""Stabilizer throughput on the synthetic clip:  python -I bench/bench_stab.py W H T [downscale] [iters]
Generates a clip, runs the full pipeline (flow analysis, planning, render + encode) and prints the report as JSON."""
import json, sys, tempfile, os
from mlxstab import synth, media
from mlxstab.stabilize import Config, stabilize
from mlxstab.__main__ import DEFAULT_WEIGHTS

W, H, T = (int(v) for v in sys.argv[1:4]); ds = int(sys.argv[4]) if len(sys.argv) > 4 else 1; it = int(sys.argv[5]) if len(sys.argv) > 5 else 12
d = tempfile.mkdtemp(); src = os.path.join(d, "in.mp4")
vid, _ = synth.render(T=T, W=W, H=H); e = media.Encoder(src, W, H, 30, crf=12, preset="veryfast")
for f in vid: e.write(f)
e.close()
rep = stabilize(src, os.path.join(d, "out.mp4"), os.path.join(d, "mask.mp4"), DEFAULT_WEIGHTS, Config(flow_iters=it, flow_downscale=ds), log=lambda m: None)
rep.update({"W": W, "H": H, "flow_downscale": ds, "iters": it}); print(json.dumps(rep))
