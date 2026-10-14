import argparse, json, sys
from pathlib import Path
from .stabilize import Config, stabilize

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "vendor/weights/conv/raft-things.safetensors"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m mlxstab", description="No-crop video stabilization on Apple Silicon (MLX + RAFT).")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--mask", help="write the per-frame border mask (white = no source) to this video")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="RAFT safetensors (see mlxstab.raft.convert)")
    p.add_argument("--smooth", choices=["l1", "gauss"], default="l1")
    p.add_argument("--sigma", type=float, default=15.0, help="gaussian sigma in frames (--smooth gauss)")
    p.add_argument("--budget", type=float, default=0.04, help="border budget: max path deviation, fraction of frame width")
    p.add_argument("--roll", help="JSON roll series in degrees (positive = counter-clockwise lean) to level the horizon")
    p.add_argument("--mesh", default="16x9", help="residual mesh cells COLSxROWS, or 0x0 to disable")
    p.add_argument("--rs-rows", type=int, default=18, help="rolling-shutter row bands, 0 disables")
    p.add_argument("--iters", type=int, default=24); p.add_argument("--flow-downscale", type=int, default=1)
    p.add_argument("--kernel", choices=["bilinear", "bicubic", "lanczos3"], default="lanczos3")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--start", type=float); p.add_argument("--duration", type=float)
    p.add_argument("--fps", type=float, help="output frame rate (default: processing rate; interlaced sources are 2x field rate)")
    p.add_argument("--report", help="write a JSON report")
    a = p.parse_args(argv)
    gc, gr = (int(v) for v in a.mesh.lower().split("x"))
    cfg = Config(smooth=a.smooth, sigma=a.sigma, budget=a.budget, mesh=(gc, gr), rs_rows=a.rs_rows, flow_iters=a.iters,
                 flow_downscale=a.flow_downscale, kernel=a.kernel, zoom=a.zoom)
    rep = stabilize(a.input, a.output, a.mask, a.weights, cfg, a.roll, a.start, a.duration, a.fps, report=a.report,
                    log=lambda m: print(m, file=sys.stderr))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
