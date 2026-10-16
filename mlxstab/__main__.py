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
    p.add_argument("--fidelity", type=float, default=0.1, help="L1 pull of the smoothed path towards the raw path (0 = pure L1, larger = hugs the raw path)")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--crop-zoom", type=float, default=1.0,
                   help="opt-in: isotropic center zoom on output coordinates (finite, >= 1), composed before the "
                        "global output-to-source map; 1.0 (default) leaves the warp unchanged")
    p.add_argument("--start", type=float); p.add_argument("--duration", type=float)
    p.add_argument("--fps", type=float, help="output frame rate (default: processing rate; interlaced sources are 2x field rate)")
    p.add_argument("--report", help="write a JSON report")
    p.add_argument("--warp-diag", action="store_true", help="include complete-warp shape/coverage diagnostics in --report (opt-in)")
    p.add_argument("--max-sampled-border-frac", type=float, default=None,
                   help="opt-in: cap the sampled output-out-of-bounds fraction (finite, in [0,1)); one clip-wide "
                        "correction scale in [0,1] is chosen so every frame's sampled fraction stays <= this. "
                        "Sampled probe-grid estimate, not a per-pixel guarantee; default disables the backoff")
    a = p.parse_args(argv)
    gc, gr = (int(v) for v in a.mesh.lower().split("x"))
    cfg = Config(smooth=a.smooth, sigma=a.sigma, budget=a.budget, mesh=(gc, gr), rs_rows=a.rs_rows, flow_iters=a.iters,
                 flow_downscale=a.flow_downscale, kernel=a.kernel, zoom=a.zoom, fidelity=a.fidelity, warp_diag=a.warp_diag,
                 max_sampled_border_frac=a.max_sampled_border_frac, crop_zoom=a.crop_zoom)
    rep = stabilize(a.input, a.output, a.mask, a.weights, cfg, a.roll, a.start, a.duration, a.fps, report=a.report,
                    log=lambda m: print(m, file=sys.stderr))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
