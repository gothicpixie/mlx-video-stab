import argparse, json, math, sys
from pathlib import Path
from .stabilize import Config, stabilize

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "vendor/weights/conv/raft-things.safetensors"


def _flow_downscale_arg(value):
    if value == "auto":
        return "auto"
    try:
        scale = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--flow-downscale must be 1, 2, 4 or auto") from None
    if scale not in (1, 2, 4):
        raise argparse.ArgumentTypeError("--flow-downscale must be 1, 2, 4 or auto")
    return scale


def _cut_frames_arg(value):
    """Parse ``--cut-frames`` as comma-separated internal shot-start frame indices."""
    parts = [p.strip() for p in value.split(",")]
    if not parts or any(not p for p in parts):
        raise argparse.ArgumentTypeError("--cut-frames needs at least one integer frame index")
    out = []
    for part in parts:
        try:
            v = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--cut-frames contains a non-integer: {part!r}") from None
        if v < 0:
            raise argparse.ArgumentTypeError(f"--cut-frames must be non-negative, got {v}")
        out.append(v)
    return tuple(sorted(set(out)))


def _analysis_stride_arg(value):
    try:
        stride = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--analysis-stride must be a positive integer") from None
    if stride <= 0:
        raise argparse.ArgumentTypeError("--analysis-stride must be a positive integer")
    return stride


def _positive_gib_bytes(value):
    try:
        gib = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--max-flow-corr-gib must be a positive finite number of GiB") from None
    if not math.isfinite(gib) or gib <= 0:
        raise argparse.ArgumentTypeError("--max-flow-corr-gib must be a positive finite number of GiB")
    if gib > sys.maxsize / (1 << 30):
        raise argparse.ArgumentTypeError("--max-flow-corr-gib is too large to express as a byte budget")
    nbytes = int(gib * (1 << 30))
    if nbytes <= 0:
        raise argparse.ArgumentTypeError("--max-flow-corr-gib is too small to express a positive byte budget")
    return nbytes


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
    p.add_argument("--mesh-shape-strength", type=float, default=0.0,
                   help="experimental: opt-in shape-preserving residual-mesh regularization strength "
                        "(finite, >= 0); disabled by default (0)")
    p.add_argument("--rs-rows", type=int, default=18, help="rolling-shutter row bands, 0 disables")
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--flow-downscale", type=_flow_downscale_arg, default=1,
                   help="RAFT flow resolution: 1, 2, 4, or auto (smallest scale whose bidirectional correlation "
                        "pyramid fits --max-flow-corr-gib; auto requires the built-in MLX RAFT backend)")
    p.add_argument("--max-flow-corr-gib", type=_positive_gib_bytes, default=None,
                   help="opt-in correlation-pyramid payload budget in GiB (positive finite; not peak process "
                        "memory); required by --flow-downscale auto and enforced for a fixed scale before RAFT construction")
    p.add_argument("--kernel", choices=["bilinear", "bicubic", "lanczos3"], default="lanczos3")
    p.add_argument("--fidelity", type=float, default=0.1, help="L1 pull of the smoothed path towards the raw path (0 = pure L1, larger = hugs the raw path)")
    p.add_argument("--l1-low-freq-fidelity", type=float, default=0.0,
                   help="opt-in L1 pull towards a low-passed camera path (0 disables; try on a copy and compare)")
    p.add_argument("--l1-low-freq-sigma", type=float, default=15.0,
                   help="Gaussian sigma in frames for the opt-in low-frequency L1 target")
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
    p.add_argument("--cut-frames", type=_cut_frames_arg, default=None, metavar="N[,N...]",
                   help="opt-in: comma-separated internal frame indices that begin a new shot; no flow pair is "
                        "measured across a cut and planning restarts per shot (relative to the decoded clip; "
                        "must align with --analysis-stride)")
    p.add_argument("--analysis-stride", type=_analysis_stride_arg, default=1,
                   help="analyse every n-th frame and interpolate the path (default 1)")
    a = p.parse_args(argv)
    gc, gr = (int(v) for v in a.mesh.lower().split("x"))
    cfg = Config(smooth=a.smooth, sigma=a.sigma, budget=a.budget, mesh=(gc, gr), rs_rows=a.rs_rows, flow_iters=a.iters,
                 mesh_shape_strength=a.mesh_shape_strength,
                 flow_downscale=a.flow_downscale, max_flow_corr_bytes=a.max_flow_corr_gib,
                 kernel=a.kernel, zoom=a.zoom, fidelity=a.fidelity,
                 l1_low_freq_fidelity=a.l1_low_freq_fidelity, l1_low_freq_sigma=a.l1_low_freq_sigma,
                 warp_diag=a.warp_diag,
                 max_sampled_border_frac=a.max_sampled_border_frac, crop_zoom=a.crop_zoom,
                 analysis_stride=a.analysis_stride)
    rep = stabilize(a.input, a.output, a.mask, a.weights, cfg, a.roll, a.start, a.duration, a.fps, report=a.report,
                    log=lambda m: print(m, file=sys.stderr), cuts=a.cut_frames)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
