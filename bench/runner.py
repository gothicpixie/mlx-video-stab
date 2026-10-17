"""Stage 0 benchmark runner: one explicit case -> one atomic, JSON-safe manifest.

    python -I bench/run_case.py --case baseline --source synthetic --duration 1
    python -I bench/run_case.py --case baseline --source /path/to/clip.MTS --start 60 --duration 1 \
        --diag-every 15 --lock on

The runner does not invent a single quality score. It records stabilization timing (including model load and
encoder close), input/output metrics measured identically at a common cadence with validated timelines,
explicit failure/support flags, complete-warp diagnostic values sampled from the **actual full-window plan**,
and enough provenance (source/weight hashes, full experiment identity, working-tree code digests, versions,
hardware) to invalidate a comparison. Private paths live only in the ignored ``out/`` manifest.
"""
from __future__ import annotations

import re
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import tee
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxstab import media, metrics, synth, tracks, warpmetrics
from mlxstab.stabilize import Config, RaftFlow, resolve_flow_downscale, stabilize
from mlxstab.__main__ import DEFAULT_WEIGHTS

from . import cases, manifest as MF
from .heavylock import HeavyLock

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "out" / "bench"
DEFAULT_TORCH_WEIGHTS = ROOT / "vendor" / "weights" / "things" / "raft-things.pth"


@dataclass
class RunOptions:
    case: str = "baseline"
    source: str = "synthetic"
    weights: str = str(DEFAULT_WEIGHTS)
    backend: str = "mlx"               # mlx | torch-mps (flow only; planner/render/evaluator remain shared)
    torch_weights: str = str(DEFAULT_TORCH_WEIGHTS)
    start: float | None = None
    duration: float | None = None
    out_dir: str | None = None
    overwrite: bool = False
    seed: int = 0
    repeats: int = 1
    eval_iters: int = 12
    eval_downscale: int = 2
    eval_flow_dtype: str = "float32"
    eval_corr_dtype: str = "float32"
    eval_vf: str | None = None
    eval_tracks: bool = False         # opt-in: also measure the independent sparse-KLT estimator + agreement
    output_fps: str | None = None
    diag_every: int = 0
    dump_plan: bool = False            # capture the retained repeat's Plan as a compressed, non-pickle NPZ
    dump_motion: bool = False          # capture the retained repeat's fitted Motion as a compressed, non-pickle NPZ
    dump_flow_samples: bool = False    # capture restored flow vectors on the exact fitting grid (repeat 0 only)
    lock: str = "auto"                 # auto | on | off
    lock_chunk: int = 0
    lock_pause_s: float = 0.0
    synth_frames: int = 30
    synth_w: int = 320
    synth_h: int = 180
    synth_fps: float = 30.0
    synth_intent_mode: str = "smooth"
    synth_intent_profile: str = "historical"
    synth_layer_motion_factors: tuple = (1.0, 0.55, 0.25)
    synth_subject_mode: str = "moving"       # absent | static | moving
    synth_subject_speed: float = 1.6
    synth_jitter_px: float = 6.0
    synth_rot_jitter_deg: float = 0.6
    synth_rho: float = 0.8
    selection_note: str | None = None
    overrides: dict = field(default_factory=dict)
    command: list | None = None
    log: object = print


# --------------------------------------------------------------------------------------
# small media helpers
# --------------------------------------------------------------------------------------

def make_synthetic(path, W, H, T, seed, fps, *, intent_mode="smooth", layer_motion_factors=(1.0, 0.55, 0.25),
                   subject_mode="moving", subject_speed=1.6, jitter_px=6.0, rot_jitter_deg=0.6, rho=0.8,
                   intent_profile="historical"):
    has_subject = subject_mode != "absent"
    speed = 0.0 if subject_mode == "static" else float(subject_speed)
    depths = tuple(float(v) for v in layer_motion_factors)
    vid, intended, truth = synth.render(T=T, W=W, H=H, seed=seed, rho=rho, jitter=jitter_px,
                                       rot_jit=rot_jitter_deg, depths=depths, subject=has_subject,
                                       subject_speed=speed, intent_mode=intent_mode,
                                       intent_profile=intent_profile, return_truth=True)
    intended_fn, actual_fn = synth.camera(T, seed + 1, jitter_px, rot_jitter_deg, intent_mode=intent_mode,
                                          intent_profile=intent_profile)
    time = np.arange(T, dtype=np.float64)
    intended_pose = intended_fn(time)
    actual_pose = actual_fn(time)
    enc = media.Encoder(str(path), W, H, fps, crf=18, preset="veryfast")
    for f in vid:
        enc.write(f)
    enc.close()
    truth_path = Path(path).with_name("synthetic_truth.npz")
    np.savez_compressed(truth_path, visible_layer_id=np.asarray(truth["visible_layer_id"], np.uint8))
    return {"frames_requested": T, "width": W, "height": H, "fps": fps,
            "intent_mode": intent_mode,
            "intent_profile": intent_profile,
            "layer_motion_factors": list(depths),
            "layer_factor_semantics": "multipliers on camera translation; far layer fixed at 1.0 for factorial comparisons",
            "subject_mode": subject_mode, "subject_present": has_subject,
            "subject_speed_px_per_frame": speed if has_subject else None,
            "jitter_px": float(jitter_px), "rot_jitter_deg": float(rot_jitter_deg), "rolling_shutter_rho": float(rho),
            "intended_camera_pose_px_px_deg": intended_pose.tolist(),
            "actual_camera_pose_px_px_deg": actual_pose.tolist(),
            "camera_shake_rms_px_px_deg": np.sqrt(np.mean((actual_pose - intended_pose) ** 2, axis=0)).tolist(),
            "truth_file": truth_path.name, "truth_sha256": MF.sha256_file(truth_path),
            "truth_layer_ids": {"far": 0, "mid": 1, "near": 2, "subject": 3}}


def decode_meta(path, info, start, duration, vf=None):
    """Decode to count frames without keeping them; returns count, effective fps and duration."""
    frames, fps, (W, H) = media.decode(path, info, start, duration, vf=vf)
    n = sum(1 for _ in frames)
    return {"frames": int(n), "fps": float(fps), "width": int(W), "height": int(H),
            "duration_s": (n / fps) if fps else None}


def _normalized_timeline(n, fps, kind="frame_index_normalized"):
    """Frame-index time for a raw decode that exposes no source PTS (explicitly not original media PTS)."""
    last = ((n - 1) / fps) if (n and fps) else None
    return {"kind": kind, "frames": int(n), "first_s": 0.0 if n else None, "last_s": last}


def evaluate_clip(path, info, start, duration, vf, flow, mask_path=None, on_step=None, tracks_enabled=False, seed=0):
    """Measure one clip with a cold flow state. Returns (metrics, meta).

    The decode does not expose source PTS, so ``meta['timeline']`` is normalised frame-index time. Encoded
    output/mask timelines are measured separately with ``media.frame_pts``. With ``tracks_enabled`` the same
    single decode pass also feeds the independent sparse-KLT estimator, whose metrics are attached under
    ``metrics['tracks']``."""
    frames, fps, (W, H) = media.decode(path, info, start, duration, vf=vf)
    count = {"n": 0}

    def counted():
        for f in frames:
            count["n"] += 1
            yield f

    masks = None
    if mask_path is not None:
        mi = media.probe(mask_path)
        mf, _, _ = media.decode(mask_path, mi, vf=vf)

        def safe_masks():
            for m in mf:
                yield (m[..., 0] > 127)
        masks = safe_masks()
    flow.reset()
    acc = tracks.TrackAccumulator(W, H, fps, seed=seed) if tracks_enabled else None
    track_masks = None
    if acc is not None and masks is not None:
        track_masks, masks = tee(masks)
    evaluated_frames = acc.wrap(counted(), track_masks) if acc else counted()
    m = metrics.measure(evaluated_frames, flow, W, H, fps, masks=masks, on_step=on_step)
    if acc:
        m["tracks"] = acc.metrics()
    meta = {"frames": count["n"], "fps": float(fps), "width": int(W), "height": int(H),
            "duration_s": (count["n"] / fps) if fps else None,
            "timeline": _normalized_timeline(count["n"], fps)}
    return m, meta


# --------------------------------------------------------------------------------------
# option validation and run directory
# --------------------------------------------------------------------------------------

def validate_options(opts: RunOptions) -> list[str]:
    from fractions import Fraction
    errs = []
    for label, value in (("output-fps", opts.output_fps), ("eval-vf", opts.eval_vf)):
        if value is None or (label == "output-fps" and value == "native"):
            continue
        try:
            rate = str(value)
            if label == "eval-vf":
                if not rate.startswith("fps="):
                    raise ValueError("expected fps=RATE")
                rate = rate[4:]
            if Fraction(rate) <= 0:
                raise ValueError("non-positive rate")
        except (ValueError, ZeroDivisionError):
            errs.append(f"--{label} requires a positive frame rate" + (" as fps=RATE" if label == "eval-vf" else " or native"))
    if opts.repeats < 1:
        errs.append("--repeats must be >= 1")
    if opts.eval_iters < 1:
        errs.append("--eval-iters must be >= 1")
    if opts.eval_downscale < 1:
        errs.append("--eval-downscale must be >= 1")
    if opts.eval_tracks and not tracks.has_cv2():
        errs.append("--eval-tracks needs OpenCV (cv2), which is not available in this environment")
    if opts.diag_every < 0:
        errs.append("--diag-every must be >= 0")
    if opts.lock not in ("auto", "on", "off"):
        errs.append("--lock must be auto|on|off")
    if opts.lock_chunk < 0:
        errs.append("--lock-chunk must be >= 0")
    if opts.lock_pause_s < 0:
        errs.append("--lock-pause must be >= 0")
    if opts.duration is not None and opts.duration <= 0:
        errs.append("--duration must be > 0")
    if opts.start is not None and opts.start < 0:
        errs.append("--start must be >= 0")
    if opts.synth_frames < 1 and opts.source == "synthetic":
        errs.append("--synth-frames must be >= 1")
    if opts.source == "synthetic":
        if opts.synth_w < 1 or opts.synth_h < 1:
            errs.append("synthetic width/height must be positive")
        if not np.isfinite(opts.synth_fps) or opts.synth_fps <= 0:
            errs.append("--synth-fps must be finite and > 0")
        if opts.synth_intent_mode not in ("smooth", "stochastic", "static"):
            errs.append("--synth-intent-mode must be smooth|stochastic|static")
        if opts.synth_intent_profile not in ("historical", "matched-spectrum"):
            errs.append("--synth-intent-profile must be historical|matched-spectrum")
        if opts.synth_intent_profile == "matched-spectrum" and opts.synth_frames < 16:
            errs.append("matched-spectrum intent requires --synth-frames >= 16")
        if opts.synth_subject_mode not in ("absent", "static", "moving"):
            errs.append("--synth-subject-mode must be absent|static|moving")
        try:
            factors = tuple(float(v) for v in opts.synth_layer_motion_factors)
            if len(factors) != 3 or not np.isfinite(factors).all() or any(v <= 0 for v in factors):
                raise ValueError
            if not np.isclose(factors[0], 1.0):
                errs.append("the synthetic far-layer motion factor must remain 1.0 to isolate parallax separation")
        except (TypeError, ValueError):
            errs.append("synthetic layer motion factors must be three finite positive values")
        if not np.isfinite(opts.synth_subject_speed) or opts.synth_subject_speed < 0:
            errs.append("--synth-subject-speed must be finite and >= 0")
        if not np.isfinite(opts.synth_jitter_px) or opts.synth_jitter_px < 0:
            errs.append("--synth-jitter must be finite and >= 0")
        if not np.isfinite(opts.synth_rot_jitter_deg) or opts.synth_rot_jitter_deg < 0:
            errs.append("--synth-rot-jitter must be finite and >= 0")
        if not np.isfinite(opts.synth_rho) or not 0 <= opts.synth_rho <= 1:
            errs.append("--synth-rho must be in [0,1]")
    if opts.source != "synthetic" and not Path(opts.source).exists():
        errs.append(f"source not found: {opts.source}")
    if opts.backend not in ("mlx", "torch-mps"):
        errs.append("--backend must be mlx|torch-mps")
    if not Path(opts.weights).exists():
        errs.append(f"weights not found: {opts.weights}")
    if opts.backend == "torch-mps":
        if not Path(opts.torch_weights).exists():
            errs.append(f"PyTorch RAFT weights not found: {opts.torch_weights}")
    return errs


def _sanitize(s):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s)).strip("_")[:60]


def _fmt(v):
    return "none" if v is None else f"{v:g}"


def default_run_dir(case_name, source, start, duration):
    tag = "synthetic" if source == "synthetic" else _sanitize(Path(source).stem)
    win = f"_t{_fmt(start)}+{_fmt(duration)}" if (start is not None or duration is not None) else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")     # real microseconds: unique within a second
    return DEFAULT_OUT / f"{stamp}_{_sanitize(case_name)}_{tag}{win}"


# --------------------------------------------------------------------------------------
# manifest skeleton and checkpoints
# --------------------------------------------------------------------------------------

def _empty_manifest(opts, run_dir):
    return {
        "schema": MF.MANIFEST_SCHEMA,
        "run_id": run_dir.name,
        "created": MF.now_iso(),
        "updated": MF.now_iso(),
        "stage": "init",
        "status": "running",
        "error": None,
        "case": None,
        "source": {"path": None, "sha256": None, "probe": None,
                   "window": {"start_s": opts.start, "duration_s": opts.duration},
                   "selection_note": opts.selection_note, "preprocessing": None, "decoded": None},
        "weights": {"path": str(opts.weights), "sha256": None, "size_bytes": None},
        "flow_backend": {"name": opts.backend, "weights_path": None, "weights_sha256": None,
                         "scope": "flow backend only; planner, renderer, media I/O and evaluation are shared mlxstab code"},
        "config": None,
        "config_digest": None,
        "flow_downscale": None,
        "experiment_identity": None,
        "content_key": None,
        "code": None,
        "environment": None,
        "hardware": None,
        "seed_policy": None,
        "evaluation": None,
        "command": list(opts.command if opts.command is not None else sys.argv),
        "timing": {"provenance_s": None, "model_construct_s": None, "repeat0_total_s": None,
                   "end_to_end_stabilization_s": None, "end_to_end_with_evaluation_s": None,
                   "total_wall_s": None, "lock_scope": None},
        "runtime_s": {"stabilize_wall": None, "stages": None, "evaluation": None,
                      "diagnostics_total": None, "observed_nondiag_render": None},
        "memory": None,
        "metrics": {"definitions_version": None, "input": None, "output": None,
                    "cadence": None, "reliability": None},
        "diagnostics": None,
        "outputs": {"video": None, "mask": None, "report": None, "plan": None, "motion": None, "flow_samples": None,
                    "retained_repeat": 0, "manifest": None,
                    "encoded_frames": None, "encoded_pts": None, "mask_pts": None},
        "repeats": [],
        "lock": None,
        "unavailable": {},
        "notes": [],
    }


def _checkpoint(man, path, stage, required=False):
    """Write a running manifest. Intermediate writes are best effort; initial/final writes must surface."""
    man["stage"] = stage
    man["updated"] = MF.now_iso()
    if required:
        MF.write_manifest(path, man)
        return
    try:
        MF.write_manifest(path, man)
    except Exception:  # noqa: BLE001 - an intermediate checkpoint must not mask the real error
        pass


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def _experiment_identity(opts, cfg_kwargs, source_hashes, weights_sha, code, eval_cfg, output_fps, eval_vf, diag_every,
                         dump_plan=False, dump_motion=False, dump_flow_samples=False, flow_downscale=None):
    identity = {
        "source": {"sha256": source_hashes.get("sha256"), "window": source_hashes.get("window"),
                   "preprocessing": source_hashes.get("preprocessing"),
                   "selection_note": source_hashes.get("selection_note")},
        "weights_sha256": weights_sha,
        "flow_backend": {"name": opts.backend},
        "config": cfg_kwargs,
        "output_cadence": output_fps,
        "evaluation": eval_cfg,
        "seed_policy": {"config_seed": cfg_kwargs.get("seed"), "synthetic_seed": opts.seed},
        "diagnostics": {"every": int(diag_every)},
        "capture": {"dump_plan": bool(dump_plan), "dump_motion": bool(dump_motion),
                    "dump_flow_samples": bool(dump_flow_samples)},
        "code": {"revision": (code or {}).get("revision"), "files": (code or {}).get("files"),
                 "dirty_relevant": (code or {}).get("dirty_relevant")},
    }
    if flow_downscale is not None:
        identity["flow_downscale"] = flow_downscale
    return identity


def run_case(opts: RunOptions) -> dict:
    run_dir = Path(opts.out_dir) if opts.out_dir else default_run_dir(opts.case, opts.source, opts.start, opts.duration)
    man_path = run_dir / "manifest.json"
    plan_path = run_dir / "plan_r0.npz" if opts.dump_plan else None
    motion_path = run_dir / "motion_r0.npz" if opts.dump_motion else None
    flow_samples_path = run_dir / "flow_samples_r0.npz" if opts.dump_flow_samples else None
    if not opts.overwrite and man_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence {man_path}; pass --overwrite to reuse it")
    if plan_path is not None and not opts.overwrite and plan_path.exists():
        raise FileExistsError(f"refusing to overwrite existing plan dump {plan_path}; pass --overwrite to reuse it")
    if motion_path is not None and not opts.overwrite and motion_path.exists():
        raise FileExistsError(f"refusing to overwrite existing motion dump {motion_path}; pass --overwrite to reuse it")
    if flow_samples_path is not None and not opts.overwrite and flow_samples_path.exists():
        raise FileExistsError(f"refusing to overwrite existing flow-sample dump {flow_samples_path}; pass --overwrite to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)
    man = _empty_manifest(opts, run_dir)
    man["outputs"]["manifest"] = str(man_path)
    _checkpoint(man, man_path, "validate", required=True)     # initial persistence must surface

    lock = None
    cancelled = None
    stage = "validate"
    t_start = time.perf_counter()
    try:
        case = cases.resolve_case(opts.case, opts.overrides)
        man["case"] = {"name": case.name, "description": case.description, "changed": case.changed,
                       "eval_vf": None, "output_fps": None}
        errs = validate_options(opts)
        output_fps = opts.output_fps if opts.output_fps is not None else case.output_fps
        if output_fps == "native":
            output_fps = None
        eval_vf = opts.eval_vf if opts.eval_vf is not None else case.eval_vf
        cfg_kwargs = case.resolved_config()
        cfg = Config(**cfg_kwargs)
        if opts.backend == "torch-mps" and (cfg.flow_dtype != "float32" or cfg.corr_dtype != "float32"):
            errs.append("torch-mps backend supports only float32 flow and correlation")
        if opts.backend == "torch-mps" and cfg.flow_downscale == "auto":
            errs.append("torch-mps backend does not support flow_downscale='auto' (the auto budget is defined for the "
                        "built-in MLX RAFT path); use a fixed scale")
        if cfg.warp_diag and opts.diag_every > 0:
            errs.append("choose either warp_diag (all frames) or --diag-every (sampled), not both")
        if errs:
            raise ValueError("invalid options:\n- " + "\n- ".join(errs))
        man["case"] = {"name": case.name, "description": case.description, "changed": case.changed,
                       "eval_vf": eval_vf, "output_fps": output_fps}
        man["config"] = cfg_kwargs
        man["config_digest"] = cases.config_digest(cfg_kwargs)
        man["seed_policy"] = {"config_seed": cfg_kwargs.get("seed"), "synthetic_seed": opts.seed if opts.source == "synthetic" else None,
                              "note": "config.seed seeds per-clip RANSAC fitting; synthetic_seed seeds synthetic clip generation"}

        stage = "provenance"
        tp = time.perf_counter()
        man["code"] = MF.code_fingerprint(ROOT)
        man["environment"] = MF.env_versions()
        man["metrics"]["definitions_version"] = MF.metrics_digest(ROOT)["version"]
        man["hardware"] = MF.hardware_info()
        man["timing"]["provenance_s"] = time.perf_counter() - tp

        eval_cfg_kwargs = {"mesh": [0, 0], "rs_rows": 18, "model": "similarity", "step": 8, "seed": 0}
        track_eval_params = {**tracks.DEFAULTS, "seed": opts.seed}
        man["evaluation"] = {
            "flow_iters": opts.eval_iters, "flow_downscale": opts.eval_downscale,
            "flow_dtype": opts.eval_flow_dtype, "corr_dtype": opts.eval_corr_dtype,
            "reset": "per_clip", "vf": eval_vf, "metrics_version": man["metrics"]["definitions_version"],
            "metric_config": eval_cfg_kwargs,
            "independent_tracks": ({"enabled": True, "estimator": "klt_similarity", "params": track_eval_params}
                                   if opts.eval_tracks else {"enabled": False}),
            "note": ("evaluation uses a fresh RaftFlow reset at each clip boundary; mesh disabled and rows from Config. "
                     "Input timeline is normalised frame-index time; output/mask timelines are measured ffprobe PTS."),
        }
        _checkpoint(man, man_path, "provenance")

        enabled_lock = opts.lock == "on" or (opts.lock == "auto" and opts.source != "synthetic")
        lock = HeavyLock(what=f"mlxstab bench {case.name}", enabled=enabled_lock, chunk=opts.lock_chunk,
                         pause_s=opts.lock_pause_s)

        stage = "source"
        if opts.source == "synthetic":
            src = run_dir / "synthetic.mp4"
            man["source"]["synthetic"] = make_synthetic(
                src, opts.synth_w, opts.synth_h, opts.synth_frames, opts.seed, opts.synth_fps,
                intent_mode=opts.synth_intent_mode, layer_motion_factors=opts.synth_layer_motion_factors,
                subject_mode=opts.synth_subject_mode, subject_speed=opts.synth_subject_speed,
                jitter_px=opts.synth_jitter_px, rot_jitter_deg=opts.synth_rot_jitter_deg, rho=opts.synth_rho,
                intent_profile=opts.synth_intent_profile)
            man["source"]["window"] = {"start_s": 0.0, "duration_s": opts.synth_frames / opts.synth_fps}
        else:
            src = Path(opts.source)
        man["source"]["path"] = str(src)
        man["source"]["sha256"] = MF.sha256_file(src)
        info = media.probe(src)
        man["source"]["probe"] = info
        man["source"]["path_type"] = "synthetic" if opts.source == "synthetic" else "file"
        man["source"]["preprocessing"] = {
            "interlaced": bool(info.get("interlaced")),
            "field_order": info.get("field_order"),
            "deinterlace": "bwdif=mode=send_field:parity=auto:deint=all" if info.get("interlaced") else None,
            "processing_fps": info.get("fps") * (2 if info.get("interlaced") else 1),
            "decode_vf": None,
        }
        man["weights"]["sha256"] = MF.sha256_file(opts.weights)
        man["weights"]["size_bytes"] = Path(opts.weights).stat().st_size
        flow_weights_path = opts.torch_weights if opts.backend == "torch-mps" else opts.weights
        man["flow_backend"].update({"weights_path": str(flow_weights_path),
                                    "weights_sha256": MF.sha256_file(flow_weights_path)})
        if opts.backend == "torch-mps":
            from .torch_flow import verify_weight_equivalence
            man["flow_backend"]["weight_equivalence"] = verify_weight_equivalence(opts.weights, opts.torch_weights)

        stage = "resolve_flow_downscale"
        _checkpoint(man, man_path, stage)
        resolution = resolve_flow_downscale(cfg, info["width"], info["height"],
                                            custom_flow=opts.backend == "torch-mps")
        cfg.flow_downscale = resolution.resolved          # the requested config stays in cfg_kwargs/man["config"]
        man["flow_downscale"] = resolution.manifest_fields()

        identity = _experiment_identity(opts, cfg_kwargs, man["source"], man["weights"]["sha256"], man["code"],
                                        man["evaluation"], output_fps, eval_vf, opts.diag_every, opts.dump_plan,
                                        opts.dump_motion, opts.dump_flow_samples,
                                        flow_downscale=resolution.manifest_fields())
        identity["flow_backend"]["weights_sha256"] = man["flow_backend"]["weights_sha256"]
        identity["flow_backend"]["canonical_weights_sha256"] = (
            man["flow_backend"].get("weight_equivalence", {}).get("canonical_sha256"))
        man["experiment_identity"] = identity
        man["content_key"] = MF.content_key(identity)
        _checkpoint(man, man_path, "source")

        stage = "lock"
        lock.acquire()
        man["lock"] = lock.status()
        _checkpoint(man, man_path, "lock")

        stage = "model_construct"
        _checkpoint(man, man_path, "model_construct")   # persist before the long RAFT construction
        tmc = time.perf_counter()
        if opts.backend == "torch-mps":
            from .torch_flow import TorchMPSFlow
            flow = TorchMPSFlow(opts.torch_weights, cfg.flow_iters, cfg.flow_downscale, cfg.warm_start,
                                getattr(mx, cfg.corr_dtype), getattr(mx, cfg.flow_dtype), cfg.warm_iters or None)
        else:
            flow = RaftFlow(opts.weights, cfg.flow_iters, cfg.flow_downscale, cfg.warm_start,
                            getattr(mx, cfg.corr_dtype), getattr(mx, cfg.flow_dtype), cfg.warm_iters or None)
        man["timing"]["model_construct_s"] = time.perf_counter() - tmc
        gate = lock.on_step if opts.lock_chunk else None

        diag_records = []

        def diag_cb(t, t_render, t_diag, dg):
            diag_records.append({"frame": int(t), "render_s": float(t_render), "diagnose_s": float(t_diag),
                                 "diagnostic": dg})

        stage = "stabilize"
        repeat_walls = []
        for r in range(max(int(opts.repeats), 1)):
            out_video = run_dir / f"stabilized_r{r}.mp4"
            out_mask = run_dir / f"mask_r{r}.mp4"
            rep_path = run_dir / f"report_r{r}.json"
            try:
                mx.reset_peak_memory()
            except Exception:
                pass
            if opts.log:
                opts.log(f"[{case.name}] repeat {r} ({'model cold-load' if r == 0 else 'model cached'}) -> {out_video}")
            t0 = time.perf_counter()
            flow.reset()
            cb = diag_cb if (opts.diag_every and r == 0) else None
            # Only the retained repeat (0) captures the plan; warm repeats must never clobber plan_r0.npz.
            rep_dump_plan = str(plan_path) if (plan_path is not None and r == 0) else None
            rep_dump_motion = str(motion_path) if (motion_path is not None and r == 0) else None
            stage = f"stabilize_r{r}"
            _checkpoint(man, man_path, stage)          # persist the precise repeat phase before the long pass
            rep = stabilize(str(src), str(out_video), str(out_mask), opts.weights, cfg, flow_fn=flow,
                            start=opts.start, duration=opts.duration, fps_out=output_fps,
                            report=str(rep_path), on_step=gate, diag_every=opts.diag_every if cb else 0,
                            diag_callback=cb, dump_motion_npz=rep_dump_motion, dump_plan=rep_dump_plan,
                            dump_flow_samples_npz=str(flow_samples_path) if (flow_samples_path is not None and r == 0) else None,
                            log=(lambda m: opts.log(f"[{case.name}] {m}")) if opts.log else (lambda m: None))
            rep.update(resolution.report_fields())
            Path(rep_path).write_text(json.dumps(rep, indent=1))
            wall = time.perf_counter() - t0
            repeat_walls.append(wall)
            man["repeats"].append({
                "index": r, "model_state": "cold-load" if r == 0 else "cached",
                "temporal_state": "reset (first pair cold)", "wall_s": wall,
                "video": str(out_video), "mask": str(out_mask), "report": str(rep_path),
                "rendered_frames": rep.get("rendered_frames"), "decoded_frames": rep.get("decoded_frames"),
                "stabilize_report": rep,
            })
            if r == 0:
                man["runtime_s"]["stages"] = dict(rep.get("timing_s", {}))
                man["runtime_s"]["diagnostics_total"] = rep.get("diagnostics_total_s")
                man["runtime_s"]["observed_nondiag_render"] = rep.get("observed_nondiag_render_s")
                man["outputs"]["video"] = str(out_video)
                man["outputs"]["mask"] = str(out_mask)
                man["outputs"]["report"] = str(rep_path)
                if plan_path is not None:
                    man["outputs"]["plan"] = str(plan_path)
                if motion_path is not None:
                    man["outputs"]["motion"] = str(motion_path)
                if flow_samples_path is not None:
                    man["outputs"]["flow_samples"] = str(flow_samples_path)
                man["outputs"]["retained_repeat"] = 0
            _checkpoint(man, man_path, f"stabilize_r{r}")
        man["runtime_s"]["stabilize_wall"] = sum(repeat_walls)
        man["timing"]["repeat0_total_s"] = repeat_walls[0]
        man["timing"]["end_to_end_stabilization_s"] = man["timing"]["model_construct_s"] + repeat_walls[0]
        man["timing"]["lock_scope"] = ("lock acquisition waits and chunk pauses are included in total_wall_s and may fall "
                                       "inside evaluation/diagnostic wall times; they are reported separately in "
                                       "lock.acquire_wait_s/pause_total_s and are not subtracted from stabilization")

        stage = "probe_output"
        _checkpoint(man, man_path, "probe_output")     # persist before probing/decoding the encoded outputs
        out_video = Path(man["outputs"]["video"])
        out_mask = Path(man["outputs"]["mask"])
        out_info = media.probe(out_video)
        out_meta = decode_meta(out_video, None, None, None)
        out_pts = media.frame_pts(out_video)
        mask_info = media.probe(out_mask)
        mask_pts = media.frame_pts(out_mask)
        man["source"]["decoded"] = decode_meta(src, info, opts.start, opts.duration)
        man["outputs"]["probe"] = out_info
        man["outputs"]["decoded"] = out_meta
        man["outputs"]["encoded_frames"] = out_meta.get("frames")
        man["outputs"]["encoded_pts"] = out_pts
        man["outputs"]["mask_pts"] = mask_pts
        from fractions import Fraction
        requested_rate = float(Fraction(str(output_fps))) if output_fps else man["source"]["preprocessing"]["processing_fps"]
        encoded_cadence = MF.check_encoded_cfr(out_pts, mask_pts, out_info["fps"], mask_info["fps"], requested_rate)
        man["outputs"]["timeline_alignment"] = encoded_cadence
        man["outputs"]["cadence"] = {"requested_fps": output_fps, "measured_fps": out_info.get("fps"),
                                     "measured_frames": out_meta.get("frames"), "measured_duration_s": out_meta.get("duration_s"),
                                     "requested_eval_vf": eval_vf}
        man["repeats"][0]["encoded_frames"] = out_meta.get("frames")

        stage = "evaluate"
        _checkpoint(man, man_path, "evaluate")         # persist before evaluation waits on HeavyLock
        ev_t0 = time.perf_counter()
        eval_flow = RaftFlow(opts.weights, opts.eval_iters, opts.eval_downscale, True,
                             getattr(mx, opts.eval_corr_dtype), getattr(mx, opts.eval_flow_dtype), None)
        m_in, meta_in = evaluate_clip(str(src), info, opts.start, opts.duration, eval_vf, eval_flow,
                                      on_step=gate, tracks_enabled=opts.eval_tracks, seed=opts.seed)
        m_out, meta_out = evaluate_clip(str(out_video), out_info, None, None, eval_vf, eval_flow,
                                        mask_path=str(out_mask), on_step=gate,
                                        tracks_enabled=opts.eval_tracks, seed=opts.seed)
        # Count the complete filtered mask, including any tail the metric loop did not consume.
        # Native output may be 59.94 fps while evaluation is 29.97 fps: compare each at its own clock.
        mask_meta = decode_meta(out_mask, mask_info, None, None, vf=eval_vf)
        mask_meta["timeline"] = _normalized_timeline(mask_meta["frames"], mask_meta["fps"])
        man["metrics"]["input"] = {**m_in, **meta_in}
        man["metrics"]["output"] = {**m_out, **meta_out}
        man["metrics"]["cadence"] = MF.check_cadence(meta_in, meta_out, mask=mask_meta)
        man["metrics"]["cadence"]["encoded_alignment"] = encoded_cadence
        if not encoded_cadence["aligned"]:
            man["metrics"]["cadence"]["aligned"] = False
            man["metrics"]["cadence"]["reasons"].extend(encoded_cadence["reasons"])
        man["metrics"]["cadence"]["timing_origin"] = ("input timeline is normalised frame-index time (raw decode exposes no "
                                                      "source PTS); evaluation streams share that normalized clock. Every encoded "
                                                      "video/mask PTS is separately checked at the encoded frame rate. Original "
                                                      "source PTS and VFR timing are not verified by this CFR benchmark.")
        man["metrics"]["reliability"] = MF.reliability(m_in, m_out, man["metrics"]["cadence"])
        if opts.eval_tracks:
            man["metrics"]["evaluator_agreement"] = tracks.agreement(m_in, m_out, m_in.get("tracks"), m_out.get("tracks"))
            man["metrics"]["evaluator_agreement"]["note"] = (
                "independent sparse-KLT estimator (mlxstab.tracks) vs the RAFT metrics; a cross-check between two "
                "estimators, not ground truth. KLT fails on low texture, blur and large displacement.")
        man["runtime_s"]["evaluation"] = time.perf_counter() - ev_t0
        man["timing"]["end_to_end_with_evaluation_s"] = man["timing"]["end_to_end_stabilization_s"] + man["runtime_s"]["evaluation"]

        stage = "diagnostics"
        if opts.diag_every > 0:
            overheads = [r["diagnose_s"] for r in diag_records]
            man["diagnostics"] = {
                "mode": "sampled", "enabled": True, "every": int(opts.diag_every),
                "frames": [r["frame"] for r in diag_records],
                "per_frame": diag_records,
                "total_overhead_s": float(sum(overheads)) if overheads else 0.0,
                "mean_overhead_s": float(sum(overheads) / len(overheads)) if overheads else None,
                "max_overhead_s": float(max(overheads)) if overheads else None,
                "aggregate": warpmetrics.aggregate([r["diagnostic"] for r in diag_records]) if diag_records else None,
                "note": ("sampled from the actual full-window plan during the retained repeat; recorded frames are exact "
                         "render indices. Per-sample maxima are not whole-video guarantees. render minus this total is "
                         "observed non-diagnostic interval time, not counterfactual throughput (encoding overlaps)."),
            }
        else:
            man["diagnostics"] = {"mode": "off", "enabled": False, "frames": [],
                                  "unavailable_reason": "complete-warp diagnostics disabled for this run (use --diag-every K)"}
            man["unavailable"]["diagnostics.overhead_s"] = "diagnostics disabled for this run; no overhead measured"

        man["memory"] = MF.peak_memory()
        man["memory"]["scope"] = ("process_peak_rss_bytes is lifetime peak RSS for this process (includes interpreter, "
                                  "model load and evaluation); mlx_peak_active_bytes is the MLX allocator peak since the "
                                  "last reset (reset per repeat).")
        man["status"] = "success"
    except BaseException as e:  # noqa: BLE001 - every failure/cancellation must be reviewable
        man["status"] = "cancelled" if isinstance(e, (KeyboardInterrupt, SystemExit)) else "failure"
        man["error"] = {"type": type(e).__name__, "message": str(e), "stage": stage}
        cancelled = e if isinstance(e, (KeyboardInterrupt, SystemExit)) else None
        if opts.log:
            opts.log(f"[{opts.case}] {man['status']} at {stage}: {type(e).__name__}: {e}")
    finally:
        if lock is not None:
            lock.release()

    man["timing"]["total_wall_s"] = time.perf_counter() - t_start
    if lock is not None:
        man["lock"] = lock.status()
    if man["status"] != "success":
        for k, reason in (("metrics", "run did not reach evaluation"),
                          ("memory", "run did not reach memory measurement"),
                          ("diagnostics", "run did not reach diagnostics")):
            man.setdefault("unavailable", {}).setdefault(k, reason)
    _checkpoint(man, man_path, man["stage"] if man["status"] == "success" else man["status"], required=True)
    if cancelled is not None:
        raise cancelled
    return man
