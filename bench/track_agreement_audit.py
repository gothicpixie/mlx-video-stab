"""Offline cross-check of the independent KLT estimator against a captured RAFT motion payload.

The stabilizer's RAFT flow is currently the sole judge of jitter/wobble in the benchmark. This audit compares
it with the independent sparse-KLT estimator (``mlxstab.tracks``) on the **same decoded real frames**, using a
captured ``motion_r0.npz`` from a benchmark run as the RAFT reference. It runs **no RAFT inference, no media
decode and no production code**; it reads a decoded-frame NPZ and a motion payload, and writes a path-free,
atomic JSON report under ignored ``out/``.

Scope: this is a cross-check between two *estimators*, not ground truth. A small per-pair transform gap means
the two methods see the same camera motion; it does not establish that either is correct, that the clip is
representative, or that stabilization improved. KLT fails on low texture, blur and large displacement. No
threshold is selected, no winner is declared and no production default changes.

    python -m bench.track_agreement_audit --frames-npz F.npz --motion-npz motion_r0.npz --out R.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from bench import manifest as MF
from bench.real_composition_audit import MotionPayloadError, load_motion_npz
from mlxstab import geometry as G, metrics as M, tracks

SCHEMA = "mlxstab.track_agreement_audit/1"
#: Five stable probe points spanning the frame; the transform comparison is a displacement check at these points.
_PROBE_FRACTION = np.array([[0.05, 0.05], [0.95, 0.05], [0.5, 0.5], [0.05, 0.95], [0.95, 0.95]])


def _frames_sha256(frames):
    h = hashlib.sha256()
    for frame in frames:
        h.update(np.ascontiguousarray(frame).tobytes())
    return h.hexdigest()


def _probe_points(W, H):
    return (_PROBE_FRACTION - 0.5) * np.array([W - 1.0, H - 1.0]) + 0.5


def _finite_stats(values):
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max()),
            "count": int(v.size)}


def _similarity_of(Hm, W, H):
    """Decompose a transform's action on the frame corners into a similarity scale and rotation (degrees, y-down)."""
    cs = G.corners(W, H)
    a, _t, _res = G.similarity_fit(cs - cs.mean(0), G.apply_h(Hm, cs) - cs.mean(0))
    return float(abs(a)), float(np.rad2deg(np.angle(a)))


def run_audit(frames, motion, tracker_kwargs=None):
    """Cross-check KLT per-pair transforms against ``motion`` (a reconstructed ``mlxstab.stabilize.Motion``)."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"frames must be [T,H,W,3] uint8, got {frames.shape}")
    if frames.shape[0] != motion.n_frames:
        raise ValueError(f"frame count {frames.shape[0]} != motion.n_frames {motion.n_frames}")
    if (frames.shape[2], frames.shape[1]) != (motion.W, motion.H):
        raise ValueError(f"frame size {frames.shape[2]}x{frames.shape[1]} != motion {motion.W}x{motion.H}")
    if len(motion.fits) == 0:
        raise ValueError("motion payload has no fitted pairs")

    acc = tracks.TrackAccumulator(motion.W, motion.H, motion.fps, **(tracker_kwargs or {}))
    for f in frames:
        acc.push(f)
    if len(acc.transforms) != len(motion.fits):
        raise ValueError(f"track pairs {len(acc.transforms)} != motion fits {len(motion.fits)}")

    pts = _probe_points(motion.W, motion.H)
    rd = np.array([G.apply_h(f.H, pts) - pts for f in motion.fits])
    td = np.array([G.apply_h(t[0], pts) - pts for t in acc.transforms])
    epe = np.linalg.norm(rd - td, axis=2)

    scale_d, rot_d = [], []
    for f, (Ht, _fb, _inl, _n) in zip(motion.fits, acc.transforms):
        s_r, r_r = _similarity_of(f.H, motion.W, motion.H)
        s_t, r_t = _similarity_of(Ht, motion.W, motion.H)
        scale_d.append(abs(s_t - s_r)); rot_d.append(abs(r_t - r_r))

    jr, fmr = M.jitter(motion)
    tm = acc.metrics()
    return {
        "schema": SCHEMA,
        "window": {"frames": int(frames.shape[0]), "width": int(motion.W), "height": int(motion.H),
                   "fps": float(motion.fps), "pairs": len(motion.fits)},
        "track": {"method": tm["method"], "pairs": tm["support_frames"], "fallbacks": tm["fallback_frames"],
                  "inlier_frac_mean": tm["inlier_frac_mean"], "tracks_mean": tm["tracks_mean"],
                  "jitter_px": tm["jitter_px"], "stability": tm["stability"]},
        "raft": {"jitter_px": jr, "frame_motion_px": fmr},
        "agreement": {
            "point_displacement_epe_px": _finite_stats(epe),
            "jitter_relative_gap": (float((tm["jitter_px"] - jr) / jr)
                                   if (jr is not None and jr >= tracks.MIN_RATIO_JITTER_PX) else None),
            "scale_delta_abs": _finite_stats(scale_d),
            "rotation_delta_deg_abs": _finite_stats(rot_d),
        },
        "note": ("Cross-check between the independent sparse-KLT estimator and the captured RAFT fit, not ground "
                 "truth. A small transform gap means both describe the same camera motion on these frames."),
        "limitations": [
            "the RAFT reference is the stabilizer's own fitted global homography, not independent ground truth",
            "KLT fails on low texture, motion blur, large displacement and rolling shutter",
            "one decoded window; pairs are serially dependent and not an independent sample",
            "no threshold is selected, no winner declared and no production default changes",
        ],
    }


def _verify_manifest(manifest_path, motion_path, frames_meta, motion):
    """Verify that a cached frame window and captured motion belong to one successful run/source window."""
    if not frames_meta:
        raise ValueError("frame cache has no metadata; cannot verify its source/window against the manifest")
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except Exception as exc:  # noqa: BLE001 - CLI converts to a stable mismatch result
        raise ValueError(f"manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("status") != "success":
        raise ValueError("manifest status is not success")
    source = manifest.get("source") or {}
    window = source.get("window") or {}
    cached_window = frames_meta.get("window") or {}
    for key in ("source_sha256", "frames_sha256"):
        value = frames_meta.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"frame cache metadata missing {key}")
    if source.get("sha256") != frames_meta["source_sha256"]:
        raise ValueError("frame cache source hash does not match run manifest")
    for key in ("start_s", "duration_s"):
        if key not in window or key not in cached_window or abs(float(window[key]) - float(cached_window[key])) > 1e-6:
            raise ValueError(f"frame cache {key} does not match run manifest")
    decoded = source.get("decoded") or {}
    expected = {"frames": motion.n_frames, "width": motion.W, "height": motion.H}
    for key, value in expected.items():
        if int(decoded.get(key, -1)) != int(value) or int(frames_meta.get({"frames": "count", "width": "width", "height": "height"}[key], -1)) != int(value):
            raise ValueError(f"frame cache/manifest {key} does not match captured motion")
    if (abs(float(decoded.get("fps", -1)) - motion.fps) > 1e-6 or
            abs(float(frames_meta.get("fps", -1)) - motion.fps) > 1e-6):
        raise ValueError("frame cache/manifest fps does not match captured motion")
    outputs = manifest.get("outputs") or {}
    motion_name = outputs.get("motion")
    if not motion_name or Path(motion_name).name != Path(motion_path).name:
        raise ValueError("captured motion path does not match manifest outputs.motion")
    expected_motion = Path(manifest_path).resolve().parent / Path(motion_name).name
    if expected_motion != Path(motion_path).resolve():
        raise ValueError("captured motion is not in the manifest run directory")
    return {"schema": frames_meta.get("schema"), "source_sha256": frames_meta["source_sha256"],
            "frames_sha256": frames_meta["frames_sha256"],
            "window": {"start_s": float(cached_window["start_s"]),
                       "duration_s": float(cached_window["duration_s"])}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-npz", required=True, help="decoded-frame NPZ with a 'frames' [T,H,W,3] uint8 array")
    ap.add_argument("--motion-npz", required=True, help="benchmark motion_r0.npz for the same window")
    ap.add_argument("--manifest", help="successful run_case manifest proving source/window correspondence")
    ap.add_argument("--out", required=True, help="path-free JSON report destination")
    ap.add_argument("--track-seed", type=int, default=0)
    ap.add_argument("--max-corners", type=int, default=tracks.DEFAULTS["max_corners"])
    a = ap.parse_args(argv)

    try:
        z = np.load(a.frames_npz, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - report a stable code instead of a traceback
        print(json.dumps({"status": "error", "code": "frames_unreadable", "message": str(exc)}), file=sys.stderr)
        return 2
    if "frames" not in z.files:
        print(json.dumps({"status": "error", "code": "frames_missing"}), file=sys.stderr)
        return 2
    frames = z["frames"]
    frames_meta = None
    if "meta" in z.files:
        try:
            frames_meta = json.loads(str(z["meta"].item()))
        except Exception as exc:  # noqa: BLE001 - report a stable input error instead of a traceback
            print(json.dumps({"status": "error", "code": "frames_meta_invalid", "message": str(exc)}), file=sys.stderr)
            return 2
        if not isinstance(frames_meta, dict):
            print(json.dumps({"status": "error", "code": "frames_meta_invalid"}), file=sys.stderr)
            return 2
        if frames_meta.get("frames_sha256") != _frames_sha256(frames):
            print(json.dumps({"status": "error", "code": "frames_digest_mismatch"}), file=sys.stderr)
            return 2
    try:
        motion, meta = load_motion_npz(a.motion_npz)
    except MotionPayloadError as exc:
        print(json.dumps({"status": "error", "code": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2

    try:
        report = run_audit(frames, motion, {"seed": a.track_seed, "max_corners": a.max_corners})
        verified_source = (_verify_manifest(a.manifest, a.motion_npz, frames_meta, motion)
                           if a.manifest else None)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        print(json.dumps({"status": "error", "code": "mismatch", "message": str(exc)}), file=sys.stderr)
        return 2

    report["inputs"] = {"frames_sha256": MF.sha256_file(a.frames_npz), "motion_sha256": MF.sha256_file(a.motion_npz),
                        "motion_schema": meta.get("schema")}
    if verified_source:
        report["inputs"]["verified_frame_window"] = verified_source
        report["inputs"]["manifest_sha256"] = MF.sha256_file(a.manifest)
    report["status"] = "ok"
    MF.write_manifest(a.out, report)
    print(json.dumps({"status": "ok", "out": str(Path(a.out)), "agreement": report["agreement"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
