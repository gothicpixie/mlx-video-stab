"""Atomic, JSON-safe manifests for the Stage 0 benchmark runner.

Everything here is side-effect-light and unit-testable with cheap fakes: hashing, environment/hardware
collection, JSON normalisation, atomic writes and the cadence/reliability guards. Private paths belong in
the ignored ``out/`` reports this module writes, never in public tracked docs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MANIFEST_SCHEMA = "mlxstab.bench.manifest/1"
METRICS_SCHEMA = "mlxstab.metrics/1"


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def relevant_code_files(root) -> list[Path]:
    """Working-tree source that affects a run: package, benchmark scripts and packaging metadata."""
    root = Path(root)
    out = []
    for sub in ("mlxstab", "bench"):
        d = root / sub
        if d.is_dir():
            out.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    pj = root / "pyproject.toml"
    if pj.exists():
        out.append(pj)
    return sorted(out)


def _git(args, root):
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True)


def code_fingerprint(root) -> dict:
    """HEAD revision plus sha256 of working-tree files, and a *relevant* dirty flag.

    The overall repo may be dirty because of unrelated docs (README, marketing). ``dirty_relevant`` only
    looks at the files whose hashes are recorded here, which is what invalidates a run.
    """
    root = Path(root)
    rev = _git(["rev-parse", "HEAD"], root)
    revision = rev.stdout.strip() if rev.returncode == 0 else None
    files = {}
    for p in relevant_code_files(root):
        files[str(p.relative_to(root))] = sha256_file(p)
    st = _git(["status", "--porcelain", "--", "mlxstab", "bench", "pyproject.toml"], root)
    status_lines = [l for l in st.stdout.splitlines() if l.strip()] if st.returncode == 0 else []
    return {
        "revision": revision,
        "dirty_relevant": bool(status_lines),
        "relevant_status": status_lines[:50],
        "files": files,
    }


def metrics_digest(root) -> dict:
    """Version string bound to the metric/warp-diagnostic source that produced the numbers."""
    root = Path(root)
    files = {}
    for rel in ("mlxstab/metrics.py", "mlxstab/warpmetrics.py"):
        p = root / rel
        files[rel] = sha256_file(p) if p.exists() else None
    h = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    return {"version": f"{METRICS_SCHEMA}+{h}", "files": files}


# --------------------------------------------------------------------------------------
# environment and hardware
# --------------------------------------------------------------------------------------

def env_versions() -> dict:
    out = {"python": platform.python_version()}
    for name, mod in (("numpy", "numpy"), ("scipy", "scipy"), ("mlx", "mlx.core"), ("torch", "torch")):
        try:
            m = __import__(mod, fromlist=["__version__"])
            out[name] = getattr(m, "__version__", None) or getattr(getattr(m, "__version__", None), "version", None)
        except Exception as e:
            out[name] = None
            out[f"{name}_unavailable"] = f"{type(e).__name__}: {e}"
    for exe in ("ffmpeg", "ffprobe"):
        try:
            r = subprocess.run([exe, "-version"], capture_output=True, text=True)
            out[exe] = r.stdout.splitlines()[0].strip() if r.returncode == 0 and r.stdout else None
        except Exception as e:
            out[exe] = None
            out[f"{exe}_unavailable"] = f"{type(e).__name__}: {e}"
    return out


def _sysctl(name):
    try:
        r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def hardware_info() -> dict:
    mem = _sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "os_version": platform.mac_ver()[0] or platform.version(),
        "cpu_count": os.cpu_count(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "total_memory_bytes": int(mem) if mem and mem.isdigit() else None,
    }


def peak_memory() -> dict:
    """Process peak RSS and MLX peak allocation. Values are null with a reason when unavailable."""
    rec = {"process_peak_rss_bytes": None, "mlx_peak_active_bytes": None,
           "available": False, "unavailable_reason": None}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss:
            rec["process_peak_rss_bytes"] = int(rss) if sys.platform == "darwin" else int(rss) * 1024
    except Exception as e:
        rec["unavailable_reason"] = f"resource.getrusage unavailable: {type(e).__name__}: {e}"
    try:
        import mlx.core as mx
        mlx_peak = int(mx.get_peak_memory())
        if mlx_peak > 0:
            rec["mlx_peak_active_bytes"] = mlx_peak
    except Exception as e:
        rec["unavailable_reason"] = (rec["unavailable_reason"] or "") + f" mlx.get_peak_memory unavailable: {e}"
    rec["available"] = rec["process_peak_rss_bytes"] is not None or rec["mlx_peak_active_bytes"] is not None
    if not rec["available"] and not rec["unavailable_reason"]:
        rec["unavailable_reason"] = "no memory source returned a positive value"
    return rec


# --------------------------------------------------------------------------------------
# JSON safety and atomic write
# --------------------------------------------------------------------------------------

def json_safe(obj):
    """Recursively convert to JSON-native, finite values. Non-finite floats become ``None`` (never NaN)."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy and other scalars/arrays without importing numpy at module import time
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return json_safe(obj.tolist())
        if isinstance(obj, np.generic):
            return json_safe(obj.item())
    except Exception:
        pass
    return str(obj)


def dumps_json(payload) -> str:
    """Serialize strictly: no NaN/Infinity tokens, sorted keys, indented for review."""
    return json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False)


def write_manifest(path, manifest: dict) -> str:
    """Write JSON atomically: serialize first, then fsync a temp file and os.replace it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps_json(manifest)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(path)


# --------------------------------------------------------------------------------------
# identity, cadence and reliability guards
# --------------------------------------------------------------------------------------

def content_key(identity: dict) -> str:
    """Hash the full experiment identity (source + window, config, cadence, evaluation, code digests, ...)."""
    return hashlib.sha256(json.dumps(json_safe(identity), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _finite_number(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _timeline_reason(tag, t):
    if not isinstance(t, dict):
        return f"{tag} timeline missing"
    first, last = t.get("first_s"), t.get("last_s")
    if not (_finite_number(first) and _finite_number(last)):
        return f"{tag} timeline first/last missing or non-finite"
    if last < first:
        return f"{tag} timeline reversed (first {first}, last {last})"
    if t.get("frames") is not None and _finite_number(t.get("frames")) and int(t["frames"]) > 1 and last - first <= 0:
        return f"{tag} timeline has a non-positive span"
    return None


def check_encoded_cfr(video, mask, fps, mask_fps, requested_fps=None, tol_s=1e-5):
    """Validate the entire encoded video/mask clock before comparing normalized evaluation streams.

    The raw-frame encoder starts at zero. Every measured PTS must match i/fps, not merely the endpoints.
    The tolerance accommodates ffprobe's microsecond decimal rounding, not a shifted or missing frame.
    Output and evaluation rates can differ: this check only compares the encoded pair at its own rate.
    """
    reasons = []
    for tag, stream, rate in (("video", video, fps), ("mask", mask, mask_fps)):
        if not _finite_number(rate) or float(rate) <= 0:
            reasons.append(f"encoded {tag} fps missing or non-positive")
            continue
        count, pts = stream.get("count"), stream.get("pts_s", [])
        if not _finite_number(count) or int(count) < 1 or count != int(count):
            reasons.append(f"encoded {tag} frame count missing or invalid")
            continue
        if len(pts) != count or stream.get("n_pts") != count or not all(_finite_number(t) for t in pts):
            reasons.append(f"encoded {tag} has incomplete/non-finite PTS")
            continue
        if any(abs(float(t) - i / float(rate)) > tol_s for i, t in enumerate(pts)):
            reasons.append(f"encoded {tag} PTS do not follow its zero-origin CFR clock")
    if video.get("count") != mask.get("count"):
        reasons.append("encoded video/mask frame counts differ")
    if _finite_number(fps) and _finite_number(mask_fps) and abs(float(fps) - float(mask_fps)) > 1e-6:
        reasons.append("encoded video/mask frame rates differ")
    if requested_fps is not None and _finite_number(fps) and abs(float(fps) - float(requested_fps)) > 1e-6:
        reasons.append("encoded frame rate differs from the requested output rate")
    return {"aligned": not reasons, "reasons": reasons, "tolerance_s": tol_s,
            "video_frames": video.get("count"), "mask_frames": mask.get("count")}


def check_cadence(inp: dict, out: dict, mask: dict | None = None, tol_frames: int = 0,
                  tol_duration_s: float = 0.05, tol_time_s: float | None = None) -> dict:
    """Compare evaluation cadence of input and output clips before their jitter is compared.

    ``inp``/``out`` each supply ``frames``, ``fps``, ``duration_s`` and a ``timeline`` with
    ``{kind, frames, first_s, last_s}``; ``mask`` optionally supplies the same. These are the evaluated,
    normalized streams at their common evaluation rate. Encoded video/mask PTS are validated independently
    by :func:`check_encoded_cfr` at the output rate. Missing/non-finite/reversed timing metadata is never treated as aligned, and a
    mask stream that is longer than the output is detected even if the evaluation loop only consumed a
    prefix. Both evaluation paths pass through the same fps filter, so frame counts are expected to match
    exactly (``tol_frames=0``); a caller may justify a tolerance explicitly.
    """
    reasons = []
    fi, fo = inp.get("frames"), out.get("frames")
    di, do = inp.get("duration_s"), out.get("duration_s")
    frame_delta = (fo - fi) if (fi is not None and fo is not None) else None
    duration_delta = (do - di) if (di is not None and do is not None) else None
    if not (_finite_number(fi) and _finite_number(fo)):
        reasons.append("frame count missing or non-finite")
    if not (_finite_number(di) and _finite_number(do)):
        reasons.append("duration missing or non-finite")
    if not (_finite_number(inp.get("fps")) and _finite_number(out.get("fps"))):
        reasons.append("effective fps missing or non-finite")
    for tag, meta in (("input", inp), ("output", out), ("mask", mask)):
        if meta is None:
            continue
        for key in ("frames", "fps", "duration_s"):
            if not _finite_number(meta.get(key)) or float(meta[key]) <= 0:
                reasons.append(f"{tag} {key} must be finite and positive")
        if (meta.get("timeline") or {}).get("frames") != meta.get("frames"):
            reasons.append(f"{tag} timeline frame count differs from evaluated count")
    if frame_delta is not None and abs(frame_delta) > tol_frames:
        reasons.append(f"frame counts differ by {frame_delta} (tol {tol_frames})")
    if duration_delta is not None and abs(duration_delta) > max(tol_duration_s, 0.02 * max(abs(di), 1e-9)):
        reasons.append(f"durations differ by {duration_delta:.4f}s (tol {tol_duration_s:.3f}s)")
    if _finite_number(inp.get("fps")) and _finite_number(out.get("fps")) and abs(inp["fps"] - out["fps"]) > 1e-3:
        reasons.append(f"effective fps differ: input {inp['fps']:.6f}, output {out['fps']:.6f}")

    ti, to = inp.get("timeline"), out.get("timeline")
    tm = mask.get("timeline") if mask is not None else None
    for tag, t in (("input", ti), ("output", to), ("mask", tm)):
        if mask is None and tag == "mask":
            continue
        r = _timeline_reason(tag, t)
        if r:
            reasons.append(r)
    mask_delta = None
    if mask is not None:
        mf = mask.get("frames")
        if not _finite_number(mf):
            reasons.append("mask frame count missing or non-finite")
        else:
            mask_delta = (mf - fo) if _finite_number(fo) else None
            if mask_delta is not None and mask_delta != 0:
                reasons.append(f"mask frame count differs from output by {mask_delta}")
    # Cross-check the timelines only when all participants have finite metadata.
    if _timeline_reason("input", ti) is None and _timeline_reason("output", to) is None:
        tol = tol_time_s if tol_time_s is not None else 1e-5
        for key in ("first_s", "last_s"):
            if abs(float(ti[key]) - float(to[key])) > tol:
                reasons.append(f"input/output timeline {key} differ by {float(ti[key]) - float(to[key]):.4f}s (tol {tol:.4f}s)")
        if _timeline_reason("mask", tm) is None:
            for key in ("first_s", "last_s"):
                if abs(float(tm[key]) - float(to[key])) > tol:
                    reasons.append(f"output/mask timeline {key} differ by {float(tm[key]) - float(to[key]):.4f}s (tol {tol:.4f}s)")
    return {"aligned": not reasons, "frame_delta": frame_delta, "duration_delta_s": duration_delta,
            "mask_frame_delta": mask_delta, "tol_frames": tol_frames, "reasons": reasons,
            "timeline_kinds": {"input": (ti or {}).get("kind"), "output": (to or {}).get("kind"),
                               "mask": (tm or {}).get("kind")},
            "input": inp, "output": out, "mask": mask}


# Required finite metrics and explicit support counters for a comparison to mean anything.
REQUIRED_METRICS = ("stability", "jitter_px", "wobble_px")
SUPPORT_METRICS = ("support_frames", "inlier_frac_mean", "fallback_frames")


def reliability(inp_metrics: dict, out_metrics: dict, cadence: dict) -> dict:
    """Explicit support/failure flags. Never collapses to a single 'quality' score.

    A comparison is unreliable when cadence is not aligned, when required metrics are missing or non-finite,
    when explicit support counters are absent/empty, or when any motion fit fell back to identity.
    """
    reasons = list(cadence.get("reasons", []))
    if not cadence.get("aligned", False) and not reasons:
        reasons.append("cadence not aligned")
    support = {}
    for tag, m in (("input", inp_metrics), ("output", out_metrics)):
        if not isinstance(m, dict):
            reasons.append(f"{tag} metrics missing")
            support[tag] = None
            continue
        for k in REQUIRED_METRICS:
            v = m.get(k)
            if v is None:
                reasons.append(f"{tag} {k} missing")
            elif not _finite_number(v):
                reasons.append(f"{tag} {k} non-finite ({v!r})")
        sf = m.get("support_frames")
        if sf is None:
            reasons.append(f"{tag} support_frames missing")
        elif not _finite_number(sf) or int(sf) <= 0:
            reasons.append(f"{tag} has no fitted frame pairs")
        inf = m.get("inlier_frac_mean")
        if inf is None:
            reasons.append(f"{tag} inlier_frac_mean missing")
        elif not _finite_number(inf) or float(inf) <= 0.0:
            reasons.append(f"{tag} insufficient inlier support ({inf!r})")
        fb = m.get("fallback_frames")
        if fb is None:
            reasons.append(f"{tag} fallback_frames missing")
        elif not _finite_number(fb):
            reasons.append(f"{tag} fallback_frames non-finite")
        elif int(fb) > 0:
            reasons.append(f"{tag} motion fit fell back to identity on {int(fb)} frame(s); "
                           "failed fits can score as perfectly stable and are not quality evidence")
        support[tag] = {"support_frames": sf, "inlier_frac_mean": inf, "fallback_frames": fb}
    return {"reliable": not reasons, "reasons": reasons, "support": support}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
