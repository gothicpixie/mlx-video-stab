"""Build frame-aligned raw/stabilized comparison clips from a successful benchmark run.

This local review aid verifies the run's source and encoded timelines, pairs frames by their
normalized presentation times, and writes provenance beside the MP4s. It encodes two blinded A/B
clips from the same paired frames in one pass: the original has raw on the left (``A``) and
stabilized on the right (``B``), and the position-reversed clip swaps the panels while keeping the
fixed A-left/B-right labels. One ``review.json`` records both pairs with hidden ``blind_mapping``
values that map ``A``/``B`` back to raw/stabilized. It makes no network call; the report can be
submitted separately with :mod:`bench.gemini_blind_review`.

An **opt-in candidate mode** (``--candidate-video``) instead compares the manifest's existing
stabilized ``outputs.video`` (the baseline) directly with one explicit postprocess candidate. It
decodes only those two already-encoded output streams, pairs them by verified frame index, and
labels the hidden panels generically as ``baseline``/``candidate``. Because both inputs are the same
stabilized frames, this isolates the postprocess effect from the stabilization itself; the default
raw-vs-stabilized workflow is unchanged.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from mlxstab import media

from . import manifest as MF
from . import gemini_blind_review as Gemini
from .multiscale_real_review import resize_bilinear


SCHEMA = "mlxstab.real_pair_review/1"
DEFAULT_PAIR_NAME = "raw_vs_stabilized.mp4"
DEFAULT_CANDIDATE_NAME = "baseline_vs_candidate.mp4"
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
}


class ReviewBuildError(ValueError):
    """Input evidence cannot support an exact raw/output pairing."""


def _rate(value, where):
    try:
        result = Fraction(str(value)).limit_denominator(1_000_000)
    except (ValueError, ZeroDivisionError, TypeError):
        raise ReviewBuildError(f"{where} must be a positive finite frame rate") from None
    if result <= 0:
        raise ReviewBuildError(f"{where} must be a positive finite frame rate")
    return result


def _close(a, b, *, tol=1e-5):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _validate_cfr(pts, count, fps, where):
    if not isinstance(pts, dict) or pts.get("count") != count or pts.get("n_pts") != count:
        raise ReviewBuildError(f"{where} PTS count does not match encoded frame count")
    values = pts.get("pts_s")
    if not isinstance(values, list) or len(values) != count or not pts.get("monotonic"):
        raise ReviewBuildError(f"{where} PTS are missing or non-monotonic")
    for i, value in enumerate(values):
        if not _close(value, i / float(fps)):
            raise ReviewBuildError(f"{where} is not zero-based constant-frame-rate at frame {i}")


def source_indices(source_fps, output_fps, output_count, source_count):
    """Map output frame times to nearest unique source frame times using exact rational arithmetic.

    Source and encoded clips both start at their own normalized time zero. Half-frame ties are
    refused because no unique nearest raw frame exists. Downsampling may skip source frames, but
    every output frame maps to a distinct, in-range decoded source frame.
    """
    si, of = _rate(source_fps, "source fps"), _rate(output_fps, "output fps")
    indices = []
    for i in range(int(output_count)):
        numerator = i * si.numerator * of.denominator
        denominator = si.denominator * of.numerator
        q, r = divmod(numerator, denominator)
        if 2 * r == denominator:
            raise ReviewBuildError(f"output frame {i} falls exactly between two source frames")
        index = q + (2 * r > denominator)
        if index < 0 or index >= int(source_count):
            raise ReviewBuildError(f"output frame {i} maps outside decoded source frames")
        if indices and index <= indices[-1]:
            raise ReviewBuildError("output cadence does not map to unique increasing source frames")
        indices.append(index)
    return indices


def _label(panel, word):
    panel = np.array(panel, dtype=np.uint8, copy=True)
    scale, pad, gap = 3, 5, 2
    glyphs = [FONT[c] for c in word]
    box_w = len(glyphs) * 5 * scale + (len(glyphs) - 1) * gap * scale + 2 * pad
    box_h = 7 * scale + 2 * pad
    if panel.shape[0] < box_h or panel.shape[1] < box_w:
        raise ReviewBuildError("panel is too small for the comparison label")
    panel[:box_h, :box_w] = 0
    x = pad
    for glyph in glyphs:
        for y, row in enumerate(glyph):
            for col, bit in enumerate(row):
                if bit == "1":
                    panel[pad + y * scale:pad + (y + 1) * scale,
                          x + col * scale:x + (col + 1) * scale] = 255
        x += (5 + gap) * scale
    return panel


def _compose(raw, stabilized, width, height):
    raw_panel = resize_bilinear(raw, width, height)
    stab_panel = resize_bilinear(stabilized, width, height)
    raw_panel = _label(raw_panel, "A")
    stab_panel = _label(stab_panel, "B")
    return np.concatenate([raw_panel, stab_panel], axis=1)


def _reversed_video_path(output_path):
    """Sibling path for the position-reversed clip: ``<stem>_posrev<.mp4>``."""
    return output_path.with_name(f"{output_path.stem}_posrev{output_path.suffix}")


def _default_output_path(manifest_path, candidate_video):
    """Default original-clip path for the selected mode (candidate mode uses a distinct name)."""
    name = DEFAULT_CANDIDATE_NAME if candidate_video is not None else DEFAULT_PAIR_NAME
    return Path(manifest_path).resolve().parent / name


def _validate_encoded_output(path, run_output, count, out_fps, where):
    """Independently verify one already-encoded output against the manifest's decoded metadata.

    Checks the probe's exact dimensions and frame rate plus the zero-origin constant-frame-rate PTS and
    frame count. The same helper is used for the baseline and candidate so neither is trusted from the
    other's decode.
    """
    info = media.probe(path)
    if (info["width"] != run_output.get("width") or info["height"] != run_output.get("height")
            or not _close(info["fps"], float(out_fps))):
        raise ReviewBuildError(f"{where} video probe disagrees with manifest decoded metadata")
    _validate_cfr(media.frame_pts(path), count, out_fps, where)
    return info


def _build_candidate(manifest_path, manifest, candidate_video, *, out_video, out_report, out_reversed,
                     panel_width, panel_height):
    """Compare the manifest's stabilized baseline output with one explicit postprocess candidate.

    This opt-in mode never re-decodes the raw source. It validates baseline and candidate independently
    against the manifest's exact output dimensions/frame rate/frame count/zero-origin CFR PTS, decodes only
    the two already-encoded output streams and pairs them by verified frame index (rejecting early EOF,
    extra frames and shape mismatch). The hidden A/B panels are labelled generically ``baseline`` and
    ``candidate``; the fixed review brief is untouched and no network call is made.
    """
    source = manifest.get("source") or {}
    outputs = manifest.get("outputs") or {}
    window = source.get("window") or {}
    source_path = Path(source.get("path") or "").expanduser()
    if not source_path.is_file() or source.get("path_type") != "file":
        raise ReviewBuildError("manifest must reference an existing local source file")
    source_sha = source.get("sha256")
    if not source_sha or MF.sha256_file(source_path) != source_sha:
        raise ReviewBuildError("source SHA-256 does not match benchmark manifest")
    run_output = outputs.get("decoded") or {}
    count = outputs.get("encoded_frames")
    if not isinstance(count, int) or count < 1 or run_output.get("frames") != count:
        raise ReviewBuildError("manifest decoded/encoded frame counts are missing or inconsistent")
    if (not _close(window.get("start_s"), window.get("start_s"))
            or not _close(window.get("duration_s"), window.get("duration_s"))
            or float(window["start_s"]) < 0 or float(window["duration_s"]) <= 0):
        raise ReviewBuildError("manifest source window is invalid")
    out_fps = _rate(run_output.get("fps"), "manifest output fps")
    cadence = outputs.get("cadence") or {}
    if not _close(cadence.get("measured_fps"), float(out_fps)) or cadence.get("measured_frames") != count:
        raise ReviewBuildError("manifest measured cadence disagrees with decoded output")
    _validate_cfr(outputs.get("encoded_pts"), count, out_fps, "manifest output")
    if outputs.get("timeline_alignment", {}).get("aligned") is not True:
        raise ReviewBuildError("manifest output timeline alignment failed")
    if panel_width < 32 or panel_height < 32 or panel_width * 2 * panel_height > 4_194_304:
        raise ReviewBuildError("panel size must be at least 32x32 and total at most 4,194,304 pixels")

    baseline_path = Path(outputs.get("video") or "").expanduser()
    candidate_path = Path(candidate_video).expanduser()
    if not baseline_path.is_file():
        raise ReviewBuildError("manifest output video is missing")
    if not candidate_path.is_file():
        raise ReviewBuildError("candidate video is missing")
    source_path = source_path.resolve()
    baseline_path, candidate_path = baseline_path.resolve(), candidate_path.resolve()
    if baseline_path == source_path:
        raise ReviewBuildError("manifest output video collides with the raw source")
    if candidate_path == baseline_path:
        raise ReviewBuildError("candidate video must differ from the manifest baseline output")
    if candidate_path in (source_path, manifest_path):
        raise ReviewBuildError("candidate video collides with a manifest/source input")

    output_path = Path(out_video) if out_video else _default_output_path(manifest_path, candidate_video)
    reversed_path = Path(out_reversed) if out_reversed else _reversed_video_path(output_path)
    report_path = Path(out_report) if out_report else output_path.with_suffix(".json")
    output_path, reversed_path, report_path = (
        output_path.resolve(), reversed_path.resolve(), report_path.resolve())
    if (output_path.suffix.lower() != ".mp4" or reversed_path.suffix.lower() != ".mp4"
            or report_path.suffix.lower() != ".json"):
        raise ReviewBuildError("review videos must use .mp4 and review manifest must use .json")
    protected = (manifest_path, source_path, baseline_path, candidate_path)
    if (len({output_path, reversed_path, report_path}) != 3
            or output_path in protected or reversed_path in protected or report_path in protected):
        raise ReviewBuildError("output paths collide with an input or each other")
    for path in (output_path, reversed_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    if not (output_path.parent == reversed_path.parent == report_path.parent):
        raise ReviewBuildError(
            "review MP4s and review.json must be in the same directory for Gemini clip resolution")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_info = _validate_encoded_output(baseline_path, run_output, count, out_fps, "baseline")
    candidate_info = _validate_encoded_output(candidate_path, run_output, count, out_fps, "candidate")
    baseline_sha = MF.sha256_file(baseline_path)
    candidate_sha = MF.sha256_file(candidate_path)
    baseline_frames, baseline_decode_fps, baseline_size = media.decode(baseline_path, baseline_info)
    candidate_frames, candidate_decode_fps, candidate_size = media.decode(candidate_path, candidate_info)
    if not _close(baseline_decode_fps, float(out_fps)) or not _close(candidate_decode_fps, float(out_fps)):
        raise ReviewBuildError("decoded output frame rates disagree with manifest")
    if (baseline_size != (run_output.get("width"), run_output.get("height"))
            or candidate_size != (run_output.get("width"), run_output.get("height"))):
        raise ReviewBuildError("decoded output dimensions disagree with manifest")
    if baseline_size != candidate_size:
        raise ReviewBuildError("baseline and candidate dimensions differ")

    jobs = (
        ("original", output_path, False),
        ("position_reversed", reversed_path, True),
    )
    tmp_paths, encoders = {}, {}
    try:
        for role, path, _swap in jobs:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent)
            os.close(fd)
            Path(tmp_name).unlink()
            tmp_paths[role] = tmp_name
            encoders[role] = media.Encoder(tmp_name, panel_width * 2, panel_height, float(out_fps),
                                           crf=20, preset="slow")
        baseline_iter, candidate_iter = iter(baseline_frames), iter(candidate_frames)
        for i in range(count):
            baseline = next(baseline_iter, None)
            candidate = next(candidate_iter, None)
            if baseline is None:
                raise ReviewBuildError("baseline decode ended before manifest frame count")
            if candidate is None:
                raise ReviewBuildError("candidate decode ended before manifest frame count")
            if baseline.shape[:2] != (baseline_size[1], baseline_size[0]):
                raise ReviewBuildError("decoded baseline frame dimensions changed midstream")
            if candidate.shape[:2] != (candidate_size[1], candidate_size[0]):
                raise ReviewBuildError("decoded candidate frame dimensions changed midstream")
            if baseline.shape != candidate.shape:
                raise ReviewBuildError(f"baseline and candidate frame shapes differ at frame {i}")
            for role, _path, swap in jobs:
                left, right = (candidate, baseline) if swap else (baseline, candidate)
                encoders[role].write(_compose(left, right, panel_width, panel_height))
        # Drain both iterators so extras are rejected rather than silently truncated.
        baseline_extra = sum(1 for _ in baseline_iter)
        candidate_extra = sum(1 for _ in candidate_iter)
        if baseline_extra:
            raise ReviewBuildError("baseline decoded frame count exceeds manifest")
        if candidate_extra:
            raise ReviewBuildError("candidate decoded frame count exceeds manifest")
        for encoder in encoders.values():
            encoder.close()
        encoders = {}
        encoded_pts = {}
        for role, _path, _swap in jobs:
            encoded_pts[role] = media.frame_pts(tmp_paths[role])
            _validate_cfr(encoded_pts[role], count, out_fps, f"generated {role} review")
            size = Path(tmp_paths[role]).stat().st_size
            if size > Gemini.MAX_INLINE_BYTES:
                raise ReviewBuildError(
                    f"generated {role} clip is {size} bytes, over Gemini's "
                    f"{Gemini.MAX_INLINE_BYTES}-byte inline limit; "
                    "retry with smaller --panel-width/--panel-height"
                )
        os.replace(tmp_paths["original"], output_path)
        os.replace(tmp_paths["position_reversed"], reversed_path)
        tmp_paths = {}
        run_id = str(manifest.get("run_id") or manifest_path.parent.name)
        pairing_rule = "verified same already-encoded output frame index; early EOF/extra frames rejected"
        video_labels = {"left": "A", "right": "B"}
        # The group embeds both input hashes so two different candidate/baseline pairs for the same source
        # window stay distinct, while the original and position-reversed roles still share one group.
        group_id = (f"{run_id}|{source_sha}|{float(window['start_s'])}|{float(window['duration_s'])}|"
                    f"{baseline_sha}|{candidate_sha}")
        report = {
            "schema": Gemini.INPUT_SCHEMA,
            "ok": True,
            "output": {"dir": output_path.parent.name,
                       "clips": [output_path.name, reversed_path.name]},
            "pairs": [
                {"pair_index": 0, "clip": output_path.name,
                 "blind_mapping": {"left_panel": "baseline", "right_panel": "candidate",
                                   "video_labels": dict(video_labels)},
                 "pairing": {"rule": pairing_rule},
                 "position_reversal": {
                     "schema": Gemini.POSITION_REVERSAL_SCHEMA,
                     "group": group_id, "role": "original"}},
                {"pair_index": 1, "clip": reversed_path.name,
                 "blind_mapping": {"left_panel": "candidate", "right_panel": "baseline",
                                   "video_labels": dict(video_labels)},
                 "pairing": {"rule": pairing_rule},
                 "position_reversal": {
                     "schema": Gemini.POSITION_REVERSAL_SCHEMA,
                     "group": group_id, "role": "position_reversed"}},
            ],
            "position_reversal": {
                "group": group_id,
                "original_clip": output_path.name,
                "reversed_clip": reversed_path.name,
                "panel_swap": (
                    "both clips are encoded from the same paired baseline/candidate already-encoded output "
                    "frames in one pass; the position-reversed clip swaps the two panels while the fixed "
                    "A-left/B-right labels and the A/B review brief are unchanged"
                ),
                "mapping_reversal": (
                    "the original clip maps left/A to baseline and right/B to candidate; the "
                    "position-reversed clip maps left/A to candidate and right/B to baseline"
                ),
                "no_network": True,
            },
            "provenance": {"run": run_id, "run_id": run_id, "mode": "candidate",
                           "case": (manifest.get("case") or {}).get("name"),
                           "source_sha256": source_sha,
                           "baseline_sha256": baseline_sha, "candidate_sha256": candidate_sha,
                           "source_window": {"start_s": float(window["start_s"]),
                                             "duration_s": float(window["duration_s"])}},
            "evidence": {
                "schema": SCHEMA,
                "mode": "candidate",
                "source_manifest": {"path": str(manifest_path), "sha256": MF.sha256_file(manifest_path)},
                "source": {"path": str(source_path), "sha256": source_sha,
                           "window": {"start_s": float(window["start_s"]),
                                      "duration_s": float(window["duration_s"])}},
                "baseline": {"path": str(baseline_path), "sha256": baseline_sha,
                             "fps": [out_fps.numerator, out_fps.denominator], "frames": count},
                "candidate": {"path": str(candidate_path), "sha256": candidate_sha,
                              "fps": [out_fps.numerator, out_fps.denominator], "frames": count},
                "cadence": {"fps": [out_fps.numerator, out_fps.denominator], "frames": count,
                            "measured_fps": cadence.get("measured_fps"),
                            "measured_frames": cadence.get("measured_frames")},
                "mapping": {"rule": "verified same already-encoded output frame index; early EOF, extra "
                                     "frames and shape mismatch rejected",
                            "frame_indices": list(range(count)), "unique": True},
                "panels": {"left": "A", "right": "B", "width": panel_width, "height": panel_height,
                           "original": {"clip": output_path.name, "left_panel": "baseline",
                                        "right_panel": "candidate"},
                           "position_reversed": {"clip": reversed_path.name, "left_panel": "candidate",
                                                 "right_panel": "baseline"}},
                "outputs": [
                    {"clip": output_path.name, "role": "original", "path": str(output_path),
                     "sha256": MF.sha256_file(output_path),
                     "fps": [out_fps.numerator, out_fps.denominator], "frames": count,
                     "width": panel_width * 2, "height": panel_height, "pts": encoded_pts["original"]},
                    {"clip": reversed_path.name, "role": "position_reversed", "path": str(reversed_path),
                     "sha256": MF.sha256_file(reversed_path),
                     "fps": [out_fps.numerator, out_fps.denominator], "frames": count,
                     "width": panel_width * 2, "height": panel_height,
                     "pts": encoded_pts["position_reversed"]},
                ],
            },
        }
        MF.write_manifest(report_path, report)
        return report
    except BaseException:
        for encoder in encoders.values():
            try:
                encoder.close()
            except Exception:
                pass
        for tmp_name in tmp_paths.values():
            Path(tmp_name).unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        reversed_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise


def build(manifest_path, *, out_video=None, out_report=None, out_reversed=None,
          panel_width=640, panel_height=360, candidate_video=None):
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewBuildError(f"cannot read benchmark manifest: {exc}") from exc
    if manifest.get("status") != "success":
        raise ReviewBuildError("benchmark manifest status must be success")
    if candidate_video is not None:
        return _build_candidate(manifest_path, manifest, candidate_video, out_video=out_video,
                                out_report=out_report, out_reversed=out_reversed,
                                panel_width=panel_width, panel_height=panel_height)
    source = manifest.get("source") or {}
    outputs = manifest.get("outputs") or {}
    window = source.get("window") or {}
    source_path = Path(source.get("path") or "").expanduser()
    if not source_path.is_file() or source.get("path_type") != "file":
        raise ReviewBuildError("manifest must reference an existing local source file")
    source_sha = source.get("sha256")
    if not source_sha or MF.sha256_file(source_path) != source_sha:
        raise ReviewBuildError("source SHA-256 does not match benchmark manifest")
    run_source = source.get("decoded") or {}
    run_output = outputs.get("decoded") or {}
    count = outputs.get("encoded_frames")
    if (not isinstance(count, int) or count < 1 or run_output.get("frames") != count
            or run_source.get("frames") is None):
        raise ReviewBuildError("manifest decoded/encoded frame counts are missing or inconsistent")
    if (not _close(window.get("start_s"), window.get("start_s"))
            or not _close(window.get("duration_s"), window.get("duration_s"))
            or float(window["start_s"]) < 0 or float(window["duration_s"]) <= 0):
        raise ReviewBuildError("manifest source window is invalid")
    out_fps = _rate(run_output.get("fps"), "manifest output fps")
    source_fps = _rate(run_source.get("fps"), "manifest source processing fps")
    cadence = outputs.get("cadence") or {}
    if not _close(cadence.get("measured_fps"), float(out_fps)) or cadence.get("measured_frames") != count:
        raise ReviewBuildError("manifest measured cadence disagrees with decoded output")
    encoded_count = outputs.get("encoded_frames")
    _validate_cfr(outputs.get("encoded_pts"), encoded_count, out_fps, "manifest output")
    if outputs.get("timeline_alignment", {}).get("aligned") is not True:
        raise ReviewBuildError("manifest output timeline alignment failed")
    indices = source_indices(source_fps, out_fps, count, int(run_source["frames"]))
    if panel_width < 32 or panel_height < 32 or panel_width * 2 * panel_height > 4_194_304:
        raise ReviewBuildError("panel size must be at least 32x32 and total at most 4,194,304 pixels")

    output_path = Path(out_video) if out_video else manifest_path.parent / "raw_vs_stabilized.mp4"
    reversed_path = Path(out_reversed) if out_reversed else _reversed_video_path(output_path)
    report_path = Path(out_report) if out_report else output_path.with_suffix(".json")
    output_path, reversed_path, report_path = (
        output_path.resolve(), reversed_path.resolve(), report_path.resolve())
    if (output_path.suffix.lower() != ".mp4" or reversed_path.suffix.lower() != ".mp4"
            or report_path.suffix.lower() != ".json"):
        raise ReviewBuildError("review videos must use .mp4 and review manifest must use .json")
    for path in (output_path, reversed_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    if (len({output_path, reversed_path, report_path}) != 3
            or output_path in (manifest_path, source_path) or reversed_path in (manifest_path, source_path)
            or report_path in (manifest_path, source_path)):
        raise ReviewBuildError("output paths collide with an input or each other")
    if not (output_path.parent == reversed_path.parent == report_path.parent):
        raise ReviewBuildError(
            "review MP4s and review.json must be in the same directory for Gemini clip resolution")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = media.probe(source_path)
    preprocessing = source.get("preprocessing") or {}
    deinterlace = bool(preprocessing.get("interlaced"))
    if deinterlace and preprocessing.get("deinterlace") != "bwdif=mode=send_field:parity=auto:deint=all":
        raise ReviewBuildError("unsupported or missing manifest deinterlace filter")
    raw_frames, raw_fps, raw_size = media.decode(source_path, info, window["start_s"], window["duration_s"],
                                                 deinterlace=deinterlace, vf=preprocessing.get("decode_vf"))
    stabilized_path = Path(outputs.get("video") or "")
    if not stabilized_path.is_file():
        raise ReviewBuildError("manifest output video is missing")
    if stabilized_path.resolve() == source_path.resolve():
        raise ReviewBuildError("manifest output video collides with the raw source")
    output_sha = MF.sha256_file(stabilized_path)
    stab_info = media.probe(stabilized_path)
    if (stab_info["width"] != run_output.get("width") or stab_info["height"] != run_output.get("height")
            or not _close(stab_info["fps"], float(out_fps))):
        raise ReviewBuildError("output video probe disagrees with manifest decoded metadata")
    _validate_cfr(media.frame_pts(stabilized_path), count, out_fps, "stabilized video")
    stab_frames, stab_fps, stab_size = media.decode(stabilized_path, stab_info)
    if not _close(raw_fps, float(source_fps)) or not _close(stab_fps, float(out_fps)):
        raise ReviewBuildError("decoded source/output frame rates disagree with manifest")
    if raw_size != (run_source.get("width"), run_source.get("height")):
        raise ReviewBuildError("decoded source dimensions disagree with manifest")
    if stab_size != (run_output.get("width"), run_output.get("height")):
        raise ReviewBuildError("decoded output dimensions disagree with manifest")

    # Both clips are encoded from the same paired frames in one pass: the original has raw on the left
    # (A) and stabilized on the right (B); the position-reversed clip swaps the panels but keeps the fixed
    # A-left/B-right labels, so only the hidden A/B mapping distinguishes the two clips.
    jobs = (
        ("original", output_path, False),
        ("position_reversed", reversed_path, True),
    )
    tmp_paths, encoders = {}, {}
    try:
        for role, path, _swap in jobs:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent)
            os.close(fd)
            Path(tmp_name).unlink()
            tmp_paths[role] = tmp_name
            encoders[role] = media.Encoder(tmp_name, panel_width * 2, panel_height, float(out_fps),
                                           crf=20, preset="slow")
        raw_iter, stab_iter = iter(raw_frames), iter(stab_frames)
        raw_index, raw_count = -1, 0
        current_raw = None
        for i, wanted in enumerate(indices):
            while raw_index < wanted:
                current_raw = next(raw_iter, None)
                raw_index += 1
                raw_count += current_raw is not None
                if current_raw is None:
                    raise ReviewBuildError("source decode ended before mapped frame")
            raw = current_raw
            stabilized = next(stab_iter, None)
            if stabilized is None:
                raise ReviewBuildError("stabilized decode ended before manifest frame count")
            if raw.shape[:2] != (raw_size[1], raw_size[0]):
                raise ReviewBuildError("decoded source frame dimensions changed midstream")
            if stabilized.shape[:2] != (stab_size[1], stab_size[0]):
                raise ReviewBuildError("decoded output frame dimensions changed midstream")
            for role, _path, swap in jobs:
                left, right = (stabilized, raw) if swap else (raw, stabilized)
                encoders[role].write(_compose(left, right, panel_width, panel_height))
        # Drain both iterators so the count check covers extra frames as well as early EOF.
        raw_count += sum(1 for _ in raw_iter)
        extra_output = sum(1 for _ in stab_iter)
        if raw_count != run_source["frames"]:
            raise ReviewBuildError("source decoded frame count disagrees with manifest")
        if count + extra_output != run_output["frames"]:
            raise ReviewBuildError("stabilized decoded frame count disagrees with manifest")
        for encoder in encoders.values():
            encoder.close()
        encoders = {}
        encoded_pts = {}
        for role, _path, _swap in jobs:
            encoded_pts[role] = media.frame_pts(tmp_paths[role])
            _validate_cfr(encoded_pts[role], count, out_fps, f"generated {role} review")
            size = Path(tmp_paths[role]).stat().st_size
            if size > Gemini.MAX_INLINE_BYTES:
                raise ReviewBuildError(
                    f"generated {role} clip is {size} bytes, over Gemini's "
                    f"{Gemini.MAX_INLINE_BYTES}-byte inline limit; "
                    "retry with smaller --panel-width/--panel-height"
                )
        os.replace(tmp_paths["original"], output_path)
        os.replace(tmp_paths["position_reversed"], reversed_path)
        tmp_paths = {}
        run_id = str(manifest.get("run_id") or manifest_path.parent.name)
        pairing_rule = "same source window; nearest normalized presentation timestamp"
        video_labels = {"left": "A", "right": "B"}
        # Shared stable identifier for the paired original/position-reversed items: both roles carry the same
        # group so the offline position-bias audit can pair them without relying on the clip name or index.
        group_id = (f"{run_id}|{source_sha}|{float(window['start_s'])}|{float(window['duration_s'])}")
        report = {
            "schema": Gemini.INPUT_SCHEMA,
            "ok": True,
            "output": {"dir": output_path.parent.name,
                       "clips": [output_path.name, reversed_path.name]},
            "pairs": [
                {"pair_index": 0, "clip": output_path.name,
                 "blind_mapping": {"left_panel": "raw", "right_panel": "stabilized",
                                   "video_labels": dict(video_labels)},
                 "pairing": {"rule": pairing_rule},
                 "position_reversal": {
                     "schema": Gemini.POSITION_REVERSAL_SCHEMA,
                     "group": group_id, "role": "original"}},
                {"pair_index": 1, "clip": reversed_path.name,
                 "blind_mapping": {"left_panel": "stabilized", "right_panel": "raw",
                                   "video_labels": dict(video_labels)},
                 "pairing": {"rule": pairing_rule},
                 "position_reversal": {
                     "schema": Gemini.POSITION_REVERSAL_SCHEMA,
                     "group": group_id, "role": "position_reversed"}},
            ],
            "position_reversal": {
                "group": group_id,
                "original_clip": output_path.name,
                "reversed_clip": reversed_path.name,
                "panel_swap": (
                    "both clips are encoded from the same paired raw/stabilized frames in one pass; the "
                    "position-reversed clip swaps the two panels while the fixed A-left/B-right labels and "
                    "the A/B review brief are unchanged"
                ),
                "mapping_reversal": (
                    "the original clip maps left/A to raw and right/B to stabilized; the position-reversed "
                    "clip maps left/A to stabilized and right/B to raw"
                ),
                "no_network": True,
            },
            "provenance": {"run": run_id, "run_id": run_id, "case": (manifest.get("case") or {}).get("name"),
                           "source_sha256": source_sha,
                           "source_window": {"start_s": float(window["start_s"]),
                                             "duration_s": float(window["duration_s"])}},
            "evidence": {
                "schema": SCHEMA,
                "source_manifest": {"path": str(manifest_path), "sha256": MF.sha256_file(manifest_path)},
                "source": {"path": str(source_path), "sha256": source_sha,
                           "window": {"start_s": float(window["start_s"]),
                                      "duration_s": float(window["duration_s"])},
                           "fps": [source_fps.numerator, source_fps.denominator], "frames": raw_count},
                "stabilized": {"path": str(stabilized_path), "sha256": output_sha,
                               "fps": [out_fps.numerator, out_fps.denominator], "frames": count},
                "mapping": {"rule": "nearest normalized presentation timestamp; exact half-frame ties rejected",
                            "source_frame_indices": indices, "unique": len(set(indices)) == count},
                "panels": {"left": "A", "right": "B", "width": panel_width, "height": panel_height,
                           "original": {"clip": output_path.name, "left_panel": "raw",
                                        "right_panel": "stabilized"},
                           "position_reversed": {"clip": reversed_path.name, "left_panel": "stabilized",
                                                 "right_panel": "raw"}},
                "outputs": [
                    {"clip": output_path.name, "role": "original", "path": str(output_path),
                     "sha256": MF.sha256_file(output_path),
                     "fps": [out_fps.numerator, out_fps.denominator], "frames": count,
                     "width": panel_width * 2, "height": panel_height, "pts": encoded_pts["original"]},
                    {"clip": reversed_path.name, "role": "position_reversed", "path": str(reversed_path),
                     "sha256": MF.sha256_file(reversed_path),
                     "fps": [out_fps.numerator, out_fps.denominator], "frames": count,
                     "width": panel_width * 2, "height": panel_height,
                     "pts": encoded_pts["position_reversed"]},
                ],
            },
        }
        MF.write_manifest(report_path, report)
        return report
    except BaseException:
        for encoder in encoders.values():
            try:
                encoder.close()
            except Exception:
                pass
        for tmp_name in tmp_paths.values():
            Path(tmp_name).unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        reversed_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="successful bench/run_case manifest.json")
    parser.add_argument("--candidate-video",
                        help="opt-in postprocess candidate MP4 to compare against the manifest's existing "
                             "stabilized outputs.video (baseline); isolates the postprocess effect against "
                             "the same stabilized frames. Default raw-vs-stabilized mode is unchanged")
    parser.add_argument("--out-video", help="original A-left/B-right MP4")
    parser.add_argument("--out-reversed", help="position-reversed MP4 (default: <out-video>_posrev.mp4)")
    parser.add_argument("--out-report")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    args = parser.parse_args(argv)
    try:
        result = build(args.manifest, out_video=args.out_video, out_reversed=args.out_reversed,
                       out_report=args.out_report, panel_width=args.panel_width,
                       panel_height=args.panel_height, candidate_video=args.candidate_video)
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    output_path = (Path(args.out_video) if args.out_video
                   else _default_output_path(args.manifest, args.candidate_video))
    reversed_path = Path(args.out_reversed) if args.out_reversed else _reversed_video_path(output_path)
    report_path = Path(args.out_report) if args.out_report else output_path.with_suffix(".json")
    print(json.dumps({"status": "success", "output": str(output_path), "reversed": str(reversed_path),
                      "report": str(report_path),
                      "frames": result["evidence"]["outputs"][0]["frames"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
