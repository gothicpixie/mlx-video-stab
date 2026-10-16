"""Research-only local position-reversal of blinded A/B review clips.

What this does
--------------
``bench.multiscale_real_review`` writes blinded two-panel A/B MP4s whose hidden ``blind_mapping`` records
which side (left/A vs right/B) is the multiscale-retention candidate and which is its nearest measured-crop
L1 counterpart. This module builds a **position-reversed local clone** of a caller-selected subset of those
pairs:

1. Load and validate an existing ``review.json`` with :func:`bench.gemini_blind_review.load_review` (plain
   ``.mp4`` clip basenames, a plain ``output.dir`` and validated per-pair ``blind_mapping``) and resolve each
   selected source clip inside the review directory with
   :func:`bench.gemini_blind_review.resolve_clip_path`.
2. Decode the already-rendered progressive A/B clip with :mod:`mlxstab.media`, exchange its two equal-width
   panels, and repaint the fixed A-left / B-right bitmap labels with
   :func:`bench.multiscale_real_review.draw_bitmap_label`, so a reviewer still sees ``A`` on the left and
   ``B`` on the right but the pixels behind the labels are swapped.
3. Encode a new local MP4 at the original fps and dimensions, then write a copied ``review.json`` that keeps
   only the selected pairs, renames their clips, flips ``blind_mapping.left_panel``/``right_panel`` for every
   swapped pair and records explicit ``position_reversal`` provenance.

The new report keeps the ``mlxstab.multiscale_real_review/1`` schema, so it remains a valid input to
:mod:`bench.gemini_blind_review`; the source clips and report are only read, never modified. All artifacts
stay under the caller-selected ``--out-dir`` (default ignored
``out/bench/gemini_review/position_reversal/<review-dir-name>``). Nothing here touches the network.

What this is not
----------------
It decodes and re-encodes local review clips and rewrites the hidden A/B mapping. It computes no
image-quality metric, no ranking, no winner and no aggregate score, and it changes no production default.

Usage::

    python -m bench.multiscale_position_reversal out/bench/gemini_review/RUN/review.json \\
        --out-dir out/bench/gemini_review/RUN/position_reversed [--pair-index 0 --pair-index 2]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from mlxstab import media

from . import gemini_blind_review as G
from . import multiscale_real_review as R
from .manifest import sha256_file, write_manifest

SCHEMA = R.SCHEMA
METADATA_SCHEMA = "mlxstab.position_reversal/1"

DEFAULT_OUT_ROOT = Path("out/bench/gemini_review/position_reversal")

#: Local source clips are only read and re-encoded; this generous bound still refuses absurd/empty inputs while
#: avoiding the much smaller Vertex inline-upload cap that :func:`bench.gemini_blind_review.validate_clip`
#: applies by default.
MAX_SOURCE_CLIP_BYTES = 1 << 31

REPORT_NAME = "review.json"

SCOPE_NOTE = (
    "Research-only local position reversal of blinded multiscale A/B review clips. It decodes an existing "
    "review clip, exchanges its equal-width A/B panels, repaints the fixed A-left/B-right labels, re-encodes "
    "at the original fps/dimensions, and copies the review report with only the selected pairs, their clips "
    "renamed and blind_mapping.left_panel/right_panel reversed. It computes no image-quality metric, ranking, "
    "winner or score and changes no production default."
)


class PositionReversalError(ValueError):
    """A position reversal was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise PositionReversalError(code, message)


# --------------------------------------------------------------------------------------
# panel swap and fixed labels (pure pixel operations)
# --------------------------------------------------------------------------------------

def swap_panels(frame):
    """Exchange the two equal-width halves of an A/B frame ``[H, W, 3]``; returns a new array.

    ``W`` must be even. The result keeps the left/right pixel layout but places the original right panel on
    the left and the original left panel on the right.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be [H, W, 3], got shape {frame.shape}")
    width = int(frame.shape[1])
    if width < 2 or width % 2 != 0:
        raise ValueError(f"frame width must be a positive even number, got {width}")
    half = width // 2
    return np.concatenate([frame[:, half:], frame[:, :half]], axis=1)


def swap_and_relabel(frame):
    """Swap the A/B panels and repaint the fixed A-left / B-right bitmap labels.

    The already-rendered clip bakes an ``A`` label into the left half and a ``B`` label into the right half.
    After the halves are exchanged those baked labels are in the wrong places, so
    :func:`bench.multiscale_real_review.draw_bitmap_label` paints the label box back into each new half:
    ``A`` on the left, ``B`` on the right. The label box fully covers the baked bitmap (same origin, scale and
    padding), so the result carries exactly one fixed label per side.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be [H, W, 3], got shape {frame.shape}")
    width = int(frame.shape[1])
    if width < 2 or width % 2 != 0:
        raise ValueError(f"frame width must be a positive even number, got {width}")
    half = width // 2
    new_left = np.array(frame[:, half:], copy=True)   # original right panel, baked 'B'
    new_right = np.array(frame[:, :half], copy=True)  # original left panel, baked 'A'
    R.draw_bitmap_label(new_left, "A")
    R.draw_bitmap_label(new_right, "B")
    return np.concatenate([new_left, new_right], axis=1)


# --------------------------------------------------------------------------------------
# naming and selection
# --------------------------------------------------------------------------------------

def reversed_clip_name(clip):
    """Deterministic new basename for a swapped clip: ``<stem>_posrev<.mp4>``."""
    path = Path(clip)
    return f"{path.stem}_posrev{path.suffix}"


def _plain_dir_name(name):
    return isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name


def _pair_indices(review):
    """Return the (list position, pair_index, pair) triples, refusing duplicate or malformed indices."""
    triples = []
    seen = set()
    for position, pair in enumerate(review["pairs"]):
        _require(isinstance(pair, dict), "pair_malformed", f"review pair {position} must be a JSON object")
        idx = pair.get("pair_index", position)
        _require(isinstance(idx, int) and not isinstance(idx, bool), "pair_malformed",
                 f"review pair {position} pair_index must be an integer, got {idx!r}")
        _require(idx not in seen, "duplicate_pair_index",
                 f"review pair_index {idx} appears more than once")
        seen.add(idx)
        triples.append((position, idx, pair))
    return triples


def _select_pairs(review, pair_indices):
    triples = _pair_indices(review)
    available = [idx for _, idx, _ in triples]
    if pair_indices is None:
        want = list(available)
    else:
        want = []
        for value in pair_indices:
            _require(isinstance(value, int) and not isinstance(value, bool), "invalid_pair_index",
                     f"--pair-index must be an integer, got {value!r}")
            _require(value in available, "unknown_pair_index",
                     f"--pair-index {value} is not in this review (available: {available})")
            if value not in want:
                want.append(value)
        _require(bool(want), "no_pairs_selected", "no --pair-index values were selected")
    wanted = set(want)
    return [(position, idx, pair) for position, idx, pair in triples if idx in wanted]


# --------------------------------------------------------------------------------------
# output planning and collision refusal
# --------------------------------------------------------------------------------------

def _default_out_dir(review_path, review):
    name = review_path.parent.name
    if not _plain_dir_name(name):
        name = review["output"]["dir"]
    return DEFAULT_OUT_ROOT / name


def _resolve(path):
    return Path(path).resolve()


def _plan_outputs(review_path, review, selected, out_dir):
    """Resolve source clips, plan new clip names/report path and refuse any output/input collision."""
    review_path = Path(review_path)
    review_dir = _resolve(review_path.parent)

    all_clips = []
    for pair in review["pairs"]:
        resolved = _resolve(G.resolve_clip_path(review_path, review, pair["clip"]))
        all_clips.append(resolved)

    out_dir_path = Path(out_dir) if out_dir is not None else _default_out_dir(review_path, review)
    _require(_plain_dir_name(out_dir_path.name), "unsafe_out_dir",
             "output directory must have a plain final component (no '.', '..' or empty name)")
    out_resolved = _resolve(out_dir_path)
    _require(out_resolved != review_dir, "output_collision",
             f"output directory {out_dir_path} is the review directory; refusing to write alongside the "
             f"source review")
    source_dirs = {clip.parent for clip in all_clips}
    _require(out_resolved not in source_dirs, "output_collision",
             f"output directory {out_dir_path} is a source clip directory; refusing to write alongside the "
             f"source clips")

    new_names = {}
    for _, idx, pair in selected:
        source = _resolve(G.resolve_clip_path(review_path, review, pair["clip"]))
        G.validate_clip(source, limit_bytes=MAX_SOURCE_CLIP_BYTES)
        name = reversed_clip_name(pair["clip"])
        _require(G._plain_mp4_basename(name), "unsafe_clip_name",
                 f"reversed clip name {name!r} is not a plain .mp4 basename")
        new_names[idx] = name
    _require(len(set(new_names.values())) == len(new_names), "duplicate_clip_name",
             "two selected pairs would be written to the same reversed clip name")

    protected = {_resolve(review_path)} | set(all_clips)
    report_path = out_dir_path / REPORT_NAME
    _require(_resolve(report_path) not in protected, "output_collision",
             f"output report {report_path} would overwrite an input review or clip")
    for name in new_names.values():
        _require(_resolve(out_dir_path / name) not in protected, "output_collision",
                 f"output clip {name!r} would overwrite an input review or clip")
    return out_dir_path, new_names, report_path


# --------------------------------------------------------------------------------------
# decoding, swapping and re-encoding
# --------------------------------------------------------------------------------------

def _reverse_clip(source_path, out_path, *, write):
    """Decode one progressive A/B clip, swap/relabel every frame and encode it at the original fps/size."""
    source_path = Path(source_path)
    info = media.probe(source_path)
    _require(not bool(info.get("interlaced", False)), "clip_not_progressive",
             f"source clip {source_path.name!r} is interlaced; the review clips are progressive")
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    fps = float(info.get("fps", 0.0))
    _require(width > 0 and height > 0 and fps > 0.0, "clip_probe_invalid",
             f"could not read positive dimensions/fps from {source_path.name!r}")
    _require(width % 2 == 0, "clip_width_not_even",
             f"source clip {source_path.name!r} width {width} is not even; A/B panels must be equal width")

    decoder, effective_fps, (decoded_w, decoded_h) = media.decode(
        source_path, info, deinterlace=False)
    _require((int(decoded_w), int(decoded_h)) == (width, height), "clip_dimension_mismatch",
             f"decoded {decoded_w}x{decoded_h} does not match probed {width}x{height}")
    fps = float(effective_fps)

    encoder = None
    if write:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoder = media.Encoder(str(out_path), width, height, fps)
    frames = 0
    try:
        for frame in decoder:
            frame = np.asarray(frame)
            _require(frame.ndim == 3 and frame.shape[2] == 3
                     and int(frame.shape[0]) == height and int(frame.shape[1]) == width,
                     "clip_frame_mismatch",
                     f"decoded frame {frame.shape} does not match {height}x{width}x3")
            if encoder is not None:
                encoder.write(swap_and_relabel(frame))
            frames += 1
    except BaseException:
        # A decode/write/frame-validation failure is the primary error. Still release the encoder, but a
        # close failure must not mask it: the caller needs the specific decode/write reason, and a failure to
        # flush a clip that is already abandoned is secondary.
        if encoder is not None:
            try:
                encoder.close()
            except BaseException:
                pass
        raise
    else:
        # Success path: a close failure (e.g. a dropped ffmpeg pipe) is the only error and must propagate.
        if encoder is not None:
            encoder.close()
    _require(frames > 0, "clip_no_frames", f"source clip {source_path.name!r} decoded zero frames")
    return {"width": width, "height": height, "panel_width": width // 2, "fps": fps, "frames": frames}


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def _run_reversal(review_path, *, out_dir, pair_indices, write):
    review_path = Path(review_path)
    review = G.load_review(review_path)  # validates schema, ok, output.dir, pairs and blind_mapping
    selected = _select_pairs(review, pair_indices)
    out_dir_path, new_names, report_path = _plan_outputs(review_path, review, selected, out_dir)

    selected_indices = [idx for _, idx, _ in selected]
    source_sha = {idx: sha256_file(_resolve(G.resolve_clip_path(review_path, review, pair["clip"])))
                  for _, idx, pair in selected}
    probe_meta = {}
    new_sha = {}
    for _, idx, pair in selected:
        source = _resolve(G.resolve_clip_path(review_path, review, pair["clip"]))
        probe_meta[idx] = _reverse_clip(source, out_dir_path / new_names[idx], write=write)
        if write:
            target = out_dir_path / new_names[idx]
            new_sha[new_names[idx]] = sha256_file(target) if target.is_file() else None

    report = copy.deepcopy(review)
    first = probe_meta[selected_indices[0]]
    report["output"]["dir"] = out_dir_path.name
    report["output"]["clips"] = [new_names[idx] for _, idx, _ in selected]
    report["output"]["comparisons"] = len(selected)
    report["output"]["panel_width"] = first["panel_width"]
    report["output"]["panel_height"] = first["height"]
    report["output"]["frame_width"] = first["width"]
    report["output"]["frame_height"] = first["height"]
    report["output"]["fps"] = first["fps"]
    report["output"]["frames"] = first["frames"]

    report_pairs = []
    for _, idx, pair in selected:
        copied = copy.deepcopy(pair)
        copied["clip"] = new_names[idx]
        mapping = copied["blind_mapping"]
        original = pair["blind_mapping"]
        mapping["left_panel"], mapping["right_panel"] = original["right_panel"], original["left_panel"]
        mapping["video_labels"] = {"left": "A", "right": "B"}
        report_pairs.append(copied)
    report["pairs"] = report_pairs

    report["position_reversal"] = {
        "metadata_schema": METADATA_SCHEMA,
        "tool": "bench/multiscale_position_reversal.py",
        "source_review": review_path.name,
        "source_review_sha256": sha256_file(review_path),
        "source_review_dir": review_path.parent.name,
        "source_output_dir": review["output"]["dir"],
        "selected_pair_indices": selected_indices,
        "swapped_pair_indices": selected_indices,
        "clip_renames": {pair["clip"]: new_names[idx] for _, idx, pair in selected},
        "source_clip_sha256": {pair["clip"]: source_sha[idx] for _, idx, pair in selected},
        "new_clip_sha256": {new_names[idx]: new_sha.get(new_names[idx]) for _, idx, _ in selected},
        "panel_swap": (
            "the two equal-width panels of each selected clip were exchanged; the fixed A-left/B-right "
            "bitmap labels were repainted with bench.multiscale_real_review.draw_bitmap_label so A stays on "
            "the left and B stays on the right"
        ),
        "mapping_reversal": (
            "for every swapped pair, blind_mapping.left_panel and blind_mapping.right_panel were exchanged; "
            "blind_mapping.video_labels stay {left: 'A', right: 'B'}"
        ),
        "mapping_semantics": (
            "the A/B sides a reviewer sees are unchanged; only which method hides behind A vs B is flipped so "
            "the recorded mapping matches the swapped pixels"
        ),
        "top_level_blinding_note": (
            "the copied blinding.seed/assignment describe the pre-reversal source review; the per-pair "
            "blind_mapping is the authoritative post-swap mapping"
        ),
        "no_network": True,
        "note": (
            "Local position reversal only: no image-quality metric, ranking, winner or score is computed and "
            "no source review or clip is modified."
        ),
    }
    # Replace the copied scope note with the reversal scope while keeping the report self-describing.
    report["scope_note"] = SCOPE_NOTE
    report["ok"] = True

    if write:
        write_manifest(report_path, report)
    return report


def run_reversal(review_path, *, out_dir=None, pair_indices=None, write=True):
    """Build a position-reversed local clone of selected pairs from one ``multiscale_real_review`` report.

    ``review_path`` is an existing ``review.json``; ``out_dir`` defaults to the ignored
    ``out/bench/gemini_review/position_reversal/<review-dir-name>``. ``pair_indices`` selects which
    ``pair_index`` values to swap (``None`` selects all). Each selected clip is decoded, its equal-width
    panels exchanged, its fixed A-left/B-right labels repainted, and re-encoded at the original fps and
    dimensions. The returned report copies the source report, keeps only the selected pairs, renames their
    clips, reverses ``blind_mapping.left_panel``/``right_panel`` and records ``position_reversal`` metadata.
    ``write=False`` runs the whole pipeline but writes no clips or report. Refusals raise
    :class:`PositionReversalError` (path-validation refusals from
    :mod:`bench.gemini_blind_review` are re-raised with the same stable ``code``). No network is used.
    """
    try:
        return _run_reversal(review_path, out_dir=out_dir, pair_indices=pair_indices, write=write)
    except G.GeminiReviewError as exc:
        raise PositionReversalError(exc.code, str(exc)) from exc


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="bench/multiscale_position_reversal.py",
        description="Local position reversal of blinded multiscale A/B review clips: decode a selected "
                    "existing review clip, swap its A/B panels, repaint the fixed A-left/B-right labels, "
                    "re-encode it and copy the review report with the hidden mapping reversed for every "
                    "swapped pair. No network, no quality metric.")
    parser.add_argument("review", help="path to a multiscale_real_review.py review.json")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: "
                             "out/bench/gemini_review/position_reversal/<review-dir-name>)")
    parser.add_argument("--pair-index", type=int, action="append", dest="pair_indices", default=None,
                        metavar="N", help="pair_index to swap (repeatable; default: every pair)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_reversal(args.review, out_dir=args.out_dir, pair_indices=args.pair_indices)
    except PositionReversalError as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False,
                          "error": {"code": exc.code, "message": str(exc)}}, sort_keys=True), file=sys.stderr)
        return 2
    print(f"schema={report['schema']} comparisons={report['output']['comparisons']} "
          f"clips={report['output']['clips']} out={report['output']['dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "METADATA_SCHEMA",
    "DEFAULT_OUT_ROOT",
    "MAX_SOURCE_CLIP_BYTES",
    "REPORT_NAME",
    "SCOPE_NOTE",
    "PositionReversalError",
    "swap_panels",
    "swap_and_relabel",
    "reversed_clip_name",
    "run_reversal",
    "build_parser",
    "main",
]
