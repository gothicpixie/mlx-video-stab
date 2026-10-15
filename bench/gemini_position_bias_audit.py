"""Research-only, deterministic auditor for Vertex Gemini positional-counterbalance reviews.

What this does
--------------
A ``bench.multiscale_real_review`` run writes blinded A/B review clips whose hidden ``blind_mapping`` records
which side (left/A versus right/B) is the multiscale-retention candidate and which is its nearest measured-crop
L1 counterpart. ``bench.gemini_blind_review`` sends those clips to Vertex Gemini and stores, per clip, the raw
A/B/tie side choice and the hidden method that side mapped to. ``bench.multiscale_position_reversal`` builds a
local **position-reversed** clone of selected pairs: it exchanges the two panels, repaints the fixed
A-left/B-right labels, renames the clips, but **keeps the original run/source/window/pair identity** and swaps
``blind_mapping.left_panel``/``right_panel``.

This module reads those already-recorded result JSONs (and, for the oldest legacy original report, a supplied
``multiscale_real_review`` ``review.json``) and audits the counterbalance **offline**. It performs no network
or model call, decodes no video, imports no media stack, and never modifies an input file. For each original /
reversal pair it reports the screen-side choices and the hidden-method choices **separately**, then derives:

* ``side_following`` — the same raw A/B side in both (non-tie), i.e. a screen-position choice;
* ``method_agreement`` — the same hidden method in both (non-tie), i.e. a content choice;
* ``opposite_physical_sides`` — the two raw sides differ (non-tie), which a correctly reversed mapping pairs
  with ``method_agreement``;
* ``position_tracks_while_method_changes`` — the same side was chosen while the hidden method behind that side
  changed, the exact positional-confound pattern.

Observational evidence only
---------------------------
The counts are descriptive properties of these recorded judgments. The sample is small and dependent, the
reviewer is a single model over short low-resolution translation-only clips, and a count of side-following
pairs says nothing about the reviewer in general. This module computes **no** winner, ranking, total,
significance test or quality score, selects no method and calls no method better.

Join and refusal rules
----------------------
Original and reversal records are joined only on stable identity: ``run_id``, ``source_sha256``, the source
window ``start_s``/``duration_s``, and ``pair_index``. Clip name and clip digest are deliberately excluded
because a reversed control changes them. Preferences and hidden mappings are normalized and validated;
duplicate, malformed, missing-mapping and (unless ``--allow-unmatched`` is given) unmatched identity keys fail
clearly instead of silently dropping records. The report is written atomically and carries input/report hashes.

Usage::

    python -m bench.gemini_position_bias_audit \\
        --original-result ORIG.json [--original-result ...] \\
        --reversal-result REV.json [--reversal-result ...] \\
        --source-review REVIEW.json [--source-review ...] \\
        --out OUT.json [--allow-unmatched]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from . import gemini_blind_review as G
from .manifest import dumps_json, sha256_file, write_manifest

SCHEMA = "mlxstab.gemini_position_bias_audit/1"
#: Schema of a standard ``bench.gemini_blind_review`` result, for both originals and reversals.
ORIGINAL_SCHEMA = G.OUTPUT_SCHEMA
#: Schema of the ``bench.multiscale_real_review`` report used to recover legacy originals.
SOURCE_REVIEW_SCHEMA = G.INPUT_SCHEMA

RAW_SIDES = ("A", "B", "tie")
METHODS = ("multiscale", "l1", "tie")

SCOPE_NOTE = (
    "Read-only, deterministic audit of already-recorded Vertex Gemini positional-counterbalance review JSON. "
    "It performs no network or model call, decodes no video and changes no default. It joins each original and "
    "position-reversed judgment on stable identity (run_id, source_sha256, source window start/duration, "
    "pair_index) and reports what each recorded judgment selected: its raw A/B screen side and the hidden "
    "method that side mapped to. Derived booleans compare the two: side_following means the same raw side "
    "(non-tie), method_agreement means the same mapped method (non-tie), and opposite_physical_sides means the "
    "two raw sides differ (non-tie), which a correctly reversed mapping pairs with method agreement. "
    "Observational evidence only: the sample is small and dependent, the reviewer is a single model over short "
    "low-resolution translation-only clips, and no causal, significance or product-quality conclusion follows."
)

NO_QUALITY_SCORE_NOTE = (
    "Observational audit only. It reports recorded A/B side choices and the hidden method they map to, "
    "separately, plus position-versus-method agreement counts. It computes no winner, ranking, total, "
    "significance test or quality score and calls no method better; a count of side-following pairs is a "
    "property of these recorded judgments, not an estimate of the reviewer's behavior."
)


class PositionBiasAuditError(ValueError):
    """An audit input was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise PositionBiasAuditError(code, message)


# --------------------------------------------------------------------------------------
# JSON and normalization helpers (pure, no I/O beyond reading an input file)
# --------------------------------------------------------------------------------------

def _finite_number(value) -> bool:
    """True for a real, finite int/float (bools excluded); ``None``, NaN and +/-inf are not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_json(path, *, kind):
    """Read one input file and return ``(document, raw_bytes)``; refusals carry stable codes."""
    path = Path(path)
    _require(path.is_file(), "input_missing", f"no {kind} file at {path}")
    try:
        raw = path.read_bytes()
    except OSError:
        raise PositionBiasAuditError("input_unreadable", f"cannot read {kind} file {path}")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise PositionBiasAuditError("input_unreadable", f"{kind} file {path} is not valid JSON")
    _require(isinstance(document, dict), "input_malformed",
             f"{kind} file {path} top level must be a JSON object")
    return document, raw


def _validated_mapping(mapping, where):
    """Validate an A-left / B-right hidden mapping and return a copy; missing mappings are distinct."""
    if mapping is None:
        raise PositionBiasAuditError("missing_mapping", f"{where}: blind_mapping is missing")
    try:
        return dict(G.validate_blind_mapping(mapping))
    except G.GeminiReviewError as exc:
        raise PositionBiasAuditError(exc.code, f"{where}: {exc}") from exc


def _mapped(mapping, side, where):
    try:
        return G.map_preference(mapping, side)
    except G.GeminiReviewError as exc:
        raise PositionBiasAuditError(exc.code, f"{where}: {exc}") from exc


def _normalize_side(value, where):
    _require(value in G.PREFERENCES, "preference_invalid",
             f"{where}: raw side preference {value!r} is not one of {list(G.PREFERENCES)}")
    return value


def _normalize_confidence(value, where):
    _require(value in G.CONFIDENCES, "confidence_invalid",
             f"{where}: confidence {value!r} is not one of {list(G.CONFIDENCES)}")
    return value


def _normalize_window(value, where):
    _require(isinstance(value, dict), "judgment_malformed", f"{where}: source_window must be an object")
    start, duration = value.get("start_s"), value.get("duration_s")
    _require(_finite_number(start) and _finite_number(duration), "judgment_malformed",
             f"{where}: source_window.start_s/duration_s must be finite numbers")
    _require(float(start) >= 0.0 and float(duration) > 0.0, "judgment_malformed",
             f"{where}: source_window.start_s must be non-negative and duration_s must be positive")
    return {"start_s": float(start), "duration_s": float(duration)}


def _normalize_pair_index(value, where):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "judgment_malformed",
             f"{where}: pair_index must be a non-negative integer")
    return int(value)


def _normalize_sha256(value, where):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
             "judgment_malformed", f"{where}: source_sha256 must be a 64-character hexadecimal SHA-256")
    return value.lower()


# --------------------------------------------------------------------------------------
# stable identity
# --------------------------------------------------------------------------------------

def _identity(record):
    return (record["run_id"], record["source_sha256"],
            record["source_window"]["start_s"], record["source_window"]["duration_s"],
            record["pair_index"])


def _key_dict(key):
    run_id, source_sha256, start, duration, pair_index = key
    return {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "source_window": {"start_s": start, "duration_s": duration},
        "pair_index": pair_index,
    }


def _index(records, which):
    """Index records by stable identity, refusing any duplicate key instead of overwriting one."""
    index = {}
    for record in records:
        key = _identity(record)
        if key in index:
            raise PositionBiasAuditError("duplicate_key",
                                         f"duplicate {which} identity {json.dumps(_key_dict(key), sort_keys=True)}")
        index[key] = record
    return index


# --------------------------------------------------------------------------------------
# record normalization: standard and legacy result shapes
# --------------------------------------------------------------------------------------

def _record_from_standard(judgment, where, *, source_result):
    """Normalize one ``mlxstab.gemini_blind_review/1`` judgment entry."""
    _require(isinstance(judgment, dict), "judgment_malformed", f"{where}: judgment must be a JSON object")
    run_id = judgment.get("run_id", judgment.get("run"))
    _require(isinstance(run_id, str) and run_id.strip(), "judgment_malformed",
             f"{where}: run_id is missing or empty")
    source_sha256 = _normalize_sha256(judgment.get("source_sha256"), where)
    source_window = _normalize_window(judgment.get("source_window"), where)
    pair_index = _normalize_pair_index(judgment.get("pair_index"), where)
    side = _normalize_side(judgment.get("raw_side_preference"), where)
    mapping = _validated_mapping(judgment.get("blind_mapping"), where)
    mapped = _mapped(mapping, side, where)
    stored = judgment.get("mapped_preference")
    if stored is not None:
        _require(stored == mapped, "mapped_preference_mismatch",
                 f"{where}: mapped_preference {stored!r} does not match blind_mapping + raw side {mapped!r}")
    confidence = _normalize_confidence(judgment.get("confidence"), where)
    return {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "source_window": source_window,
        "pair_index": pair_index,
        "blind_mapping": mapping,
        "raw_side": side,
        "mapped_method": mapped,
        "confidence": confidence,
        "clip": judgment.get("clip"),
        "clip_sha256": judgment.get("clip_sha256"),
        "source_kind": "standard",
        "source_result": source_result,
    }


def _collect_originals(paths, reviews):
    """Load every original result; legacy (schema-less) results are recovered through ``reviews``."""
    records = []
    inputs = []
    for path in paths:
        document, _raw = _read_json(path, kind="original result")
        kind = _classify(document, path)
        entries = document.get("judgments")
        _require(isinstance(entries, list), "input_malformed", f"{path}: judgments must be a JSON list")
        for index, judgment in enumerate(entries):
            where = f"original {path} judgment {index}"
            if kind == "standard":
                records.append(_record_from_standard(judgment, where, source_result=str(path)))
            else:
                records.append(_record_from_legacy(judgment, reviews, where, source_result=str(path)))
        inputs.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "kind": kind,
            "schema": document.get("schema"),
            "judgments": len(entries),
        })
    return records, inputs


def _collect_reversals(paths):
    """Load every reversal result; reversals must be standard ``gemini_blind_review`` reports."""
    records = []
    inputs = []
    for path in paths:
        document, _raw = _read_json(path, kind="reversal result")
        _require(document.get("schema") == ORIGINAL_SCHEMA, "schema_unsupported",
                 f"reversal result {path}: schema {document.get('schema')!r} must be {ORIGINAL_SCHEMA!r}")
        entries = document.get("judgments")
        _require(isinstance(entries, list), "input_malformed", f"{path}: judgments must be a JSON list")
        for index, judgment in enumerate(entries):
            where = f"reversal {path} judgment {index}"
            records.append(_record_from_standard(judgment, where, source_result=str(path)))
        inputs.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "kind": "standard",
            "schema": document.get("schema"),
            "judgments": len(entries),
        })
    return records, inputs


def _classify(document, path):
    schema = document.get("schema")
    if schema == ORIGINAL_SCHEMA:
        return "standard"
    if schema is None:
        return "legacy"
    raise PositionBiasAuditError("schema_unsupported",
                                 f"{path}: result schema {schema!r} is neither {ORIGINAL_SCHEMA!r} nor the "
                                 f"legacy (schema-less) shape")


# --------------------------------------------------------------------------------------
# legacy original recovery through a supplied multiscale_real_review source report
# --------------------------------------------------------------------------------------

def _load_source_reviews(paths):
    """Validate each ``multiscale_real_review`` report and index its pairs by clip."""
    reviews = []
    inputs = []
    for path in paths:
        _read_json(path, kind="source review")
        try:
            review = G.load_review(path)
        except G.GeminiReviewError as exc:
            raise PositionBiasAuditError(exc.code, f"source review {path}: {exc}") from exc
        _require(review.get("schema") == SOURCE_REVIEW_SCHEMA, "schema_unsupported",
                 f"source review {path}: schema {review.get('schema')!r} must be {SOURCE_REVIEW_SCHEMA!r}")
        entries = []
        for position, pair in enumerate(review["pairs"]):
            index = _normalize_pair_index(pair.get("pair_index", position),
                                          f"source review {path} pair {position}")
            entries.append((index, pair))
        provenance = review["provenance"]
        reviews.append({
            "path": str(path),
            "output_dir": review["output"]["dir"],
            "parent_name": Path(path).parent.name,
            "provenance": provenance,
            "pairs": entries,
        })
        inputs.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "schema": review.get("schema"),
            "output_dir": review["output"]["dir"],
            "run_id": provenance.get("run_id"),
            "source_sha256": provenance.get("source_sha256"),
            "source_window": _normalize_window(provenance.get("source_window"), f"source review {path}"),
            "pairs": len(entries),
        })
    return reviews, inputs


def _match_source_review(window, reviews, where):
    matches = [review for review in reviews
               if window == review["output_dir"] or window == review["parent_name"]]
    if not matches:
        raise PositionBiasAuditError(
            "legacy_window_unmatched",
            f"{where}: legacy window {window!r} matches no supplied --source-review output.dir or parent "
            f"directory")
    if len(matches) > 1:
        raise PositionBiasAuditError(
            "legacy_window_ambiguous",
            f"{where}: legacy window {window!r} matches {len(matches)} supplied source reviews")
    return matches[0]


def _match_pair(clip, review, where):
    matches = [entry for entry in review["pairs"] if entry[1].get("clip") == clip]
    if not matches:
        raise PositionBiasAuditError(
            "legacy_clip_unmatched",
            f"{where}: legacy clip {clip!r} matches no pair in source review {review['output_dir']!r}")
    if len(matches) > 1:
        raise PositionBiasAuditError(
            "legacy_clip_ambiguous",
            f"{where}: legacy clip {clip!r} matches {len(matches)} pairs in source review "
            f"{review['output_dir']!r}")
    return matches[0]


def _record_from_legacy(judgment, reviews, where, *, source_result):
    """Recover a legacy ``{window, clip, bytes, judgment:{...}}`` entry from its source review mapping."""
    _require(isinstance(judgment, dict), "judgment_malformed", f"{where}: judgment must be a JSON object")
    window = judgment.get("window")
    _require(isinstance(window, str) and window.strip(), "judgment_malformed",
             f"{where}: legacy window is missing or empty")
    clip = judgment.get("clip")
    _require(isinstance(clip, str) and clip.strip(), "judgment_malformed",
             f"{where}: legacy clip is missing or empty")
    review = _match_source_review(window, reviews, where)
    pair_index, pair = _match_pair(clip, review, where)
    inner = judgment.get("judgment")
    _require(isinstance(inner, dict), "judgment_malformed",
             f"{where}: legacy judgment.judgment must be a JSON object")
    side = _normalize_side(inner.get("preference"), where)
    confidence = _normalize_confidence(inner.get("confidence"), where)
    mapping = _validated_mapping(pair.get("blind_mapping"), where)
    mapped = _mapped(mapping, side, where)
    provenance = review["provenance"]
    run_id = provenance.get("run_id", provenance.get("run"))
    _require(isinstance(run_id, str) and run_id.strip(), "judgment_malformed",
             f"{where}: source review provenance.run_id is missing or empty")
    source_sha256 = _normalize_sha256(provenance.get("source_sha256"), where)
    return {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "source_window": _normalize_window(provenance.get("source_window"), where),
        "pair_index": pair_index,
        "blind_mapping": mapping,
        "raw_side": side,
        "mapped_method": mapped,
        "confidence": confidence,
        "clip": clip,
        "clip_sha256": None,
        "source_kind": "legacy",
        "source_result": source_result,
    }


# --------------------------------------------------------------------------------------
# derived math and aggregates
# --------------------------------------------------------------------------------------

def _derive(original, reversal):
    """Screen-side versus hidden-method agreement for one complete original/reversal pair."""
    o_side, r_side = original["raw_side"], reversal["raw_side"]
    o_method, r_method = original["mapped_method"], reversal["mapped_method"]
    both_non_tie = o_side != "tie" and r_side != "tie"
    side_following = both_non_tie and o_side == r_side
    method_agreement = o_method != "tie" and r_method != "tie" and o_method == r_method
    opposite_physical_sides = both_non_tie and o_side != r_side
    position_tracks_while_method_changes = side_following and o_method != r_method
    original_mapping = original["blind_mapping"]
    reversal_mapping = reversal["blind_mapping"]
    mapping_reversed = (
        original_mapping["left_panel"] == reversal_mapping["right_panel"]
        and original_mapping["right_panel"] == reversal_mapping["left_panel"]
    )
    return {
        "both_non_tie": both_non_tie,
        "side_following": side_following,
        "method_agreement": method_agreement,
        "opposite_physical_sides": opposite_physical_sides,
        "position_tracks_while_method_changes": position_tracks_while_method_changes,
        "mapping_reversed": mapping_reversed,
    }


def _counts(values, universe):
    return {name: sum(1 for value in values if value == name) for name in universe}


def _aggregate(pairs):
    original_sides = [pair["original"]["raw_side"] for pair in pairs]
    reversal_sides = [pair["reversal"]["raw_side"] for pair in pairs]
    original_methods = [pair["original"]["mapped_method"] for pair in pairs]
    reversal_methods = [pair["reversal"]["mapped_method"] for pair in pairs]
    derived = [pair["derived"] for pair in pairs]
    non_tie = [item for item in derived if item["both_non_tie"]]
    return {
        "both_non_tie_pairs": len(non_tie),
        "original_raw_side_counts": _counts(original_sides, RAW_SIDES),
        "reversal_raw_side_counts": _counts(reversal_sides, RAW_SIDES),
        "original_mapped_method_counts": _counts(original_methods, METHODS),
        "reversal_mapped_method_counts": _counts(reversal_methods, METHODS),
        "side_following_pairs": sum(1 for item in derived if item["side_following"]),
        "method_agreement_pairs": sum(1 for item in derived if item["method_agreement"]),
        "opposite_physical_sides_pairs": sum(1 for item in non_tie if item["opposite_physical_sides"]),
        "position_tracks_while_method_changes_pairs":
            sum(1 for item in derived if item["position_tracks_while_method_changes"]),
        "mapping_reversed_pairs": sum(1 for item in derived if item["mapping_reversed"]),
        "mapping_not_reversed_pairs": sum(1 for item in derived if not item["mapping_reversed"]),
    }


def _side_record(record):
    """Project one normalized record onto the reported per-side fields."""
    return {
        "raw_side": record["raw_side"],
        "mapped_method": record["mapped_method"],
        "confidence": record["confidence"],
        "blind_mapping": dict(record["blind_mapping"]),
        "clip": record["clip"],
        "clip_sha256": record["clip_sha256"],
        "source_kind": record["source_kind"],
        "result": record["source_result"],
    }


def _unmatched_message(unmatched_originals, unmatched_reversals):
    def _preview(keys):
        shown = [f"{key[0]}#{key[4]}@{key[2]}/{key[3]}:{key[1][:12]}" for key in keys[:3]]
        joined = ", ".join(shown)
        if len(keys) > len(shown):
            joined += f", +{len(keys) - len(shown)} more"
        return joined

    parts = []
    if unmatched_originals:
        parts.append(f"{len(unmatched_originals)} original key(s) with no reversal: {_preview(unmatched_originals)}")
    if unmatched_reversals:
        parts.append(f"{len(unmatched_reversals)} reversal key(s) with no original: {_preview(unmatched_reversals)}")
    return ("unmatched identity keys; pass --allow-unmatched to record them without failing. "
            + "; ".join(parts))


# --------------------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------------------

def audit(original_paths, reversal_paths, source_review_paths=(), *, allow_unmatched=False, out_path=None):
    """Audit a positional-counterbalance review and return the deterministic report document.

    ``original_paths``/``reversal_paths`` are result JSONs joined on stable identity;
    ``source_review_paths`` recover legacy (schema-less) originals. With ``allow_unmatched`` false (the
    default) any identity key present on only one side raises :class:`PositionBiasAuditError`; with it true the
    unmatched keys are recorded in ``coverage`` and the complete pairs are still audited. ``out_path`` is only
    used to refuse a write that would overwrite an input.
    """
    original_paths = list(original_paths)
    reversal_paths = list(reversal_paths)
    source_review_paths = list(source_review_paths)
    _require(original_paths, "no_inputs", "at least one --original-result is required")
    _require(reversal_paths, "no_inputs", "at least one --reversal-result is required")
    if out_path is not None:
        protected = {Path(path).resolve() for path in original_paths + reversal_paths + source_review_paths}
        _require(Path(out_path).resolve() not in protected, "input_collision",
                 f"--out {out_path} would overwrite an input file")

    reviews, source_inputs = _load_source_reviews(source_review_paths)
    originals, original_inputs = _collect_originals(original_paths, reviews)
    reversals, reversal_inputs = _collect_reversals(reversal_paths)

    original_index = _index(originals, "original")
    reversal_index = _index(reversals, "reversal")
    unmatched_originals = sorted(set(original_index) - set(reversal_index))
    unmatched_reversals = sorted(set(reversal_index) - set(original_index))
    if not allow_unmatched and (unmatched_originals or unmatched_reversals):
        raise PositionBiasAuditError("unmatched_key",
                                     _unmatched_message(unmatched_originals, unmatched_reversals))

    pairs = []
    for key in sorted(set(original_index) & set(reversal_index)):
        original = original_index[key]
        reversal = reversal_index[key]
        pairs.append({
            "identity": _key_dict(key),
            "original": _side_record(original),
            "reversal": _side_record(reversal),
            "derived": _derive(original, reversal),
        })

    report = {
        "schema": SCHEMA,
        "ok": True,
        "scope_note": SCOPE_NOTE,
        "no_quality_score_note": NO_QUALITY_SCORE_NOTE,
        "identity": {
            "join_key": ["run_id", "source_sha256", "source_window.start_s", "source_window.duration_s",
                         "pair_index"],
            "note": "Original and reversal records are joined only on this stable identity. Clip name and "
                    "clip digest are excluded because a position-reversed control changes them.",
        },
        "inputs": {
            "original_results": sorted(original_inputs, key=lambda item: item["path"]),
            "reversal_results": sorted(reversal_inputs, key=lambda item: item["path"]),
            "source_reviews": sorted(source_inputs, key=lambda item: item["path"]),
        },
        "coverage": {
            "original_records": len(originals),
            "reversal_records": len(reversals),
            "complete_pairs": len(pairs),
            "unmatched_original_count": len(unmatched_originals),
            "unmatched_reversal_count": len(unmatched_reversals),
            "unmatched_originals": [_key_dict(key) for key in unmatched_originals],
            "unmatched_reversals": [_key_dict(key) for key in unmatched_reversals],
            "allow_unmatched": bool(allow_unmatched),
        },
        "aggregates": _aggregate(pairs),
        "pairs": pairs,
    }
    report["report_sha256"] = _report_sha256(report)
    return report


def _report_sha256(report):
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="bench/gemini_position_bias_audit.py",
        description="Research-only, deterministic JSON-only audit of a Vertex Gemini positional-counterbalance "
                    "review. It joins original and position-reversed blind-review results on stable identity, "
                    "reports screen-side choices separately from hidden-method choices, and computes no winner, "
                    "ranking, score or significance claim. No network, no model call, no video decoding.")
    parser.add_argument("--original-result", action="append", required=True, metavar="JSON",
                        dest="original_results",
                        help="original gemini_blind_review result (repeatable; standard or legacy shape)")
    parser.add_argument("--reversal-result", action="append", required=True, metavar="JSON",
                        dest="reversal_results",
                        help="position-reversed gemini_blind_review result (repeatable)")
    parser.add_argument("--source-review", action="append", default=[], metavar="REVIEW.json",
                        dest="source_reviews",
                        help="multiscale_real_review review.json used to recover legacy originals (repeatable)")
    parser.add_argument("--out", required=True,
                        help="path to the versioned audit JSON (written atomically)")
    parser.add_argument("--allow-unmatched", action="store_true",
                        help="record unmatched identity keys in the report instead of failing (strict by "
                             "default)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit(args.original_results, args.reversal_results, args.source_reviews,
                       allow_unmatched=args.allow_unmatched, out_path=args.out)
    except PositionBiasAuditError as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False,
                          "error": {"code": exc.code, "message": str(exc)}}, sort_keys=True), file=sys.stderr)
        return 2
    write_manifest(args.out, report)
    print(f"schema={SCHEMA} pairs={report['coverage']['complete_pairs']} "
          f"non_tie={report['aggregates']['both_non_tie_pairs']} ok=True out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "ORIGINAL_SCHEMA",
    "SOURCE_REVIEW_SCHEMA",
    "RAW_SIDES",
    "METHODS",
    "SCOPE_NOTE",
    "NO_QUALITY_SCORE_NOTE",
    "PositionBiasAuditError",
    "audit",
    "build_parser",
    "main",
]
