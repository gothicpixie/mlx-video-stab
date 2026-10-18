"""Focused tests for the research-only Gemini positional-counterbalance audit.

All inputs are small JSON fixtures: no network, no model call, no MLX, no FFmpeg. Coverage: the
screen-side/hidden-method mapping math, legacy original recovery through a supplied source review, strict
duplicate/missing/unmatched behavior (and the explicit ``--allow-unmatched`` opt-out), tie and confidence
handling, refusal codes, the absence of winner/ranking/score fields, atomic output and byte-stable JSON.
"""
import json

import pytest

from bench import gemini_blind_review as G
from bench import gemini_position_bias_audit as A
from bench.manifest import dumps_json


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def _mapping(left, right):
    return {"left_panel": left, "right_panel": right, "video_labels": {"left": "A", "right": "B"}}


def _standard_judgment(*, run_id="run-1", source_sha256="a" * 64, start=0.0, duration=1.0, pair_index=0,
                       raw="A", confidence="medium", mapping=None, mapped=None, clip="pair_00_ab.mp4",
                       clip_sha256="b" * 64):
    mapping = mapping if mapping is not None else _mapping("multiscale", "l1")
    if mapped is None:
        mapped = {"A": mapping["left_panel"], "B": mapping["right_panel"], "tie": "tie"}[raw]
    return {
        "run_id": run_id,
        "source_sha256": source_sha256,
        "source_window": {"start_s": start, "duration_s": duration},
        "pair_index": pair_index,
        "blind_mapping": mapping,
        "raw_side_preference": raw,
        "mapped_preference": mapped,
        "confidence": confidence,
        "clip": clip,
        "clip_sha256": clip_sha256,
    }


def _write_standard_result(path, judgments):
    path.write_text(json.dumps({"schema": G.OUTPUT_SCHEMA, "complete": True, "judgments": judgments}),
                    encoding="utf-8")
    return path


def _write_source_review(path, *, output_dir, run_id, source_sha256="d" * 64, start=0.0, duration=1.0,
                         pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": G.INPUT_SCHEMA,
        "ok": True,
        "output": {"dir": output_dir},
        "provenance": {"run": run_id, "run_id": run_id, "source_sha256": source_sha256,
                       "source_window": {"start_s": start, "duration_s": duration}},
        "pairs": [{"pair_index": index, "clip": clip, "blind_mapping": mapping}
                  for index, clip, mapping in pairs],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_legacy_result(path, judgments):
    document = {
        "model": "gemini-3.8-flash",
        "provider": "Vertex AI REST, authenticated via gcloud user login",
        "prompt": "legacy review brief",
        "judgments": [
            {"window": window, "clip": clip, "bytes": 1,
             "judgment": {"preference": preference, "confidence": confidence}}
            for window, clip, preference, confidence in judgments
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _pair(audit_report, pair_index):
    for pair in audit_report["pairs"]:
        if pair["identity"]["pair_index"] == pair_index:
            return pair
    raise AssertionError(f"no audited pair with pair_index {pair_index}")


# --------------------------------------------------------------------------------------
# mapping math and derived booleans
# --------------------------------------------------------------------------------------

def test_side_following_when_same_side_hides_opposite_methods(tmp_path):
    # Original: multiscale left / L1 right; the reviewer picks B/right -> L1.
    # Reversal: mapping swapped, same B/right now hides multiscale. Same side, different method.
    original = _write_standard_result(
        tmp_path / "orig.json",
        [_standard_judgment(run_id="r", pair_index=3, raw="B", mapping=_mapping("multiscale", "l1"))])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=3, raw="B", mapping=_mapping("l1", "multiscale"),
                            clip="pair_03_ab_posrev.mp4", clip_sha256="c" * 64)])
    report = A.audit([original], [reversal], [])
    pair = _pair(report, 3)
    assert pair["original"]["raw_side"] == "B" and pair["original"]["mapped_method"] == "l1"
    assert pair["reversal"]["raw_side"] == "B" and pair["reversal"]["mapped_method"] == "multiscale"
    assert pair["derived"] == {
        "both_non_tie": True,
        "side_following": True,
        "method_agreement": False,
        "opposite_physical_sides": False,
        "position_tracks_while_method_changes": True,
        "mapping_reversed": True,
    }
    aggregates = report["aggregates"]
    assert aggregates["side_following_pairs"] == 1
    assert aggregates["position_tracks_while_method_changes_pairs"] == 1
    assert aggregates["method_agreement_pairs"] == 0


def test_aggregates_expose_each_count_once(tmp_path):
    # Guards the report shape: no aggregate key duplicates another field or carries a misleading name.
    report, _, _ = _sample_report(tmp_path)
    aggregates = report["aggregates"]
    assert "complete_pairs" not in aggregates          # already reported under coverage
    assert "right_left_consistent_non_tie_pairs" not in aggregates
    # side_following already implies both_non_tie, so the count is the side-following count.
    assert aggregates["side_following_pairs"] <= aggregates["both_non_tie_pairs"]
    assert set(aggregates) == {
        "both_non_tie_pairs",
        "original_raw_side_counts",
        "reversal_raw_side_counts",
        "original_mapped_method_counts",
        "reversal_mapped_method_counts",
        "side_following_pairs",
        "method_agreement_pairs",
        "opposite_physical_sides_pairs",
        "position_tracks_while_method_changes_pairs",
        "mapping_reversed_pairs",
        "mapping_not_reversed_pairs",
    }


def test_method_agreement_when_reviewer_switches_side_with_the_method(tmp_path):
    # Original picks A/left = multiscale; reversal picks B/right, which after the swap is again multiscale.
    original = _write_standard_result(
        tmp_path / "orig.json",
        [_standard_judgment(run_id="r", pair_index=0, raw="A", mapping=_mapping("multiscale", "l1"))])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=0, raw="B", mapping=_mapping("l1", "multiscale"))])
    pair = _pair(A.audit([original], [reversal], []), 0)
    assert pair["derived"]["side_following"] is False
    assert pair["derived"]["method_agreement"] is True
    assert pair["derived"]["opposite_physical_sides"] is True
    assert pair["derived"]["position_tracks_while_method_changes"] is False


def test_ties_are_complete_but_follow_neither_side_nor_method(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json",
        [_standard_judgment(run_id="r", pair_index=1, raw="tie", confidence="low",
                            mapping=_mapping("multiscale", "l1"))])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=1, raw="tie", confidence="low",
                            mapping=_mapping("l1", "multiscale"))])
    report = A.audit([original], [reversal], [])
    pair = _pair(report, 1)
    assert report["coverage"]["complete_pairs"] == 1
    assert pair["original"]["raw_side"] == "tie" and pair["original"]["mapped_method"] == "tie"
    assert pair["reversal"]["raw_side"] == "tie" and pair["reversal"]["mapped_method"] == "tie"
    for derived in ("side_following", "method_agreement", "opposite_physical_sides",
                    "position_tracks_while_method_changes"):
        assert pair["derived"][derived] is False
    assert pair["derived"]["mapping_reversed"] is True
    assert report["aggregates"]["original_raw_side_counts"]["tie"] == 1
    assert report["aggregates"]["reversal_mapped_method_counts"]["tie"] == 1


def test_mixed_tie_and_non_tie_pair_is_not_side_following(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=2, raw="B")])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=2, raw="tie", mapping=_mapping("l1", "multiscale"))])
    pair = _pair(A.audit([original], [reversal], []), 2)
    assert pair["derived"]["both_non_tie"] is False
    assert pair["derived"]["side_following"] is False
    assert pair["derived"]["method_agreement"] is False
    assert pair["derived"]["opposite_physical_sides"] is False


def test_confidences_are_reported_per_side(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=0, raw="B", confidence="high")])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=0, raw="B", confidence="low",
                            mapping=_mapping("l1", "multiscale"))])
    pair = _pair(A.audit([original], [reversal], []), 0)
    assert pair["original"]["confidence"] == "high"
    assert pair["reversal"]["confidence"] == "low"


# --------------------------------------------------------------------------------------
# join identity: clip name/digest change is ignored
# --------------------------------------------------------------------------------------

def test_join_ignores_clip_name_and_digest(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json",
        [_standard_judgment(run_id="r", pair_index=3, raw="B", clip="pair_03_ab.mp4",
                            clip_sha256="a" * 64)])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=3, raw="B", mapping=_mapping("l1", "multiscale"),
                            clip="pair_03_ab_posrev.mp4", clip_sha256="z" * 64)])
    report = A.audit([original], [reversal], [])
    assert report["coverage"]["complete_pairs"] == 1
    assert report["pairs"][0]["original"]["clip"] == "pair_03_ab.mp4"
    assert report["pairs"][0]["reversal"]["clip"] == "pair_03_ab_posrev.mp4"


def test_same_identity_different_window_does_not_join(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", start=0.0, duration=1.0, pair_index=3)])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", start=60.0, duration=1.0, pair_index=3,
                            mapping=_mapping("l1", "multiscale"))])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "unmatched_key"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha256", "not-a-sha256"),
        ("pair_index", -1),
        ("source_window", {"start_s": -0.1, "duration_s": 1.0}),
        ("source_window", {"start_s": 0.0, "duration_s": 0.0}),
    ],
)
def test_malformed_stable_identity_fails(tmp_path, field, value):
    judgment = _standard_judgment(run_id="r", pair_index=0)
    judgment[field] = value
    original = _write_standard_result(tmp_path / "orig.json", [judgment])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=0, mapping=_mapping("l1", "multiscale"))],
    )
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "judgment_malformed"


# --------------------------------------------------------------------------------------
# legacy original recovery
# --------------------------------------------------------------------------------------

def test_legacy_recovery_via_output_dir(tmp_path):
    source = _write_source_review(
        tmp_path / "src" / "review.json", output_dir="runwin", run_id="legacy-run", source_sha256="d" * 64,
        start=2.0, duration=1.0, pairs=[(3, "pair_03_ab.mp4", _mapping("multiscale", "l1"))])
    legacy = _write_legacy_result(tmp_path / "legacy.json", [("runwin", "pair_03_ab.mp4", "B", "medium")])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="legacy-run", source_sha256="d" * 64, start=2.0, duration=1.0,
                            pair_index=3, raw="B", mapping=_mapping("l1", "multiscale"))])
    report = A.audit([legacy], [reversal], [source])
    pair = _pair(report, 3)
    assert pair["original"]["source_kind"] == "legacy"
    assert pair["original"]["raw_side"] == "B"
    assert pair["original"]["mapped_method"] == "l1"   # recovered from the source review's blind_mapping
    assert pair["reversal"]["mapped_method"] == "multiscale"
    assert pair["derived"]["side_following"] is True
    # The source review contributes provenance to the report.
    assert report["inputs"]["source_reviews"][0]["run_id"] == "legacy-run"


def test_legacy_recovery_via_parent_directory_name(tmp_path):
    source = _write_source_review(
        tmp_path / "parentdir" / "review.json", output_dir="other", run_id="legacy-run",
        pairs=[(0, "pair_00_ab.mp4", _mapping("multiscale", "l1"))])
    legacy = _write_legacy_result(tmp_path / "legacy.json", [("parentdir", "pair_00_ab.mp4", "A", "high")])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="legacy-run", source_sha256="d" * 64, pair_index=0, raw="A",
                            mapping=_mapping("l1", "multiscale"))])
    pair = _pair(A.audit([legacy], [reversal], [source]), 0)
    assert pair["original"]["mapped_method"] == "multiscale"


def test_legacy_window_that_matches_nothing_fails(tmp_path):
    source = _write_source_review(
        tmp_path / "src" / "review.json", output_dir="runwin", run_id="legacy-run",
        pairs=[(0, "pair_00_ab.mp4", _mapping("multiscale", "l1"))])
    legacy = _write_legacy_result(tmp_path / "legacy.json", [("otherwin", "pair_00_ab.mp4", "A", "high")])
    reversal = _write_standard_result(
        tmp_path / "rev.json", [_standard_judgment(run_id="legacy-run", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([legacy], [reversal], [source])
    assert exc.value.code == "legacy_window_unmatched"


def test_legacy_clip_that_matches_no_pair_fails(tmp_path):
    source = _write_source_review(
        tmp_path / "src" / "review.json", output_dir="runwin", run_id="legacy-run",
        pairs=[(0, "pair_00_ab.mp4", _mapping("multiscale", "l1"))])
    legacy = _write_legacy_result(tmp_path / "legacy.json", [("runwin", "pair_09_ab.mp4", "A", "high")])
    reversal = _write_standard_result(
        tmp_path / "rev.json", [_standard_judgment(run_id="legacy-run", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([legacy], [reversal], [source])
    assert exc.value.code == "legacy_clip_unmatched"


def test_legacy_ambiguous_window_fails(tmp_path):
    source_a = _write_source_review(
        tmp_path / "a" / "review.json", output_dir="dup", run_id="r", pairs=[(0, "pair_00_ab.mp4",
                                                                             _mapping("multiscale", "l1"))])
    source_b = _write_source_review(
        tmp_path / "b" / "review.json", output_dir="dup", run_id="r", pairs=[(0, "pair_00_ab.mp4",
                                                                             _mapping("multiscale", "l1"))])
    legacy = _write_legacy_result(tmp_path / "legacy.json", [("dup", "pair_00_ab.mp4", "A", "high")])
    reversal = _write_standard_result(
        tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([legacy], [reversal], [source_a, source_b])
    assert exc.value.code == "legacy_window_ambiguous"


# --------------------------------------------------------------------------------------
# strict duplicate / missing / unmatched behavior
# --------------------------------------------------------------------------------------

def test_duplicate_original_identity_fails(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json",
        [_standard_judgment(run_id="r", pair_index=0), _standard_judgment(run_id="r", pair_index=0)])
    reversal = _write_standard_result(
        tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "duplicate_key"


def test_duplicate_reversal_identity_fails(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=0)])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", pair_index=0), _standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "duplicate_key"


def test_unmatched_keys_fail_by_default_and_are_recorded_when_allowed(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=0)])
    reversal = _write_standard_result(
        tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=1)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "unmatched_key"
    assert "original key" in str(exc.value) and "reversal key" in str(exc.value)

    report = A.audit([original], [reversal], [], allow_unmatched=True)
    assert report["coverage"]["complete_pairs"] == 0
    assert report["coverage"]["unmatched_original_count"] == 1
    assert report["coverage"]["unmatched_reversal_count"] == 1
    assert report["coverage"]["unmatched_originals"][0]["pair_index"] == 0
    assert report["coverage"]["unmatched_reversals"][0]["pair_index"] == 1


def test_missing_mapping_fails(tmp_path):
    judgment = _standard_judgment(run_id="r", pair_index=0)
    del judgment["blind_mapping"]
    original = _write_standard_result(tmp_path / "orig.json", [judgment])
    reversal = _write_standard_result(tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "missing_mapping"


def test_mapped_preference_mismatch_fails(tmp_path):
    judgment = _standard_judgment(run_id="r", pair_index=0, raw="A", mapped="l1", mapping=_mapping("multiscale",
                                                                                                   "l1"))
    original = _write_standard_result(tmp_path / "orig.json", [judgment])
    reversal = _write_standard_result(tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "mapped_preference_mismatch"


def test_invalid_preference_and_confidence_fail(tmp_path):
    bad_preference = _standard_judgment(run_id="r", pair_index=0)
    bad_preference["raw_side_preference"] = "Z"
    original = _write_standard_result(tmp_path / "orig.json", [bad_preference])
    reversal = _write_standard_result(tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "preference_invalid"

    bad_confidence = _standard_judgment(run_id="r", pair_index=0)
    bad_confidence["confidence"] = "certain"
    original = _write_standard_result(tmp_path / "orig.json", [bad_confidence])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "confidence_invalid"


def test_invalid_blind_mapping_fails(tmp_path):
    judgment = _standard_judgment(run_id="r", pair_index=0)
    judgment["blind_mapping"] = _mapping("multiscale", "multiscale")
    original = _write_standard_result(tmp_path / "orig.json", [judgment])
    reversal = _write_standard_result(tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [])
    assert exc.value.code == "blind_mapping_invalid"


def test_unknown_schema_fails(tmp_path):
    path = tmp_path / "orig.json"
    path.write_text(json.dumps({"schema": "other/1", "judgments": []}), encoding="utf-8")
    reversal = _write_standard_result(tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([path], [reversal], [])
    assert exc.value.code == "schema_unsupported"


def test_legacy_shaped_reversal_is_rejected(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=0)])
    legacy = _write_legacy_result(tmp_path / "rev.json", [("runwin", "pair_00_ab.mp4", "A", "high")])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [legacy], [])
    assert exc.value.code == "schema_unsupported"


def test_missing_and_unreadable_inputs_fail(tmp_path):
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([tmp_path / "absent.json"], [tmp_path / "rev.json"], [])
    assert exc.value.code == "input_missing"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([bad], [tmp_path / "rev.json"], [])
    assert exc.value.code == "input_unreadable"


def test_out_collision_with_input_is_refused(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=0)])
    reversal = _write_standard_result(tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([original], [reversal], [], out_path=original)
    assert exc.value.code == "input_collision"


# --------------------------------------------------------------------------------------
# report content, determinism and atomic output
# --------------------------------------------------------------------------------------

def _sample_report(tmp_path):
    original = _write_standard_result(
        tmp_path / "orig.json",
        [_standard_judgment(run_id="r", start=0.0, duration=1.0, pair_index=3, raw="B"),
         _standard_judgment(run_id="r", start=60.0, duration=2.0, pair_index=0, raw="tie")])
    reversal = _write_standard_result(
        tmp_path / "rev.json",
        [_standard_judgment(run_id="r", start=0.0, duration=1.0, pair_index=3, raw="B",
                            mapping=_mapping("l1", "multiscale")),
         _standard_judgment(run_id="r", start=60.0, duration=2.0, pair_index=0, raw="tie",
                            mapping=_mapping("l1", "multiscale"))])
    return A.audit([original], [reversal], []), original, reversal


def test_report_has_no_winner_ranking_score_or_significance(tmp_path):
    report, _, _ = _sample_report(tmp_path)
    forbidden = {"winner", "ranking", "rank", "score", "significance", "better", "aggregate"}
    assert not (set(report) & forbidden)
    assert "observational evidence only" in report["scope_note"].lower()
    assert "no winner" in report["no_quality_score_note"].lower()
    # The word "winner" must only ever appear inside a negative disclaimer.
    encoded = dumps_json(report)
    assert '"winner"' not in encoded


def test_scope_note_does_not_claim_translation_only_clips(tmp_path):
    report, _, _ = _sample_report(tmp_path)
    note = report["scope_note"].lower()
    assert "translation-only" not in note
    assert "single model" in note
    assert "short" in note
    assert "low-resolution" in note
    assert "no causal, significance or product-quality conclusion follows" in note


def test_report_is_byte_stable_and_self_hashed(tmp_path):
    first, original, reversal = _sample_report(tmp_path)
    second = A.audit([original], [reversal], [])
    assert dumps_json(first) == dumps_json(second)
    payload = {key: value for key, value in first.items() if key != "report_sha256"}
    import hashlib
    assert first["report_sha256"] == hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()


def test_pairs_are_sorted_by_stable_identity(tmp_path):
    report, _, _ = _sample_report(tmp_path)
    keys = [(pair["identity"]["run_id"], pair["identity"]["source_window"]["start_s"],
             pair["identity"]["pair_index"]) for pair in report["pairs"]]
    assert keys == sorted(keys)
    assert [pair["identity"]["pair_index"] for pair in report["pairs"]] == [3, 0]  # 0.0 s start before 60.0 s


def test_input_provenance_records_hashes(tmp_path):
    report, original, reversal = _sample_report(tmp_path)
    original_input = report["inputs"]["original_results"][0]
    reversal_input = report["inputs"]["reversal_results"][0]
    assert original_input["path"] == str(original)
    assert reversal_input["path"] == str(reversal)
    assert len(original_input["sha256"]) == 64 and len(reversal_input["sha256"]) == 64
    assert original_input["kind"] == "standard" and original_input["judgments"] == 2


def test_main_writes_atomically_and_rejects_unmatched(tmp_path, capsys):
    _report, original_path, reversal_path = _sample_report(tmp_path)
    out = tmp_path / "audit.json"
    assert A.main(["--original-result", str(original_path), "--reversal-result", str(reversal_path),
                   "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == A.SCHEMA and written["ok"] is True
    assert list(tmp_path.glob("*.tmp")) == []
    assert "pairs=2" in capsys.readouterr().out

    # Strict mode refuses the real-file shape where reversals cover only a subset of originals.
    only_original = _write_standard_result(
        tmp_path / "only.json", [_standard_judgment(run_id="solo", pair_index=0)])
    assert A.main(["--original-result", str(only_original), "--reversal-result", str(reversal_path),
                   "--out", str(tmp_path / "nope.json")]) == 2
    err = capsys.readouterr().err
    assert "unmatched_key" in err
    assert not (tmp_path / "nope.json").exists()


def test_cli_requires_out_and_an_input_source():
    with pytest.raises(SystemExit):
        A.build_parser().parse_args(["--original-result", "a.json", "--reversal-result", "b.json"])
    # With --paired-result available, original/reversal are no longer parse-time required; the audit still
    # refuses a run that supplies no input at all.
    args = A.build_parser().parse_args(["--out", "x.json"])
    assert args.original_results == [] and args.paired_results == []
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [], out_path="x.json")
    assert exc.value.code == "no_inputs"


def test_allow_unmatched_cli_records_coverage(tmp_path, capsys):
    original = _write_standard_result(
        tmp_path / "orig.json", [_standard_judgment(run_id="r", pair_index=0)])
    reversal = _write_standard_result(
        tmp_path / "rev.json", [_standard_judgment(run_id="r", pair_index=1)])
    out = tmp_path / "audit.json"
    assert A.main(["--original-result", str(original), "--reversal-result", str(reversal),
                   "--out", str(out), "--allow-unmatched"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["coverage"]["complete_pairs"] == 0
    assert report["coverage"]["unmatched_original_count"] == 1


# --------------------------------------------------------------------------------------
# paired results: one file carrying the original and its position-reversed control
# --------------------------------------------------------------------------------------

def _paired_judgment(*, group, role, **kwargs):
    judgment = _standard_judgment(**kwargs)
    judgment["position_reversal"] = {"schema": G.POSITION_REVERSAL_SCHEMA, "group": group, "role": role}
    return judgment


def _write_paired_result(path, judgments):
    return _write_standard_result(path, judgments)


def test_paired_result_requires_role_metadata(tmp_path):
    path = _write_paired_result(tmp_path / "paired.json", [_standard_judgment(run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [path])
    assert exc.value.code == "paired_role_missing"


def test_paired_result_rejects_invalid_role(tmp_path):
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="g", role="control", run_id="r", pair_index=0),
         _paired_judgment(group="g", role="position_reversed", run_id="r", pair_index=1,
                          raw="B", mapping=_mapping("stabilized", "raw"))])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [path])
    assert exc.value.code == "position_reversal_invalid"


def test_paired_result_rejects_duplicate_role(tmp_path):
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="g", role="original", run_id="r", pair_index=0),
         _paired_judgment(group="g", role="original", run_id="r", pair_index=1)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [path])
    assert exc.value.code == "paired_duplicate_role"


def test_paired_result_rejects_missing_role(tmp_path):
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="g", role="original", run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [path])
    assert exc.value.code == "paired_missing_role"


def test_paired_result_rejects_group_split(tmp_path):
    # An original and a reversal under different groups leave both groups one-sided.
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="g1", role="original", run_id="r", pair_index=0),
         _paired_judgment(group="g2", role="position_reversed", run_id="r", pair_index=1,
                          raw="B", mapping=_mapping("stabilized", "raw"))])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [path])
    assert exc.value.code == "paired_missing_role"


def test_paired_result_cannot_self_pair(tmp_path):
    # Splitting by role means a single original record cannot join itself; strict mode reports the unmatched
    # reversal side rather than fabricating a pair.
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="g", role="original", run_id="r", pair_index=0)])
    with pytest.raises(A.PositionBiasAuditError) as exc:
        A.audit([], [], [], [path])
    assert exc.value.code == "paired_missing_role"


def test_paired_result_audit_maps_raw_in_both_orders(tmp_path):
    # Build a paired result outright (no Gemini call is made by the audit) with the reviewer choosing the
    # raw panel behind A in the original and behind B in the reversal: two screen-side preferences that both
    # map to raw.
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="grp", role="original", run_id="r", source_sha256="a" * 64,
                          pair_index=0, raw="A", mapping=_mapping("raw", "stabilized")),
         _paired_judgment(group="grp", role="position_reversed", run_id="r", source_sha256="a" * 64,
                          pair_index=1, raw="B", mapping=_mapping("stabilized", "raw"))])
    report = A.audit([], [], [], [path])
    assert report["coverage"]["complete_pairs"] == 1
    assert report["coverage"]["paired_original_records"] == 1
    assert report["coverage"]["paired_reversal_records"] == 1
    pair = report["pairs"][0]
    assert pair["identity"]["group"] == "grp"
    assert pair["identity"].get("pair_index") is None
    assert pair["original"]["raw_side"] == "A" and pair["original"]["mapped_method"] == "raw"
    assert pair["reversal"]["raw_side"] == "B" and pair["reversal"]["mapped_method"] == "raw"
    assert pair["derived"]["method_agreement"] is True
    assert pair["derived"]["opposite_physical_sides"] is True
    assert pair["derived"]["side_following"] is False
    assert pair["derived"]["mapping_reversed"] is True
    aggregates = report["aggregates"]
    assert aggregates["original_mapped_method_counts"]["raw"] == 1
    assert aggregates["reversal_mapped_method_counts"]["raw"] == 1
    # raw/stabilized are part of the mapped-method universe now.
    assert set(aggregates["original_mapped_method_counts"]) == set(A.METHODS)
    assert set(aggregates["reversal_mapped_method_counts"]) == set(A.METHODS)


def test_paired_result_cli_round_trip(tmp_path, capsys):
    path = _write_paired_result(
        tmp_path / "paired.json",
        [_paired_judgment(group="grp", role="original", run_id="r", pair_index=0,
                          raw="A", mapping=_mapping("raw", "stabilized")),
         _paired_judgment(group="grp", role="position_reversed", run_id="r", pair_index=1,
                          raw="B", mapping=_mapping("stabilized", "raw"))])
    out = tmp_path / "audit.json"
    assert A.main(["--paired-result", str(path), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["coverage"]["complete_pairs"] == 1
    assert "pairs=1" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# end-to-end: paired review JSON -> Gemini tasks/judgments (fake API) -> audit
# --------------------------------------------------------------------------------------

class _FakeVertexResponse:
    def __init__(self, preference):
        judgment = {"preference": preference, "confidence": "medium",
                    "visible_evidence": "visible", "artifacts": "none", "limitations": "short clip"}
        self._payload = json.dumps({"candidates": [{"content": {"parts": [
            {"text": json.dumps(judgment)}]}}]}).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _RecordingOpener:
    """Return one fake Vertex response per call and record every request body."""

    def __init__(self, preferences):
        self.preferences = list(preferences)
        self.bodies = []

    def __call__(self, request, timeout=None):
        self.bodies.append(json.loads(request.data.decode("utf-8")))
        return _FakeVertexResponse(self.preferences.pop(0))


def _write_paired_review(root, *, group="grp"):
    """A real_pair_review-style review.json with both roles and their position_reversal metadata."""
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    original = "raw_vs_stabilized.mp4"
    reversed_clip = "raw_vs_stabilized_posrev.mp4"
    (out_dir / original).write_bytes(b"original clip bytes")
    (out_dir / reversed_clip).write_bytes(b"reversed clip bytes")
    document = {
        "schema": G.INPUT_SCHEMA,
        "ok": True,
        "output": {"dir": "review", "clips": [original, reversed_clip]},
        "provenance": {"run": "run-pair", "run_id": "run-pair", "case": "case-pair",
                       "source_sha256": "e" * 64,
                       "source_window": {"start_s": 1.0, "duration_s": 2.0}},
        "pairs": [
            {"pair_index": 0, "clip": original,
             "blind_mapping": _mapping("raw", "stabilized"),
             "position_reversal": {"schema": G.POSITION_REVERSAL_SCHEMA, "group": group,
                                   "role": "original"}},
            {"pair_index": 1, "clip": reversed_clip,
             "blind_mapping": _mapping("stabilized", "raw"),
             "position_reversal": {"schema": G.POSITION_REVERSAL_SCHEMA, "group": group,
                                   "role": "position_reversed"}},
        ],
    }
    review_path = out_dir / "review.json"
    review_path.write_text(json.dumps(document), encoding="utf-8")
    return review_path, document


def test_end_to_end_paired_review_raw_in_both_orders(tmp_path):
    review_path, _ = _write_paired_review(tmp_path / "pair")

    # collect_tasks carries the per-item metadata without placing it in any Vertex request.
    tasks = G.collect_tasks([review_path])
    assert [task["position_reversal"]["role"] for task in tasks] == ["original", "position_reversed"]
    assert {task["position_reversal"]["group"] for task in tasks} == {"grp"}

    opener = _RecordingOpener(["A", "B"])  # original A -> raw; reversal B -> raw
    out = tmp_path / "paired_result.json"
    document = G.run_review_set([review_path], out, token="tok", opener=opener, sleep=lambda _s: None)

    # The request bodies carry only the clip bytes and the fixed brief: no role/group/mapping metadata.
    for body in opener.bodies:
        text = body["contents"][0]["parts"][1]["text"]
        assert text == G.REVIEW_BRIEF
        assert "position_reversal" not in text
        assert "grp" not in json.dumps(body)
        assert "raw" not in json.dumps(body) and "stabilized" not in json.dumps(body)

    # The recorded judgments preserve the roles; mapping stays local.
    assert [record["position_reversal"]["role"] for record in document["judgments"]] == [
        "original", "position_reversed"]
    assert [record["mapped_preference"] for record in document["judgments"]] == ["raw", "raw"]

    report = A.audit([], [], [], [out])
    assert report["coverage"]["complete_pairs"] == 1
    pair = report["pairs"][0]
    assert pair["identity"]["group"] == "grp"
    assert pair["original"]["raw_side"] == "A" and pair["original"]["mapped_method"] == "raw"
    assert pair["reversal"]["raw_side"] == "B" and pair["reversal"]["mapped_method"] == "raw"
    assert pair["derived"] == {
        "both_non_tie": True,
        "side_following": False,
        "method_agreement": True,
        "opposite_physical_sides": True,
        "position_tracks_while_method_changes": False,
        "mapping_reversed": True,
    }
    aggregates = report["aggregates"]
    assert aggregates["original_mapped_method_counts"]["raw"] == 1
    assert aggregates["reversal_mapped_method_counts"]["raw"] == 1
