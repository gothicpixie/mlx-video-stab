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


def test_cli_requires_original_reversal_and_out():
    with pytest.raises(SystemExit):
        A.build_parser().parse_args(["--out", "x.json"])
    with pytest.raises(SystemExit):
        A.build_parser().parse_args(["--original-result", "a.json", "--out", "x.json"])
    with pytest.raises(SystemExit):
        A.build_parser().parse_args(["--original-result", "a.json", "--reversal-result", "b.json"])


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
