"""Focused stdlib/pytest tests for the research-only Vertex Gemini blind-review CLI.

Everything is mocked: no Vertex call, no gcloud, no MLX, no FFmpeg. Coverage: A/B/tie mapping through the
hidden ``blind_mapping``, that the request carries only the clip bytes plus the fixed brief, resume-cache
reuse and invalidation on clip digest / model / prompt / schema change, transient HTTP retry with bounded
backoff, sanitized permanent errors, review/clip validation refusals, path-traversal/absolute-path and
output-collision refusals, file-layout resolution and atomic result-JSON safety.
"""
import base64
import io
import json
import urllib.error

import pytest

from bench import gemini_blind_review as G


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _SequenceOpener:
    """Open returns/raises the queued outcomes in order; fails loudly if called too many times."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("opener called more times than expected")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)


def _vertex_judgment(**overrides):
    judgment = {
        "preference": "A",
        "confidence": "medium",
        "visible_evidence": "the lights track more smoothly",
        "artifacts": "none beyond compression noise",
        "limitations": "short, dark, low-resolution clip",
    }
    judgment.update(overrides)
    return judgment


def _vertex_response(judgment):
    return json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(judgment)}]}}]}).encode("utf-8")


def _write_review(root, *, dir_name="review", size=1024, left="multiscale", right="l1",
                  labels=("A", "B"), pairs=1, schema=G.INPUT_SCHEMA, ok=True, nested=False, provenance=None):
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_list = []
    for index in range(pairs):
        name = f"pair_{index:02d}_ab.mp4"
        (out_dir / name).write_bytes(bytes([index % 256]) * size)
        pairs_list.append({
            "pair_index": index,
            "clip": name,
            "blind_mapping": {"left_panel": left, "right_panel": right,
                              "video_labels": {"left": labels[0], "right": labels[1]}},
        })
    document = {
        "schema": schema,
        "ok": ok,
        "output": {"dir": dir_name, "clips": [pair["clip"] for pair in pairs_list]},
        "pairs": pairs_list,
        "provenance": provenance or {"run": "run-1", "run_id": "run-1", "case": "case-1",
                                     "source_sha256": "a" * 64,
                                     "source_window": {"start_s": 0.0, "duration_s": 2.0}},
    }
    review_path = (root / "review.json") if nested else (out_dir / "review.json")
    review_path.write_text(json.dumps(document), encoding="utf-8")
    return review_path, document


# --------------------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------------------

def test_map_preference_resolves_a_b_and_tie():
    mapping = {"left_panel": "multiscale", "right_panel": "l1",
               "video_labels": {"left": "A", "right": "B"}}
    assert G.map_preference(mapping, "A") == "multiscale"
    assert G.map_preference(mapping, "B") == "l1"
    assert G.map_preference(mapping, "tie") == "tie"
    swapped = {"left_panel": "l1", "right_panel": "multiscale",
               "video_labels": {"left": "A", "right": "B"}}
    assert G.map_preference(swapped, "A") == "l1"
    assert G.map_preference(swapped, "B") == "multiscale"
    assert G.map_preference(swapped, "tie") == "tie"


@pytest.mark.parametrize("bad", [
    None,
    {},
    {"left_panel": "multiscale", "right_panel": "multiscale",
     "video_labels": {"left": "A", "right": "B"}},
    {"left_panel": "multiscale", "right_panel": "l1", "video_labels": {"left": "B", "right": "A"}},
    {"left_panel": "multiscale", "right_panel": "l1"},
])
def test_validate_blind_mapping_rejects_bad_shapes(bad):
    with pytest.raises(G.GeminiReviewError) as exc:
        G.validate_blind_mapping(bad)
    assert exc.value.code == "blind_mapping_invalid"


def test_map_preference_rejects_unknown_side():
    mapping = {"left_panel": "multiscale", "right_panel": "l1",
               "video_labels": {"left": "A", "right": "B"}}
    with pytest.raises(G.GeminiReviewError) as exc:
        G.map_preference(mapping, "C")
    assert exc.value.code == "preference_invalid"


# --------------------------------------------------------------------------------------
# request construction carries only bytes + the fixed brief
# --------------------------------------------------------------------------------------

def test_build_request_is_only_video_bytes_and_fixed_brief():
    payload = b"\x00\x01\x02video-bytes"
    body = G.build_request(payload)
    parts = body["contents"][0]["parts"]
    assert len(parts) == 2
    assert set(parts[0]) == {"inlineData"}
    inline = parts[0]["inlineData"]
    assert inline["mimeType"] == "video/mp4"
    assert base64.b64decode(inline["data"]) == payload
    assert parts[1] == {"text": G.REVIEW_BRIEF}
    assert set(body["generationConfig"]) == {"responseMimeType", "responseSchema", "temperature"}
    for token in ["pair_", ".mp4", "multiscale", "source_sha256", "run_id", "source_window", "blind_mapping",
                  "left_panel", "right_panel", "retention", "candidate"]:
        assert token not in G.REVIEW_BRIEF


def test_request_never_carries_hidden_metadata(tmp_path):
    review_path, document = _write_review(tmp_path / "r", pairs=1)
    captured = {}

    class _Recorder:
        def __call__(self, request, timeout=None):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(_vertex_response(_vertex_judgment()))

    G.run_review_set([review_path], tmp_path / "out.json", token="tok", opener=_Recorder(),
                     sleep=lambda _s: None)
    body = captured["body"]
    parts = body["contents"][0]["parts"]
    assert parts[1]["text"] == G.REVIEW_BRIEF
    brief = parts[1]["text"]
    for token in ["run-1", document["provenance"]["source_sha256"], "pair_00_ab.mp4", "blind_mapping",
                  "multiscale", "left_panel", "source_window", "case-1"]:
        assert token not in brief
    # The single non-brief part is the inline video; there is no accompanying filename/metadata part.
    assert len(parts) == 2
    assert set(parts[0]) == {"inlineData"}


# --------------------------------------------------------------------------------------
# review.json validation
# --------------------------------------------------------------------------------------

def test_load_review_missing_unreadable_schema_and_ok(tmp_path):
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(tmp_path / "absent.json")
    assert exc.value.code == "review_missing"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(bad)
    assert exc.value.code == "review_unreadable"

    review_path, _ = _write_review(tmp_path / "schema", schema="other/1")
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(review_path)
    assert exc.value.code == "review_schema_unsupported"

    review_path, _ = _write_review(tmp_path / "notok", ok=False)
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(review_path)
    assert exc.value.code == "review_not_ok"


def test_load_review_rejects_bad_hidden_mapping(tmp_path):
    review_path, _ = _write_review(tmp_path / "mapping", left="multiscale", right="multiscale")
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(review_path)
    assert exc.value.code == "blind_mapping_invalid"

    review_path, _ = _write_review(tmp_path / "labels", labels=("B", "A"))
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(review_path)
    assert exc.value.code == "blind_mapping_invalid"


@pytest.mark.parametrize("bad_clip", [
    "../evil.mp4",
    "../../etc/evil.mp4",
    "/tmp/evil.mp4",
    "sub/evil.mp4",
    "sub\\evil.mp4",
    "evil.mov",
    "noext",
])
def test_load_review_rejects_clip_traversal_absolute_and_non_mp4(tmp_path, bad_clip):
    review_path, document = _write_review(tmp_path / "clip")
    document["pairs"][0]["clip"] = bad_clip
    review_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(review_path)
    assert exc.value.code == G.UNSAFE_PATH


@pytest.mark.parametrize("bad_dir", ["/abs", "a/b", "a\\b", "..", ".", "C:\\x", "../sibling"])
def test_load_review_rejects_bad_output_dir(tmp_path, bad_dir):
    review_path, document = _write_review(tmp_path / "dir")
    document["output"]["dir"] = bad_dir
    review_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(G.GeminiReviewError) as exc:
        G.load_review(review_path)
    assert exc.value.code == G.UNSAFE_PATH


@pytest.mark.parametrize("plain", ["pair_00_ab.mp4", "pair.mp4", "a.b.mp4", "PAIR.MP4"])
def test_load_review_accepts_plain_mp4_basenames(tmp_path, plain):
    review_path, document = _write_review(tmp_path / "plain")
    document["pairs"][0]["clip"] = plain
    review_path.write_text(json.dumps(document), encoding="utf-8")
    assert G.load_review(review_path)["pairs"][0]["clip"] == plain


def test_load_review_accepts_documented_same_dir_and_copied_layouts(tmp_path):
    sibling, _ = _write_review(tmp_path / "sibling", dir_name="review")
    G.load_review(sibling)
    assert G.collect_tasks([sibling])[0]["clip_path"] == sibling.parent / "pair_00_ab.mp4"

    nested, _ = _write_review(tmp_path / "nested", dir_name="review", nested=True)
    G.load_review(nested)
    assert G.collect_tasks([nested])[0]["clip_path"] == nested.parent / "review" / "pair_00_ab.mp4"


def test_resolve_clip_path_handles_sibling_and_nested_layouts(tmp_path):
    sibling, _ = _write_review(tmp_path / "sibling", dir_name="review")
    assert G.resolve_clip_path(sibling, json.loads(sibling.read_text()), "pair_00_ab.mp4") == \
        sibling.parent / "pair_00_ab.mp4"
    nested, _ = _write_review(tmp_path / "nested", dir_name="review", nested=True)
    assert G.resolve_clip_path(nested, json.loads(nested.read_text()), "pair_00_ab.mp4") == \
        nested.parent / "review" / "pair_00_ab.mp4"


def test_resolve_clip_path_rejects_symlink_escaping_review_directory(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    outside_clip = external / "outside.mp4"
    outside_clip.write_bytes(b"external clip bytes")

    review_path, _ = _write_review(tmp_path / "review", dir_name="review")
    clip_link = review_path.parent / "pair_00_ab.mp4"
    clip_link.unlink()
    try:
        clip_link.symlink_to(outside_clip)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(G.GeminiReviewError) as exc:
        G.resolve_clip_path(review_path, json.loads(review_path.read_text()), "pair_00_ab.mp4")
    assert exc.value.code == G.UNSAFE_PATH

    # The same escaping clip is refused by the validating task collector, before any bytes are read.
    with pytest.raises(G.GeminiReviewError) as exc:
        G.collect_tasks([review_path])
    assert exc.value.code == G.UNSAFE_PATH


def test_collect_tasks_records_provenance_and_digest(tmp_path):
    review_path, _ = _write_review(tmp_path / "r", pairs=2)
    tasks = G.collect_tasks([review_path])
    assert len(tasks) == 2
    assert [task["pair_index"] for task in tasks] == [0, 1]
    for task in tasks:
        assert len(task["clip_sha256"]) == 64
        assert task["run"] == "run-1" and task["run_id"] == "run-1"
        assert task["source_window"] == {"start_s": 0.0, "duration_s": 2.0}
        assert task["blind_mapping"]["left_panel"] == "multiscale"


def test_collect_tasks_clip_validation_refusals(tmp_path):
    review_path, _ = _write_review(tmp_path / "missing")
    (review_path.parent / "pair_00_ab.mp4").unlink()
    with pytest.raises(G.GeminiReviewError) as exc:
        G.collect_tasks([review_path])
    assert exc.value.code == "clip_missing"

    review_path, _ = _write_review(tmp_path / "empty", size=0)
    with pytest.raises(G.GeminiReviewError) as exc:
        G.collect_tasks([review_path])
    assert exc.value.code == "clip_empty"

    review_path, _ = _write_review(tmp_path / "big", size=64)
    with pytest.raises(G.GeminiReviewError) as exc:
        G.collect_tasks([review_path], limit_bytes=32)
    assert exc.value.code == "clip_too_large"


# --------------------------------------------------------------------------------------
# Vertex transport: retries and sanitized errors
# --------------------------------------------------------------------------------------

def _http_error(status, body=b"SECRET-RESPONSE-BODY"):
    return urllib.error.HTTPError("https://example.invalid", status, "err", {}, io.BytesIO(body))


def test_call_vertex_retries_transient_then_recovers():
    opener = _SequenceOpener([_http_error(429), _http_error(503), _http_error(500),
                              _vertex_response(_vertex_judgment())])
    sleeps = []
    response = G.call_vertex(G.build_request(b"x"), token="tok", opener=opener, sleep=sleeps.append)
    assert opener.calls == 4
    assert sleeps == [G.BACKOFF_BASE_S, G.BACKOFF_BASE_S * 2, G.BACKOFF_BASE_S * 4]
    assert response["candidates"][0]["content"]["parts"][0]["text"]


def test_call_vertex_permanent_error_is_not_retried_and_hides_body():
    opener = _SequenceOpener([_http_error(400)])
    sleeps = []
    with pytest.raises(G.GeminiReviewError) as exc:
        G.call_vertex(G.build_request(b"x"), token="tok", opener=opener, sleep=sleeps.append)
    assert exc.value.code == "http_400"
    assert "SECRET-RESPONSE-BODY" not in str(exc.value)
    assert "tok" not in str(exc.value)
    assert opener.calls == 1 and sleeps == []


def test_call_vertex_retries_network_then_gives_up():
    opener = _SequenceOpener([urllib.error.URLError("boom"), urllib.error.URLError("boom"),
                              urllib.error.URLError("boom")])
    sleeps = []
    with pytest.raises(G.GeminiReviewError) as exc:
        G.call_vertex(G.build_request(b"x"), token="tok", opener=opener, sleep=sleeps.append, attempts=3)
    assert exc.value.code == "network_error"
    assert len(sleeps) == 2
    assert opener.calls == 3


def test_parse_response_validates_and_sanitizes():
    good = G.parse_response({"candidates": [{"content": {"parts": [
        {"text": json.dumps(_vertex_judgment())}]}}]})
    assert good["preference"] == "A" and good["confidence"] == "medium"

    with pytest.raises(G.GeminiReviewError) as exc:
        G.parse_response({"candidates": []})
    assert exc.value.code == "response_no_candidates"

    with pytest.raises(G.GeminiReviewError) as exc:
        G.parse_response({"candidates": [{"content": {"parts": [{"text": "not json SECRET"}]}}]})
    assert exc.value.code == "response_unparseable"
    assert "SECRET" not in str(exc.value)

    with pytest.raises(G.GeminiReviewError) as exc:
        G.parse_response({"candidates": [{"content": {"parts": [
            {"text": json.dumps(_vertex_judgment(preference="Z"))}]}}]})
    assert exc.value.code == "response_invalid"


def test_access_token_returns_and_fails_cleanly():
    class _Proc:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode

    assert G.access_token(runner=lambda *a, **k: _Proc("  tok-value \n")) == "tok-value"
    with pytest.raises(G.GeminiReviewError) as exc:
        G.access_token(runner=lambda *a, **k: _Proc("", returncode=1))
    assert exc.value.code == "auth_failed"


# --------------------------------------------------------------------------------------
# end-to-end run, mapping in the record, resume
# --------------------------------------------------------------------------------------

def test_run_review_set_maps_and_records(tmp_path):
    review_path, _ = _write_review(tmp_path / "r", pairs=2)
    opener = _SequenceOpener([_vertex_response(_vertex_judgment(preference="B"))] * 2)
    out = tmp_path / "out.json"
    document = G.run_review_set([review_path], out, token="tok", opener=opener, sleep=lambda _s: None)

    assert document["complete"] is True
    assert len(document["judgments"]) == 2
    for judgment in document["judgments"]:
        assert judgment["raw_side_preference"] == "B"
        assert judgment["mapped_preference"] == "l1"          # left=multiscale, right=l1
        assert len(judgment["clip_sha256"]) == 64
        assert judgment["run_id"] == "run-1"
        assert judgment["model"] == G.MODEL
        assert judgment["prompt_version"] == G.PROMPT_VERSION
        assert judgment["response_schema_version"] == G.RESPONSE_SCHEMA_VERSION
        assert judgment["timestamp"]
    saved = out.read_text(encoding="utf-8")
    assert json.loads(saved)["complete"] is True
    assert "tok" not in saved
    assert list(tmp_path.glob("*.tmp")) == []


def test_run_review_set_resume_reuses_then_invalidates_on_new_clip_bytes(tmp_path):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    out = tmp_path / "out.json"
    first = _SequenceOpener([_vertex_response(_vertex_judgment(preference="A"))])
    G.run_review_set([review_path], out, token="tok", opener=first, sleep=lambda _s: None)
    assert first.calls == 1
    first_record = json.loads(out.read_text(encoding="utf-8"))["judgments"][0]

    reused = _SequenceOpener([])  # would fail if the cache were not hit
    document = G.run_review_set([review_path], out, token="tok", opener=reused, sleep=lambda _s: None)
    assert reused.calls == 0
    assert document["judgments"][0]["timestamp"] == first_record["timestamp"]
    assert document["judgments"][0]["clip_sha256"] == first_record["clip_sha256"]

    # Replacing the clip bytes changes the digest, so the cached judgment must not be reused.
    (review_path.parent / "pair_00_ab.mp4").write_bytes(b"different clip bytes" * 64)
    refreshed = _SequenceOpener([_vertex_response(_vertex_judgment(preference="B"))])
    document = G.run_review_set([review_path], out, token="tok", opener=refreshed, sleep=lambda _s: None)
    assert refreshed.calls == 1
    assert document["judgments"][0]["clip_sha256"] != first_record["clip_sha256"]
    assert document["judgments"][0]["raw_side_preference"] == "B"


def test_run_review_set_resume_invalidates_on_prompt_schema_and_model(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    out = tmp_path / "out.json"
    opener = _SequenceOpener([_vertex_response(_vertex_judgment())])
    G.run_review_set([review_path], out, token="tok", opener=opener, sleep=lambda _s: None)
    assert opener.calls == 1

    monkeypatch.setattr(G, "PROMPT_VERSION", "mlxstab.gemini_blind_review.prompt/2")
    bumped = _SequenceOpener([_vertex_response(_vertex_judgment())])
    G.run_review_set([review_path], out, token="tok", opener=bumped, sleep=lambda _s: None)
    assert bumped.calls == 1

    monkeypatch.setattr(G, "RESPONSE_SCHEMA_VERSION", "mlxstab.gemini_blind_review.response/2")
    bumped = _SequenceOpener([_vertex_response(_vertex_judgment())])
    G.run_review_set([review_path], out, token="tok", opener=bumped, sleep=lambda _s: None)
    assert bumped.calls == 1

    monkeypatch.undo()
    model = _SequenceOpener([_vertex_response(_vertex_judgment())])
    G.run_review_set([review_path], out, token="tok", opener=model, sleep=lambda _s: None,
                     model="gemini-other")
    assert model.calls == 1
    assert json.loads(out.read_text())["judgments"][0]["model"] == "gemini-other"


def test_run_review_set_no_resume_repeats_requests(tmp_path):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    out = tmp_path / "out.json"
    opener = _SequenceOpener([_vertex_response(_vertex_judgment())])
    G.run_review_set([review_path], out, token="tok", opener=opener, sleep=lambda _s: None)
    again = _SequenceOpener([_vertex_response(_vertex_judgment())])
    G.run_review_set([review_path], out, token="tok", opener=again, sleep=lambda _s: None, resume=False)
    assert again.calls == 1


def test_run_review_set_rejects_bad_resume_file(tmp_path):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    out = tmp_path / "out.json"
    out.write_text("{not json", encoding="utf-8")
    with pytest.raises(G.GeminiReviewError) as exc:
        G.run_review_set([review_path], out, token="tok",
                         opener=_SequenceOpener([_vertex_response(_vertex_judgment())]), sleep=lambda _s: None)
    assert exc.value.code == "resume_unreadable"


def test_run_review_set_rejects_out_colliding_with_review_or_clip(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    clip_path = review_path.parent / "pair_00_ab.mp4"
    original = clip_path.read_bytes()

    def _no_token():
        raise AssertionError("access token must not be acquired before the collision refusal")

    class _NoNetwork:
        def __call__(self, *args, **kwargs):
            raise AssertionError("network must not be used before the collision refusal")

    monkeypatch.setattr(G, "access_token", _no_token)
    colliding = [review_path, clip_path, review_path.parent / "sub" / ".." / review_path.name]
    for out in colliding:
        with pytest.raises(G.GeminiReviewError) as exc:
            G.run_review_set([review_path], out, token=None, opener=_NoNetwork(), sleep=lambda _s: None)
        assert exc.value.code == G.UNSAFE_PATH
    assert clip_path.read_bytes() == original
    assert not (review_path.parent / "sub").exists()


def test_document_computes_no_aggregate_winner_or_score(tmp_path):
    review_path, _ = _write_review(tmp_path / "r", pairs=2)
    opener = _SequenceOpener([_vertex_response(_vertex_judgment(preference="A"))] * 2)
    document = G.run_review_set([review_path], tmp_path / "out.json", token="tok", opener=opener,
                                sleep=lambda _s: None)
    forbidden = {"winner", "ranking", "rank", "score", "aggregate", "overall", "preference_counts", "summary"}
    assert not (set(document) & forbidden)
    assert "aggregate winner" in document["no_quality_score_note"]


# --------------------------------------------------------------------------------------
# atomic output safety
# --------------------------------------------------------------------------------------

def test_write_json_atomic_roundtrip_has_no_temp_files(tmp_path):
    out = tmp_path / "r.json"
    G.write_json_atomic(out, {"b": 1, "a": 2})
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 2, "b": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_atomic_replace_failure_preserves_destination(tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    out.write_text("ORIGINAL", encoding="utf-8")

    def _boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(G.os, "replace", _boom)
    with pytest.raises(OSError):
        G.write_json_atomic(out, {"a": 1})
    assert out.read_text(encoding="utf-8") == "ORIGINAL"
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_atomic_serialization_failure_preserves_destination(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    with pytest.raises(TypeError):
        G.write_json_atomic(out, {"bad": object()})
    assert out.read_text(encoding="utf-8") == "ORIGINAL"
    assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_requires_out():
    with pytest.raises(SystemExit):
        G.build_parser().parse_args(["review.json"])


def test_main_runs_and_maps_with_mocked_network_and_auth(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    out = tmp_path / "out.json"
    monkeypatch.setattr(G, "access_token", lambda: "tok")
    monkeypatch.setattr(G.urllib.request, "urlopen",
                        _SequenceOpener([_vertex_response(_vertex_judgment(preference="tie"))]))
    assert G.main([str(review_path), "--out", str(out)]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["complete"] is True
    assert document["judgments"][0]["raw_side_preference"] == "tie"
    assert document["judgments"][0]["mapped_preference"] == "tie"


def test_main_reports_validation_errors_without_secrets(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert G.main([str(bad), "--out", str(tmp_path / "o.json")]) == 2
    err = capsys.readouterr().err
    assert "review_unreadable" in err
