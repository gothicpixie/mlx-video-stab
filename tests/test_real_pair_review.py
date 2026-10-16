import json
from pathlib import Path

import numpy as np
import pytest

from mlxstab import media
from bench import manifest as MF
from bench import real_pair_review as R


class FakeEncoder:
    instances = []

    def __init__(self, path, width, height, fps, **kwargs):
        self.path, self.width, self.height, self.fps = str(path), width, height, fps
        self.frames = []
        self.instances.append(self)

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def close(self):
        Path(self.path).write_bytes(b"fake-video" + bytes([len(self.frames)]))


def _pts(count, fps):
    return {"count": count, "n_pts": count, "monotonic": True,
            "pts_s": [i / fps for i in range(count)]}


def _manifest(tmp_path, *, status="success", source_frames=4, output_frames=2):
    source = tmp_path / "source.mts"
    source.write_bytes(b"source")
    video = tmp_path / "stabilized.mp4"
    video.write_bytes(b"stabilized")
    return {
        "status": status,
        "source": {
            "path": str(source), "path_type": "file", "sha256": MF.sha256_file(source),
            "window": {"start_s": 1.0, "duration_s": 0.1},
            "preprocessing": {"interlaced": True,
                              "deinterlace": "bwdif=mode=send_field:parity=auto:deint=all", "decode_vf": None},
            "decoded": {"frames": source_frames, "fps": 60.0, "width": 8, "height": 6},
        },
        "outputs": {
            "video": str(video), "encoded_frames": output_frames,
            "decoded": {"frames": output_frames, "fps": 30.0, "width": 8, "height": 6},
            "cadence": {"measured_fps": 30.0, "measured_frames": output_frames},
            "encoded_pts": _pts(output_frames, 30.0), "timeline_alignment": {"aligned": True},
        },
    }


def _setup(monkeypatch, tmp_path, source_frames=4, output_frames=2):
    raw = [np.full((6, 8, 3), i * 30, np.uint8) for i in range(source_frames)]
    stab = [np.full((6, 8, 3), 200 + i * 10, np.uint8) for i in range(output_frames)]
    manifest = _manifest(tmp_path, source_frames=source_frames, output_frames=output_frames)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    FakeEncoder.instances = []
    def fake_probe(path):
        fps = 60.0 if Path(path).name == "source.mts" else 30.0
        return {"width": 8, "height": 6, "fps": fps, "interlaced": Path(path).name == "source.mts"}
    def fake_decode(path, info=None, start=None, duration=None, deinterlace=True, vf=None):
        frames = raw if Path(path).name == "source.mts" else stab
        fps = 60.0 if frames is raw else 30.0
        return iter(frames), fps, (8, 6)
    monkeypatch.setattr(media, "probe", fake_probe)
    monkeypatch.setattr(media, "decode", fake_decode)
    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    monkeypatch.setattr(media, "frame_pts", lambda path: _pts(2, 30.0))
    return manifest_path, raw, stab


def test_source_indices_exact_nearest_unique_and_refuses_ambiguity():
    assert R.source_indices("60000/1001", "30000/1001", 4, 8) == [0, 2, 4, 6]
    assert R.source_indices(30, 30, 4, 4) == [0, 1, 2, 3]
    with pytest.raises(R.ReviewBuildError, match="between two"):
        R.source_indices(60, 40, 2, 4)  # output frame 1 is at source frame 1.5
    with pytest.raises(R.ReviewBuildError, match="outside"):
        R.source_indices(60, 30, 3, 4)


def test_build_writes_original_and_position_reversed_clips_with_provenance(tmp_path, monkeypatch):
    manifest_path, raw, stab = _setup(monkeypatch, tmp_path)
    video_path = tmp_path / "paired.mp4"
    report_path = tmp_path / "paired.json"
    report = R.build(manifest_path, out_video=video_path, out_report=report_path,
                     panel_width=40, panel_height=32)
    assert len(FakeEncoder.instances) == 2
    original, reversed_clip = FakeEncoder.instances
    assert ".paired." in original.path and "_posrev" not in original.path
    assert ".paired_posrev." in reversed_clip.path
    for encoder in (original, reversed_clip):
        assert len(encoder.frames) == 2
        assert encoder.frames[0].shape == (32, 80, 3)
    # Original: raw frame zero (0) on the left, stabilized frame zero (200) on the right.
    assert np.mean(original.frames[0][28:, :40]) < 1
    assert np.all(original.frames[0][28:, 75:] == 200)
    # Position-reversed: the same paired frames with the two panels exchanged (sample right of the label box).
    assert np.all(reversed_clip.frames[0][28:, 30:40] == 200)
    assert np.mean(reversed_clip.frames[0][28:, 75:]) < 1
    assert report["evidence"]["mapping"]["source_frame_indices"] == [0, 2]
    assert report["evidence"]["panels"]["original"] == {"clip": "paired.mp4", "left_panel": "raw",
                                                         "right_panel": "stabilized"}
    assert report["evidence"]["panels"]["position_reversed"] == {"clip": "paired_posrev.mp4",
                                                                  "left_panel": "stabilized",
                                                                  "right_panel": "raw"}
    assert [entry["role"] for entry in report["evidence"]["outputs"]] == ["original", "position_reversed"]
    assert [entry["frames"] for entry in report["evidence"]["outputs"]] == [2, 2]
    api_input = json.loads(report_path.read_text())
    assert api_input["output"]["clips"] == ["paired.mp4", "paired_posrev.mp4"]
    from bench import gemini_blind_review as G
    roles = [pair["position_reversal"]["role"] for pair in api_input["pairs"]]
    assert roles == ["original", "position_reversed"]
    groups = {pair["position_reversal"]["group"] for pair in api_input["pairs"]}
    assert len(groups) == 1 and all(group for group in groups)
    assert all(pair["position_reversal"]["schema"] == G.POSITION_REVERSAL_SCHEMA
               for pair in api_input["pairs"])
    validated = G.load_review(report_path)
    assert [pair["clip"] for pair in validated["pairs"]] == ["paired.mp4", "paired_posrev.mp4"]
    assert G.map_preference(validated["pairs"][0]["blind_mapping"], "A") == "raw"
    assert G.map_preference(validated["pairs"][0]["blind_mapping"], "B") == "stabilized"
    assert G.map_preference(validated["pairs"][1]["blind_mapping"], "A") == "stabilized"
    assert G.map_preference(validated["pairs"][1]["blind_mapping"], "B") == "raw"
    # The Gemini reviewer validates and enumerates both clips with their hidden mappings.
    tasks = G.collect_tasks([report_path])
    assert [task["pair_index"] for task in tasks] == [0, 1]
    assert tasks[0]["blind_mapping"]["left_panel"] == "raw"
    assert tasks[1]["blind_mapping"]["left_panel"] == "stabilized"
    assert tasks[1]["blind_mapping"]["right_panel"] == "raw"
    assert [task["position_reversal"]["role"] for task in tasks] == ["original", "position_reversed"]
    assert {task["position_reversal"]["group"] for task in tasks} == groups


def test_build_enforces_inline_cap_for_each_clip(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)

    class SizedEncoder(FakeEncoder):
        def close(self):
            Path(self.path).write_bytes(b"x" * (4096 if "posrev" in self.path else 8))

    monkeypatch.setattr(media, "Encoder", SizedEncoder)
    monkeypatch.setattr(R.Gemini, "MAX_INLINE_BYTES", 1024)
    with pytest.raises(R.ReviewBuildError, match="position_reversed"):
        R.build(manifest_path, panel_width=40, panel_height=32)

    monkeypatch.setattr(R.Gemini, "MAX_INLINE_BYTES", 8192)
    report = R.build(manifest_path, out_video=tmp_path / "ok.mp4", out_report=tmp_path / "ok.json",
                     panel_width=40, panel_height=32)
    assert [entry["role"] for entry in report["evidence"]["outputs"]] == ["original", "position_reversed"]


def test_build_refuses_existing_reversed_artifact(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    reversed_path = tmp_path / "paired_posrev.mp4"
    reversed_path.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        R.build(manifest_path, out_video=tmp_path / "paired.mp4", out_reversed=reversed_path,
                panel_width=40, panel_height=32)
    assert reversed_path.read_bytes() == b"keep"


def test_build_requires_both_clips_and_report_in_one_directory(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(R.ReviewBuildError, match="same directory"):
        R.build(manifest_path, out_video=tmp_path / "paired.mp4",
                out_reversed=other / "paired_posrev.mp4", out_report=tmp_path / "paired.json",
                panel_width=40, panel_height=32)


def test_build_rejects_bad_pts_for_reversed_clip(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    good = media.frame_pts
    monkeypatch.setattr(media, "frame_pts",
                        lambda path: _pts(2, 25.0) if "_posrev" in str(path) else good(path))
    with pytest.raises(R.ReviewBuildError, match="generated position_reversed"):
        R.build(manifest_path, out_video=tmp_path / "paired.mp4", panel_width=40, panel_height=32)


def test_build_rejects_source_frame_dimensions_changing_midstream(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    decode = media.decode

    def wrong_source_frame(path, *args, **kwargs):
        frames, fps, size = decode(path, *args, **kwargs)
        if Path(path).name == "source.mts":
            frames = iter([np.zeros((4, 8, 3), np.uint8), *list(frames)[1:]])
        return frames, fps, size

    monkeypatch.setattr(media, "decode", wrong_source_frame)
    with pytest.raises(R.ReviewBuildError, match="source frame dimensions changed"):
        R.build(manifest_path, panel_width=40, panel_height=32)


def test_cli_builds_both_clips_and_reports_reversed(tmp_path, monkeypatch, capsys):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    video_path = tmp_path / "cli.mp4"
    report_path = tmp_path / "cli.json"
    assert R.main([str(manifest_path), "--out-video", str(video_path),
                   "--out-report", str(report_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "success", "output": str(video_path),
                       "reversed": str(tmp_path / "cli_posrev.mp4"),
                       "report": str(report_path), "frames": 2}
    assert len(FakeEncoder.instances) == 2


@pytest.mark.parametrize("change, message", [
    (lambda m: m.update(status="running"), "status must be success"),
    (lambda m: m["source"].update(sha256="0" * 64), "SHA-256"),
    (lambda m: m["outputs"].update(encoded_frames=3), "counts are missing or inconsistent"),
    (lambda m: m["outputs"].update(encoded_pts=_pts(2, 25.0)), "not zero-based constant-frame-rate"),
])
def test_build_rejects_bad_manifest(tmp_path, monkeypatch, change, message):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    doc = json.loads(manifest_path.read_text())
    change(doc)
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(R.ReviewBuildError, match=message):
        R.build(manifest_path, panel_width=40, panel_height=32)


def test_build_refuses_output_collisions(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    existing = tmp_path / "already.mp4"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        R.build(manifest_path, out_video=existing, panel_width=40, panel_height=32)
    assert existing.read_bytes() == b"keep"


def test_build_rejects_source_decode_count_mismatch(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path, source_frames=3, output_frames=2)
    # The manifest says 3 source frames, but the nearest map still fits; report must match actual decoded count.
    doc = json.loads(manifest_path.read_text())
    doc["source"]["decoded"]["frames"] = 4
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(R.ReviewBuildError, match="source decoded frame count"):
        R.build(manifest_path, panel_width=40, panel_height=32)


def test_build_rejects_stabilized_file_with_bad_pts(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    original = media.frame_pts
    monkeypatch.setattr(media, "frame_pts", lambda path: _pts(2, 25.0)
                        if Path(path).name == "stabilized.mp4" else original(path))
    with pytest.raises(R.ReviewBuildError, match="stabilized video is not zero-based"):
        R.build(manifest_path, panel_width=40, panel_height=32)


# --------------------------------------------------------------------------------------
# opt-in candidate mode: stabilized baseline output vs an explicit postprocess candidate
# --------------------------------------------------------------------------------------

def _candidate_setup(monkeypatch, tmp_path, candidates, *, output_frames=2, source_frames=4):
    """Set up the default fake run plus one or more candidate MP4s.

    ``candidates`` maps a candidate filename to its base pixel value; each candidate decodes to constant
    frames so panel assignment is exact. The default fake decodes the manifest's stabilized output to
    values 200/210/... and the raw source to 0/30/...
    """
    manifest_path, _, baseline_frames = _setup(monkeypatch, tmp_path, source_frames=source_frames,
                                               output_frames=output_frames)
    frames_by_name = {}
    for name, value in candidates.items():
        (tmp_path / name).write_bytes(b"candidate-" + name.encode("ascii"))
        frames_by_name[name] = [np.full((6, 8, 3), value + i * 5, np.uint8) for i in range(output_frames)]
    decode = media.decode

    def fake_decode(path, *args, **kwargs):
        name = Path(path).name
        if name in frames_by_name:
            return iter(frames_by_name[name]), 30.0, (8, 6)
        return decode(path, *args, **kwargs)

    monkeypatch.setattr(media, "decode", fake_decode)
    return manifest_path, frames_by_name


def test_build_candidate_isolates_postprocess_and_assigns_panels_both_ways(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    candidate_path = tmp_path / "candidate.mp4"
    report_path = tmp_path / "cand.json"
    report = R.build(manifest_path, candidate_video=candidate_path, out_video=tmp_path / "cand.mp4",
                     out_report=report_path, panel_width=40, panel_height=32)
    assert len(FakeEncoder.instances) == 2
    original, reversed_clip = FakeEncoder.instances
    for encoder in (original, reversed_clip):
        assert len(encoder.frames) == 2
        assert encoder.frames[0].shape == (32, 80, 3)
    # Original: baseline stabilized frame zero (200) left/A, candidate frame zero (50) right/B.
    assert np.all(original.frames[0][28:, 30:40] == 200)
    assert np.all(original.frames[0][28:, 75:] == 50)
    # Position-reversed: the same exact paired output frames with the two panels exchanged.
    assert np.all(reversed_clip.frames[0][28:, 30:40] == 50)
    assert np.all(reversed_clip.frames[0][28:, 75:] == 200)
    # Hidden mappings use the generic baseline/candidate labels in both orders.
    assert report["pairs"][0]["blind_mapping"]["left_panel"] == "baseline"
    assert report["pairs"][0]["blind_mapping"]["right_panel"] == "candidate"
    assert report["pairs"][1]["blind_mapping"]["left_panel"] == "candidate"
    assert report["pairs"][1]["blind_mapping"]["right_panel"] == "baseline"
    # Evidence records both input hashes, exact cadence, frame-index identity, panel mappings and hashes.
    evidence = report["evidence"]
    assert evidence["mode"] == "candidate"
    assert evidence["baseline"]["sha256"] == MF.sha256_file(tmp_path / "stabilized.mp4")
    assert evidence["candidate"]["sha256"] == MF.sha256_file(candidate_path)
    assert evidence["mapping"]["frame_indices"] == [0, 1]
    assert evidence["mapping"]["unique"] is True
    assert evidence["cadence"] == {"fps": [30, 1], "frames": 2, "measured_fps": 30.0,
                                   "measured_frames": 2}
    assert evidence["panels"]["original"] == {"clip": "cand.mp4", "left_panel": "baseline",
                                              "right_panel": "candidate"}
    assert evidence["panels"]["position_reversed"] == {"clip": "cand_posrev.mp4",
                                                       "left_panel": "candidate",
                                                       "right_panel": "baseline"}
    assert [entry["role"] for entry in evidence["outputs"]] == ["original", "position_reversed"]
    assert {entry["sha256"] for entry in evidence["outputs"]} == {
        MF.sha256_file(tmp_path / "cand.mp4"), MF.sha256_file(tmp_path / "cand_posrev.mp4")}
    assert report["provenance"]["baseline_sha256"] == evidence["baseline"]["sha256"]
    assert report["provenance"]["candidate_sha256"] == evidence["candidate"]["sha256"]


def test_candidate_default_output_name_differs_from_raw_mode(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    report = R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                     panel_width=40, panel_height=32)
    assert report["output"]["clips"] == ["baseline_vs_candidate.mp4",
                                         "baseline_vs_candidate_posrev.mp4"]
    for name in ("baseline_vs_candidate.mp4", "baseline_vs_candidate_posrev.mp4",
                 "baseline_vs_candidate.json"):
        assert (tmp_path / name).is_file()


@pytest.mark.parametrize("bad_name, where", [("stabilized.mp4", "baseline"),
                                             ("candidate.mp4", "candidate")])
def test_build_candidate_independently_rejects_output_cfr(tmp_path, monkeypatch, bad_name, where):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    current = media.frame_pts
    monkeypatch.setattr(media, "frame_pts",
                        lambda path: _pts(2, 25.0) if Path(path).name == bad_name else current(path))
    with pytest.raises(R.ReviewBuildError, match=f"{where} is not zero-based"):
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                out_video=tmp_path / "x.mp4", out_report=tmp_path / "x.json",
                panel_width=40, panel_height=32)


@pytest.mark.parametrize("bad_name, where", [("stabilized.mp4", "baseline"),
                                             ("candidate.mp4", "candidate")])
def test_build_candidate_independently_rejects_dimensions(tmp_path, monkeypatch, bad_name, where):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    current = media.probe

    def fake_probe(path):
        info = dict(current(path))
        if Path(path).name == bad_name:
            info["width"] = 9
        return info

    monkeypatch.setattr(media, "probe", fake_probe)
    with pytest.raises(R.ReviewBuildError, match=f"{where} video probe disagrees"):
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                out_video=tmp_path / "x.mp4", out_report=tmp_path / "x.json",
                panel_width=40, panel_height=32)


@pytest.mark.parametrize("bad_name, delta, message", [
    ("stabilized.mp4", 1, "baseline decoded frame count exceeds manifest"),
    ("candidate.mp4", 1, "candidate decoded frame count exceeds manifest"),
    ("stabilized.mp4", -1, "baseline decode ended before manifest frame count"),
    ("candidate.mp4", -1, "candidate decode ended before manifest frame count"),
])
def test_build_candidate_independently_rejects_frame_count(tmp_path, monkeypatch, bad_name, delta, message):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    current = media.decode

    def fake_decode(path, *args, **kwargs):
        frames, fps, size = current(path, *args, **kwargs)
        if Path(path).name == bad_name:
            frames = list(frames)
            frames = frames + [frames[-1]] if delta > 0 else frames[:-1]
        return frames, fps, size

    monkeypatch.setattr(media, "decode", fake_decode)
    with pytest.raises(R.ReviewBuildError, match=message):
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                out_video=tmp_path / "x.mp4", out_report=tmp_path / "x.json",
                panel_width=40, panel_height=32)


def test_build_candidate_rejects_shape_mismatch(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    current = media.decode

    def fake_decode(path, *args, **kwargs):
        frames, fps, size = current(path, *args, **kwargs)
        if Path(path).name == "candidate.mp4":
            frames = list(frames)
            frames[0] = np.zeros((6, 8, 4), np.uint8)
        return frames, fps, size

    monkeypatch.setattr(media, "decode", fake_decode)
    with pytest.raises(R.ReviewBuildError, match="shapes differ"):
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                out_video=tmp_path / "x.mp4", out_report=tmp_path / "x.json",
                panel_width=40, panel_height=32)


def test_build_candidate_refuses_collisions_missing_and_overwrite(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    with pytest.raises(R.ReviewBuildError, match="differ from the manifest baseline"):
        R.build(manifest_path, candidate_video=tmp_path / "stabilized.mp4",
                panel_width=40, panel_height=32)
    with pytest.raises(R.ReviewBuildError, match="collides with a manifest/source input"):
        R.build(manifest_path, candidate_video=tmp_path / "source.mts",
                panel_width=40, panel_height=32)
    with pytest.raises(R.ReviewBuildError, match="candidate video is missing"):
        R.build(manifest_path, candidate_video=tmp_path / "nope.mp4",
                panel_width=40, panel_height=32)
    existing = tmp_path / "keep.mp4"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4", out_video=existing,
                panel_width=40, panel_height=32)
    assert existing.read_bytes() == b"keep"


def test_candidate_review_groups_distinguish_inputs_and_join_roles(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"c1.mp4": 50, "c2.mp4": 60})
    report_a = R.build(manifest_path, candidate_video=tmp_path / "c1.mp4",
                       out_video=tmp_path / "a.mp4", out_report=tmp_path / "a.json",
                       panel_width=40, panel_height=32)
    report_b = R.build(manifest_path, candidate_video=tmp_path / "c2.mp4",
                       out_video=tmp_path / "b.mp4", out_report=tmp_path / "b.json",
                       panel_width=40, panel_height=32)
    groups_a = {pair["position_reversal"]["group"] for pair in report_a["pairs"]}
    groups_b = {pair["position_reversal"]["group"] for pair in report_b["pairs"]}
    assert len(groups_a) == len(groups_b) == 1
    assert report_a["position_reversal"]["group"] in groups_a
    # Same source window, different candidate hash: the groups must stay distinct.
    assert groups_a != groups_b
    assert [pair["position_reversal"]["role"] for pair in report_a["pairs"]] == [
        "original", "position_reversed"]


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
    def __init__(self, preferences):
        self.preferences = list(preferences)
        self.bodies = []

    def __call__(self, request, timeout=None):
        self.bodies.append(json.loads(request.data.decode("utf-8")))
        return _FakeVertexResponse(self.preferences.pop(0))


def test_candidate_review_hidden_mapping_and_paired_position_audit(tmp_path, monkeypatch):
    from bench import gemini_blind_review as G
    from bench import gemini_position_bias_audit as A

    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    report_path = tmp_path / "cand.json"
    report = R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                     out_video=tmp_path / "cand.mp4", out_report=report_path,
                     panel_width=40, panel_height=32)
    validated = G.load_review(report_path)
    assert [G.map_preference(pair["blind_mapping"], side)
            for pair in validated["pairs"] for side in ("A", "B")] == [
        "baseline", "candidate", "candidate", "baseline"]
    tasks = G.collect_tasks([report_path])
    assert [task["position_reversal"]["role"] for task in tasks] == ["original", "position_reversed"]
    assert {task["position_reversal"]["group"] for task in tasks} == \
        {report["position_reversal"]["group"]}

    # Give the two encoded clips distinct bytes so the fake reviewer's resume cache cannot collapse them.
    original_clip, reversed_clip = report["output"]["clips"]
    for name, marker in ((original_clip, b"-baseline"), (reversed_clip, b"-candidate")):
        clip = tmp_path / name
        clip.write_bytes(clip.read_bytes() + marker)

    opener = _RecordingOpener(["A", "B"])  # original A -> baseline; reversal B -> baseline
    out = tmp_path / "candidate_result.json"
    document = G.run_review_set([report_path], out, token="tok", opener=opener,
                                sleep=lambda _s: None)
    # The Vertex request carries only the clip bytes and the fixed brief; no generic panel labels leak.
    for body in opener.bodies:
        assert body["contents"][0]["parts"][1]["text"] == G.REVIEW_BRIEF
    assert [record["mapped_preference"] for record in document["judgments"]] == [
        "baseline", "baseline"]

    audit = A.audit([], [], [], [out])
    assert audit["coverage"]["complete_pairs"] == 1
    pair = audit["pairs"][0]
    assert pair["identity"]["group"] == report["position_reversal"]["group"]
    assert pair["original"]["mapped_method"] == "baseline"
    assert pair["reversal"]["mapped_method"] == "baseline"
    assert pair["original"]["blind_mapping"]["left_panel"] == "baseline"
    assert pair["reversal"]["blind_mapping"]["left_panel"] == "candidate"
    assert pair["derived"]["method_agreement"] is True
    assert pair["derived"]["mapping_reversed"] is True
    assert "baseline" in audit["aggregates"]["original_mapped_method_counts"]
    assert audit["aggregates"]["original_mapped_method_counts"]["baseline"] == 1
    assert audit["aggregates"]["reversal_mapped_method_counts"]["baseline"] == 1


def test_cli_candidate_mode_builds_and_reports(tmp_path, monkeypatch, capsys):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})
    video_path = tmp_path / "cli_cand.mp4"
    report_path = tmp_path / "cli_cand.json"
    assert R.main([str(manifest_path), "--candidate-video", str(tmp_path / "candidate.mp4"),
                   "--out-video", str(video_path), "--out-report", str(report_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "success", "output": str(video_path),
                       "reversed": str(tmp_path / "cli_cand_posrev.mp4"),
                       "report": str(report_path), "frames": 2}
    assert len(FakeEncoder.instances) == 2


class _TrackedEncoder(FakeEncoder):
    """FakeEncoder that records whether close() was invoked (so sibling cleanup is provable)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.closed = False

    def close(self):
        self.closed = True
        super().close()


def test_build_closes_both_paired_clips_and_propagates_first_close_failure(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)

    class CloseBoomEncoder(_TrackedEncoder):
        def close(self):
            self.closed = True
            raise RuntimeError(f"close {self.path}")

    monkeypatch.setattr(media, "Encoder", CloseBoomEncoder)
    with pytest.raises(RuntimeError) as ei:
        R.build(manifest_path, out_video=tmp_path / "paired.mp4", out_report=tmp_path / "paired.json",
                panel_width=40, panel_height=32)
    assert len(FakeEncoder.instances) == 2
    assert all(encoder.closed for encoder in FakeEncoder.instances)          # both paired clips closed
    assert str(ei.value) == f"close {FakeEncoder.instances[0].path}"         # first close failure, not the later one


def test_build_preserves_build_failure_while_closing_both_paired_clips(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)

    class BuildBoom(Exception):
        pass

    class WriteAndCloseBoomEncoder(_TrackedEncoder):
        def write(self, frame):
            raise BuildBoom("write failed")
        def close(self):
            self.closed = True
            raise RuntimeError(f"close {self.path}")

    monkeypatch.setattr(media, "Encoder", WriteAndCloseBoomEncoder)
    with pytest.raises(BuildBoom, match="write failed"):
        R.build(manifest_path, out_video=tmp_path / "paired.mp4", out_report=tmp_path / "paired.json",
                panel_width=40, panel_height=32)
    assert len(FakeEncoder.instances) == 2
    assert all(encoder.closed for encoder in FakeEncoder.instances)          # closes attempted despite failure


def test_build_candidate_closes_both_clips_and_propagates_first_close_failure(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})

    class CloseBoomEncoder(_TrackedEncoder):
        def close(self):
            self.closed = True
            raise RuntimeError(f"close {self.path}")

    monkeypatch.setattr(media, "Encoder", CloseBoomEncoder)
    with pytest.raises(RuntimeError) as ei:
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                out_video=tmp_path / "cand.mp4", out_report=tmp_path / "cand.json",
                panel_width=40, panel_height=32)
    assert len(FakeEncoder.instances) == 2
    assert all(encoder.closed for encoder in FakeEncoder.instances)
    assert str(ei.value) == f"close {FakeEncoder.instances[0].path}"


def test_build_candidate_preserves_build_failure_while_closing_both_clips(tmp_path, monkeypatch):
    manifest_path, _ = _candidate_setup(monkeypatch, tmp_path, {"candidate.mp4": 50})

    class BuildBoom(Exception):
        pass

    class WriteAndCloseBoomEncoder(_TrackedEncoder):
        def write(self, frame):
            raise BuildBoom("write failed")
        def close(self):
            self.closed = True
            raise RuntimeError(f"close {self.path}")

    monkeypatch.setattr(media, "Encoder", WriteAndCloseBoomEncoder)
    with pytest.raises(BuildBoom, match="write failed"):
        R.build(manifest_path, candidate_video=tmp_path / "candidate.mp4",
                out_video=tmp_path / "cand.mp4", out_report=tmp_path / "cand.json",
                panel_width=40, panel_height=32)
    assert len(FakeEncoder.instances) == 2
    assert all(encoder.closed for encoder in FakeEncoder.instances)
