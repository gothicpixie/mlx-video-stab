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
