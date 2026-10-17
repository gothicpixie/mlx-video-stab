"""Weight-free, media-free tests for the research-only temporal border-fill pilot.

The pure fill core is exercised directly on tiny arrays; the end-to-end ``build`` path uses injected fakes
for ``mlxstab.media`` so no ffmpeg, no footage, no MLX and no network are touched. Coverage: nearest-donor
preference, symmetric tie averaging, no-donor passthrough, untouched valid pixels, positive-integer radius,
stream count/cadence mismatch and overwrite refusal.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mlxstab import media
from bench import manifest as MF
from bench import temporal_border_fill_pilot as P

FPS = 30.0


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------

class FakeEncoder:
    instances = []

    def __init__(self, path, width, height, fps, **kwargs):
        self.path, self.width, self.height, self.fps = str(path), width, height, fps
        self.kwargs = kwargs
        self.frames = []
        FakeEncoder.instances.append(self)

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def close(self):
        Path(self.path).write_bytes(b"fake-video-" + bytes([len(self.frames)]))


def _pts(count, fps):
    return {"count": int(count), "n_pts": int(count), "monotonic": True,
            "pts_s": [i / float(fps) for i in range(int(count))]}


def _setup(monkeypatch, tmp_path, *, count=3, width=4, height=4, video=None, mask=None,
           video_pts=None, mask_pts=None):
    if video is None:
        video = [np.full((height, width, 3), 10 + 10 * i, np.uint8) for i in range(count)]
    if mask is None:
        mask = [np.zeros((height, width, 3), np.uint8) for _ in range(count)]

    source = tmp_path / "source.mts"
    source.write_bytes(b"source")
    video_path = tmp_path / "stabilized.mp4"
    video_path.write_bytes(b"video")
    mask_path = tmp_path / "mask.mp4"
    mask_path.write_bytes(b"mask")
    manifest = {
        "schema": MF.MANIFEST_SCHEMA,
        "status": "success",
        "run_id": "run",
        "outputs": {
            "video": str(video_path), "mask": str(mask_path),
            "encoded_frames": count,
            "decoded": {"frames": count, "fps": FPS, "width": width, "height": height},
            "probe": {"width": width, "height": height, "fps": FPS},
            "encoded_pts": _pts(count, FPS), "mask_pts": _pts(count, FPS),
            "timeline_alignment": {"aligned": True},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fake_probe(path):
        return {"width": width, "height": height, "fps": FPS, "interlaced": False}

    def fake_decode(path, info=None, start=None, duration=None, deinterlace=True, vf=None):
        if Path(path).name == "mask.mp4":
            return iter([np.array(f, copy=True) for f in mask]), FPS, (width, height)
        return iter([np.array(f, copy=True) for f in video]), FPS, (width, height)

    def fake_frame_pts(path):
        name = Path(path).name
        if name == "stabilized.mp4":
            return video_pts or _pts(count, FPS)
        if name == "mask.mp4":
            return mask_pts or _pts(count, FPS)
        return _pts(len(FakeEncoder.instances[-1].frames), FPS)

    FakeEncoder.instances = []
    monkeypatch.setattr(media, "probe", fake_probe)
    monkeypatch.setattr(media, "decode", fake_decode)
    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    monkeypatch.setattr(media, "frame_pts", fake_frame_pts)
    return manifest_path, video, mask


# --------------------------------------------------------------------------------------
# pure fill core
# --------------------------------------------------------------------------------------

def _blank_window(n):
    return np.zeros((n, 1, 1, 3), np.uint8), np.zeros((n, 1, 1), bool)


def test_nearest_donor_prefers_the_closer_valid_neighbor():
    frames, masks = _blank_window(5)          # radius 2, center 2
    masks[2, 0, 0] = True                     # center invalid
    masks[1, 0, 0] = True                     # d=1 previous invalid
    frames[3, 0, 0] = [7, 8, 9]               # d=1 next valid (nearest)
    frames[0, 0, 0] = [100, 100, 100]         # d=2 previous also valid, must not win
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 2, 2)
    assert out[0, 0].tolist() == [7, 8, 9]
    assert (invalid, filled, unresolved) == (1, 1, 0)

    frames, masks = _blank_window(5)
    masks[2, 0, 0] = True
    masks[3, 0, 0] = True                     # d=1 next invalid
    frames[1, 0, 0] = [4, 5, 6]               # d=1 previous valid (nearest)
    frames[4, 0, 0] = [200, 200, 200]
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 2, 2)
    assert out[0, 0].tolist() == [4, 5, 6]


def test_symmetric_tie_averages_previous_and_next():
    frames, masks = _blank_window(5)
    masks[2, 0, 0] = True
    masks[1, 0, 0] = True                     # d=1 both invalid
    masks[3, 0, 0] = True
    frames[0, 0, 0] = [10, 20, 30]            # d=2 previous
    frames[4, 0, 0] = [40, 50, 60]            # d=2 next
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 2, 2)
    assert out[0, 0].tolist() == [25, 35, 45]
    assert (invalid, filled, unresolved) == (1, 1, 0)


def test_bilateral_agreement_gate_fills_only_when_two_donors_agree():
    frames, masks = _blank_window(3)
    masks[1, 0, 0] = True
    frames[0, 0, 0] = [10, 20, 30]
    frames[2, 0, 0] = [12, 23, 28]
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 1, 1, max_bilateral_delta=5)
    assert out[0, 0].tolist() == [11, 21, 29]
    assert (invalid, filled, unresolved) == (1, 1, 0)

    frames[2, 0, 0] = [60, 23, 28]
    frames[1, 0, 0] = [1, 2, 3]
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 1, 1, max_bilateral_delta=5)
    assert out[0, 0].tolist() == [1, 2, 3]
    assert (invalid, filled, unresolved) == (1, 0, 1)


def test_bilateral_gate_does_not_fall_back_after_nearest_disagreement():
    frames, masks = _blank_window(5)
    masks[2, 0, 0] = True
    frames[1, 0, 0] = [0, 0, 0]
    frames[3, 0, 0] = [100, 100, 100]  # nearest bilateral pair disagrees
    frames[0, 0, 0] = [10, 10, 10]
    frames[4, 0, 0] = [11, 11, 11]    # farther agreement is deliberately not used
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 2, 2, max_bilateral_delta=5)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert (invalid, filled, unresolved) == (1, 0, 1)


def test_no_donor_leaves_the_pixel_unchanged_and_unresolved():
    frames, masks = _blank_window(5)
    masks[2, 0, 0] = True
    for index in (0, 1, 3, 4):
        masks[index, 0, 0] = True
    frames[2, 0, 0] = [1, 2, 3]
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 2, 2)
    assert out[0, 0].tolist() == [1, 2, 3]
    assert (invalid, filled, unresolved) == (1, 0, 1)


def test_valid_pixels_are_preserved():
    frames, masks = _blank_window(3)
    frames[1, 0, 0] = [5, 6, 7]
    frames[0, 0, 0] = [9, 9, 9]
    frames[2, 0, 0] = [8, 8, 8]
    out, invalid, filled, unresolved = P.fill_center(frames, masks, 1, 1)
    assert out[0, 0].tolist() == [5, 6, 7]
    assert (invalid, filled, unresolved) == (0, 0, 0)


def test_stream_fill_handles_boundary_frames_and_keeps_a_bounded_window():
    count = 5
    frames = np.zeros((count, 2, 2, 3), np.uint8)
    masks = np.zeros((count, 2, 2, 3), np.uint8)
    masks[0, 0, 0] = 255                       # first frame border pixel
    masks[4, 1, 1] = 255                       # last frame border pixel
    frames[1, 0, 0] = [11, 22, 33]
    frames[3, 1, 1] = [44, 55, 66]
    seen = []
    window_sizes = []

    def on_frame(frame, record):
        seen.append(record["index"])

    records, totals = P.stream_fill(zip(iter(frames), iter(masks)), count, 1, on_frame,
                                   window_probe=window_sizes.append)
    assert seen == [0, 1, 2, 3, 4]
    assert max(window_sizes) <= 2 * 1 + 1
    assert records[0]["filled"] == 1 and records[0]["unresolved"] == 0
    assert records[4]["filled"] == 1 and records[4]["unresolved"] == 0
    assert totals["filled"] == 2


# --------------------------------------------------------------------------------------
# end-to-end build (fake media)
# --------------------------------------------------------------------------------------

def test_build_writes_candidate_and_adjacent_json(tmp_path, monkeypatch):
    video = [np.full((4, 4, 3), 10 + 10 * i, np.uint8) for i in range(3)]
    mask = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    mask[1][0, 0] = 255
    manifest_path, _, _ = _setup(monkeypatch, tmp_path, video=video, mask=mask)
    out_video = tmp_path / "candidate.mp4"
    report = P.build(manifest_path, out_video=out_video, radius=1)

    assert out_video.is_file()
    assert (tmp_path / "candidate.json").is_file()
    assert report["status"] == "success" and report["schema"] == P.SCHEMA
    assert report["radius"] == 1 and report["max_window_frames"] == 3
    assert report["candidate"]["frames"] == 3 and report["candidate"]["within_inline_cap"] is True
    assert report["candidate"]["cfr"]["aligned"] is True
    assert report["fill"]["totals"]["invalid"] == 1
    assert report["fill"]["totals"]["filled"] == 1
    assert report["fill"]["totals"]["unresolved"] == 0
    # equidistant frame 0 / frame 2 valid -> integer mean at frame 1 (10 and 30 -> 20)
    assert FakeEncoder.instances[-1].frames[1][0, 0].tolist() == [20, 20, 20]
    # all valid pixels stay exactly as decoded
    assert np.array_equal(FakeEncoder.instances[-1].frames[0], video[0])
    assert np.array_equal(FakeEncoder.instances[-1].frames[2], video[2])
    on_disk = json.loads((tmp_path / "candidate.json").read_text())
    assert on_disk["candidate"]["sha256"] == MF.sha256_file(out_video)


def test_build_records_and_applies_bilateral_threshold(tmp_path, monkeypatch):
    video = [np.full((4, 4, 3), value, np.uint8) for value in (10, 20, 30)]
    mask = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    mask[1][0, 0] = 255
    manifest_path, _, _ = _setup(monkeypatch, tmp_path, video=video, mask=mask)
    report = P.build(manifest_path, out_video=tmp_path / "gated.mp4", radius=1, max_bilateral_delta=5)
    assert report["max_bilateral_delta"] == 5
    assert "bilateral_only" in report["policy"]
    assert report["fill"]["totals"]["filled"] == 0
    assert report["fill"]["totals"]["unresolved"] == 1
    assert FakeEncoder.instances[-1].frames[1][0, 0].tolist() == [20, 20, 20]


def test_build_requires_a_successful_run_case_manifest(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    doc = json.loads(manifest_path.read_text())
    doc["status"] = "running"
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(P.TemporalFillError, match="status must be success"):
        P.build(manifest_path, radius=1)


@pytest.mark.parametrize("radius", [0, -1, 31, True, 1.5])
def test_build_rejects_non_positive_or_unbounded_radius(tmp_path, monkeypatch, radius):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    with pytest.raises(P.TemporalFillError, match="radius"):
        P.build(manifest_path, radius=radius)


@pytest.mark.parametrize("threshold", [-1, 256, float("nan"), float("inf"), True])
def test_build_rejects_invalid_bilateral_threshold(tmp_path, monkeypatch, threshold):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    with pytest.raises(P.TemporalFillError, match="max_bilateral_delta"):
        P.build(manifest_path, radius=1, max_bilateral_delta=threshold)


def test_build_rejects_decoded_frame_count_mismatch(tmp_path, monkeypatch):
    video = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]     # decode yields 2
    mask = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    manifest_path, _, _ = _setup(monkeypatch, tmp_path, count=3, video=video, mask=mask)
    with pytest.raises(P.TemporalFillError, match="decoded frame count"):
        P.build(manifest_path, radius=1)


def test_build_rejects_cadence_mismatch(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path, video_pts=_pts(3, 25.0))
    with pytest.raises(P.TemporalFillError, match="CFR"):
        P.build(manifest_path, radius=1)


def test_build_rejects_disagreeing_manifest_timeline(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    doc = json.loads(manifest_path.read_text())
    doc["outputs"]["timeline_alignment"] = {"aligned": False}
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(P.TemporalFillError, match="timeline alignment"):
        P.build(manifest_path, radius=1)


def test_build_refuses_to_overwrite_existing_artifacts(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    existing = tmp_path / "keep.mp4"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        P.build(manifest_path, out_video=existing, radius=1)
    assert existing.read_bytes() == b"keep"

    report_path = tmp_path / "keep.json"
    report_path.write_bytes(b"keep-json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        P.build(manifest_path, out_video=tmp_path / "new.mp4", out_json=report_path, radius=1)
    assert report_path.read_bytes() == b"keep-json"


def test_build_enforces_gemini_inline_cap(tmp_path, monkeypatch):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)

    class BigEncoder(FakeEncoder):
        def close(self):
            Path(self.path).write_bytes(b"x" * 4096)

    monkeypatch.setattr(media, "Encoder", BigEncoder)
    monkeypatch.setattr(P.Gemini, "MAX_INLINE_BYTES", 1024)
    with pytest.raises(P.TemporalFillError, match="inline cap"):
        P.build(manifest_path, out_video=tmp_path / "capped.mp4", radius=1)
    assert not (tmp_path / "capped.mp4").exists()


def test_cli_reports_success(tmp_path, monkeypatch, capsys):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    out_video = tmp_path / "cli.mp4"
    assert P.main([str(manifest_path), "--radius", "1", "--out-video", str(out_video)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "success", "video": str(out_video),
                       "report": str(tmp_path / "cli.json"), "radius": 1, "frames": 3}


def test_cli_reports_structured_failure(tmp_path, monkeypatch, capsys):
    manifest_path, _, _ = _setup(monkeypatch, tmp_path)
    assert P.main([str(manifest_path), "--radius", "0"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failure" and payload["error"]["type"] == "TemporalFillError"
