"""Focused tests for the research-only local position-reversal of blinded A/B review clips.

Everything is mocked for media: no FFmpeg, no MLX, no network. Coverage: the pixel panel swap and the fixed
A-left/B-right relabelling, hidden ``blind_mapping`` reversal and clip renaming in the copied report, a
single selected-pair end-to-end run through a mocked decoder/encoder, and refusals for malformed mapping,
bad/missing clips, unknown pair indices and output/input path collisions. The source review and clips are
never modified.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mlxstab import media

from bench import gemini_blind_review as G
from bench import multiscale_position_reversal as P
from bench import multiscale_real_review as R


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

class _FakeEncoder:
    instances = []
    #: Class-level toggles let a test make writes or the final flush fail without a bespoke fake; both are
    #: reset by :func:`_mock_media` so ordinary tests keep the success behavior.
    write_exc = None
    close_exc = None

    def __init__(self, path, width, height, fps, **kwargs):
        self.path = str(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
        self.closed = False
        self.close_attempts = 0
        _FakeEncoder.instances.append(self)

    def write(self, frame):
        if _FakeEncoder.write_exc is not None:
            raise _FakeEncoder.write_exc
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        self.closed = True
        self.close_attempts += 1
        if _FakeEncoder.close_exc is not None:
            raise _FakeEncoder.close_exc
        # Emulate a real encoder writing an MP4 so downstream digests/existence checks are exercised.
        Path(self.path).write_bytes(b"fake-mp4" + bytes(len(self.frames)))


def _frames(count, height, width, *, left=40, right=200):
    frames = []
    for _ in range(count):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :width // 2] = left
        frame[:, width // 2:] = right
        frames.append(frame)
    return frames


def _mock_media(monkeypatch, width, height, fps, frames):
    _FakeEncoder.instances = []
    _FakeEncoder.write_exc = None
    _FakeEncoder.close_exc = None
    monkeypatch.setattr(media, "probe",
                        lambda path: {"width": width, "height": height, "fps": fps, "interlaced": False,
                                      "field_order": "progressive", "duration": 1.0})
    monkeypatch.setattr(
        media, "decode",
        lambda path, info=None, start=None, duration=None, deinterlace=False, vf=None:
        (iter(frames), fps, (width, height)))
    monkeypatch.setattr(media, "Encoder", _FakeEncoder)


def _provenance():
    return {
        "run": "run-1",
        "run_id": "run-1",
        "case": "case-1",
        "source_sha256": "a" * 64,
        "source_window": {"start_s": 0.0, "duration_s": 2.0},
    }


def _write_review(root, *, nested=False, pairs=2, left="multiscale", right="l1",
                  clip_names=None, bad_clip=None, bad_mapping=False, size=64):
    """Write a valid multiscale review document plus clip files; return (review_path, document)."""
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_list = []
    for index in range(pairs):
        name = (clip_names[index] if clip_names else f"pair_{index:02d}_ab.mp4")
        if index == 0 and bad_clip:
            name = bad_clip
        if name and "/" not in name and not name.startswith("."):
            (out_dir / name).write_bytes(bytes([index + 1]) * size)
        mapping = {"left_panel": left, "right_panel": right,
                   "video_labels": {"left": "A", "right": "B"}}
        if index == 0 and bad_mapping:
            mapping = {"left_panel": left, "right_panel": left,
                       "video_labels": {"left": "A", "right": "B"}}
        pairs_list.append({
            "pair_index": index,
            "clip": name,
            "blind_mapping": mapping,
            "multiscale": {"name": "ms-0", "crop_proxy_px": 1.0},
            "l1": {"name": "budget_1", "crop_proxy_px": 1.2},
            "pairing": {"rule": "nearest"},
        })
    document = {
        "schema": P.SCHEMA,
        "ok": True,
        "output": {"dir": "review", "clips": [pair["clip"] for pair in pairs_list]},
        "pairs": pairs_list,
        "provenance": _provenance(),
    }
    review_path = (root / "review.json") if nested else (out_dir / "review.json")
    review_path.write_text(json.dumps(document), encoding="utf-8")
    return review_path, document


# --------------------------------------------------------------------------------------
# pixel panel swap and fixed labels
# --------------------------------------------------------------------------------------

def test_swap_panels_exchanges_equal_halves():
    frame = np.zeros((40, 48, 3), np.uint8)
    frame[:, :24] = 10
    frame[:, 24:] = 200
    swapped = P.swap_panels(frame)
    assert swapped.shape == frame.shape
    assert np.all(swapped[:, :24] == 200)
    assert np.all(swapped[:, 24:] == 10)
    # The original is untouched.
    assert np.all(frame[:, :24] == 10) and np.all(frame[:, 24:] == 200)
    with pytest.raises(ValueError):
        P.swap_panels(np.zeros((40, 47, 3), np.uint8))
    with pytest.raises(ValueError):
        P.swap_panels(np.zeros((40, 48), np.uint8))


def test_swap_and_relabel_repaints_fixed_a_left_b_right():
    # Simulate an already-rendered clip: A baked on the left, B baked on the right.
    left = np.full((40, 24, 3), 10, np.uint8)
    right = np.full((40, 24, 3), 200, np.uint8)
    R.draw_bitmap_label(left, "A")
    R.draw_bitmap_label(right, "B")
    frame = np.concatenate([left, right], axis=1)

    out = P.swap_and_relabel(frame)
    # Pixel content is swapped away from the label box.
    assert out[39, 4, 0] == 200 and out[39, 28, 0] == 10

    # The fixed labels are exactly A on the new left and B on the new right.
    expected_left = np.full((40, 24, 3), 200, np.uint8)
    R.draw_bitmap_label(expected_left, "A")
    expected_right = np.full((40, 24, 3), 10, np.uint8)
    R.draw_bitmap_label(expected_right, "B")
    assert np.array_equal(out[:, :24], expected_left)
    assert np.array_equal(out[:, 24:], expected_right)


def test_reversed_clip_name_is_a_plain_mp4_basename():
    assert P.reversed_clip_name("pair_00_ab.mp4") == "pair_00_ab_posrev.mp4"
    assert G._plain_mp4_basename(P.reversed_clip_name("pair_03_ab.mp4"))


# --------------------------------------------------------------------------------------
# hidden mapping reversal in the copied report
# --------------------------------------------------------------------------------------

def test_run_reversal_swaps_mapping_renames_clips_and_stays_loadable(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=2, left="l1", right="multiscale")
    source_review = review_path.read_bytes()
    source_clip = (review_path.parent / "pair_00_ab.mp4").read_bytes()
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(3, 40, 48))

    out_dir = tmp_path / "out"
    report = P.run_reversal(review_path, out_dir=out_dir, write=True)

    assert report["schema"] == P.SCHEMA
    assert report["ok"] is True
    assert report["output"]["comparisons"] == 2
    assert report["output"]["dir"] == "out"
    assert len(report["pairs"]) == 2
    for pair in report["pairs"]:
        # Original left=l1/right=multiscale becomes left=multiscale/right=l1.
        assert pair["blind_mapping"]["left_panel"] == "multiscale"
        assert pair["blind_mapping"]["right_panel"] == "l1"
        assert pair["blind_mapping"]["video_labels"] == {"left": "A", "right": "B"}
        assert pair["clip"].endswith("_posrev.mp4")
        assert "multiscale" in pair and "l1" in pair  # measurements copied

    meta = report["position_reversal"]
    assert meta["metadata_schema"] == P.METADATA_SCHEMA
    assert meta["selected_pair_indices"] == [0, 1]
    assert meta["clip_renames"] == {"pair_00_ab.mp4": "pair_00_ab_posrev.mp4",
                                    "pair_01_ab.mp4": "pair_01_ab_posrev.mp4"}
    assert meta["no_network"] is True

    # The new report is a valid gemini_blind_review input, and the visible A/B sides map back to the flipped
    # hidden methods.
    loaded = G.load_review(out_dir / "review.json")
    assert loaded["output"]["dir"] == "out"
    assert loaded["pairs"][0]["clip"] == "pair_00_ab_posrev.mp4"
    assert G.map_preference(loaded["pairs"][0]["blind_mapping"], "A") == "multiscale"
    assert G.map_preference(loaded["pairs"][0]["blind_mapping"], "B") == "l1"

    # Source report and clips are never modified.
    assert review_path.read_bytes() == source_review
    assert (review_path.parent / "pair_00_ab.mp4").read_bytes() == source_clip


# --------------------------------------------------------------------------------------
# one selected-pair end-to-end with mocked media
# --------------------------------------------------------------------------------------

def test_run_reversal_single_selected_pair_end_to_end(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=3)
    frames = _frames(4, 40, 48, left=10, right=210)
    _mock_media(monkeypatch, 48, 40, 25.0, frames)

    out_dir = tmp_path / "out"
    report = P.run_reversal(review_path, out_dir=out_dir, pair_indices=[1], write=True)

    assert report["output"]["comparisons"] == 1
    assert [pair["pair_index"] for pair in report["pairs"]] == [1]
    assert report["output"]["clips"] == ["pair_01_ab_posrev.mp4"]
    assert report["position_reversal"]["selected_pair_indices"] == [1]

    assert len(_FakeEncoder.instances) == 1
    encoder = _FakeEncoder.instances[0]
    assert encoder.closed is True
    assert (encoder.width, encoder.height) == (48, 40)
    assert encoder.fps == 25.0
    assert len(encoder.frames) == 4
    for frame in encoder.frames:
        assert frame.shape == (40, 48, 3)
        # Swapped away from the label box: left now holds the original right panel.
        assert frame[39, 4, 0] == 210
        assert frame[39, 28, 0] == 10

    # Only the selected reversed clip and report are written under out_dir.
    assert (out_dir / "pair_01_ab_posrev.mp4").parent == out_dir
    assert (out_dir / "review.json").is_file()
    assert sorted(path.name for path in out_dir.iterdir()) == ["pair_01_ab_posrev.mp4", "review.json"]


def test_run_reversal_propagates_encoder_close_failure_on_success(tmp_path, monkeypatch):
    """With no decode/write error, a failing close is the primary error and must propagate."""
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(2, 40, 48))
    _FakeEncoder.close_exc = RuntimeError("close boom")

    with pytest.raises(RuntimeError, match="close boom"):
        P.run_reversal(review_path, out_dir=tmp_path / "out")

    # Close was attempted exactly once and the failure was not swallowed.
    assert len(_FakeEncoder.instances) == 1
    assert _FakeEncoder.instances[0].closed is True
    assert _FakeEncoder.instances[0].close_attempts == 1


def test_run_reversal_preserves_write_error_when_close_also_fails(tmp_path, monkeypatch):
    """A write/decode failure is primary: close is still attempted, but its failure cannot mask the write."""
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(2, 40, 48))
    _FakeEncoder.write_exc = RuntimeError("write boom")
    _FakeEncoder.close_exc = RuntimeError("close boom")

    with pytest.raises(RuntimeError, match="write boom"):
        P.run_reversal(review_path, out_dir=tmp_path / "out")

    # The cleanup path still tried to release the encoder despite the close failure.
    assert len(_FakeEncoder.instances) == 1
    assert _FakeEncoder.instances[0].closed is True
    assert _FakeEncoder.instances[0].close_attempts == 1


def test_run_reversal_write_false_skips_clips_and_report(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(2, 40, 48))

    out_dir = tmp_path / "out"
    report = P.run_reversal(review_path, out_dir=out_dir, write=False)
    assert report["output"]["comparisons"] == 1
    assert _FakeEncoder.instances == []
    assert not out_dir.exists()


def test_run_reversal_default_out_dir_uses_review_dir_name(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)  # review.json lives in .../r/review/
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(1, 40, 48))

    report = P.run_reversal(review_path, write=False)
    assert report["output"]["dir"] == "review"
    assert P.DEFAULT_OUT_ROOT.name == "position_reversal"


# --------------------------------------------------------------------------------------
# refusals: malformed mapping/clip, selection and collisions
# --------------------------------------------------------------------------------------

def test_run_reversal_refuses_malformed_mapping(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1, bad_mapping=True)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(1, 40, 48))
    with pytest.raises(P.PositionReversalError) as exc:
        P.run_reversal(review_path, out_dir=tmp_path / "out")
    assert exc.value.code == "blind_mapping_invalid"


def test_run_reversal_refuses_bad_and_missing_clips(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "traversal", pairs=1, bad_clip="../evil.mp4")
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(1, 40, 48))
    with pytest.raises(P.PositionReversalError) as exc:
        P.run_reversal(review_path, out_dir=tmp_path / "out_a")
    assert exc.value.code == G.UNSAFE_PATH

    review_path, _ = _write_review(tmp_path / "missing", pairs=1)
    (review_path.parent / "pair_00_ab.mp4").unlink()
    with pytest.raises(P.PositionReversalError) as exc:
        P.run_reversal(review_path, out_dir=tmp_path / "out_b")
    assert exc.value.code == "clip_missing"


def test_run_reversal_refuses_unknown_pair_index(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=2)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(1, 40, 48))
    with pytest.raises(P.PositionReversalError) as exc:
        P.run_reversal(review_path, out_dir=tmp_path / "out", pair_indices=[99])
    assert exc.value.code == "unknown_pair_index"


def test_run_reversal_refuses_output_input_collisions(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", nested=True, pairs=1)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(1, 40, 48))

    review_dir = review_path.parent              # holds review.json
    clips_dir = review_dir / "review"            # holds the source clips
    for out_dir in (review_dir, clips_dir):
        with pytest.raises(P.PositionReversalError) as exc:
            P.run_reversal(review_path, out_dir=out_dir)
        assert exc.value.code == "output_collision"

    # A valid sibling out-dir still works and leaves the source untouched.
    report = P.run_reversal(review_path, out_dir=tmp_path / "reversed")
    assert report["output"]["comparisons"] == 1


def test_run_reversal_refuses_interlaced_source(tmp_path, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    _FakeEncoder.instances = []
    monkeypatch.setattr(media, "probe",
                        lambda path: {"width": 48, "height": 40, "fps": 30.0, "interlaced": True})
    monkeypatch.setattr(media, "decode", lambda *a, **k: (iter([]), 30.0, (48, 40)))
    monkeypatch.setattr(media, "Encoder", _FakeEncoder)
    with pytest.raises(P.PositionReversalError) as exc:
        P.run_reversal(review_path, out_dir=tmp_path / "out")
    assert exc.value.code == "clip_not_progressive"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_build_parser_collects_repeatable_pair_indices():
    args = P.build_parser().parse_args(["review.json", "--pair-index", "0", "--pair-index", "2",
                                        "--out-dir", "o"])
    assert args.pair_indices == [0, 2]
    assert args.out_dir == "o"


def test_main_reports_refusals_to_stderr(tmp_path, capsys, monkeypatch):
    review_path, _ = _write_review(tmp_path / "r", pairs=1)
    _mock_media(monkeypatch, 48, 40, 30.0, _frames(1, 40, 48))
    assert P.main([str(review_path), "--out-dir", str(tmp_path / "o"), "--pair-index", "5"]) == 2
    assert "unknown_pair_index" in capsys.readouterr().err
