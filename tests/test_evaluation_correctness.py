"""Regression tests for the Stage 0 evaluation-correctness work package.

Three defects are pinned down here:

1. `RaftFlow` kept bidirectional warm state and cached frame features across sequence
   boundaries; only the legacy `prev` dict was cleared. `reset()` must make the next
   pair cold (no warm start, both frames re-encoded).
2. `exclude_masked` sampled *both* the source and the destination invalid masks at the
   source position. The destination mask belongs at `pts + flow`, and fractional target
   positions must be sampled conservatively.
3. `analyze` skipped mask exclusion entirely when the target mask was absent, silently
   ignoring an available source mask.

The flow-model fixtures are intentionally fake and cheap: these tests assert lifecycle
and coordinate behaviour, not RAFT numerics.
"""
import runpy
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

from mlxstab import media, metrics, motion as MO, stabilize as S

ROOT = Path(__file__).resolve().parent.parent


def _obs(pts, flow, valid=None):
    pts = np.asarray(pts, np.float64)
    flow = np.asarray(flow, np.float64)
    if valid is None:
        valid = np.ones(len(pts), bool)
    return MO.Obs(pts, flow, np.asarray(valid, bool), 64, 48)


# --------------------------------------------------------------------------------------
# mask exclusion: source vs destination coordinates
# --------------------------------------------------------------------------------------

def test_target_mask_is_sampled_at_destination_not_source():
    """An invalid destination pixel must reject the correspondence even when the source is clean."""
    H, W = 48, 64
    obs = _obs([[10, 10]], [[20, 0]])                      # destination (30, 10)
    m_src = np.zeros((H, W), bool)
    m_dst = np.zeros((H, W), bool); m_dst[10, 30] = True   # wrong under the old source sampling
    MO.exclude_masked(obs, m_src, m_dst, pad=0)
    assert not obs.valid[0]


def test_target_mask_at_source_position_is_ignored():
    """Regression counterexample: a target pixel coinciding with the source position must not matter."""
    H, W = 48, 64
    obs = _obs([[10, 10]], [[20, 0]])                      # destination (30, 10)
    m_src = np.zeros((H, W), bool)
    m_dst = np.zeros((H, W), bool); m_dst[10, 10] = True   # only the source position is invalid in m_dst
    MO.exclude_masked(obs, m_src, m_dst, pad=0)
    assert obs.valid[0]                                    # old code rejected this


def test_source_mask_is_sampled_at_source_position():
    H, W = 48, 64
    obs = _obs([[10, 10]], [[20, 0]])
    m_src = np.zeros((H, W), bool); m_src[10, 10] = True
    MO.exclude_masked(obs, m_src, np.zeros((H, W), bool), pad=0)
    assert not obs.valid[0]


@pytest.mark.parametrize("flow,invalid_xy,expected", [
    # horizontal-only fractional offset: dst = (30.5, 10); taps x in {30, 31}, y = 10
    ((20.5, 0.0), (30, 10), False),
    ((20.5, 0.0), (31, 10), False),
    ((20.5, 0.0), (10, 10), True),     # source position, outside the target support
    ((20.5, 0.0), (32, 10), True),
    # vertical-only fractional offset: dst = (30, 10.5); taps x = 30, y in {10, 11}
    ((20.0, 0.5), (30, 10), False),
    ((20.0, 0.5), (30, 11), False),
    ((20.0, 0.5), (30, 12), True),
    # diagonal fractional offset: dst = (30.5, 10.5); all four taps reject
    ((20.5, 0.5), (30, 10), False),
    ((20.5, 0.5), (31, 10), False),
    ((20.5, 0.5), (30, 11), False),
    ((20.5, 0.5), (31, 11), False),
    ((20.5, 0.5), (32, 11), True),
])
def test_fractional_target_sampling_uses_bilinear_support(flow, invalid_xy, expected):
    """A single invalid bilinear tap rejects the correspondence; taps outside it do not."""
    H, W = 48, 64
    obs = _obs([[10, 10]], [flow])
    m_dst = np.zeros((H, W), bool); m_dst[invalid_xy[1], invalid_xy[0]] = True
    MO.exclude_masked(obs, np.zeros((H, W), bool), m_dst, pad=0)
    assert bool(obs.valid[0]) is expected


def test_exact_integer_target_is_not_rejected_by_invalid_neighbours():
    """At an exact pixel centre only one bilinear tap has weight; adjacent invalid pixels must not leak in."""
    H, W = 48, 64
    obs = _obs([[10, 10]], [[20.0, 0.0]])                  # dst = (30, 10) exactly
    m_dst = np.zeros((H, W), bool)
    m_dst[10, 31] = m_dst[11, 30] = m_dst[11, 31] = True   # all neighbours invalid, centre valid
    MO.exclude_masked(obs, np.zeros((H, W), bool), m_dst, pad=0)
    assert obs.valid[0]


def test_exact_valid_edge_coordinate_is_not_rejected():
    """A valid pixel centre at the image corner has a single tap and must survive."""
    H, W = 48, 64
    obs = _obs([[W - 1, H - 1]], [[0, 0]])
    MO.exclude_masked(obs, np.zeros((H, W), bool), np.zeros((H, W), bool), pad=0)
    assert obs.valid[0]


def test_invalid_mask_dilation_reaches_nearby_but_not_distant_points():
    H, W = 48, 64
    m = np.zeros((H, W), bool); m[20, 10] = True           # 10 px above the sample at (10, 10)
    near = _obs([[10, 10]], [[0, 0]])
    MO.exclude_masked(near, m, None, pad=12)
    assert not near.valid[0]
    far = _obs([[10, 10]], [[0, 0]])
    MO.exclude_masked(far, m, None, pad=5)
    assert far.valid[0]


def test_out_of_bounds_and_nonfinite_positions_are_rejected():
    H, W = 48, 64
    zero = np.zeros((H, W), bool)

    oob_src = _obs([[W, 10]], [[0, 0]])                    # x == W is outside [0, W-1]
    MO.exclude_masked(oob_src, zero, None, pad=0)
    assert not oob_src.valid[0]

    oob_dst = _obs([[W - 1, 10]], [[5.0, 0]])              # destination leaves the frame
    MO.exclude_masked(oob_dst, None, zero, pad=0)
    assert not oob_dst.valid[0]

    nan_src = _obs([[np.nan, 10]], [[0, 0]])
    MO.exclude_masked(nan_src, zero, None, pad=0)
    assert not nan_src.valid[0]

    nan_dst = _obs([[10, 10]], [[np.nan, 0]])
    MO.exclude_masked(nan_dst, None, zero, pad=0)
    assert not nan_dst.valid[0]


def test_optional_masks_are_independent_and_valid_is_never_reactivated():
    H, W = 48, 64
    zero = np.zeros((H, W), bool)
    one = np.ones((H, W), bool)

    # A source mask without a target mask still applies.
    only_src = _obs([[10, 10]], [[20, 0]])
    MO.exclude_masked(only_src, one.copy(), None, pad=0)
    assert not only_src.valid[0]

    # A target mask without a source mask still applies.
    only_dst = _obs([[10, 10]], [[20, 0]])
    MO.exclude_masked(only_dst, None, one.copy(), pad=0)
    assert not only_dst.valid[0]

    # Both absent is a no-op.
    none_obs = _obs([[10, 10]], [[20, 0]])
    MO.exclude_masked(none_obs, None, None, pad=0)
    assert none_obs.valid[0]

    # Already-invalid observations stay invalid even when both masks are clean.
    keep = _obs([[10, 10]], [[0, 0]], valid=[False])
    MO.exclude_masked(keep, zero, zero, pad=0)
    assert not keep.valid[0]

    # And a clean mask never reactivates an invalid point while invalidating another.
    mix = _obs([[10, 10], [20, 20]], [[0, 0], [0, 0]], valid=[False, True])
    MO.exclude_masked(mix, zero, zero, pad=0)
    assert list(mix.valid) == [False, True]


# --------------------------------------------------------------------------------------
# analyze caller: apply an available source mask even when the target mask is missing
# --------------------------------------------------------------------------------------

def test_analyze_applies_source_mask_when_target_mask_absent(monkeypatch):
    H = W = 32
    seen = []
    pts = np.array([[1.0, 1.0]], np.float64)

    def fake_observe(F, B, step):
        o = MO.Obs(pts.copy(), np.zeros((1, 2)), np.ones(1, bool), W, H)
        seen.append(o)
        return o

    monkeypatch.setattr(MO, "observe", fake_observe)
    monkeypatch.setattr(MO, "fit_motion", lambda *a, **k: object())

    class FakeFlow:
        def bidir(self, a, b):
            return mx.zeros((H, W, 2)), mx.zeros((H, W, 2))

    m0 = np.zeros((H, W), bool); m0[1, 1] = True
    masks = iter([m0, None, None])                         # source frame masked, target frames not
    cfg = S.Config(mesh=(0, 0), rs_rows=0)
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(3)]

    S.analyze(frames, FakeFlow(), cfg, W, H, 30.0, masks=masks, threads=False)
    assert seen[0].valid.tolist() == [False]               # old `if cm is not None` skipped this
    assert seen[1].valid.tolist() == [True]                # no source mask left for the second pair


# --------------------------------------------------------------------------------------
# RaftFlow.reset lifecycle (fake RAFT)
# --------------------------------------------------------------------------------------

class _FakeFeat:
    def __init__(self, shape):
        self.shape = shape
        self.fmap = mx.zeros((1, max(shape[0] // 8, 1), max(shape[1] // 8, 1), 1))


class _FakeRaft:
    """Records encoder reuse and warm-start initialisation without running RAFT."""

    def __init__(self):
        self.frame_features_calls = 0
        self.flow_calls = []

    def frame_features(self, x):
        self.frame_features_calls += 1
        return _FakeFeat(tuple(x.shape[1:3]))

    def flow_bidir(self, fa, fb, iters, init_f, init_b, corr_dtype):
        self.flow_calls.append({"iters": iters, "warm": init_f is not None and init_b is not None})
        h, w = fa.shape
        low = mx.zeros((1, max(h // 8, 1), max(w // 8, 1), 2))
        up = mx.zeros((1, h, w, 2))
        return low, up, low, up

    def __call__(self, a, b, iters, flow_init=None, corr_dtype=mx.float32):
        self.flow_calls.append({"iters": iters, "warm": flow_init is not None})
        h, w = a.shape[1:3]
        return mx.zeros((1, max(h // 8, 1), max(w // 8, 1), 2)), mx.zeros((1, h, w, 2))


def _fake_flow(monkeypatch, fake):
    monkeypatch.setattr(S, "load_raft", lambda weights: fake)
    return S.RaftFlow("unused", iters=6, downscale=1, warm=True, warm_iters=3)


def test_bidir_reuses_features_and_warm_starts_inside_a_sequence(monkeypatch):
    fake = _FakeRaft()
    flow = _fake_flow(monkeypatch, fake)
    f = [np.zeros((16, 24, 3), np.uint8) for _ in range(3)]

    flow.bidir(f[0], f[1])
    assert fake.frame_features_calls == 2                  # both frames encoded, cold
    assert fake.flow_calls[-1] == {"iters": 6, "warm": False}

    flow.bidir(f[1], f[2])
    assert fake.frame_features_calls == 3                  # only the new frame is encoded
    assert fake.flow_calls[-1] == {"iters": 3, "warm": True}   # warm_iters used


def test_reset_starts_next_sequence_cold_and_refeatures(monkeypatch):
    fake = _FakeRaft()
    flow = _fake_flow(monkeypatch, fake)
    f = [np.zeros((16, 24, 3), np.uint8) for _ in range(4)]

    flow.bidir(f[0], f[1])
    flow.bidir(f[1], f[2])                                 # _last holds f[2]; warm state set
    calls, features = len(fake.flow_calls), fake.frame_features_calls

    flow.reset()
    flow.bidir(f[2], f[3])                                 # same object f[2] as the cached first frame
    assert len(fake.flow_calls) == calls + 1
    assert fake.flow_calls[-1] == {"iters": 6, "warm": False}
    assert fake.frame_features_calls == features + 2       # identity cache bypassed: both re-encoded


def test_reset_clears_legacy_warm_cache_and_diagnostic_counter(monkeypatch):
    fake = _FakeRaft()
    flow = _fake_flow(monkeypatch, fake)
    f = [np.zeros((16, 24, 3), np.uint8) for _ in range(3)]

    flow(f[0], f[1])                                        # legacy path increments `syncs`
    assert flow.prev and flow.syncs == 1
    flow.reset()
    assert flow.prev == {} and flow.prev_f is None and flow.prev_b is None and flow._last is None
    assert flow.syncs == 0


# --------------------------------------------------------------------------------------
# evaluator integration: bench/evaluate.py must reset between its input and output clips
# --------------------------------------------------------------------------------------

def test_evaluate_resets_flow_at_clip_boundaries(monkeypatch):
    """Execute bench/evaluate.py's real input/output loop with a fake RAFT and no media files.

    Each clip's first pair must be cold (no warm init, both frames encoded); later pairs stay warm and
    reuse features. Reverting the evaluator to `flow.prev.clear()` leaves `prev_f`/`prev_b` set, so the
    output clip's first pair would be warm and this test would fail even though `RaftFlow.reset()` remains
    implemented.
    """
    fake = _FakeRaft()
    monkeypatch.setattr(S, "load_raft", lambda weights: fake)

    H, W, T = 16, 24, 4
    clip = {"n": 0}

    def fake_probe(path):
        return {"path": path}

    def fake_decode(path, info, start=None, duration=None, vf=None):
        clip["n"] += 1
        frames = [np.full((H, W, 3), 40 * clip["n"], np.uint8) for _ in range(T)]
        return frames, 30.0, (W, H)

    monkeypatch.setattr(media, "probe", fake_probe)
    monkeypatch.setattr(media, "decode", fake_decode)

    per_clip = []

    def fake_measure(frames, flow, width, height, fps, masks=None, keep_frames=False, on_step=None):
        recs = []; prev = None
        for fr in frames:
            if prev is not None:
                before = fake.frame_features_calls
                flow.bidir(prev, fr)
                recs.append({"warm": fake.flow_calls[-1]["warm"],
                             "feat": fake.frame_features_calls - before})
            prev = fr
        per_clip.append(recs)
        return {"stability": 1.0, "jitter_px": 0.0, "wobble_px": 0.0, "fallback_frames": 0}

    monkeypatch.setattr(metrics, "measure", fake_measure)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "input.mp4", "output.mp4"])

    runpy.run_path(str(ROOT / "bench" / "evaluate.py"), run_name="evaluate_under_test")

    assert len(per_clip) == 2                              # input and output clips both measured
    for recs in per_clip:
        assert recs[0] == {"warm": False, "feat": 2}       # cold first pair of every clip
        assert recs[1] == {"warm": True, "feat": 1}        # warm start + feature reuse inside the clip
