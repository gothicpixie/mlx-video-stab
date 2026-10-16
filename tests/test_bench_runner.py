"""Focused tests for the Stage 0 benchmark runner (bench/).

All model/media work is faked: these pin configuration/experiment identity, state isolation, manifest hashing
and invalidation, atomic/JSON-safe writes, failure and cancellation reporting, cadence and support guards,
lock accounting and the diagnostics sampling policy. They do not run RAFT or touch private footage.
"""
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

from bench import cases, manifest as MF, runner

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# case definitions and configuration identity
# --------------------------------------------------------------------------------------

def test_baseline_is_historical_preset_and_variants_change_declared_variables_only():
    base = cases.get_case("baseline").config
    assert base == cases.HISTORICAL_PRESET
    expected = {
        "fp16-network": {"flow_dtype"},
        "fp16-corr": {"corr_dtype"},
        "combined-fp16": {"flow_dtype", "corr_dtype"},
        "warm-8": {"warm_iters"},
        "stride-2": {"analysis_stride"},
        "global-only": {"mesh", "rs_rows"},
        "mesh-only": {"rs_rows"},
        "row-only": {"mesh"},
    }
    for name, keys in expected.items():
        cfg = cases.get_case(name).config
        diff = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
        assert diff == keys, (name, diff)
    assert cases.get_case("baseline").eval_vf == "fps=30000/1001"
    assert cases.get_case("baseline").output_fps == "30000/1001"


def test_config_identity_is_stable_under_ordering_and_sensitive_to_values():
    a = {"flow_iters": 20, "flow_downscale": 2, "warm_start": True}
    b = {"warm_start": True, "flow_downscale": 2, "flow_iters": 20}
    assert cases.canonical_config(a) == cases.canonical_config(b)
    assert cases.config_digest(a) == cases.config_digest(b)
    assert cases.config_digest(dict(a, flow_iters=21)) != cases.config_digest(a)


def test_config_identity_normalizes_tuple_and_list_mesh():
    assert cases.config_digest({"mesh": (16, 9), "flow_iters": 20}) == cases.config_digest({"mesh": [16, 9], "flow_iters": 20})


def test_resolve_case_applies_and_validates_overrides_and_seed():
    spec = cases.resolve_case("baseline", {"flow_iters": 5, "seed": 3})
    assert spec.config["flow_iters"] == 5 and spec.config["seed"] == 3
    assert spec.config["flow_downscale"] == 2
    assert cases.config_digest(cases.resolve_case("baseline", {"seed": 3}).resolved_config()) != \
        cases.config_digest(cases.resolve_case("baseline", {"seed": 4}).resolved_config())
    with pytest.raises(KeyError):
        cases.resolve_case("baseline", {"not_a_field": 1})


def test_resolved_config_is_a_full_config():
    from mlxstab.stabilize import Config
    from dataclasses import fields
    resolved = cases.get_case("baseline").resolved_config()
    assert set(resolved) == {f.name for f in fields(Config)}


def test_residual_ablation_resolved_config_vs_baseline():
    base = cases.resolve_case("baseline").resolved_config()
    assert base["mesh"] == (16, 9) and base["rs_rows"] == 18        # baseline keeps full mesh + rows
    expected = {
        "global-only": {"mesh", "rs_rows"},
        "mesh-only": {"rs_rows"},
        "row-only": {"mesh"},
    }
    for name, keys in expected.items():
        resolved = cases.resolve_case(name).resolved_config()
        diff = {k for k in resolved if resolved[k] != base[k]}
        assert diff == keys, (name, diff)
    assert cases.resolve_case("global-only").resolved_config()["mesh"] == (0, 0)
    assert cases.resolve_case("global-only").resolved_config()["rs_rows"] == 0
    assert cases.resolve_case("mesh-only").resolved_config()["mesh"] == (16, 9)
    assert cases.resolve_case("mesh-only").resolved_config()["rs_rows"] == 0
    assert cases.resolve_case("row-only").resolved_config()["mesh"] == (0, 0)
    assert cases.resolve_case("row-only").resolved_config()["rs_rows"] == 18


def test_residual_ablation_case_names_labels_and_descriptions():
    for name in ("global-only", "mesh-only", "row-only"):
        assert name in cases.case_names
        spec = cases.get_case(name)
        assert spec.changed and cases.resolve_case(name).changed == spec.changed
        assert "refit" in spec.description.lower()
    assert cases.get_case("global-only").changed == "mesh+rs_rows"   # two channels disabled
    assert cases.get_case("mesh-only").changed == "rs_rows"         # one channel
    assert cases.get_case("row-only").changed == "mesh"             # one channel


def test_residual_ablation_config_digests_distinct_and_stable():
    names = ("baseline", "global-only", "mesh-only", "row-only")
    digests = {n: cases.config_digest(cases.resolve_case(n).resolved_config()) for n in names}
    assert len(set(digests.values())) == len(names)
    for n in names:
        assert cases.config_digest(cases.resolve_case(n).resolved_config()) == digests[n]


# --------------------------------------------------------------------------------------
# JSON safety, atomic writes and hashing/invalidation
# --------------------------------------------------------------------------------------

def test_json_safe_converts_numpy_nonfinite_and_paths():
    obj = {"a": np.float64("nan"), "b": np.float32("inf"), "c": np.int64(3),
           "d": np.array([1.0, 2.0]), "e": Path("/tmp/x"), "f": {"g": np.bool_(True)}}
    safe = MF.json_safe(obj)
    assert safe["a"] is None and safe["b"] is None and safe["c"] == 3
    assert safe["d"] == [1.0, 2.0] and safe["e"] == "/tmp/x" and safe["f"]["g"] is True
    text = json.dumps(safe, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_write_manifest_atomic_reload_and_no_temp_left(tmp_path):
    p = tmp_path / "m.json"
    MF.write_manifest(p, {"x": float("nan"), "y": 1})
    assert json.loads(p.read_text()) == {"x": None, "y": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_manifest_does_not_clobber_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "m.json"; p.write_text('{"old": true}')
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        MF.write_manifest(p, {"new": True})
    assert json.loads(p.read_text()) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []


def _identity():
    return {"source": {"sha256": "s", "window": {"start_s": 60, "duration_s": 10}, "preprocessing": {"processing_fps": 59.94}},
            "weights_sha256": "w", "config": {"flow_iters": 20, "seed": 0},
            "output_cadence": "30000/1001", "evaluation": {"flow_iters": 12, "flow_downscale": 2},
            "seed_policy": {"config_seed": 0, "synthetic_seed": 0},
            "diagnostics": {"every": 0}, "code": {"revision": "r", "files": {"mlxstab/stabilize.py": "h1"}}}


def test_content_key_invalidates_window_code_seed_eval_and_diagnostics():
    base = MF.content_key(_identity())
    for path, value in [
        (("source", "sha256"), "s2"),
        (("source", "window"), {"start_s": 800, "duration_s": 10}),
        (("source", "preprocessing"), {"processing_fps": 29.97}),
        (("weights_sha256",), "w2"),
        (("config", "flow_iters"), 24),
        (("config", "seed"), 7),
        (("output_cadence",), None),
        (("evaluation", "flow_iters"), 20),
        (("diagnostics", "every"), 15),
        (("code", "files"), {"mlxstab/stabilize.py": "h2"}),
        (("code", "revision"), "r2"),
    ]:
        ident = _identity()
        cur = ident
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
        assert MF.content_key(ident) != base, path
    assert MF.content_key(_identity()) == base


def test_experiment_identity_records_plan_capture():
    """The plan/motion-dump options are part of the experiment identity, so toggling them invalidates the key."""
    src = {"sha256": "s", "window": {}, "preprocessing": {}, "selection_note": None}
    opts = runner.RunOptions()
    off = runner._experiment_identity(opts, {"seed": 0}, src, "w", {}, {}, None, None, 0, False)
    on = runner._experiment_identity(opts, {"seed": 0}, src, "w", {}, {}, None, None, 0, True)
    assert off["capture"] == {"dump_plan": False, "dump_motion": False}
    assert on["capture"] == {"dump_plan": True, "dump_motion": False}
    assert MF.content_key(off) != MF.content_key(on)
    motion = runner._experiment_identity(opts, {"seed": 0}, src, "w", {}, {}, None, None, 0, False, True)
    assert motion["capture"] == {"dump_plan": False, "dump_motion": True}
    assert MF.content_key(off) != MF.content_key(motion)


def test_code_fingerprint_hashes_relevant_code_not_docs():
    fp = MF.code_fingerprint(ROOT)
    assert "mlxstab/stabilize.py" in fp["files"]
    assert "bench/runner.py" in fp["files"]
    assert not any(f.startswith("README") or f.startswith("marketing") for f in fp["files"])


# --------------------------------------------------------------------------------------
# cadence and reliability guards
# --------------------------------------------------------------------------------------

def _tl(first, last, frames, kind="encoded_pts"):
    return {"kind": kind, "frames": frames, "first_s": first, "last_s": last}


def test_check_cadence_requires_and_validates_timelines():
    fps = 30000 / 1001
    n = 300
    last = (n - 1) / fps
    inp = {"frames": n, "fps": fps, "duration_s": 10.01, "timeline": _tl(0.0, last, n, "frame_index_normalized")}
    out = {"frames": n, "fps": fps, "duration_s": 10.01, "timeline": _tl(0.0, last, n, "encoded_pts")}
    mask = {"frames": n, "fps": fps, "duration_s": 10.01, "timeline": _tl(0.0, last, n, "encoded_pts")}
    ok = MF.check_cadence(inp, out, mask=mask)
    assert ok["aligned"] and ok["reasons"] == []
    assert ok["timeline_kinds"]["input"] == "frame_index_normalized"
    # equal frames/fps/duration but a shifted encoded timeline must fail
    shifted = {**out, "timeline": _tl(1.0, 1.0 + last, n)}
    bad = MF.check_cadence(inp, shifted, mask=mask)
    assert not bad["aligned"] and any("timeline first_s" in r for r in bad["reasons"])
    # an extra mask tail is detected even though the evaluation loop consumed only a prefix
    extra = {"frames": n + 1, "fps": fps, "duration_s": 10.01, "timeline": _tl(0.0, 1.0 + last, n + 1)}
    bad_mask = MF.check_cadence(inp, out, mask=extra)
    assert not bad_mask["aligned"] and any("mask frame count" in r for r in bad_mask["reasons"])
    # missing / reversed / non-finite timing must not be aligned
    assert not MF.check_cadence(inp, {**out, "timeline": None}, mask=mask)["aligned"]
    assert not MF.check_cadence(inp, {**out, "timeline": _tl(5.0, 1.0, n)}, mask=mask)["aligned"]
    assert not MF.check_cadence(inp, {**out, "timeline": _tl(0.0, float("nan"), n)}, mask=mask)["aligned"]
    # count / duration checks still apply
    assert not MF.check_cadence(inp, {**out, "frames": 299})["aligned"]
    assert not MF.check_cadence({**inp, "duration_s": float("nan")}, out)["aligned"]


def _metric(**kw):
    base = {"stability": 0.9, "jitter_px": 0.1, "wobble_px": 0.01, "support_frames": 100,
            "inlier_frac_mean": 0.9, "fallback_frames": 0}
    base.update(kw)
    return base


def test_reliability_requires_support_and_finite_metrics():
    cad_ok = {"aligned": True, "reasons": []}
    clean = MF.reliability(_metric(), _metric(), cad_ok)
    assert clean["reliable"] and clean["reasons"] == []
    zero_support = MF.reliability(_metric(inlier_frac_mean=0.0), _metric(), cad_ok)
    assert not zero_support["reliable"]                              # was inverted before
    empty = MF.reliability(_metric(support_frames=0), _metric(), cad_ok)
    assert not empty["reliable"]
    missing = MF.reliability({"stability": 1.0}, _metric(), cad_ok)
    assert not missing["reliable"] and any("support_frames missing" in r for r in missing["reasons"])
    nonfinite = MF.reliability(_metric(jitter_px=float("nan")), _metric(), cad_ok)
    assert not nonfinite["reliable"]
    none_metrics = MF.reliability(None, _metric(), cad_ok)
    assert not none_metrics["reliable"]
    fb = MF.reliability(_metric(fallback_frames=2), _metric(), cad_ok)
    assert not fb["reliable"]
    bad = MF.reliability(_metric(), _metric(), {"aligned": False, "reasons": ["mismatch"]})
    assert not bad["reliable"] and "mismatch" in bad["reasons"]


# --------------------------------------------------------------------------------------
# state isolation and evaluation clip reset lifecycle
# --------------------------------------------------------------------------------------

class _FakeFlow:
    def __init__(self, *a, **k):
        self.resets = 0

    def reset(self):
        self.resets += 1


def test_evaluate_clip_resets_flow_each_clip_and_reports_normalized_timeline(monkeypatch):
    monkeypatch.setattr(runner.media, "probe", lambda p: {"fps": 30.0, "width": 4, "height": 4})
    monkeypatch.setattr(runner.media, "decode",
                        lambda *a, **k: (iter([np.zeros((4, 4, 3), np.uint8)] * 3), 30.0, (4, 4)))

    def fake_measure(frames, flow, W, H, fps, masks=None, on_step=None):
        n = sum(1 for _ in frames)
        mk = list(masks) if masks is not None else []
        return {"stability": 1.0, "jitter_px": 0.0, "wobble_px": 0.0, "fallback_frames": 0,
                "support_frames": n, "inlier_frac_mean": 0.9, "_masks": len(mk)}

    monkeypatch.setattr(runner.metrics, "measure", fake_measure)
    flow = _FakeFlow()
    m1, meta1 = runner.evaluate_clip("a.mp4", {}, None, None, None, flow)
    m2, meta2 = runner.evaluate_clip("b.mp4", {}, None, None, None, flow, mask_path="m.mp4")
    assert flow.resets == 2
    assert meta1["frames"] == 3 and meta1 == meta2
    assert meta1["timeline"] == {"kind": "frame_index_normalized", "frames": 3, "first_s": 0.0, "last_s": 2 / 30.0}
    assert m2["_masks"] == 3


def test_evaluate_clip_tracks_enabled_attaches_independent_metrics(monkeypatch):
    """`--eval-tracks` measures the independent KLT estimator in the same single decode pass (opt-in)."""
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    frames = [np.ascontiguousarray(base[50:146, 50 + t:50 + t + 96]) for t in range(6)]
    monkeypatch.setattr(runner.media, "probe", lambda p: {"fps": 30.0, "width": 96, "height": 96})
    monkeypatch.setattr(runner.media, "decode", lambda *a, **k: (iter(frames), 30.0, (96, 96)))

    def fake_measure(fs, flow, W, H, fps, masks=None, on_step=None):
        n = sum(1 for _ in fs)
        return {"stability": 1.0, "jitter_px": 0.0, "wobble_px": 0.0, "fallback_frames": 0, "support_frames": n}
    monkeypatch.setattr(runner.metrics, "measure", fake_measure)

    m, _ = runner.evaluate_clip("a.mp4", {}, None, None, None, _FakeFlow(), tracks_enabled=True)
    assert m["tracks"]["method"] == "klt_similarity"
    assert m["tracks"]["n_frames"] == 6 and m["tracks"]["support_frames"] == 5
    assert m["tracks"]["fallback_frames"] == 0

    m2, _ = runner.evaluate_clip("a.mp4", {}, None, None, None, _FakeFlow())
    assert "tracks" not in m2                              # opt-in only


def test_evaluate_clip_tracks_consume_the_same_invalid_mask_stream(monkeypatch):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    frames = [np.ascontiguousarray(base[50:146, 50 + t:50 + t + 96]) for t in range(5)]
    invalid = [np.full((96, 96, 3), 255, np.uint8) for _ in frames]
    monkeypatch.setattr(runner.media, "probe", lambda p: {"fps": 30.0, "width": 96, "height": 96})
    monkeypatch.setattr(runner.media, "decode",
                        lambda p, *a, **k: (iter(invalid if p == "mask.mp4" else frames), 30.0, (96, 96)))

    def fake_measure(fs, flow, W, H, fps, masks=None, on_step=None):
        n = 0
        for _ in fs:
            if masks is not None:
                next(masks)
            n += 1
        return {"stability": 1.0, "jitter_px": 0.0, "wobble_px": 0.0,
                "fallback_frames": 0, "support_frames": n - 1}
    monkeypatch.setattr(runner.metrics, "measure", fake_measure)

    m, _ = runner.evaluate_clip("a.mp4", {}, None, None, None, _FakeFlow(),
                                mask_path="mask.mp4", tracks_enabled=True)
    assert m["tracks"]["fallback_frames"] == 4
    assert m["tracks"]["jitter_px"] is None
    assert "failed pair" in m["tracks"]["unavailable_reason"]


def test_run_case_records_independent_evaluator_agreement_when_enabled(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    monkeypatch.setattr(runner, "stabilize", lambda *a, **k: {
        "frames": 3, "decoded_frames": 3, "rendered_frames": 3,
        "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
        "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2})

    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(tmp_path / "tr"), lock="off", seed=17, command=["run_case"],
                                            eval_tracks=True))
    assert man["status"] == "success", man.get("error")
    assert man["evaluation"]["independent_tracks"]["enabled"] is True
    assert man["evaluation"]["independent_tracks"]["params"]["seed"] == 17
    assert man["metrics"]["input"]["tracks"]["method"] == "klt_similarity"
    ag = man["metrics"]["evaluator_agreement"]
    assert ag["reliable"] is False and ag["raft_jitter_ratio"] is None   # 4x4 blank frames: no support
    assert "estimators" in ag["note"]


def test_run_case_omits_tracks_by_default(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    monkeypatch.setattr(runner, "stabilize", lambda *a, **k: {
        "frames": 3, "decoded_frames": 3, "rendered_frames": 3,
        "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
        "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2})
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(tmp_path / "notr"), lock="off", command=["run_case"]))
    assert man["status"] == "success"
    assert man["evaluation"]["independent_tracks"]["enabled"] is False
    assert "tracks" not in man["metrics"]["input"]
    assert "evaluator_agreement" not in man["metrics"]


# --------------------------------------------------------------------------------------
# lock accounting
# --------------------------------------------------------------------------------------

class _FakeCM:
    def __init__(self): self.entered = 0
    def __enter__(self): self.entered += 1; return self
    def __exit__(self, *a): return False


def test_heavylock_chunk_pause_accounting(monkeypatch):
    from bench import heavylock
    cm = _FakeCM()
    fake_mod = types.SimpleNamespace(LOCK="/tmp/fake.lock", heavy=lambda what, poll: cm)
    monkeypatch.setattr(heavylock, "_load_film_heavy", lambda: (fake_mod, Path("/tmp")))
    slept = []
    lock = heavylock.HeavyLock("test", enabled=True, chunk=2, pause_s=0.05, sleep=lambda s: slept.append(s))
    with lock:
        lock.on_step(0)          # not a multiple
        lock.on_step(2)          # release, pause, re-acquire
        lock.on_step(4)
    assert lock.releases == 3     # three releases: two chunk yields + final exit
    assert slept == [0.05, 0.05]
    assert lock.acquire_wait_s >= 0 and lock.pause_total_s >= 0
    assert cm.entered == 3        # acquire + two re-acquires
    st = lock.status()
    assert st["releases"] == 3 and st["pause_total_s"] >= 0 and st["acquire_wait_s"] >= 0


# --------------------------------------------------------------------------------------
# run_case: failure, cancellation, validation and a full success manifest with fakes
# --------------------------------------------------------------------------------------

def _patch_provenance(monkeypatch):
    monkeypatch.setattr(MF, "sha256_file", lambda p, *a, **k: "deadbeef" * 8)
    monkeypatch.setattr(MF, "code_fingerprint", lambda root: {"revision": "abc123", "dirty_relevant": True,
                                                              "relevant_status": [], "files": {"mlxstab/stabilize.py": "h"}})
    monkeypatch.setattr(MF, "env_versions", lambda: {"python": "3.11", "mlx": "0.32"})
    monkeypatch.setattr(MF, "hardware_info", lambda: {"chip": "Apple M5", "cpu_count": 10})
    monkeypatch.setattr(MF, "metrics_digest", lambda root: {"version": "mlxstab.metrics/1+00000000", "files": {}})
    monkeypatch.setattr(MF, "peak_memory", lambda: {"process_peak_rss_bytes": 123, "mlx_peak_active_bytes": 456,
                                                    "available": True, "unavailable_reason": None})


def _patch_media(monkeypatch, source_fps=30.0, output_fps=30000 / 1001, frames=3):
    from fractions import Fraction

    def probe(p):
        encoded = Path(p).name.startswith(("stabilized_r", "mask_r"))
        fps = output_fps if encoded else source_fps
        return {"fps": fps, "width": 4, "height": 4, "interlaced": False, "duration": frames / fps}

    def decode(p, info=None, start=None, duration=None, vf=None):
        rate = probe(p)["fps"]
        fps = float(Fraction(vf[4:])) if vf else rate
        n = round(frames * fps / rate)
        return iter([np.zeros((4, 4, 3), np.uint8)] * n), fps, (4, 4)

    def pts(p):
        times = [i / probe(p)["fps"] for i in range(frames)]
        return {"count": frames, "n_pts": frames, "pts_s": times, "first_pts_s": times[0],
                "last_pts_s": times[-1], "monotonic": True}

    monkeypatch.setattr(runner.media, "probe", probe)
    monkeypatch.setattr(runner.media, "decode", decode)
    monkeypatch.setattr(runner.media, "frame_pts", pts)

    def measure(frames, flow, W, H, fps, masks=None, on_step=None):
        n = sum(1 for _ in frames)
        if masks is not None:
            list(masks)
        return {"stability": 1.0, "jitter_px": 0.0, "wobble_px": 0.0, "fallback_frames": 0,
                "support_frames": n, "inlier_frac_mean": 0.9}
    monkeypatch.setattr(runner.metrics, "measure", measure)


def test_run_case_failure_writes_valid_manifest(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner.media, "probe", lambda p: {"fps": 30.0, "width": 4, "height": 4,
                                                          "interlaced": False, "duration": 1.0})
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)

    def boom(*a, **k):
        raise RuntimeError("stabilizer exploded")
    monkeypatch.setattr(runner, "stabilize", boom)

    run_dir = tmp_path / "run"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", command=["run_case"]))
    assert man["status"] == "failure"
    assert man["error"]["stage"] == "stabilize_r0" and man["error"]["type"] == "RuntimeError"
    on_disk = json.loads((run_dir / "manifest.json").read_text())
    assert on_disk["status"] == "failure" and on_disk["unavailable"]


def test_run_case_cancellation_writes_manifest_and_reraises(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner.media, "probe", lambda p: {"fps": 30.0, "width": 4, "height": 4,
                                                          "interlaced": False, "duration": 1.0})
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)

    def interrupt(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, "stabilize", interrupt)

    run_dir = tmp_path / "cancel"
    with pytest.raises(KeyboardInterrupt):
        runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                          out_dir=str(run_dir), lock="off", command=["run_case"]))
    on_disk = json.loads((run_dir / "manifest.json").read_text())
    assert on_disk["status"] == "cancelled" and on_disk["error"]["type"] == "KeyboardInterrupt"


def test_run_case_validates_options_and_reports_failure(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    run_dir = tmp_path / "bad"
    man = runner.run_case(runner.RunOptions(case="baseline", source="synthetic", out_dir=str(run_dir),
                                            lock="off", repeats=0, command=["run_case"]))
    assert man["status"] == "failure" and man["error"]["stage"] == "validate"
    assert "repeats" in man["error"]["message"]
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "failure"


def test_default_run_dir_handles_duration_without_start():
    p = runner.default_run_dir("baseline", "/tmp/clip.MTS", None, 1.0)
    assert "tnone+1" in p.name


def test_invalid_cadence_is_rejected_before_starting_ffmpeg():
    for rate in ("0", "-30", "nan", "1/0"):
        errors = runner.validate_options(runner.RunOptions(output_fps=rate))
        assert any("output-fps" in error for error in errors)
    for vf in ("fps=0", "fps=nan", "scale=32:24", "fps=30,scale=32:24"):
        errors = runner.validate_options(runner.RunOptions(eval_vf=vf))
        assert any("eval-vf" in error for error in errors)


def test_run_case_success_manifest_has_required_sections(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)

    calls = []

    def fake_stabilize(*a, **k):
        calls.append(k.get("diag_callback"))
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3, "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)

    run_dir = tmp_path / "run2"
    man = runner.run_case(runner.RunOptions(case="fp16-network", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", repeats=2, command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    for key in ("schema", "run_id", "case", "source", "weights", "config", "config_digest", "experiment_identity",
                "content_key", "code", "environment", "hardware", "seed_policy", "evaluation", "command",
                "timing", "runtime_s", "memory", "metrics", "diagnostics", "outputs", "repeats", "lock"):
        assert key in man, key
    assert man["source"]["sha256"] and man["weights"]["sha256"] and man["content_key"]
    assert man["config"]["flow_dtype"] == "float16"
    assert [r["model_state"] for r in man["repeats"]] == ["cold-load", "cached"]
    assert [r["temporal_state"] for r in man["repeats"]] == ["reset (first pair cold)"] * 2
    assert man["repeats"][0]["video"] != man["repeats"][1]["video"]       # distinct evidence paths
    assert man["repeats"][0]["rendered_frames"] == 3 and man["repeats"][0]["encoded_frames"] == 3
    assert man["outputs"]["encoded_frames"] == 3 and man["outputs"]["encoded_pts"]["count"] == 3
    assert man["metrics"]["cadence"]["aligned"] is True
    assert man["metrics"]["reliability"]["reliable"] is True
    assert man["diagnostics"]["mode"] == "off"
    assert "mlx_peak_active_bytes" in man["memory"] and "scope" in man["memory"]
    assert man["timing"]["end_to_end_stabilization_s"] >= man["repeats"][0]["wall_s"]
    text = (run_dir / "manifest.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def _run_case_with_stage_spies(tmp_path, monkeypatch, spy_attr, predicate=None):
    """Run a fully faked success case while a spy records the on-disk manifest stage.

    ``spy_attr`` is the runner attribute to wrap (``evaluate_clip`` or ``media.probe``); the spy appends the
    stage currently persisted in ``manifest.json`` each time the target runs (optionally only when ``predicate``
    accepts the call), then delegates to the real object.
    """
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    monkeypatch.setattr(runner, "stabilize", lambda *a, **k: {
        "frames": 3, "decoded_frames": 3, "rendered_frames": 3,
        "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
        "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2})

    run_dir = tmp_path / "stages"
    man_path = run_dir / "manifest.json"
    stages = []
    target = runner
    for part in spy_attr.split(".")[:-1]:
        target = getattr(target, part)
    attr = spy_attr.split(".")[-1]
    real = getattr(target, attr)

    def spy(*a, **k):
        if predicate is None or predicate(*a, **k):
            stages.append(json.loads(man_path.read_text())["stage"])
        return real(*a, **k)
    monkeypatch.setattr(target, attr, spy)

    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", command=["run_case"]))
    return man, man_path, stages


def test_run_case_persists_evaluate_stage_before_evaluate_clip_called(tmp_path, monkeypatch):
    man, man_path, stages = _run_case_with_stage_spies(tmp_path, monkeypatch, "evaluate_clip")
    assert man["status"] == "success", man.get("error")
    # Both evaluate_clip calls observed an on-disk manifest already marked evaluate; the run still finishes.
    assert stages == ["evaluate", "evaluate"]
    on_disk = json.loads(man_path.read_text())
    assert on_disk["status"] == "success"


def test_run_case_persists_probe_output_stage_before_probing_outputs(tmp_path, monkeypatch):
    encoded_video = lambda p: Path(p).name.startswith("stabilized_r")
    man, man_path, stages = _run_case_with_stage_spies(tmp_path, monkeypatch, "media.probe",
                                                       predicate=encoded_video)
    assert man["status"] == "success", man.get("error")
    # probe_output is persisted before the encoded video is probed; the manifest still ends success.
    assert stages == ["probe_output"]
    on_disk = json.loads(man_path.read_text())
    assert on_disk["status"] == "success"


def test_run_case_persists_model_construct_and_repeat_stages_before_long_calls(tmp_path, monkeypatch):
    """The on-disk stage must already name the phase when RAFT loads or a repeat's stabilize begins."""
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    run_dir = tmp_path / "phases"
    man_path = run_dir / "manifest.json"

    construct_stages = []

    def raft_spy(*a, **k):
        # RaftFlow is constructed once for stabilization and once during evaluation.
        construct_stages.append(json.loads(man_path.read_text())["stage"])
        return _FakeFlow(*a, **k)

    stabilize_stages = []

    def stabilize_spy(*a, **k):
        stabilize_stages.append(json.loads(man_path.read_text())["stage"])
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3,
                "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0,
                "observed_nondiag_render_s": 0.2}

    monkeypatch.setattr(runner, "RaftFlow", raft_spy)
    monkeypatch.setattr(runner, "stabilize", stabilize_spy)

    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", repeats=2, command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    # At RAFT construction time the manifest already says model_construct (not lock).
    assert construct_stages[0] == "model_construct"
    # Each repeat's long stabilize call is entered with that repeat's stage already persisted.
    assert stabilize_stages == ["stabilize_r0", "stabilize_r1"]
    assert json.loads(man_path.read_text())["status"] == "success"


def test_native_encoding_and_lower_rate_evaluation_use_separate_clocks(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch, source_fps=60.0, output_fps=60.0, frames=6)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    weights = tmp_path / "w.safetensors"; weights.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    monkeypatch.setattr(runner, "stabilize", lambda *a, **k: {
        "rendered_frames": 6, "timing_s": {"analysis": 1., "plan": .1, "render": .2}})
    man = runner.run_case(runner.RunOptions(source=str(src), weights=str(weights),
        output_fps="native", eval_vf="fps=30", out_dir=str(tmp_path / "run"), lock="off", log=None))
    assert man["status"] == "success", man.get("error")
    assert man["outputs"]["encoded_frames"] == 6
    assert man["metrics"]["output"]["frames"] == 3
    assert man["metrics"]["cadence"]["mask"]["frames"] == 3
    assert man["metrics"]["cadence"]["aligned"], man["metrics"]["cadence"]["reasons"]
    assert man["outputs"]["timeline_alignment"]["aligned"]


def test_encoded_validation_checks_interior_pts_and_mask_tail():
    good = {"count": 4, "n_pts": 4, "pts_s": [0., .1, .2, .3]}
    assert MF.check_encoded_cfr(good, good, 10., 10.)["aligned"]
    # Equal endpoints/counts can still conceal a duplicate or misplaced interior frame.
    for times in ([0., .2, .1, .3], [0., .1, .1, .3], [0., .11, .2, .3], [0., None, .2, .3]):
        assert not MF.check_encoded_cfr({**good, "pts_s": times}, good, 10., 10.)["aligned"]
    tail = {"count": 5, "n_pts": 5, "pts_s": [0., .1, .2, .3, .4]}
    assert not MF.check_encoded_cfr(good, tail, 10., 10.)["aligned"]
    assert not MF.check_encoded_cfr(good, good, 10., 10., requested_fps=30.)["aligned"]


def test_cadence_rejects_one_frame_shift_and_nonpositive_metadata():
    base = {"frames": 3, "fps": 30., "duration_s": .1,
            "timeline": _tl(0., 2 / 30, 3, "frame_index_normalized")}
    shift = {**base, "timeline": _tl(1 / 30, 3 / 30, 3)}
    assert not MF.check_cadence(base, shift)["aligned"]
    for key in ("frames", "fps", "duration_s"):
        assert not MF.check_cadence({**base, key: 0}, {**base, key: 0})["aligned"]


def test_run_case_sampled_diagnostics_use_actual_plan_frames(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)

    def fake_stabilize(*a, **k):
        cb = k.get("diag_callback")
        every = k.get("diag_every")
        assert every == 2
        for t in (0, 2, 4):                       # exactly the frames the runner asked to sample
            cb(t, 0.01, 0.5, {"shape.det_negative_frac": 0.0})
        return {"frames": 5, "decoded_frames": 5, "rendered_frames": 5, "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 1.6},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 1.5, "observed_nondiag_render_s": 0.1}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)

    run_dir = tmp_path / "diag"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w), out_dir=str(run_dir),
                                            lock="off", diag_every=2, command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    assert man["diagnostics"]["mode"] == "sampled" and man["diagnostics"]["frames"] == [0, 2, 4]
    assert man["diagnostics"]["every"] == 2
    assert len(man["diagnostics"]["per_frame"]) == 3
    assert man["diagnostics"]["aggregate"] is not None
    assert man["diagnostics"]["total_overhead_s"] == pytest.approx(1.5)
    assert "not whole-video guarantees" in man["diagnostics"]["note"]


# --------------------------------------------------------------------------------------
# CLI paths (list, dry-run parity)
# --------------------------------------------------------------------------------------

def test_cli_forwards_dump_plan_option(tmp_path, monkeypatch, capsys):
    from bench import run_case as cli
    captured = {}

    def fake_run_case(opts):
        captured["opts"] = opts
        return {"status": "success", "case": {"name": opts.case}, "outputs": {"manifest": "m"}, "error": None}
    monkeypatch.setattr(cli.runner, "run_case", fake_run_case)
    assert cli.main(["--dump-plan", "--source", "synthetic", "--lock", "off", "--out-dir", str(tmp_path / "d")]) == 0
    assert captured["opts"].dump_plan is True
    assert cli.main(["--source", "synthetic", "--lock", "off", "--out-dir", str(tmp_path / "d2")]) == 0
    assert captured["opts"].dump_plan is False


def test_cli_forwards_eval_tracks_option(tmp_path, monkeypatch, capsys):
    from bench import run_case as cli
    captured = {}

    def fake_run_case(opts):
        captured["opts"] = opts
        return {"status": "success", "case": {"name": opts.case}, "outputs": {"manifest": "m"}, "error": None}
    monkeypatch.setattr(cli.runner, "run_case", fake_run_case)
    assert cli.main(["--eval-tracks", "--source", "synthetic", "--lock", "off", "--out-dir", str(tmp_path / "et")]) == 0
    assert captured["opts"].eval_tracks is True
    assert cli.main(["--source", "synthetic", "--lock", "off", "--out-dir", str(tmp_path / "et2")]) == 0
    assert captured["opts"].eval_tracks is False


def test_cli_forwards_dump_motion_option(tmp_path, monkeypatch, capsys):
    from bench import run_case as cli
    captured = {}

    def fake_run_case(opts):
        captured["opts"] = opts
        return {"status": "success", "case": {"name": opts.case}, "outputs": {"manifest": "m"}, "error": None}
    monkeypatch.setattr(cli.runner, "run_case", fake_run_case)
    assert cli.main(["--dump-motion", "--source", "synthetic", "--lock", "off", "--out-dir", str(tmp_path / "dm")]) == 0
    assert captured["opts"].dump_motion is True
    assert cli.main(["--source", "synthetic", "--lock", "off", "--out-dir", str(tmp_path / "dm2")]) == 0
    assert captured["opts"].dump_motion is False


def test_cli_list_cases_and_dry_run_parity(capsys):
    from bench import run_case as cli
    assert cli.main(["--list-cases"]) == 0
    listed = capsys.readouterr().out
    assert "baseline" in listed and "fp16-corr" in listed and "combined-fp16" in listed

    assert cli.main(["--dry-run", "--case", "baseline", "--seed", "0"]) == 0
    d = json.loads(capsys.readouterr().out)
    resolved = cases.resolve_case("baseline", {"seed": 0}).resolved_config()
    assert d["config_digest"] == cases.config_digest(resolved)     # dry-run hashes the fully resolved config
    assert d["resolved_config"]["seed"] == 0
    assert d["output_fps"] == "30000/1001" and d["eval_vf"] == "fps=30000/1001"
    assert d["resolved_config"]["flow_downscale"] == 2


def test_cli_lists_and_dry_runs_residual_ablation_cases(capsys):
    from bench import run_case as cli
    assert cli.main(["--list-cases"]) == 0
    listed = capsys.readouterr().out
    for name in ("global-only", "mesh-only", "row-only"):
        assert name in listed
    expected = {"global-only": ([0, 0], 0), "mesh-only": ([16, 9], 0), "row-only": ([0, 0], 18)}
    for name, (mesh, rs) in expected.items():
        assert cli.main(["--dry-run", "--case", name, "--seed", "0"]) == 0
        d = json.loads(capsys.readouterr().out)
        resolved = cases.resolve_case(name, {"seed": 0}).resolved_config()
        assert d["config_digest"] == cases.config_digest(resolved)
        assert d["resolved_config"]["mesh"] == mesh and d["resolved_config"]["rs_rows"] == rs
        assert d["output_fps"] == "30000/1001" and d["eval_vf"] == "fps=30000/1001"


# --------------------------------------------------------------------------------------
# review-2 contract tests: change labels, run-dir uniqueness/overwrite, CLI failures,
# distinct rendered/encoded counts, diag-mode conflict
# --------------------------------------------------------------------------------------

def test_resolve_case_change_label_ignores_noop_seed():
    assert cases.resolve_case("baseline", {"seed": 0}).changed == ""
    assert cases.resolve_case("baseline", {"flow_iters": 7}).changed == "flow_iters"
    assert cases.resolve_case("fp16-network", {"seed": 0}).changed == "flow_dtype"
    assert cases.resolve_case("combined-fp16").changed == "flow_dtype+corr_dtype"
    assert cases.resolve_case("combined-fp16", {"seed": 0}).changed == "corr_dtype,flow_dtype"
    assert cases.resolve_case("global-only", {"seed": 0}).changed == "mesh,rs_rows"
    assert cases.resolve_case("mesh-only", {"seed": 0}).changed == "rs_rows"
    assert cases.resolve_case("row-only", {"seed": 0}).changed == "mesh"


def test_default_run_dir_is_unique_within_a_second():
    a = runner.default_run_dir("baseline", "/tmp/clip.MTS", 60, 10)
    b = runner.default_run_dir("baseline", "/tmp/clip.MTS", 60, 10)
    assert a != b


def test_run_case_refuses_to_overwrite_existing_evidence(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    run_dir = tmp_path / "existing"; run_dir.mkdir(); (run_dir / "manifest.json").write_text('{"old": true}')
    with pytest.raises(FileExistsError):
        runner.run_case(runner.RunOptions(case="baseline", source="synthetic", out_dir=str(run_dir), lock="off"))
    assert json.loads((run_dir / "manifest.json").read_text()) == {"old": True}


def test_cli_missing_source_structured_failure_not_typeerror(tmp_path, capsys):
    from bench import run_case as cli
    rc = cli.main(["--source", "/does/not/exist.MTS", "--out-dir", str(tmp_path / "missing"), "--lock", "off"])
    assert rc == 1
    d = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert d["status"] == "failure" and d["case"] == "baseline"
    assert d["manifest"] and Path(d["manifest"]).exists()          # failure manifest written, no TypeError
    on_disk = json.loads(Path(d["manifest"]).read_text())
    assert on_disk["status"] == "failure" and on_disk["error"]["stage"] == "validate"


def test_cli_overwrite_refusal_is_structured(tmp_path, capsys):
    from bench import run_case as cli
    run_dir = tmp_path / "existing"; run_dir.mkdir(); (run_dir / "manifest.json").write_text("{}")
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    rc = cli.main(["--source", str(src), "--out-dir", str(run_dir), "--lock", "off"])
    assert rc == 1
    d = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert d["status"] == "failure" and d["error"]["type"] == "FileExistsError"
    assert json.loads((run_dir / "manifest.json").read_text()) == {}


def test_run_case_keeps_rendered_and_encoded_counts_distinct(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)

    def fake_stabilize(*a, **k):
        return {"frames": 6, "decoded_frames": 6, "rendered_frames": 6, "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 60.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(tmp_path / "counts"), lock="off"))
    assert man["repeats"][0]["rendered_frames"] == 6
    assert man["repeats"][0]["encoded_frames"] == 3
    assert man["outputs"]["encoded_frames"] == 3


def test_run_case_dump_plan_captures_only_retained_repeat(tmp_path, monkeypatch):
    """With --dump-plan the runner forwards a run-dir plan path only for repeat 0 and records it in outputs."""
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    seen = []

    def fake_stabilize(*a, **k):
        seen.append(k.get("dump_plan"))
        if k.get("dump_plan"):
            Path(k["dump_plan"]).write_bytes(b"npz")
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3,
                "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)

    run_dir = tmp_path / "dump"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", repeats=3,
                                            dump_plan=True, command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    assert seen == [str(run_dir / "plan_r0.npz"), None, None]           # warm repeats never capture the plan
    assert man["outputs"]["plan"] == str(run_dir / "plan_r0.npz")
    assert man["outputs"]["retained_repeat"] == 0
    assert man["experiment_identity"]["capture"] == {"dump_plan": True, "dump_motion": False}
    assert (run_dir / "plan_r0.npz").read_bytes() == b"npz"


def test_run_case_without_dump_plan_records_no_plan(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    captured = {}

    def fake_stabilize(*a, **k):
        captured.update(k)
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3,
                "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)
    run_dir = tmp_path / "nodump"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    assert captured["dump_plan"] is None and man["outputs"]["plan"] is None
    assert not (run_dir / "plan_r0.npz").exists()
    assert man["experiment_identity"]["capture"] == {"dump_plan": False, "dump_motion": False}


def test_run_case_dump_motion_captures_only_retained_repeat(tmp_path, monkeypatch):
    """With --dump-motion the runner forwards a run-dir motion path only for repeat 0 and records it in outputs."""
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    seen = []

    def fake_stabilize(*a, **k):
        seen.append(k.get("dump_motion_npz"))
        if k.get("dump_motion_npz"):
            Path(k["dump_motion_npz"]).write_bytes(b"npz")
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3,
                "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)

    run_dir = tmp_path / "dump_motion"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", repeats=3,
                                            dump_motion=True, command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    assert seen == [str(run_dir / "motion_r0.npz"), None, None]        # warm repeats never capture the motion
    assert man["outputs"]["motion"] == str(run_dir / "motion_r0.npz")
    assert man["outputs"]["plan"] is None
    assert man["outputs"]["retained_repeat"] == 0
    assert man["experiment_identity"]["capture"] == {"dump_plan": False, "dump_motion": True}
    assert (run_dir / "motion_r0.npz").read_bytes() == b"npz"


def test_run_case_without_dump_motion_records_no_motion(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    captured = {}

    def fake_stabilize(*a, **k):
        captured.update(k)
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3,
                "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)
    run_dir = tmp_path / "nodump_motion"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    assert captured["dump_motion_npz"] is None and man["outputs"]["motion"] is None
    assert not (run_dir / "motion_r0.npz").exists()
    assert man["experiment_identity"]["capture"] == {"dump_plan": False, "dump_motion": False}


def test_run_case_refuses_to_overwrite_existing_motion_dump(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    run_dir = tmp_path / "existing_motion"; run_dir.mkdir()
    (run_dir / "motion_r0.npz").write_bytes(b"old")
    with pytest.raises(FileExistsError, match="motion dump"):
        runner.run_case(runner.RunOptions(case="baseline", source="synthetic", out_dir=str(run_dir),
                                          lock="off", dump_motion=True))
    assert (run_dir / "motion_r0.npz").read_bytes() == b"old"          # untouched


def test_run_case_dump_plan_without_dump_motion_keeps_plan_only_contract(tmp_path, monkeypatch):
    """The plan-only capture contract is unchanged when dump_motion is false."""
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    seen = []

    def fake_stabilize(*a, **k):
        seen.append((k.get("dump_plan"), k.get("dump_motion_npz")))
        if k.get("dump_plan"):
            Path(k["dump_plan"]).write_bytes(b"npz")
        return {"frames": 3, "decoded_frames": 3, "rendered_frames": 3,
                "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
                "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2}
    monkeypatch.setattr(runner, "stabilize", fake_stabilize)
    run_dir = tmp_path / "plan_only"
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(run_dir), lock="off", dump_plan=True, command=["run_case"]))
    assert man["status"] == "success", man.get("error")
    assert seen == [(str(run_dir / "plan_r0.npz"), None)]
    assert man["outputs"]["plan"] == str(run_dir / "plan_r0.npz") and man["outputs"]["motion"] is None
    assert man["experiment_identity"]["capture"] == {"dump_plan": True, "dump_motion": False}


def test_run_case_refuses_to_overwrite_existing_plan_dump(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    run_dir = tmp_path / "existing"; run_dir.mkdir()
    (run_dir / "plan_r0.npz").write_bytes(b"old")
    with pytest.raises(FileExistsError, match="plan dump"):
        runner.run_case(runner.RunOptions(case="baseline", source="synthetic", out_dir=str(run_dir),
                                          lock="off", dump_plan=True))
    assert (run_dir / "plan_r0.npz").read_bytes() == b"old"          # untouched


def test_run_case_rejects_warp_diag_plus_sampled_diagnostics(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch)
    man = runner.run_case(runner.RunOptions(case="baseline", source="synthetic", out_dir=str(tmp_path / "both"),
                                            lock="off", diag_every=5, overrides={"warp_diag": True}))
    assert man["status"] == "failure" and man["error"]["stage"] == "validate"
    assert "not both" in man["error"]["message"]


def test_run_case_records_encoded_pts_timeline(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    monkeypatch.setattr(runner, "stabilize", lambda *a, **k: {
        "frames": 3, "decoded_frames": 3, "rendered_frames": 3, "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
        "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2})
    man = runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                            out_dir=str(tmp_path / "pts"), lock="off"))
    assert man["metrics"]["output"]["timeline"]["kind"] == "frame_index_normalized"
    assert man["metrics"]["input"]["timeline"]["kind"] == "frame_index_normalized"
    assert man["metrics"]["cadence"]["mask"] is not None
    assert man["outputs"]["encoded_pts"]["pts_s"] == [i / (30000 / 1001) for i in range(3)]
    assert man["outputs"]["timeline_alignment"]["aligned"]


def test_run_case_surfaces_initial_write_failure(tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(path, man):
        calls["n"] += 1
        raise OSError("disk full at initial")

    monkeypatch.setattr(MF, "write_manifest", boom)
    with pytest.raises(OSError):
        runner.run_case(runner.RunOptions(case="baseline", source="synthetic",
                                          out_dir=str(tmp_path / "initfail"), lock="off"))
    assert calls["n"] == 1                              # surfaced before any expensive work


def test_run_case_surfaces_final_write_failure(tmp_path, monkeypatch):
    _patch_provenance(monkeypatch); _patch_media(monkeypatch)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    w = tmp_path / "w.safetensors"; w.write_bytes(b"w")
    monkeypatch.setattr(runner, "RaftFlow", _FakeFlow)
    monkeypatch.setattr(runner, "stabilize", lambda *a, **k: {
        "frames": 3, "decoded_frames": 3, "rendered_frames": 3, "timing_s": {"analysis": 1.0, "plan": 0.1, "render": 0.2},
        "fps": 30.0, "fallback_frames": 0, "diagnostics_total_s": 0.0, "observed_nondiag_render_s": 0.2})
    real = MF.write_manifest

    def maybe_fail(path, manifest):
        if manifest.get("status") == "success":
            raise OSError("disk full at final")
        return real(path, manifest)

    monkeypatch.setattr(MF, "write_manifest", maybe_fail)
    with pytest.raises(OSError):
        runner.run_case(runner.RunOptions(case="baseline", source=str(src), weights=str(w),
                                          out_dir=str(tmp_path / "finalfail"), lock="off"))
