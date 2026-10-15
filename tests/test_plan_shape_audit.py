"""Focused tests for the captured-plan geometry audit (``bench/plan_shape_audit.py``).

Covers: NPZ payload validation (schema/keys/shapes/dims/finiteness/no-pickle), Plan reconstruction, the four
output-space probes through production ``source_coords`` (straight homography control, bowed mesh/row residual
detection), source-bounds coverage accounting, sparse-vs-full ``source_coords`` parity, frame selection that
always includes the final frame, JSON-safe reports and the CLI. No media, FFmpeg, or changed production
defaults.
"""
import copy
import json

import numpy as np
import pytest
import mlx.core as mx

from mlxstab import stabilize as S
from bench import plan_shape_audit as P
from bench import shape_audit as SA

W, H, T = 64, 48, 3


# --------------------------------------------------------------------------------------
# synthetic plans and NPZ writers
# --------------------------------------------------------------------------------------

def make_plan(*, G=None, mesh=None, row=None, t=T, w=W, h=H):
    if G is None:
        G = [np.eye(3) for _ in range(t)]
    if mesh is None:
        mesh = np.zeros((t, 1, 1, 2))
    if row is None:
        row = np.zeros((t, 1, 2))
    return S.Plan(G=[np.array(g, np.float64) for g in G], mesh_corr=np.asarray(mesh, np.float64),
                  row_corr=np.asarray(row, np.float64), raw_quads=np.zeros((t, 4, 2)),
                  path_quads=np.zeros((t, 4, 2)))


def save_real_plan(tmp_path, plan, *, w=W, h=H, fps=30.0, name="plan.npz"):
    path = tmp_path / name
    S.save_plan(plan, str(path), w, h, fps)          # the real production writer
    return path


def base_payload(t=T, w=W, h=H):
    return {
        "schema": np.asarray(S.PLAN_SCHEMA), "width": np.int64(w), "height": np.int64(h),
        "frames": np.int64(t), "fps": np.float64(30.0), "G": np.tile(np.eye(3), (t, 1, 1)),
        "mesh_corr": np.zeros((t, 1, 1, 2)), "row_corr": np.zeros((t, 1, 2)),
        "raw_quads": np.zeros((t, 4, 2)), "path_quads": np.zeros((t, 4, 2)), "fallbacks": np.int64(0),
        "sampled_border_scale": np.float64(1.0), "sampled_border_worst_frac": np.float64(0.0),
        "sampled_border_measured": np.bool_(False),
    }


def write_payload(tmp_path, payload, name="payload.npz"):
    path = tmp_path / name
    np.savez_compressed(path, **payload)
    return path


def seg_metrics(plan, t, w=W, h=H, samples=201):
    segs = P.output_segments(w, h)
    return {name: P.evaluate_segment(plan, t, w, h, segs[name], samples) for name in P.SEGMENT_ORDER}


def devs(metrics):
    return {name: metrics[name]["mapped"]["dev_max_px"] for name in P.SEGMENT_ORDER}


# --------------------------------------------------------------------------------------
# loading / validation
# --------------------------------------------------------------------------------------

def test_load_plan_roundtrip_reconstructs_plan(tmp_path):
    gc, gr, nr = 4, 3, 5
    mesh = np.arange(T * (gr + 1) * (gc + 1) * 2, dtype=np.float64).reshape(T, gr + 1, gc + 1, 2) * 0.0
    row = np.zeros((T, nr, 2))
    plan = make_plan(mesh=mesh, row=row)
    path = save_real_plan(tmp_path, plan)
    loaded, meta = P.load_plan_npz(path)
    assert isinstance(loaded, S.Plan) and len(loaded.G) == T
    np.testing.assert_array_equal(np.asarray(loaded.G), np.asarray(plan.G))
    np.testing.assert_array_equal(loaded.mesh_corr, plan.mesh_corr)
    np.testing.assert_array_equal(loaded.row_corr, plan.row_corr)
    assert (meta["width"], meta["height"], meta["frames"], meta["fps"]) == (W, H, T, 30.0)
    assert meta["schema"] == S.PLAN_SCHEMA
    assert meta["mesh_cells"] == [gc, gr] and meta["row_bands"] == nr
    # A plan built without the backoff records the disabled defaults (scale 1, no probe).
    assert loaded.sampled_border_scale == 1.0 and loaded.sampled_border_worst_frac == 0.0
    assert loaded.sampled_border_measured is False
    assert meta["sampled_border_measured"] is False


def test_legacy_plan_v1_loads_with_border_defaults(tmp_path):
    """A schema-/1 payload (no sampled border fields) stays loadable read-only with disabled defaults."""
    p = base_payload()
    p["schema"] = np.asarray(S.PLAN_SCHEMA_V1)
    for k in P.BORDER_KEYS:
        del p[k]
    loaded, meta = P.load_plan_npz(write_payload(tmp_path, p, name="legacy.npz"))
    assert meta["schema"] == S.PLAN_SCHEMA_V1
    assert loaded.sampled_border_scale == 1.0 and loaded.sampled_border_worst_frac == 0.0
    assert loaded.sampled_border_measured is False


def test_current_plan_requires_border_keys(tmp_path):
    """The current schema must carry all three sampled border fields; their absence is malformed."""
    p = base_payload()
    del p["sampled_border_scale"]
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p, name="missing_border.npz"))
    assert ei.value.code == "missing_keys" and "sampled_border_scale" in str(ei.value)


@pytest.mark.parametrize("key,val", [("sampled_border_scale", np.float64(2.0)),
                                     ("sampled_border_worst_frac", np.float64(-0.1))])
def test_rejects_border_fields_out_of_range(tmp_path, key, val):
    p = base_payload(); p[key] = val
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p, name="border_range.npz"))
    assert ei.value.code == "bad_range"


def test_rejects_wrong_schema(tmp_path):
    p = base_payload(); p["schema"] = np.asarray("mlxstab.plan/999")
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p))
    assert ei.value.code == "bad_schema" and "does not match" in str(ei.value)


def test_rejects_missing_key(tmp_path):
    p = base_payload(); del p["mesh_corr"]
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p))
    assert ei.value.code == "missing_keys" and "mesh_corr" in str(ei.value)


@pytest.mark.parametrize("key,bad", [
    ("G", np.zeros((T, 2, 3))),
    ("mesh_corr", np.zeros((T, 3, 2))),
    ("row_corr", np.zeros((T, 2))),
    ("raw_quads", np.zeros((T, 3, 2))),
    ("path_quads", np.zeros((T, 4, 3))),
])
def test_rejects_bad_shapes(tmp_path, key, bad):
    p = base_payload(); p[key] = bad
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p))
    assert ei.value.code == "bad_shape"


def test_rejects_mesh_zero_row_cells_positive_column_cells(tmp_path):
    # gr=0 with gc>0 would make production source_coords divide by H/gr (ZeroDivisionError); reject it.
    p = base_payload()
    p["mesh_corr"] = np.zeros((T, 1, 5, 2))          # gr=0, gc=4
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p, name="zero_rows.npz"))
    assert ei.value.code == "bad_shape"
    assert "gr" in str(ei.value) and "row cells" in str(ei.value)
    # The same column count with one row cell (gr=1, gc=4) is a valid mesh and must still load.
    p["mesh_corr"] = np.zeros((T, 2, 5, 2))
    loaded, meta = P.load_plan_npz(write_payload(tmp_path, p, name="one_row.npz"))
    assert isinstance(loaded, S.Plan) and meta["mesh_cells"] == [4, 1]


@pytest.mark.parametrize("key", ["G", "mesh_corr", "row_corr", "raw_quads", "path_quads"])
def test_rejects_non_finite_arrays(tmp_path, key):
    p = base_payload(); p[key] = p[key].astype(np.float64).copy(); p[key].reshape(-1)[0] = np.nan
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p))
    assert ei.value.code == "nonfinite"


@pytest.mark.parametrize("key,val,code", [
    ("width", np.int64(0), "bad_dims"),
    ("height", np.int64(1), "bad_dims"),
    ("frames", np.int64(0), "bad_dims"),
    ("fps", np.float64(0.0), "bad_scalar"),
    ("fps", np.float64(np.inf), "bad_scalar"),
])
def test_rejects_bad_dimensions_and_scalars(tmp_path, key, val, code):
    p = base_payload(); p[key] = val
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p))
    assert ei.value.code == code


def test_rejects_object_array_without_unpickling(tmp_path):
    p = base_payload()
    obj = np.empty((T, 3, 3), dtype=object)
    obj[:] = np.eye(3)
    p["G"] = obj
    with pytest.raises(P.PlanPayloadError) as ei:
        P.load_plan_npz(write_payload(tmp_path, p))
    assert ei.value.code == "unreadable"


# --------------------------------------------------------------------------------------
# straight control / bend detection
# --------------------------------------------------------------------------------------

def test_global_homography_maps_all_segments_straight(tmp_path):
    Pm = SA.projective(2.5e-3, -1.5e-3)
    plan = make_plan(G=[Pm.copy() for _ in range(T)])
    loaded, _ = P.load_plan_npz(save_real_plan(tmp_path, plan))
    m = seg_metrics(loaded, 1)
    for name, dev in devs(m).items():
        assert m[name]["mapped"]["ok"] and dev < 1e-3, (name, dev)
        assert m[name]["coverage"]["frac"] == pytest.approx(1.0)


def test_bowed_mesh_bends_horizontal_but_not_vertical(tmp_path):
    gc, gr = 16, 12
    mesh = np.zeros((T, gr + 1, gc + 1, 2))
    xs = np.linspace(0.0, 1.0, gc + 1)
    mesh[:, :, :, 1] = (6.0 * np.sin(np.pi * xs))[None, None, :]     # vertical bow across columns
    loaded, _ = P.load_plan_npz(save_real_plan(tmp_path, make_plan(mesh=mesh)))
    m = seg_metrics(loaded, 1)
    assert m["horizontal"]["mapped"]["ok"] and m["horizontal"]["mapped"]["dev_max_px"] > 1.0
    assert m["vertical"]["mapped"]["dev_max_px"] < 1e-3             # a purely vertical shift stays collinear


def test_row_residual_bends_vertical(tmp_path):
    nr = 18
    row = np.zeros((T, nr, 2))
    ys = np.linspace(0.0, 1.0, nr)
    row[:, :, 0] = (6.0 * np.sin(np.pi * ys))[None, :]              # horizontal row shift vs y bends vertical lines
    loaded, _ = P.load_plan_npz(save_real_plan(tmp_path, make_plan(row=row)))
    m = seg_metrics(loaded, 1)
    assert m["vertical"]["mapped"]["ok"] and m["vertical"]["mapped"]["dev_max_px"] > 1.0
    assert m["horizontal"]["mapped"]["dev_max_px"] < 1e-3


def test_coverage_accounting_in_and_out_of_bounds(tmp_path):
    inside, _ = P.load_plan_npz(save_real_plan(tmp_path, make_plan(), name="in.npz"))
    m = seg_metrics(inside, 0)
    for name in P.SEGMENT_ORDER:
        c = m[name]["coverage"]
        assert c["n"] == 201 and c["in_bounds"] == 201 and c["frac"] == pytest.approx(1.0)
        assert c["fully_in_bounds"] is True

    Gt = np.eye(3); Gt[0, 2] = 3 * W
    outside, _ = P.load_plan_npz(save_real_plan(tmp_path, make_plan(G=[Gt.copy() for _ in range(T)]), name="out.npz"))
    m = seg_metrics(outside, 0)
    for name in P.SEGMENT_ORDER:
        c = m[name]["coverage"]
        assert c["in_bounds"] == 0 and c["frac"] == pytest.approx(0.0) and c["fully_in_bounds"] is False
        assert c["x_min"] > W                                             # mapped x is off to the right


# --------------------------------------------------------------------------------------
# sparse vs full grid parity
# --------------------------------------------------------------------------------------

def test_sparse_source_coords_matches_values_from_full_grid(tmp_path):
    w, h, t = 24, 18, 1
    gc, gr, nr = 6, 4, 6
    rng = np.random.default_rng(0)
    plan = make_plan(G=[SA.projective(1.5e-3, -1.0e-3)], mesh=rng.normal(0, 1, (t, gr + 1, gc + 1, 2)),
                     row=rng.normal(0, 1, (t, nr, 2)), t=t, w=w, h=h)
    loaded, _ = P.load_plan_npz(save_real_plan(tmp_path, plan, w=w, h=h))

    full = np.array(S.source_coords(loaded, 0, w, h)[0], np.float64)          # [H, W, 2]
    assert full.shape == (h, w, 2)
    coords = [(0, 0), (2, 3), (5, 7), (11, 9), (23, 17)]
    xs = np.array([c[0] for c in coords], np.float32)
    ys = np.array([c[1] for c in coords], np.float32)
    sparse = np.array(S.source_coords(loaded, 0, w, h, (mx.array(xs)[None], mx.array(ys)[None]))[0, 0], np.float64)
    assert sparse.shape == (len(coords), 2)
    expected = np.array([full[y, x] for x, y in coords])
    np.testing.assert_allclose(sparse, expected, atol=1e-5, rtol=0.0)
    # The corrections genuinely move the points (parity is not trivially identity).
    assert np.abs(sparse - np.array(coords, np.float64)).max() > 0.1


# --------------------------------------------------------------------------------------
# frame selection
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("t,every,expected", [
    (65, 30, [0, 30, 60, 64]),
    (60, 30, [0, 30, 59]),
    (90, 30, [0, 30, 60, 89]),
    (1, 30, [0]),
    (5, 1, [0, 1, 2, 3, 4]),
])
def test_frame_indices_always_include_final_frame(t, every, expected):
    got = P.frame_indices(t, every)
    assert got == expected and got[-1] == t - 1


def test_frame_indices_rejects_bad_stride():
    with pytest.raises(ValueError):
        P.frame_indices(10, 0)


# --------------------------------------------------------------------------------------
# report / CLI
# --------------------------------------------------------------------------------------

def test_report_is_json_safe_and_records_exact_frames(tmp_path):
    path = save_real_plan(tmp_path, make_plan(G=[SA.projective(1.0e-3, -0.5e-3) for _ in range(T)]))
    rep = P.run_plan_audit(path, every=10, samples=101)
    json.dumps(P.json_safe(rep), allow_nan=False)
    assert rep["schema"] == P.SCHEMA and rep["ok"] is True
    assert rep["selection"]["frames"] == [0, 2] and rep["selection"]["final_frame_included"] is True
    assert rep["plan"]["frames"] == T
    assert set(rep["segments"]) == set(P.SEGMENT_ORDER)
    for pf in rep["per_frame"]:
        for name in P.SEGMENT_ORDER:
            assert pf["segments"][name]["input_line"]["p0"] == rep["segments"][name]["p0"]
            assert pf["segments"][name]["coverage"]["n"] == 101
    assert "not" in rep["scope"].lower() and "perceptual" in rep["scope"].lower()


def test_cli_writes_json_with_final_frame(tmp_path):
    plan = make_plan(t=65)
    path = save_real_plan(tmp_path, plan, name="long.npz")
    out = tmp_path / "audit.json"
    assert P.main([str(path), "--every", "30", "--samples", "101", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == P.SCHEMA and rep["selection"]["frames"] == [0, 30, 60, 64]


def test_cli_prints_json_to_stdout(tmp_path, capsys):
    path = save_real_plan(tmp_path, make_plan())
    assert P.main([str(path), "--every", "2", "--samples", "51"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["ok"] is True and rep["selection"]["frames"] == [0, 2]


def test_cli_emits_json_safe_error_for_bad_payload(tmp_path, capsys):
    p = base_payload(); p["schema"] = np.asarray("wrong")
    path = write_payload(tmp_path, p)
    assert P.main([str(path)]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False and err["error"]["code"] == "bad_schema"
    json.dumps(err, allow_nan=False)


def test_cli_error_written_to_out_file(tmp_path):
    path = write_payload(tmp_path, {}, name="empty.npz")
    out = tmp_path / "err.json"
    assert P.main([str(path), "--out", str(out)]) == 2
    err = json.loads(out.read_text())
    assert err["ok"] is False and err["error"]["code"] == "missing_keys"


def test_run_plan_audit_rejects_bad_selection(tmp_path):
    path = save_real_plan(tmp_path, make_plan())
    with pytest.raises(ValueError):
        P.run_plan_audit(path, every=0)
    with pytest.raises(ValueError):
        P.run_plan_audit(path, samples=1)


# --------------------------------------------------------------------------------------
# matched multi-plan comparison
# --------------------------------------------------------------------------------------

def write_manifest(run_dir, *, case, source_sha="s" * 64, window=None, preprocessing=None,
                   weights_sha="w" * 64, config=None, plan_name="plan_r0.npz", dump_plan=True,
                   outputs_plan=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "schema": "mlxstab.bench.manifest/1",
        "run_id": run_dir.name,
        "case": {"name": case},
        "source": {"sha256": source_sha,
                   "window": window if window is not None else {"start_s": 0.0, "duration_s": 1.0},
                   "preprocessing": preprocessing if preprocessing is not None else {"interlaced": False}},
        "weights": {"sha256": weights_sha},
        "config": copy.deepcopy(BASELINE_CONFIG) if config is None else config,
        "outputs": {"plan": outputs_plan if outputs_plan is not None else str(run_dir / plan_name)},
        "experiment_identity": {"capture": {"dump_plan": dump_plan}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return run_dir


def make_case(root, case, *, run_name=None, plan=None, name="plan_r0.npz", w=W, h=H, fps=30.0,
              **manifest_kwargs):
    """A plan dump plus its matching sibling manifest inside ``root/<run_name>``."""
    run_dir = root / (run_name or case)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = save_real_plan(run_dir, plan if plan is not None else make_plan(), w=w, h=h, fps=fps, name=name)
    write_manifest(run_dir, case=case, plan_name=name, **manifest_kwargs)
    return path


def bow_mesh(t=T, gc=16, gr=12):
    mesh = np.zeros((t, gr + 1, gc + 1, 2))
    mesh[:, :, :, 1] = (6.0 * np.sin(np.pi * np.linspace(0.0, 1.0, gc + 1)))[None, None, :]
    return mesh


# Shared resolved stabilization Config used by every comparison manifest. Tests vary only the intended
# residual-channel ablation keys (mesh, rs_rows) unless they are explicitly checking a refusal.
BASELINE_CONFIG = {
    "smooth": "l1", "sigma": 15.0, "l1_weights": [10.0, 1.0, 100.0], "budget": 0.04,
    "mesh": [16, 9], "mesh_budget": 0.012, "rs_rows": 18, "flow_iters": 20,
    "flow_downscale": 2, "warm_start": True, "step": 8, "kernel": "lanczos3",
    "zoom": 1.0, "seed": 0, "analysis_stride": 1, "flow_dtype": "float32",
    "corr_dtype": "float32", "warm_iters": 0, "model": "similarity", "fidelity": 0.1,
    "warp_diag": False,
}


def row_bend(t=T, nr=18):
    row = np.zeros((t, nr, 2))
    row[:, :, 0] = (6.0 * np.sin(np.pi * np.linspace(0.0, 1.0, nr)))[None, :]
    return row


def test_compare_matched_baseline_global_mesh_row_manifests(tmp_path):
    base = make_case(tmp_path, "baseline", plan=make_plan())
    glob = make_case(tmp_path, "global_only", plan=make_plan(G=[SA.projective(2.5e-3, -1.5e-3) for _ in range(T)]),
                     config={**BASELINE_CONFIG, "mesh": [0, 0], "rs_rows": 0})
    mesh = make_case(tmp_path, "mesh_only", plan=make_plan(mesh=bow_mesh()),
                     config={**BASELINE_CONFIG, "rs_rows": 0})
    row = make_case(tmp_path, "row_only", plan=make_plan(row=row_bend()),
                    config={**BASELINE_CONFIG, "mesh": [0, 0]})

    rep = P.compare_plans([base, glob, mesh, row], every=2, samples=101)
    json.dumps(P.json_safe(rep), allow_nan=False)
    assert rep["schema"] == P.COMPARISON_SCHEMA and rep["ok"] is True
    assert rep["reference"] == "baseline"
    assert [c["case"] for c in rep["cases"]] == ["baseline", "global_only", "mesh_only", "row_only"]
    assert [c["is_reference"] for c in rep["cases"]] == [True, False, False, False]
    assert rep["equality"]["manifests_matched"] is True
    assert rep["equality"]["allowed_config_diff_keys"] == ["mesh", "rs_rows"]
    assert rep["provenance"]["source_sha256"] == "s" * 64
    assert rep["provenance"]["weights_sha256"] == "w" * 64
    # Each case carries its own plan provenance and audit summary.
    assert all("plan" in c and "summary" in c and c["manifest_path"].endswith("manifest.json")
               for c in rep["cases"])

    by_case = {c["case"]: c for c in rep["cases"]}
    seg = lambda c, s: by_case[c]["per_segment"][s]["dev_max_px"][-1]
    # The grid-only/global-only plan maps lines straight; the mesh and row residuals bend distinct probes.
    assert seg("global_only", "horizontal") < 1e-3 and seg("global_only", "vertical") < 1e-3
    assert seg("mesh_only", "horizontal") > 1.0 and seg("mesh_only", "vertical") < 1e-3
    assert seg("row_only", "vertical") > 1.0 and seg("row_only", "horizontal") < 1e-3


def test_compare_reference_is_first_without_baseline(tmp_path):
    a = make_case(tmp_path, "alpha", plan=make_plan())
    b = make_case(tmp_path, "beta", plan=make_plan())
    rep = P.compare_plans([a, b], every=2, samples=51)
    assert rep["reference"] == "alpha" and rep["cases"][0]["is_reference"] is True
    assert rep["cases"][1]["is_reference"] is False


def test_compare_metrics_are_independent_and_deltas_zero_for_reference(tmp_path):
    base = make_case(tmp_path, "baseline", plan=make_plan())
    other = make_case(tmp_path, "other", plan=make_plan(mesh=bow_mesh()))
    rep = P.compare_plans([base, other], every=2, samples=51)
    ref_case = rep["cases"][0]
    for name in P.SEGMENT_ORDER:
        block = ref_case["per_segment"][name]
        # Coverage and curvature are distinct, per-frame series and are never merged into one number.
        assert set(block) >= {"dev_max_px", "dev_p95_px", "coverage_frac", "delta_vs_reference"}
        assert len(block["dev_max_px"]) == len(rep["selection"]["frames"])
        assert len(block["coverage_frac"]) == len(rep["selection"]["frames"])
        assert all(v == 0.0 for v in block["delta_vs_reference"]["dev_max_px"])
        assert all(v == 0.0 for v in block["delta_vs_reference"]["coverage_frac"])
    assert "winner" in rep["not_a_winner_score"] and "score" in rep["not_a_winner_score"]
    for bad in ("score", "winner", "quality"):
        assert bad not in rep


def test_compare_final_frame_sampling(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan(t=65))
    b = make_case(tmp_path, "other", plan=make_plan(t=65))
    rep = P.compare_plans([a, b], every=30, samples=51)
    assert rep["selection"]["frames"] == [0, 30, 60, 64]
    assert rep["selection"]["final_frame_included"] is True
    for case in rep["cases"]:
        assert [pf["frame"] for pf in case["per_frame"]] == [0, 30, 60, 64]
        assert len(case["per_segment"]["horizontal"]["dev_max_px"]) == 4


@pytest.mark.parametrize("field", ["source_sha256", "source_window", "preprocessing", "weights_sha256"])
def test_compare_refuses_mismatched_manifest_identity(tmp_path, field):
    base = make_case(tmp_path, "baseline", plan=make_plan())
    if field == "source_window":
        kwargs, code = {"window": {"start_s": 5.0, "duration_s": 1.0}}, "mismatched_window"
    elif field == "preprocessing":
        kwargs, code = {"preprocessing": {"interlaced": True}}, "mismatched_preprocessing"
    elif field == "source_sha256":
        kwargs, code = {"source_sha": "x" * 64}, "mismatched_source"
    else:
        kwargs, code = {"weights_sha": "y" * 64}, "mismatched_weights"
    other = make_case(tmp_path, "other", plan=make_plan(), **kwargs)
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([base, other], every=2, samples=51)
    assert ei.value.code == code
    assert "unproven" in str(ei.value)


def test_compare_refuses_duplicate_case_names(tmp_path):
    a = make_case(tmp_path, "same", run_name="a", plan=make_plan())
    b = make_case(tmp_path, "same", run_name="b", plan=make_plan())
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "duplicate_case" and "same" in str(ei.value)


def test_compare_refuses_missing_manifest(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    orphan = save_real_plan(orphan_dir, make_plan(), name="plan_r0.npz")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, orphan], every=2, samples=51)
    assert ei.value.code == "manifest_missing"


def test_compare_refuses_malformed_manifest(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    bad = tmp_path / "bad"
    bad.mkdir()
    save_real_plan(bad, make_plan(), name="plan_r0.npz")
    (bad / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, bad / "plan_r0.npz"], every=2, samples=51)
    assert ei.value.code == "manifest_unreadable"

    # Well-formed JSON that is missing a required identity field is malformed, not silently accepted.
    bad2 = tmp_path / "bad2"
    write_manifest(bad2, case="other")
    save_real_plan(bad2, make_plan(), name="plan_r0.npz")
    man = json.loads((bad2 / "manifest.json").read_text())
    del man["source"]["sha256"]
    (bad2 / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, bad2 / "plan_r0.npz"], every=2, samples=51)
    assert ei.value.code == "manifest_malformed"


@pytest.mark.parametrize("override", [
    {"seed": 1},
    {"flow_iters": 8},
    {"smooth": "gauss"},
    {"sigma": 9.0},
    {"fidelity": 0.2},
    {"analysis_stride": 2},
    {"warm_iters": 8},
    {"corr_dtype": "float16"},
])
def test_compare_refuses_other_config_differences(tmp_path, override):
    base = make_case(tmp_path, "baseline", plan=make_plan())
    other = make_case(tmp_path, "other", plan=make_plan(), config={**BASELINE_CONFIG, **override})
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([base, other], every=2, samples=51)
    assert ei.value.code == "mismatched_config"
    changed_key = next(iter(override))
    assert changed_key in str(ei.value)
    assert "mesh" in str(ei.value) and "rs_rows" in str(ei.value)


def test_compare_refuses_missing_resolved_config_key(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    config = dict(BASELINE_CONFIG)
    del config["seed"]
    b = make_case(tmp_path, "other", plan=make_plan(), config=config)
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "mismatched_config" and "seed" in str(ei.value)


def test_compare_refuses_missing_config(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "other", plan=make_plan())
    man = json.loads((tmp_path / "other" / "manifest.json").read_text())
    del man["config"]
    (tmp_path / "other" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "manifest_malformed" and "config" in str(ei.value)


def test_compare_refuses_missing_capture_proof(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    # (1) experiment_identity present but capture object missing entirely
    b = make_case(tmp_path, "no_capture", plan=make_plan())
    man = json.loads((tmp_path / "no_capture" / "manifest.json").read_text())
    del man["experiment_identity"]["capture"]
    (tmp_path / "no_capture" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "manifest_malformed" and "capture" in str(ei.value)

    # (2) capture present but dump_plan key missing
    c = make_case(tmp_path, "no_dump_plan", plan=make_plan())
    man = json.loads((tmp_path / "no_dump_plan" / "manifest.json").read_text())
    del man["experiment_identity"]["capture"]["dump_plan"]
    (tmp_path / "no_dump_plan" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, c], every=2, samples=51)
    assert ei.value.code == "manifest_malformed" and "dump_plan" in str(ei.value)

    # (3) experiment_identity missing entirely
    d = make_case(tmp_path, "no_identity", plan=make_plan())
    man = json.loads((tmp_path / "no_identity" / "manifest.json").read_text())
    del man["experiment_identity"]
    (tmp_path / "no_identity" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, d], every=2, samples=51)
    assert ei.value.code == "manifest_malformed" and "experiment_identity" in str(ei.value)


def test_compare_refuses_false_dump_plan(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "not_captured", plan=make_plan(), dump_plan=False)
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "manifest_plan_not_captured"
    assert "dump_plan" in str(ei.value) and "not True" in str(ei.value)


def test_compare_refuses_mismatched_output_plan_basename(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "other", plan=make_plan(), outputs_plan="somewhere/else/plan_r9.npz")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "mismatched_plan_output"
    assert "plan_r9.npz" in str(ei.value) and "plan_r0.npz" in str(ei.value)


def test_compare_refuses_missing_or_empty_output_plan(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    for run_name, value in (("no_outputs", None), ("empty_plan", ""), ("bad_type", 7)):
        b = make_case(tmp_path, run_name, plan=make_plan())
        man = json.loads((tmp_path / run_name / "manifest.json").read_text())
        if value is None:
            del man["outputs"]
        else:
            man["outputs"]["plan"] = value
        (tmp_path / run_name / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        with pytest.raises(P.ComparisonError) as ei:
            P.compare_plans([a, b], every=2, samples=51)
        assert ei.value.code == "manifest_malformed"


def test_compare_matches_output_plan_by_basename_only(tmp_path, monkeypatch):
    # ``outputs.plan`` may be recorded relative to the run dir or the caller's CWD (or bare). Only the basename
    # is compared, and it is never resolved against the current working directory, because the plan is a sibling
    # of its manifest by construction.
    a = make_case(tmp_path, "baseline", plan=make_plan(), outputs_plan="plan_r0.npz")
    b = make_case(tmp_path, "other", plan=make_plan(),
                  outputs_plan="out/bench/other/plan_r0.npz")
    monkeypatch.chdir(tmp_path)
    rep = P.compare_plans([a, b], every=2, samples=51)
    assert rep["ok"] is True


def test_compare_report_records_capture_provenance(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "other", plan=make_plan())
    rep = P.compare_plans([a, b], every=2, samples=51)
    json.dumps(P.json_safe(rep), allow_nan=False)
    cap = rep["provenance"]["capture"]
    assert cap["dump_plan"] is True
    assert cap["checked_fields"] == ["outputs.plan", "experiment_identity.capture.dump_plan"]
    assert set(cap["output_plans"]) == {"baseline", "other"}
    assert all(p.endswith("plan_r0.npz") for p in cap["output_plans"].values())
    assert rep["equality"]["checked_capture_fields"] == [
        "outputs.plan", "experiment_identity.capture.dump_plan"]
    for entry in rep["provenance"]["plans"]:
        assert entry["capture_dump_plan"] is True
        assert entry["output_plan"].endswith("plan_r0.npz")


def test_cli_compare_refuses_false_dump_plan(tmp_path, capsys):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "other", plan=make_plan(), dump_plan=False)
    assert P.main(["--compare", str(a), str(b), "--every", "2"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "manifest_plan_not_captured"
    json.dumps(err, allow_nan=False)


def test_compare_refuses_non_object_config(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "other", plan=make_plan())
    man = json.loads((tmp_path / "other" / "manifest.json").read_text())
    man["config"] = ["not", "an", "object"]
    (tmp_path / "other" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == "manifest_malformed" and "config" in str(ei.value)


def test_compare_allows_mesh_and_rs_rows_config_differences(tmp_path):
    base = make_case(tmp_path, "baseline", plan=make_plan())
    other = make_case(tmp_path, "row_only", plan=make_plan(row=row_bend()),
                      config={**BASELINE_CONFIG, "mesh": [0, 0], "rs_rows": 0})
    rep = P.compare_plans([base, other], every=2, samples=51)
    assert rep["ok"] is True and rep["equality"]["manifests_matched"] is True
    assert rep["equality"]["allowed_config_diff_keys"] == ["mesh", "rs_rows"]
    assert set(rep["equality"]["checked_config_keys"]) == set(BASELINE_CONFIG)
    assert rep["provenance"]["config"] == BASELINE_CONFIG
    json.dumps(P.json_safe(rep), allow_nan=False)


@pytest.mark.parametrize("kind,code", [("dims", "mismatched_dimensions"),
                                       ("frames", "mismatched_frame_count"),
                                       ("fps", "mismatched_fps")])
def test_compare_refuses_mismatched_plan_geometry(tmp_path, kind, code):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    if kind == "dims":
        # Same manifest provenance, different frame dimensions.
        b = make_case(tmp_path, "other", plan=make_plan(w=32, h=24), w=32, h=24)
    elif kind == "frames":
        b = make_case(tmp_path, "other", plan=make_plan(t=5))
    else:
        b = make_case(tmp_path, "other", plan=make_plan(), fps=25.0)
    with pytest.raises(P.ComparisonError) as ei:
        P.compare_plans([a, b], every=2, samples=51)
    assert ei.value.code == code


def test_cli_compare_prints_comparison_schema(tmp_path, capsys):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "mesh_only", plan=make_plan(mesh=bow_mesh()))
    assert P.main(["--compare", str(a), str(b), "--every", "2", "--samples", "51"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["schema"] == P.COMPARISON_SCHEMA and rep["ok"] is True
    assert rep["reference"] == "baseline" and len(rep["cases"]) == 2


def test_cli_compare_refusal_is_json_error_exit_2(tmp_path, capsys):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    b = make_case(tmp_path, "other", plan=make_plan(), source_sha="x" * 64)
    assert P.main(["--compare", str(a), str(b), "--every", "2"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["schema"] == P.COMPARISON_SCHEMA and err["ok"] is False
    assert err["error"]["code"] == "mismatched_source"
    json.dumps(err, allow_nan=False)


def test_cli_compare_requires_at_least_two_and_no_positional(tmp_path):
    a = make_case(tmp_path, "baseline", plan=make_plan())
    with pytest.raises(SystemExit):
        P.main(["--compare", str(a)])
    b = make_case(tmp_path, "other", plan=make_plan())
    with pytest.raises(SystemExit):
        P.main([str(a), "--compare", str(a), str(b)])
