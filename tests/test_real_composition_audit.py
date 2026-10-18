"""Focused tests for the captured plan+motion composition audit (``bench/real_composition_audit.py``).

Covers: Motion NPZ round-trip and zero-fit shapes, strict schema/shape/finiteness/no-pickle validation,
sibling-manifest capture proof for both artifacts (plan and motion basenames + capture flags), analytic
projective/parallax Motion/Plan fixtures, sparse probe sampling, both high-pass chart modes with precomputed
smoothing, final-frame selection, JSON-safe reports with no composite winner score, and the CLI. No media,
FFmpeg, or changed production defaults.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mlxstab import stabilize as S
from mlxstab import geometry as G
from mlxstab import motion as MO
from mlxstab.motion import Fit

from bench import composition_audit as CA
from bench import real_composition_audit as R
from bench import shape_audit as SA

W, H = 64, 48

CONFIG_FIELDS = ("smooth", "sigma", "l1_weights", "budget", "mesh", "mesh_budget", "rs_rows", "flow_iters",
                 "flow_downscale", "warm_start", "step", "kernel", "zoom", "seed", "analysis_stride",
                 "flow_dtype", "corr_dtype", "warm_iters", "model", "fidelity", "warp_diag")


# --------------------------------------------------------------------------------------
# fixtures (analytic projective / parallax motion + plan)
# --------------------------------------------------------------------------------------

def _config_dict(cfg):
    d = {k: getattr(cfg, k) for k in CONFIG_FIELDS}
    d["l1_weights"] = list(d["l1_weights"])
    d["mesh"] = list(d["mesh"])
    return d


def _fits(pairs, nr=1):
    return [Fit(np.asarray(p.H, np.float64).copy(), np.asarray(p.mesh, np.float64).copy(),
                (np.asarray(p.rows, np.float64).copy() if p.rows is not None else np.zeros((nr, 2))),
                0.95, 0.01, False) for p in pairs]


def _motion_plan(pairs, *, W=W, H=H, gc=4, gr=3, rs_rows=0, pin=True):
    T = len(pairs) + 1
    mot = S.Motion(W, H, 30.0, fits=_fits(pairs, max(rs_rows, 1)), stride=1, n_frames=T)
    fx = CA.Fixture("fx", "fixture", W, H, T, pairs, (gc, gr), rs_rows=rs_rows)
    plan, cfg = CA.build_plan(fx, pin=pin)
    return mot, plan, cfg


def _projective_pairs(W=W, H=H, gc=4, gr=3):
    """Each true pair map is a projective homography, reported as ``H=I`` + the vertex residual mesh."""
    Ps = [SA.projective(7.5e-3, -4.5e-3), SA.projective(-6.0e-3, 6.0e-3)]
    return [CA.PairMotion(np.eye(3), CA.field_mesh(W, H, gc, gr, lambda p, P=P: G.apply_h(P, p) - p))
            for P in Ps]


def _parallax_fixture():
    fx = CA.build_fixtures()["parallax_two_layer"]
    return fx.pairs, fx.W, fx.H, fx.mesh_cells[0], fx.mesh_cells[1]


def _long_pairs(W=W, H=H, gc=4, gr=3, T=8):
    """Rotation globals + a varying residual over time, so the two temporal-smoothing charts differ."""
    pairs = []
    for k in range(T - 1):
        Rm = CA.rotation_about_clockwise(W, H, 2.0 * (k + 1))
        amp = 1.5 + 0.5 * k

        def resid(p, amp=amp):
            return np.stack([amp * np.sin(np.pi * (p[:, 0] + 0.5) / W),
                             0.5 * amp * np.cos(np.pi * (p[:, 1] + 0.5) / H)], 1)

        pairs.append(CA.PairMotion(Rm, CA.field_mesh(W, H, gc, gr, resid)))
    return pairs


def _identity_pairs(T, W=W, H=H, gc=4, gr=3):
    return [CA.PairMotion(np.eye(3), np.zeros((gr + 1, gc + 1, 2))) for _ in range(T - 1)]


def write_manifest(run, plan_path, motion_path, *, case="baseline", config=None, dump_plan=True,
                   dump_motion=True, outputs_plan=None, outputs_motion=None, source_sha="s" * 64,
                   weights_sha="w" * 64):
    man = {
        "schema": "mlxstab.bench.manifest/1",
        "run_id": run.name,
        "case": {"name": case},
        "source": {"sha256": source_sha, "window": {"start_s": 0.0, "duration_s": 1.0},
                   "preprocessing": {"interlaced": False}},
        "weights": {"sha256": weights_sha},
        "config": copy.deepcopy(config),
        "outputs": {"plan": outputs_plan if outputs_plan is not None else str(plan_path),
                    "motion": outputs_motion if outputs_motion is not None else str(motion_path)},
        "experiment_identity": {"capture": {"dump_plan": dump_plan, "dump_motion": dump_motion}},
    }
    (run / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return man


def save_case(root, *, name="run", case="baseline", pairs=None, W=W, H=H, gc=4, gr=3, rs_rows=0, pin=True,
              plan_name="plan_r0.npz", motion_name="motion_r0.npz", dump_plan=True, dump_motion=True,
              config=None, outputs_plan=None, outputs_motion=None, write=True):
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    pairs = _projective_pairs(W, H, gc, gr) if pairs is None else pairs
    mot, plan, cfg = _motion_plan(pairs, W=W, H=H, gc=gc, gr=gr, rs_rows=rs_rows, pin=pin)
    plan_path = run / plan_name
    motion_path = run / motion_name
    S.save_plan(plan, str(plan_path), W, H, 30.0)
    S.save_motion_npz(mot, str(motion_path), cfg)
    if write:
        write_manifest(run, plan_path, motion_path, case=case,
                       config=_config_dict(cfg) if config is None else config,
                       dump_plan=dump_plan, dump_motion=dump_motion,
                       outputs_plan=outputs_plan, outputs_motion=outputs_motion)
    return plan_path, motion_path, run, mot, plan, cfg


# --------------------------------------------------------------------------------------
# motion NPZ round-trip / zero-fit
# --------------------------------------------------------------------------------------

def test_motion_npz_roundtrip_reconstructs_fits(tmp_path):
    mot, _, _ = _motion_plan(_projective_pairs(), pin=False)
    path = tmp_path / "m.npz"
    S.save_motion_npz(mot, str(path), S.Config(mesh=(4, 3), rs_rows=0))
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {"schema", "width", "height", "n_frames", "fps", "stride", "n_fits",
                                "H", "mesh", "rows", "inlier_frac", "mover_frac", "fallback",
                                "n_valid_samples", "n_ransac_inliers", "fallback_reason_code", "fit_meta_available",
                                "mesh_support_mass", "row_support_mass",
                                "cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass",
                                "cycle_map_available", "cut_frames"}
        assert all(z[k].dtype.kind in "fiuUb" for k in z.files)
        assert z["mesh_support_mass"].shape == (2, 4, 5) and z["row_support_mass"].shape == (2, 1)
        # The fixtures carry no cycle maps, so save dumps zero maps tagged unavailable.
        assert z["cycle_err_rms"].shape == (2, 4, 5) and z["cycle_sample_mass"].shape == (2, 4, 5)
        assert not z["cycle_map_available"].any()
    loaded, meta = R.load_motion_npz(path)
    assert isinstance(loaded, S.Motion) and len(loaded.fits) == len(mot.fits)
    assert loaded.W == mot.W and loaded.H == mot.H and loaded.n_frames == mot.n_frames and loaded.stride == 1
    for a, b in zip(loaded.fits, mot.fits):
        np.testing.assert_array_equal(a.H, b.H)
        np.testing.assert_array_equal(a.mesh, b.mesh)
        np.testing.assert_array_equal(a.rows, b.rows)
        assert a.inlier_frac == pytest.approx(b.inlier_frac) and a.mover_frac == pytest.approx(b.mover_frac)
        # The fixtures carry no support mass, so save dumps shape-correct zeros.
        assert a.mesh_support_mass.shape == a.mesh.shape[:2] and np.all(a.mesh_support_mass == 0)
        assert a.row_support_mass.shape == (a.rows.shape[0],) and np.all(a.row_support_mass == 0)
    assert meta["mesh_cells"] == [4, 3] and meta["row_bands"] == 1 and meta["n_fits"] == 2
    assert meta["support_mass"]["present"] is True and meta["support_mass"]["threshold"] == 3.0


def test_save_motion_npz_zero_fit_uses_config_shapes(tmp_path):
    mot = S.Motion(W, H, 30.0, fits=[], stride=1, n_frames=1)
    path = tmp_path / "zero.npz"
    S.save_motion_npz(mot, str(path), S.Config(mesh=(4, 3), rs_rows=5))
    loaded, meta = R.load_motion_npz(path)
    assert loaded.fits == [] and meta["n_fits"] == 0
    assert meta["mesh_cells"] == [4, 3] and meta["row_bands"] == 5   # from cfg.mesh / cfg.rs_rows
    with np.load(path, allow_pickle=False) as z:                     # zero-fit support arrays keep config shapes
        assert z["mesh_support_mass"].shape == (0, 4, 5) and z["row_support_mass"].shape == (0, 5)
    # With no config at all the shapes fall back to the analyze minimum (one cell, one row band).
    S.save_motion_npz(mot, str(tmp_path / "zero2.npz"))
    _, meta2 = R.load_motion_npz(tmp_path / "zero2.npz")
    assert meta2["mesh_cells"] == [1, 1] and meta2["row_bands"] == 1


def test_motion_v2_roundtrip_preserves_support_mass(tmp_path):
    gc, gr, nr = 4, 3, 3
    mesh = np.zeros((gr + 1, gc + 1, 2))
    rows = np.zeros((nr, 2))
    msm = np.full((gr + 1, gc + 1), 3.0); msm[0, 0] = 1.0        # one diffusion-filled vertex
    rsm = np.array([0.5, 4.0, 2.0])
    fit = Fit(np.eye(3), mesh, rows, 0.9, 0.0, False,
              mesh_support_mass=msm, row_support_mass=rsm)
    mot = S.Motion(W, H, 30.0, fits=[fit], stride=1, n_frames=2)
    path = tmp_path / "support.npz"
    S.save_motion_npz(mot, str(path), S.Config(mesh=(gc, gr), rs_rows=nr))

    with np.load(path, allow_pickle=False) as z:
        assert str(z["schema"]) == "mlxstab.motion/5"
        np.testing.assert_array_equal(z["mesh_support_mass"][0], msm)
        np.testing.assert_array_equal(z["row_support_mass"][0], rsm)

    loaded, meta = R.load_motion_npz(path)
    np.testing.assert_array_equal(loaded.fits[0].mesh_support_mass, msm)
    np.testing.assert_array_equal(loaded.fits[0].row_support_mass, rsm)
    assert meta["support_mass"]["present"] is True and meta["support_mass"]["threshold"] == 3.0


def test_save_motion_npz_none_support_serializes_zero_masses(tmp_path):
    fit = Fit(np.eye(3), np.zeros((3, 5, 2)), np.zeros((2, 2)), 0.9, 0.0, False)   # support defaults None
    assert fit.mesh_support_mass is None and fit.row_support_mass is None
    mot = S.Motion(W, H, 30.0, fits=[fit], stride=1, n_frames=2)
    path = tmp_path / "none_support.npz"
    S.save_motion_npz(mot, str(path), S.Config(mesh=(4, 2), rs_rows=2))
    with np.load(path, allow_pickle=False) as z:
        assert z["mesh_support_mass"].shape == (1, 3, 5) and z["row_support_mass"].shape == (1, 2)
        assert np.all(z["mesh_support_mass"] == 0) and np.all(z["row_support_mass"] == 0)
    loaded, meta = R.load_motion_npz(path)
    assert loaded.fits[0].mesh_support_mass.shape == (3, 5)
    assert loaded.fits[0].row_support_mass.shape == (2,)
    assert np.all(loaded.fits[0].mesh_support_mass == 0) and np.all(loaded.fits[0].row_support_mass == 0)
    assert meta["support_mass"]["present"] is True


def test_legacy_v1_payload_loads_with_support_none(tmp_path):
    p = base_motion_payload()
    p["schema"] = np.asarray("mlxstab.motion/1")
    for key in R.MOTION_SUPPORT_KEYS + R.MOTION_CYCLE_KEYS + R.MOTION_FIT_META_KEYS:
        del p[key]
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="v1.npz"))
    assert meta["schema"] == "mlxstab.motion/1" and meta["support_mass"]["present"] is False
    assert meta["cycle_maps"]["present"] is False and meta["cycle_maps"]["available_fits"] is None
    assert len(loaded.fits) == 2
    for fit in loaded.fits:
        assert fit.mesh_support_mass is None and fit.row_support_mass is None
        assert (fit.cycle_err_rms is None and fit.cycle_inconsistency_frac is None
                and fit.cycle_sample_mass is None)
        assert (fit.n_valid_samples is None and fit.n_ransac_inliers is None
                and fit.fallback_reason is None)


def test_legacy_v2_payload_loads_with_support_but_no_cycle(tmp_path):
    p = base_motion_payload()
    p["schema"] = np.asarray("mlxstab.motion/2")
    for key in R.MOTION_CYCLE_KEYS + R.MOTION_FIT_META_KEYS:
        del p[key]
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="v2.npz"))
    assert meta["schema"] == "mlxstab.motion/2"
    assert meta["support_mass"]["present"] is True and meta["cycle_maps"]["present"] is False
    for fit in loaded.fits:
        assert fit.mesh_support_mass is not None and fit.row_support_mass is not None
        assert (fit.cycle_err_rms is None and fit.cycle_inconsistency_frac is None
                and fit.cycle_sample_mass is None)
        assert (fit.n_valid_samples is None and fit.n_ransac_inliers is None
                and fit.fallback_reason is None)


def test_legacy_v3_payload_loads_with_support_and_cycle_but_no_fit_meta(tmp_path):
    p = base_motion_payload()
    p["schema"] = np.asarray("mlxstab.motion/3")
    for key in R.MOTION_FIT_META_KEYS:
        del p[key]
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="v3.npz"))
    assert meta["schema"] == "mlxstab.motion/3"
    assert meta["support_mass"]["present"] is True and meta["cycle_maps"]["present"] is True
    for fit in loaded.fits:
        assert fit.mesh_support_mass is not None and fit.row_support_mass is not None
        # /3 predates the fit-gate diagnostics: unavailable/null, never invented.
        assert (fit.n_valid_samples is None and fit.n_ransac_inliers is None
                and fit.fallback_reason is None)


# --------------------------------------------------------------------------------------
# strict payload validation
# --------------------------------------------------------------------------------------

def base_motion_payload(n_fits=2, mesh_cells=(4, 3), n_rows=1, n_frames=3):
    gc, gr = mesh_cells
    return {
        "schema": np.asarray(S.MOTION_SCHEMA_V4), "width": np.int64(W), "height": np.int64(H),
        "n_frames": np.int64(n_frames), "fps": np.float64(30.0), "stride": np.int64(1),
        "n_fits": np.int64(n_fits),
        "H": np.tile(np.eye(3), (n_fits, 1, 1)), "mesh": np.zeros((n_fits, gr + 1, gc + 1, 2)),
        "rows": np.zeros((n_fits, n_rows, 2)), "inlier_frac": np.full((n_fits,), 0.9),
        "mover_frac": np.zeros((n_fits,)), "fallback": np.zeros((n_fits,), bool),
        "n_valid_samples": np.full((n_fits,), -1, np.int64),
        "n_ransac_inliers": np.full((n_fits,), -1, np.int64),
        "fallback_reason_code": np.full((n_fits,), -1, np.int64),
        "fit_meta_available": np.zeros((n_fits,), bool),
        "mesh_support_mass": np.zeros((n_fits, gr + 1, gc + 1)),
        "row_support_mass": np.zeros((n_fits, n_rows)),
        "cycle_err_rms": np.zeros((n_fits, gr + 1, gc + 1)),
        "cycle_inconsistency_frac": np.zeros((n_fits, gr + 1, gc + 1)),
        "cycle_sample_mass": np.zeros((n_fits, gr + 1, gc + 1)),
        "cycle_map_available": np.zeros((n_fits,), bool),
    }


def base_motion_v5_payload(n_fits=2, mesh_cells=(4, 3), n_rows=1, n_frames=3):
    payload = base_motion_payload(n_fits, mesh_cells, n_rows, n_frames)
    payload["schema"] = np.asarray(S.MOTION_SCHEMA)
    payload["cut_frames"] = np.zeros((0,), np.int64)
    return payload


def write_payload(tmp_path, payload, name="payload.npz"):
    path = tmp_path / name
    np.savez_compressed(path, **payload)
    return path


def test_rejects_wrong_schema(tmp_path):
    p = base_motion_payload(); p["schema"] = np.asarray("mlxstab.motion/999")
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p))
    assert ei.value.code == "bad_schema" and "is not one of" in str(ei.value)


def test_rejects_missing_key(tmp_path):
    p = base_motion_payload(); del p["mesh"]
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p))
    assert ei.value.code == "missing_keys" and "mesh" in str(ei.value)


@pytest.mark.parametrize("key,bad", [
    ("H", np.zeros((2, 2, 3))),
    ("mesh", np.zeros((2, 3, 2))),
    ("rows", np.zeros((2, 2))),
    ("inlier_frac", np.zeros((2, 1))),
    ("fallback", np.zeros((2, 1), bool)),
])
def test_rejects_bad_shapes(tmp_path, key, bad):
    p = base_motion_payload(); p[key] = bad
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p))
    assert ei.value.code == "bad_shape"


def test_rejects_mesh_zero_row_cells_positive_column_cells(tmp_path):
    p = base_motion_payload()
    p["mesh"] = np.zeros((2, 1, 5, 2))          # gr=0, gc=4
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="zero_rows.npz"))
    assert ei.value.code == "bad_shape" and "row cells" in str(ei.value)
    p["mesh"] = np.zeros((2, 2, 5, 2))          # gr=1, gc=4 is valid
    p["mesh_support_mass"] = np.zeros((2, 2, 5))    # support must follow the mesh shape
    for _k in ("cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass"):
        p[_k] = np.zeros((2, 2, 5))                  # cycle maps follow the mesh shape too
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="one_row.npz"))
    assert len(loaded.fits) == 2 and meta["mesh_cells"] == [4, 1]


@pytest.mark.parametrize("key", ["H", "mesh", "rows", "inlier_frac", "mover_frac"])
def test_rejects_non_finite_arrays(tmp_path, key):
    p = base_motion_payload(); p[key] = p[key].astype(np.float64).copy(); p[key].reshape(-1)[0] = np.nan
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p))
    assert ei.value.code == "nonfinite"


@pytest.mark.parametrize("key,val,code", [
    ("width", np.int64(0), "bad_dims"),
    ("height", np.int64(1), "bad_dims"),
    ("n_frames", np.int64(0), "bad_dims"),
    ("stride", np.int64(0), "bad_dims"),
    ("n_fits", np.int64(-1), "bad_dims"),
    ("fps", np.float64(0.0), "bad_scalar"),
    ("fps", np.float64(np.inf), "bad_scalar"),
])
def test_rejects_bad_dimensions_and_scalars(tmp_path, key, val, code):
    p = base_motion_payload(); p[key] = val
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p))
    assert ei.value.code == code


def test_rejects_object_array_without_unpickling(tmp_path):
    p = base_motion_payload()
    obj = np.empty((2, 3, 3), dtype=object)
    obj[:] = np.eye(3)
    p["H"] = obj
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p))
    assert ei.value.code == "unreadable"


def test_rejects_v2_missing_support_keys(tmp_path):
    for key in ("mesh_support_mass", "row_support_mass"):
        p = base_motion_payload()
        del p[key]
        with pytest.raises(R.MotionPayloadError) as ei:
            R.load_motion_npz(write_payload(tmp_path, p, name=f"missing_{key}.npz"))
        assert ei.value.code == "missing_keys" and key in str(ei.value)


def test_rejects_v2_bad_support_shapes(tmp_path):
    p = base_motion_payload()
    p["mesh_support_mass"] = np.zeros((2, 3, 3))                # expected (2, 4, 5)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_mesh_mass.npz"))
    assert ei.value.code == "bad_shape"
    p = base_motion_payload()
    p["row_support_mass"] = np.zeros((2, 2))                    # expected (2, 1)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_row_mass.npz"))
    assert ei.value.code == "bad_shape"


@pytest.mark.parametrize("key", ["mesh_support_mass", "row_support_mass"])
def test_rejects_v2_non_finite_support(tmp_path, key):
    p = base_motion_payload()
    p[key] = p[key].astype(np.float64).copy()
    p[key].reshape(-1)[0] = np.nan
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"nonfinite_{key}.npz"))
    assert ei.value.code == "nonfinite"


@pytest.mark.parametrize("key", ["mesh_support_mass", "row_support_mass"])
def test_rejects_v2_negative_support_mass(tmp_path, key):
    p = base_motion_payload()
    p[key] = p[key].astype(np.float64).copy()
    p[key].reshape(-1)[0] = -0.5                       # support mass is a count and cannot be negative
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"negative_{key}.npz"))
    assert ei.value.code == "negative" and key in str(ei.value)


def test_legacy_v1_negative_support_is_absent_and_loads(tmp_path):
    p = base_motion_payload()
    p["schema"] = np.asarray("mlxstab.motion/1")
    for key in ("mesh_support_mass", "row_support_mass"):
        del p[key]
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="v1_no_support.npz"))
    assert meta["support_mass"]["present"] is False
    assert all(f.mesh_support_mass is None and f.row_support_mass is None for f in loaded.fits)


def test_rejects_support_object_array_without_unpickling(tmp_path):
    p = base_motion_payload()
    obj = np.empty((2, 4, 5), dtype=object)
    obj[:] = 0.0
    p["mesh_support_mass"] = obj
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="obj_mass.npz"))
    assert ei.value.code == "unreadable"


def test_motion_v5_roundtrip_preserves_cycle_maps_fit_meta_and_cuts(tmp_path):
    gc, gr, nr = 4, 3, 2
    mesh = np.zeros((gr + 1, gc + 1, 2))
    rows = np.zeros((nr, 2))
    rms = np.full((gr + 1, gc + 1), 0.25); rms[0, 0] = 0.0
    frac = np.full((gr + 1, gc + 1), 0.5); frac[0, 0] = 0.0
    mass = np.full((gr + 1, gc + 1), 7.0); mass[0, 0] = 0.0           # zero-mass vertex -> zero map
    ok = Fit(np.eye(3), mesh, rows, 0.9, 0.0, False,
             n_valid_samples=140, n_ransac_inliers=132, fallback_reason="none",
             cycle_err_rms=rms, cycle_inconsistency_frac=frac, cycle_sample_mass=mass)
    manual = Fit(np.eye(3), mesh, rows, 0.9, 0.0, False)             # cycle and fit-meta fields default None
    mot = S.Motion(W, H, 30.0, fits=[ok, manual], stride=1, n_frames=3)
    path = tmp_path / "cycle.npz"
    S.save_motion_npz(mot, str(path), S.Config(mesh=(gc, gr), rs_rows=nr))

    with np.load(path, allow_pickle=False) as z:
        assert str(z["schema"]) == S.MOTION_SCHEMA == "mlxstab.motion/5"
        assert z["cut_frames"].shape == (0,)
        np.testing.assert_array_equal(z["cycle_err_rms"][0], rms)
        np.testing.assert_array_equal(z["cycle_inconsistency_frac"][0], frac)
        np.testing.assert_array_equal(z["cycle_sample_mass"][0], mass)
        assert list(z["cycle_map_available"]) == [True, False]
        # A manual Fit serializes zero maps, tagged unavailable rather than silently available.
        assert np.all(z["cycle_err_rms"][1] == 0.0)
        assert not z["cycle_map_available"][1]
        # Fit-gate metadata round-trips as the integer reason code; a manual Fit uses the unavailable sentinels.
        assert list(z["fit_meta_available"]) == [True, False]
        assert list(z["n_valid_samples"]) == [140, -1] and list(z["n_ransac_inliers"]) == [132, -1]
        assert list(z["fallback_reason_code"]) == [0, -1]

    loaded, meta = R.load_motion_npz(path)
    np.testing.assert_array_equal(loaded.fits[0].cycle_err_rms, rms)
    np.testing.assert_array_equal(loaded.fits[0].cycle_inconsistency_frac, frac)
    np.testing.assert_array_equal(loaded.fits[0].cycle_sample_mass, mass)
    assert loaded.fits[0].n_valid_samples == 140 and loaded.fits[0].n_ransac_inliers == 132
    assert loaded.fits[0].fallback_reason == "none"
    assert loaded.fits[1].cycle_err_rms is None and loaded.fits[1].cycle_inconsistency_frac is None
    assert loaded.fits[1].cycle_sample_mass is None
    assert loaded.fits[1].n_valid_samples is None and loaded.fits[1].n_ransac_inliers is None
    assert loaded.fits[1].fallback_reason is None
    assert meta["cycle_maps"]["present"] is True and meta["cycle_maps"]["available_fits"] == 1


def test_motion_v4_remains_loadable_without_cut_metadata(tmp_path):
    p = base_motion_payload()
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="motion_v4.npz"))
    assert loaded.cuts == () and meta["cut_frames"] == []


def test_motion_v4_rejects_v5_sequence_boundary_reason_code(tmp_path):
    p = base_motion_payload(n_fits=1, n_frames=2)
    p["n_valid_samples"] = np.array([0], np.int64)
    p["n_ransac_inliers"] = np.array([0], np.int64)
    p["fallback_reason_code"] = np.array([MO.FALLBACK_REASON_CODES[MO.FALLBACK_REASON_SEQUENCE_BOUNDARY]],
                                          np.int64)
    p["fit_meta_available"] = np.array([True])
    p["fallback"] = np.array([True])
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="v4_new_reason.npz"))
    assert ei.value.code == "bad_range"


def test_motion_v5_requires_cut_metadata(tmp_path):
    p = base_motion_v5_payload()
    del p["cut_frames"]
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="v5_missing_cuts.npz"))
    assert ei.value.code == "missing_keys" and "cut_frames" in str(ei.value)


def test_motion_v5_requires_boundary_fit_reason_to_match_cut_frames(tmp_path):
    p = base_motion_v5_payload(n_fits=1, n_frames=2)
    p["n_valid_samples"] = np.array([0], np.int64)
    p["n_ransac_inliers"] = np.array([0], np.int64)
    p["fallback_reason_code"] = np.array(
        [MO.FALLBACK_REASON_CODES[MO.FALLBACK_REASON_SEQUENCE_BOUNDARY]], np.int64)
    p["fit_meta_available"] = np.array([True])
    p["fallback"] = np.array([True])
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="orphan_boundary_reason.npz"))
    assert ei.value.code == "bad_range" and "must match cut_frames" in str(ei.value)


def test_rejects_v4_missing_fit_meta_keys(tmp_path):
    for key in R.MOTION_FIT_META_KEYS:
        p = base_motion_payload()
        del p[key]
        with pytest.raises(R.MotionPayloadError) as ei:
            R.load_motion_npz(write_payload(tmp_path, p, name=f"missing_{key}.npz"))
        assert ei.value.code == "missing_keys" and key in str(ei.value)


def test_rejects_v4_bad_fit_meta_shapes(tmp_path):
    p = base_motion_payload(); p["n_valid_samples"] = np.full((2, 1), -1, np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_nvalid.npz"))
    assert ei.value.code == "bad_shape"
    p = base_motion_payload(); p["n_ransac_inliers"] = np.full((2, 1), -1, np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_ninliers.npz"))
    assert ei.value.code == "bad_shape"
    p = base_motion_payload(); p["fallback_reason_code"] = np.full((2, 1), -1, np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_reason.npz"))
    assert ei.value.code == "bad_shape"
    p = base_motion_payload(); p["fit_meta_available"] = np.zeros((2, 1), bool)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_avail.npz"))
    assert ei.value.code == "bad_shape"


def test_rejects_v4_non_integer_fit_counts(tmp_path):
    p = base_motion_payload(); p["n_valid_samples"] = np.full((2,), 1.5)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="float_counts.npz"))
    assert ei.value.code == "bad_shape" and "integer" in str(ei.value)


@pytest.mark.parametrize("count", [-2, -5])
@pytest.mark.parametrize("key", ["n_valid_samples", "n_ransac_inliers"])
def test_rejects_v4_fit_counts_below_sentinel(tmp_path, key, count):
    p = base_motion_payload(); p[key] = np.full((2,), count, np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"low_{key}.npz"))
    assert ei.value.code == "bad_range" and "-1" in str(ei.value)


def test_rejects_v4_unknown_fallback_reason(tmp_path):
    p = base_motion_payload()
    p["fit_meta_available"] = np.asarray([True, False])
    p["n_valid_samples"] = np.asarray([10, -1], np.int64)
    p["n_ransac_inliers"] = np.asarray([9, -1], np.int64)
    p["fallback_reason_code"] = np.asarray([99, -1], np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="unknown_reason.npz"))
    assert ei.value.code == "bad_range" and "fallback_reason_code" in str(ei.value)


def test_rejects_v4_fallback_reason_code_out_of_range(tmp_path):
    # The numeric codes are 0..3 with -1 for "unavailable"; anything outside that is malformed.
    for bad in (MO.FALLBACK_REASON_UNAVAILABLE_CODE - 1, max(MO.FALLBACK_REASON_BY_CODE) + 1):
        p = base_motion_payload()
        p["fit_meta_available"] = np.asarray([True, False])
        p["n_valid_samples"] = np.asarray([10, -1], np.int64)
        p["n_ransac_inliers"] = np.asarray([9, -1], np.int64)
        p["fallback_reason_code"] = np.asarray([bad, -1], np.int64)
        with pytest.raises(R.MotionPayloadError) as ei:
            R.load_motion_npz(write_payload(tmp_path, p, name=f"range_{bad}.npz"))
        assert ei.value.code == "bad_range" and "fallback_reason_code" in str(ei.value)


def test_rejects_v4_non_integer_fallback_reason_code(tmp_path):
    p = base_motion_payload(); p["fallback_reason_code"] = np.full((2,), 1.5)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="float_reason.npz"))
    assert ei.value.code == "bad_shape" and "integer" in str(ei.value)


def test_rejects_v4_available_fit_with_unavailable_reason_code(tmp_path):
    # An available fit must carry a real 0..3 code, never the -1 unavailable sentinel.
    p = base_motion_payload()
    p["fit_meta_available"] = np.asarray([True, False])
    p["n_valid_samples"] = np.asarray([10, -1], np.int64)
    p["n_ransac_inliers"] = np.asarray([9, -1], np.int64)
    p["fallback_reason_code"] = np.asarray([MO.FALLBACK_REASON_UNAVAILABLE_CODE, -1], np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="avail_unavailable_reason.npz"))
    assert ei.value.code == "bad_range" and "fallback_reason_code" in str(ei.value)


def test_rejects_v4_unavailable_fit_meta_with_non_sentinel_data(tmp_path):
    p = base_motion_payload()
    p["fit_meta_available"] = np.asarray([False, False])
    p["n_valid_samples"] = np.asarray([10, -1], np.int64)          # first fit is tagged unavailable
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="unavail_count.npz"))
    assert ei.value.code == "bad_range" and "n_valid_samples" in str(ei.value)

    p = base_motion_payload()
    p["fit_meta_available"] = np.asarray([False, False])
    p["fallback_reason_code"] = np.asarray([0, -1], np.int64)   # first reason is nonzero
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="unavail_reason.npz"))
    assert ei.value.code == "bad_range" and "fallback_reason_code" in str(ei.value)


def test_rejects_v4_fit_meta_availability_not_binary(tmp_path):
    p = base_motion_payload(); p["fit_meta_available"] = np.asarray([2, 0], np.int64)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="avail_two.npz"))
    assert ei.value.code == "bad_range" and "0 or 1" in str(ei.value)


def test_v4_available_fit_meta_roundtrips_counts_and_reason(tmp_path):
    p = base_motion_payload()
    p["fit_meta_available"] = np.asarray([True, False])
    p["n_valid_samples"] = np.asarray([120, -1], np.int64)
    p["n_ransac_inliers"] = np.asarray([110, -1], np.int64)
    p["fallback_reason_code"] = np.asarray([0, -1], np.int64)
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="fitmeta.npz"))
    assert loaded.fits[0].n_valid_samples == 120 and loaded.fits[0].n_ransac_inliers == 110
    assert loaded.fits[0].fallback_reason == "none"
    assert loaded.fits[1].n_valid_samples is None and loaded.fits[1].fallback_reason is None


def test_v4_prestandardization_fallback_reason_alias_loads(tmp_path):
    # The just-captured /4 payloads named the integer code array ``fallback_reason`` before the standard
    # ``fallback_reason_code``; the loader reads that alias so the preserved capture still loads.
    p = base_motion_payload()
    del p["fallback_reason_code"]
    p["fallback_reason"] = np.asarray([0, -1], np.int64)
    p["fit_meta_available"] = np.asarray([True, False])
    p["n_valid_samples"] = np.asarray([120, -1], np.int64)
    p["n_ransac_inliers"] = np.asarray([110, -1], np.int64)
    loaded, meta = R.load_motion_npz(write_payload(tmp_path, p, name="alias.npz"))
    assert loaded.fits[0].fallback_reason == "none"
    assert loaded.fits[1].n_valid_samples is None and loaded.fits[1].fallback_reason is None


def test_v4_prestandardization_alias_still_requires_integer_codes(tmp_path):
    # The alias only renames the key: a Unicode reason array is still rejected exactly as the int64 array was.
    p = base_motion_payload()
    del p["fallback_reason_code"]
    p["fallback_reason"] = np.asarray(["none", ""], dtype="<U4")
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="alias_string.npz"))
    assert ei.value.code == "bad_shape" and "integer" in str(ei.value)


def test_rejects_v3_missing_cycle_keys(tmp_path):
    for key in R.MOTION_CYCLE_KEYS:
        p = base_motion_payload()
        del p[key]
        with pytest.raises(R.MotionPayloadError) as ei:
            R.load_motion_npz(write_payload(tmp_path, p, name=f"missing_{key}.npz"))
        assert ei.value.code == "missing_keys" and key in str(ei.value)


@pytest.mark.parametrize("key", ["cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass"])
def test_rejects_v3_bad_cycle_map_shape(tmp_path, key):
    p = base_motion_payload()
    p[key] = np.zeros((2, 3, 3))                    # expected (2, 4, 5)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"badshape_{key}.npz"))
    assert ei.value.code == "bad_shape"


def test_rejects_v3_bad_cycle_availability_shape(tmp_path):
    p = base_motion_payload()
    p["cycle_map_available"] = np.zeros((2, 1), bool)
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="bad_avail.npz"))
    assert ei.value.code == "bad_shape"


@pytest.mark.parametrize("key", ["cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass"])
def test_rejects_v3_non_finite_cycle_maps(tmp_path, key):
    p = base_motion_payload()
    p[key] = p[key].astype(np.float64).copy()
    p[key].reshape(-1)[0] = np.nan
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"nonfinite_{key}.npz"))
    assert ei.value.code == "nonfinite"


@pytest.mark.parametrize("key", ["cycle_err_rms", "cycle_sample_mass"])
def test_rejects_v3_negative_cycle_maps(tmp_path, key):
    p = base_motion_payload()
    p[key] = p[key].astype(np.float64).copy()
    p[key].reshape(-1)[0] = -0.5                    # RMS and sample mass cannot be negative
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"negative_{key}.npz"))
    assert ei.value.code == "negative" and key in str(ei.value)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_rejects_v3_inconsistency_fraction_out_of_range(tmp_path, bad):
    p = base_motion_payload()
    p["cycle_inconsistency_frac"] = p["cycle_inconsistency_frac"].astype(np.float64).copy()
    p["cycle_inconsistency_frac"].reshape(-1)[0] = bad
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"range_{bad}.npz"))
    assert ei.value.code == "bad_range" and "[0, 1]" in str(ei.value)


def test_rejects_v3_cycle_availability_not_binary(tmp_path):
    p = base_motion_payload()
    p["cycle_map_available"] = np.asarray([2, 0], np.int64)   # 2 is not a valid 0/1 availability flag
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name="avail_two.npz"))
    assert ei.value.code == "bad_range" and "0 or 1" in str(ei.value)


@pytest.mark.parametrize("key", ["cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass"])
def test_rejects_v3_unavailable_cycle_maps_with_nonzero_data(tmp_path, key):
    p = base_motion_payload()
    p["cycle_map_available"] = np.asarray([0, 0], np.bool_)    # all fits tagged unavailable
    p[key] = p[key].astype(np.float64).copy()
    p[key].reshape(-1)[0] = 0.5                                # nonzero data under an unavailable fit
    with pytest.raises(R.MotionPayloadError) as ei:
        R.load_motion_npz(write_payload(tmp_path, p, name=f"unavail_nonzero_{key}.npz"))
    assert ei.value.code == "bad_range" and key in str(ei.value)


# --------------------------------------------------------------------------------------
# composition audit on analytic fixtures
# --------------------------------------------------------------------------------------

def test_projective_fixture_separates_composition_from_production(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path)
    rep = R.run_real_composition_audit(plan_path, motion_path, every=1, samples=201)
    assert rep["schema"] == R.SCHEMA and rep["ok"] is True
    assert [m["mode"] for m in rep["candidate_modes"]] == list(R.CANDIDATE_MODES)
    last = rep["per_frame"][-1]
    horiz = last["segments"]["horizontal"]
    prod = horiz["production"]["mapped"]["dev_max_px"]
    exact = horiz["candidates"]["exact_composition"]["mapped"]["dev_max_px"]
    assert prod is not None and exact is not None and exact < prod
    disp = horiz["candidates"]["exact_composition"]["displacement_from_production_px"]
    assert disp["max_px"] > 1.0 and disp["n"] == 201
    # The exact composition mode maps the probe much closer to straight than production's fixed residual sum.
    assert rep["summary"]["production_max_mapped_dev_max_px"] >= exact


def test_parallax_fixture_runs_all_modes(tmp_path):
    pairs, Wp, Hp, gcp, grp = _parallax_fixture()
    plan_path, motion_path, *_ = save_case(tmp_path, name="parallax", pairs=pairs, W=Wp, H=Hp, gc=gcp, gr=grp)
    rep = R.run_real_composition_audit(plan_path, motion_path, every=1, samples=101)
    assert rep["ok"] is True and rep["plan"]["frames"] == 3
    for pf in rep["per_frame"]:
        for name in R.PSA.SEGMENT_ORDER:
            seg = pf["segments"][name]
            assert seg["production"]["coverage"]["n"] == 101
            for mode in R.CANDIDATE_MODES:
                cand = seg["candidates"][mode]
                assert cand["coverage"]["n"] == 101
                assert cand["displacement_from_production_px"]["n"] == 101
                assert cand["mapped"]["ok"] in (True, False)   # never silently hidden


def test_both_highpass_charts_present_and_precomputed_once(tmp_path):
    pairs = _long_pairs(T=8)
    plan_path, motion_path, run, mot, plan, cfg = save_case(tmp_path, name="long", pairs=pairs, pin=False)
    rep = R.run_real_composition_audit(plan_path, motion_path, every=1, samples=101)
    modes = {m["mode"] for m in rep["candidate_modes"]}
    assert modes == set(R.CANDIDATE_MODES)
    smoothed = R.precompute_smoothed(mot, plan, cfg)
    assert set(smoothed) == {"source_highpass", "reference_highpass"}
    T = rep["plan"]["frames"]
    nref = (plan.mesh_corr.shape[1]) * (plan.mesh_corr.shape[2])
    for key, arr in smoothed.items():
        assert arr.shape == (T, nref, 2) and np.isfinite(arr).all()
    # The two charts are genuinely different for non-identity globals with a varying residual.
    assert np.abs(smoothed["source_highpass"] - smoothed["reference_highpass"]).max() > 1e-6
    for pf in rep["per_frame"]:
        for mode in ("source_highpass", "reference_highpass"):
            cand = pf["segments"]["horizontal"]["candidates"][mode]
            assert cand["mapped"]["ok"] is True and cand["displacement_from_production_px"]["n"] == 101


def test_sparse_sampling_uses_requested_points(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path, name="sparse")
    rep = R.run_real_composition_audit(plan_path, motion_path, every=2, samples=33)
    for pf in rep["per_frame"]:
        for name in R.PSA.SEGMENT_ORDER:
            seg = pf["segments"][name]
            assert seg["input"]["n"] == 33
            assert seg["production"]["coverage"]["n"] == 33
            for mode in R.CANDIDATE_MODES:
                assert seg["candidates"][mode]["mapped"]["n"] == 33
                assert seg["candidates"][mode]["displacement_from_production_px"]["n"] == 33


def test_final_frame_selection(tmp_path):
    pairs = _identity_pairs(T=65)
    plan_path, motion_path, *_ = save_case(tmp_path, name="long65", pairs=pairs, pin=False)
    rep = R.run_real_composition_audit(plan_path, motion_path, every=30, samples=51)
    assert rep["selection"]["frames"] == [0, 30, 60, 64]
    assert rep["selection"]["final_frame_included"] is True
    assert [pf["frame"] for pf in rep["per_frame"]] == [0, 30, 60, 64]


def test_report_is_json_safe_and_has_no_winner_score(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path, name="json")
    rep = R.run_real_composition_audit(plan_path, motion_path, every=2, samples=51)
    json.dumps(R.json_safe(rep), allow_nan=False)
    for bad in ("quality_score", "winner", "ranking"):
        assert bad not in rep
    assert "winner" in rep["no_composite_winner_score"] and "score" in rep["no_composite_winner_score"]
    assert "not" in rep["scope"].lower() and "perceptual" in rep["scope"].lower()
    assert rep["provenance"]["capture"]["checked_fields"] == [
        "outputs.plan", "outputs.motion",
        "experiment_identity.capture.dump_plan", "experiment_identity.capture.dump_motion"]


# --------------------------------------------------------------------------------------
# manifest provenance and scope refusals
# --------------------------------------------------------------------------------------

def test_manifest_identity_records_both_captures(tmp_path):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="ident")
    ident = R.load_manifest_identity(plan_path, motion_path)
    assert ident["capture_dump_plan"] is True and ident["capture_dump_motion"] is True
    assert ident["output_plan"].endswith("plan_r0.npz") and ident["output_motion"].endswith("motion_r0.npz")
    assert ident["manifest_path"].endswith("manifest.json") and ident["config"]


def test_refuses_false_dump_motion(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path, name="nodm", dump_motion=False)
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "manifest_dump_motion_not_captured" and "dump_motion" in str(ei.value)


def test_refuses_mismatched_motion_output_basename(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path, name="mism", outputs_motion="elsewhere/motion_r9.npz")
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "mismatched_motion_output"
    assert "motion_r9.npz" in str(ei.value) and "motion_r0.npz" in str(ei.value)


def test_refuses_missing_or_empty_output_motion(tmp_path):
    for i, (name, value) in enumerate((("no_outputs", None), ("empty_motion", ""), ("bad_type", 7))):
        plan_path, motion_path, run, *_ = save_case(tmp_path, name=f"om{i}")
        man = json.loads((run / "manifest.json").read_text())
        if value is None:
            del man["outputs"]["motion"]
        else:
            man["outputs"]["motion"] = value
        (run / "manifest.json").write_text(json.dumps(man))
        with pytest.raises(R.CompositionAuditError) as ei:
            R.run_real_composition_audit(plan_path, motion_path)
        assert ei.value.code == "manifest_malformed"


def test_refuses_missing_or_non_object_config(tmp_path):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="noconfig")
    man = json.loads((run / "manifest.json").read_text())
    del man["config"]
    (run / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "manifest_malformed" and "config" in str(ei.value)

    plan_path, motion_path, run, *_ = save_case(tmp_path, name="listconfig", config=["not", "obj"])
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "manifest_malformed"


def test_refuses_missing_and_unreadable_manifest(tmp_path):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="noman")
    (run / "manifest.json").unlink()
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "manifest_missing"

    plan_path, motion_path, run, *_ = save_case(tmp_path, name="badman")
    (run / "manifest.json").write_text("{not json")
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "manifest_unreadable"


def test_refuses_non_object_manifest(tmp_path):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="listman")
    (run / "manifest.json").write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "manifest_malformed" and "top level" in str(ei.value)


def test_refuses_artifacts_not_siblings(tmp_path):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="sib")
    other = tmp_path / "elsewhere"; other.mkdir()
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, other / "motion_r0.npz")
    assert ei.value.code == "artifacts_not_siblings"


def test_refuses_unsupported_stride(tmp_path):
    pairs = _projective_pairs()
    run = tmp_path / "stride"; run.mkdir()
    mot, plan, cfg = _motion_plan(pairs)
    mot.stride = 2
    plan_path, motion_path = run / "plan_r0.npz", run / "motion_r0.npz"
    S.save_plan(plan, str(plan_path), W, H, 30.0)
    S.save_motion_npz(mot, str(motion_path), cfg)
    write_manifest(run, plan_path, motion_path, config=_config_dict(cfg))
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "unsupported_stride"


def test_refuses_mismatched_lengths(tmp_path):
    pairs = _projective_pairs()
    run = tmp_path / "len"; run.mkdir()
    mot, plan, cfg = _motion_plan(pairs)
    mot.n_frames = mot.n_frames + 1
    plan_path, motion_path = run / "plan_r0.npz", run / "motion_r0.npz"
    S.save_plan(plan, str(plan_path), W, H, 30.0)
    S.save_motion_npz(mot, str(motion_path), cfg)
    write_manifest(run, plan_path, motion_path, config=_config_dict(cfg))
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "mismatched_lengths"


def test_refuses_config_shape_inconsistency(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path, name="badshape")
    run = plan_path.parent
    man = json.loads((run / "manifest.json").read_text())
    man["config"]["mesh"] = [8, 6]           # does not match the captured plan/motion mesh cells
    (run / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(R.CompositionAuditError) as ei:
        R.run_real_composition_audit(plan_path, motion_path)
    assert ei.value.code == "inconsistent_config"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_writes_json_and_final_frame(tmp_path):
    pairs = _identity_pairs(T=65)
    plan_path, motion_path, *_ = save_case(tmp_path, name="cli65", pairs=pairs, pin=False)
    out = tmp_path / "audit.json"
    assert R.main([str(plan_path), str(motion_path), "--every", "30", "--samples", "51", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == R.SCHEMA and rep["selection"]["frames"] == [0, 30, 60, 64]


def test_cli_prints_json_to_stdout(tmp_path, capsys):
    plan_path, motion_path, *_ = save_case(tmp_path, name="clistd")
    assert R.main([str(plan_path), str(motion_path), "--every", "2", "--samples", "51"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["ok"] is True and rep["selection"]["frames"] == [0, 2]


def test_cli_emits_json_safe_error_for_bad_motion(tmp_path, capsys):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="clierr")
    bad = run / "motion_r0.npz"
    p = base_motion_payload(); p["schema"] = np.asarray("wrong")
    np.savez_compressed(bad, **p)
    assert R.main([str(plan_path), str(bad)]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False and err["error"]["code"] == "bad_schema"
    json.dumps(err, allow_nan=False)


def test_cli_error_written_to_out_file(tmp_path):
    plan_path, motion_path, run, *_ = save_case(tmp_path, name="clierrout")
    (run / "manifest.json").write_text("{nope")
    out = tmp_path / "err.json"
    assert R.main([str(plan_path), str(motion_path), "--out", str(out)]) == 2
    err = json.loads(out.read_text())
    assert err["ok"] is False and err["error"]["code"] == "manifest_unreadable"


def test_cli_requires_both_paths(tmp_path):
    plan_path, motion_path, *_ = save_case(tmp_path, name="cliargs")
    with pytest.raises(SystemExit):
        R.main([str(plan_path)])
    with pytest.raises(SystemExit):
        R.main([])
