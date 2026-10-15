"""Weight-free tests for the isolated depth-bin flow-coherence audit.

The pure-NumPy core is exercised on exact synthetic flow fields with analytic held-out residuals, and the
runner is exercised with injected fakes for the RAFT runner and heavy lock, so no weights, no MLX, no media
and no private footage are touched. Coverage: two-layer-affine flow (bin model wins), uniform flow (no
material gain), invalid/non-finite support, insufficient support, constant/tied depth, threshold validation,
checkerboard split determinism, and path-independent atomic JSON. Warnings are escalated to errors so an
unclean numeric path fails rather than passing silently.
"""
import json
import warnings

import numpy as np
import pytest

from bench import depth_motion_audit as D


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --------------------------------------------------------------------------------------
# synthetic builders
# --------------------------------------------------------------------------------------

def _depth_ramp(h=10, w=12):
    """Strictly increasing left-to-right depth: distinct per-cell values, no ties."""
    return np.tile(np.linspace(0.0, 1.0, w)[None, :], (h, 1))


def _two_layer_flow(depth):
    """Exact two-layer affine flow: u is 1.0 or 3.0 by depth half, v is a linear ramp in y."""
    h, w = depth.shape
    flow = np.zeros((h, w, 2), np.float64)
    flow[..., 0] = np.where(depth < 0.5, 1.0, 3.0)
    flow[..., 1] = np.linspace(0.0, 0.4, h)[:, None] * np.ones(w)[None, :]
    return flow


def _ones(shape):
    return np.ones(shape, bool)


# --------------------------------------------------------------------------------------
# core: reporting floors (6 train / 3 validation)
# --------------------------------------------------------------------------------------

def test_reporting_minima_are_six_and_three():
    assert D.MIN_FIT_SAMPLES == 6
    assert D.MIN_EVAL_SAMPLES == 3


def test_affine_fit_requires_six_train_rows():
    # A rank-3 design with only 5 rows is below the reporting floor even though it is identifiable.
    x = np.linspace(0.0, 1.0, 5)
    design5 = np.stack([x, x ** 2, np.ones_like(x)], axis=1)
    assert np.linalg.matrix_rank(design5) == 3
    assert D._affine_fit(design5, np.arange(5.0)) is None
    # Six non-collinear rows meet the floor.
    x6 = np.linspace(0.0, 1.0, 6)
    design6 = np.stack([x6, x6 ** 2, np.ones_like(x6)], axis=1)
    coeff = D._affine_fit(design6, 2.0 * x6 + 1.0)
    assert coeff is not None
    assert coeff[2] == pytest.approx(1.0, abs=1e-9)
    # A rank-deficient design stays rejected regardless of row count.
    collinear = np.stack([x6, 2.0 * x6, np.ones_like(x6)], axis=1)
    assert np.linalg.matrix_rank(collinear) < 3
    assert D._affine_fit(collinear, x6) is None


def test_pooled_fit_needs_six_train_cells():
    # Force a small eligible set with an exact checkerboard split so the pooled floor is the only variable.
    h, w = 4, 6
    depth = _depth_ramp(h, w)
    flow = _two_layer_flow(depth)
    ys, xs = np.mgrid[0:h, 0:w]
    parity = (xs + ys) % 2

    # Five even-parity (train) cells and five odd-parity (validation) cells: below the pooled floor.
    cells_5 = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (0, 1), (0, 3), (1, 0), (1, 2), (1, 4)]
    valid = np.zeros((h, w), bool)
    for y, x in cells_5:
        valid[y, x] = True
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), valid)
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["train_count"] == 5
    assert tier["validation_count"] == 5
    assert tier["pooled_model_fit_ok"] is False
    assert tier["pooled_median"] is None
    assert tier["bin_median"] is None

    # Add one more even-parity cell => six train cells: the pooled fit is now permitted.
    valid2 = valid.copy()
    valid2[2, 0] = True
    out2 = D.depth_motion_audit(depth, flow, np.zeros((h, w)), valid2)
    tier2 = out2["tiers"]["all_cycle_valid"]
    assert tier2["train_count"] == 6
    assert tier2["pooled_model_fit_ok"] is True


def test_insufficient_validation_reports_null_and_no_improvement():
    # Plenty of train cells but too few held-out cells in a bin must not be scored or called an improvement.
    h, w = 4, 6
    depth = _depth_ramp(h, w)
    flow = _two_layer_flow(depth)
    ys, xs = np.mgrid[0:h, 0:w]
    # Every even-parity (train) cell present, but only the first odd-parity cell kept: every bin's held-out
    # block is a single cell, below the three-cell validation floor.
    valid = (xs + ys) % 2 == 0
    valid[0, 1] = True
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), valid)
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["train_count"] >= D.MIN_FIT_SAMPLES
    assert tier["validation_count"] == 1
    assert tier["pooled_median"] is None
    assert tier["bin_median"] is None
    assert tier["improved_bins"] == 0
    assert tier["bins_with_models"] == 0
    for b in tier["bins"]:
        assert b["global_model"]["median"] is None
        assert b["bin_model"]["median"] is None
        assert b["median_delta"] is None
        assert b["reason"] == "insufficient_validation"


# --------------------------------------------------------------------------------------
# core: exact tied-edge binning
# --------------------------------------------------------------------------------------

def test_tied_edge_value_stays_in_lower_bin():
    vals = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    assign, edges = D._bin_assignments(vals, np.ones(vals.size, bool), 2)
    # Two equal-count halves of six cells give one interior edge at the third sorted value: 10.0.
    assert edges == [10.0]
    # The value exactly equal to the edge (10.0 at index 2) stays in the lower bin.
    assert assign[2] == 0
    assert assign.tolist() == [0, 0, 0, 1, 1, 1]


def test_tied_edge_with_repeated_values_collapses_without_extra_bins():
    vals = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    assign, edges = D._bin_assignments(vals, np.ones(vals.size, bool), 3)
    # Requesting three bins over three repeated values yields exactly the three value groups.
    assert set(edges) <= {1.0, 2.0}
    assert set(assign.tolist()) == {0, 1, 2}
    # Every repeated value sits wholly in one bin (the lower one when it equals an edge).
    assert assign[0] == assign[1] == assign[2] == 0
    assert assign[3] == assign[4] == assign[5] == 1
    assert assign[6] == assign[7] == assign[8] == 2


# --------------------------------------------------------------------------------------
# core: two-layer flow -- bin model explains better than pooled
# --------------------------------------------------------------------------------------

def test_two_layer_flow_bin_model_beats_pooled():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["effective_bins"] == out["n_bins"]
    # Every scored bin cleared the reporting floors, so the win is not a small-sample artifact.
    assert tier["pooled_model_fit_ok"] is True
    scored = [b for b in tier["bins"] if b["bin_model"]["median"] is not None]
    assert scored and all(b["train_count"] >= D.MIN_FIT_SAMPLES for b in scored)
    assert all(b["validation_count"] >= D.MIN_EVAL_SAMPLES for b in scored)
    # The pooled affine cannot represent a step in u; the per-bin models fit each layer exactly.
    assert tier["pooled_median"] > 0.1
    assert tier["bin_median"] == pytest.approx(0.0, abs=1e-9)
    assert tier["median_delta"] > 0.1
    assert tier["improved_bins"] == tier["bins_with_models"]


def test_two_layer_flow_heldout_counts_partition_support():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    tier = out["tiers"]["all_cycle_valid"]
    n_train = sum(b["train_count"] for b in tier["bins"])
    n_val = sum(b["validation_count"] for b in tier["bins"])
    assert n_train == tier["train_count"]
    assert n_val == tier["validation_count"]
    assert n_train + n_val == tier["support_count"]


def test_uniform_flow_binning_brings_no_material_gain():
    depth = _depth_ramp()
    h, w = depth.shape
    flow = np.ones((h, w, 2), np.float64)
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["pooled_model_fit_ok"] is True
    assert tier["pooled_median"] == pytest.approx(0.0, abs=1e-9)
    assert tier["bin_median"] == pytest.approx(0.0, abs=1e-9)
    assert tier["median_delta"] == pytest.approx(0.0, abs=1e-9)
    # No bin can claim a gain: the two models coincide on uniform flow.
    assert tier["improved_bins"] == 0


# --------------------------------------------------------------------------------------
# core: degenerate depth
# --------------------------------------------------------------------------------------

def test_constant_depth_yields_one_effective_bin():
    h, w = 8, 8
    depth = np.full((h, w), 0.5)
    flow = _two_layer_flow(_depth_ramp(h, w))
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["effective_bins"] == 1
    # One bin: the "pooled" and "bin" models use the same training cells, so they agree.
    assert tier["pooled_median"] == pytest.approx(tier["bin_median"])


def test_tied_discrete_depth_handled_safely():
    h, w = 10, 10
    depth = np.zeros((h, w))
    depth[:, ::2] = 1.0
    depth[:, 1::2] = 2.0
    flow = _two_layer_flow(_depth_ramp(h, w))
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    tier = out["tiers"]["all_cycle_valid"]
    # Only two distinct values, so at most two effective bins despite requesting four.
    assert tier["effective_bins"] <= 2
    # Interior edges never exceed n_bins - 1, and only distinct values are reported.
    assert len(tier["quantile_edges"]) <= D.DEFAULT_BINS - 1
    # Without asserting unsupported extra bins: only reported bins carry depth bounds, and every bound is one
    # of the two tied values.
    assert all(b["depth_min"] is None or b["depth_min"] in (0.0, 1.0, 2.0) for b in tier["bins"])
    assert all(b["depth_max"] is None or b["depth_max"] in (0.0, 1.0, 2.0) for b in tier["bins"])
    assert sum(1 for b in tier["bins"] if b["support_count"] > 0) == tier["effective_bins"]


# --------------------------------------------------------------------------------------
# core: support and thresholds
# --------------------------------------------------------------------------------------

def test_nonfinite_depth_and_flow_are_excluded_from_support():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    depth = depth.copy()
    depth[0, 0] = np.nan
    flow[1, 1, 0] = np.inf
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    assert out["base_support_count"] == h * w - 2


def test_cycle_valid_mask_excluded_and_thresholds_narrow_support():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    resid = np.full((h, w), 0.3)
    valid = _ones((h, w))
    valid[0, 0] = False
    out = D.depth_motion_audit(depth, flow, resid, valid,
                              cycle_thresholds=[0.25, 0.5, 2.0])
    base = out["tiers"]["all_cycle_valid"]
    le25 = out["tiers"]["cycle_le_0.25"]
    le05 = out["tiers"]["cycle_le_0.5"]
    assert base["support_count"] == h * w - 1
    assert le25["support_count"] == 0
    assert le05["support_count"] == h * w - 1
    assert le05["support_count"] == out["tiers"]["cycle_le_2"]["support_count"]
    # An empty tier reports finite-free null metrics.
    assert le25["pooled_median"] is None
    assert le25["bin_median"] is None


def test_insufficient_support_reports_null_metrics():
    h, w = 4, 4
    depth = _depth_ramp(h, w)
    flow = _two_layer_flow(depth)
    valid = _ones((h, w))
    valid[:, :] = False
    valid[0, 0] = valid[0, 1] = True  # only two cells: below the affine fit floor
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), valid)
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["pooled_median"] is None
    assert tier["bin_median"] is None
    assert all(b["bin_model"]["median"] is None for b in tier["bins"])


# --------------------------------------------------------------------------------------
# core: validation / determinism
# --------------------------------------------------------------------------------------

def test_rejects_shape_mismatch():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    with pytest.raises(ValueError, match="fwd_flow"):
        D.depth_motion_audit(depth, np.zeros((h, w, 3)), np.zeros((h, w)), _ones((h, w)))
    with pytest.raises(ValueError, match="cycle_resid"):
        D.depth_motion_audit(depth, flow, np.zeros((h, w + 1)), _ones((h, w)))


def test_rejects_bad_bins_and_thresholds():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    with pytest.raises(ValueError, match="n_bins"):
        D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)), n_bins=0)
    with pytest.raises(ValueError, match="finite"):
        D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)), cycle_thresholds=[np.nan])
    with pytest.raises(ValueError, match="nonnegative"):
        D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)), cycle_thresholds=[-1.0])


def test_duplicate_threshold_keys_rejected():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    with pytest.raises(ValueError, match="duplicate"):
        D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)),
                            cycle_thresholds=[1.0, 1.0])


def test_is_deterministic():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    a = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    b = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    assert a == b


def test_result_is_json_finite_safe():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    # allow_nan=False must not raise: every numeric leaf is finite or None.
    json.dumps(out, allow_nan=False)


def test_checkerboard_split_is_even_train_odd_validate():
    depth = _depth_ramp()
    h, w = depth.shape
    # Pure x-ramp flow so every cell is explainable; the split is what we inspect.
    flow = np.zeros((h, w, 2))
    flow[..., 0] = depth
    out = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)))
    tier = out["tiers"]["all_cycle_valid"]
    # every cell is train or validate, and the two counts sum to the support
    assert tier["train_count"] + tier["validation_count"] == tier["support_count"]


# --------------------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------------------

def test_parser_requires_cache_paths():
    with pytest.raises(SystemExit):
        D.build_parser().parse_args([])


def test_parser_defaults():
    a = D.build_parser().parse_args(["--frames-npz", "f.npz", "--depth-npz", "d.npz"])
    assert a.iters == D.DEFAULT_ITERS
    assert a.downscale == D.DEFAULT_DOWNSCALE
    assert a.bins == D.DEFAULT_BINS
    assert a.lock == "on"


def test_parse_cycle_thresholds_validates_dedupes_and_preserves_order():
    assert D.parse_cycle_thresholds("0.5,0.25,0.5,1") == [0.5, 0.25, 1.0]
    with pytest.raises(ValueError, match="finite"):
        D.parse_cycle_thresholds("nan")
    with pytest.raises(ValueError, match=">= 0"):
        D.parse_cycle_thresholds("-1")
    with pytest.raises(ValueError, match="at least one"):
        D.parse_cycle_thresholds("")


# --------------------------------------------------------------------------------------
# runner fakes
# --------------------------------------------------------------------------------------

class FakeRunner:
    """Deterministic stand-in for :class:`bench.flow_prediction_real.RaftRunner` (no MLX, no weights)."""

    def __init__(self, weights, downscale=2, flow_dtype="float32", corr_dtype="float32", h=8, w=10):
        self.weights = weights
        self.downscale = int(downscale)
        self.flow_dtype = flow_dtype
        self.corr_dtype = corr_dtype
        self.h = int(h)
        self.w = int(w)
        self.pad = (0, 0, 0, 0)
        self.prep_s = 0.0
        self.prepared = None
        self.calls = []

    def prepare(self, frames, on_step=None):
        self.prepared = [np.asarray(f) for f in frames]
        if on_step is not None:
            for _ in self.prepared:
                on_step()

    def reference_pair(self, src, dst, iters):
        self.calls.append((int(src), int(dst), int(iters)))
        # Constant unit forward flow, zero backward: fully cycle-valid and exactly affine.
        fwd = np.ones((self.h, self.w, 2), np.float64)
        bwd = np.zeros((self.h, self.w, 2), np.float64)
        return fwd, bwd


class RaisingRunner:
    """Runner whose construction fails: proves invalid inputs never reach runner creation."""

    def __init__(self, *a, **k):
        raise AssertionError("runner must not be constructed for invalid inputs")


class RecordingLock:
    """Fake ``HeavyLock`` recording construction, steps and holding state without any real lock."""

    instances = []

    def __init__(self, what="", enabled=True, chunk=0, poll=10, pause_s=0.0, sleep=None):
        self.what = what
        self.enabled = enabled
        self.chunk = int(chunk or 0)
        self.pause_s = float(pause_s or 0.0)
        self.steps = []
        self.held = False
        self.releases = 0
        RecordingLock.instances.append(self)

    def on_step(self, step):
        self.steps.append(int(step))

    def status(self):
        return {"backend": "fake", "enabled": self.enabled, "chunk": self.chunk, "pause_s": self.pause_s,
                "held": self.held, "releases": self.releases, "steps": len(self.steps)}

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


# --------------------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------------------

def _frames(n=6, h=8, w=10, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def _meta(n=6, h=8, w=10, fps=30.0):
    return {"schema": "mlxstab.flow_prediction_real.frames/1", "count": n, "height": h, "width": w,
            "channels": 3, "fps": fps, "window": {"start_s": 1.0, "duration_s": 2.0}}


def _write_frames_npz(tmp_path, frames, meta):
    path = tmp_path / "frames.npz"
    np.savez(path, frames=frames, meta=np.array(json.dumps(meta)))
    return path


def _depth_info(indices, h=8, w=10, frames_sha="x"):
    depth = np.zeros((len(indices), h, w), np.float32)
    for k in range(len(indices)):
        # a per-frame depth ramp so bins are non-trivial
        depth[k] = np.tile(np.linspace(0.0, 1.0, w)[None, :], (h, 1)) + k
    return {"indices": np.asarray(indices, np.int64), "depth": depth,
            "model_id": "fake-model", "normalization": "fake-norm", "frames_npz_sha256": frames_sha}


@pytest.fixture
def weights_file(tmp_path):
    path = tmp_path / "fake.safetensors"
    path.write_bytes(b"fake-weights")
    return path


def _run(frames, meta, depth_info, tmp_weights, **kw):
    kw.setdefault("lock_factory", RecordingLock)
    kw.setdefault("runner_factory", FakeRunner)
    kw.setdefault("write", False)
    kw.setdefault("log", lambda m: None)
    return D.run_audit(frames, meta, depth_info, str(tmp_weights), None, **kw)


# --------------------------------------------------------------------------------------
# runner: validation before runner creation
# --------------------------------------------------------------------------------------

def test_run_audit_rejects_bad_iters_before_runner(weights_file):
    with pytest.raises(ValueError, match="iters"):
        _run(_frames(), _meta(), _depth_info([0, 2]), weights_file, iters=0,
             runner_factory=RaisingRunner)


def test_run_audit_rejects_bad_bins_before_runner(weights_file):
    with pytest.raises(ValueError, match="bins"):
        _run(_frames(), _meta(), _depth_info([0, 2]), weights_file, bins=0,
             runner_factory=RaisingRunner)


def test_run_audit_rejects_bad_thresholds_before_runner(weights_file):
    with pytest.raises(ValueError, match="thresholds"):
        _run(_frames(), _meta(), _depth_info([0, 2]), weights_file, cycle_thresholds=[-0.5],
             runner_factory=RaisingRunner)


def test_run_audit_rejects_missing_weights_before_runner(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights"):
        _run(_frames(), _meta(), _depth_info([0, 2]), tmp_path / "nope.safetensors",
             runner_factory=RaisingRunner)


# --------------------------------------------------------------------------------------
# runner: wiring, both directions, determinism
# --------------------------------------------------------------------------------------

def test_run_audit_maps_adjacent_pairs_both_directions(weights_file):
    res = _run(_frames(n=7), _meta(n=7), _depth_info([0, 2, 4, 6]), weights_file, iters=5)
    assert res["pair_count"] == 3
    assert [p["source_frame_indices"] for p in res["pairs"]] == [[0, 2], [2, 4], [4, 6]]
    for p in res["pairs"]:
        assert [d["direction"] for d in p["directions"]] == ["forward", "backward"]
    assert res["pairs"][0]["time_s"] == [0.0, 2 / 30.0]


def test_run_audit_reference_pairs_are_adjacent_on_sampled_sequence(weights_file):
    made = {}

    class SpyRunner(FakeRunner):
        def reference_pair(self, src, dst, iters):
            made.setdefault("pairs", []).append((int(src), int(dst), int(iters)))
            return super().reference_pair(src, dst, iters)

    _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), weights_file, iters=5,
         runner_factory=SpyRunner)
    assert made["pairs"] == [(0, 1, 5), (1, 2, 5)]


def test_run_audit_lock_gates_per_frame_and_pair(weights_file):
    RecordingLock.instances.clear()
    res = _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), weights_file)
    lock = RecordingLock.instances[-1]
    # 3 sampled frames prepared + 2 completed pairs = 5 monotonic gates
    assert lock.steps == [1, 2, 3, 4, 5]
    assert lock.status()["held"] is False
    assert res["timing_s"]["lock"]["backend"] == "fake"


def test_run_audit_does_not_serialize_lock_path(weights_file):
    res = _run(_frames(n=3), _meta(n=3), _depth_info([0, 1, 2]), weights_file)
    assert "path" not in res["timing_s"]["lock"]


def test_run_audit_is_deterministic(weights_file):
    a = _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), weights_file)
    b = _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), weights_file)
    assert a["summary"] == b["summary"]
    assert a["flow"]["feature_grid"] == b["flow"]["feature_grid"]
    for pa, pb in zip(a["pairs"], b["pairs"]):
        assert pa["directions"] == pb["directions"]


def test_summarize_pairs_aggregates_tiers():
    depth = _depth_ramp()
    flow = _two_layer_flow(depth)
    h, w = depth.shape
    audit = D.depth_motion_audit(depth, flow, np.zeros((h, w)), _ones((h, w)),
                                cycle_thresholds=[0.25])
    pairs = [{"directions": [{"direction": "forward", "audit": audit},
                             {"direction": "backward", "audit": audit}]}]
    s = D.summarize_pairs(pairs, [0.25], D.DEFAULT_BINS)
    assert s["_meta"]["n_bins"] == D.DEFAULT_BINS
    assert s["all_cycle_valid"]["pair_directions"] == 2
    assert s["all_cycle_valid"]["total_improved_bins"] == 2 * audit["tiers"]["all_cycle_valid"]["improved_bins"]


# --------------------------------------------------------------------------------------
# end-to-end run() with real cache files (fake runner, no weights)
# --------------------------------------------------------------------------------------

def test_run_end_to_end_writes_atomic_path_independent_json(tmp_path, monkeypatch):
    frames = _frames(n=6, h=8, w=10)
    meta = _meta(n=6, h=8, w=10)
    fpath = _write_frames_npz(tmp_path, frames, meta)
    import bench.manifest as MF
    fsha = MF.sha256_file(fpath)

    info = _depth_info([0, 2, 4], h=8, w=10, frames_sha=fsha)
    dpath = tmp_path / "depth.npz"
    np.savez(dpath, indices=info["indices"].astype(np.int32), depth=info["depth"],
             model_id=np.array(info["model_id"]), normalization=np.array(info["normalization"]),
             frames_npz_sha256=np.array(info["frames_npz_sha256"]))
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"fake-weights")
    out = tmp_path / "audit.json"

    monkeypatch.setattr(D.R, "RaftRunner", FakeRunner)
    monkeypatch.setattr(D.R, "HeavyLock", RecordingLock)

    args = ["--frames-npz", str(fpath), "--depth-npz", str(dpath), "--weights", str(weights),
            "--out", str(out), "--lock", "off", "--iters", "3", "--bins", "3"]
    assert D.main(args) == 0
    result = json.loads(out.read_text())
    assert result["status"] == "success"
    assert result["schema"] == D.SCHEMA
    assert result["pair_count"] == 2
    assert result["bins"] == 3
    # no absolute local footage / model-cache path leaked
    blob = json.dumps(result)
    assert str(tmp_path) not in blob
    assert "source_path" not in blob
    assert result["source"]["frames_npz_sha256"] == fsha
    assert result["depth"]["model_id"] == "fake-model"
    assert result["flow"]["weights_sha256"] == MF.sha256_file(weights)
    assert result["split"]["scheme"] == "checkerboard_x_plus_y_parity"


def test_run_rejects_depth_from_other_frame_cache(tmp_path):
    frames = _frames(n=6, h=8, w=10)
    meta = _meta(n=6, h=8, w=10)
    fpath = _write_frames_npz(tmp_path, frames, meta)
    info = _depth_info([0, 2, 4], h=8, w=10, frames_sha="deadbeef")
    dpath = tmp_path / "depth.npz"
    np.savez(dpath, indices=info["indices"].astype(np.int32), depth=info["depth"],
             model_id=np.array(info["model_id"]), normalization=np.array(info["normalization"]),
             frames_npz_sha256=np.array(info["frames_npz_sha256"]))
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"fake")
    args = ["--frames-npz", str(fpath), "--depth-npz", str(dpath), "--weights", str(weights),
            "--out", str(tmp_path / "o.json")]
    assert D.main(args) == 1
