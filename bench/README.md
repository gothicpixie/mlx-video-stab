# Stage 0 benchmark runner

`bench/run_case.py` runs one **explicit** stabilization case over a synthetic clip or a local video window
and writes one atomic, JSON-safe manifest with enough provenance to invalidate a comparison. It is Stage 0
measurement infrastructure, not a quality claim: it does not compute a single score, does not change
stabilization defaults, and records failures and missing support explicitly.

## Cases

Each case changes one variable relative to the historical practical preset
(`flow_downscale=2`, `flow_iters=20`, `fidelity=0.1`, `analysis_stride=1`, fp32, warm start on, output and
metrics at `30000/1001`; **not** the literal current `Config` defaults of full-res flow / 24 iterations).
`combined-fp16` (two precision fields) and `global-only` (two residual channels) deliberately change two
fields each and must be read alongside their isolated counterparts.

| case | changed variable | meaning |
| --- | --- | --- |
| `baseline` | none | historical practical preset (full mesh + rows) |
| `fp16-network` | `flow_dtype=float16` | fp16 encoders/update block, fp32 correlation |
| `fp16-corr` | `corr_dtype=float16` | fp32 network, fp16 correlation pyramid |
| `combined-fp16` | `flow_dtype` + `corr_dtype` | fp16 network and fp16 correlation pyramid (two variables) |
| `warm-8` | `warm_iters=8` | 8 iterations once warm-started (cold pair still 20) |
| `stride-2` | `analysis_stride=2` | analyse every second frame, interpolate the path |
| `global-only` | `mesh` + `rs_rows` = 0 | both residual channels disabled; global model refit |
| `mesh-only` | `rs_rows=0` | rolling-shutter rows disabled; global + mesh refit |
| `row-only` | `mesh=(0,0)` | residual mesh disabled; global + rows refit |

The residual-channel cases (`global-only`, `mesh-only`, `row-only`) are practical configuration ablations,
**not** fixed-plan causal isolation: disabling a channel makes the fit re-estimate the remaining motion
fields, so the comparison measures the configuration's output rather than the isolated effect of a frozen
plan. `global-only` disables two channels and is only interpretable alongside the isolated `mesh-only` and
`row-only` cases. These observations still require rigid-structure/visual review and 10 s-or-longer evaluation
before any decision.

`--set KEY=VALUE` overrides a single `Config` field; `--config-json` accepts a JSON object. Unknown fields
are rejected. `--seed` sets both synthetic clip generation and the per-clip fitting RNG (`--set seed=`
overrides the fitting seed). Config identity is canonical JSON (sorted keys, tuples as lists) under SHA-256,
so any field change invalidates a manifest.

## Usage

Public synthetic smoke (no private footage, heavy lock off, small):

```bash
python -I bench/run_case.py --case baseline --source synthetic --duration 1 --synth-frames 30
```

Local file source (footage stays local; keep outputs in this repo's ignored `out/`):

```bash
python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS \
    --start 60 --duration 1 --diag-every 15 \
    --out-dir out/bench/smoke_t060 --log out/bench/smoke_t060.log
```

Full 10 s window, shared heavy lock held for the run:

```bash
python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS \
    --start 60 --duration 10 --lock on --out-dir out/bench/baseline_t060 --log out/bench/baseline_t060.log
```

List cases: `python -I bench/run_case.py --list-cases`. Inspect a case without running:
`python -I bench/run_case.py --dry-run --case warm-8`.

Useful options:

- `--start`, `--duration` – precise source window.
- `--eval-iters`, `--eval-downscale`, `--eval-flow-dtype`, `--eval-corr-dtype` – evaluation flow.
- `--output-fps VALUE` / `--eval-vf VALUE` – override the case output cadence (rational, or `native`) and the
  evaluation `fps` filter. The resolved values are recorded in `case` and folded into `experiment_identity`.
- `--overwrite` – reuse an existing run directory. By default an existing `manifest.json` is protected and the
  CLI returns a structured nonzero failure instead of clobbering evidence. Default run directories carry a
  microsecond stamp and are unique.
- `--selection-note TEXT` – provenance note (e.g. the 60/800 s windows were inferred from historical output
  filenames, not a recovered original manifest). Started in `source.selection_note` and the identity.
- `--repeats N` – one cold run plus `N-1` warm runs in the same process; each repeat writes distinct
  `stabilized_r{i}.mp4` / `mask_r{i}.mp4` / `report_r{i}.json` and records its model/temporal state and wall
  time. `rendered_frames` (writes) and the separately measured `encoded_frames` are kept distinct. The
  retained output for evaluation is repeat 0 (`outputs.retained_repeat`).
- `--diag-every K` – sample complete-warp diagnostics every `K` rendered frames. Mutually exclusive with
  `Config.warp_diag` (all frames): the runner rejects the combination during validation.
- `--dump-plan` – write the retained repeat-0 plan to `<out-dir>/plan_r0.npz` for later shape analysis. The
  file is a compressed NPZ (no pickle; loads with `np.load(path, allow_pickle=False)`) holding the schema tag,
  frame width/height, frame count, analysis fps, per-frame global homographies `G`, `mesh_corr`, `row_corr`,
  `raw_quads`, `path_quads` and the fallback count. It is derived camera/motion geometry of the footage, so
  treat it as private exactly like the manifest (keep it in the ignored `out/`); it carries no pixels but can
  be larger than the manifest and scales with frame count × mesh/row resolution. Only the retained repeat is
  captured (warm repeats never clobber `plan_r0.npz`), and an existing dump is protected unless `--overwrite`
  is passed. The option is recorded in `experiment_identity.capture` so toggling it changes the `content_key`.
- `--dump-motion` – write the retained repeat-0 fitted motion to `<out-dir>/motion_r0.npz` for the
  real-motion composition audit. The file is a compressed NPZ (no pickle; loads with
  `np.load(path, allow_pickle=False)`) holding the schema tag (`mlxstab.motion/4`), frame dimensions,
  `n_frames`, analysis fps, `stride`, the fit count and the stacked per-pair fits (`H [n_fits,3,3]`, `mesh`,
  `rows`, `inlier_frac`, `mover_frac`, `fallback`) plus the per-fit gate diagnostics
  (`n_valid_samples [n_fits]`, `n_ransac_inliers [n_fits]`, the integer `fallback_reason_code [n_fits]` and the
  per-fit `fit_meta_available [n_fits]` flag), the observation-coverage support masses
  (`mesh_support_mass [n_fits,gr+1,gc+1]`, `row_support_mass [n_fits,nr]`; a Fit built without them dumps
  shape-correct zeros) and the forward-backward cycle-consistency maps
  (`cycle_err_rms`, `cycle_inconsistency_frac`, `cycle_sample_mass`, each `[n_fits,gr+1,gc+1]`, and the per-fit
  `cycle_map_available [n_fits]` flag; a Fit built without them dumps zero maps tagged unavailable). Support
  mass is the non-robust kernel-weighted count of usable non-mover flow samples at each mesh vertex / row
  band, recorded before diffusion -- an observation-coverage diagnostic, **not** a probability and **not** flow
  confidence. The cycle maps are a forward-backward **cycle-consistency** diagnostic only (weighted RMS of
  `||f + sampled_b||`, weighted fraction of in-bounds samples failing the existing `fb_abs + fb_rel*||f||`
  tolerance, and spatial sample mass), built from every finite, in-bounds forward endpoint independently of the
  fit's valid/non-mover mask -- **not** a true correspondence error, **not** ground-truth error, **not** a
  probability and **not** calibrated uncertainty; they never feed stabilization or fitting. The payload stores
  `fallback_reason_code`, one of the stable integer codes `0..3` (`0` = `none`, `1..3` = the first failing fit
  gate), with `-1` for an unavailable fit, and the audit maps it back to the human-readable `fallback_reason`
  label; a manually built Fit dumps the `-1, -1, -1` sentinels tagged `fit_meta_available=False`. These are
  diagnostics only and do not change the fit or pixels.
  The loader still
  accepts legacy `mlxstab.motion/3` (support + cycle, no fit metadata), `mlxstab.motion/2` (support, no cycle
  maps) and `mlxstab.motion/1` (neither) payloads, loading the missing fields as `None`, and reads the
  pre-standardization `/4` reason-code key `fallback_reason` as an alias for `fallback_reason_code` so the
  preserved just-captured payload still loads. Like the plan it is
  private derived motion geometry: keep it in the
  ignored
  `out/`. Only the retained repeat is captured, an existing dump is protected unless `--overwrite` is passed,
  and the option is recorded in `experiment_identity.capture` so toggling it changes the `content_key`. Plan
  and motion capture are independent; `--dump-plan` alone keeps the plan-only comparison contract.
- `--lock {auto,on,off}`, `--lock-chunk N`, `--lock-pause S` – shared heavy lock. `auto` uses the lock for
  file sources and not for synthetic. A `chunk` releases the lock, yields for `--lock-pause` seconds (the
  shared-project etiquette is ~30 frames with a ~12 s gap) and re-acquires; acquisition wait and pause are
  accounted separately from processing. The default holds the lock for the whole run and always releases in
  a `finally`; cancellation writes a `cancelled` manifest before re-raising.

## Cadence

The historical preset renders at the source processing rate and encodes the output at `30000/1001` using an
explicit FFmpeg `fps` filter (`-vf fps=... -r ... -fps_mode cfr`). Relying on output `-r` alone was wrong: it
produced 32 frames for a 1 s 60 fps clip at 30 fps. Conversion is validated by
`tests/test_media_cadence.py` for 60→30 and 60000/1001→30000/1001, including mask alignment. `stabilize`
reports `rendered_frames` (writes) and the runner measures the true `outputs.encoded_frames` by probing the
finished file; the two are never conflated.

Metrics use the same `fps=RATE` filter on input, output and mask. Their complete decoded frame counts and
normalized frame-index timelines must agree. Separately, every encoded video and mask PTS is checked against
the requested zero-origin CFR clock, with a 10-microsecond tolerance for ffprobe's decimal rounding. This
detects shifted clocks, missing timestamps, duplicate/interior timing errors and extra mask tails. The
encoded rate may differ from evaluation: 60 fps video/mask can correctly be evaluated at 30 fps. Output
and evaluation rates must be positive; arbitrary filter chains are not supported by `--eval-vf`.

The source decoder does not expose original frame PTS. The benchmark records that limitation and uses
normalized frame-index time for evaluation; it does not validate original VFR timing or absolute source
timestamps. Raw container `start_time`/`time_base` are retained as provenance.

The `Config` field `analysis_stride` controls analysis cadence separately from output cadence.

## Diagnostics

Complete-warp diagnostics are opt-in and expensive. `--diag-every K` samples the **actual full-window plan of
the retained repeat** during rendering (no extra RAFT inference) at exactly the render indices where
`t % K == 0`. Each sampled frame is rendered once without diagnostics and the `warpmetrics.diagnose` call is
timed separately; the manifest stores the exact frame indices, per-frame diagnostic dictionaries, aggregates
and the separated overhead. `Config.warp_diag=True` (all-frame diagnostics) is mutually exclusive with
`--diag-every` and the runner rejects the combination. `render` minus the diagnostic total is labelled
**observed non-diagnostic interval time**, not counterfactual throughput, because encoding overlaps with
rendering. Per-sample maxima are labelled as such and are not whole-video guarantees.

## Manifest

Each run writes `<out-dir>/manifest.json` (atomic temp-write + `os.replace`). It contains:

- `schema`, `created`, `status` (`success`/`failure`), the most recently persisted `stage`, and `error` (`type`,
  `message`, `stage`). Stage checkpoints name the phase before each long operation: `model_construct` is
  persisted before RAFT is constructed, `stabilize_r{i}` immediately before each repeat's stabilization pass
  (so a failure identifies the exact repeat), and `probe_output`/`evaluate` before output probing and
  evaluation. A long run can therefore be told apart from a stalled model load, a stalled stabilization pass,
  or a stalled probe/evaluation, rather than appearing frozen at the last acquisition stage.
- `case` (name, changed variable, output/eval cadence), full resolved `config`, `config_digest`, the expanded
  `experiment_identity` (source + window + preprocessing, weights, config, output/eval cadence, evaluation
  parameters, seed policy, diagnostic selection, working-tree code digests) and its `content_key`.
- `source` (sha256, ffprobe metadata, window, preprocessing, decoded frame count/fps/duration) and `weights` sha256.
- `code` (git revision + sha256 of `mlxstab/**`, `bench/**`, `pyproject.toml`, plus a relevant-dirty flag that
  ignores unrelated docs), `evaluation` (eval flow precision/iters/downscale, per-clip reset, metric config)
  and `metrics.definitions_version` bound to the metric source hashes, and `seed_policy`.
- `environment` (Python/MLX/Torch/NumPy/SciPy/FFmpeg/ffprobe) and `hardware`.
- `timing` (provenance, model construction, repeat-0 wall, end-to-end stabilization including model load and
  encoder close, end-to-end with evaluation, and a `lock_scope` note that acquisition waits/pauses are inside
  `total_wall_s` and not subtracted from stabilization) and `runtime_s` (analysis/plan/render stages,
  evaluation, diagnostics total, observed non-diagnostic render time, total wall) plus per-repeat
  `model_state`/`temporal_state`/`rendered_frames`/`encoded_frames`/wall time and distinct output paths.
- `outputs` (`encoded_frames`, `encoded_pts`, `mask_pts`, cadence, and, with `--dump-plan`/`--dump-motion`,
  the `plan_r0.npz`/`motion_r0.npz` paths) and `source.selection_note`.
- `memory` (process peak RSS and MLX peak, each with scope, else `null` with a reason), `metrics` (input/
  output, cadence with validated timelines, reliability + support stats and reasons), `diagnostics`, `lock`
  (backend and acquire/pause accounting), and `unavailable` reasons for anything not measured.

Unavailable measurements are written as `null` with an explanation, never as zero. Serialization uses
`allow_nan=False`; non-finite values are converted to `null` and never emitted as `NaN`/`Infinity`.

## Reliability

`reliability()` marks a comparison unusable (never "stable") when cadence is misaligned, when required
metrics are missing or non-finite, when explicit support counters (`support_frames`, `inlier_frac_mean`,
`fallback_frames`) are absent or show no adequate support, or when any motion fit fell back to an identity
transform. Failed fits can otherwise score as perfectly stable footage. The manifest keeps the input and
output metric dicts and support statistics separate; there is no aggregate score.

## Locking and private data

For file sources the runner reuses the anniversary project's shared lock (`media/.heavy.lock`) by importing
its `tools/heavy.py`; if it is unavailable a local lock under this repo's ignored `out/` is used and recorded
as `lock.backend`. Lock paths and private source paths appear only in ignored `out/` manifests, never in
tracked docs. No credentials or arbitrary environment variables are recorded; only the optional
`MLXSTAB_FILM_TOOLS` path override is read.

## Captured artifacts and the real-motion composition audit

`--dump-plan` and `--dump-motion` let a run capture the actual plan and fitted motion for offline geometry
audits without re-running the stabilizer. `bench/plan_shape_audit.py` probes production `source_coords` on
captured plans alone; `bench/real_composition_audit.py` additionally loads the captured motion and compares
production against the Stage 1 composed candidate in three modes (exact composition and both temporal
high-pass charts) on the same four sparse line probes and frame selection. Both audits validate their NPZ
payloads with `allow_pickle=False` and require the sibling manifest to prove the run captured the supplied
artifact(s) (`outputs.plan`/`outputs.motion` basenames and the `experiment_identity.capture` flags). They are
warp-geometry proxies only: no media, no rendering, no production wiring, and no perceived-quality claim. See
[`docs/shape_audit.md`](../docs/shape_audit.md) for the metric, contracts and limitations.

## Captured-motion field audit

`bench/real_motion_field_audit.py` is an **isolated** companion that looks inside the captured fitted motion
instead of composing it. Given one or more benchmark `manifest.json` files it verifies each manifest is a
`status == "success"` run, that `outputs.motion` resolves to an existing captured `motion_r0.npz` whose
payload dimensions/fps match the manifest's recorded `source.decoded`, and that several manifests share a
source identity (source sha256, preprocessing, weights sha256 and decoded size/fps). The **source window is
deliberately not required to match**: t060 and t800 are separate inferred windows of the same source, not
held-out clips, and each was captured at both 1 s and 10 s (the one-second slice nested within its 10 s run),
so the report records each window per case instead of pretending they are one selection. Case names are
deliberately not required to be unique either, so captures that share the case name `baseline` are retained
and distinguished by `run_id`, `manifest_path` and `source_window`. The
motion is loaded through
the same strict non-pickle loader (`np.load(..., allow_pickle=False)`, schema/keys/shapes/dtypes/finiteness)
used by `real_composition_audit`. No media is decoded, no frames are rendered and no production code or default
changes.

For **every** analysed frame pair it reports:

- the best-fit **similarity** of the fitted global `H` (frame k -> k+1) over an inset span grid: a uniform
  `scale` and an in-plane `rotation_deg_y_down`, defined in image pixel coordinates with **y pointing down**
  (positive = visual clockwise content rotation, `atan2(A[1,0], A[0,0])`; the same matrix is counter-clockwise
  in a y-up frame);
- separate RMS/median/p95/max magnitudes of the **row correction** (`Fit.rows`) and **mesh residual**
  (`Fit.mesh`) in destination-frame pixels, with the per-vertex vectors and magnitudes;
- the residual **spatial grid**: one averaged displacement vector and magnitude per mesh cell plus
  valid/fallback metadata (`Fit.fallback`, `Fit.inlier_frac`, `Fit.mover_frac`) and the schema-`/4` per-pair
  gate diagnostics (`n_valid_samples`, `n_ransac_inliers`, `fallback_reason`, `fit_metadata_available`);
- a cautious, explicitly defined **dispersion / neighbour-coherence** proxy of the per-cell residual field
  (`dispersion_rms_px` and a magnitude-weighted neighbour cosine);
- the per-vertex and per-band **support mass** (`mesh_vertex_support_mass`, `row_band_support_mass`) and the
  residual magnitudes split into directly supported (mass >= 3.0) and diffusion-filled (mass < 3.0) groups,
  per pair and pooled per case. Support mass is a non-robust kernel-weighted usable-sample count (observation
  coverage), not a probability and not flow confidence.
- the x-independent **row-band line profile**: an unweighted least-squares line fit of both components of
  `Fit.rows` against the row-band centre y positions `(j + 0.5)*height/nr - 0.5`, reporting `available`
  (at least 2 bands), `n_bands`, `slope_px_per_row` (px displacement per px row), `delta_first_to_last_px`
  (the fitted first-to-last-band change), `nonlinear_residual_rms_px` (pooled residual RMS around the line)
  and `linear_explained_fraction` (pooled multivariate `1 - RSS/TSS`), with per-case descriptive statistics
  and per-pair series. Slope/delta/explained-fraction are `null` when not identifiable (a constant field or
  fewer than two bands), and constant/one-band fields stay finite-safe. The fit deliberately **ignores the
  support masses**, so legacy `/1` and `/2`//`/3` captures share the same definition. They are not combined
  into a score or correlated with the cycle maps.
- the forward-backward **cycle-consistency maps** for schema-`/3` motion: `mesh_cycle_error_rms_px`,
  `mesh_cycle_inconsistency_fraction` and `mesh_cycle_sample_mass` (per-vertex, Gaussian spatial aggregation
  of the per-sample cycle diagnostics), with a per-pair `cycle_consistency` summary and pooled per-case
  statistics. They are built from every finite, in-bounds forward endpoint regardless of the fit's
  valid/non-mover mask, so a sample rejected by the tolerance is still counted. They are reported as `null`
  for manually constructed and legacy `/1`//`/2` fits, which carry no cycle maps.

Honesty boundary: row/mesh residuals are *fit outputs* that mix scene parallax, independent object motion,
**model mismatch**, leftover **rolling-shutter** structure and **optical-flow error**. They are **not depth**.
The module never infers metric depth, object identity or foreground segmentation; true depth needs camera
intrinsics plus known camera motion (structure from motion) or an external depth sensor/network, neither of
which is present. A spatially coherent residual is consistent with parallax *and* with a smooth model mismatch
or rolling-shutter pattern, so the spatial proxy is a descriptive summary, not depth evidence. Support mass
is an **observation-coverage** diagnostic only: it says where the fit was directly observed versus filled, not
how accurate the residual is, and it is not a calibrated uncertainty. The cycle-consistency maps are likewise
a **forward-backward consistency** diagnostic only: a consistent cycle does not prove the correspondence is
correct and an inconsistent cycle does not locate the error, so they are not true correspondence error, not
ground-truth error, not a probability and not calibrated uncertainty. No correlation between the cycle maps and
the residual maps is computed or implied. The row-band line profile is **not** a rolling-shutter detector and
**not** a physical readout estimate: a smooth vertical row displacement is rolling-shutter-*like*, but it can
equally arise from other motion or model mismatch, and no camera readout time, focal calibration or
ground-truth readout is inferred. The `fallback_reason` is likewise a label for the existing `fit_motion`
refusal gates (the first failing gate wins), not a quality judgement. Every report (schema
`mlxstab.real_motion_field_audit/5`) stores this in `scope`, `similarity_definition`, `residual_definition`,
`spatial_proxy_definition`, `support_definition`, `fallback_reason_definition`, `row_line_definition`,
`cycle_consistency_definition` and `limitations`. Legacy `mlxstab.motion/1` payloads
load with `support=None`, so their support fields are null; legacy `/1` and `/2` payloads load with
`cycle_*=None`, so their cycle maps and cycle summaries are null/unavailable; legacy `/1`, `/2` and `/3`
payloads carry no fit-gate metadata, so `n_valid_samples`/`n_ransac_inliers`/`fallback_reason` are null with
`fit_metadata_available=false`; the row-band line profile is
computed from `Fit.rows`, which every schema carries, and stays available. JSON output is deterministic,
`sort_keys` and finite-safe (`json_safe` + `allow_nan=False`). Tests: `tests/test_real_motion_field_audit.py`.

```bash
python -m bench.real_motion_field_audit out/bench/<run>/manifest.json --out out/bench/report.json
# several manifests are accepted only when they share the same source identity (sha/preprocessing/weights/
# decoded size+fps); differing source windows are recorded per case, not refused
python -m bench.real_motion_field_audit run_a/manifest.json run_b/manifest.json --out out/bench/report.json
```

## E04 real-footage flow-initialization benchmark

`bench/flow_prediction_real.py` is an **isolated** E04 experiment (no production wiring, no default change, not
imported by `stabilize`). It replays cold, current same-grid reuse (`RaftFlow.prev_f`/`prev_b`), coordinate-
transported previous flow, constant-velocity and constant-acceleration initializations on the **same real
consecutive frame pairs**, in a forward and a separate reverse-temporal backward traversal. Candidate
construction reuses `bench/flow_prediction.predict`/`transport`; the only changed assumption is that the pair
map is RAFT's own forward/backward flow sampled bilinearly, i.e. an *approximate* inverse with no
occlusion/disocclusion truth (validity is in-bounds and finite only).

Fairness: the window is decoded once, the reference (a **cold high-iteration RAFT run**, default 24
iterations) and every candidate share the same `uint8` frames and the same cached per-frame network features,
and only the initialization differs. Reported per pair: endpoint error (EPE) of the refined flow against the
reference in **feature-grid units** (multiply by `units_to_source_px = 8 * downscale` for source pixels), the
initialization's own EPE, raw/effective coverage, fallback flag and reason, iteration count, and separately
measured initialization (NumPy transport) and refinement (GPU) wall time. The reference is model-derived, so
this measures distance to the converged cold solution, **not** true optical-flow error and not production
speed.

Reference-derived transport maps (offline assumption). Every transport `Pair` map, in both directions, is
built from the **full-window cold high-iteration reference flows** (`--ref-iters`), not from the candidate's
own low-iteration outputs. The candidate never sees or consumes its own map outputs, so the transport map is
exactly the expensive converged flow the benchmark is trying to avoid. This is an offline,
reference-derived-map assumption: it bounds an idealized replay, does not model an online/production pipeline,
and implies no online or production readiness.

```bash
# tiny public synthetic smoke (no private footage; heavy lock off; still runs real RAFT)
python -m bench.flow_prediction_real --source synthetic --synth-frames 5 --synth-w 64 --synth-h 48 \
    --out-dir out/bench/flow_prediction_real/smoke

# local real window (footage stays local; shared heavy lock; cache the exact replayed frames)
python -m bench.flow_prediction_real --source /path/to/local/clip.MTS --start 60 --duration 0.2 \
    --downscale 2 --iters 12 --ref-iters 24 --lock on \
    --cache-frames out/bench/flow_prediction_real/t060_frames.npz --out-dir out/bench/flow_prediction_real/t060
```

Options: `--source`/`--start`/`--duration`/`--vf` (window), `--out-dir`/`--overwrite` (output), `--weights`,
`--iters`/`--ref-iters`, `--downscale`, `--flow-dtype`/`--corr-dtype`, `--methods`/`--directions`,
`--min-coverage`, `--max-pairs`, `--cache-frames`, `--lock {auto,on,off}`, the `--synth-*` smoke controls and
the opt-in `--budget-scan`/`--budget-max-iters`/`--budget-thresholds` oracle scan (which also records the
candidate-observable `flow_update_low` update-magnitude and `photometric_low` luminance-residual diagnostics,
but implements no stopping rule). A separate, mutually exclusive opt-in `--cycle-scan` (with a bounded
`--cycle-max-iters`, default 24) runs the cold-only bidirectional cycle-consistency pilot described below.
`--cache-frames` stores the decoded `uint8` frames; a cache is trusted only when its recorded request
fingerprint (source sha256/path, window, filter, synth parameters) and its own frame digest both match, so a
stale or tampered cache is refused rather than silently mixed. The cache file also stores private-frame
derived data, so keep it in the ignored `out/`.

Each run writes an atomic `manifest.json` with `schema` (`mlxstab.flow_prediction_real/1`), `status`/`stage`/
`error`, source (`sha256`, window, preprocessing/downscale/dtypes, decoded dimensions/fps, `frames_sha256`),
weights sha256, `config`/`config_digest`, `experiment_identity`/`content_key`, the code fingerprint,
environment/hardware, the feature `grid` (padded image / 8, pad, `units_to_source_px`), the `reference` block,
per-direction `common_targets` (the pair set every method can initialize, so aggregates are comparable),
per-method `summary` and `summary_common` plus every `per_pair` row, timing, lock accounting, `scope` and
`limitations`. Frame load/decode/cache restore is reported as `timing_s.frames_load`: it excludes the
shared-lock acquire wait, which `lock.acquire_wait_s` records separately, and is accurate for both freshly
decoded and cache-restored frames. Non-finite values are written as `null` (`allow_nan=False`).

Limitations (also stored in every manifest): the reference is RAFT's own cold output with shared weights, not
ground truth; transport geometry is approximate RAFT flow without occlusion validity; **every transport map
is built from the full-window high-iteration reference flows rather than the candidate's own outputs, an
offline reference-derived-map assumption that does not model online/production use**; fields live on the
feature grid including replicate-padded borders; candidate refinement is single-direction (batch 1) while the
reference is batch 2, so reference timing is reported separately; only initialization/refinement are timed
(frame load, feature extraction and evaluation are reported separately and the frame-load figure excludes the
lock wait); a single inferred/selected window is not clip-wide, held-out, perceptual or multi-scene evidence;
`photometric_low` maps a feature cell to the top-left preprocessed pixel of its 8x8 upsampled block and does
not reconstruct RAFT's learned convex-upsampling sub-cell weighting, so an alignment-dependent residual floor
is possible even at the true flow.

Opt-in per-iteration budget scan (offline oracle). `--budget-scan` (with `--budget-max-iters`, default 24, and
comma-separated `--budget-thresholds`, default `0.005,0.01,0.02` feature-grid px) keeps the same decoded
frames, cached features, cold high-iteration reference, pair ordering, candidate history and shared-lock
chunking, but refines each target pair one iteration at a time from the initialization the normal policy
produced, up to `--budget-max-iters`, recording the mean (and p95/max) low-grid EPE against that pair's cold
reference after every iteration. For each threshold the manifest records the first iteration at or below it,
or that it was **censored** (not reached by the max). **History invariant:** the flow handed to the following
pair is always the state after exactly `--iters` iterations -- the normal fixed-budget replay policy -- so
scan iterations beyond `--iters` never leak into a later initialization; `--budget-max-iters` must be
`>= --iters`. The fixed-budget `summary`/`summary_common` remain the `--iters` equivalents and the scan adds a
separate `budget_scan` block per method (plus `budget_scan`/`config.budget` in the manifest) with summarised
reach rates, crossing iterations and **separately measured** cumulative device-update and host EPE/oracle-check
times. Each per-iteration device result is evaluated before the EPE check, so the two times are not mixed. The
summary keeps the **full-scan totals** (the trace deliberately runs to `--budget-max-iters` to measure the
whole curve) and adds **prefix-summed measured costs per threshold**: for every usable pair the prefix is the
first crossing iteration when reached and otherwise the number of measured scan steps (normally the max, less
only if output went non-finite early), and that prefix's per-step device-update and host EPE/oracle-check times
are summed. Censored pairs therefore still contribute their full measured work, and these prefixes are
computed from the single full trace rather than from separate early-stopped executions; the summary also
reports mean iterations consumed and mean per-pair costs.
This is an **oracle-based, reference-derived and offline** scan: it measures against a cold reference the
pipeline would not have, is **not a deployable stopping rule**, and claims no speedup, online or
production-readiness. Fixed-budget runs with the mode off are unchanged (no `budget` config, no `budget_scan`
block).

Candidate-observable diagnostics (no stop rule selected). Every per-iteration scan row also records
`flow_update_low`: the mean/p95/max per-pixel magnitude `|low - previous_low|` in feature-grid px, where
`previous_low` is the effective initialization on the first iteration and the previous step's flow afterwards.
It is computed **only from the candidate's own initialization/current/previous prediction** -- the cold
reference never enters it -- so it is a *candidate-observable* signal, deliberately kept distinct from the
oracle `epe_low`. Its host computation time is measured separately as `update_mag_check_s` and is never mixed
into `update_s` (device) or `epe_check_s` (oracle EPE); the summary reports prefix-summed `signal_check_s_*`
and full-trace `full_scan_signal_check_s_*` totals, with `total_with_signal_s_*` adding the signal time to the
otherwise unchanged `total_s_*` (which stays `update_s + epe_check_s`). These diagnostics **could** support a
future convergence-based stopping rule, but **no stop rule is implemented or selected** -- the threshold scan
remains an offline oracle comparison to the cold reference, and this is a diagnostic, not a production or
readiness claim.

Every per-iteration scan row also records `photometric_low`, a second candidate-observable signal distinct from
`epe_low` and `flow_update_low`: robust (mean/median/p95/max) absolute luminance-residual statistics after
warping the **source** preprocessed frame toward the **target** preprocessed frame at the candidate's current
low-resolution flow. Luminance is Rec.601 of the same integer box downscale `RaftRunner._prep_pad` applies, so
the diagnostic sees the exact preprocessed content. Feature cell `(fx, fy)` maps to preprocessed pixel
`(8*fx - pad_left, 8*fy - pad_top)` (the `8 * downscale` feature-to-source scaling, with the replicate-pad
offset removed) and the warped target is that location plus `8 * flow`. Samples whose source cell is in the
replicate pad, whose target is out of the preprocessed frame, or whose flow/value is non-finite are excluded
(`valid_frac` reports the retained fraction); a global exposure shift is removed with the per-pair luminance
median difference. Because flat regions carry no photometric signal, `resid_textured` restricts the same
statistics to source cells whose central gradient exceeds `PHOTOMETRIC_TEXTURE_MIN` and `texture_frac` reports
their share, so a low `valid_frac`/`texture_frac` means the signal is only weakly observed. Its host
computation cost is measured separately as `photometric_check_s` and never mixed into `update_s` (device),
`epe_check_s` (oracle EPE) or `update_mag_check_s` (the other candidate-observable signal); the summary adds
prefix-summed `photometric_check_s_*` and full-trace `full_scan_photometric_check_s_*` totals plus
`total_with_observables_s_*`. A single held-out real run now records this signal (see the observable-signal
audit below), but one window is not enough to select a threshold; none is selected and no stopping rule or
speedup is claimed.

Post-hoc observable-signal audit (offline evaluator; no threshold selected).
`bench/flow_prediction_observable_audit.py` reads one or more successful `mlxstab.flow_prediction_real/1`
manifests with an enabled budget scan, validates the required structure (refusing a missing/non-JSON/
wrong-schema/non-success/scan-disabled manifest as a per-case error with exit code 2), and correlates each
candidate-observable signal against the oracle `epe_low.mean` **within each trajectory** (one
method/direction/target pair, varying across the recorded scan iterations). It covers the previously screened
update magnitude (`flow_update_low.mean`) and the new photometric residual
(`photometric_low.resid_textured.median`) with a dependency-free Spearman rank correlation using average ties,
reporting per manifest and signal the valid/usable trajectory counts, the median/mean correlation and the
positive fraction, stratified by method (and direction), plus a descriptive cross-sectional-by-iteration
table. It also accepts a cold bidirectional `cycle_scan` manifest and the
`cycle_low.resid.mean` forward-backward cycle residual (selected explicitly with
`--signal cycle_low.resid.mean`, so the default budget report is unchanged). Pairs whose initialization fell
back with `insufficient_history` are excluded and counted; scan rows
whose signal or EPE is null/non-finite are excluded from pairing and counted. It also reports the median
valid/texture fractions and the separately measured per-iteration signal/photometric check costs, and, when
the audited rows are a cycle scan, the `cycle_low.valid_frac` median/min and the `cycle_check_s` mean/total
(the fields for whichever kind is absent stay `null`, so one coverage shape covers both). The report
is JSON-safe and deterministic, writes atomically with `--out` (default stdout), and records provenance
(manifest sha256, source hash/window, weights, config/config digest, code revision) plus explicit caveats.

This is association only, not accuracy: a high correlation only says the observable and the oracle-EPE curve
move together monotonically, the oracle is the cold high-iteration RAFT reference (model-derived, not ground
truth), and the cross-sectional table reuses dependent pairs/methods/iterations so it is descriptive only.
**No threshold is searched, fitted or selected and no deployable stopping rule is recommended**; nothing here
implies a speedup or readiness.

```bash
python -m bench.flow_prediction_observable_audit MANIFEST.json [MANIFEST.json ...] [--out report.json]
```

Verified on the held-out real manifest
`out/bench/flow_prediction_real/t063_budget_scan_photo_iters12/manifest.json` (`00177.MTS`, 63-64 s window, 59
pairs, 2x downscale, 12/24 iters, both directions; manifest sha256
`9ac3366e939b10a8001e5c8d71413d2cae9fc6c42b3d68483477e7f6e427e0ad`): 590 trajectories (2 directions x 5
methods x 59 targets), 576 usable, 14 excluded as `insufficient_history`. Per-trajectory signal-vs-EPE
Spearman: `flow_update_low.mean` median 0.970 / mean 0.929 / positive fraction 1.000; and
`photometric_low.resid_textured.median` median 0.556 / mean 0.408 / positive fraction 0.792 -- both computed
over the same 576 usable trajectories with no null-signal or null-EPE samples. Stratified by method, the
photometric median ranged from 0.170 (`cold`) to 0.874 (`constant_acceleration`), so its association is much
less consistent than the update magnitude's on this single window. Median `valid_frac` / `texture_frac` were
0.985 / 0.167 (mins 0.970 / 0.143), the per-iteration photometric check cost averaged 1.09 ms (15.09 s over
all 13824 usable scan rows) against 0.139 ms for the update-magnitude check, and the descriptive
cross-sectional correlation fell from 0.92 (update) / 0.81 (photometric) at iteration 1 to 0.19 / 0.02 at
iteration 24. These are one window's association numbers only, with no threshold selected; the JSON artifact
is `out/bench/flow_prediction_observable_audit/t063_photo_iters12.json` (ignored `out/`).

A second held-out window was then audited at
`out/bench/flow_prediction_real/t300_budget_scan_photo_iters12/manifest.json` (status `success`; different
source file `00176.MTS`, inferred 300-301 s window; manifest sha256
`edf8686baccf62374f1d8910d81ad306dd8409d9169cfd3167f882e9d97f35e0`), and a third at
`out/bench/flow_prediction_real/t090_budget_scan_photo_iters12_halfsec/manifest.json` (status `success`; same
`00177.MTS` source as t063 but a non-overlapping 90.0-90.5 s window, 29 pairs; manifest sha256
`730a72221e289c17e43b446fdfd9921a917492b2dcd6fca9ff83eac9fb71c3ca`). Both new runs share t063's
`config_digest` `1070e8e89ab4e542b186880773452e6c752fb81bca0d89f52714892fcf5c6543`, weights (`84c1f730...`)
and code revision, with 2x downscale, 12/24 iters, both directions and zero fallbacks (t300 59 pairs, t090 29
pairs). The combined three-case audit is
`out/bench/flow_prediction_observable_audit/t063_t300_t090_photo_iters12.json` (schema
`mlxstab.flow_prediction_observable_audit/1`). At t300 the per-trajectory Spearman was median 0.234 / mean
0.293 / positive fraction 0.941 for `flow_update_low.mean` and median **-0.080** / mean 0.044 / positive
fraction 0.458 for `photometric_low.resid_textured.median`, with per-method photometric medians from -0.586
(`cold`) to 0.988 (`constant_acceleration`), median `texture_frac` **0.833** (min 0.814) and the photometric
check averaging 1.13 ms. At t090 the update-magnitude Spearman was median 0.634 / positive fraction 1.000 and
the photometric residual median **0.453** / positive fraction 0.826, with per-method photometric medians from
-0.087 (`cold`) to 0.978 (`constant_acceleration`) and median `texture_frac` 0.622 (min 0.611). So t300 has
much higher texture coverage (~0.833 vs ~0.167 at t063 and ~0.622 at t090) yet a **degraded** photometric
association (median ~-0.080 vs ~0.556 and ~0.453): texture coverage alone does not account for the
discrepancy. The three windows are non-overlapping, but t063 and t090 share one source file/hash while t300
uses a different file/hash, and they are only three short windows (two one-second, one half-second) that are
**not** statistically independent; no generalization, threshold, flow accuracy, speedup or readiness claim
follows. **Provisionally, the photometric residual should not be accepted as a standalone convergence proxy**,
pending validation on independent and longer windows.

Tests: `tests/test_flow_prediction_observable_audit.py` (34 weight-free tests) pin the average-tie rank
correlation and its constant/too-few-point reasons, the manifest structure/success/scan refusals, the
insufficient-history and null/non-finite signal skips, method/direction stratification, the cross-sectional
view, the cycle-scan manifest/signal acceptance and coverage, deterministic JSON safety and the stdout/atomic-`--out` CLI.
`tests/test_flow_prediction_real.py` (73 tests) pin deterministic replay/orchestration,
the traversal/direction mapping, the transport direction, identity-map transported == same-grid, fallback and
insufficient-history reporting, the common-pair intersection, the frame-cache request/tamper guards, JSON/
manifest safety (non-finite -> null, atomic writes), and that `timing_s.frames_load` excludes the lock acquire
wait while `lock.acquire_wait_s` keeps tracking it. The budget-scan tests cover threshold parsing/validation,
the `max_iters >= iters` refusal, first-crossing/reach-rate/censor summaries, per-threshold prefix cost
totals/means for both reached and censored pairs (including `total_s = update_s + epe_check_s`), the
fixed-history-at-`--iters` invariant, separated update vs EPE-check timing, non-finite-step stopping,
insufficient-history semantics and
manifest wiring (including that off-mode runs carry no `budget` block). Additional candidate-observable tests
pin the exact expected `flow_update_low` mean/p95/max for a known fake trajectory, prove the signal ignores the
cold reference (changing the reference changes `epe_low` but not `flow_update_low`), and prove its host time is
measured separately from `epe_check_s`/`update_s` (prefix and full-trace `signal_check_s_*` totals, with
`total_s_*` unchanged). They do not assert wall-clock values. The photometric tests pin the Rec.601 box
downscale against a manual computation, the 8x feature-grid-to-preprocessed-pixel scaling with an exactly
recoverable known translation, the replicate-pad coordinate offset, out-of-bounds and non-finite flow/value
exclusion (empty statistics stay `null`), global-brightness-offset removal, the low-texture mask, the
traversal-direction routing of the warper closure, and that the signal ignores the cold reference. Budget-scan
timings keep device update, update-magnitude check, photometric check and oracle EPE check as four separate
measured costs (prefix and full-trace `photometric_check_s_*`, with `total_with_observables_s_*` adding both
signal costs), and the manifest/JSON wiring asserts the `photometric_low` statistics, the per-iteration
`photometric_check_s`, and the `candidate_observable`/`stopping_rule` descriptions. The cycle tests are
weight-free except the paired-stepper parity test, which runs only when the RAFT weights are present: they
pin the analytic cycle residual (inverse-consistent fields -> 0, a known translation disagreement -> 1 px),
explicit border/out-of-bounds/non-finite exclusion, validity-mask handling, that the cycle residual ignores
the cold reference (changing the reference changes oracle EPE but not `cycle_low`), per-pair/same-iteration
paired state, separated device/host-oracle/host-cycle timing, non-finite stopping, the manifest/JSON wiring,
the `--cycle-scan`/`--budget-scan` mutual exclusion and the parity of a full paired scan to `ref_iters` with
`RaftRunner.reference_pair` within the repository's `< 1e-3` feature-grid EPE tolerance. One additional
weight-free test is a **visibility toy check** on the exact `parallax_two_layer` analytic fixture
(48x36, four pairs): it builds the forward/backward fields from `pair.forward(grid) - grid` and
`pair.inverse(grid) - grid`, calls `cycle_residual_fields` with **no ground-truth masks**, and scores the
forward residual on the helper's `fwd_valid` support against `~pair.forward_valid` -- the forward occlusion
AUROC is **1.0**. For the reversed pairs it scores the reversed **forward** residual (`rfields['fwd_resid']`)
on the reversed `fwd_valid` support against `~rev.forward_valid`, whose label is exactly the original
disocclusion (the reversed pair's forward source is original frame k+1) -- the reverse disocclusion AUROC is
also **1.0**. The score and label are aligned on the same source grid; scoring the reversed pair's backward
residual against the reversed forward-valid label would compare different source grids. Both numbers are
exact-analytic fields on **one 48x36 toy with four pairs only**: they are not a real model result, not
real-footage evidence, and imply no production or occlusion-understanding claim.

Opt-in cold-only bidirectional cycle-consistency pilot (diagnostic only; no threshold selected).
`--cycle-scan` is a separate, cycle-only mode (it cannot be combined with `--budget-scan`). For each pair the
forward (`k -> k+1`) and backward (`k+1 -> k`) candidates both start from the **cold zero initialization**
and are advanced **together, one iteration at a time**, through the production batch-2 path (an isolated
`RaftRunner.begin_bidir_scan` that mirrors `RAFT.flow_bidir`/`reference_pair`, without touching `RaftFlow`).
After every iteration the two **same-iteration, same-pair, same-model** fields produce a forward-backward
cycle residual in feature-grid pixels, `|fwd(p) + bwd(p + fwd(p))|` and `|bwd(q) + fwd(q + bwd(q))|`, with an
in-bounds/finite validity filter and robust mean/median/p95/max plus `valid_frac`; the calculation accepts
only the two candidate fields, the grid geometry and optional validity masks and **never** a reference field
or EPE. The cold high-iteration references are still computed and the oracle EPE is recorded next to, but
separately from, the cycle residual; host cycle-check time (`cycle_check_s`) is measured separately from the
device `update_s` and the oracle `epe_check_s`. The pilot is stored under an independent `cycle_scan` block
(the normal `results` block is empty and no production default changes).

This is a **cycle-consistency / visibility** diagnostic only: cycle errors can also mean wrong flow, blur or
independent motion; a consistent cycle does not prove visibility or correspondence correctness; and a low
cycle residual is not true-flow accuracy. On the five short held-out windows below the per-trajectory
association with the oracle EPE was weak and inconsistent (median Spearman 0.017 / -0.590 / 0.137 / -0.450 /
0.538 at t063 / t300 / t090 / t303 / stream00000). Only three distinct source files are involved (t063/t090
share `00177.MTS`, t300/t303 share `00176.MTS`, and the new `STREAM_00000.MTS` is distinct), so the 119 pairs
of the 2.0 s windows are not independent clips; the pooled residual typically fell to a minimum around
iteration 6-15 before rising again. No threshold is selected and no stop rule is implemented; these are five
short, non-independent windows and no convergence improvement is claimed.

Completed cycle-only pilot (ignored `out/`, 2x downscale/float32, 24 paired candidate steps, 24-iteration cold
reference, `--lock on --lock-chunk 30 --lock-pause 12`). All five runs reused the existing verified frame
caches and succeeded, with zero non-finite stops and `coverage_min` 1.0. `cycle_low_resid_mean` is the mean
over all pair/iteration `cycle_low.resid.mean` values; `association` is the per-trajectory
`cycle_low.resid.mean` vs oracle `epe_low.mean` Spearman from
`bench/flow_prediction_observable_audit.py --signal cycle_low.resid.mean`:

| window | pairs | valid_frac | cycle_low mean (median across all per-pair/per-iteration residual means) | assoc median / mean / pos.frac | update_s / cycle_check_s / oracle_epe_check_s | manifest sha256 (prefix) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| t063 (00177.MTS 63-64 s) | 59 | 0.99992 | 0.04330 (0.02841) | 0.017 / -0.050 / 0.542 | 43.615 / 2.769 / 0.290 | `a9fad6a01b40ed84` |
| t300 (00176.MTS 300-301 s) | 59 | 1.00000 | 0.01770 (0.01720) | -0.590 / -0.603 / 0.000 | 44.466 / 2.740 / 0.295 | `cdd1c871f80e3561` |
| t090 (00177.MTS 90.0-90.5 s) | 29 | 1.00000 | 0.01899 (0.01800) | 0.137 / 0.123 / 0.828 | 20.621 / 1.319 / 0.135 | `31dc05c42a076448` |
| t303 (00176.MTS 303-305 s, 2.0 s) | 119 | 1.00000 | 0.01734 (0.01671) | -0.450 / -0.476 / 0.000 | 90.694 / 5.657 / 0.595 | `efc299c0f6a2d80a` |
| stream00000 (STREAM_00000.MTS 25-27 s, 2.0 s) | 119 | 0.98279 | 0.20630 (0.06689) | 0.538 / 0.419 / 0.807 | 90.207 / 5.619 / 0.580 | `edace8cea096f10b` |

All five manifests have `config.cycle = {"scan": true, "max_iters": 24}` and the same `config_digest`
`e873905d5285b3af...`; the combined audit artifact is
`out/bench/flow_prediction_observable_audit/t063_t300_t090_t303_stream00000_cycle_iters24.json` (all five cases
`ok`). At
the last iteration the oracle EPE is exactly 0 by construction (24 paired steps reproduce the 24-iteration
cold reference), so that cross-sectional row is reported as `constant_epe` and skipped rather than treated as
a data point. The t303 window is a non-overlapping 303-305 s 2.0 s window on the same `00176.MTS` source file
as t300; t063 and t090 share `00177.MTS`, while the new stream00000 run uses a distinct `STREAM_00000.MTS`
source. The descriptive per-iteration cycle residual is
non-monotonic (median across each window's pair-level means at each iteration: stream00000 0.50389 at iter 1 ->
minimum 0.05477 at iter 15 -> 0.06038 at iter 24; t303 0.02543 at iter 1 -> minimum 0.01397 at iter 7 ->
0.01932 at iter 24), consistent with
the other windows, so it does not track the oracle EPE monotonically. The per-trajectory association is weak
and inconsistent, so the cycle residual is **not** a standalone convergence cue here; oracle EPE and its check
cost remain clearly separate from the cycle signal. This is diagnostic evidence on five short windows, not a
threshold, a stop rule or a convergence improvement.

## E07 cache-driven relative-depth temporal audit

`bench/depth_temporal_pilot.py` is an **isolated, cache-only** E07 runner (no production wiring, no default
change, not imported by `stabilize`, no depth inference and no media decode). It reads three cached inputs --
a decoded-frame `.npz`, a cached per-frame monocular *relative* depth `.npz` and a local RAFT weights file --
validates the frame/depth hashes, shapes and index correspondence, then runs RAFT only to transport one
sampled depth map onto its neighbour and hands the paired feature-grid depth maps to the pure-NumPy
`bench.depth_temporal_audit.relative_depth_pair_audit`. Because monocular relative depth is defined only up to
a per-frame scale/shift, each pair is compared under an **independent robust per-pair affine fit that discards
metric depth**: the residuals are *structure self-consistency under RAFT transport*, not metric-depth accuracy,
not a depth benchmark and not an occlusion detector. Cycle filtering is model-derived and may select easy
regions; every support filter is diagnostic and **no threshold is selected**. The report is written under
ignored `out/` and is never source-controlled.

```bash
python -m bench.depth_temporal_pilot \
    --frames-npz out/bench/flow_prediction_real/stream00000_t25_frames.npz \
    --depth-npz out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz \
    --weights vendor/weights/conv/raft-things.safetensors \
    --out out/bench/flow_prediction_real/stream00000_t25_depth_temporal_audit.json \
    --iters 24 --downscale 2 --flow-dtype float32 --corr-dtype float32 \
    --cycle-thresholds 0.25,0.5,1,2 --lock on
```

**Optional depth-cache builder.** `bench/depth_cache.py` is an optional companion that turns an
already-decoded, already-verified frame cache into the depth `.npz` contract the pilot consumes (keys
`indices`, `depth`, `model_id`, `normalization`, `frames_npz_sha256`, the last being the content hash of the
input frame cache). It is not required by the pilot and is not wired into any pipeline. Real inference needs
**optional local heavy dependencies** -- a Python environment with `torch`, `transformers` and `PIL` -- plus
an **offline Hugging Face cache** selected through `HF_HOME`; use placeholders (no absolute local path is
recorded). `--local-files-only` **refuses network** and `--lock on` coordinates MPS with the shared
`HeavyLock`. The output NPZ records no path, no source file name and no machine detail.

```bash
HF_HOME=/path/to/local/hf-cache /path/to/python -m bench.depth_cache \
    --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz \
    --out out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz \
    --every 4 --model-id depth-anything/Depth-Anything-V2-Base-hf \
    --device mps --max-side 1036 --local-files-only --lock on

python -m bench.depth_temporal_pilot \
    --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz \
    --depth-npz out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz \
    --weights vendor/weights/conv/raft-things.safetensors \
    --out out/bench/flow_prediction_real/00176_t60_depth_temporal_audit.json \
    --iters 24 --downscale 2 --flow-dtype float32 --corr-dtype float32 \
    --cycle-thresholds 0.25,0.5,1,2 --lock on
```

Reproduced from local caches: **two distinct source files/windows from the same camera/event family**. Window A
is 30 sampled frames (indices 0,4,...,116) from the 120-frame 25-27 s distinct `STREAM_00000.MTS` cache (29
adjacent sampled pairs; night scene). Window B is 30 sampled indices (0,4,...,116) from a 60-62 s decoded window
of `00176.MTS` (29 adjacent sampled pairs; an outdoor ceremony scene with guests walking between the camera and
a deep tree background, described for context only). Both use Depth Anything V2 Base, model id
`depth-anything/Depth-Anything-V2-Base-hf`, each frame independently percentile-normalized [1,99] and clipped
[0,1]; RAFT MLX fp32/fp32, 2x downscale, 24 iterations, 68x120 feature grid. The Window A pilot used the
**original** depth cache `out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz` produced
earlier by the inline inference process, and that cache was the **input to the Window A pilot**. The optional
`bench/depth_cache.py` builder was later run on the **same frame cache** to a separate rebuilt cache and compared
against the original; it was **not** the pilot's input. These are **not two independent
events/clips**, and pairs within each window are serially dependent, so **no pooled significance** is claimed.

Cache-builder parity: re-running the builder on the same original 30-frame `STREAM_00000` frame cache gave
**per-map max abs diff 1.8507242e-5, mean abs diff 1.5569479e-7, p99 abs 9.5367432e-7** against the original
`stream00000_t25_depth_v2base_30frames.npz` depth cache -- same indices and reproducible preprocessing, so the
builder **reproduced the original maps at near machine precision**. This is a builder parity check only; the
rebuilt cache was **not** the input to the Window A pilot. Window B provenance hashes (all under ignored
`out/`, no absolute paths in the report): frame cache
`7838a1265e3dbb020bf93cb5a8101e0c9257df7cf395f96197ed7162f67e4565`, depth cache
`51e311c719415260e9c2c3bdbddf17b2bcfc151b8cc1ddfa28d71659a35f2749`, report
`029c5aaa92274968e47c9f05164b273a6dedfba1676dd0a80da9ac59e3ba6652`.

All-cycle-valid median pair median normalized residual (Window A): **0.0221479 (same-index)** vs **0.0215240
(flow-warp)**; mean pair median 0.0322434 vs 0.0333583; median pair p95 0.115408 vs 0.0965191; median support
0.912868; flow-warp lower on **15/29 pairs**. Here `same-index` compares the source depth against the target
frame's depth at the **same feature-grid pixel coordinates**, while `flow-warp` samples the target depth at the
**forward-flow-mapped coordinates**. `cycle <= 0.25` tier: median 0.0199386 vs 0.0185233, mean
0.0311942 vs 0.0286459, median p95 0.0934196 vs 0.0641095, support 0.804289; flow-warp lower on **17/29
pairs**. Window B all-cycle-valid: same-index median/mean/p95 **0.005629/0.006387/0.064359** vs flow-warp
**0.006932/0.007393/0.070942**, support **0.9727**; `cycle <= 0.25` tier same **0.005608/0.006327/0.060456** vs
flow **0.006842/0.007309/0.067943**, support **0.9593**, flow-warp lower on only **5/29 pairs** in both tiers
(check the JSON before quoting an aggregate). The thresholds are a **measurement sweep only**.

Window B **reverses the mostly mixed/slight-positive night case**: it is **worse for flow-warp in every
aggregate tier**, **direct evidence against a general "flow warp improves temporal depth consistency"
assumption**. With only two windows from the same camera/event family there is **not enough evidence to support
a general claim in either direction and no grounds for pooled significance**. Depth temporal self-consistency
under RAFT transport is therefore **scene-dependent and not a usable generic visibility/confidence cue by
itself**; keep it as a **measurement method only**. It is **insufficient to establish that this visibility/depth
signal has value**: no stabilized-output or downstream-quality comparison, no depth metric truth, ground truth,
semantic or occlusion claim, no production wiring/default, stop rule or speedup, and no claim that the model
understands content or true depth. Runtime is a reproduction observation only (**not a speed claim**, and no
runtime claim is made for depth inference): Window B's RAFT audit took 32.62 s for the 29 pairs; Window A's
canonical report (after the privacy scrub) records 26.670004125 s total, 2.1007635 s feature prep,
23.932933585 s flow across the 29 pairs (mean 0.825273572 s/pair). Next experiments include more distinct
longer scenes from other sources, stronger matched static/deforming/moving cases, downstream stabilization
crop/jitter/warp-validity comparison, and a **scene-conditioned (not pooled)** analysis.

Tests: `tests/test_depth_cache.py` holds **31 cache-builder tests** and `tests/test_depth_temporal_pilot.py`
holds **31 tests** covering the depth audit and the real flow path; the focused depth/cache/flow tests passed
**158 tests**. The full `.claude/verify.sh` passed **672 tests in 24.06 s** (second run; the first run
transiently failed the pre-existing `test_random_init_stages` at EPE 0.00290 against a 0.002 threshold, an
isolated rerun passed at 0.00095, then the full suite passed). CodeGraph is synced/up-to-date and `affected`
found no graph-linked test file for the modules, so the direct tests were run.

## E08 depth-rank motion-layer audit

`bench/depth_motion_audit.py` is an **isolated, cache-only** E08 runner (no production wiring, no default
change, not imported by `stabilize`, no depth inference and no media decode). It asks one narrow question: on a
**held-out checkerboard** split of the feature grid, does stratifying a **model-predicted flow** field by
**per-frame relative-depth quantile bins** make the local displacement field more predictable than a single
pooled 2-D affine model?

Each cell carries a source relative-depth value and a model-predicted forward displacement. The displacement
is modelled as a linear function of the cell's **normalized** `(x/W, y/H, 1)` coordinates with independent OLS
coefficients for `u` and `v` (plain `numpy.linalg.lstsq`, no seeding, no RANSAC). A **pooled** affine model is
fitted on all training cells; a **bin-specific** model is fitted on that bin's training cells only. The split
is deterministic **cell parity of `x + y`** (even cells train, odd cells validate), so train and held-out cells
are interleaved and each bin keeps a held-out subset. The pooled-held-out residual within each bin and the
bin-specific-held-out residual are reported together with their delta (pooled minus bin; a positive delta means
the bin model is lower/better). Source relative depth is divided into **4 equal-count relative-depth rank
bins**; **no bin is labelled true foreground/background or near/far**, and there is **no true flow / ground
truth**. Residuals are Euclidean in feature-grid pixels.

The runner samples **30 frames -> 29 adjacent pairs in each existing 2-second window** and audits **both
directions**, giving **58 pair-direction rows per window**; per-window summaries pool pairs **within a single
window only** (the two windows are distinct files from the same camera/event family and are **not** pooled
together, and pairs within a window are serially dependent).

```bash
python -m bench.depth_motion_audit \
    --frames-npz out/bench/flow_prediction_real/stream00000_t25_frames.npz \
    --depth-npz out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz \
    --weights vendor/weights/conv/raft-things.safetensors \
    --out out/bench/flow_prediction_real/stream00000_t25_depth_motion_audit.json \
    --iters 24 --downscale 2 --bins 4 --cycle-thresholds 0.25,0.5,1,2 --lock on

python -m bench.depth_motion_audit \
    --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz \
    --depth-npz out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz \
    --weights vendor/weights/conv/raft-things.safetensors \
    --out out/bench/flow_prediction_real/00176_t60_depth_motion_audit.json \
    --iters 24 --downscale 2 --bins 4 --cycle-thresholds 0.25,0.5,1,2 --lock on
```

**Window A** `STREAM_00000.MTS` 25-27 s (report
`out/bench/flow_prediction_real/stream00000_t25_depth_motion_audit.json`, ignored): all-cycle-valid median
support **0.9327**; across usable pair-directions the median of the pooled-model held-out per-bin median
residual is **0.23513** vs **0.12132** for the bin-specific model, with a median per-row delta of **0.11937**;
median pooled/bin p95 **0.53985 / 0.29863**; **215/232 bins** have lower residual with the bin model; usable
**58/58 pair-directions**. `cycle <= 0.25` tier: support **0.8095**; median pooled/bin **0.21644 / 0.10537**;
p95 **0.45624 / 0.23598**; **207/224 bins** improve.

**Window B** `00176.MTS` 60-62 s (report `out/bench/flow_prediction_real/00176_t60_depth_motion_audit.json`,
ignored): all-cycle-valid median support **0.9853**; median pooled/bin per-bin held-out median
**0.04665 / 0.03197**, median per-row delta **0.00898**; median pooled/bin p95 **0.21806 / 0.20867**;
**157/232 bins** improve; usable **58/58 pair-directions**. `cycle <= 0.25` tier: support **0.9700**; pooled/bin
median **0.04275 / 0.02963**, p95 **0.19361 / 0.18303**; **157/232 bins** improve. Reports carry **no absolute
paths**; report sha256: A `3a3b5fd9b1f7e25cd79291ad923fead090058aa9a2d54fa7f90d6ec0ecb63d5b`, B
`7f42b6c8f05a718bc8aca742481e62532d54715eb5b63e9e17636fed9d5c72d`.

Interpretation: rank stratification makes RAFT's own local displacement field **more affine-predictable** in
these two windows, with a much larger effect at night and a smaller absolute improvement outdoors. This is
evidence of **depth-rank association with model-predicted RAFT flow structure only**; it does **not** validate
depth estimates or predict stabilization benefit. It is **not proven**, **not a general claim**, and does not
establish a usable depth-confidence cue.

Explicit limitations: RAFT flow is **model-derived**; there is **no metric depth, no optical-flow accuracy, no
camera-motion ground truth and no stabilization outcome**. Quantile edges use **all eligible depth values
including held-out cells** (unsupervised distribution leakage), and checkerboard train/held-out neighbours are
**correlated**, so the held-out residual is a structural diagnostic, not a generalisation guarantee. The
per-window summaries pool pairs within a single window only; the two windows are **distinct files from the same
camera/event family and serially dependent**, and must **not** be pooled. No threshold is selected and no
default changes.

Combined with E07 (temporal self-consistency was **scene-dependent**), flow-field stratification is positive on
**both** windows; these are **different signals** and must **not** be conflated. Next decision gate: an isolated
candidate that uses depth-rank motion layers to choose **background-lock vs foreground-preserve** paths on
**synthetic known-truth** scenes, then a matched downstream **crop/jitter/warp-validity** ablation on these two
windows. Do **not** wire production or choose a depth threshold until it improves the user-visible tradeoff;
no generic depth-confidence claim yet.

Tests: `bench/depth_motion_audit.py` is covered by `tests/test_depth_motion_audit.py`; the direct new tests
passed **35 tests**. The full `.claude/verify.sh` passed **707 tests in 23.91 s**. CodeGraph was synced and
`affected` found no graph-linked tests, so the direct tests were run. No production code changed.

## E09 depth-rank layer synthetic known-truth gate

`bench/depth_layer_synthetic.py` is an **isolated, synthetic, pure-NumPy** known-truth runner (no production
wiring, no default change, not imported by `stabilize`, no media decode, no RAFT/MLX import, no weights, heavy
lock untouched). It implements the E08 next-decision-gate question: on a **deterministic two-layer affine
coordinate-model fixture**, can selecting cells by an **explicit depth rank** recover a requested coherent
motion layer, and what do separate **background-lock** and **foreground-preserve** policies cost downstream?

Scene. A full-frame **background** plane (layer 0) and a rectangular **foreground** overlay (layer 1) drawn on
top so it occludes the background. Per frame `t` each layer `l` has a **known** 2x3 affine `M[l][t]`; both
layers share a camera transform (intended smooth pan/zoom composed with deterministic per-frame jitter), and
the overlay adds an **independent deforming local motion** composed with the camera, so the two layers genuinely
differ. The observed forward flow is the visible layer's displacement. Everything is floated in float64 with
fixed seeds/permutations; the report is deterministic and **path-free** (atomic write via `bench.manifest`).

Depth semantics. Raw depth values are **arbitrary labels**. `RANK_ORIENTATIONS` makes the raw-value -> rank
mapping **explicit and caller-chosen**: `low_raw_low_rank` (smallest raw value is rank 0) or
`low_raw_high_rank` (largest raw value is rank 0). The module **never** infers physical near/far and never
claims semantic foreground/background. The policy rank is **chosen via the spec** (`lock_rank` / `preserve_rank`
under the configured orientation), **not by semantic inference**; a rank carrying no layer is `unavailable` with
a machine-readable reason and is never silently clamped.

Method. A deterministic **checkerboard split** (even `x + y` train, odd held out) is used. A **pooled** baseline
fits one affine displacement field on all valid train cells; each **policy** fits the same affine family on only
its selected rank's valid train cells, fitted **from source depth, flow and validity alone** (the true
per-pixel layer id is never used for fitting), and scored against the **known layer flow** of its target layer
on the identical held-out block. **Known truth is used only to score.** Both policies are **separate blocks**
with their own selected rank, fitted model and statistics; nothing is pooled between them, no threshold is
selected or swept, and **no winner is declared**. Fixed support floors (`MIN_FIT_SAMPLES = 6`,
`MIN_EVAL_SAMPLES = 3`) make an accidental single-cell win impossible.

Downstream model. Stabilization by an estimated per-frame path `E[t]` composes `E[t]^-1` with the known layer
transform, giving **downstream RMS second-difference residual jitter** and **in-bounds inverse-warp coverage**
-- **coordinate-model metrics, not rendered-pixel or perceptual metrics** (no frame is rendered, no production
stabilizer is run). The pooled baseline targets neither layer: to compare it downstream on identical footing,
`baseline.downstream_by_target_layer` is the **one common pooled path counterfactually evaluated** with target
layer **0** and target layer **1** as the intended target (`baseline.targets_layer` is `None`); if any pooled fit
fails, the block is unavailable with a machine-readable reason and no metrics.

Degenerate controls. `depth_mode` can make depth **uninformative** (both layers collapse to one rank; each
policy declines with reason `rank_ambiguous_uninformative_depth`) or **shuffled** (a deterministic permutation
that destroys the rank/layer association; both policies are declared **unavailable** with reason
`rank_layer_association_destroyed`, a `None` downstream block and **zero** material/comparable wins). The pooled
baseline is retained as a diagnostic in every regime but supports **no policy claim**. A **false-win issue was
caught in the shuffled control** -- a rank-selected fit could score an accidental frame-wise "win" -- and was
**fixed by suppressing policy claims** in that regime rather than reporting the spurious improvement.

```bash
python -m bench.depth_layer_synthetic --out out/bench/depth_layer_synthetic.json
python -m bench.depth_layer_synthetic --depth-mode shuffled --out out/bench/depth_layer_synthetic_shuffled.json
```

Default one-fixture example metrics (T=10, W=48, H=36, depths `0.2,0.8`, `low_raw_low_rank`, `ranked`, lock
rank 0 / preserve rank 1, seed 0). Both policies available.
Held-out **target median error** is the exact-recovery **0.0 px** for the rank policy against the pooled
baseline's **0.420816 px** (background_lock target layer 0) and **1.630723 px** (foreground_preserve target
layer 1). Rank-policy **locked-layer residual jitter** is **0.0 px** against the pooled **counterfactual**
**0.441816 px** (target 0) and **1.226369 px** (target 1). Policy **crop coverage** is **0.816667** (target 0)
and **0.838484** (target 1) against the pooled **0.821238** for each target. Pooled fits succeeded **10/10**.

Interpretation (cautious): the exact-recovery numbers are **expected from a noiseless affine coordinate-model
fixture** and validate that the plumbing (rank resolution, held-out scoring, the two independent policy blocks,
the counterfactual pooled downstream, and the degenerate refusals) **works as specified** -- they do **not**
demonstrate real-world benefit, and there is **no generic claim** and **no winner**. The results are
**synthetic**; raw depth values are **arbitrary labels** with explicit rank orientation; checkerboard
train/held-out neighbours are **spatially correlated**, so held-out error is a structural diagnostic, not a
generalisation guarantee; downstream jitter and coverage are coordinate-model metrics, **not perceptual
quality**. No production policy is wired or selected and no default changes. Nothing here is proven or general.

Next decision gate (E10): before any production policy choice, **add noise, non-affine deformation and depth
errors** to the fixture (and/or matched **real footage**) and re-measure the **downstream crop/jitter/warp-validity**
tradeoff. Do **not** wire production or choose a depth threshold until it improves the user-visible tradeoff.

Tests: `bench/depth_layer_synthetic.py` is covered by `tests/test_depth_layer_synthetic.py`; the focused
depth/layer tests passed **59 tests**. The full `.claude/verify.sh` passed **766 tests in 23.69 s**. CodeGraph is
synced/up-to-date and `affected` found **no graph-linked tests** for the new module, so the direct tests were
run. No production code changed.

## E10 robustness controls on the depth-layer synthetic gate

E10 adds **three opt-in, default-zero robustness controls** to the **same** isolated synthetic runner
`bench/depth_layer_synthetic.py` (identical scene, checkerboard split, separate policy blocks and counterfactual
pooled downstream; still no production wiring, no default change, not imported by `stabilize`, no media decode,
no RAFT/MLX import, no weights, heavy lock untouched, path-free atomic JSON report). It answers the E09 next
decision gate by re-measuring the same **frame-median held-out target flow error**, **target-layer residual
jitter** and **inverse-warp coverage** tradeoff under corruption. Nothing is swept and **no winner is declared**.

Zero-default controls (defaults preserve E09 exactly; the default spec is bit-for-bit the E09 fixture):

- **`flow_noise_px`** adds zero-mean i.i.d. Gaussian **measurement noise** (sigma, pixels) under a fixed
  named-offset seeded RNG to the **observed** flow only; the known `layer_flow` **ground truth is never
  perturbed**, so noisy observations and exact scoring stay separable. Default `0.0`.
- **`depth_error_frac`** (in `[0, 1]`) applies **deterministic flips** of that fraction of the **visible ranked**
  binary depth labels between the two raw classes (fixed seeded permutation), and **only to `depth_mode="ranked"`**
  (uninformative/shuffled left untouched). The **realized count/fraction is reported**. It changes the observed
  per-pixel rank the fit consumes but **never** the ground-truth layer labels, transforms or flow. Default `0.0`.
- **`fg_nonaffine_px`** adds a **deterministic smooth, spatially varying** (sinusoidal, frame-varying)
  foreground displacement folded into **both** the known `layer_flow` and the observed flow; the affine
  `layer_transform` is **kept separately** as the affine component. Default `0.0`.

All three default to zero, so the E09 fixture is unchanged, and the **shuffled** and **uninformative**
degenerate refusals (`rank_layer_association_destroyed`, `rank_ambiguous_uninformative_depth`) remain in force.

Measured policy-vs-pooled values. One **deterministic fixture/seed** (T=10, W=48, H=36, depths `0.2,0.8`,
`low_raw_low_rank`, `ranked`, lock rank 0 / preserve rank 1, seed 0). Each cell is
**frame-median held-out target flow error px / target-layer residual jitter px / inverse-warp coverage**.
**Synthetic only, no threshold search, no winner.**

| corruption (fixed fixture, seed 0) | layer | policy | pooled counterfactual |
|---|---|---|---|
| ideal (all defaults) | background | 0.000000 / 0.000000 / 0.816667 | 0.420816 / 0.441816 / 0.821238 |
| ideal (all defaults) | foreground | 0.000000 / 0.000000 / 0.838484 | 1.630723 / 1.226369 / 0.821238 |
| flow noise sigma 0.5 px | background | 0.045862 / 0.189726 / 0.817361 | 0.397158 / 0.473539 / 0.821528 |
| flow noise sigma 0.5 px | foreground | 0.080578 / 0.212981 / 0.840162 | 1.650267 / 1.238590 / 0.821528 |
| 25% depth rank flips | background | 0.151815 / 0.155288 / 0.819039 | 0.435668 / 0.441816 / 0.821238 |
| 25% depth rank flips | foreground | 1.471806 / 0.808365 / 0.826273 | 1.792963 / 1.226369 / 0.821238 |
| fg non-affine amp 2 px | background | 0.000000 / 0.000000 / 0.816667 | 0.477180 / 0.444151 / 0.821875 |
| fg non-affine amp 2 px | foreground | 0.645553 / 0.353022 / 0.797454 | 2.119175 / 1.370548 / 0.821875 |
| combined sigma 0.5 / flip 0.25 / non-affine 2 | background | 0.168082 / 0.223581 / 0.819907 | 0.484729 / 0.476453 / 0.822049 |
| combined sigma 0.5 / flip 0.25 / non-affine 2 | foreground | 2.075177 / 0.986376 / 0.825752 | 2.351953 / 1.381617 / 0.822049 |

For the 25% depth-error row the fixture realized **432/1728 flipped pixels** (0.25).

Interpretation (cautious): the **different corruption parameters must be interpreted separately** -- a higher
policy error under one knob does not transfer to another, and **no aggregate score or winner** is claimed. In
this construction **depth-label errors hurt the foreground much more than the background**, **non-affine
foreground motion is model misspecification** (the fitted affine family cannot represent it), and **measurement
noise perturbs the flow estimates** on both layers while truth stays exact. These are **synthetic
coordinate-model corruptions, not calibrated sensor/depth/motion models**, and they **do not establish a robust
real-video benefit**; the all-zero default still validates **plumbing, not benefit**.

Explicit limitations: unchanged from E09 (synthetic, arbitrary depth labels, explicit rank orientation and no
physical near/far or semantic foreground/background, spatially correlated checkerboard neighbours, coordinate
model rather than rendered-pixel/perceptual metrics), plus the three controls are **synthetic corruptions of the
coordinate model**, not calibrated models of real sensor noise, depth errors or non-rigid motion. No threshold is
selected or swept, no winner declared, and no production wiring or default change. Nothing is proven or general,
and no robust real-footage benefit is established.

```bash
python -m bench.depth_layer_synthetic --out out/bench/depth_layer_synthetic.json                                              # ideal defaults (E09)
python -m bench.depth_layer_synthetic --flow-noise-px 0.5 --out out/bench/depth_layer_synthetic_noise.json                    # flow measurement noise
python -m bench.depth_layer_synthetic --depth-error-frac 0.25 --out out/bench/depth_layer_synthetic_flip.json                  # 25% depth rank flips (ranked only)
python -m bench.depth_layer_synthetic --fg-nonaffine-px 2.0 --out out/bench/depth_layer_synthetic_nonaffine.json              # non-affine foreground motion
python -m bench.depth_layer_synthetic --flow-noise-px 0.5 --depth-error-frac 0.25 --fg-nonaffine-px 2.0 --out out/bench/depth_layer_synthetic_combined.json  # combined
python -m bench.depth_layer_synthetic --depth-mode shuffled --out out/bench/depth_layer_synthetic_shuffled.json               # shuffled refusal (still)
python -m bench.depth_layer_synthetic --depth-mode uninformative --out out/bench/depth_layer_synthetic_uninformative.json     # uninformative refusal (still)
```

Next decision gate (E11): E10 is one fixture and one seed with synthetic corruptions. Next **run multiple seeds
and scene geometries**, then check **textured/rendered validity** and a **matched real-footage downstream
crop/jitter/warp-validity** comparison. Do **not** wire production or choose a depth threshold until then.

Tests: `bench/depth_layer_synthetic.py` (with the three controls) is covered by
`tests/test_depth_layer_synthetic.py`; the focused depth/layer tests passed **78 tests**. The full
`.claude/verify.sh` passed **785 tests in 28.11 s**. CodeGraph was synced and `affected` found **no graph-linked
tests** for the module, so the direct tests were run. No production code changed.

## E11 multi-seed / foreground-geometry synthetic suite

E11 adds the isolated suite `bench/depth_layer_synthetic_suite.py`, a thin deterministic harness around
`bench.depth_layer_synthetic.run_experiment`. It adds **no new model, metric, threshold or winner**; it only runs
the existing E09/E10 experiment across a **fixed, predeclared grid** and reports per-cell numbers. Still no
media decode, no RAFT/MLX/production import, no weights, heavy lock untouched, path-free atomic JSON report.

Fixed grid (frozen constants, not swept):

- **Seeds**: `0, 1, 7, 42` (a small deterministic convenience sample, not a random sample).
- **Foreground geometries**, normalized fractions `(cx, cy, half_w, half_h)`: `default_center`
  `(0.56, 0.52, 0.20, 0.20)` (the E09/E10 default rectangle), `offcenter_vertical` `(0.32, 0.48, 0.12, 0.35)`,
  and `upper_wide` `(0.64, 0.35, 0.30, 0.14)`.
- **Conditions**, run separately and never mixed: `ideal` (all E10 controls zero) and `stress` with the fixed
  `flow_noise_px=0.5, depth_error_frac=0.10, fg_nonaffine_px=1.0`.

That is `4 seeds x 3 geometries x 2 conditions = 24 cases`; each `(seed, geometry, condition)` is run exactly
once and its individual report preserved verbatim under `cases`. Summaries are keyed by
`geometry|condition|policy` and are **kept separate by geometry, condition and policy** -- **no pooling across
geometries**. The values are **descriptive only, not inferential**: no significance claim, no winner, no
threshold, no production wiring (and no import by `stabilize`).

The suite runs the base module, whose foreground rectangle is an explicit **`fg_rect`** field
(`SyntheticSpec(fg_rect=...)`, normalized `(cx, cy, half_w, half_h)`); the **default `(0.56, 0.52, 0.20,
0.20)` preserves the E09/E10 geometry**, so `default_center` is bit-for-bit the existing fixture. On the base
CLI it is exposed as **`--fg-rect`** (`cx,cy,half_w,half_h`, default `0.56,0.52,0.20,0.20`).

Summary cells (all 12). Each entry is **error median [min,max] px; target-layer jitter median [min,max] px;
coverage median [min,max]; material-win seeds / frame wins**. Values were **independently verified** from the
suite report. **Synthetic only, descriptive only, no winner, no threshold.**

| geometry \| condition \| policy | error median [min,max] px | target-layer jitter median [min,max] px | coverage median [min,max] | material-win seeds / frame wins |
|---|---|---|---|---|
| default_center \| ideal \| background_lock | 3.55e-15 [3.55e-15, 3.55e-15] | 1.433e-14 [1.159e-14, 1.735e-14] | .844763 [.816667, .872975] | 4/4 seeds, 36/40 frames |
| default_center \| ideal \| foreground_preserve | 3.55e-15 [3.55e-15, 3.55e-15] | 9.292e-15 [7.959e-15, 1.323e-14] | .856916 [.837326, .887211] | 4/4 seeds, 36/40 frames |
| default_center \| stress \| background_lock | .074652 [.059796, .087068] | .180607 [.164005, .188279] | .845081 [.816898, .872512] | 4/4 seeds, 40/40 frames |
| default_center \| stress \| foreground_preserve | 1.097184 [1.036723, 1.153596] | .553389 [.517897, .584807] | .850203 [.824421, .876215] | 4/4 seeds, 40/40 frames |
| offcenter_vertical \| ideal \| background_lock | 3.948e-15 [3.553e-15, 4.651e-15] | 1.459e-14 [1.142e-14, 1.802e-14] | .844763 [.816667, .872975] | 4/4 seeds, 36/40 frames |
| offcenter_vertical \| ideal \| foreground_preserve | 3.553e-15 [3.553e-15, 3.553e-15] | 1.132e-14 [8.367e-15, 1.318e-14] | .856916 [.837326, .887211] | 4/4 seeds, 36/40 frames |
| offcenter_vertical \| stress \| background_lock | .085059 [.081109, .085198] | .184236 [.172043, .188164] | .845139 [.819965, .875000] | 4/4 seeds, 39/40 frames |
| offcenter_vertical \| stress \| foreground_preserve | .793693 [.779753, .851519] | .440549 [.405352, .462743] | .839959 [.813831, .875463] | 4/4 seeds, 36/40 frames |
| upper_wide \| ideal \| background_lock | 3.109e-15 [2.221e-15, 3.553e-15] | 1.251e-14 [1.010e-14, 1.556e-14] | .844763 [.816667, .872975] | 4/4 seeds, 36/40 frames |
| upper_wide \| ideal \| foreground_preserve | 3.397e-15 [1.776e-15, 5.329e-15] | 1.009e-14 [9.597e-15, 1.223e-14] | .856916 [.837326, .887211] | 4/4 seeds, 36/40 frames |
| upper_wide \| stress \| background_lock | .091512 [.074313, .110384] | .222435 [.189997, .259119] | .845197 [.820370, .873380] | 4/4 seeds, 38/40 frames |
| upper_wide \| stress \| foreground_preserve | .854979 [.854523, .894360] | .460494 [.398733, .476876] | .859722 [.841319, .891435] | 4/4 seeds, 37/40 frames |

Interpretation (cautious): under `ideal` the errors are at floating-point level (near machine zero), which is
**expected from the exact affine ground truth** -- it validates plumbing, not benefit. `stress` **raises errors
and jitter relative to `ideal`** in every cell, consistent with E10. The **stress magnitudes differ across
geometries**, but with **only four fixed seeds that is not evidence of any ordering** between geometries, and
no geometry, condition or policy is declared better. All counts (material-win seeds/frames, coverage) are
**descriptive only**. There is **no textured render yet**: the next gate is a **textured/rendered
warp-validity** check plus a **matched real-footage downstream crop/jitter/warp-validity** comparison. **No
production policy until then.**

Limitations: unchanged from E09/E10 (synthetic coordinate model with known ground truth, arbitrary depth
labels, explicit rank orientation, no physical near/far or semantic foreground/background, coordinate-model
rather than rendered-pixel or perceptual metrics) plus the E11 scope caveat: the seed list and geometries are a
**small predeclared deterministic convenience sample, not a random sample**, so medians and min/max describe
those fixed realisations only; the results are **exploratory and not inferential**. No threshold is selected or
swept, no winner is declared, and no production wiring, import or default change is implied. Nothing here is
proven or general, and no robust real-footage benefit is established (no textured render exists yet).

```bash
python -m bench.depth_layer_synthetic_suite --out out/bench/depth_layer_synthetic_suite_e11.json   # E11 fixed grid (24 cases)
python -m bench.depth_layer_synthetic --fg-rect 0.32,0.48,0.12,0.35 --out out/bench/depth_layer_synthetic_offcenter.json   # base: explicit fg rect
```

Tests: `bench/depth_layer_synthetic_suite.py` composes `bench/depth_layer_synthetic.py`, covered by
`tests/test_depth_layer_synthetic.py`; the focused suite + base tests passed **105 tests**. The full
`.claude/verify.sh` passed **812 tests in 26.12 s**. CodeGraph was synced and `affected` found **no graph-linked
tests** for the suite module, so the direct suite tests were run. No production code changed.

## E12 rendered-validity audit

E12 closes the "no textured render yet" gap left by E09/E10/E11: the coordinate-model diagnostics report a
*geometric* coverage fraction, never a statement about which *rendered* pixels survive the warp.
`bench/depth_layer_render_audit.py` is an isolated **pure-NumPy** module that adds rendered-pixel bookkeeping.
It adds **no new model, threshold or winner**, is **not imported by** `stabilize`/`source_coords`/any
production module, changes no default, and touches no renderer. No media is decoded, no RAFT/MLX/production
code is imported, no weights are read, the heavy lock is untouched, and the report is **path-free** and
finite-safe (atomic write).

Rendering model (explicit, deterministic):

- **Textures.** `build_textures` builds a `[2, H, W, 3]` float64 RGB stack from fixed sinusoids of the
  reference coordinates (background) plus a distinct frequency set and a domain-separating offset (foreground).
  **No RNG, no assets, no media, no weights**; values are finite and clipped to `[0, 1]`.
- **Forward splat per frame.** For frame `t` and layer `l` the reference pixel `p` maps to `q = M[l][t](p) +
  d_l(p)`, where `M[l][t] = seq.layer_transform[l, t]` is the known affine and `d_l = seq.layer_nonaffine[l, t]`
  is the known sampled non-affine displacement (exact zeros for the background and for zero non-affine
  amplitude). `q` is rounded to the nearest integer pixel (`rasterization = nearest_forward_splat`);
  destinations outside `[0, W-1] x [0, H-1]` are dropped (the source of splat holes).
- **Compositing and collision policy.** The scene composites layer 0 first, then layer 1 on top, so the explicit
  collision policy is **later layer wins**; within one layer the **last reference pixel in deterministic
  row-major order wins** (`collision_policy = later_layer_wins_then_row_major_last`). A per-pixel
  render-validity mask marks destinations that received at least one splat; an unwritten destination stays at
  zero RGB with `valid=False` (a hole). **Holes and collisions are synthetic** — artifacts of the forward splat
  and collision rule, not of any stabilizer.
- **Pull-warp.** `pull_warp` maps every output pixel `o` back to `s = E[t]^-1(o)` and **bilinearly samples** the
  rendered frame there with a **conservative validity mask**: a pixel is valid only when all four contributing
  source-mask taps are valid **and** the mapped sample is strictly inside the pixel-edge box `[0, W-1] x
  [0, H-1]` (all four bilinear taps in bounds). The warp uses each policy's estimated `E[t]` from
  `bench.depth_layer_synthetic._fit_policy`.

Policies and the pooled baseline. `background_lock` selects `spec.lock_rank`; `foreground_preserve` selects
`spec.preserve_rank`; each has its own estimated `E[t]` per frame and therefore its own rendered-validity
numbers, and **nothing is pooled between them**. The **pooled baseline** fits one affine on all valid train
cells and is a single common diagnostic path. When a policy's rank is unavailable -- under **uninformative**
depth the rank collapses, under **shuffled** depth the rank/layer association is destroyed, and a rank outside
the scene is absent -- the policy is reported `available=False` with a machine-readable reason and **no render
is fabricated** (its `render` block is `None`); the pooled baseline remains a diagnostic in those regimes.

Report. Per frame and in aggregate it records **in-bounds coverage**, **rendered-valid fraction**, and **finite
RGB fraction / variance / counts**. The report contains **no pixel buffers** (no image pixels in JSON).

Independently reproduced CLI results for `T=10, W=48, H=36`, default spec and the stress spec
`flow_noise_px=0.5, depth_error_frac=0.2, fg_nonaffine_px=2.0`. Metrics are **mean rendered-valid coverage /
mean in-bounds coverage / mean finite-RGB fraction**:

| spec | policy | rendered-valid / in-bounds / finite-RGB |
|---|---|---|
| default | pooled | .595081 / .813252 / 1.0 |
| default | background_lock | .594213 / .810069 / 1.0 |
| default | foreground_preserve | .603183 / .822859 / 1.0 |
| stress | pooled | .543461 / .801794 / 1.0 |
| stress | background_lock | .545775 / .804630 / 1.0 |
| stress | foreground_preserve | .549479 / .805324 / 1.0 |

Cautious note. The **lower rendered-valid coverage relative to geometric in-bounds coverage** reflects forward-
splat **holes plus the conservative four-tap mask**; that difference is a **fixture / rasterization property**,
**not a measured real-stabilizer crop or an advantage** of any policy. These are **one deterministic synthetic
configuration**, not a policy winner. This is **not perceptual evidence**, and the forward-splat hole rate is an
artifact of the synthetic splat, not a property of stabilization.

```bash
python -m bench.depth_layer_render_audit --T 10 --W 48 --H 36 --out out/bench/depth_layer_render_audit_e12_default.json
python -m bench.depth_layer_render_audit --T 10 --W 48 --H 36 --flow-noise-px 0.5 --depth-error-frac 0.2 --fg-nonaffine-px 2.0 --out out/bench/depth_layer_render_audit_e12_stress.json
```

Tests: focused `tests/test_depth_layer_render_audit.py` passed **28 tests**; the full `.claude/verify.sh` passed
**840 tests in 25.74 s**. CodeGraph was synced and `affected bench/depth_layer_render_audit.py` found **no
graph-linked tests**, so the direct focused tests were run. No production code changed and no production default
changed.

### E13 matched real-footage downstream geometry

`bench.depth_layer_real_runner` validates one frame/depth cache pair, runs bidirectional RAFT, and supplies the
forward flow plus cycle-map-valid support to the pure NumPy `bench.depth_layer_real_audit` core. The core
compares a pooled affine fit with each of four per-frame relative-depth rank fits, composes each into a path,
applies a component-wise Gaussian path smoother (`sigma=1 pair`), and reports jitter and inverse-warp geometric /
conservative bilinear coverage. Rank bins are neutral numeric strata; they are not foreground/background or
metric depth. Cycle-map-valid means finite and in bounds, not a selected residual threshold.

Two independent existing cache windows were measured using RAFT-things MLX, 24 iterations, downscale 2. Results
are not pooled across windows:

| window | policy | jitter px | in-bounds | rendered-valid |
|---|---|---:|---:|---:|
| stream00000 | pooled | 1.70667 | 0.98089 | 0.96578 |
| stream00000 | rank 1 | 1.66567 | 0.97942 | 0.96441 |
| stream00000 | rank 3 | 1.99646 | 0.97663 | 0.96194 |
| 00176 | pooled | 0.03575 | 1.00000 | 0.97698 |
| 00176 | rank 1 | 0.03647 | 1.00000 | 0.97643 |
| 00176 | rank 3 | 0.03853 | 1.00000 | 0.97740 |

Other rank bins also move the trade-off; no rank policy consistently improves jitter and coverage. This is a
model-derived affine path proxy, not ground-truth camera motion or rendered stabilization quality. Results are
path-free and ignored under `out/bench/flow_prediction_real/`; no clips or dense flow maps are committed.

The optional `--sweep-bins 1,2,4,8 --sweep-sigmas 0,0.5,1,2` runs 16 variants from the same RAFT observations
for each window. The zero-sigma control is exact identity correction (0 jitter, 1.0 coverage); a regression
test guards against floating-point edge loss there. Across sigma values, the first window's pooled jitter /
rendered-valid coverage moves from `0 / 1.0` at sigma 0 to `2.36869 / 0.94545` at sigma 2. In the second it
moves from `0 / 1.0` to `0.06299 / 0.97779`. Across rank policies at 2, 4 and 8 bins, jitter ranges are
`0.53474–3.40642` for stream00000 and `0.01078–0.10561` for 00176 (over the full sigma sweep). These ranges
show strong scene and parameter sensitivity; they do not identify a winner. Full per-window values are in
`STATUS.md`.

```bash
python -m bench.depth_layer_real_runner --frames-npz out/bench/flow_prediction_real/stream00000_t25_frames.npz --depth-npz out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz --weights vendor/weights/conv/raft-things.safetensors --out out/bench/flow_prediction_real/stream00000_t25_depth_layer_real.json --iters 24 --downscale 2 --bins 4 --smooth-sigma 1
python -m bench.depth_layer_real_runner --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz --depth-npz out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz --weights vendor/weights/conv/raft-things.safetensors --out out/bench/flow_prediction_real/00176_t60_depth_layer_real.json --iters 24 --downscale 2 --bins 4 --smooth-sigma 1
```

Re-run the same windows with the sensitivity grid:

```bash
python -m bench.depth_layer_real_runner --frames-npz out/bench/flow_prediction_real/stream00000_t25_frames.npz --depth-npz out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz --weights vendor/weights/conv/raft-things.safetensors --out out/bench/flow_prediction_real/stream00000_t25_depth_layer_real_sweep.json --iters 24 --downscale 2 --sweep-bins 1,2,4,8 --sweep-sigmas 0,0.5,1,2
python -m bench.depth_layer_real_runner --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz --depth-npz out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz --weights vendor/weights/conv/raft-things.safetensors --out out/bench/flow_prediction_real/00176_t60_depth_layer_real_sweep.json --iters 24 --downscale 2 --sweep-bins 1,2,4,8 --sweep-sigmas 0,0.5,1,2
```

Focused core/runner tests passed **35 tests**; the synthetic, multi-seed and rendered depth-layer suites plus
depth-motion audit passed another **168 tests**. CodeGraph found no graph-linked tests, so these tests were run
directly. No production code or default changed. The next gate is more independent windows plus a
human-reviewed rendered-quality comparison before any production wiring.

## Tests

`tests/test_bench_runner.py` covers configuration/experiment identity, change labelling against the fully
resolved config, state isolation, manifest hashing/invalidation, atomic and JSON-safe writes, failure and
cancellation reporting, overwrite protection, run-dir uniqueness, cadence/timeline and mask-tail mismatch,
support flags, rendered-vs-encoded counts, diagnostics sampling policy and the CLI failure paths, all with
cheap model/media fakes. `tests/test_media_cadence.py` adds real (CPU-only) ffmpeg regressions and a fake-flow
end-to-end 60-render/30-encode case that needs no RAFT.
