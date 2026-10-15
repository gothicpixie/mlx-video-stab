# mlxstab status

## Stage 0 progress

### Slice 1: evaluator reset and mask coordinates

Implemented the evaluator-reset and mask-coordinate fixes from the plan, with regression tests.

- `RaftFlow.reset()` clears the legacy `prev` cache, the bidirectional warm flows (`prev_f`, `prev_b`), the cached frame features (`_last`) and the `syncs` diagnostic. `bench/evaluate.py` now calls it at each clip boundary instead of `flow.prev.clear()`, so one clip cannot warm-start or reuse features from another. Feature reuse and warm starts are unchanged inside a sequence.
- `motion.exclude_masked` samples the source mask at `obs.pts` and the target mask at `obs.pts + obs.flow`. Dilation semantics are preserved; fractional positions are sampled conservatively (any non-zero bilinear weight on an invalid pixel rejects the correspondence), exact pixel centres are not rejected by rounding, out-of-image and non-finite positions are rejected, and `valid` is only ever reduced.
- `analyze` no longer skips exclusion when the target mask is absent but a source mask is available.
- `tests/test_evaluation_correctness.py`: 25 tests covering source-vs-destination mask positions, horizontal/vertical/diagonal fractional target sampling plus an exact-integer zero-weight guard, independent optional masks, dilation, bounds/non-finite inputs, invalidity preservation, the `analyze` source-only-mask caller, a fake-RAFT reset lifecycle, and a `bench/evaluate.py` integration test that runs its real input/output loop with fake media and asserts each clip starts cold. The initial 15-test version produced 9 failures against the pre-fix code (checked then); the expanded suite was not re-run against the old code.

### Slice 2: complete-warp diagnostics and fp16 correlation coverage

- `mlxstab/warpmetrics.py` measures the complete source-coordinate field (global homography + residual mesh + row correction). Each output cell is a bilinear patch and is checked at all **four corner Jacobians** (forward differences), so an opposite-corner fold, a concave cell and a non-finite corner are detected rather than missed by a single top-left Jacobian. It reports anisotropy/stretch/shear, signed determinant (corner min/mean, foldover and collapse fractions), a cell-count partition (`nonfinite_cells` / `overflow_cells` / `usable_cells`), and `center_valid` versus full-kernel-support coverage for bilinear, bicubic and Lanczos3 with a documented `|weight| <= 1e-6` zero-tap rule. Singular values use a scale-normalised 2x2 SVD (`s2 = |det|/s1`) so high-shear matrices do not suffer cancellation, and finite-but-overflowing coordinates are counted as unusable instead of scoring as perfect. Non-finite/degenerate inputs return explicit `None` values and every result, including cross-frame aggregation, is JSON-safe. Semantics and conventions are in `docs/warp_diagnostics.md`.
- Wired into the report path: `Config(warp_diag=True)`, CLI `--warp-diag`, and `render_frame(..., diagnose=True)`, which reuses the source field already computed for rendering (no second warp pass). The historical global-only `distortion_min`/`distortion_mean` are unchanged and now labelled `distortion_scope = "global_homography_only"`; when the option is off, `warp_diag` is `null`.
- Tests: `tests/test_warpmetrics.py` (15 analytic/integration cases: identity, similarity, anisotropic scale versus shear, reflection, half-collapse, an alternating fold that centred differences would cancel, the reviewed opposite-corner fold and concave-cell counterexamples, a missing finite corner, stable singular values against `numpy.linalg.svd` including the `[[1,1e4],[0,1]]` high-shear matrix, overflow accounting and JSON safety, the `scale=1e154` representable-determinant mean, scaled `_stats` reductions for large equal and opposite-sign values, non-finite/degenerate behaviour, translated borders and filter support, and bent row/mesh through the real `source_coords`/`render_frame` path); `tests/test_stabilize.py` adds a monkeypatched `stabilize` report integration plus a `_distortion` invalid-mapping guard test.
- fp16 precision: `tests/test_raft.py::test_real_weights_bidir_fp16_correlation` runs the bidirectional path with fp32-network/fp16-correlation and fp16-network/fp16-correlation against the PyTorch reference, and spies on the `CorrBlock` instances `flow_bidir` actually creates (forward and backward, every pyramid level) so the keyword cannot be silently ignored. Measured on the local raft-things demo frames (both directions): fp32 0.00222 px, fp16 net/fp32 corr 0.00269 px, fp32 net/fp16 corr 0.00224 px, fp16 net + fp16 corr 0.00260 px (existing gates: fp32 < 0.01 px, fp16 < 0.05 px).
- Measured overhead at 1920x1080 (one frame, minimum of 3 runs): render-only ~17 ms, render + diagnostics ~755 ms (coverage ~432 ms, four-corner shape ~245 ms). Complete-warp diagnostics are therefore opt-in.
- Review repairs on the same branch: four-corner Jacobians; stable 2x2 SVD; explicit corner overflow accounting; scaled mean/quantile/cross-cell reductions so representable finite values (e.g. four corner determinants of 1e308, or opposite-sign extrema) never become `inf`; `_distortion` input guards; the `CorrBlock` spy over every pyramid level; and documentation corrected so `det_collapse_frac` is a corner-collapse fraction (mixed-sign cells are caught by `det_negative_frac`) and anisotropy drops collapsed corners rather than whole cells.

Check: `.claude/verify.sh` -> 58 passed (offline; the earlier 40 still pass).

### Slice 3: reproducible benchmark runner

`bench/run_case.py` runs explicit cases with a full experiment identity: source and weight hashes,
source window and preprocessing, resolved configuration, output/evaluation rates, diagnostic selection,
seed policy, code revision and working-tree code hashes. The baseline uses half-resolution flow,
20 iterations, fidelity 0.1 and 30000/1001 output/evaluation fps. This historical preset differs from
`Config()` defaults (full-resolution flow, 24 iterations). Cases isolate fp16 network, fp16 correlation,
warm-8 and stride-2; combined-fp16 explicitly changes both precision settings.

- Diagnostics sample the actual full-window render field, with exact frame indices, values and separately
  measured cost. They require no second RAFT pass. All-frame and sampled diagnostic scopes are mutually
  exclusive. Sampled maxima do not bound unsampled frames.
- Fixed output-rate conversion: 60 input frames at 60 fps previously became 32 frames at 30 fps.
  The encoder now uses an explicit fps filter, producing 30. Tests also cover NTSC rates and mask alignment.
  Render writes and measured encoded frame counts are reported separately.
- Evaluation uses common-rate normalized frame-index time. Every encoded video/mask PTS is separately
  checked against its zero-origin CFR clock, including interior timestamps. This permits native-rate
  output with lower-rate evaluation and rejects extra mask tails, duplicate/shifted timestamps and missing
  timing. Original source PTS and VFR timing are not verified by this CFR benchmark.
- Atomic manifests record running, success, failure and cancellation states. Initial/final persistence
  failures surface; directories are protected against accidental overwrite. Missing/non-finite metrics,
  absent support or identity-fit fallbacks make comparisons unreliable.
- Repeats reset temporal state, use distinct output paths and distinguish initial model loading from cached
  runs. Timing includes model construction and encoder close. Lock waits and configurable chunk pauses
  have separate counters; stage wall times include pauses. Render time minus measured diagnostic intervals
  is not a counterfactual throughput estimate because encoding overlaps.

Usage, schema, rate overrides, window-selection notes and lock options are in `bench/README.md`.
Verification: `.claude/verify.sh` -> **99 passed** (offline); focused benchmark/media checks and the native-rate smoke also passed.

Evidence under ignored `out/` (no private media is tracked):

- `out/bench/final_t060_1s/`: one-second camcorder smoke at the inferred 60-second window, 60 deinterlaced
  fields -> 30 output/mask frames at 30000/1001, 1.001 s. Input/output evaluation had 29 supported pairs
  each and no identity fallbacks. Diagnostics sampled actual render indices [0,15,30,45]. The retained
  original report contains a stale encoded-count field; inspection records distinguish it from measured
  counts. The 60/800-second window choices come from historical filenames, not a recovered formal manifest.
- `out/bench/native_cadence_audit/`: public synthetic run with 60 fps encoding and 30 fps evaluation;
  60 encoded video/mask frames, 30 evaluated frames, matching complete encoded timelines and normalized
  evaluation clocks. This specifically checks that the two rates are not conflated.
- Earlier runner drafts remain marked unaccepted. Smoke runs establish execution and reporting behavior;
  ten-second footage baselines and approximation-quality comparisons remain outstanding.

- `out/bench/baseline_t060_10s_1c60047/`: completed the 10 s baseline on local camcorder file `00176.MTS`,
  using the 60–70 s window inferred from historical output names (no formal selection manifest was recovered).
  The runner deinterlaced 600 fields into 300 output frames at 30000/1001 fps; video and mask cadence matched,
  the evaluation was reliable, and no identity-fit fallbacks occurred. Input → stabilized jitter was
  1.587 → 0.519 px and wobble was 0.0762 → 0.0536 px; stability was 0.843 → 0.926. Stabilization took
  1096 s and evaluation brought total wall time to 2997 s. This is one inferred window, not a clip-wide or
  held-out quality claim. Sampled complete-warp diagnostics ran on frames
  [0,60,120,180,240,300,360,420,480,540]: mean/min center-valid support 97.77%/94.64% and mean/min Lanczos3
  full-support 97.37%/94.13%; no sampled negative determinants or non-finite warp cells were found, and these
  samples do not bound unsampled frames.

- `out/bench/baseline_t800_10s_14f6f60/`: completed the 10 s baseline on the same local camcorder file
  `00176.MTS`, using the 800–810 s window inferred from historical mlxstab_t800 output filenames (no formal
  selection manifest recovered). Status `success`, cadence aligned: 600 deinterlaced fields -> 300 output and
  mask frames at 30000/1001 fps, reliable with zero identity-fit fallbacks. Input → stabilized jitter was
  1.036 → 0.0632 px and wobble 0.0654 → 0.0343 px; the Liu stability composite moved 0.784 → 0.722 (it
  decreased, and the score is scale-free and punishes near-static paths, so it must not be read alone).
  Stabilization took 1026 s and total wall time 1653 s. Sampled complete-warp diagnostics ran on frames
  [0,60,120,180,240,300,360,420,480,540]: mean/min center-valid support 98.657%/96.880% and mean/min Lanczos3
  full-support 98.227%/96.503%; no sampled negative determinants or non-finite warp cells were found, and these
  samples do not bound unsampled frames. One inferred window, not a full-clip or held-out quality claim.

Matched one-second residual ablations (all eight manifests `success` and reliable, 29 support frames, zero fit
fallbacks; identical input metrics within each window, matched source hash/window, the same relevant code
hashes, and zero sampled negative determinant / non-finite-cell fractions). Each residual-disabled case
re-estimates the remaining model, so this is a practical configuration ablation, not fixed-plan causal
isolation: `full` = mesh+rows, `global-only` = both residuals off, `mesh-only` = rows off, `row-only` = mesh
off.

| window | configuration | output jitter px | output wobble px | Lanczos3 full-support mean |
| --- | --- | --- | --- | --- |
| t060 | full | 0.04653 | 0.03227 | 0.99176 |
| t060 | global-only | 0.06413 | 0.09079 | 0.99204 |
| t060 | mesh-only | 0.04603 | 0.05492 | 0.99182 |
| t060 | row-only | 0.08036 | 0.04096 | 0.99185 |
| t800 | full | 0.04580 | 0.02944 | 0.98868 |
| t800 | global-only | 0.05186 | 0.06211 | 0.98854 |
| t800 | mesh-only | 0.04314 | 0.04162 | 0.98864 |
| t800 | row-only | 0.06196 | 0.03029 | 0.98859 |

At t060 the full model had the lowest jitter and the lowest wobble; at t800 mesh-only had the lowest jitter
while the full model had the lowest wobble. These results are not collapsed into one score and no winner is
claimed. Both windows are one second and inferred from historical filenames; diagnostics are sampled at
[0,15,30,45] only; this is not clip-wide, held-out, perceptual, or un-deinterlaced evidence.

Matched two-second residual ablations (all eight manifests `success` and reliable, 59 support frames, zero fit
fallbacks; same source/window/input metrics and code hash within each window, and zero sampled
negative-determinant/non-finite-cell fractions). Diagnostics were sampled at [0,30,60,90] only. Each
residual-disabled case re-estimates the remaining model fields, so this is a practical configuration
ablation, not fixed-plan causal isolation.

| window | configuration | output jitter px | output wobble px | Lanczos3 full-support mean | stabilization s |
| --- | --- | --- | --- | --- | --- |
| t060 | full | 0.04843 | 0.03678 | 0.98254 | 179.7 |
| t060 | global-only | 0.06655 | 0.08438 | 0.98286 | 177.1 |
| t060 | mesh-only | 0.05069 | 0.05503 | 0.98272 | 184.6 |
| t060 | row-only | 0.07250 | 0.04012 | 0.98236 | 182.6 |
| t800 | full | 0.04781 | 0.03543 | 0.98872 | 183.8 |
| t800 | global-only | 0.05118 | 0.07295 | 0.98873 | 177.3 |
| t800 | mesh-only | 0.04834 | 0.04909 | 0.98867 | 203.7 |
| t800 | row-only | 0.06957 | 0.03442 | 0.98841 | 175.6 |

`full` has the lowest jitter on both windows; at t060 it also has the lowest wobble, while at t800 `row-only`
is slightly lower on wobble. `global-only` has much higher wobble in both windows. The metrics are kept
separate; no winner is claimed and stability is not used as a composite decision. These are still two-second
windows inferred from filenames, not held-out, full-clip or perceptual proof; each ablation re-estimates the
remaining model fields.

Across these four paired window/duration comparisons (t060 and t800, each 1 s and 2 s), relative to `full`:
`global-only` raises output jitter by ~+7% to +38% and wobble by ~+106% to +181%; `mesh-only` changes jitter by
~-6% to +5% and raises wobble by ~+39% to +70%; `row-only` raises jitter by ~+35% to +73% with wobble from
~-3% to +27%. These windows start at the same inferred t060/t800 offsets and are nested (1 s inside 2 s), not
independent scene samples, so the ranges are not confidence intervals or held-out results. The consistent
`global-only` wobble penalty is a useful hypothesis about which residual channels matter, but rigid-structure/
visual review and 10 s or longer evaluation are still required before any decision.

Matched one-second configuration screens (five config manifests per window plus the per-window `full`
baseline; all succeeded and were reliable with 29 support frames and zero fit fallbacks, matched to their
per-window baseline on source hash/window, input metrics and relevant code hashes, and with zero sampled
negative-determinant/non-finite-cell fractions). `Liu stability` is secondary only and is not used to declare
a quality winner; total wall includes evaluation.

| window | case | output jitter px | output wobble px | Liu stability (secondary only) | stabilization wall s | total wall s |
| --- | --- | --- | --- | --- | --- | --- |
| t060 | full | 0.04653 | 0.03227 | 0.92714 | 81.4 | 119.8 |
| t060 | fp16-network | 0.04669 | 0.03156 | 0.92199 | 77.6 | 116.8 |
| t060 | fp16-corr | 0.04505 | 0.03162 | 0.92187 | 85.6 | 125.4 |
| t060 | combined-fp16 | 0.04287 | 0.03204 | 0.92428 | 76.5 | 117.4 |
| t060 | warm-8 | 0.04658 | 0.03331 | 0.90231 | 61.2 | 105.6 |
| t060 | stride-2 | 0.02977 | 0.03474 | 0.93274 | 60.9 | 99.0 |
| t800 | full | 0.04580 | 0.02944 | 0.91586 | 83.4 | 122.3 |
| t800 | fp16-network | 0.04699 | 0.03039 | 0.90766 | 73.5 | 111.5 |
| t800 | fp16-corr | 0.04564 | 0.02947 | 0.89861 | 85.7 | 125.6 |
| t800 | combined-fp16 | 0.04590 | 0.03064 | 0.90170 | 76.7 | 116.3 |
| t800 | warm-8 | 0.04615 | 0.02925 | 0.89254 | 64.2 | 102.4 |
| t800 | stride-2 | 0.03436 | 0.02898 | 0.92515 | 57.6 | 97.8 |

Short, preliminary observation (not a recommendation): `fp16-network`/`combined-fp16` and `warm-8`/`stride-2`
are faster in stabilization on both windows than `full`, while `fp16-corr` is slower; `warm-8`/`stride-2`
reduce stabilization time by about 21–31%. Timing differences are preliminary, these are one-second
inferred-window screens, and they require longer matched clips and quality review. They do not justify
enabling any default.

Remaining Stage 0 work: the matched one- and two-second residual ablations and the one-second configuration
screens are complete; 10-second/full-window versions are still needed. Also outstanding are the real-footage
reproduction of the reported tree deformation and floaty motion (the CPU-only E02 shape audit is analytic
only), the motion-field rotation/parallax audit on real windows, production integration, and clip-wide/held-out
quality, so Stage 0 is not complete. Fixture parity (fp32/fp16) does not establish full-clip quality of the
approximations.

## Stage 1 progress

### Composition candidate (experimental, not wired in)

`bench/composition_candidate.py` is a NumPy-only Stage 1 experiment. It is **not** imported by `stabilize`,
`source_coords` or any other production module and changes no defaults or existing outputs. It rebuilds the
output-to-source field by composing the per-pair global homographies with the measured residuals,
`F_k(p) = H_k(p) + rows_k(p_y) + mesh_k(p)`, evaluated at their source positions, instead of adding the
residuals on the fixed output grid. The smoothed global path is preserved exactly as `Plan.G`
(`Psi_t(V_t(x)) = Plan.G[t](x)`), and `highpass=False` returns the exact composed oracle.

Two temporal-smoothing charts are implemented and **neither is selected**: `smoothing_basis="source"`
(default, unchanged behavior) smooths the frame-`t` source displacement `delta_t = Phi_t - Psi_t`;
`"reference"` smooths `eta_t = inv(Psi_t)(Phi_t(z)) - z` in the reference chart and pushes the high-pass back
with the full projective `Psi_t(z + eta_hp) - Psi_t(z)` (no linear/Jacobian block, so projective `H_k` is
supported). The budget is in pixels of the smoothed chart, so the two modes are not directly comparable; the
open decision is recorded in `SMOOTHING_FRAME_NOTE`. Support/fold reporting is finite-difference and strict:
`candidate_field(strict=True)` rejects folded or collapsed cells, non-finite cells, and finite cells whose
corner determinant overflowed; those categories are excluded from the finite summaries, which stay JSON-safe.

Verification: the candidate's 24 tests and the audit's 15 tests passed together under the shared heavy lock
(39 total). They check analytic fixtures, closed-form rotation transport, the location error, parallax
localisation, mesh/row order, exact projective composition and the source-vs-reference chart difference
against independently derived oracles. These are analytic results only: they do **not** establish
real-footage quality, tuning, coverage behaviour or perceptual benefit, and no production integration is
proposed. Both real-footage 10 s baselines (t060 and t800) are now complete; their evidence is recorded under
Stage 0 above, each being one inferred window rather than a clip-wide or held-out quality claim.

### Rigid-line shape audit (E02, CPU-only analytic)

`bench/shape_audit.py` is a CPU-only analytic mechanism check for E02 ("what causes deformation?"). It
compares the current fixed-grid residual map (`composition_audit.naive_from_plan`) with the Stage 1 composed
candidate (`composition_candidate.candidate_source_coords(..., highpass=False)`) using a best-fit-line (SVD/PCA)
deviation metric and a frame-to-frame `dev_max_px` time series. It is **not** footage or perceptual proof,
**does not wire the candidate into production**, and changes no defaults; no media, FFmpeg or MLX/Metal kernels
are used. Conventions, controls and tolerances are in `docs/shape_audit.md`.

Measured on the two-step projective fixture (`n=401`, frames 0-2, frame values are `dev_max_px`):

- ground truth by matrix product `P1 @ P0`: max line deviation ~1.14e-14 px.
- candidate (composed): max 0.001814 px, max temporal step 0.001770 px, max pointwise error to truth
  0.005059 px.
- fixed-grid sum (`naive_from_plan`): max 1.028842 px, max temporal step 1.028798 px, max pointwise error to
  truth 8.023751 px.

`tests/test_shape_audit.py` (18 tests) passed; only that module was run, not the full suite. The fixture is one
orientation, one parameterisation and a 48x36 mesh, so the candidate deviation is mesh-limited. The reported
real-footage tree deformation is still unmeasured by this audit; it is a controlled mechanism check only.

### Captured-plan geometry path (implemented; real-footage audit pending)

`stabilize(..., dump_plan=...)` and benchmark `--dump-plan` now save the retained plan as compressed,
non-pickle `plan_r0.npz` with its global maps, mesh/row corrections, dimensions, rate, schema and fallback
count. The benchmark manifest records the artifact path and whether capture was enabled in the experiment
identity. `bench/plan_shape_audit.py` validates this payload and evaluates four sparse straight-line probes
through the production `source_coords` implementation, reporting mapped-line deviation and source-bound
coverage at selected exact frame indices. Its `--compare` mode compares full/global-only/mesh-only/row-only plans
only when sibling manifests prove the same source hash, time window, preprocessing, weights and all resolved
configuration fields except the intentionally varied `mesh` and `rs_rows`. It reports curvature and coverage
separately, with no winner score. This remains a warp-geometry proxy, not tree detection or perceptual proof.

Verification: plan-dump integration tests and the plan-audit loader, malformed-payload, homography, mesh/row,
coverage, sparse/full-map parity, frame-selection, matched-plan/config refusals and CLI tests pass. A 61-frame
identity-plan CLI smoke selected frames `[0,30,60]`, with final-frame inclusion, near-zero mapped deviation
and full probe coverage.

Matched one-second captures now exist at the inferred source windows 60–61 s and 800–801 s. For each window,
baseline, global-only, mesh-only and row-only all completed with reliable benchmark metrics, and
`--compare` verified equal source hash/window/preprocessing/weights, dimensions, frame count, fps and resolved
configuration apart from `mesh`/`rs_rows`. The audit sampled frames `[0,15,30,45,59]` (59.94 fps) and 201
points on each of four probe lines. The table reports the maximum mapped probe deviation and minimum probe
coverage separately from the benchmark's output jitter and wobble:

| window | case | max mapped probe deviation px | min probe coverage | output jitter px | output wobble px |
| --- | --- | ---: | ---: | ---: | ---: |
| t060 | baseline | 9.709 | 0.9851 | 0.04653 | 0.03227 |
| t060 | global-only | 0.00018 | 0.9801 | 0.06413 | 0.09079 |
| t060 | mesh-only | 8.107 | 0.9851 | 0.04603 | 0.05492 |
| t060 | row-only | 4.859 | 0.9801 | 0.08036 | 0.04096 |
| t800 | baseline | 3.288 | 0.9900 | 0.04580 | 0.02944 |
| t800 | global-only | 0.00017 | 0.9900 | 0.05186 | 0.06211 |
| t800 | mesh-only | 2.913 | 0.9900 | 0.04314 | 0.04162 |
| t800 | row-only | 1.295 | 0.9900 | 0.06196 | 0.03029 |

On these two inferred one-second windows, the residual fields introduce measurable warp-geometry curvature;
global-only is near-straight but has worse output jitter and wobble than the full plan. Mesh-only preserves
similar or slightly lower jitter while increasing wobble; row-only has higher jitter. This supports testing
shape-preserving residual composition/regularization, not disabling residuals as a general fix. The probe
lines are not detected tree trunks, and these results do not establish perceptual quality, clip-wide behavior,
or a default change. The 10-second baselines predate plan capture; next, evaluate the composed candidate on
real motion fits, then test longer and held-out windows with visual review.

### Real plan+motion composition audit (implemented; two short real-footage audits complete)

`stabilize(..., dump_motion_npz=...)` and benchmark `--dump-motion` now save the retained repeat-0 fitted
motion as a compressed, non-pickle `motion_r0.npz` (schema tag `mlxstab.motion/4`, frame dimensions,
`n_frames`, fps, stride, fit count, stacked per-pair `H`/`mesh`/`rows`/`inlier_frac`/`mover_frac`/`fallback`,
the per-fit **fit-gate diagnostics** `n_valid_samples`/`n_ransac_inliers`/`fallback_reason_code` plus the
`fit_meta_available` flag, the observation-coverage support masses `mesh_support_mass`/`row_support_mass`, and
the forward-backward
cycle-consistency maps `cycle_err_rms`/`cycle_inconsistency_frac`/`cycle_sample_mass` plus the per-fit
`cycle_map_available` flag; zero-fit shapes come from `Config.mesh`/`rs_rows`, and a Fit built without support
mass, cycle maps or gate metadata dumps shape-correct zeros / `-1, -1, -1` sentinels, the latter two tagged
unavailable). Support mass is the
non-robust kernel-weighted count of usable non-mover flow samples at each mesh vertex / row band, recorded
before `_fill` diffusion: `>= 3.0` marks directly observed locations and `< 3.0` marks diffusion-filled ones.
It is an **observation-coverage diagnostic only**, not a probability, not flow confidence and not a calibrated
uncertainty. The cycle maps are a forward-backward **cycle-consistency** diagnostic only, not a true
correspondence error and not calibrated uncertainty (see the dedicated subsection below). The loader still
accepts legacy `mlxstab.motion/3` (support + cycle, no fit metadata), `mlxstab.motion/2` (support, no cycle
maps) and `mlxstab.motion/1` payloads, loading the
missing fields as `None`, and reads the pre-standardization `/4` reason-code key `fallback_reason` as an alias
for `fallback_reason_code` (the canonical key) so the one preserved just-captured `/4` payload still loads. The
benchmark
manifest records `outputs.motion` and `experiment_identity.capture.dump_motion`; plan and motion capture are
independent, and `--dump-plan` alone keeps the plan-only comparison contract unchanged.

`bench/real_composition_audit.py` loads a captured plan **and** its captured motion from the same run,
verifies the sibling manifest proves both artifacts were captured (`outputs.plan`/`outputs.motion` basenames
and `capture.dump_plan`/`capture.dump_motion`), reconstructs `Plan`, `Motion`, `Fit` and `Config`, and enforces
the composition scope (`stride == 1`, `len(plan.G) == len(motion.fits) + 1 == motion.n_frames`, matching
dimensions/fps, and `Config.mesh`/`rs_rows` consistent with the plan/motion array shapes). It then compares
production `source_coords` against the composed candidate in three modes -- exact composition
(`highpass=False`) and both temporal high-pass charts (source, reference, each smoothed once per clip) -- on
the same four sparse line probes and frame selection as `bench.plan_shape_audit`. It reports per-frame
curvature, source-bounds coverage and pointwise displacement from production, and deliberately no composite
winner score. This remains a warp-geometry proxy on real fitted motions: not tree detection, scene
understanding, a perceptual evaluation, or evidence of perceived tree straightness, and no production
integration (no default or output change).

Verification: `stabilize`/runner motion-dump integration, the motion loader's schema/shape/finiteness/no-pickle
validation, manifest capture/basename refusals, scope and stride/length refusals, analytic projective and
parallax fixtures, sparse sampling, both high-pass charts, final-frame selection and the CLI are covered by
the focused tests. The full suite passes (322 tests). Matched baseline captures were added at the same inferred
t060 and t800 one-second windows as the plan-shape audit. Both captures passed the manifest/provenance checks;
each audit sampled frames `[0, 30, 59]` on the same four 401-point probes. Aggregate maximum probe deviation,
minimum bounds coverage and maximum displacement from production were:

| Window | Map | Max deviation (px) | Min coverage | Max displacement from production (px) |
|---|---|---:|---:|---:|
| t060 | production | 9.757 | 0.9850 | 0 |
| t060 | exact composition | 19.298 | 0.9875 | 13.567 |
| t060 | source high-pass | 10.260 | 0.9850 | 1.498 |
| t060 | reference high-pass | 10.244 | 0.9850 | 1.497 |
| t800 | production | 3.159 | 0.9975 | 0 |
| t800 | exact composition | 4.496 | 0.9975 | 3.460 |
| t800 | source high-pass | 3.043 | 0.9975 | 0.758 |
| t800 | reference high-pass | 3.044 | 0.9975 | 0.754 |

In these sparse probes, exact composition is less straight than production in both windows. The high-pass
charts remain close to production: they slightly reduce maximum deviation at t800 but increase it at t060,
with under 1.5 px maximum displacement in either window. Coverage is effectively unchanged. This is not
evidence of a useful improvement: the candidate is not uniformly better, these are only three frames from
inferred one-second windows, and straight-line probes do not evaluate visual content, temporal perception,
full-frame behavior or output-video quality. The plan-shape ablation table above and these candidate results
remain separate diagnostics; no chart is selected and no default or output changes.

Next composition-candidate gate: repeat on longer and held-out windows with visual review and temporal
stability measures. No production integration is proposed until a candidate demonstrates a consistent,
perceptually validated improvement.

### Captured-motion field audit (implemented; four motion-field audits complete: t060/t800, 1 s and 10 s)

`bench/real_motion_field_audit.py` is an **isolated** companion to the composition audit that inspects the
captured fitted motion itself rather than composing it. It consumes one or more benchmark `manifest.json`
files, requires each to be a `status == "success"` run whose `outputs.motion` resolves to an existing
`motion_r0.npz` whose payload dimensions/fps match the manifest's recorded `source.decoded`, and loads the
motion through the same strict non-pickle loader (`np.load(..., allow_pickle=False)`). Several manifests must
share a source identity (source sha256, preprocessing, weights sha256 and decoded size/fps); the **source
window is deliberately not required to match**, because t060 and t800 are separate inferred windows of the
same source, not held-out clips, and each was captured at both 1 s and 10 s (the one-second slice nested
within its 10 s run), so the report records each window per case rather than pretending they are one
selection. For every
analysed frame pair it reports the best-fit image-y-down similarity of the fitted global `H` (uniform `scale`
and in-plane `rotation_deg_y_down`, positive = visual clockwise content rotation), separate RMS/median/p95/max
magnitudes of the row and mesh residual vectors in destination-frame pixels, per-vertex and per-cell mesh
residual vectors/magnitudes with valid/fallback metadata, a cautious dispersion/neighbour-coherence proxy, and
the per-vertex/per-band support mass with the residual magnitudes split into directly supported (`mass >= 3.0`)
and diffusion-filled (`mass < 3.0`) groups. JSON is deterministic, `sort_keys` and finite-safe.

Scope: row/mesh residuals are fit outputs mixing scene parallax, independent object motion, **model
mismatch**, leftover **rolling-shutter** structure and **optical-flow error**. They are **not depth**; the
audit never infers metric depth, object identity or foreground segmentation, and states that true depth needs
camera intrinsics plus camera motion or external depth. The dispersion/coherence proxy is descriptive only: a
coherent residual is consistent with parallax *and* with smooth model mismatch, so it is not depth evidence.
Support mass is likewise an **observation-coverage** diagnostic only: the non-robust kernel-weighted count of
usable non-mover flow samples at each vertex/band is **not** a probability, **not** flow confidence and **not**
a calibrated uncertainty estimate, and the supported-vs-filled split describes where the fit was directly
observed, not how accurate the residual is. The motion NPZ gains these diagnostic arrays; stabilization
defaults and rendered video output are unchanged.

Row-band line profile (report schema `mlxstab.real_motion_field_audit/4`). The report now also describes the
x-independent row correction `Fit.rows` directly by fitting, per pair, an **unweighted** least-squares line to
each component against the actual row-band centre y positions `(j + 0.5)*height/nr - 0.5` (the same positions
`fit_motion` used). Per pair it reports availability (at least two bands), the band count, the slope vector in
px displacement per px row, the fitted first-to-last-band displacement delta, the pooled nonlinear residual
RMS around the line, and a pooled multivariate linear-explained fraction (`1 - RSS/TSS` with per-component
means). Slope, delta and the fraction are `null` when they are not identifiable (a constant field, or fewer
than two bands), and constant/one-band fields stay finite-safe rather than emitting `nan`/`inf`. The fit
**ignores the support masses**, so legacy `/1` and `/2`//`/3` captures share exactly the same definition; the
report aggregates per-case descriptive statistics and per-pair series for the slope magnitude, delta magnitude,
nonlinear RMS and explained fraction, without combining them into a score and without correlating them with the
cycle maps. This is explicitly **not** a rolling-shutter detector and **not** a physical readout estimate: a
smooth vertical row displacement is rolling-shutter-*like* but can equally arise from other motion or model
mismatch, and no camera readout time, focal calibration or ground-truth readout is inferred. The schema bumped
from `/3` to `/4` because the report JSON shape changed; the motion payload schema is now `mlxstab.motion/4`
(see the fit-gate fallback diagnostics note next) and stabilization behavior is untouched.

Fit-gate fallback diagnostics (motion schema `mlxstab.motion/4`; audit report schema `/5`). The motion-field
audits exposed a case the existing `fallback`/`inlier_frac` fields could not explain: the independent
`00177.MTS` t060 2-second capture (`out/bench/independent_00177_t060_2s/`, motion payload `/3`) has exactly two
fallback pairs, 73 and 75, and both report `fallback=true`, `inlier_frac=0.0` and all-zero
`mesh`/`rows`/support mass. A `/3` payload records neither the valid-sample count, the RANSAC-inlier count nor
which threshold fired, so the reason for those two already-captured fallbacks is **not knowable from that
file** and is not claimed here. `fit_motion` now records `n_valid_samples`, `n_ransac_inliers` and a stable
`fallback_reason` (`none`, `insufficient_valid_samples`, `insufficient_ransac_inliers`,
`low_ransac_inlier_fraction`; the first failing gate wins, preserving the existing
`inl.sum() < min_valid or inlier_frac < 0.15` refusal exactly), serialized as `mlxstab.motion/4` with numeric
counts and the clearly named integer reason code `fallback_reason_code` (`0..3`, with `-1` when unavailable;
an unavailable/manual fit uses `-1, -1, -1` and
`fit_meta_available=false`). `bench/real_motion_field_audit.py` (report schema bumped to
`mlxstab.real_motion_field_audit/5`) reports the counts/reason per pair and case-level reason counts, and
legacy `/1`-`/3` payloads report them as null with `fit_metadata_available=false` rather than invented. A fresh
`/4` capture of these windows (`out/bench/independent_00177_t060_2s_v4/`) labels both pairs 73 and 75 as
`insufficient_valid_samples` (0 valid samples, 0 RANSAC inliers); that preserved capture predates the standard
field name and stores the same integer codes under the pre-standardization `fallback_reason` key, which the
loader reads as an alias. The fresh `/4` capture is a repeat of the already audited 60–62 s window and does not
backfill the old `/3` payload; the old pair labels remain unavailable in that artifact. These are diagnostics
only: the fitted `H`/mesh/rows, the plan and the rendered
pixels are unchanged.

Verification: `tests/test_real_motion_field_audit.py` (81 tests) covers the analytic similarity scale/rotation
round-trip (including a projective `H`), exact residual magnitude statistics, the analytic row-band line
profile (exact linear shear, nonlinear and constant fields, one-band/empty finite-safe fields, legacy `/1`
JSON behavior), unvarying/gradient spatial-cell
proxies, fallback metadata and the per-pair/case-level fit-gate counts/reason (including the legacy
`fit_metadata_available=false` path), frame indexing under analysis stride, strict manifest success/provenance checks,
missing/malformed manifest cases, motion-vs-manifest dimension/fps cross-checks, cross-manifest
source-identity refusals, the non-matching-window allowance, repeated case names retained and distinguished by
`run_id`/`manifest_path`/`source_window`, the support-mass threshold split and per-pair/per-case
supported-vs-filled statistics, the legacy `/1` support-absent path, the no-pickle guarantee (a patched
`np.load` must receive `allow_pickle=False`; object arrays are rejected), JSON-safe finite output and the CLI.
The four real audits below were run on the same inferred-window captures as the composition audit, and the
combined `out/bench/real_motion_field_audit/t060_t800_short_long.json` is generated from the same four
manifests, writing JSON only under the ignored `out/bench/real_motion_field_audit/`.

t060 and t800 are **separate inferred windows of the same local 1080i `00176.MTS`** (60 s and 800 s offsets),
not independent or held-out clips. Each window was captured at two durations: a one-second slice and a
10-second run, with the one-second slice **nested within** its own 10-second capture (t060 1 s inside t060
10 s, t800 1 s inside t800 10 s), so the two durations per window are not independent samples of each other.
All four captures share the case name `baseline`; the combined
`out/bench/real_motion_field_audit/t060_t800_short_long.json` audits them together, keeping four entries
distinguished by `run_id` (`e02_comp_t060_baseline_1s`, `e02_comp_t800_baseline_1s`,
`real_motion_field_capture_t060_10s`, `real_motion_field_capture_t800_10s`), `manifest_path` and
`source_window` (1 s and 10 s at each offset). Every audit analysed all of its consecutive pairs with zero
fallbacks; the fitted globals were similarities with identity-adjacent transforms. Per-pair maxima:

| window | pairs | rotation deg y-down cw (min, max) | scale (min, max) | row RMS max px | row max max px | mesh RMS max px | mesh max max px | dispersion RMS max px |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| t060 1s | 59 | -0.0174, 0.0176 | 0.999487, 1.000671 | 0.190 | 0.322 | 0.267 | 0.975 | 0.262 |
| t800 1s | 59 | -0.0192, 0.0161 | 0.999811, 1.000304 | 0.290 | 0.391 | 0.317 | 0.966 | 0.281 |
| t060 10s | 599 | -0.0933, 0.0489 | 0.999166, 1.001260 | 1.362 | 2.345 | 1.842 | 5.756 | 1.594 |
| t800 10s | 599 | -0.030420, 0.026183 | 0.999625, 1.000414 | 0.290 | 0.448 | 0.475 | 2.195 | 0.420 |

On the one-second windows the fitted inter-frame motion is very close to identity (rotation well under
0.02 deg, scale within about 0.07%), while the residual fields are sub-pixel but non-trivial (mesh maxima
~0.97 px, dispersion RMS ~0.26-0.28 px). Both 10-second windows still have **near-identity global motion**
(rotation within about 0.09 deg at t060 and about 0.03 deg at t800; scale within about 0.13% at t060 and about
0.04% at t800) yet show residual maxima above one pixel (t060 mesh max 5.756 px, t800 mesh max 2.195 px). t060
has the larger maxima, with a mesh maximum of 5.756 px versus 2.195 px at t800 and a dispersion RMS maximum
of 1.594 px versus 0.420 px. The spatial dispersion is smooth (neighbour coherence is high) but, per the
honesty boundary, that is equally consistent with parallax and with model mismatch / rolling-shutter leftovers;
it does not measure depth and is not an object/tree finding. The residuals are still fit outputs that mix
parallax, independent object motion, model mismatch, rolling-shutter leftovers and optical-flow error: they
are **not depth and not object semantics**. These are four inferred windows, not clip-wide, held-out or
perceptual evidence; the residual magnitudes are the fit arrays at band/vertex resolution, the one-second
slices are nested within their 10-second runs, and no default or production wiring changes.

Schema-v2 support-map spot checks added two one-second baseline captures at t060 and t065. The t060 window
had direct support (`mass >= 3.0`) at all 10,030 mesh vertices across 59 pairs. The t065 window had 9,748 of
10,030 vertices supported (97.19%); the 282 filled vertices were concentrated at the lower-left image edge.
At pair 14 in the t065 sample (approximately the same source moment as pair 314 in the t060 10-second run,
whose maximum vertex residual was 5.756 px), 166/170 vertices were directly supported. The maximum vertex
residual there was 5.747 px and belonged to a supported vertex; the four filled vertices had 3.109 px RMS.
This indicates that diffusion fill alone does not explain that high residual. It does not establish flow
accuracy or temporal visual quality: support mass counts usable fit samples, not correspondence correctness,
and these windows come from the same source rather than a held-out clip. The captures and JSON reports are
ignored under `out/bench/support_capture_t060_1s`, `out/bench/support_capture_t065_1s` and
`out/bench/real_motion_field_audit/`.

A third one-second schema-v2 capture sampled the previously flagged t800 edge event. All 10,030 mesh vertices
across its 59 pairs were supported. At pair 23 (the source moment corresponding approximately to pair 112 of
the t800 10-second run), the bottom-right vertex had the largest residual at 1.988 px and support mass 343.4;
all 170 vertices in that pair were directly supported. The older 10-second fit reports a 2.195 px maximum at
the same edge region. This also argues against fill as the cause of that peak, but says nothing about optical-
flow correctness. This capture is stored under `out/bench/support_capture_t800_spike_1s/`.

### Forward-backward cycle-consistency diagnostic (implemented; schema `mlxstab.motion/4`)

The support maps above say *where* a residual was observed, but not whether the optical-flow correspondences
feeding the fit are correct. This motion-field gate adds a spatial forward-backward **cycle-consistency**
diagnostic, never a ground-truth error or a confidence. `observe` already computed the per-sample cycle error
`fb_err = ||f + sampled_b||`, the tolerance `fb_abs + fb_rel*||f||` and whether the forward endpoint is
in-bounds; these are now retained on every sampled point (including points the tolerance rejects, which
`Obs.valid` alone would hide). `fit_motion` aggregates them into three maps on the mesh grid --
`cycle_err_rms` (Gaussian-spatially-weighted RMS of the cycle error in px), `cycle_inconsistency_frac` (the
spatially weighted fraction of finite, in-bounds samples failing the existing tolerance, in [0,1]) and
`cycle_sample_mass` (spatial sample coverage). The maps are built from every finite, in-bounds forward
endpoint **independent of the fit's valid/non-mover mask**, zero-mass vertices are zero, and they are computed
even when the geometric fit falls back. They are a forward-backward **cycle-consistency** diagnostic only:
a consistent cycle does not prove the correspondence is right and an inconsistent cycle does not locate the
error, so this is **not** true correspondence error, **not** ground-truth error, **not** a probability and
**not** a calibrated uncertainty. Stabilization pixels, the fitted `H`/mesh/rows and every fitting decision
are unchanged; the maps are never read by the stabilizer.

`mlxstab.motion/4` stores the three maps `[n_fits, gr+1, gc+1]` plus a per-fit `cycle_map_available` flag and
the fit-gate diagnostics; a
manually constructed Fit (no maps) dumps zero maps tagged unavailable, and one without gate metadata dumps
`-1, -1, -1` (unavailable integer reason code) tagged `fit_meta_available=false`. The loader (`allow_pickle=False`,
unchanged stable error codes plus `bad_range`) still accepts `/1` (no support, no cycle maps) and `/2`
(support, no cycle maps) and `/3` (support + cycle, no fit metadata), loading the missing fields as `None`.
`bench/real_motion_field_audit.py` (report
schema was `/3`, then `/4` with the row-band line profile, now `/5` with the fit-gate diagnostics)
reports the three maps per pair, a per-pair `cycle_consistency` summary and pooled per-case statistics, all
`null`/unavailable for manually constructed and legacy fits. JSON stays deterministic and finite-safe; **no correlation between the cycle maps and the residual
maps is computed or implied**. The three existing short real captures (`support_capture_t060_1s`,
`support_capture_t065_1s`, `support_capture_t800_spike_1s`) are schema `/2` and carry no cycle maps, so they
report null cycle fields. Two focused one-second schema-`/3` captures now cover the t065 and t800 residual
peaks. At the t065 pair-14 5.747 px vertex residual (row 9, column 3), support mass was 48.0, cycle RMS was
1.011 px and 2.78% of the local cycle weight failed the consistency tolerance; pair-wide mass-weighted
inconsistency was 0.95%. At the t800 pair-23 1.990 px bottom-right vertex residual (row 9, column 16), support
mass was 343.2, cycle RMS was 0.557 px and 2.46% of local cycle weight failed; pair-wide mass-weighted
inconsistency was 0.094%. The cycle diagnostic is low at both peaks, but cannot verify correctness: forward
and backward flows can share errors. These windows are the same source, not held-out evidence. Captures and
reports are ignored under `out/bench/support_cycle_t065_1s`, `out/bench/support_cycle_t800_spike_1s` and
`out/bench/real_motion_field_audit/`.

Verification: `tests/test_motion_cycle.py` covers the retained per-sample fields (including rejected-but-
in-bounds and out-of-bounds forward endpoints), the analytic Gaussian spatial aggregation (constant-error,
excluded out-of-bounds/non-finite samples, zero-mass vertices), the fallback path preserving diagnostics and
`None` maps for hand-built observations; `tests/test_real_composition_audit.py` covers the `/4` round-trip,
manual-Fit zero-map/sentinel serialization, malformed `/4` shape/finiteness/range/availability/reason rejection
and legacy `/1`/`/2`/`/3` loading with fit metadata `None`; `tests/test_motion_fit_meta.py` covers each
fallback branch and the reason precedence; `tests/test_real_motion_field_audit.py` covers the per-pair maps, pooled case
statistics, the fit-gate counts/reason and case-level reason counts, JSON safety and the `null` legacy behavior.

Independent-file t060 profile comparison (diagnostic only). `00177.MTS` has SHA-256
`8c5d98d15448ad0557537ba9d02a28fc426185b4e32bdf1be3eb4964fdbea714`, distinct from `00176.MTS`
(`78ba322cfd013054d6ff4846524511e5462ffc9cfbac5acd0200f85bfcc10e93`). A first-second smoke on the former
completed 59 motion pairs and validated the capture/audit path. The matched-offset run then captured
`00177.MTS` at 60–62 s (`out/bench/independent_00177_t060_2s/`, 119 pairs, 2 fallback fits); this offset was
chosen to match the existing `00176.MTS` t060 runs. Both used the same baseline config, weights, 1920x1080
dimensions and 59.94006 Hz decoded cadence. For a duration-matched description, the comparison uses the first
119 pairs from `00176.MTS`'s existing 10-second t060 motion capture and all 119 pairs from the new 2-second
capture:

| row-profile measure | `00176.MTS`, first 119 pairs | `00177.MTS`, 119 pairs |
| --- | ---: | ---: |
| slope magnitude mean / p95 (px displacement per px row) | 0.000222 / 0.000423 | 0.000181 / 0.000387 |
| first-to-last row delta mean / p95 (px) | 0.226 / 0.431 | 0.185 / 0.395 |
| nonlinear residual RMS mean / p95 (px) | 0.0638 / 0.0952 | 0.0392 / 0.0734 |
| linear-explained fraction mean / p95 | 0.369 / 0.759 | 0.462 / 0.855 (117 defined; 2 constant fallback fields) |

The original `/3` `00177.MTS` capture cannot reveal why pairs 73 and 75 fell back. A repeat of the same 60–62 s
window using motion schema `/4` (`out/bench/independent_00177_t060_2s_v4/`) records both as
`insufficient_valid_samples`, with 0 valid samples and 0 RANSAC inliers. Its cycle maps have nonzero in-bounds
sample mass and a 100% mass-weighted tolerance-failure fraction at both pairs (per-vertex cycle-error RMS
308.88 px and 311.30 px). This locates the failure at the existing forward-backward validity gate; it does
not by itself identify the underlying image or flow cause. A contact sheet of the same deinterlaced window
shows pairs 73 and 75 in a very dim interval immediately after two bright frames; the 480 px thumbnails have
mean luma 19.8/19.2 versus 94.9/94.8 for those bright frames. This is consistent with a low-light/low-texture
risk, but it does not prove that darkness caused the cycle failure or distinguish it from blur, motion or
deinterlacing effects. Cycle-map comparison across files remains unavailable because
the older `00176.MTS` capture is `/2`, while `00177.MTS` is `/3` and the repeat is `/4`. Although config,
weights and decoded identity match, the captures' recorded source-code hashes differ because the later captures
include diagnostic instrumentation; this is not a strict same-code replication. Different source-file hashes
and a matched timestamp do not establish independent scene content. The row profiles are descriptive only, not
a rolling-shutter finding or a stabilization-quality comparison.

Next motion-field gate: test support/cycle/row-profile patterns on independent windows before drawing
content-level conclusions. The row-band line profile is a descriptive smoothness/shape summary only, not a
rolling-shutter readout measurement: turning it into a readout-time or focal-calibration estimate would need
independent ground truth and is explicitly out of scope. Cycle consistency is not ground-truth flow error. No
depth or production claim is made.

## Stage 2 progress: E04 flow-initialization prediction (analytic only)

`bench/flow_prediction.py` compares cold, current same-grid warm reuse, map-transported previous flow,
constant-velocity extrapolation, and constant-acceleration extrapolation on exact synthetic pair maps. It
transports each field through the inverse source-to-target maps onto the next pair's source grid and carries
validity through occlusions, disocclusions, and borders. Low-coverage candidates fall back to a zero
initialization. The module is NumPy-only and is not wired into `RaftFlow`; it establishes neither RAFT
initialization quality nor a real-footage speedup. Results are in ignored `out/bench/flow_prediction_e04/`.

On the controlled fixtures, constant velocity exactly predicts linearly increasing translations and constant
acceleration exactly predicts quadratic translations. For repeated 8-degree rotations, same-grid reuse has
0 px mean EPE while transported reuse has 0.195 px; constant-velocity and constant-acceleration
extrapolation lower that to 0.0266 and 0.00461 px. For 4/8/12/16-degree rotations, same-grid and transported
mean EPE are 0.714 and 0.785 px, while the extrapolations reach 0.355 and 0.173 px. In the two-layer
parallax fixture, same-grid EPE is 0.199 px and transported EPE is 0 on shared valid support; transported
support is lower (0.894 versus 0.922 of the frame). In the time-reversed fixture, same-grid EPE is 0.165 px
and transported EPE is 0, with shared support 0.894 versus 0.914. These motion-specific outcomes do not
identify a universal warm-start policy.

`tests/test_flow_prediction.py`, `tests/test_composition_audit.py`, `tests/test_composition_candidate.py`, and
`tests/test_evaluation_correctness.py`: **93 passed** after fixing exact-center mask sampling to ignore
zero-weight neighbours, correcting the image-coordinate rotation helper labels, reversing temporal order for
backward pair sequences, and aligning raw fallback coverage with actual support. No RAFT kernels or media were
involved.

### E04 real-footage flow-initialization benchmark (isolated; two inferred windows, fixed-reference iteration comparison)

`bench/flow_prediction_real.py` is the second E04 gate: the same initialization candidates replayed on
**real consecutive frame pairs**. For each direction (forward and a separate reverse-temporal backward
traversal) it compares `cold`, the current production same-grid reuse, coordinate-transported previous flow,
constant velocity and constant acceleration. Candidate construction reuses `bench.flow_prediction.predict` /
`transport` unchanged; the only changed assumption is the pair map, which is RAFT's own forward/backward flow
sampled bilinearly (an *approximate* inverse with no occlusion/disocclusion truth), so map validity is only
in-bounds and finite. The module is not imported by `stabilize`, changes no default and is not wired into
`RaftFlow`.

Fairness and target. The window is decoded once into `uint8` frames; the reference (a **cold high-iteration
RAFT run**, 24 iterations) and every candidate share those exact frames and the same per-frame network
features. Reported error is the endpoint error of the *refined* flow against the reference in feature-grid
units (multiply by 8 x downscale for source pixels), plus the initialization's own error, coverage, fallback
reason, iteration count and separately measured initialization/refinement wall time. The reference is
model-derived, so the numbers answer "does a predicted initialization land closer to the converged cold
solution at a fixed iteration count?" and are **not** optical-flow ground truth. Each manifest carries the
source sha256, window, preprocessing/downscale/dtypes, weights sha256, frame-stack sha256, config digest and
working-tree code digests; outputs live only under ignored `out/`. Frame load/decode/cache restore is timed as
`timing_s.frames_load`, which excludes the shared-lock acquire wait (recorded separately as
`lock.acquire_wait_s`) and is accurate for both freshly decoded and cache-restored frames. For long real
windows the harness supports shared-lock chunking (`--lock-chunk N`, `--lock-pause S`): one monotonic step
counter is threaded through feature preparation (once per encoded frame), reference generation (once per
reference pair) and candidate replay (once per method/pair), and `HeavyLock.on_step` releases/re-acquires the
lock every N completed frame/pair boundaries. Chunking changes throughput timing only, never the decoded
frames, cached features, references or reported errors; the yielded time is recorded in `lock.pause_total_s`.

Reference-derived maps (offline assumption). Every transport Pair map, in both directions, is built from the
**full-window cold high-iteration reference flows** (`--ref-iters`), not from the candidate's own
low-iteration outputs; the candidate never sees or consumes its own map outputs. The map is therefore exactly
the expensive converged flow the benchmark is trying to avoid. This is an offline, reference-derived-map
assumption that bounds an idealized replay, does not model an online/production pipeline, and implies no
online or production readiness.

One inferred window at two candidate iteration counts (12 and 3), not two real windows. It is the inferred
t060 offset on a local 1080i camcorder source, 60.0-60.2 s, deinterlaced to 59.94 fps, 12 frames, 2x flow
downscale, 11 pairs, cold reference at 24 iterations. Both directions were evaluated on the common pair set
3..10 (constant acceleration needs three prior pairs); coverage was 1.0 and there were no fallbacks in any
row. Mean refined EPE in feature-grid px against the 24-iteration cold reference:

| iters | direction | cold | same_grid | transported | constant_vel | constant_accel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 12 | forward | 0.00884 | 0.01139 | 0.01042 | 0.01313 | 0.01463 |
| 12 | backward | 0.008826 | 0.01049 | 0.00964 | 0.01283 | 0.01559 |
| 3 | forward | 0.01589 | 0.01747 | 0.01709 | 0.02021 | 0.02478 |
| 3 | backward | 0.01547 | 0.01717 | 0.01672 | 0.02059 | 0.02650 |

On this short, low-motion window cold was already closest to the converged reference at both iteration
counts: transport was slightly closer than same-grid reuse, but every warm start was *further* from the
reference than cold, and extrapolation was worse still. Feature-grid EPE of about 0.009-0.027 is roughly
0.14-0.43 source px at 2x downscale. The parity test on the batch-2 bidirectional path versus two independent
fp32 calls is a <1e-3 EPE check on a random-init case, not a bit-identical equality proof, so the sign is
best read as consistent across the tested paths rather than proven free of any path artefact. This is a
negative result for warm starting **on this window only**; it does not establish a clip-wide policy, a
speedup, a downstream stabilization change, or behaviour on faster motion, and no production wiring is
proposed.

T800 window at two candidate iteration counts (3 and 12). A second inferred window on the same local 1080i
camcorder source is the inferred 800-801 s offset (`00176.MTS`), not a second real clip and not used to derive
the t060 result. It was run with the shared-lock chunking (`--lock on --lock-chunk 30 --lock-pause 12`) at
candidate iteration counts 3 and 12 against the same fixed 24-iteration cold reference. Both runs replayed the
**same verified cached `uint8` frame stack** (source sha256
`78ba322cfd013054d6ff4846524511e5462ffc9cfbac5acd0200f85bfcc10e93`; frame-stack sha256
`be256da2b786879242725a580471cce25ce18e7bf16d73f79838dce6c2100a66`; 60 decoded deinterlaced frames at
59.94005994 fps, 59 pairs, 2x downscale). Both directions were evaluated on the common target set exactly
3..58 (56 pairs); all common summaries had **zero fallbacks**, and coverage minimum was 1.0 except constant
acceleration (forward 0.99901961, backward 0.99914216, at both iteration counts).

Mean refined EPE in feature-grid px, from `results[direction][method].summary_common.refined_epe_mean_low`,
against the 24-iteration cold reference:

| iters | direction | cold | same_grid | transported | constant_vel | constant_accel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | forward | 0.01512177 | 0.01801981 | 0.01693655 | 0.02096558 | 0.04030430 |
| 3 | backward | 0.01517254 | 0.01781562 | 0.01703884 | 0.02160822 | 0.04321982 |
| 12 | forward | 0.00788815 | 0.01200960 | 0.01014106 | 0.01381540 | 0.02847226 |
| 12 | backward | 0.00800310 | 0.01181215 | 0.01006289 | 0.01411258 | 0.02748056 |

Cold was closest to the high-iteration reference in all four direction/iteration conditions; transported was
modestly closer than same-grid, and both extrapolation approaches were worse. This is a **negative result for
these warm starts on this window only**, with the same caveats as t060: the target is the 24-iteration cold
RAFT reference, **not optical-flow ground truth**; the transport maps are reference-derived and offline; and
this is one inferred one-second window, not broad evidence or production readiness. At 2x downscale the
source-pixel EPE conversion factor is 16 (about 0.13-0.69 source px for the values above).

Each manifest succeeded (`out/bench/flow_prediction_real/t800_iters3/manifest.json` and
`.../t800_iters12/manifest.json`, both under ignored `out/`). Both record 24 shared-lock releases with
`chunk=30`, `pause=12 s`, and aggregate pauses of 276.203 s (3 iters) and 276.137 s (12 iters);
`timing_s.benchmark` is 375.029 s and 479.663 s respectively and **includes these pauses**, so it is not a
compute-only time. Local footage and generated outputs remain untracked/ignored.

Opt-in per-iteration budget scan (offline oracle; one completed real t800 window). A new
offline-only, non-default mode on the same harness answers the follow-up E04 question "how many refinement
iterations does each initialization need to reach a given distance to the cold reference?". `--budget-scan`
(plus `--budget-max-iters`, default 24, and comma-separated `--budget-thresholds`, default `0.005,0.01,0.02`
feature-grid px) reuses the same decoded frames, cached features, cold high-iteration reference, pair ordering,
fallback/insufficient-history semantics and shared-lock chunking, and for every method/direction refines each
target pair one iteration at a time from the initialization the normal policy produced, up to the max, recording
the mean (and p95/max) low-grid EPE against that pair's cold reference after every iteration. For each
threshold it records the first iteration at or below it, or that it was **censored** (not reached by the max).
The **history invariant** is enforced: the flow written for the following pair's initialization is always the
state after exactly `--iters` iterations (the normal fixed-budget replay policy), so scan iterations beyond
`--iters` never leak into a later initialization; `--budget-max-iters` must be `>= --iters`. Each per-iteration
device result is evaluated before the EPE check, and per-iteration device-update time is recorded separately
from host EPE/oracle-check time. The fixed-budget `summary`/`summary_common` remain the `--iters` equivalents
and the mode adds a `budget_scan` block (per method, with summarised reach rates, crossing iterations and
cumulative separated times) to the atomic manifest. Because the trace deliberately runs every pair to the
scan end to measure the whole EPE curve, the block keeps those sums as clearly labelled **full-scan** totals
and adds **prefix-summed measured costs per threshold**: each usable pair consumes its first crossing
iteration when reached and otherwise its number of measured scan steps (normally the max, less only if output
went non-finite early), and the per-step device-update and host EPE/oracle-check times over that prefix are
summed (with mean iterations consumed and mean per-pair costs); censored pairs still contribute their work and
there are no separate early-stopped executions. Each per-iteration row additionally records a
**candidate-observable** update magnitude `flow_update_low`: the mean/p95/max per-pixel magnitude of
`low - previous_low` (feature-grid px), where `previous_low` is the effective initialization on the first
iteration and the previous step's flow afterwards. It is computed **only from the candidate's own
initialization/current/previous prediction** -- the cold reference never enters it -- so it is deliberately
distinct from the oracle `epe_low`. Its host computation cost is measured separately as `update_mag_check_s`
and never mixed into `update_s` (device) or `epe_check_s` (oracle EPE); the summary adds prefix-summed
`signal_check_s_*` and full-trace `full_scan_signal_check_s_*` timings, with `total_with_signal_s_*` adding the
signal time to the otherwise unchanged `total_s_*` (`= update_s + epe_check_s`). A second candidate-observable
signal, `photometric_low`, is described below. These diagnostics **could**
support a future convergence-based stopping rule, but **no stop rule is implemented or selected** -- the
threshold scan remains an offline oracle comparison to the cold reference. This is **oracle-based,
reference-derived and offline**; it is **not a deployable stopping rule** and makes no speedup, online or
production-readiness claim. When the
mode is off, fixed-budget run behaviour and manifests are unchanged (no `budget` config and no `budget_scan`
block). A tiny public synthetic smoke (4 frames, 64x48, `--iters 3 --budget-max-iters 6`, ignored `out/`) ran
end to end with real RAFT, but its numbers are a smoke test only.

**Completed real t800 scan.** Authoritative run: ignored manifest
`out/bench/flow_prediction_real/t800_budget_scan_iters12/manifest.json`, status `success`. It is the same
inferred 800-801 s offset on local 1080i `00176.MTS` and the same verified cached stack used in the fixed
t800 runs (source sha256 `78ba322cfd013054d6ff4846524511e5462ffc9cfbac5acd0200f85bfcc10e93`; frame-stack
sha256 `be256da2b786879242725a580471cce25ce18e7bf16d73f79838dce6c2100a66`; 60 deinterlaced frames at
59.94005994 fps, 59 pairs, 2x downscale). Both directions were evaluated on the common target set exactly
3..58 (56 pairs). Candidate history is 12 iterations, the scan ran to a maximum of 24, and the cold reference
is 24 iterations; thresholds are 0.005, 0.01 and 0.02 feature-grid px (source-grid conversion factor 16 at 2x
downscale). Every method/direction had zero fallbacks and zero non-finite stops; coverage minimum was 1.0
except constant acceleration (forward 0.99901961, backward 0.99914216). Because the scan runs every pair to
24 to measure the full EPE curve, the per-threshold costs below are **prefix sums from that measured full
trace** (each usable pair consumes its first crossing iteration when reached, otherwise its number of measured
scan steps), not separate early-stopped runs.

For the 0.01 threshold, reach rate, mean iterations consumed (censored pairs consume 24, so this is averaged
over all 56 pairs) and mean prefix total seconds per pair (`update_s_mean + epe_check_s_mean`):

| direction | method | reach | mean iters | mean prefix s/pair |
| --- | --- | ---: | ---: | ---: |
| forward | cold | 100% | 9.18 | 0.1666 |
| forward | same_grid | 64.3% | 17.09 | 0.2931 |
| forward | transported | 71.4% | 13.54 | 0.2385 |
| forward | constant_vel | 55.4% | 19.00 | 0.3356 |
| forward | constant_accel | 3.6% | 23.38 | 0.4008 |
| backward | cold | 100% | 9.39 | 0.1708 |
| backward | same_grid | 67.9% | 16.05 | 0.2717 |
| backward | transported | 73.2% | 13.25 | 0.2224 |
| backward | constant_vel | 50.0% | 19.77 | 0.3263 |
| backward | constant_accel | 7.1% | 23.52 | 0.4146 |

At 0.02, cold reached 100% in both directions (mean 1.93/1.89 steps, 0.0490/0.0506 s per pair) and
transported also reached 100% in both (2.16/2.07 steps, 0.0539/0.0491 s per pair); constant velocity reached
87.5% in each direction (mean consumed 6.29/6.30) and constant acceleration reached 94.6% forward and 100%
backward (15.07/14.38 steps). At the strict 0.005 threshold cold reached 100% in both directions
(18.29/18.46 steps) while every warm method reached 0 except constant acceleration forward, which reached
1/56 (1.8%).

Interpretation. Cold remains the fastest and most reliable path at these tested thresholds; transported is the
strongest warm initializer but is slower and less reliable than cold. This is a negative result for warm
starts on this one inferred window, not truth-quality or clip-wide evidence. The thresholds are mean EPE to
the 24-iteration cold RAFT reference, not optical-flow ground truth, and the transport maps remain
reference-derived and offline. Every iteration explicitly evaluates (`mx.eval`) before the host oracle EPE
check; update time and oracle-check time are separated, but these are instrumented timings, not
throughput/production claims. The benchmark wall time 566.789 s includes 276.202 s of aggregate shared-lock
pauses (24 releases, chunk 30, pause 12 s) and is not compute-only time.

Verification: `tests/test_flow_prediction_real.py` (59 tests) covers the direction/traversal mapping, the
transport direction (a non-zero inverse map must shift the sample), identity-map equivalence of transported
and same-grid reuse, fallback and insufficient-history reporting, the common-pair intersection, deterministic
replay/orchestration with a fake model, the frame-cache request guard and tamper detection, JSON/manifest
safety (including non-finite -> null and atomic writes), that `timing_s.frames_load` excludes the shared-lock
acquisition wait while `lock.acquire_wait_s` keeps tracking it, and the shared-lock chunking: the one monotonic
step gate spanning feature preparation/reference generation/candidate replay, and `--lock-chunk`/`--lock-pause`
option wiring and config/manifest persistence through a fake lock, so default synthetic runs never touch a real
lock. The budget-scan tests add threshold parsing/validation, the `max_iters >= iters` refusal, first-crossing
reach-rate/censor summaries, per-threshold prefix cost totals/means for reached and censored pairs (including
`total_s = update_s + epe_check_s`), the fixed-history-at-`iters` invariant, separated update vs EPE-check
timing, non-finite-step stopping and off-mode-manifest invariance. Additional candidate-observable tests pin
the exact expected `flow_update_low` mean/p95/max for a known fake trajectory, prove the signal ignores the
cold reference (changing the reference changes `epe_low` but not `flow_update_low`) and prove its host time is
measured separately from `epe_check_s`/`update_s` (prefix and full-trace `signal_check_s_*` totals, with
unchanged `total_s_*`), and the manifest wiring asserts the `candidate_observable`/`flow_update_low` fields.
The photometric tests pin the Rec.601 box downscale against a manual computation, the 8x feature-grid-to-
preprocessed-pixel scaling with an exactly recoverable known translation, the replicate-pad coordinate offset,
out-of-bounds and non-finite flow/value exclusion (empty statistics stay `null`), global-brightness-offset
removal, the low-texture mask, traversal-direction routing of the warper closure, and that the signal ignores
the cold reference; the budget-scan timing tests keep device update, update-magnitude check, photometric check
and oracle EPE check as four separate measured costs, and the manifest/JSON wiring asserts the `photometric_low`
statistics, the per-iteration `photometric_check_s`, and the `candidate_observable`/`stopping_rule`
descriptions.
CodeGraph `affected` found no linked test
file for the benchmark module, hence direct tests; `tests/test_flow_prediction_real.py` passed 59 tests and
direct full verification `.claude/verify.sh` passed **539 tests in 22.39 s** (independent run). A tiny public
synthetic smoke
(`--source synthetic`, 64x48) also completes end to end with real RAFT and exercises the step gate.

**Completed real t800 candidate-observable update-magnitude scan.** A second completed run on the same inferred
horizon records the per-iteration candidate-observable signal in the ignored atomic manifest
`out/bench/flow_prediction_real/t800_budget_scan_observable_iters12/manifest.json` (status `success`). It reuses
the same verified cached `uint8` frame stack as the fixed t800 runs (source sha256
`78ba322cfd013054d6ff4846524511e5462ffc9cfbac5acd0200f85bfcc10e93`; frame-stack sha256
`be256da2b786879242725a580471cce25ce18e7bf16d73f79838dce6c2100a66`; 60 deinterlaced frames at
59.94005994 fps, 59 pairs, 2x downscale), with candidate history 12 iterations, reference/scan maximum 24
iterations and thresholds 0.005/0.01/0.02 feature-grid px. Both directions were evaluated on the common target
set exactly 3..58 (56 pairs). The run recorded 24 shared-lock releases and 276.212 s aggregate pause;
`timing_s.benchmark` is 567.828 s and **includes those pauses**, so it is not compute-only time. No local
footage or generated output is tracked.

At iteration 12 the mean candidate flow-update magnitude (`flow_update_low`, mean/p95 over pixels, feature-grid
px) is slightly **smaller** for the warm starts even when their reference EPE is worse:

| direction | method | update mean | update p95 | EPE vs 24-iter cold |
| --- | --- | ---: | ---: | ---: |
| forward | cold | 0.003019 | 0.008417 | 0.007888 |
| forward | same_grid | 0.002941 | 0.007881 | 0.012010 |
| forward | transported | 0.002955 | 0.007977 | 0.010141 |
| backward | cold | 0.003080 | 0.008608 | 0.008003 |
| backward | same_grid | 0.002962 | 0.007877 | 0.011812 |
| backward | transported | 0.002977 | 0.007957 | 0.010063 |

This directly cautions against treating a small RAFT update as confidence by itself.

**Exploratory stop-threshold sweep (in-sample).** The companion ignored `.../update_signal_sweep.json` pools the
560 common pair sequences (10 method/direction rows x 56 pairs), evaluates mean EPE <= 0.01 feature-grid px
against the 24-iteration cold reference, and triggers on **only** the candidate update mean/p95 (the reference
never triggers), requiring both below their thresholds for 1-4 consecutive steps. It searches the 13 mean
thresholds and 12 p95 thresholds recorded in the JSON (624 combinations). Thresholds were searched and scored on
the same one-second window, so this is descriptive/in-sample, not held-out validation. The best zero-false-stop
rule reached only 3/560 pairs (0.54%) and consumed 23.95 iterations on average. The best rule with at least 70%
reach used patience 4 with mean <= 0.0032 and p95 <= 0.01, but had 205 false stops among 392 stops (52.3%) and
consumed 15.51 iterations averaged over all pairs. This does not support selecting a deployable stopping
threshold. It complements the prior oracle scan: warm starts remain worse than cold at fixed EPE on this window.
The fixed-reference caveats above apply unchanged: the target is the 24-iteration cold RAFT reference, not
optical-flow ground truth; the transport maps are reference-derived and offline; this is one inferred one-second
window; and no stop rule is implemented or selected.

**Candidate-observable photometric-consistency diagnostic (held-out t063 association audit below; no threshold selected).** The
opt-in budget scan now also records a second candidate-observable signal, `photometric_low`, in every
per-iteration row: robust (mean/median/p95/max) absolute luminance-residual statistics after warping the
**source** preprocessed frame toward the **target** preprocessed frame at the candidate's own current
low-resolution flow. It uses only the two decoded/preprocessed frames and the candidate flow -- the cold
reference flow and the oracle EPE never participate -- so it is distinct from both `epe_low` and
`flow_update_low`. Luminance is Rec.601 of the same integer box downscale `RaftRunner._prep_pad` applies, so the
residual is measured on the exact preprocessed content. Geometry: feature cell `(fx, fy)` maps to preprocessed
pixel `(8*fx - pad_left, 8*fy - pad_top)` (the `8 * downscale` feature-to-source scaling with the replicate-pad
offset removed) and the warped target is that location plus `8 * flow`; source cells in the pad, targets out of
the preprocessed frame, and non-finite flow/values are excluded, with `valid_frac` reporting the retained
fraction. A per-pair luminance-median difference removes a global exposure shift, and because flat regions
carry no photometric signal `resid_textured` restricts the statistics to source cells above
`PHOTOMETRIC_TEXTURE_MIN` with `texture_frac` reporting their share. Its host cost is measured separately as
`photometric_check_s`, never mixed into `update_s` (device), `epe_check_s` (oracle EPE) or `update_mag_check_s`
(the other candidate-observable signal); the summary adds prefix-summed `photometric_check_s_*` and full-trace
`full_scan_photometric_check_s_*` totals, with `total_with_observables_s_*` adding both signal costs to the
unchanged `total_s_*`. A documented boundary: the warp maps a feature cell to the top-left preprocessed pixel of
its 8x8 upsampled block and applies `8 * flow` directly rather than reconstructing RAFT's learned
convex-upsampling sub-cell weighting, so an alignment-dependent residual floor is possible even at the true
flow; non-global exposure change, blur, rolling shutter, compression and non-Lambertian content can also
dominate it. This was originally an implementation and diagnostic-cost record only. A single held-out real run
now records the signal (see the completed t063 run below); even so, that one window selects no threshold, and
no stopping rule or speedup is claimed.

**Completed held-out t063 run and observable-signal audit (association only; no threshold selected).** The
held-out real run is complete at `out/bench/flow_prediction_real/t063_budget_scan_photo_iters12/manifest.json`
(status `success`; source `00177.MTS`, 63-64 s window, 59 pairs, 2x downscale, 12 candidate / 24 reference
iterations, both directions; manifest sha256
`9ac3366e939b10a8001e5c8d71413d2cae9fc6c42b3d68483477e7f6e427e0ad`). A new offline evaluator,
`bench/flow_prediction_observable_audit.py` (schema `mlxstab.flow_prediction_observable_audit/1`), reads
successful budget-scan manifests, validates their structure (refusing missing/non-JSON/wrong-schema/
non-success/scan-disabled input as a per-case error with exit code 2), excludes and counts
`insufficient_history` trajectories and null/non-finite signal/EPE samples, and reports a dependency-free
average-tie Spearman rank correlation between each candidate-observable signal and `epe_low.mean` **within
each trajectory** (one method/direction/target pair across the scan iterations). It runs no RAFT, decodes no
media, renders no frame, changes no default, and searches/selects no threshold: it is an offline association
diagnostic, not accuracy and not stop-policy validation.

On the held-out manifest: 590 trajectories (2 directions x 5 methods x 59 targets), 576 usable, 14
`insufficient_history`. Per-trajectory signal-vs-EPE Spearman was median 0.970 / mean 0.929 / positive
fraction 1.000 for `flow_update_low.mean`, and median 0.556 / mean 0.408 / positive fraction 0.792 for
`photometric_low.resid_textured.median`, both over the same 576 trajectories with no null signal/EPE samples.
Stratified by method, the photometric median ranged 0.170 (`cold`) to 0.874 (`constant_acceleration`), so the
photometric association was much less consistent across initializations than the update magnitude's. Median
`valid_frac`/`texture_frac` were 0.985/0.167 (mins 0.970/0.143); the per-iteration photometric check averaged
1.09 ms (15.09 s over 13824 usable scan rows) against 0.139 ms for the update-magnitude check. The descriptive
cross-sectional view (dependent samples: repeated pairs/methods/iterations, and the oracle reference is not
ground truth) fell from 0.92 (update) / 0.81 (photometric) at iteration 1 to 0.19 / 0.02 at iteration 24. The
JSON artifact is ignored `out/bench/flow_prediction_observable_audit/t063_photo_iters12.json`. **These are one
window's association numbers; they do not select a threshold and must not be read as flow accuracy or as
stop-rule validation.**

Verification for the audit: `tests/test_flow_prediction_observable_audit.py` (28 weight-free tests) pins the
average-tie rank correlation and its constant/too-few-point reasons, the manifest structure/success/budget-scan
refusals, `insufficient_history` and null/non-finite signal skips, method/direction stratification, the
descriptive cross-sectional view, deterministic JSON safety and the stdout/atomic-`--out` CLI. The direct test
run passed 87 tests for the audit plus `tests/test_flow_prediction_real.py`, and `.claude/verify.sh` passed
**567 tests in 20.98 s** (independent run). CodeGraph `affected` found no linked test file for the new
standalone module, so the direct tests above were run.

**Second and third held-out runs (t300, t090) and combined three-window observable-signal audit (association
only; no threshold selected; provisional photometric decision).** A second held-out real budget-scan run is complete at
`out/bench/flow_prediction_real/t300_budget_scan_photo_iters12/manifest.json` (status `success`; manifest
sha256 `edf8686baccf62374f1d8910d81ad306dd8409d9169cfd3167f882e9d97f35e0`). It is a **different source file
and window** from t063: local 1080i `00176.MTS` (sha256
`78ba322cfd013054d6ff4846524511e5462ffc9cfbac5acd0200f85bfcc10e93`) at the inferred 300-301 s window, its
own verified 60-frame deinterlaced stack (`frames_sha256`
`24bdf5e1e5183685e7cee0a1f261de25a57300e6d44159986d53e957a273acd4`, 59.94005994 fps, 59 pairs). All ten
method/direction common summaries had zero fallbacks and `coverage_min` 1.0. `timing_s.benchmark` was 583.115 s
including 276.226 s of aggregate shared-lock pauses (24 releases, chunk 30, pause 12 s), so it is not
compute-only; `timing_s.frames_load` was 2.112 s, excluding the separately recorded 0.031 s acquire wait.

A third held-out real budget-scan run is complete at
`out/bench/flow_prediction_real/t090_budget_scan_photo_iters12_halfsec/manifest.json` (status `success`;
manifest sha256 `730a72221e289c17e43b446fdfd9921a917492b2dcd6fca9ff83eac9fb71c3ca`). It is a 0.5 s window at
the inferred 90.0-90.5 s offset on the **same** local 1080i `00177.MTS` source as t063 (`source_sha256`
`8c5d98d15448ad0557537ba9d02a28fc426185b4e32bdf1be3eb4964fdbea714`) but a **non-overlapping** window, with
its own verified 30-frame deinterlaced stack (`frames_sha256`
`2189dd0939381837012057872cbc4fdeb7dd8c3b8b9c08250853fe835ac54ab9`, 59.94005994 fps, 29 pairs). All ten
method/direction common summaries had zero fallbacks and `coverage_min` 1.0. `timing_s.benchmark` was 281.945 s
including 132.112 s of aggregate shared-lock pauses (12 releases, chunk 30, pause 12 s), so it is not
compute-only; `timing_s.frames_load` was 0.614 s, excluding the separately recorded 0.017 s acquire wait.

All three held-out runs share the same resolved config and `config_digest`
`1070e8e89ab4e542b186880773452e6c752fb81bca0d89f52714892fcf5c6543`: 2x downscale, 12 candidate / 24
reference+scan iterations, thresholds 0.005/0.01/0.02 feature-grid px, float32 flow and correlation, both
directions (forward and a separate reverse-temporal backward traversal), the same five methods (`cold`,
`same_grid`, `transported`, `constant_velocity`, `constant_acceleration`), the same weights sha256
`84c1f730d4620ef4dd0a2eeeef97db7a729fc4190ee05cdd0169030fc92fccc1` and code revision
`14f6f606189857d7e89515466888f8a48f63d196` (`dirty_relevant=true`). t063 and t300 each have 59 pairs; t090 has
29 pairs. The three windows are **non-overlapping**, but t063 and t090 **share one source file and hash**
(`00177.MTS`) while t300 uses a **different source file and hash** (`00176.MTS`, sha256
`78ba322c...`). They are still only **three short windows** (two one-second, one half-second): they are not
statistically independent (within-window temporal autocorrelation, shared processing, and for t063/t090 a
shared source file, with possibly shared scene/equipment character) and they do not support generalization, a
scene-level hold-out, or any clip-wide claim. The combined offline audit is
`out/bench/flow_prediction_observable_audit/t063_t300_t090_photo_iters12.json` (schema
`mlxstab.flow_prediction_observable_audit/1`; all three cases `ok`; 2 signals). t063 and t300 each had 590
trajectories (2 directions x 5 methods x 59 targets), 576 usable and 14 excluded as `insufficient_history`;
t090 had 290 trajectories (2 x 5 x 29), 276 usable and 14 excluded. No case had null-signal or null-EPE
samples. Per-trajectory signal-vs-oracle-EPE Spearman and the photometric coverage/overhead:

| window | signal | median | mean | positive frac | valid median (min) | texture median (min) | check mean (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t063 | `flow_update_low.mean` | 0.970 | 0.929 | 1.000 | - | - | 0.139 (update) |
| t063 | `photometric_low.resid_textured.median` | 0.556 | 0.408 | 0.792 | 0.985 (0.970) | 0.167 (0.143) | 1.091 (15.088 s / 13824 rows) |
| t300 | `flow_update_low.mean` | 0.234 | 0.293 | 0.941 | - | - | 0.135 (update) |
| t300 | `photometric_low.resid_textured.median` | -0.080 | 0.044 | 0.458 | 0.985 (0.978) | 0.833 (0.814) | 1.126 (15.564 s / 13824 rows) |
| t090 | `flow_update_low.mean` | 0.634 | 0.623 | 1.000 | - | - | 0.128 (update) |
| t090 | `photometric_low.resid_textured.median` | 0.453 | 0.401 | 0.826 | 0.985 (0.982) | 0.622 (0.611) | 1.077 (7.135 s / 6624 rows) |

Per-method medians of the same two signals (each method pooled over both directions and its window's targets:
59 for t063/t300, 29 for t090):

| window | method | update median | photometric median |
| --- | --- | ---: | ---: |
| t063 | cold | 0.965 | 0.170 |
| t063 | same_grid | 0.967 | 0.507 |
| t063 | transported | 0.969 | 0.482 |
| t063 | constant_velocity | 0.977 | 0.687 |
| t063 | constant_acceleration | 0.967 | 0.874 |
| t300 | cold | 0.177 | -0.586 |
| t300 | same_grid | 0.197 | -0.153 |
| t300 | transported | 0.207 | -0.366 |
| t300 | constant_velocity | 0.239 | 0.199 |
| t300 | constant_acceleration | 0.766 | 0.988 |
| t090 | cold | 0.620 | -0.087 |
| t090 | same_grid | 0.615 | 0.375 |
| t090 | transported | 0.612 | 0.360 |
| t090 | constant_velocity | 0.636 | 0.564 |
| t090 | constant_acceleration | 0.779 | 0.978 |

Across the three windows the update-magnitude association is always positive (median 0.970 / 0.234 / 0.634;
positive in 100% / 94.1% / 100% of trajectories) while the photometric association swings widely (median 0.556 /
-0.080 / 0.453; positive in 79.2% / 45.8% / 82.6%). Texture coverage does not track it: t300 has the highest
median texture fraction (**~0.833** versus ~0.167 at t063 and ~0.622 at t090) yet the weakest photometric
association. The per-method photometric medians stay strongly initialization-dependent at every window, from
-0.586 (`cold`, t300) / -0.087 (`cold`, t090) to 0.988 (`constant_acceleration`, t300) / 0.978
(`constant_acceleration`, t090). The photometric residual is therefore inconsistent across windows and across
methods within a window. The descriptive cross-sectional view (dependent samples: repeated
pairs/methods/iterations, and the oracle reference is not ground truth) fell over iterations 1-24 from 0.921 to
0.185 (update) / 0.811 to 0.018 (photometric) at t063, 0.823 to 0.297 / 0.778 to 0.011 at t300, and 0.852 to
0.365 / 0.900 to 0.078 at t090.

**Provisional experiment decision:** on the combined evidence of these three short windows, the photometric
residual should **not** be accepted as a standalone convergence proxy. This is a provisional experiment
decision, not a settled negative result and not an inferential rejection: the windows are non-overlapping but
t063 and t090 share one source file, only t300 uses a different file and hash, and the windows are far too
short to generalize; **future validation on independent and longer windows is still required** before the
signal is rejected or a threshold is chosen. **No threshold is selected or searched, no true optical-flow
accuracy is claimed, and no speedup or production readiness is implied.** The photometric check remains the
more expensive observable (~1.08-1.13 ms per iteration versus ~0.128-0.139 ms for the update-magnitude check)
for a signal that is not consistently associated with the oracle EPE.

Next E04 gate. The basic flow-update magnitude signal has now been screened and rejected as sufficient by itself
on these three windows. The photometric residual is now available as a second, independent candidate-observable
signal and has been run on three non-overlapping (but not statistically independent) held-out windows (t063,
t300 and t090) for association diagnosis; the combined audit suggests it is inconsistent and, provisionally,
should not be accepted as a standalone convergence proxy, but no threshold has been selected and it has not
been shown to help. A candidate-observable stopping rule still needs validation on independent and longer
windows and stronger evidence such as depth/occlusion/content-aware confidence; matched downstream
stabilization quality also remains untested. No production wiring/default change or readiness claim is made,
and these remain three short inferred windows.

### E04 cold-only bidirectional cycle-consistency pilot (opt-in, isolated; diagnostic only; no threshold)

PLAN/STATUS name occlusion/visibility as a missing evidence source, and CodeGraph inspection found
`RAFT.flow_bidir` already performs same-pair forward/backward refinements as one batch-2 call while the
per-iteration runner scanned one direction at a time. A new **opt-in, cycle-only** pilot in
`bench/flow_prediction_real.py` therefore scans the forward (`k -> k+1`) and backward (`k+1 -> k`) candidates of
each pair **together, from the cold zero initialization, one iteration at a time** through an isolated
`RaftRunner.begin_bidir_scan` that mirrors `RAFT.flow_bidir`/`reference_pair` (production `RaftFlow` is not
touched). `--cycle-scan` (with bounded `--cycle-max-iters`, default 24) cannot be combined with
`--budget-scan`; the standard method/direction replay is skipped, the `results` block is empty and everything
is written under an independent `cycle_scan` block, so manifests/readers and schema `/1` are preserved and no
default changes.

The candidate-observable statistic `cycle_low` is the forward-backward cycle residual in feature-grid pixels,
`|fwd(p) + bwd(p + fwd(p))|` and `|bwd(q) + fwd(q + bwd(q))|`, aggregated to mean/median/p95/max plus
`valid_frac`. `cycle_residual(fwd_flat, bwd_flat, h, w, fwd_valid=None, bwd_valid=None)` accepts only the two
**same-iteration, same-pair, same-model** candidate fields, the grid geometry and optional validity masks --
never a cold reference field or EPE; out-of-bounds mapped points and non-finite vectors are excluded
explicitly. Device `update_s`, host `cycle_check_s` and host oracle `epe_check_s` are measured separately, and
the oracle EPE against the existing 24-iteration cold reference is recorded next to but separate from the
cycle residual. This is a cycle-consistency/visibility **diagnostic**: cycle errors can also mean wrong flow,
blur or independent motion, a consistent cycle does not prove visibility, and a low residual is not true-flow
accuracy. No threshold is selected and no stop rule is implemented.

Verification. `tests/test_flow_prediction_real.py` (73 tests) adds deterministic cycle tests: analytic
inverse-consistent fields give a zero residual, a known translation disagreement gives exactly 1 px, and
border/out-of-bounds/non-finite coverage and validity masks are pinned. They prove the signal ignores the cold
reference (changing the reference changes oracle EPE but not `cycle_low`), pin per-pair/same-iteration paired
state, separated device/host-cycle/host-oracle timing (including that `full_scan_candidate_total_s = update_s +
cycle_check_s` and `full_scan_total_with_oracle_s` adds only the oracle), non-finite stopping, the manifest/JSON
wiring, and the `--cycle-scan`/`--budget-scan` mutual exclusion. A weight-dependent parity test runs the paired
stepper a full `ref_iters` and matches `RaftRunner.reference_pair` within the repository's established `< 1e-3`
feature-grid EPE tolerance (it reproduces the reference exactly, EPE 0.0, on the small fixture). The offline
audit (`bench/flow_prediction_observable_audit.py`) now also accepts a `cycle_scan` manifest and the
`cycle_low.resid.mean` signal (explicit `--signal`, so the default report is unchanged);
`tests/test_flow_prediction_observable_audit.py` (34 tests) covers that acceptance and signal, and the new
cycle coverage entries (`cycle_low.valid_frac` median/min, `cycle_check_s` mean/total) with nulls for the
absent kind. CodeGraph
`affected` found no linked test file for either benchmark module, so the direct tests above were run.

**Visibility toy check (48x36, four exact pairs).** One weight-free test drives `cycle_residual_fields` with
the exact `parallax_two_layer` analytic pairs only (forward field `pair.forward(grid) - grid`, backward field
`pair.inverse(grid) - grid`, no ground-truth masks). It scores the forward residual on the helper's
`fwd_valid` support against the pair's own occlusion `~pair.forward_valid`, giving a forward occlusion AUROC
of **1.0**. For the reversed pairs it scores the reversed **forward** residual (`rfields['fwd_resid']`) on the
reversed `fwd_valid` support against `~rev.forward_valid` -- the reversed pair's forward source is original
frame k+1, so this label is exactly the original disocclusion -- giving a reverse disocclusion AUROC of
**1.0**. The score and label are taken on the same source grid in each case; comparing the reversed pair's
backward residual against the reversed forward-valid label would mix two different source grids and is not
meaningful. Both AUROCs are **exact analytic fields on one 48x36 toy with four pairs**, not a real model or
real-footage result. It is not semantic or depth-understanding evidence and implies no production or
occlusion-understanding claim.
`.claude/verify.sh` passed **586 tests in 21.10 s** (independent run); the direct pair passed 106 tests. No
changes landed in `mlxstab/` and no production default changed (the pre-existing working-tree edits in
`mlxstab/motion.py`/`mlxstab/stabilize.py` are unrelated to this pilot).

**Completed pilot on the existing held-out frame caches** (ignored `out/`, 2x downscale, float32 flow and
correlation, 24 paired candidate steps, 24-iteration cold reference, `--lock on --lock-chunk 30 --lock-pause
12`). All five runs were `status=success` with zero non-finite stops and `coverage_min` 1.0; they share
`config.cycle = {"scan": true, "max_iters": 24}` and `config_digest` `e873905d5285b3af...`. `cycle_low_mean`
is the mean over all pair/iteration `cycle_low.resid.mean`; association is the per-trajectory Spearman of
`cycle_low.resid.mean` vs oracle `epe_low.mean`:

| window | pairs | valid_frac | cycle_low mean (median across all per-pair/per-iteration residual means) | assoc median / mean / pos.frac | update_s / cycle_check_s / oracle_epe_check_s | manifest sha256 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| t063 (`00177.MTS`, 63-64 s) | 59 | 0.99992 | 0.04330 (0.02841) | 0.017 / -0.050 / 0.542 | 43.615 / 2.769 / 0.290 | `a9fad6a01b40ed84defa066e37e969a0d17d448491a991705e19456bbca46d3c` |
| t300 (`00176.MTS`, 300-301 s) | 59 | 1.00000 | 0.01770 (0.01720) | -0.590 / -0.603 / 0.000 | 44.466 / 2.740 / 0.295 | `cdd1c871f80e35610175d5410cf0870b5285445ac736828b7f40f5645411e7e7` |
| t090 (`00177.MTS`, 90.0-90.5 s) | 29 | 1.00000 | 0.01899 (0.01800) | 0.137 / 0.123 / 0.828 | 20.621 / 1.319 / 0.135 | `31dc05c42a076448ff899cf578e04a6763361b38b4049e0dd63d74a62e2f5f93` |
| t303 (`00176.MTS`, 303-305 s, 2.0 s) | 119 | 1.00000 | 0.01734 (0.01671) | -0.450 / -0.476 / 0.000 | 90.694 / 5.657 / 0.595 | `efc299c0f6a2d80ad5b2d6ff794bbfa2ebe918194bafb543f8c75540f2e6a0b2` |
| stream00000 (`STREAM_00000.MTS`, 25-27 s, 2.0 s) | 119 | 0.98279 | 0.20630 (0.06689) | 0.538 / 0.419 / 0.807 | 90.207 / 5.619 / 0.580 | `edace8cea096f10b4b42797df7879bbfb9830fa60b47fb753228e9fe8aeffaa2` |

The pooled per-iteration cycle residual typically falls to a minimum around iteration 6-15 and then rises
(median across each window's pair-level means at each iteration: t063 0.0407 at iter 1 -> 0.0247 at iter 9 ->
0.0320 at iter 24; t300 0.0254 -> 0.0134 at iter 6 -> 0.0219; t090 0.0312 -> 0.0161 at iter 10 -> 0.0200; t303
0.02543 at iter 1 -> minimum 0.01397 at iter 7 -> 0.01932 at iter 24; stream00000 0.50389 at iter 1 -> minimum
0.05477 at iter 15 -> 0.06038 at iter 24), so it does not track the oracle EPE monotonically. The combined
audit artifact is
`out/bench/flow_prediction_observable_audit/t063_t300_t090_t303_stream00000_cycle_iters24.json` (all five cases
`ok`; manifest sha256 `e27b0eb0617340e12154262fe5d226213470fb568b849981b5e73fd1ba82f0ce`). At the last iteration the
oracle EPE is exactly 0 by construction (the 24 paired steps reproduce the 24-iteration cold reference), so
that cross-sectional row is reported as `constant_epe` and skipped rather than treated as a data point. Wall
times: `timing_s.benchmark` 155.468 s / 158.329 s / 69.555 s / 331.855 s / 329.541 s including aggregate
shared-lock pauses of 60.045 s / 60.051 s / 24.020 s / 132.101 s / 132.095 s (6/6/3/12/12 releases at chunk 30,
pause 12 s), so they are not compute-only; `frames_load` was 0.301 s / 0.922 s / 0.220 s / 2.611 s / 1.238 s
(excluding the separately recorded 0.013 s acquire wait for t303 and 0.015 s for stream00000). The t303 run is
a **non-overlapping 303-305 s** 2.0 s window on the same local
1080i `00176.MTS` source as t300 (`source_sha256`
`78ba322cfd013054d6ff4846524511e5462ffc9cfbac5acd0200f85bfcc10e93`, its own verified 120-frame deinterlaced
stack `frames_sha256` `034ea39a6816f9df280f5a1a2c6baa0b638145c53cebb494ae9b28c014bb60de`, 59.94005994 fps, 119
pairs); t063 and t090 share the `00177.MTS` source file/hash, and the new stream00000 run is a distinct
`STREAM_00000.MTS` source (`source_sha256`
`2ed3213fe667d6adda242c5914b401468185343b15106a30ede4788aaacec636`, its own verified 120-frame deinterlaced
stack `frames_sha256` `94962e581aae39b408a049a094ca6962f429a5a919dbe987d591dc131dba4b58`, 59.94005994 fps, 119
pairs). All five share the weights sha256
`84c1f730d4620ef4dd0a2eeeef97db7a729fc4190ee05cdd0169030fc92fccc1`, code revision
`14f6f606189857d7e89515466888f8a48f63d196` (`dirty_relevant=true`), `iters` 12 and `ref_iters` 24.

**Provisional result.** On these five short, non-independent windows the cycle residual's per-trajectory
association with the oracle EPE is weak and inconsistent (median 0.017 / -0.590 / 0.137 / -0.450 / 0.538), so
it should **not** be accepted as a standalone convergence/visibility proxy and **no convergence improvement is
claimed**. This is diagnostic association evidence only: the oracle is a model-derived cold reference (not
ground truth), the windows are two one-second, one half-second and two two-second windows, only three distinct
source files are involved (the 119 pairs in the two 2.0 s windows are not independent clips), and cycle
consistency does not prove visibility. **No threshold is selected or searched, no stop rule is implemented, no true-flow
accuracy is claimed, and no production
wiring/default change or readiness is implied.**

Next E04 gate. The cycle-consistency pilot is a negative/provisional result on the same five short windows,
consistent with the photometric residual; it adds a genuine candidate-observable visibility-flavoured signal
but does not, by itself, supply the missing occlusion/content-aware confidence. Before any stopping rule or
warm-start policy is considered, the next gate is validation on **independent and longer windows** with a
stronger visibility/depth-aware signal and matched downstream stabilization quality; the cycle signal may be
one input but must not be selected as a threshold from these windows.

### E07 cache-driven relative-depth temporal audit (offline, cache-only; diagnostic only; no threshold)

The E04 follow-up gate above asked for a stronger visibility/depth-aware signal. A new isolated runner,
`bench/depth_temporal_pilot.py`, therefore audits **relative-depth temporal self-consistency** on the existing
held-out frame caches. It does **no depth inference, no media decode and no production-code change**: it reads a
decoded-frame `.npz`, a cached per-frame monocular *relative* depth `.npz` and a local RAFT weights file,
validates the frame/depth **hashes, shapes and index correspondence**, and uses the cached frames and depth
plus the local weights to rebuild the audit. The pure-NumPy audit lives in
`bench/depth_temporal_audit.relative_depth_pair_audit`; RAFT is run only to transport one sampled depth map
onto its neighbour, and the paired feature-grid depth maps are handed to that audit. The runner is not imported
by `stabilize`, is not wired into any pipeline, adds no default and selects no threshold. Its JSON report is
written under ignored `out/` and is never source-controlled.

Monocular relative depth is defined only up to a per-frame scale/shift, so each pair is compared under an
**independent robust per-pair affine fit that deliberately discards metric depth**. The reported residuals are
*structure self-consistency under RAFT transport*, not metric-depth accuracy, not a depth benchmark and not an
occlusion detector. Every support filter (finite/bounds/cycle) is diagnostic, and cycle filtering is
model-derived and may preferentially select easy regions, so it can flatter the result.

Command (reproduced from local caches):

```
python -m bench.depth_temporal_pilot \
    --frames-npz out/bench/flow_prediction_real/stream00000_t25_frames.npz \
    --depth-npz out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz \
    --weights vendor/weights/conv/raft-things.safetensors \
    --out out/bench/flow_prediction_real/stream00000_t25_depth_temporal_audit.json \
    --iters 24 --downscale 2 --flow-dtype float32 --corr-dtype float32 \
    --cycle-thresholds 0.25,0.5,1,2 --lock on
```

Reproduction. Two **distinct source files/windows from the same camera/event family** are audited; these are
**not two independent events** and the per-window pairs are serially dependent, so no pooled significance is
claimed.

- **Window A** is **30 sampled frames (indices 0,4,...,116) drawn from the 120-frame 25-27 s distinct
  `STREAM_00000.MTS` decoded cache** (29 adjacent sampled pairs; night scene).
- **Window B** is **30 sampled indices (0,4,...,116) drawn from a 60-62 s decoded window of `00176.MTS`**
  (29 adjacent sampled pairs; an outdoor ceremony scene with guests walking between the camera and a deep tree
  background; described for context only, no identity).

Both windows use **independent per-frame percentile normalization [1,99] clipped [0,1]** (no cross-frame metric
or calibration assumed). RAFT runs at MLX fp32/fp32, 2x downscale, 24 iterations, on a 68x120 feature grid. The
runner validates hashes/shape/index correspondence and then works from the cached depth and frames plus the local
RAFT weights; it performs **no depth inference and no media decode**. The Window A pilot used the **original**
depth cache `out/bench/flow_prediction_real/stream00000_t25_depth_v2base_30frames.npz` produced earlier by the
inline inference process; that cache was the **input to the Window A pilot**. The optional
`bench/depth_cache.py` builder (below) was later run on the **same frame cache** to a separate rebuilt cache
(outside the repo) and compared against the original cache; the builder **reproduced the
original maps at near machine precision** (same indices, per-map max abs diff 1.8507242e-5, mean abs diff
1.5569479e-7, p99 abs 9.5367432e-7). The rebuilt cache was **not** the input to the Window A pilot.

`same-index` compares a frame's **source depth against the target frame's depth at the same feature-grid pixel
coordinates**; `flow-warp` compares the same feature-grid coordinates but samples the target depth at the
**forward-flow-mapped coordinates** instead. All values below are **median pair median normalized residual** at the all-cycle-valid tier and the
`cycle <= 0.25` tier (mean pair median, median pair p95 and support in parentheses in the JSON report); the
0.25/0.5/1/2 values are a **measurement sweep only -- no threshold is selected**.

| Window (source file) | Tier | same-index | flow-warp | flow-warp lower |
| --- | --- | --- | --- | --- |
| A `STREAM_00000.MTS` 25-27 s | all-cycle-valid | 0.0221479 | 0.0215240 | 15/29 pairs |
| A `STREAM_00000.MTS` 25-27 s | `cycle <= 0.25` | 0.0199386 | 0.0185233 | 17/29 pairs |
| B `00176.MTS` 60-62 s | all-cycle-valid | 0.005629 | 0.006932 | **5/29 pairs** |
| B `00176.MTS` 60-62 s | `cycle <= 0.25` | 0.005608 | 0.006842 | **5/29 pairs** |

Window A (all-cycle-valid): same-index median/mean/p95 0.0221479/0.0322434/0.115408, flow-warp
0.0215240/0.0333583/0.0965191, support median 0.912868; `<= 0.25` tier same 0.0199386/0.0311942/0.0934196,
flow 0.0185233/0.0286459/0.0641095, support 0.804289.

Window B (all-cycle-valid): same-index median/mean/p95 **0.005629/0.006387/0.064359**, flow-warp
**0.006932/0.007393/0.070942**, support median **0.9727**; `<= 0.25` tier same
**0.005608/0.006327/0.060456**, flow **0.006842/0.007309/0.067943**, support **0.9593**. Flow-warp was lower
on only **5/29 pairs** in both tiers (verify against the JSON before quoting any aggregate).

**The second window reverses the mostly mixed/slight-positive night case.** The new report is **worse for
flow-warp in every aggregate tier**, which is **direct evidence against a general "flow warp improves temporal
depth consistency" assumption**. With only two windows from the same camera/event family (two distinct source
files/windows, **not** independent events/clips, and serially dependent pairs within each window) there is **not
enough evidence to support a general claim in either direction and no grounds for pooled significance**.

**Scene-dependent conclusion.** Depth temporal self-consistency under RAFT transport is **scene-dependent and
not a usable generic visibility/confidence cue by itself**. Keep it as a **measurement method only**: no
downstream stabilization/crop/jitter evaluation, no depth metric truth, no semantic or occlusion ground truth,
no production wiring/default, and no threshold or speedup are claimed. The per-pair robust affine scale/offset
deliberately discards metric depth, and the cycle filter is model-derived; this is not a claim that the model
understands content or true depth.

Runtime is a reproduction observation only (**not a speed claim**, and no runtime claim is made for depth
inference): Window B's RAFT audit took **32.62 s for the 29 pairs**. Window A's canonical report (after the
privacy scrub) records **26.670004125 s** total, **2.1007635 s** feature prep, **23.932933585 s** flow across
the 29 pairs (mean **0.825273572 s/pair**).

### E07 optional depth-cache builder (offline, cache-only)

`bench/depth_cache.py` is an **optional** companion that turns an already-decoded, already-verified frame cache
into the depth `.npz` contract the pilot consumes (keys `indices`, `depth`, `model_id`, `normalization`,
`frames_npz_sha256`; the last is the content hash of the input frame cache). It is **not required** by the pilot
and is **not wired into any pipeline**. Real inference needs **optional local heavy dependencies** -- a Python
environment with `torch`, `transformers` and `PIL` -- plus an **offline Hugging Face cache** selected through
`HF_HOME`; placeholders are used below and no absolute local path is recorded anywhere. `--local-files-only`
**refuses network**, and `--lock on` coordinates MPS with the shared `HeavyLock`. The output NPZ records **no
path**, no source file name and no machine detail.

```bash
HF_HOME=/path/to/local/hf-cache /path/to/python -m bench.depth_cache \
    --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz \
    --out out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz \
    --every 4 --model-id depth-anything/Depth-Anything-V2-Base-hf \
    --device mps --max-side 1036 --local-files-only --lock on
```

The pilot command for the second window then consumes that cache:

```bash
python -m bench.depth_temporal_pilot \
    --frames-npz out/bench/flow_prediction_real/00176_t60_frames.npz \
    --depth-npz out/bench/flow_prediction_real/00176_t60_depth_v2base_every4.npz \
    --weights vendor/weights/conv/raft-things.safetensors \
    --out out/bench/flow_prediction_real/00176_t60_depth_temporal_audit.json \
    --iters 24 --downscale 2 --flow-dtype float32 --corr-dtype float32 \
    --cycle-thresholds 0.25,0.5,1,2 --lock on
```

Cache-builder real-inference parity. Re-running the builder on the same original 30-frame `STREAM_00000` frame
cache gave **per-map max abs diff 1.8507242e-5, mean abs diff 1.5569479e-7, p99 abs 9.5367432e-7** against the
original `stream00000_t25_depth_v2base_30frames.npz` depth cache: **same indices and the same reproducible
preprocessing**, so the builder **reproduced the original maps at near machine precision**. This is a builder
parity check only; the rebuilt cache was **not** the input to the Window A pilot, which used the original cache.

Provenance hashes (Window B), all under ignored `out/` with private footage/depth arrays untracked and **no
absolute paths in the report**: frame cache
`7838a1265e3dbb020bf93cb5a8101e0c9257df7cf395f96197ed7162f67e4565`; depth cache
`51e311c719415260e9c2c3bdbddf17b2bcfc151b8cc1ddfa28d71659a35f2749`; report
`029c5aaa92274968e47c9f05164b273a6dedfba1676dd0a80da9ac59e3ba6652`.

Next E07 experiments. More **distinct** longer scenes from other sources and stronger matched
static/deforming/moving cases; a comparison of downstream stabilization crop/jitter/warp validity; and a
scene-conditioned analysis (not a pooled one) of when RAFT transport helps or hurts this depth self-consistency
statistic.

Verification. `tests/test_depth_cache.py` holds **31 cache-builder tests**; the focused depth/cache/flow tests
passed **158 tests**. The full `.claude/verify.sh` passed **672 tests in 24.06 s** (the second run; the first
run transiently failed the pre-existing `test_random_init_stages` at EPE 0.00290 against a 0.002 threshold, an
isolated rerun of that test passed at 0.00095, and then the full suite passed). CodeGraph is synced/up-to-date
and `affected` found no graph-linked test file for the new modules, so the direct tests above were run. No
changes landed in `mlxstab/`, no production default changed, and nothing was committed or pushed.

### E08 depth-rank motion-layer audit (offline, cache-only; diagnostic only; no threshold)

The E07 next-experiment gate asked for scene-conditioned analysis and a matched downstream comparison. A new
isolated runner, `bench/depth_motion_audit.py`, instead asks one narrow, falsifiable question about **flow
structure**: on a **held-out checkerboard** split of the feature grid, does stratifying a **model-predicted
flow** field by **per-frame relative-depth quantile bins** make the local displacement field more predictable
than a single pooled 2-D affine model? Like E07 it is **offline and cache-only** -- it does **no depth
inference, no media decode and no production-code change**, validates the frame/depth hashes, shapes and index
correspondence, and is **not imported by `stabilize`**, wired into any pipeline, or used to select a default or
threshold. Its JSON report is written under ignored `out/` and is never source-controlled.

Method. Each cell `p = (x, y)` carries a source **relative-depth** value and a **model-predicted** 2-D forward
displacement `(u, v) = fwd_flow[p]`. The displacement is modelled as a linear function of the cell's
**normalized** coordinates `(x / W, y / H, 1)`, with **independent OLS coefficients for `u` and `v`** (a plain
`numpy.linalg.lstsq` with `rcond=None`; no random seeding, no RANSAC). A **pooled** affine model is fitted on
every training-support cell; a **bin-specific** model is fitted on that bin's training cells only. The split is
deterministic **cell parity of `x + y`** -- even cells train, odd cells validate -- so train and held-out cells
are **interleaved** and each bin keeps a held-out subset. The pooled-held-out residual within each bin and the
bin-specific-held-out residual are reported together with their delta (pooled minus bin, so a **positive delta
means the bin model is lower/better**), evaluated on the **identical** validation cell set. Source relative
depth is divided into **4 equal-count relative-depth rank bins** over the eligible cells. **No bin is labelled
true foreground/background or near/far, there is no true flow and no ground truth**; residuals are Euclidean in
feature-grid pixels. The runner samples **30 frames -> 29 adjacent pairs in each existing 2-second window** and
audits **both directions**, giving **58 pair-direction rows per window**. Per-window summaries pool pairs
**within a single window only**; the two windows are **distinct files from the same camera/event family and are
serially dependent**, so they are **not pooled across windows** and no pooled significance is claimed.

Command (reproduced from local caches; placeholders, no absolute local path recorded):

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

**Window A** (`STREAM_00000.MTS` 25-27 s, night; report
`out/bench/flow_prediction_real/stream00000_t25_depth_motion_audit.json`, ignored): all-cycle-valid median
support **0.9327**. Across the usable pair-directions the median of the pooled-model held-out per-bin median
residual is **0.23513** vs **0.12132** for the bin-specific model, with a median per-row delta of **0.11937**;
median pooled/bin p95 **0.53985 / 0.29863**; **215/232 bins** have a lower residual with the bin model; usable
**58/58 pair-directions**. `cycle <= 0.25` tier: support **0.8095**; median pooled/bin **0.21644 / 0.10537**;
p95 **0.45624 / 0.23598**; **207/224 bins** improve.

**Window B** (`00176.MTS` 60-62 s, outdoor; report
`out/bench/flow_prediction_real/00176_t60_depth_motion_audit.json`, ignored): all-cycle-valid median support
**0.9853**. Median pooled/bin **per-bin held-out median 0.04665 / 0.03197**, median per-row delta **0.00898**;
median pooled/bin p95 **0.21806 / 0.20867**; **157/232 bins** improve; usable **58/58 pair-directions**.
`cycle <= 0.25` tier: support **0.9700**; pooled/bin median **0.04275 / 0.02963**, p95 **0.19361 / 0.18303**;
**157/232 bins** improve. Both reports carry **no absolute paths** (hashes only). Report sha256: A
`3a3b5fd9b1f7e25cd79291ad923fead090058aa9a2d54fa7f90d6ec0ecb63d5b`; B
`7f42b6c8f05a718bc8aca742481e62532d54715eb5b63e9e17636fed9d5c72d`.

Interpretation. Rank stratification makes RAFT's own local displacement field **more affine-predictable** in
these two windows, with a **much larger effect at night** and a **smaller absolute improvement outdoors**. This
is evidence of **depth-rank association with model-predicted RAFT flow structure only**; it does **not**
validate depth estimates and does **not** predict stabilization benefit. In combination with the E07 result,
**temporal self-consistency was scene-dependent while flow-field stratification is positive on both windows**;
these are **different signals and must not be conflated**.

Explicit limitations. RAFT flow is **model-derived**; there is **no metric depth, no optical-flow accuracy, no
camera-motion ground truth and no stabilization outcome**. The quantile edges are computed from **all eligible
depth values including held-out cells** (an **unsupervised distribution leakage**), and **checkerboard
train/held-out neighbours are correlated**, so the held-out residual is a structural diagnostic and **not** a
generalisation guarantee. The per-window summaries pool pairs **within a single window only**; the two windows
are distinct files from the same camera/event family and **serially dependent**, so they must **not** be pooled.
No threshold is selected or searched and no production wiring/default change is implied. Nothing here is
**proven** or **general**, and **no generic depth-confidence claim is made yet**.

Next decision gate (E09). An isolated candidate that uses **depth-rank motion layers** to choose
**background-lock vs foreground-preserve** paths on **synthetic known-truth** scenes, then a matched downstream
**crop/jitter/warp-validity ablation** on these two windows. **Do not wire production or choose a depth
threshold until it improves the user-visible tradeoff.**

Verification. The new module `bench/depth_motion_audit.py` is covered by `tests/test_depth_motion_audit.py`;
the direct new tests passed **35 tests**. The full `.claude/verify.sh` passed **707 tests in 23.91 s**. CodeGraph
was synced and `affected` found no graph-linked tests for the new module, so the direct tests were run. No
production code changed, no production default changed, and nothing was committed or pushed.

### E09 depth-rank layer synthetic known-truth gate (offline, synthetic, pure NumPy; diagnostic only; no threshold)

The E08 next decision gate asked for a synthetic known-truth scene that uses **depth-rank motion layers** to
choose **background-lock vs foreground-preserve** paths. A new isolated runner, `bench/depth_layer_synthetic.py`,
implements that gate as a **deterministic two-layer affine coordinate-model fixture**: a full-frame background
plane (layer 0) and a rectangular foreground distractor overlay (layer 1) drawn on top so it occludes the
background. Per frame `t` each layer `l` has a **known** 2x3 affine `M[l][t]`; both layers share a camera
transform (intended smooth pan/zoom composed with a deterministic per-frame jitter), and the overlay adds an
**independent deforming local motion** composed with the camera, so the two layers genuinely differ. The
observed forward flow at a reference coordinate is the visible layer's displacement; the scene is built so the
ground truth stays inside the affine family both policies fit. It is **pure NumPy float64 with fixed
seeds/permutations**: no media is decoded, no RAFT/MLX/production code is imported, no weights are read, the
heavy lock is untouched, the JSON report is deterministic and **path-free** (written atomically through
`bench.manifest`), and **no production default is changed**.

Explicit depth semantics. The scene's raw depth values are **arbitrary labels**. `RANK_ORIENTATIONS` makes the
mapping from raw value to rank **explicit and caller-chosen**: `low_raw_low_rank` (smallest raw value is rank 0)
and `low_raw_high_rank` (largest raw value is rank 0). The module **never** infers that a rank is physical near
or physical far and never claims a rank is semantic foreground/background. A policy's rank is **chosen via the
spec** (`lock_rank` / `preserve_rank` under the configured orientation), **not by semantic inference**. A rank
that carries no layer is reported `unavailable` with a machine-readable reason and is never silently clamped.

Method. Estimation uses a deterministic **checkerboard split** (even `x + y` cells train, odd `x + y` cells
held out). A **pooled** baseline fits one affine displacement field on all valid train cells; each **policy**
fits the same affine family on only its selected rank's valid train cells, fitted **from source depth, flow and
validity alone** -- the true per-pixel layer id is **never** used for fitting, and **known truth is used only to
score**. Each policy is scored against the **known layer flow** of the policy's target layer on the identical
held-out block (so the comparison is on identical cells). Both policies are **separate blocks** with their own
selected rank, fitted model and statistics; nothing is pooled between them, no threshold is selected or swept,
and **no winner is declared**. Fixed support floors (`MIN_FIT_SAMPLES = 6`, `MIN_EVAL_SAMPLES = 3`) make an
accidental single-cell win impossible.

Downstream coordinate model. Stabilization by an estimated per-frame path `E[t]` is modelled as composing
`E[t]^-1` with the known layer transform `M[l][t]`; the metrics are the **RMS of the discrete second difference
of the residual path over `t`** (a motion-smoothness proxy for downstream residual jitter, per layer and
aggregated) and the fraction of the output grid whose `E[t]^-1` maps **in-bounds** (valid inverse-warp
coverage), plus the held-out target-flow error. These are **coordinate-model metrics, not rendered-pixel or
perceptual quality**; no frame is rendered and no production stabilizer is run. The pooled baseline targets
neither layer: to compare its downstream cost with both policies on identical footing, `baseline` reports
`downstream_by_target_layer`, i.e. the **same one common pooled path counterfactually evaluated** with target
layer **0** and with target layer **1** as the *intended* target (identity intended residual for the indexed
layer, its intended in-frame motion otherwise; `baseline.targets_layer` is `None`). If any pooled fit fails the
block is reported unavailable with a machine-readable reason and no metrics, rather than substituting a partial
path.

Degenerate controls. `depth_mode` can make depth **uninformative** (both layers share one raw value, collapsing
to a single rank) or **shuffled** (a deterministic permutation that destroys the rank/layer association). Under
uninformative depth a policy's explicit rank no longer isolates a single layer, so it declines with reason
`rank_ambiguous_uninformative_depth`; under shuffled depth the rank/layer association is destroyed by
construction, so both policies are declared **unavailable** with reason `rank_layer_association_destroyed`, a
`None` downstream block and **zero** material/comparable wins. The pooled baseline is retained as a diagnostic
in every regime but supports **no policy claim**.

Commands (default and shuffled negative control):

```bash
python -m bench.depth_layer_synthetic --out out/bench/depth_layer_synthetic.json
python -m bench.depth_layer_synthetic --depth-mode shuffled --out out/bench/depth_layer_synthetic_shuffled.json
```

Default one-fixture example metrics (T=10, W=48, H=36, depths `0.2,0.8`, `low_raw_low_rank`, `ranked`, lock
rank 0 / preserve rank 1, seed 0). Both policies are available.
Held-out **target median error** is the exact-recovery **0.0 px** (below print precision) for the rank policy
against the pooled baseline's **0.420816 px** (background_lock target layer 0) and **1.630723 px**
(foreground_preserve target layer 1). Rank-policy **locked-layer residual jitter** is **0.0 px** against the
pooled **counterfactual** jitter **0.441816 px** (target 0) and **1.226369 px** (target 1). Policy **crop
coverage** is **0.816667** (target 0) and **0.838484** (target 1) against the pooled **0.821238** for each
target. Pooled fits succeeded on **10/10** frames.

Interpretation (cautious). The exact-recovery numbers are **expected from a noiseless affine coordinate-model
fixture** and primarily validate that the plumbing -- rank resolution, held-out scoring, the two independent
policy blocks, the counterfactual pooled downstream, and the degenerate refusals -- **works as specified**. They
do **not** demonstrate real-world benefit; there is **no generic claim** and **no winner**. A **false-win issue
was caught in the shuffled control** -- a rank-selected fit could score an accidental frame-wise "win" against
the pooled baseline even though the observed rank field carries no layer evidence -- and was **fixed by
suppressing policy claims** in that regime (both policies unavailable, zero wins) rather than reporting the
spurious improvement.

Explicit limitations. All results are **synthetic**: a deterministic coordinate model with known ground-truth
layer transforms; they say nothing about real footage, RAFT accuracy or metric depth. Raw depth values are
**arbitrary labels** and rank orientation is explicit, so no physical near/far or semantic foreground/background
claim is made. The checkerboard train/held-out neighbours are **spatially correlated**, so held-out error is a
structural diagnostic, not an independent-sample generalisation guarantee. Downstream jitter and coverage are
coordinate-model metrics, **not rendered-pixel or perceptual quality**. No threshold is selected or swept, no
policy is declared a winner, and no production wiring or default change is implied. Nothing here is **proven** or
**general**.

Next decision gate (E10). The fixture validates plumbing, not benefit. Before any production policy choice,
next **add noise, non-affine deformation and depth errors** to the fixture (and/or matched **real footage**) and
re-measure the **downstream crop/jitter/warp-validity** tradeoff. **Do not wire production or choose a depth
threshold until it improves the user-visible tradeoff.**

Verification. The new module `bench/depth_layer_synthetic.py` is covered by `tests/test_depth_layer_synthetic.py`;
the focused depth/layer tests passed **59 tests**. The full `.claude/verify.sh` passed **766 tests in 23.69 s**.
CodeGraph is synced/up-to-date and `affected` found **no graph-linked tests** for the new module, so the direct
tests were run. No production code changed, no production default changed, and nothing was committed or pushed.

### E10 robustness controls on the depth-layer synthetic gate (offline, synthetic, pure NumPy; diagnostic only; no threshold)

The E09 next decision gate asked to **add noise, non-affine deformation and depth errors** to the synthetic
fixture before any production policy choice. E10 extends the **same** isolated runner
`bench/depth_layer_synthetic.py` (identical scene, checkerboard split, policy blocks and counterfactual pooled
downstream; still no media decode, no RAFT/MLX/production import, no weights, heavy lock untouched, path-free
atomic JSON report). It adds **three opt-in, default-zero controls** and re-measures the same **frame-median
held-out target flow error**, **target-layer residual jitter** and **inverse-warp coverage** tradeoff. Nothing
is swept and **no winner is declared**.

Zero-default controls (defaults preserve E09 exactly; bit-for-bit on the default spec):

- **`flow_noise_px`** adds zero-mean i.i.d. Gaussian **measurement noise** (sigma, pixels) under a fixed
  named-offset seeded RNG to the **observed** flow only. The known `layer_flow` **ground truth is never
  perturbed**, so noisy observations and exact scoring stay separable. Default `0.0`.
- **`depth_error_frac`** (in `[0, 1]`) applies **deterministic flips** of that fraction of the **visible ranked**
  binary depth labels between the two raw classes (fixed seeded permutation), and **only to `depth_mode="ranked"`**
  (uninformative/shuffled regimes untouched). The **realized count/fraction is reported**. It changes the
  observed per-pixel rank the fit consumes but **never** the ground-truth layer labels, transforms or flow.
  Default `0.0`.
- **`fg_nonaffine_px`** adds a **deterministic smooth, spatially varying** (sinusoidal, frame-varying)
  foreground displacement that is folded into **both** the known `layer_flow` and the observed flow; the affine
  `layer_transform` is **kept separately** as the affine component. Default `0.0`.

All three default to zero, so the E09 fixture is unchanged; the **shuffled** and **uninformative** degenerate
refusals (`rank_layer_association_destroyed`, `rank_ambiguous_uninformative_depth`) remain in force.

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

Interpretation (cautious). The **different corruption parameters must be interpreted separately**: a higher
policy error under one knob does not transfer to another and **no aggregate score or winner** is claimed. In
this construction, **depth-label errors hurt the foreground much more than the background** (the foreground
policy error rises to about 1.47 px and 2.08 px under flips and combined), **non-affine foreground motion is
model misspecification** (the fitted affine family cannot represent it, so the foreground policy carries a
persistent ~0.65 px error even when the background is exact), and **measurement noise perturbs the flow
estimates** (small, roughly additive policy error on both layers while truth stays exact). Coverage stays near
0.82 across all cases. These are **synthetic coordinate-model corruptions, not calibrated sensor/depth/motion
models**, and they **do not establish a robust real-video benefit**. The default all-zero behaviour still
validates **plumbing, not benefit**; the shuffled/uninformative refusals remain and support no policy claim.

Explicit limitations. Unchanged from E09 and now also covering the controls: all results are **synthetic**, the
raw depth values are **arbitrary labels** (explicit rank orientation, no physical near/far, no semantic
foreground/background), checkerboard train/held-out neighbours are **spatially correlated**, and downstream
jitter/coverage are **coordinate-model metrics, not rendered-pixel or perceptual quality**. The three controls
are **synthetic corruptions of the coordinate model**, not calibrated models of real sensor noise, depth errors
or non-rigid motion. No threshold is selected or swept, no policy is declared a winner, and no production wiring
or default change is implied. Nothing here is **proven** or **general**, and **no robust real-footage benefit is
established**.

Commands (example for each case; all optional, all zero by default):

```bash
python -m bench.depth_layer_synthetic --out out/bench/depth_layer_synthetic.json                                              # ideal defaults (E09)
python -m bench.depth_layer_synthetic --flow-noise-px 0.5 --out out/bench/depth_layer_synthetic_noise.json                    # flow measurement noise
python -m bench.depth_layer_synthetic --depth-error-frac 0.25 --out out/bench/depth_layer_synthetic_flip.json                  # 25% depth rank flips (ranked only)
python -m bench.depth_layer_synthetic --fg-nonaffine-px 2.0 --out out/bench/depth_layer_synthetic_nonaffine.json              # non-affine foreground motion
python -m bench.depth_layer_synthetic --flow-noise-px 0.5 --depth-error-frac 0.25 --fg-nonaffine-px 2.0 --out out/bench/depth_layer_synthetic_combined.json  # combined
python -m bench.depth_layer_synthetic --depth-mode shuffled --out out/bench/depth_layer_synthetic_shuffled.json               # shuffled refusal (still)
python -m bench.depth_layer_synthetic --depth-mode uninformative --out out/bench/depth_layer_synthetic_uninformative.json     # uninformative refusal (still)
```

Next decision gate (E11). E10 keeps one fixture and one seed; the controls are still synthetic corruptions.
Before any production policy choice, next **run multiple seeds and scene geometries**, then check
**textured/rendered validity** and a **matched real-footage downstream crop/jitter/warp-validity** comparison.
**Do not wire production or choose a depth threshold until then.**

Verification. `bench/depth_layer_synthetic.py` (with the three controls) is covered by
`tests/test_depth_layer_synthetic.py`; the focused depth/layer tests passed **78 tests**. The full
`.claude/verify.sh` passed **785 tests in 28.11 s**. CodeGraph was synced and `affected` found **no graph-linked
tests** for the module, so the direct tests were run. No production code changed, no production default changed,
and nothing was committed or pushed.

### E11 multi-seed / foreground-geometry synthetic suite (offline, synthetic, pure NumPy; diagnostic only; no threshold)

The E10 next decision gate asked to **run multiple seeds and scene geometries** before any production policy
choice. E11 adds the isolated suite `bench/depth_layer_synthetic_suite.py`, a thin deterministic harness around
`bench.depth_layer_synthetic.run_experiment`. It adds **no new model, metric, threshold or winner**; it only
runs the existing E09/E10 experiment across a **fixed, predeclared grid** and reports per-cell numbers.

Fixed grid (frozen constants, not swept):

- **Seeds**: `0, 1, 7, 42` (a small deterministic convenience sample, not a random sample of anything).
- **Foreground geometries**, normalized fractions `(cx, cy, half_w, half_h)`: `default_center` `(0.56, 0.52,
  0.20, 0.20)` (the E09/E10 default rectangle), `offcenter_vertical` `(0.32, 0.48, 0.12, 0.35)`, and
  `upper_wide` `(0.64, 0.35, 0.30, 0.14)`.
- **Conditions**, run separately and never mixed: `ideal` (all E10 controls zero) and `stress` with the fixed
  `flow_noise_px=0.5, depth_error_frac=0.10, fg_nonaffine_px=1.0`.

That is `4 seeds x 3 geometries x 2 conditions = 24 cases`; each `(seed, geometry, condition)` is run exactly
once and its individual report kept verbatim under `cases`. Summaries are keyed by
`geometry|condition|policy` and are **kept separate by geometry, condition and policy** -- **no pooling across
geometries** (or conditions/seeds). The values are **descriptive only, not inferential**: no significance
claim, no winner, no threshold, no production wiring or default change is implied or performed.

The suite runs the base E09/E10 module, whose foreground rectangle is now an explicit
**`fg_rect`** field (`SyntheticSpec(fg_rect=...)`, normalized `(cx, cy, half_w, half_h)`); the **default
`(0.56, 0.52, 0.20, 0.20)` preserves the E09/E10 geometry**, so `default_center` is bit-for-bit the existing
fixture. The base module exposes it on the CLI as **`--fg-rect`** (`cx,cy,half_w,half_h`, default
`0.56,0.52,0.20,0.20`).

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

Interpretation (cautious). Under `ideal` the errors are at floating-point level (near machine zero), which is
**expected from the exact affine ground truth** -- it validates plumbing, not benefit. `stress` **raises errors
and jitter relative to `ideal`** in every cell, consistent with E10. The **stress magnitudes differ across
geometries**, but with **only four fixed seeds that is not evidence of any ordering** between geometries, and
no geometry, condition or policy is declared better. All counts (material-win seeds/frames, coverage) are
**descriptive only**. There is **no textured render yet**: the next gate is a **textured/rendered warp-validity**
check plus a **matched real-footage downstream crop/jitter/warp-validity** comparison. **No production policy
until then.**

Limitations. Unchanged from E09/E10 (synthetic coordinate model with known ground truth, arbitrary depth
labels, explicit rank orientation, no physical near/far or semantic foreground/background, coordinate-model
rather than rendered-pixel or perceptual metrics) plus the E11 scope caveat: the seed list and geometries are a
**small predeclared deterministic convenience sample, not a random sample**, so medians and min/max describe
those fixed realisations only; the results are **exploratory and not inferential**. No threshold is selected or
swept, no winner is declared, and no production wiring, import or default change is implied. Nothing here is
proven or general, and no robust real-footage benefit is established (no textured render exists yet).

```bash
python -m bench.depth_layer_synthetic_suite --out out/bench/depth_layer_synthetic_suite_e11.json   # E11 fixed grid (24 cases)
```

The suite composes the base module, so E09/E10 commands still apply, including the new foreground-rectangle
knob:

```bash
python -m bench.depth_layer_synthetic --fg-rect 0.32,0.48,0.12,0.35 --out out/bench/depth_layer_synthetic_offcenter.json   # explicit fg rect
```

Verification. `bench/depth_layer_synthetic_suite.py` composes `bench/depth_layer_synthetic.py`, covered by
`tests/test_depth_layer_synthetic.py`; the focused suite + base tests passed **105 tests**. The full
`.claude/verify.sh` passed **812 tests in 26.12 s**. CodeGraph was synced and `affected` found **no graph-linked
tests** for the suite module, so the direct suite tests were run. The suite report is path-free. No production
code changed, no production default changed, and no private paths or private data appear here.

### E12 rendered-validity audit (offline, synthetic, pure NumPy; diagnostic only; no threshold)

E12 closes the "no textured render yet" gap left by E09/E10/E11. The coordinate-model diagnostics measure a
*coordinate model* (a residual path, a crop-coverage fraction, a flow error) but never rasterize a pixel, so
their "valid coverage" is a geometric fraction, not a statement about which *rendered* pixels survive the warp.
`bench/depth_layer_render_audit.py` is an isolated, **pure-NumPy** module that adds a rendered-pixel
bookkeeping stage. It adds **no new model, threshold or winner**; it is not imported by `stabilize`,
`source_coords` or any production module, changes no default, and touches no renderer.

Rendering model (explicit, deterministic):

- **Textures.** `build_textures` builds a `[2, H, W, 3]` float64 RGB stack from fixed sinusoids of the
  reference coordinates (background) plus a distinct frequency set and a domain-separating offset (foreground).
  **No RNG, no assets, no media, no weights**; values are finite and clipped to `[0, 1]`.
- **Forward splat per frame.** For frame `t` and layer `l` the reference pixel `p` maps to `q = M[l][t](p) +
  d_l(p)` where `M[l][t] = seq.layer_transform[l, t]` is the known affine and `d_l = seq.layer_nonaffine[l, t]`
  is the known sampled non-affine displacement (exact zeros for the background and for zero non-affine
  amplitude). `q` is rounded to the **nearest integer pixel** (`rasterization =
  nearest_forward_splat`); destinations outside `[0, W-1] x [0, H-1]` are dropped (the source of splat holes).
- **Compositing and collision policy.** The scene composites layer 0 first, then layer 1 on top, so the explicit
  collision policy is **later layer wins**; within one layer the **last reference pixel in deterministic
  row-major order wins** (`collision_policy = later_layer_wins_then_row_major_last`). A per-pixel
  render-validity mask marks destinations that received at least one splat; an unwritten destination stays at
  zero RGB with `valid=False` (a hole). **Holes and collisions are synthetic** — they are artifacts of the
  forward splat and the collision rule, not of any stabilizer.
- **Pull-warp.** `pull_warp` maps every output pixel `o` back to `s = E[t]^-1(o)` and **bilinearly samples** the
  rendered frame there with a **conservative validity mask**: a pixel is valid only when all four contributing
  source-mask taps are valid **and** the mapped sample is strictly inside the pixel-edge box `[0, W-1] x
  [0, H-1]` (all four bilinear taps in bounds). The conservative rule makes the warped valid mask a subset of
  the source mask's interior, so an output pixel adjacent to a hole is dropped rather than mixed with an
  invalid neighbour. The `E[t]` used is the **estimated** `E[t]` from `bench.depth_layer_synthetic._fit_policy`
  (per policy, pooled/background/foreground respectively).

Policies and the pooled baseline. `background_lock` selects `spec.lock_rank` and `foreground_preserve` selects
`spec.preserve_rank`; each has its own estimated `E[t]` per frame and therefore its own rendered-validity
numbers, and **nothing is pooled between them**. The **pooled baseline** fits one affine on all valid train
cells and is reported as a single common diagnostic path. When a policy's rank is unavailable -- under
uninformative depth the rank collapses, under shuffled depth the rank/layer association is destroyed, and a rank
outside the scene is simply absent -- the policy is reported `available=False` with a machine-readable reason
and **no rendered frames are fabricated** (its `render` block is `None`); the pooled baseline remains a
diagnostic in those regimes.

Report. `bench/depth_layer_render_audit.py` records, per frame and as an aggregate, **in-bounds coverage**,
**rendered-valid fraction**, and **finite RGB fraction / variance / counts**. The report contains **no image
pixels**, is **path-free** and finite-safe (written atomically through `bench.manifest`). No media is decoded,
no RAFT/MLX/production code is imported, no weights are read, the heavy lock is untouched, and **no production
wiring, default or threshold, and no winner** is selected or changed.

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
splat **holes plus the conservative four-tap mask**; the difference is a **fixture / rasterization property**,
**not a measured real-stabilizer crop or an advantage** of any policy. These values are **one deterministic
synthetic configuration**, not a policy winner. This is **not perceptual evidence**, and the forward-splat hole
rate is an artifact of the synthetic splat, not a property of stabilization.

Commands:

```bash
python -m bench.depth_layer_render_audit --T 10 --W 48 --H 36 --out out/bench/depth_layer_render_audit_e12_default.json
python -m bench.depth_layer_render_audit --T 10 --W 48 --H 36 --flow-noise-px 0.5 --depth-error-frac 0.2 --fg-nonaffine-px 2.0 --out out/bench/depth_layer_render_audit_e12_stress.json
```

Verification. Focused test module `tests/test_depth_layer_render_audit.py` passed **28 tests**; the full
`.claude/verify.sh` passed **840 tests in 25.74 s**. CodeGraph was synced and `affected
bench/depth_layer_render_audit.py` found **no graph-linked tests**, so the direct focused tests were run. The
report is path-free; no production code changed, no production default changed, and **no private paths or
private data** appear here.

### E13 matched real-footage depth-rank path audit (implemented; two windows measured)

`bench/depth_layer_real_audit.py` compares a pooled affine flow fit with four neutral per-frame relative-depth
rank fits. It composes each fit into a path, smooths the six affine parameters component-wise with a Gaussian
(`sigma=1 pair`, nearest boundary), then measures the raw-versus-smoothed path jitter at five image points and
the inverse correction's geometric and conservative bilinear valid coverage. `bench/depth_layer_real_runner.py`
validates paired frame/depth caches, runs bidirectional RAFT, passes forward flow and cycle-map-valid support to
the pure NumPy core, and writes a path-free report. Rank bins are recomputed per pair; there is no near/far label,
no image render, and no model output enters production.

Both existing 30-frame windows were run separately with RAFT-things MLX, 24 iterations, downscale 2, four
rank bins, and `sigma=1`. The values below are each window's own estimates; they are not pooled across clips.

| window | policy | path jitter (px) | geometric in-bounds | conservative rendered-valid |
|---|---|---:|---:|---:|
| stream00000 | pooled | 1.70667 | 0.98089 | 0.96578 |
| stream00000 | rank 0 | 1.70129 | 0.97964 | 0.96590 |
| stream00000 | rank 1 | 1.66567 | 0.97942 | 0.96441 |
| stream00000 | rank 2 | 1.86733 | 0.97920 | 0.96415 |
| stream00000 | rank 3 | 1.99646 | 0.97663 | 0.96194 |
| 00176 | pooled | 0.03575 | 1.00000 | 0.97698 |
| 00176 | rank 0 | 0.03665 | 1.00000 | 0.97687 |
| 00176 | rank 1 | 0.03647 | 1.00000 | 0.97643 |
| 00176 | rank 2 | 0.03662 | 1.00000 | 0.97641 |
| 00176 | rank 3 | 0.03853 | 1.00000 | 0.97740 |

The candidate ranking changes by metric and window. Rank 1 lowers the jitter proxy in the first window but
increases it slightly in the second, while its rendered-valid coverage is lower than pooled in the second
window. Rank 0 improves coverage slightly in the first but does not lower jitter in the second. There is **no
consistent rank policy win**. These are fitted
RAFT-field affine path proxies, not ground-truth camera motion or stabilized image quality. The cycle mask means
the bidirectional sample is finite and in bounds; no cycle-residual threshold was selected. The audit is still
one smoothing setting on two windows, and the synthetic splat audit's limitations remain separate.

An optional sweep mode was added without changing the single-run report or API.
`audit_depth_layer_sweep(observations, *, bin_counts=(1, 2, 4, 8), smooth_sigmas=(0, 0.5, 1, 2))` runs the
existing per-window audit independently for every Cartesian combination and returns each full per-window report
verbatim under a stable string key (for example `bins_4__sigma_1`), with no scoring, ranking or winner
selection. `run_cached_audit` accepts optional `sweep_bins`/`sweep_sigmas` (CLI `--sweep-bins`/`--sweep-sigmas`);
when either is supplied RAFT still runs exactly once and every variant reuses the same collected observations,
and a missing grid axis falls back to the single-run `n_bins`/`smooth_sigma`. Omitting both keeps the existing
single-run report unchanged. The sweep is a per-window diagnostic enumeration only; it selects no smoothing or
bin policy and adds no production wiring.

### E14 real-window sensitivity sweep (three windows; 16 variants each)

The two E13 windows plus one additional `stream00016` window were each run once with
`bin_counts=(1,2,4,8)` and `smooth_sigmas=(0,0.5,1,2)`; each RAFT pass produced 16 NumPy audit variants without
rerunning flow. The zero-sigma regression exposed and fixed floating-point edge loss: with no smoothing the
correction is exactly identity, so jitter is 0 and both coverage measures are exactly 1.0. This is a
no-stabilization control, not a quality result.

The table reports pooled `jitter / rendered-valid coverage` and the min-to-max range over every depth-rank
policy at bin counts 2, 4 and 8. These envelopes are descriptive across different model resolutions, not a
ranking or pooled cross-window score.

| window | sigma (pairs) | pooled jitter / valid | rank-policy jitter range | rank-policy valid range |
|---|---:|---:|---:|---:|
| stream00000 | 0 | 0.00000 / 1.00000 | 0.00000–0.00000 | 1.00000–1.00000 |
| stream00000 | 0.5 | 0.60033 / 0.97641 | 0.53474–0.93775 | 0.97370–0.97742 |
| stream00000 | 1 | 1.70667 / 0.96578 | 1.55994–2.57133 | 0.95473–0.96590 |
| stream00000 | 2 | 2.36869 / 0.94545 | 2.22245–3.40642 | 0.92446–0.94514 |
| 00176 | 0 | 0.00000 / 1.00000 | 0.00000–0.00000 | 1.00000–1.00000 |
| 00176 | 0.5 | 0.01079 / 0.97691 | 0.01078–0.01687 | 0.97501–0.97728 |
| 00176 | 1 | 0.03575 / 0.97698 | 0.03574–0.05510 | 0.97487–0.97757 |
| 00176 | 2 | 0.06299 / 0.97779 | 0.06206–0.10561 | 0.97525–0.97826 |
| stream00016 | 0 | 0.00000 / 1.00000 | 0.00000–0.00000 | 1.00000–1.00000 |
| stream00016 | 0.5 | 0.00460 / 0.97833 | 0.00512–0.01546 | 0.97510–0.97861 |
| stream00016 | 1 | 0.01264 / 0.97775 | 0.01368–0.03984 | 0.97644–0.97875 |
| stream00016 | 2 | 0.01704 / 0.97667 | 0.01734–0.05841 | 0.97623–0.97923 |

The smoother choice changes the jitter/coverage pair substantially in the first window; the second window's
coverage is comparatively flat and its jitter remains much smaller. The third window also has a substantially
lower jitter scale than the first. More bins produce larger jitter in some rank policies in every window. These
results show parameter and scene sensitivity, not a robust depth-rank advantage. No combination is selected.
Full per-pair reports are local-only under ignored `out/bench/flow_prediction_real/`.

### E15 geometry-only real-frame preview

The audit report is now schema `/2` and carries each frame's source-to-stabilized affine correction in feature
coordinates. The runner conjugates that correction into original-pixel coordinates and offers optional
`--preview-out`, `--preview-bins` and `--preview-sigma` flags. The contact sheet shows the original plus pooled
and all rank-bin warps for up to six evenly sampled frames. Preview selection must resolve to an explicitly
requested variant before RAFT runs; output paths may not collide with the input caches, weights or JSON report.
The preview resamples actual RGB pixels, but it does not use depth-aware occlusion handling or compute an image
quality score.

Local contact sheets for stream00000 and stream00016 are under ignored
`out/bench/flow_prediction_real/*_depth_layer_preview.png`. The stream00016 columns are nearly identical,
matching that window's small path estimates; stream00000 shows modest frame shifts, but the policy differences
remain subtle at this contact-sheet scale. This is a geometry diagnostic for visual inspection, not a human
quality verdict or production evidence. No input cache was modified and the preview path is not written into the
JSON report.

Focused core/runner tests passed **49 tests**, including feature-to-pixel transform conversion, preview creation,
input immutability, path non-leakage and output-path collision rejection. The complete `.claude/verify.sh` passed
**889 tests in 30.89 s**. CodeGraph was synced; it found no linked tests, so the focused modules were run directly.
No production code or default changed.

### E16 fourth distinct source-window sweep

The `stream00025` window was decoded from a fourth source and depth-cached independently (30 frames sampled at
four-frame intervals). One RAFT pass at 24 iterations and downscale 2 supplied all 16 combinations of
`bin_counts=(1,2,4,8)` and `smooth_sigmas=(0,0.5,1,2)`. The table reports the pooled affine path's jitter /
rendered-valid coverage and the envelope over relative-depth rank policies at 2, 4 and 8 bins. It is one window,
not an aggregate score:

| window | sigma (pairs) | pooled jitter / valid | rank-policy jitter range | rank-policy valid range |
|---|---:|---:|---:|---:|
| stream00025 | 0 | 0.00000 / 1.00000 | 0.00000–0.00000 | 1.00000–1.00000 |
| stream00025 | 0.5 | 0.08672 / 0.97687 | 0.06279–0.45795 | 0.97517–0.98086 |
| stream00025 | 1 | 0.26622 / 0.97747 | 0.18457–1.34030 | 0.97248–0.97794 |
| stream00025 | 2 | 0.49200 / 0.97502 | 0.31211–2.15227 | 0.96336–0.97596 |

The rank-policy envelope includes both lower- and higher-jitter outcomes, while its coverage also changes with
rank resolution. This adds evidence of scene and parameter sensitivity, not a depth-rank advantage. Model-derived
flow and monocular depth remain proxies; these metrics are not rendered perceptual quality. The full path-free
report is local-only under ignored `out/bench/flow_prediction_real/`. No production code or default changed.

### E17 fifth distinct source-window sweep

The `t063_photo` source was also depth-cached independently (15 frames sampled every four frames) and evaluated
with the same RAFT settings and 16-variant grid. Its pooled jitter / rendered-valid coverage is `0 / 1.0` at
sigma 0, then `0.01483 / 0.97822`, `0.04267 / 0.97838` and `0.06339 / 0.98016` at sigma 0.5, 1 and 2. Across
2, 4 and 8 bins, depth-rank policies have jitter ranges `0.01687–0.08555`, `0.04918–0.23148` and
`0.06829–0.32510` at those nonzero sigmas, respectively. In this window every sampled rank policy has higher
jitter than pooled at each sigma, while coverage differences remain small. This is a fifth per-window
observation, not a general policy verdict: all flow/depth values are model-derived geometric proxies, with no
rendered perceptual assessment. The full report remains local-only under ignored
`out/bench/flow_prediction_real/`; no production code or default changed.

Reports are under ignored `out/bench/flow_prediction_real/`; no source clips or dense flow maps are committed.
Focused tests for the core and cache runner passed **35 tests** (including the sweep grid validation, the
one-bin rank-equals-pooled sanity check and a fake-runner check that one RAFT pass feeds every requested
variant); CodeGraph was synced and found no linked tests,
so these tests were run directly. No production code or default changed.

Next gate: repeat on more distinct source windows and evaluate rendered image quality against a human-reviewed
baseline before considering any policy or production wiring.

## Stage 4 progress: timestamp-aware multiscale baseline (research-only)

Added `bench/multiscale_motion.py`, a pure-NumPy decomposition of a `[T, D]` camera-motion signal using
normalized Gaussian low-pass kernels over explicit timestamps in seconds. It supports irregular frame intervals,
processes bounded time-row chunks, returns fine-to-coarse detail bands plus a trend, and reports per-band RMS
energy. Reconstruction is exact up to floating-point round-off; all-one detail retention returns the original
signal exactly. The synthetic pan-plus-shake fixture makes a slow pan and two faster components inspectable.

This is only the Stage 4 measurement baseline. It does not estimate a fractal/Hurst exponent, select a band,
consume real footage or change production behavior. Its 25 focused tests passed.

### E10 fixed-grid comparison with nearest-crop pairing

Added `bench/multiscale_motion_audit.py`, comparing four fixed multiscale-retention settings with a predeclared
L1 budget grid from `0.0` to `2.0` in `0.1` steps on deterministic, uniform-clock synthetic translation. Each
multiscale result is paired with the L1 result whose measured max-translation correction is nearest, with a
maximum grid tolerance of `0.05` path units for this pilot. Both methods have offline access to the same entire
clip; the audit caps the sequence at the L1 single-window limit of 600 samples. The L1 control uses fidelity
`0.1`, matching the production default; an initial zero-fidelity run exposed a free constant-offset ambiguity
on static input, so that run is not evidence.

The default run covers 180 samples at 60 fps with three seeds and five controls: `static`, `pan_no_shake`,
`pan_shake`, `pan_stochastic_shake`, and `pan_jolt`. Every method preserved the static control exactly. For
periodic pan plus shake, these per-pair ranges across the three seeds show the tradeoff; each metric is
`multiscale / paired L1`, and units are synthetic path units, not rendered pixels or perceptual scores:

| multiscale setting ↔ L1 budget | crop mismatch | pan error RMS | residual second-difference RMS |
| --- | ---: | ---: | ---: |
| `drop_all_detail` ↔ `1.7` | 0.00065–0.04741 | 0.1198–0.1288 / 0.1867–0.2597 | 0.000082–0.000085 / 0.000163–0.000216 |
| `drop_finest` ↔ `0.5` | 0.04288–0.04364 | 0.5024–0.5036 / 0.4317–0.5016 | 0.09223–0.09229 / 0.07815–0.09229 |
| `drop_finest_and_medium` ↔ `1.3` | 0.01321–0.04649 | 0.01481–0.04082 / 0.09195–0.13531 | 0.001257–0.001346 / 0.000237–0.002975 |
| `keep_all_detail` ↔ `0.0` | 0 | 0.6225 / 0.6225 | 0.75595–0.75706 / 0.75595–0.75706 |

The added non-periodic controls use the same fixed candidates. `pan_stochastic_shake` is a stationary colored
AR(1) process (explicitly not fractal and not 1/f noise); `pan_jolt` is a one-sided transient with exponential
recovery. Their matched-crop ranges are:

| control / multiscale setting ↔ L1 budget | crop mismatch | pan error RMS | residual second-difference RMS |
| --- | ---: | ---: | ---: |
| AR(1) `drop_all_detail` ↔ `1.7` | 0.00355–0.04802 | 0.1899–0.2288 / 0.2466–0.4175 | 0.000088–0.000110 / 0.000202–0.000264 |
| AR(1) `drop_finest` ↔ `0.6` | 0.01187–0.04145 | 0.3346–0.4746 / 0.2024–0.4234 | 0.0791–0.1021 / 0.00515–0.02767 |
| AR(1) `drop_finest_and_medium` ↔ `1.4` | 0.02176–0.04282 | 0.2089–0.2748 / 0.2082–0.3975 | 0.00212–0.00426 / 0.00021–0.00078 |
| jolt `drop_all_detail` ↔ `1.3` | 0.00535 | 0.2144 / 0.3895 | 0.000133 / 0.000268 |
| jolt `drop_finest` ↔ `0.6` | 0.04847 | 0.3449 / 0.5740 | 0.02810 / 0.01238 |
| jolt `drop_finest_and_medium` ↔ `0.9` | 0.00021 | 0.3023 / 0.3847 | 0.001893 / 0.000872 |

In these path proxies, `drop_all_detail` has lower pan error and residual jitter than its nearest-crop L1 pair
for each of the three shake controls, at corrections around 1.3–1.7 units. Other multiscale settings trade
pan error against jitter differently: for example, `drop_finest` has higher residual jitter than L1 in both
non-periodic controls. The pan-only control also shows candidate-dependent path error despite having no shake.
This is a promising signal for a deliberately narrow synthetic setting, not a quality result: the paths have
translation only, a short sinusoidal pan and simple periodic/AR(1)/transient shake, with no rendered images,
rotation, depth, subject motion or occlusion. It does not establish a general multiscale advantage. All 60
default crop pairs stayed within the 0.05-unit tolerance. The updated focused suite passed **64 tests** and the
full `.claude/verify.sh` passed **982 tests in 58.12 s**. Next: render these held-out-like synthetic controls
and evaluate coverage and image shape, then use distinct real windows. Consider fractal scaling only after
non-fractal multiscale behavior has a repeatable rendered benefit.

### E10.1 rendered-pixel audit of fixed multiscale vs nearest-crop L1 (implemented; pure NumPy)

E10 measured translation *paths* only. E10.1 adds the rendered-pixel stage it lacked:
`bench/multiscale_render_audit.py` builds a deterministic, procedurally textured single-layer RGB **static
scene** on a finite array with a margin, renders it into moving frames with a local pure-NumPy **bilinear**
translation-only warp along the known camera path, and compares the four fixed multiscale-retention candidates
with their **nearest measured-crop L1** counterparts (via `nearest_crop_index`, same tie rule) by rendered RGB
MAE/RMS against the known ideal frame path plus geometric valid coverage. It imports no MLX and changes no
production default; it is research-only.

Camera convention (explicit, and asserted in tests): output pixel `(y, x)` samples
`scene(y + margin + C_t.y, x + margin + C_t.x)` for the observed camera path `C_t`, the ideal view samples
`scene(y + margin + pan_t.y, x + margin + pan_t.x)`, and the candidate view backward-samples the observed frame
at `grid + P_t - C_t`. Hence `P = C` reproduces the observed frame, and with `C = pan` (so observed = ideal)
`P = pan` gives the ideal view. A source coordinate is valid only when `0 <= x <= W-1` and `0 <= y <= H-1` (all
bilinear taps in bounds); MAE/RMS are RGB-channel means over the intersection of candidate-valid and
ideal-valid pixels, so uncovered/zero pixels are never penalized. Coverage is reported over all output pixels
and as common-support coverage. Units are normalized scene intensity in `[0, 1]`.

Fixed scope and defaults (not tunable in the report): `n=60`, `fps=60`, `W=80`, `H=48`, `margin=16`, scene seed
`1729`; the same four fixed multiscale-retention candidates and the same predeclared uniform L1 budget grid
(`0.0..2.0` in `0.1` steps) as E10; five conditions (`static`, `pan_no_shake`, `pan_shake`,
`pan_stochastic_shake`, `pan_jolt`) and three seeds (`0,1,2`), i.e. 15 runs, each rendering only the four
multiscale candidates plus their (cached) paired L1 paths. The scene is a fixed procedural texture, not media.

Run:

```bash
.venv/bin/python -I -m bench.multiscale_render_audit --out /tmp/mlxstab-multiscale-render-default.json
```

Default run results across seeds 0, 1 and 2. MAE and common-valid coverage are shown as min–max across the
three independent seed records; the report itself keeps each seed and candidate pair separate. Paired L1
budgets can vary by seed, so the table lists the distinct budget(s) seen for each candidate.

| condition | multiscale setting ↔ paired L1 budget(s) | RGB MAE range (MS / L1) | common-valid fraction range (MS / L1) |
| --- | --- | ---: | ---: |
| `static` | `drop_all_detail` ↔ `0` | 0 / 0 | 1 / 1 |
| `static` | `drop_finest` ↔ `0` | 0 / 0 | 1 / 1 |
| `static` | `drop_finest_and_medium` ↔ `0` | 0 / 0 | 1 / 1 |
| `static` | `keep_all_detail` ↔ `0` | 0 / 0 | 1 / 1 |
| `pan_no_shake` | `drop_all_detail` ↔ `0.4` | 0.00875 / 0.00951 | 0.9669 / 0.9673 |
| `pan_no_shake` | `drop_finest` ↔ `0` | 0.00003 / 0 | 0.9669 / 1 |
| `pan_no_shake` | `drop_finest_and_medium` ↔ `0.1` | 0.00091 / 0.00215 | 0.9669 / 0.9677 |
| `pan_no_shake` | `keep_all_detail` ↔ `0` | 0 / 0 | 1 / 1 |
| `pan_shake` | `drop_all_detail` ↔ `1.5, 1.6` | 0.00891–0.00932 / 0.01038–0.01164 | 0.9634–0.9643 / 0.9637–0.9647 |
| `pan_shake` | `drop_finest` ↔ `0.4, 0.5` | 0.01038–0.02820 / 0.00906–0.02723 | 0.9669 / 0.9669 |
| `pan_shake` | `drop_finest_and_medium` ↔ `1.1, 1.3` | 0.00209–0.00285 / 0.00569–0.00629 | 0.9643–0.9653 / 0.9638–0.9645 |
| `pan_shake` | `keep_all_detail` ↔ `0` | 0.01617–0.03134 / 0.01617–0.03134 | 1 / 1 |
| `pan_stochastic_shake` | `drop_all_detail` ↔ `1, 1.4, 2` | 0.00771–0.00986 / 0.01003–0.01306 | 0.9635–0.9669 / 0.9632–0.9675 |
| `pan_stochastic_shake` | `drop_finest` ↔ `0.4, 0.5, 0.6` | 0.00937–0.01559 / 0.00817–0.01565 | 0.9669 / 0.9669 |
| `pan_stochastic_shake` | `drop_finest_and_medium` ↔ `0.8, 1.1, 1.2` | 0.00553–0.00996 / 0.00589–0.01210 | 0.9666–0.9669 / 0.9649–0.9669 |
| `pan_stochastic_shake` | `keep_all_detail` ↔ `0` | 0.01272–0.01873 / 0.01272–0.01873 | 1 / 1 |
| `pan_jolt` | `drop_all_detail` ↔ `1.1` | 0.02473 / 0.01482 | 0.9665 / 0.9656 |
| `pan_jolt` | `drop_finest` ↔ `0.6` | 0.02401 / 0.03456 | 0.9669 / 0.9669 |
| `pan_jolt` | `drop_finest_and_medium` ↔ `0.9` | 0.02427 / 0.02347 | 0.9669 / 0.9669 |
| `pan_jolt` | `keep_all_detail` ↔ `0` | 0.02391 / 0.02391 | 1 / 1 |

The periodic and stochastic controls show lower MAE for `drop_all_detail` than its nearest-crop L1 pair across
these three seeds; the jolt reverses that comparison, while the other settings show mixed tradeoffs. This is
not a general advantage: the texture, scene geometry, motion and sampler are all deliberately narrow, and
crop matching in path units does not force exact equality of the rendered valid coverage. Across all 60
default pairs the largest crop-proxy mismatch was **0.04983**, within the fixed **0.05** tolerance. The focused
suite passed **30 tests** and `.claude/verify.sh` passed **1,012 tests in 69.07 s**.

Limitations. Single-layer **static** scene; **translation-only** bilinear warps; a **fixed procedural texture**;
no depth, rotation, scale, perspective, rolling shutter, subject motion, occlusion or media decode; not the
production renderer and **not perceptual quality or a human preference study**; coverage is a geometric source
validity fraction, not a real stabilizer crop. The four multiscale weights and the L1 budget grid are reported
verbatim and nothing is tuned or ranked; each pair is reported separately with no pooled score or winner.
Errors here do not translate directly to real footage.

Next gate: repeat this rendered comparison on distinct real source windows with human-reviewed crops and
layer/subject-motion stress tests before considering any multiscale advantage; keep fractal scaling out of
scope until the non-fractal results are repeatable beyond this synthetic scene.

### E10.2 multi-layer rendered-pixel audit: parallax and subject-motion stress (implemented; pure NumPy)

E10.1 rendered a **single-layer, static, zero-parallax** scene, which a uniform camera warp can reproduce
exactly; depth ordering, parallax and independent subject motion were out of scope. E10.2 adds
`bench/multiscale_layered_render_audit.py`, which renders the **same** four fixed multiscale-retention camera
paths and their **nearest measured-crop L1** counterparts (same `nearest_crop_index` tie rule) through a
deterministic **three-layer** scene and compares rendered pixels. It imports no production module directly,
changes no production default and is research-only.

Scene and ground truth (explicit, asserted in tests). Three fixed layers, each a distinct deterministic
procedural texture (`make_layer_canvases`): textured **background** (full canvas), opaque rectangular
**foreground** geometry, and an opaque **subject** rectangle composited in front (so the subject/foreground
overlap exercises occlusion). Layer `l` samples its canvas at `grid + margin + depth_scales[l] * camera`
(the subject also adds its independent deterministic sinusoid `subject_motion_path`). Opacity is hard: a
bilinear-sampled layer alpha is occupied iff `alpha >= 0.5`, and the frontmost occupied layer wins **and
defines the ideal semantic region** of each pixel. The **observed** frames are the layered render under the
noisy camera `C = pan + shake`; the **ideal** frames are the layered render under the known intended `pan`
with the same true subject motion; a **candidate** camera path `P` renders as
`backward_warp(observed, P - observed)`, one uniform correction per frame. Consequences asserted in tests:
`P = C` reproduces the observed frame, and with flat depth scales (`1,1,1`) and no subject motion an integer
`P - C` equals the E10.1 translation shift. A single uniform warp therefore **cannot** represent per-layer
parallax or independent subject motion, which the per-region metrics expose rather than hide.

Fixed isolation conditions (`CONDITION_SPECS`, no tuning): `static_no_parallax` (camera `static`, depth
scales `(1,1,1)`, no subject motion: the exact-identity control), `parallax_shake` (camera `pan_shake`, depth
scales `(0.5,1.0,1.0)`, no subject motion) and `parallax_subject_jolt` (camera `pan_jolt`, depth scales
`(0.5,1.0,1.0)`, independent subject motion). Fixed scope: `n=60`, `fps=60`, `W=80`, `H=48`, `margin=20`,
three seeds `0,1,2`, so 9 runs; fixed layer texture seeds `1729,2718,3141`, foreground rect
`(0.42,0.55,0.24,0.30)`, subject rect `(0.60,0.48,0.14,0.22)`, subject period `0.8 s`, amplitude `(1.0,0.7)`
and seed offset `101`. Both families keep **matched offline whole-clip** access (`n <= L1_WINDOW`).

Reported per pair, **separately**: the E10 path metrics; whole-frame rendered RGB MAE/RMS and
candidate/ideal/common valid coverage; and the same common-support MAE/RMS/coverage **per ideal semantic
region** (background/foreground/subject) with explicit counts (`region_pixels`,
`candidate_valid_region_pixels`, `ideal_valid_region_pixels`, `common_valid_region_pixels`), so a vanishing
region cannot hide behind a pooled number. No aggregate winner, ranking, composite or score is produced.

Run:

```bash
.venv/bin/python -I -m bench.multiscale_layered_render_audit --out /tmp/mlxstab-multiscale-layered-default.json
```

Default run results across seeds `0,1,2` (MAE is normalized intensity in `[0, 1]`; ranges are min-max across
the three seed records; the report keeps every seed/pair/region record separate). The ideal semantic masks
cover 230,400 output pixels total in the static control: 154,620 background, 46,740 foreground and 29,040
subject. Paired L1 budgets can vary by seed.

| condition | multiscale setting ↔ paired L1 budget(s) | whole-frame RGB MAE range (MS / L1) | subject-region RGB MAE range (MS / L1) |
| --- | --- | ---: | ---: |
| `static_no_parallax` | all four ↔ `0` | 0 / 0 | 0 / 0 |
| `parallax_shake` | `drop_all_detail` ↔ `1.5, 1.6` | 0.01268–0.01600 / 0.01350–0.01737 | 0.01333–0.01403 / 0.01492–0.01579 |
| `parallax_shake` | `drop_finest` ↔ `0.4, 0.5` | 0.01198–0.01983 / 0.01200–0.01900 | 0.02165–0.03180 / 0.02038–0.03041 |
| `parallax_shake` | `drop_finest_and_medium` ↔ `1.1, 1.3` | 0.00851–0.01218 / 0.00991–0.01384 | 0.00696–0.00854 / 0.01036–0.01175 |
| `parallax_shake` | `keep_all_detail` ↔ `0` | 0.01435–0.02226 / same | 0.02699–0.03546 / same |
| `parallax_subject_jolt` | `drop_all_detail` ↔ `1.1` | 0.02175–0.02184 / 0.02120–0.02125 | 0.02839–0.02871 / 0.01851–0.01875 |
| `parallax_subject_jolt` | `drop_finest` ↔ `0.6` | 0.01732–0.01746 / 0.02835–0.02853 | 0.02533–0.02614 / 0.03866–0.03955 |
| `parallax_subject_jolt` | `drop_finest_and_medium` ↔ `0.9` | 0.01767–0.01783 / 0.02376–0.02383 | 0.02567–0.02647 / 0.02688–0.02724 |
| `parallax_subject_jolt` | `keep_all_detail` ↔ `0` | 0.01708–0.01720 / same | 0.02530–0.02606 / same |

In the `static_no_parallax` control every candidate reproduces the ideal layered render exactly (MAE 0,
full coverage in every region). Under `parallax_shake`, `drop_all_detail` and `drop_finest_and_medium` show
lower whole-frame MAE than their nearest-crop L1 pairs across the three seeds, while `drop_finest` trades the
whole-frame number against its L1 pair; under `parallax_subject_jolt` the reversal is mixed
(`drop_all_detail` has *higher* whole-frame MAE than its L1 pair and lower subject-region MAE, while
`drop_finest`/`drop_finest_and_medium` show lower whole-frame MAE but a comparable or higher subject-region
baseline). These are **synthetic, narrow observations, not a multiscale advantage**. The largest crop-proxy
mismatch across all 36 pairs was **0.04982**, within the fixed **0.05** tolerance; the minimum whole-frame
common-valid coverage was **0.9634** (`parallax_shake`). The background/foreground regions carry nonzero
error even for the identity rendering `keep_all_detail` in the parallax conditions, confirming that parallax
and the subject's own motion genuinely alter the known-truth render.

Limitations. Deterministic synthetic **three-layer caricature**, not real footage: layer "depth" is a fixed
scalar displacement multiplier, not a physical depth map; foreground/subject are axis-aligned rectangles with
hard thresholded opacity, not geometry, segmentation or soft alpha; warps are **translation-only** pure-NumPy
bilinear (no rotation, scale, perspective, rolling shutter or lens); the candidate is a **single uniform
camera warp per frame** that cannot represent per-layer parallax or independent subject motion; ideal
semantic masks are the ideal render's layer assignments and region metrics use the common candidate/ideal
valid support inside each region; rendered RGB MAE/RMS on a synthetic cartoon is **not perceptual quality**;
coverage is a geometric source-validity fraction, not a real stabilizer crop. The four fixed multiscale
weights and the predeclared uniform L1 budget grid are reported verbatim and nothing is tuned or ranked;
each pair and region is reported separately with no pooled score or winner. Errors here do not translate to
real footage, and no real-scene benefit is established.

Next gate: run the layered comparison on distinct real source windows with human-reviewed crops and layer/
subject-motion stress, and check whether the parallax/subject-region tradeoff is repeatable beyond this
synthetic scene, before considering any multiscale advantage. Keep fractal scaling out of scope until the
non-fractal results are repeatable.

### E10.3 blinded real-footage A/B review clips (implemented; local, isolated)

E10/E10.1/E10.2 are all synthetic. E10.3 adds a local, human-review path: `bench/multiscale_real_review.py`
turns a **successful real run directory** (`manifest.json` + `motion_r0.npz`) into **four blinded two-panel
A/B MP4s**. It validates the captured motion payload with `bench.real_composition_audit.load_motion_npz`
(`allow_pickle=False`), cross-checks it against the sibling manifest (`source.sha256`, `source.window`,
`source.decoded`, `source.preprocessing`, `outputs.motion` basename and
`experiment_identity.capture.dump_motion`), probes/decodes the manifest source window with the **recorded**
`bwdif=mode=send_field:parity=auto:deint=all` deinterlace, downscales every frame to a fixed `480x270`
panel, composes the fitted global homographies with `bench.composition_candidate.global_homography` into a
**translation path** `C_t = frame_center - Psi_t(frame_center)` (`frame_center = ((W-1)/2, (H-1)/2)`), scales
the path into panel pixels, builds the same four fixed multiscale-retention candidates and the same fixed L1
budget grid as E10, pairs each with the **nearest measured-crop** L1 (same `nearest_crop_index` tie rule) and
backward-warps one frame at a time with `bench.multiscale_render_audit.backward_warp`, writing each
comparison through `mlxstab.media.Encoder` as `A`/`B` bitmap-labelled side-by-side panels.

The sign is explicit and tested: with a pure forward translation `H` of `(tx, ty)`, `C_t = -t*(tx, ty)`, so
the path follows the scene shift and `delta = P - C` with `P = C` reproduces the observed panel. The
reduction is **translation-only**: a rotation about the frame centre gives `C_t = 0` even though the composed
map is not the identity, so rotation, scale, shear, perspective and depth are ignored. Side assignment is
randomised from a **per-run seed** that `derive_blinding_seed` derives with stdlib SHA-256 from the **fixed
base seed** `20240517` plus the run's stable provenance (`source.sha256`, `source.window.start_s`,
`source.window.duration_s`, `run_id`), so different windows generally blind independently while one run
replays bit-for-bit; the derived seed is recorded in `blinding.seed` (with the base seed in
`blinding.base_seed`), the MP4s carry only `A`/`B` and the hidden mapping lives only in the JSON report.
Manifest/provenance refusals are explicit: source-hash, window, dimension, fps,
frame-count, interlace and deinterlace mismatches (plus a non-successful run or a manifest that does not name
this motion) raise a stable `code`. The report contains path metrics, crop proxies, max correction and
geometric-support mean/min only; it writes **no image-quality score, ranking or winner** and changes no
production default.

Run (default output is `RUN_DIR/review`, under ignored `out/`):

```bash
.venv/bin/python -m bench.multiscale_real_review out/bench/independent_00177_t060_2s_v4
```

Measured on the existing 00177 capture (`motion_r0.npz`, schema `mlxstab.motion/4`): manifest
`status=success`, `source.sha256` verified **True** (247 MB source recomputed), window `start 60.0 s /
duration 2.0 s`, decoded `1920x1080 @ 59.94005994005994 fps, 120 frames`; motion `1920x1080`, `n_frames 120`,
`n_fits 119`, `stride 1`, fps `59.94005994005994`. Decode returned `1920x1080 @ 59.94`, 120 frames with the
recorded bwdif deinterlace. Four clips written, each `960x270`, h264/yuv420p, `120` frames at
`60000/1001 fps`, `0.44-0.53 MB`. Paths are in panel pixels (scale `x=0.25, y=0.25`); crop proxies and
support from the report:

| pair | multiscale ↔ paired L1 | crop proxy MS / L1 (px) | mismatch (px) | support mean/min MS | support min L1 | A side |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `drop_all_detail` ↔ `budget_0.8` | 0.8333 / 0.8000 | 0.0333 | 0.9942 / 0.9942 | 0.9942 | multiscale |
| 1 | `drop_finest` ↔ `budget_0.2` | 0.1953 / 0.2000 | 0.0047 | 0.9942 / 0.9942 | 0.9942 | multiscale |
| 2 | `drop_finest_and_medium` ↔ `budget_0.4` | 0.3754 / 0.4000 | 0.0246 | 0.9942 / 0.9942 | 0.9942 | multiscale |
| 3 | `keep_all_detail` ↔ `budget_0` | 0.0000 / 0.0000 | 0.0000 | 1.0000 / 1.0000 | 1.0000 | L1 |

All four crop-proxy mismatches are within the fixed `0.05` tolerance; the paired multiscale/L1 geometric
support is identical here (both are small translations, `min 0.9942` = border loss) and the identity
`keep_all_detail`/`budget_0` pair has full support. No quality number is computed and no side is declared
better.

Checks: `codegraph sync .` indexed the new source module and `codegraph affected bench/multiscale_real_review.py`
reported no test files (dynamic test relationship), so the relevant tests were run directly. The focused
suite `tests/test_multiscale_real_review.py` passed **19 tests**, and `.claude/verify.sh` passed **1056 tests
in 61.32 s** after the per-run blinding update. Generated clips
and `review.json` were inspected: 4 MP4s at `960x270`/120 frames/`60000/1001`, both `A`/`B` labels inked and
distinct, hidden mapping matching the recorded per-run blinding draw, and the report JSON-safe, path-free,
with no quality/winner/ranking field. Artifacts stay under ignored
`out/bench/gemini_review/t177_60_2s_multiscale_ab/` and are not committed.

A second temporal window from the same source was also rendered: `out/bench/independent_00177_t000_1s`
(`start 0.0 s`, `duration 1.0 s`, 60 frames) produced four more A/B clips under
`out/bench/gemini_review/t177_00_1s_multiscale_ab/`. Its source hash also verified. The derived assignment is
different from t060, but t000 and t060 are two windows of the **same source**, not independent scenes. Both
windows remain local review material; the blind visual verdict recorded below is the only review so far.

Review status. A blind Gemini review over the **Vertex API** has been recorded for the eight existing clips
(four per window); the results are in ignored, uncommitted
`out/bench/gemini_review/vertex_api_blind_review.json`. It returned **seven ties and one medium-confidence
preference for B**. For the single non-tie (`t177_00_1s`, `pair_03_ab.mp4`) that pair's `review.json` maps the
B panel to **L1**, so across the eight clips the set contains **one L1 preference and zero multiscale
preferences**. Both windows are drawn from the **same dark source**, and the review is a `480x270`,
translation-only reduction of the composed homography, so the dark, low-resolution, sparse-feature footage
and the translation-only model sharply constrain what could be judged. This is **weak subjective evidence,
not a product-quality result**: it establishes nothing about a multiscale advantage and selects no winner.

A second blind Gemini review over the same **Vertex API** path has now been recorded for a **second, distinct
source file** (`00176.MTS`, current SHA
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`), window start **800 s** / duration
**1 s**, rendered the same way (1920x1080 interlaced source to `480x270` translation-only review clips). The
results are in ignored, uncommitted `out/bench/gemini_review/t176_800_1s_multiscale_ab/gemini_api_review.json`.
It returned **three ties and one medium-confidence preference for B**; for the single non-tie
(`pair_02_ab.mp4`) that pair's `review.json` maps the B panel to **L1**. Gemini reported no apparent warping or
artifacts and cited the short duration and low resolution as the limits of its judgment. Combined with the
first-source `00177` API set (8 clips: 7 ties, one medium-confidence L1 preference), the recorded review now
covers **12 clips across two distinct source files: 10 ties, 2 L1 preferences and zero multiscale
preferences**. This remains **weak subjective evidence only**: the clips are short, low-resolution and
translation-only, the count is limited, and Gemini's visual claims do not establish product quality.

Limitations. The review is a **translation-only** approximation of the composed global homography (no
rotation, scale, shear, perspective or depth), a single-layer warp of real frames (no occlusion/parallax
handling), and a `480x270` visual aid, not a full-resolution render. It computes **no image-quality metric**
and selects no winner; the videos are for local human viewing only. The fixed multiscale weights and L1 grid
are reported verbatim and nothing is tuned; nearest-crop pairing is descriptive. Errors here do not
establish any multiscale advantage.

Next gate: human review of the twelve clips from both sources plus **more independent source windows** before
treating any visible difference as repeatable or claiming any multiscale advantage; keep any advantage claim
out of scope until then. No production change.

Position-bias control. The three original non-tie choices all selected screen position **B/right**, which had
happened to map to L1 in each reviewed pair. To test this confound, three clips were locally panel-swapped,
their fixed A-left/B-right labels repainted, and the hidden mapping reversed; the videos were then sent to the
same **Vertex AI `generateContent` API** reviewer, without sending the hidden mapping. The API returned two
more B/right choices and one tie. Those two B choices now map to multiscale because the pixels were reversed.
Across the five non-tie judgments, Gemini chose the right-hand position every time, irrespective of method.
This makes the method comparison **position-confounded and inconclusive**; the earlier three apparent L1
preferences are not evidence for L1. The full position-control response is in ignored
`out/bench/gemini_review/position_bias_reversal/gemini_api_review.json` (three judgments, two mapped to
multiscale and one tie). Original and reversal API reports together cover 19 judgments: 14 ties, 3 L1 and 2
multiscale mappings; the five non-ties all chose B/right.

The local, network-free reversal step is now reproducible with
`python -m bench.multiscale_position_reversal REVIEW.json --out-dir OUTPUT [--pair-index N ...]`; its copied
`review.json` remains valid input to `bench.gemini_blind_review`, with the post-swap method mapping updated.
The focused reversal, report-validation and review tests passed (**85 tests**). Generated comparison clips and
Gemini responses remain local and ignored. No production change.

## What works

- `mlxstab/warp.py`: Metal sampler (bilinear / Catmull-Rom / Lanczos-3, zeros/border, align_corners). `grid_sample` max abs error vs torch 1.2e-7 .. 2.4e-7. Forward only.
- `mlxstab/raft/`: RAFT-things in MLX + `.pth` -> safetensors converter. Final flow EPE vs PyTorch at 12 iterations: 2.2e-3 px (reference demo frames), 2.4e-3 px (synthetic); batch-2 bidirectional path: 2.2e-3 px fp32, 2.7e-3 px fp16. Gate is < 0.01 px for the fp32 path.
- `mlxstab/stabilize.py`, `python -m mlxstab`: no-crop stabilizer (similarity global fit by RANSAC with forward-backward masking and mover exclusion, residual mesh + rolling-shutter row bands, L1-optimal paths with border budget, roll levelling, one Lanczos resample, border mask video). Interlaced sources deinterlaced first (bwdif send_field).
- `.claude/verify.sh`: 138 tests pass locally, offline; weight-dependent tests skip when weights are unavailable.

### Benchmarks (RAFT-things, 12 iterations; one pair unless noted; MLX fp32 unless noted)

| size | mode | MLX | PyTorch-MPS | PyTorch-CPU |
|---|---|---|---|---|
| 960x544 | one direction, fp32 | 0.39 s | 0.42 s | 1.58 s |
| 960x544 | fwd+bwd as batch 2, fp32 | 0.67 s | 0.80 s | |
| 960x544 | fwd+bwd batch 2, fp16 | 0.54 s | 0.59 s (conv-input cast shim) | |
| 1920x1088 | one direction, fp32 | 2.17 s | 2.26 s | 11.5 s |
| 1920x1088 | fwd+bwd batch 2, fp16 | 2.30 s | did not finish in 90 s (fp32 correlation, batch 2, about 11 GB) | |

MLX is not slower than PyTorch-MPS; they are within 5 to 20% of each other. Both are compute-bound in the convolutions (see profile).

### Quality (private 1080i camcorder windows A and B, both outputs at 29.97 fps; flow 1/2 res, 20 iterations)

| clip | | stability | jitter px | wobble px |
|---|---|---|---|---|
| A | input | 0.841 | 1.43 | 0.061 |
| A | OpenCV baseline (6% crop) | 0.756 | 0.86 | 0.064 |
| A | mlxstab, fidelity 0.1 (best result) | 0.942 | 0.46 | 0.054 |
| B | input | 0.496 | 4.97 | 0.081 |
| B | OpenCV baseline | 0.692 | 3.22 | 0.091 |
| B | mlxstab, fidelity 1.0 (old default, superseded; never re-run with 0.1) | 0.594 | 2.61 | 0.063 |

A: mean border 1.7% (max 8.4%), distortion 1.0. Synthetic clip: jitter 3.43 -> 0.12 px.

## Round 1b: profiling and optimization (stopped early)

Profile of one 1080p RAFT pair, 12 iterations, before (`bench/profile.py raft`): encoders 0.84 s, correlation pyramid 0.25 s, lookup 0.06 s (5 ms per iteration), 12 GRU updates 0.93 s (78 ms each, including the upsampling-mask head every iteration), upsample 8 ms; 27 `mx.eval` calls, but removing the syncs changed nothing (shipped 2.33 s vs fully lazy 2.36 s): the time is GPU compute in convolutions, not stalls. The GRU 1x5 / 5x1 convolutions run at about 1.4 TFLOPS in both fp32 and fp16 (MLX gives no fp16 gain there; 3x3 convs do get 2x); fused z/r convolutions, shifted matmuls and conv1d were tried and were not faster.

Implemented (all in the repo, parity-tested): mask head computed once instead of per iteration, dedicated correlation-lookup kernel, `mx.compile`d update step with no per-iteration sync, encoders run once per frame and reused for both directions, forward and backward refinement as one batch-2 graph, optional fp16 network and fp16 correlation pyramid, warm start with fewer iterations, analysis at every n-th frame with path interpolation, CPU fit thread overlapping GPU flow, decode prefetch and encoder writer threads, RANSAC scoring on a subset (fit 1.4 s -> 0.2 to 0.3 s per pair at 1080p).

Full stabilizer on 60 synthetic 1080p frames, flow at 1/2 resolution, 20 iterations (`bench/profile.py pipeline`, lock waits excluded):

| configuration | frames per second (whole pipeline) |
|---|---|
| before (round 1) | 0.41 (flow 74 s, serial CPU fit 68 s of 148 s) |
| A: engineering only, exact fp32 path | 1.05 |
| B: A + fp16 network | 1.22 |
| C: B + warm start, 8 iterations | 1.97 |
| D: C + analysis at every 2nd frame (59.94p -> 29.97 analysis) | 3.57 |

Warm-start iteration study (`bench/warm_iters.py`, EPE of warm-started MLX vs the cold 12-iteration PyTorch reference on 9 demo pairs): 2 it 0.185, 3 it 0.123, 4 it 0.096, 6 it 0.063, 8 it 0.045, 12 it 0.031 px (fp16 identical to fp32 to 3 digits). 8 iterations is the smallest count under 0.05 px (mean; max 0.055).

NOT finished: the camcorder quality gate for the fast configurations. The runs (`t800:base`, then `fast` and `fast2` on both clips) were killed before producing results, so clip B still has no fidelity-0.1 baseline and nothing about C or D is validated on real footage. Therefore the shipped defaults are the exact path (fp32, no warm-start shortcut, stride 1), still including the engineering changes (bidirectional batch, mask once, new lookup kernel, RANSAC subset), which are parity-tested on RAFT but not re-measured on the camcorder clips. On the synthetic clip, stride 2 loses most of the jitter reduction (jitter 3.05 -> 1.68 instead of 2.77 -> 0.25), so stride 2 should stay opt-in; its effect on real footage is unmeasured.

## Known issues

- Reviewer (Gemini) reported floaty motion and warped tree trunks in the stabilized output; not reproduced or diagnosed here. Suspects: mesh residuals not shape-preserving, rolling-shutter row correction applied where it should not be, L1 path with constant-acceleration segments reading as floaty.
- Clip B stability score is 0.59 against the OpenCV baseline's 0.69 (old fidelity-1.0 run). The Liu stability ratio is scale-free and punishes near-static paths, so it must be read with jitter; not re-measured with the current default.
- Full-resolution 1080p flow is 0.14 fps (2 s per direction). Half-resolution flow is the practical default; batch-2 fp32 at full resolution needs about 11 GB for the correlation volumes.
- The torch fp16 baseline needs a conv-input cast shim; the reference code mixes fp32 flow with an fp16 network.
- Heavy-lock etiquette: killing a job that holds the lock leaves it stale; the scripts used per-chunk locking with a 12 s gap.

## Open design questions for round 2

- Border fill: temporal fill from neighbouring frames (cheap, no new model) vs ProPainter (non-commercial S-Lab weights; needs deformable convolution, which MLX lacks: custom Metal kernel or sampling-based rewrite).
- Mesh rigidity and shape preservation (as-similar-as-possible regularisation, protecting vertical structures).
- Per-segment shot intent: lock-off vs pan vs dolly, rather than one set of path weights.
- Speed architecture: flow resolution, analysis at 29.97, a flow cache keyed by file and time range, on-demand correlation for 4K.
- Evaluation plan: public benchmark clips (needs approval before download), blind A/B against the OpenCV stabilizer, regression thresholds on jitter, wobble, border and distortion for fixed clips.
- Packaging: Python API, chunked processing for long videos, film-engine adapter, project name and MIT copyright holder before publishing (LICENSE currently says "mlxstab contributors").

## Resume

```
cd mlx-video-stab
.claude/verify.sh                                              # green check (138 tests)
python -I bench/run_case.py --list-cases                       # Stage 0 cases
python -I bench/run_case.py --dry-run --case baseline          # resolved config + digest, no run
# public synthetic smoke (no private footage, no lock)
python -I bench/run_case.py --case baseline --source synthetic --duration 1 --synth-frames 30 --diag-every 15 --lock off
# cheap real-input smoke / 10 s window (private footage stays local; shared film lock)
python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS --start 60 --duration 1 \
    --diag-every 15 --lock on --out-dir out/bench/smoke_t060 --log out/bench/smoke_t060.log
.venv/bin/python -I bench/profile.py raft 1088 1920 12         # per-stage RAFT profile
.venv/bin/python -I bench/profile.py raftfast 1088 1920 12 float16 float16
.venv/bin/python -I bench/warm_iters.py ref && .venv/bin/python -I bench/warm_iters.py mlx
.venv/bin/python -I bench/bench_raft.py {mlx|mps|cpu} 544 960 12 3 {pair|bidir|bidir16}
python -m mlxstab in.mp4 out.mp4 --mask mask.mp4 --iters 20 --flow-downscale 2     # exact path
```
Stage 0 resume: the runner is built and reviewed; the 1 s real smoke, both 10 s baselines (t060 and t800), the
matched one- and two-second residual ablations and the one-second configuration screens (`fp16-network`,
`fp16-corr`, `combined-fp16`, `warm-8`, `stride-2`) are complete. Next are 10-second/full-window variants,
plus the tree-deformation visual reproduction and motion-field audit, before any quality conclusion. Use
`--lock on --lock-chunk 30 --lock-pause 12` for long jobs.
Setup, weights and conversion are in the README; private-footage outputs stay under this repo's
ignored `out/`, earlier film-repo outputs are under renders/v5_proto/mlxstab_*.
