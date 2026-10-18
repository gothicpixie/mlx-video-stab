# mlxstab status

### 2026-10-18 pre-registered E04 cold-12 fast-preset gate

Test whether disabling same-grid warm reuse and reducing RAFT from 20 to 12 iterations can improve complete-pipeline
throughput without degrading stabilization. The prior E04 flow-initialization benchmark used a model-derived
24-iteration reference and found cold-12 closest to it on the tested pairs; it did not measure downstream video
quality or justify production changes. This gate tests the product consequence, not another initialization/EPE
variant.

Compare historical `warm_start=true, flow_iters=20` against `warm_start=false, flow_iters=12`, holding all other
configuration, source, decode, weights, output cadence, and evaluation settings fixed. Run fresh paired arms on
four distinct source windows: `00176.MTS` t060–070, `00177.MTS` t090–100, `STREAM_00013.MTS` t032–042, and
`STREAM_00024.MTS` t109–119. Record source hashes and full manifests. Obtain independent KLT where available,
full output border metrics, and counterbalanced Gemini A/B plus panel reversals. This is a preset experiment;
neither arm changes the production default during evaluation.

Pass only if every window has zero fit fallbacks; stabilization wall time improves at least 15% and analysis
time at least 20%, with no window slower; RAFT output jitter and wobble do not increase by more than
`max(5%, 0.1 px at 1080p)`; independent KLT jitter ratio is at most 1.05 wherever available (unavailable KLT
remains unverified, never a pass); mean invalid border area worsens by at most 0.2 percentage points and
worst-frame invalid area by at most 1 point; and the counterbalanced review has no repeatable two-order
baseline preference. Static/pan controls must remain within their existing noise tolerance. If the gate passes,
document cold-12 as an optional fast preset and repeat on a held-out source before considering it for a default.
If it fails, retain warm-20 and stop this branch; attribute only the single failing window with cold-20 before
any further iteration-count experiment. All artifacts will remain under ignored `out/bench/e04_fast12_*`.

### E04 cold-12 gate result (2026-10-18; branch rejected after two windows)

Both arms were freshly run from the same current source hashes, weights and benchmark code; the source-declared
windows and all configuration except `warm_start`/`flow_iters` matched. `00176.MTS` t060–070 passed its numeric
and support gates: warm-20 versus cold-12 stabilization time was **574.70 → 388.71 s** (-32.4%) and analysis
time **521.43 → 340.04 s** (-34.8%). RAFT jitter was **0.51917 → 0.51804 px** (-0.22%); wobble was
**0.05365 → 0.05477 px** (+2.09%). KLT output/input jitter ratio changed **0.33444 → 0.33465** (+0.06%).
Mean invalid border area was **2.236% → 2.217%**, worst-frame area **5.804% → 5.835%**, and both arms had zero
stabilization fit fallbacks. Gemini called the original order a tie and the reversed order baseline-preferred;
the position-bias audit found only one non-tie judgment, so there was no repeatable two-order preference.

`00177.MTS` t090–100 failed the pre-registered gate. Warm-20 versus cold-12 stabilization time was
**516.10 → 442.03 s** (-14.4%) and analysis time **471.85 → 396.29 s** (-16.0%), both short of their required
15%/20% gains. RAFT jitter worsened **2.5028 → 3.3398 px** (+33.4%), wobble worsened **0.4783 → 0.6153 px**
(+28.6%), and worst-frame invalid border area rose **6.620% → 8.379%** (+1.76 percentage points, above the
1-point limit). The baseline had one stabilization fit fallback and the candidate had zero, but output RAFT had
five versus four fallback fits. KLT could not score either arm: the input had 37 failed track pairs and the
output path metrics were unavailable. Per the preregistered stop rule, the remaining `STREAM_00013.MTS` and
`STREAM_00024.MTS` windows were not run and no fast preset is adopted.

The required cold-20 attribution on the failing `00177.MTS` window isolated initialization from iteration
count. Cold-20 took **532.28 s** stabilization / **488.73 s** analysis versus warm-20 at **516.10 s / 471.85 s**
(+3.1% / +3.6%); cold initialization alone did not improve speed. The run produced its full motion, plan and
render, but its evaluation-only tail was cancelled after more than 40 minutes because the speed attribution was
already conclusive; its RAFT output metrics are incomplete and are not used. This gate rejects cold-12 as a
fast preset for the tested population. It does not establish that 12 iterations is always inferior or that
warm-20 is optimal; reopen only with a new, specific hypothesis and a gate that addresses the measured t090
quality regressions. Manifests and reviews are under ignored `out/bench/e04_fast12_*`.

### 2026-10-18 experimental shape-preserving residual-mesh regularization

Added a default-off `mesh_shape_strength` experiment to the per-pair residual mesh fit and exposed it through
`Config` and `--mesh-shape-strength`. It penalizes only the trace-free symmetric part of each mesh-cell
displacement gradient, so translation, local rotation, isotropic scale, the global fit and x-independent row
correction are left alone. Strength zero bypasses the sparse solve and preserves the prior fit output. Analytic
tests cover that parity, the intended null modes, localized shear reduction, support weighting, finite fallback
behavior and pipeline/CLI validation.

The first matched real-footage screen used current-source `00176.MTS` t060–061, the same 1920×1080 interlaced
decode and benchmark settings for both arms, and 30 aligned CFR output frames. On the saved 59-pair motion
capture, strength 10 reduced mean trace-free mesh strain energy by 93.1% with 0.082 px RMS and 0.620 px maximum
mesh change. The matched benchmark evaluator then measured RAFT jitter **0.04653 → 0.05733 px** (+23.2%),
wobble **0.03227 → 0.03393 px** (+5.2%), and frame motion **0.07366 → 0.10246 px** (+39.1%). Gemini 3.8 Flash
judged the original paired clip a medium-confidence tie and preferred baseline at medium confidence after
panel reversal; only one of the two judgments was non-tie, so this does not establish a stable perceptual
preference. The old saved baseline manifest had a stale source hash and was rejected; the comparison was
repeated with two fresh benchmark runs from the current source hash before review.

This strength-10 setting fails the first product gate and is not recommended or enabled by default. The option
remains experimental for a targeted future case only; do not broaden into a strength sweep without a specific
visual failure prediction and an independent quality measure. Runs, video comparisons and API judgments are
ignored under `out/bench/mesh_shape_matched_t060_1s_*`.

### 2026-10-17 opt-in low-frequency camera-path fidelity

Added an opt-in term to the existing L1 path planner that softly pulls translation and rotation/log-scale paths
towards a per-shot low-pass of the measured path. The L1 derivative penalties, per-frame border budget, window
pinning and shot-boundary resets remain active. `l1_low_freq_fidelity=0` is the default; Python `Config` and CLI
controls are available for comparisons. Mesh and rolling-shutter residual smoothing are unchanged.

On the deterministic 120-frame, 0.5 px/frame pan control, default L1 removed the sustained pan while the opt-in
setting (`fidelity=5`, `sigma=12`) retained 97.15% path gain and ended at 54.7 px of the intended 59.5 px. On the
same pan plus 2 px high-frequency jitter, the selected opt-in synthetic test suppressed high-pass velocity error
below 0.1 px/frame while preserving the pan. These are exact synthetic controls, not evidence of improvement on
natural footage. The feature remains off by default pending disjoint real-window measurements and blinded visual
review. The focused tests cover the LP target formulation, historical behavior, pan/jitter controls, cuts, windowing,
validation and CLI wiring; the full suite passed **1738 tests**.

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
`bench.gemini_position_bias_audit` now reads the recorded Vertex response JSONs and source review reports,
joins original/control judgments by run, source hash, source window and pair index, and reports screen-side and
mapped-method preferences separately. It recovers the first schema-less response through its source report,
records unmatched reviews, and performs no model/network call. Running it over the 16 original and three
position-control API judgments produced three complete counterbalanced pairs and 13 originals without
controls: all three originals chose B/right and mapped to L1; two controls again chose B/right and mapped to
multiscale, while the third was a tie. Thus two pairs explicitly track screen position while the hidden method
changes. The report confirms the position-confound reading without claiming a method winner. The focused
reversal, Gemini-review and bias-audit suites passed (**101 tests**), and the auditor reproduced these counts
from the saved Vertex reports. Generated comparison clips and Gemini responses remain local and ignored. No
production change.

## Stage 7 progress: encoder failure propagation

`mlxstab.media.Encoder` now keeps its bounded writer queue draining after an ffmpeg stdin failure, surfaces
writer/flush/close/nonzero-child-exit errors from `write`, `drain` or `close`, and makes `close` idempotent.
The `close_all` helper attempts every owned encoder before reporting the first close failure. Stabilization and
the paired/multiscale review builders use it on success; their failure cleanup closes every sibling while
preserving the original render/build exception. The local A/B position-reversal writer follows the same
primary-exception rule for its single encoder. `media.decode` now waits for ffmpeg on natural EOF, surfaces
nonzero exit status, rejects trailing partial raw frames, and closes/reaps the child after early generator close
or consumer/read errors without masking the primary exception. Successful frame bytes, cadence and rendered
pixels are unchanged. `media.prefetch` now forwards producer failures after queued frames, uses cancellation-aware
bounded puts, closes its upstream iterator on the producer thread and joins that worker when the consumer exits.
Non-positive queue capacities are refused because Python treats `Queue(0)` as unbounded.

Commits `455eb1c`, `dd41808`, `ebfef70` and `08c317b` add encoder lifecycle, owner-cleanup,
exception-precedence, decode-process and prefetch-worker coverage. The full suite passed **1317 tests** after the main decode/prefetch changes. After
adding eager queue-capacity validation and further exception-precedence cases, **77 directly affected tests**
passed locally. A real-ffmpeg smoke verified natural EOF, early-close reap and truncated-input failure.
CodeGraph is synced; `affected` found pytest links with the focused `*prefetch*` filter, and callers were run
directly. This is a reliability change only and makes no stabilization-quality claim.

The current-source 800–810 s, 10 s baseline capture remains queued at `stage=source`, `status=running` while
the anniversary-video render owns the shared heavy lock. The runner is alive and will proceed when that lock is
released; its output must be completed before a new independent-window comparison can be reviewed.

Atomic-output hardening now stages each `Encoder` render in a unique hidden same-directory `.mp4` and publishes
with `os.replace` only after a clean writer/ffmpeg exit. Failed construction, encoding, child exit or publication
removes the partial while preserving an existing destination. New-file permissions follow the OS umask without
reading or mutating process-global `umask`; replacements preserve the old destination mode. Symlinks remain
unchanged on failure and are replaced by a regular output on success (the referent is left untouched). Tests
cover publish/failure cleanup, permissions, symlink behavior and stdin-close failure. The full suite passed
**1336 tests** with the close-failure regression; **31 focused encoder/cadence tests** passed with it included.
A real ffmpeg/ffprobe smoke verified a three-frame output, umask-derived new-file mode,
existing-mode preservation and the documented symlink behavior. CodeGraph was re-synced after the change; with
the explicit `tests/**` filter it found 27 affected test files.

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

- Gemini's earlier standalone review reported floaty motion and warped tree trunks; its first paired review disagreed. A later position-reversed review of the current baseline on `00176.MTS` (60–70 s) preferred raw in both panel orders (medium confidence), mapping A→raw in the original and B→raw in the reversal. It cited a lower-edge black/missing-border wedge near 3 s. The encoded validity mask independently marks about 7.2% of the bottom four rows invalid at 2.94 s (about 1.2% invalid over that full frame), which is consistent with a small border wedge. This is one clip/model/window and a low-resolution visual proxy, not a general quality verdict. Reports and clips remain in the ignored `out/` tree. Follow-up: inspect border fill and correction shape on this window before changing mesh or rolling-shutter policy; position reversal is now part of the Gemini review workflow.
- Offline border-support attribution reproduced from the saved t060 10 s baseline (`out/bench/current_source_t060_10s_baseline/plan_r0.npz` + `mask_r0.mp4`) at output frame 88 (plan frame 176; 2.94 s at 29.97 p, the frame behind the older lower-edge finding above). The production-order map (global homography, then mesh sampled at the source coordinate, then rows sampled after the mesh shift) was rebuilt offline in NumPy and ablated. Invalid fractions over all output pixels / over the bottom four-row strip (4x1920 px): global-only 25,198 (1.2152%) / 88 (1.1458%); global+mesh 23,913 (1.1532%) / 734 (9.5573%); global+rows 25,439 (1.2268%) / 80 (1.0417%); full global+mesh+rows 23,873 (1.1513%) / 556 (7.2396%). The mesh correction is the sole source of the bottom-edge wedge (net +646 bottom-four-row invalid pixels: 668 added, 22 removed); the row correction removes 178 mesh-induced bottom-four-row invalid pixels and adds none, so it neither creates nor worsens the wedge. The full map's center-out-of-bounds mask is pixel-identical (0 of 2,073,600 differing) to the decoded `mask_r0.mp4` frame 88 and to the production MLX `stabilize.render_from_src` mask, so the attribution is anchored to the encoded artifact. Limitations: one window and one output frame, a geometry/support diagnostic rather than a perceptual or quality metric, and no production default, `source_coords` or plan is changed.
- A read-only CPU/NumPy border-backoff screen reconstructed the one-second current-source t800 plan from its captured motion/config, then exactly matched the encoded `mask_r0.mp4` on all 30 output frames (0 differing pixels across 62,208,000). The rerun with artifact-cadence checks also passed exact verification. Baseline support was 0.6117% invalid over all pixels on average and 5.9340% over the bottom four rows; the worst bottom strip was 67.6563% invalid on the last output frame. For each sampled-border limit 0.005, 0.01, 0.02 and 0.04, the existing clip-wide backoff selected scale 0.023; at that scale all measured output masks had zero invalid pixels, but the sampled worst cliff is abrupt (scale 0.024: 9.4118% vs scale 0.023: 0%). Because the scale jointly shrinks the global, mesh and row corrections toward identity, this is evidence that a single global backoff is too blunt for the localized wedge, not evidence of improved stabilization. This screen covers one second and encoded output frames only, uses a coarse probe grid, and rendered no images; it does not change defaults. The reusable audit has 44 synthetic tests. Next decision: compare local/region-aware support handling on a rendered held-out window before considering any production change; the t800 10 s baseline remains queued behind the shared render lock.
- A separate one-second t800 component-ablation screen reconstructed the same plan and again matched `mask_r0.mp4` exactly across all 30 outputs. Bottom-four-row invalid pixels were 10,932 for global-only, 16,616 for global+mesh, 12,855 for global+rows, and 13,672 for the full warp. Relative to global-only, mesh alone added 5,684 invalid bottom pixels and rows alone added 1,923; together they added 2,740, with a large non-additive interaction of -4,867. Conditional on the other component, mesh added 817 invalid pixels while rows removed 2,944, so the two residuals partly cancel at the edge. Across the whole frame, both residual components together reduced invalid pixels by 9,801 relative to global-only. This differs from the t060 single-frame attribution and shows the bottom-edge effect depends on the interaction and time window; no single residual can be named the universal cause. The result is still one window's geometry/support proxy: no source footage was decoded, images were not rendered, and there is no perceptual quality or foldover assessment. Before changing policy, extend component attribution to held-out windows and compare rendered candidates.
- A mesh-only scale sweep (`mesh_corr := s * mesh_corr`, with global and row correction fixed) on the same t800 second was also exact against the encoded mask. Over 30 outputs, bottom-four-row invalid pixels were 12,855 at `s=0`, reached a coarse-grid minimum of 12,678 (5.503%) at `s=0.5`, then rose to 13,672 (5.934%) at `s=1`; the response was non-monotonic. Even at `s=0.5`, the worst frame still had about 67.6% of its bottom four rows invalid. Full-frame invalid pixels decreased from 386,557 at `s=0` to 380,500 at `s=1`, so the full mesh improves overall coverage while slightly worsening the pooled bottom strip. This is a one-window support tradeoff, not a reason to pick a global mesh scale; no images were rendered and no scale or default changed. Next, compare component-sensitive support policies on held-out windows and render only candidates that preserve useful motion without edge defects.
- A second component-ablation check on the current-source t060 2-second cache (60–62 s) matched the decoded mask exactly on all 60 output frames. Bottom-four-row invalid pixels were 164,023 for global-only, 193,019 for global+mesh, 183,244 for global+rows, and 205,816 for the full warp (35.60%, 41.89%, 39.77%, 44.66%). Both residuals worsen the bottom strip in isolation in this window, and together add 41,793 invalid pixels over global-only with a -6,424-pixel interaction. Across the full image the combined warp adds only 7,126 invalid pixels over global-only (1,944,741 vs 1,937,615), while mesh alone improves the full-image count by 2,374. This contrasts with t800, where rows recover bottom support conditional on mesh; the attribution is strongly time-window-dependent and cannot yet select a universal component policy. It remains a one-window geometry/coverage proxy with no rendering or perceptual assessment.
- The reusable border-component audit now streams reconstructed/decoded masks in bounded chunks, verifies exact frame counts (including refusal of early EOF and extra masks), and closes iterators on all exits; 100 focused tests pass. Applied to the successful t060 10 s cache (60–70 s; 600 plan frames, 300 encoded outputs), it matched the encoded validity mask exactly across all 300 outputs. Over the full window, invalid full-image pixels were 13,914,436 for full G+mesh+rows, 13,062,272 for global-only, 13,953,696 for global+mesh, and 13,043,113 for global+rows. Bottom-four-row invalid pixels were 912,634 (39.61%), 924,448 (40.12%), 899,794 (39.05%), and 936,208 (40.63%), respectively. Thus the combined residuals reduce bottom-strip invalid pixels by 11,814 but add 852,164 invalid pixels over the whole image; the bottom result is time-window-dependent and the full-frame count is not a perceptual quality metric. Per-frame, all four bottom rows are invalid on 44 of 300 full-warp outputs, clustered at output frames 39–59, 154–162, 164–170, 175–177 and 234–237; the worst full-frame invalid fraction is 5.78%. The streaming output reproduces the prior t800 1 s and t060 2 s component reports exactly; the source footage is not decoded and no production policy changed. The 10 s report is retained under ignored `out/bench/`.
- The border-backoff screen now decodes, verifies and measures baseline/candidate masks in bounded chunks. The streamed t800 1 s report exactly matches its saved materialized verification, baseline and candidate blocks. On t060 10 s, exact mask verification passed across all 300 outputs; the baseline sampled-worst probe score was 57.65% (full-image invalid 2.2368%, bottom-four-row invalid 39.61%). All four tested limits (0.005, 0.01, 0.02, 0.04) selected scale 0.004 and yielded zero invalid pixels in the reconstructed output mask. The scale jointly retains only 0.4% of the global, mesh and row corrections, so this is effectively an identity-near coverage result, not evidence of useful stabilization or a production policy. Reports remain in ignored `out/`.
- The research-only temporal border-fill pilot was run on the same 60–70 s window at radii 1, 3 and 15. They filled 8.7%, 24.0% and 73.4% of the 13,914,436 invalid pixels respectively (0.19%, 0.54% and 1.64% of all output pixels). The first radius-3 Gemini 3.8 Flash via Vertex API comparison used raw versus filled output and was confounded by stabilization. Follow-up blinded comparisons isolate the postprocess by pairing the baseline stabilized output against each filled candidate at exact output-frame indices. Gemini called the radius-1, radius-3 and radius-15 original/reversed clips ties at medium confidence, reporting no discernible motion or image-integrity difference; the position-bias audit found two complete pairs and no non-tie judgments. This is one short, low-resolution, model-judged window and does not establish that temporal fill helps or hurts generally. Candidates and API reports remain in ignored `out/`; the pilot stays opt-in and diagnostic.
- Rendered composition candidates were evaluated on the same 60–70 s window from `00176.MTS`, using the captured baseline plan/motion and production `render_from_src`. The renderer uses the research-only padded dense-map approximation documented above; on 315 sparse points over five source frames, its maximum deviation from direct `candidate_source_coords` was 0.003 px for all three modes. Gemini 3.8 Flash via the Vertex API reviewed blind, position-reversed comparisons against the existing stabilized baseline (640x360 per panel, 10 s): exact composition was judged worse in both orders (medium confidence), citing severe tree/scene warping and black border gaps; source-chart high-pass was a tie in both orders (medium confidence), with minor lower-edge artifacts reported in both panels; reference-chart high-pass was judged worse in both orders (medium confidence), citing lower-edge black gaps. The position-bias audit confirmed method agreement for the two non-tie pairs and no non-tie judgment for source high-pass. These are three paired judgments from one short inferred window and one model; they do not establish a general ranking. On the checked probes, the near-zero exact-map error indicates the visible distortion is not explained by dense-map interpolation. Reports and clips remain in ignored `out/`. Follow-up: examine border support and the source-chart high-pass candidate on longer/held-out windows before changing the production path; no mode is selected.
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

## Completed t800–810 s baseline and blinded raw/stabilized review (2026-10-15)

Run. The ten-second baseline at the inferred 800–810 s window of local `00176.MTS` (SHA-256
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`) completed with run_case `status=success`
on the repository stabilization code, with no relevant dirty files. It used **600 interlaced analysis frames**
and encoded **300 output frames at 29.97p**; the evaluation succeeded. Timing: analysis 747.4 s, plan 25.1 s,
render 253.6 s, evaluation 548.0 s, end-to-end stabilization 1026.2 s. Border: mean fraction **1.56%**, max
**4.11%**; crop ratio **98.44%**. The run report contains **no quality metric**.

Blinded visual comparison. A synchronized, blinded raw-versus-stabilized A/B and a swapped-position control
were built (one short **480x270** pair). **Gemini 3.8 Flash judged raw preferable in both orders at medium
confidence.** The position-reversal audit found **one complete original/reversal pair with method agreement,
the compared methods on opposite physical sides, and zero side-following**. Gemini cited **black boundary
padding on the stabilized panel in both orders**, while its wording was **inconsistent about which panel looked
smoother in the reversed clip**; the mapped raw preference is recorded together with that visible-evidence
inconsistency.

Reading. The mapped preference **weakly favours raw for this one clip**: Gemini cited **black boundary padding**,
but its steadiness evidence was **inconsistent**, so this **does not isolate whether stabilization reduced
shake**; it is **not proof that raw is generally better**. It is a single short, low-resolution,
model-derived pair, so it is **not a general quality conclusion and no production/default change follows**. A
real border-component audit was begun but **interrupted before completion because of excessive cost**; it is
**not complete and no result is claimed from it**.

Follow-up temporal-fill candidate (diagnostic only). The existing `temporal_border_fill_pilot` at radius 3 was
also run on this same t800 baseline; it changes no camera path or default. Its report totals are invalid
9,675,939 pixels, filled 2,545,944 (**26.31%** of invalid pixels, **0.409%** of all output pixels) and
unresolved 7,129,995. The candidate was encoded at **CRF22** (the default encoding exceeded the 15 MiB Gemini
inline cap) as 300 frames of 1920x1080 at 29.97p (7,827,897 bytes). Position-reversed baseline-versus-fill
clips submitted to Gemini 3.8 Flash returned a **tie at medium confidence in both orders**, and the
position-reversal audit records **one complete paired tie**. Gemini reported no perceptible difference in
stability or framing and noted equal compression artifacts. This is **inconclusive for fine shake and gives no
evidence of a visible benefit**, so the fill is **not adopted from this clip**; because the candidate used
CRF22 and the A/B is low-resolution, **compression and review limits apply**.

## Matched t060–070 s baseline/warm-8 screen and blinded review (2026-10-16)

Matched 10 s screen. Two `run_case` manifests completed `status=success` on the same 60–70 s window of local
`00176.MTS`, both using source SHA
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`, with matched input metrics. The only
resolved configuration difference is `warm_iters=8` in the warm-8 case; the baseline is otherwise unchanged.
Both runs recorded **57 shared-lock releases** and about **672.3 s** of reported pause total.

Timing. Stabilization wall was **1030.965 s** for baseline versus **777.374 s** for warm-8 (**24.6% less**).
Almost all of the saving is analysis: **742.367 s** versus **490.523 s** (**33.9% less**). Plan and render were
essentially unchanged (**32.49 / 256.06 s** versus **31.13 / 255.68 s**), and evaluation was **536.83 s**
versus **555.60 s**.

Quality metrics. Output jitter was **0.5191697 px** (baseline) versus **0.5226940 px** (warm-8, **+0.68%**);
output wobble was **0.0536469 px** versus **0.0549162 px** (**+2.37%**); Liu stability was **0.9255194**
versus **0.9220134**. Both runs had **zero fit fallbacks**, and the input metrics matched.

Blinded review. A baseline-versus-warm-8 blinded, position-reversed comparison was built (one original and one
reversed pair). **Gemini 3.8 Flash returned a medium-confidence tie in both orders**, reporting no clearly
visible stability difference. The position-reversal audit records **one complete paired tie**. Both reviews are
limited by low resolution and the short ten-second duration.

Caveat. The baseline manifest's source revision is **`1fb994d`** and the warm-8 manifest's is **`b93382f`**, so
the two runs are **not byte-identical code**. Inspection found the intervening `mlxstab/stabilize.py` and
`mlxstab/media.py` changes are encoder/error-cleanup/reliability changes rather than RAFT, motion-fitting or
warp-equation changes, but strict one-variable attribution remains imperfect.

Reading. The speed gain is **promising on this clip only**: warm-8 cut stabilization wall time by **24.6%**,
almost all of it from analysis, and that is the encouraging result. The measured metrics did not match,
however: warm-8 had small but real increases in jitter (**+0.68%**, 0.5191697 to 0.5226940 px) and wobble
(**+2.37%**, 0.0536469 to 0.0549162 px) and a slightly **lower** Liu stability (0.9220134 versus 0.9255194).
The blinded Gemini 3.8 Flash review could not distinguish the two methods (a medium-confidence tie in both
orders), but that judgment is at **limited resolution and a ten-second duration**, so it does not resolve the
small measured regressions. A promising speed result with small regressions and a perceptual review that
cannot tell the methods apart is **insufficient to change a default**. Reports and clips remain in the
ignored `out/` tree.

## Opt-in output crop zoom and t0613 development screen (2026-10-16)

Implementation. Commit `964ae44` adds an opt-in output-side center zoom. `Config.crop_zoom` defaults to `1.0`
(validated finite and `>= 1`), `python -m mlxstab` gains `--crop-zoom`, and the zoom is composed on the right
of the global map (`G @ Z`), so `x -> c + (x - c)/crop_zoom` acts on output coordinates with
`c = (W/2 - .5, H/2 - .5)` before the output-to-source homography; `1.0` is the exact identity. The mesh and
row residual arrays are left untouched and re-evaluated under the new `G`. The full suite passed **1499
tests** and the focused crop-zoom suite passed **98 tests**; the branch was pushed. This is an opt-in flag,
not a default change.

Development screen. Three `run_case` manifests completed `status=success` on the same 61.3 s, 1.0 s window of
local `00176.MTS` (SHA-256 `e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`, the window
behind the earlier lower-edge wedge), each with **30 progressive outputs at 1920x1080 and `30000/1001`** and
code revision `964ae4411b1a1349fcfe5fb5cdde7bf28639a08c`. The runs are `out/bench/cropzoom_t0613_baseline`
(`crop_zoom=1`), `out/bench/cropzoom_t0613_z110` (`1.10`) and `out/bench/cropzoom_t0613_z115` (`1.15`); the
zoomed runs set the value through `--set crop_zoom=...`.

Mask/support diagnostics. The current-render `warp_diag` full-support Lanczos-3 fraction was mean
**0.9760902456**, min **0.9388276427** at `crop_zoom=1`; both `1.10` and `1.15` measured mean/min **1.0**. All
three runs had **zero fit fallback frames**.

Output metrics. Baseline jitter/wobble were **0.0613283537 / 0.0305879680 px**. At `1.10` they were
**0.0676948740 / 0.0328344449 px** (**+10.4% / +7.3%**; absolute **+0.006367 / +0.002246 px**), and at `1.15`
**0.0790517982 / 0.0349467526 px** (**+28.9% / +14.3%**). On this window the zoom removes the border exposure
at a measured jitter/wobble cost.

Blinded review. Gemini via the Vertex API reviewed position-reversed, blinded `baseline_vs_cropzoom` pairs for
both zoom ratios. Each review followed the screen side (the mapped choice flipped between the original and the
reversal), and the position-bias audit records `side_following_pairs=1` and `method_agreement_pairs=0` for each,
so there is **no usable visual preference** from this review.

Reading. This is a **single targeted development interval, not a held-out window and not a general quality
result**. `crop_zoom` stays **opt-in with default `1.0`**, and no preset or default is selected. `1.10` was the
lowest tested ratio that achieved full support on this window, but it is **not claimed to be mathematically
minimal**. Artifacts stay in the ignored `out/bench/` tree.

Distinct-source validation. `1.10` was held fixed with no tuning on two distinct source files, each a 1-second
matched baseline/candidate pair with identical configs except `crop_zoom`, **30 outputs at 1920x1080 and
`30000/1001`**, all runs `success` with **zero fit fallback frames**. `STREAM_00016.MTS` (SHA-256
`49e0aab7021bd254f46f0226ac020d4c586c29825b47102acdfbb09657c127dd`, window 60-61 s): full-support Lanczos-3
mean/min went from **0.9917992139 / 0.9900868056** at `1.0` to **1.0 / 1.0** at `1.10`; output jitter rose
**0.1329892225 -> 0.1713887097 px** (**+28.9%**, **+0.038399 px**) and wobble fell **0.0331164615 ->
0.0303146974 px** (**-8.5%**). `STREAM_00025.MTS` (SHA-256
`d6744f20554b80cf93ac7e7a1310ce16bc1d567a7409eaf96cc6d272b2b97ca0`, window 25-26 s): support went from
**0.9856643760 / 0.9712803819** to **1.0 / 1.0**; jitter rose **0.5568961500 -> 0.6741181918 px** (**+21.0%**,
**+0.117222 px**, beyond the PLAN absolute 0.1 px gate) and wobble **0.2313849032 -> 0.2414798839 px**
(**+4.4%**).

Blinded review. Gemini via the Vertex API reviewed the baseline/candidate pairs with position reversals on
both source files; all four judgments were **medium-confidence ties**, with zero side-following and no method
agreement (the source short/low-light limitations).

Artifacts. The four runs are `out/bench/cropzoom_holdout_s16_t060_baseline`,
`out/bench/cropzoom_holdout_s16_t060_z110`, `out/bench/cropzoom_holdout_s25_t025_baseline` and
`out/bench/cropzoom_holdout_s25_t025_z110`; reports and blinded clips remain ignored.

Conclusion. `crop_zoom=1.10` generalizes to full-support coverage but not to matched stability: fixed `1.10`
fails the jitter noninferiority gate on `STREAM_00025` and is **not a general preset**. Keep the default `1.0`
and select no ratio. `1.10` remains **opt-in for user-selected framing only**.

Next step. Investigate a per-clip minimal crop policy with a strict crop cap and a measured jitter tradeoff, or
drop it if no settings meet the gate; do not tune on these reserved windows.

## Independent sparse-track evaluation cross-check (2026-10-16)

Implementation. The evaluator now has an **opt-in** sparse KLT plus robust-similarity estimator in
`mlxstab/tracks.py`. `--eval-tracks` wires it into the benchmark's same decode pass; `bench.evaluate --tracks`
adds it to direct input/output evaluation; `bench.track_agreement_audit` compares it with a captured RAFT motion
payload without another flow inference. This addresses the plan's requirement to judge at least some results
with independent tracks. It changes no stabilization fit or default. Source/destination no-source masks use the
existing conservative evaluator semantics, out-of-frame LK endpoints are dropped, and KLT path metrics are
unavailable if any pair failed. We do not turn a failed pair into identity or collapse gaps into a false
continuous path. The offline audit verifies a frame cache's content digest and can match its source hash, time
window, frame count, dimensions, cadence and motion file to a successful run manifest.

Verification. `tests/test_tracks.py`, `tests/test_track_agreement_audit.py` and `tests/test_bench_runner.py`
cover transform direction, known translation, outliers, blank/partial failures, mask handling, bounds, cache
and manifest mismatch, single-pass wiring and opt-in manifest identity. The focused set passed **84 tests**;
`.claude/verify.sh` passed the full **1,531-test** suite. CodeGraph was synced after the source edits; it found
no linked tests, so the focused tests were run directly as well.

Real cross-check. On local `00176.MTS` (source SHA-256
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`), window 60-62 s, the independently
decoded 120-frame 1920x1080/59.94p cache was verified against the successful run manifest and captured
`motion_r0.npz`. KLT fit all 119 pairs (mean 499.5 tracks and 0.918 inlier fraction). Its input path jitter was
**0.41140 px** versus RAFT's **0.40619 px**; five-probe global-transform displacement EPE was **0.1343 px
mean / 0.2890 p95 / 0.4478 max**. This is cross-estimator agreement on one short inferred window, not ground
truth.

Input/output check. The same source window, baseline stabilized output and decoded no-source mask were evaluated
at `30000/1001` by `bench.evaluate --tracks`: RAFT jitter moved **0.86233 -> 0.04843 px** and KLT jitter
moved **0.86797 -> 0.07187 px**. Both estimators had support on all 59 evaluation pairs, no failed fits, and
agreed that output jitter improved; their output/input ratios were **0.0562** (RAFT) and **0.0828** (KLT).
The estimator agreement flag is true, while the ratio relative gap is 0.322. This supports the direction of the
metric result on this single two-second window only; it says nothing about other scenes, intentional-motion
preservation, local mesh/row distortion, visual preference, or superiority to PyTorch. Reports and pixel caches
remain local under ignored `out/bench/`.

### Cross-source KLT abstention checks and Gemini review (2026-10-16)

On distinct local source `00177.MTS` (SHA-256
`8c5d98d15448ad0557537ba9d02a28fc426185b4e32bdf1be3eb4964fdbea714`), window 60-62 s, the direct evaluator
reported RAFT jitter **0.22596 -> 0.16241 px** and KLT jitter **2.23561 -> 2.16868 px** (KLT ratio **0.970**,
classified unchanged at the 5% boundary). KLT covered all 59 pairs with no failed fits, but its absolute jitter
was about ten times RAFT's, so this is an estimator disagreement, not evidence that either is correct. RAFT had
one failed fit on both input and output, making the combined agreement flag **unreliable**. Gemini 3.8 Flash via
the Vertex API reviewed the raw/stabilized side-by-side and its panel reversal; both medium-confidence judgments
were **ties**, with no side-following. The reviewer described almost identical movement, but both clips are only
480x270 panels from an extremely dark two-second scene. This visual review does not settle the metric conflict.

On `STREAM_00029.MTS` (SHA-256
`2a8e01a69f13178ce12b3537bd48c750815c48879b4d87a7411dcc048936369b`), the four-second one-second baseline
window had KLT jitter **0.07303 -> 0.07149 px** (ratio **0.979**, unchanged), no KLT failed pairs and about 480
tracks per pair. RAFT input jitter was **0.04948 px**, below the predeclared 0.05 px ratio floor, so its ratio
and estimator agreement remain unavailable. This is a near-static clip where a small ratio is not interpretable.

Warm-iteration follow-up. A same-code, same-source 90-100 s `00177.MTS` baseline / `warm_iters=8` pair had
already measured RAFT jitter **2.50279 -> 2.15134 px** but wobble **0.47834 -> 0.69012 px**; end-to-end
stabilization time fell **967.9 -> 746.1 s** (22.9%). The KLT cross-check could not score either run: input
tracking failed on 37/299 pairs (12.4%); output failed on 43/299 (14.4%) for baseline and 51/299 (17.1%) for
warm-8. Path metrics were therefore unavailable rather than silently bridging gaps. This does not pass the
warm-8 quality gate; it remains a speed candidate with a wobble regression and no independent KLT quality
measurement on this dark scene. Artifacts remain under ignored `out/bench/`.

The longer-run KLT check also surfaced an empty-inlier edge case: RANSAC can lose all support during refinement,
and the caller previously passed its empty mask into `geometry.fit_model`, producing NaN runtime warnings before
falling back. `pair_similarity` now returns the explicit fallback before that empty fit; a regression test pins
the behavior. The 00177 ten-second KLT pass was rerun with `RuntimeWarning` promoted to error and completed
without warnings, while still reporting path metrics unavailable for the documented failed-pair reason.

## Projective residual-composition oracle guard (2026-10-16)

The existing E02 synthetic audit already showed that the production fixed-grid residual sum differs from exact
composition under rotation and two-layer parallax. It had not isolated a projective global with a nonzero
residual, and its `transported_field` helper applied an affine-only displacement recursion without refusing
projective homographies. Added a projective-plus-residual fixture and made that helper reject non-affine globals;
the report marks the transported comparison as not applicable instead of emitting a misleading oracle. The
fixed-grid sum differs from full composition by **1.07837 px mean / 1.62170 px max** on the analytic frame-2
fixture, while production `source_coords` matches the independent fixed-grid replication within **9.6e-06 px**.
A zero-residual projective control remains exact. This demonstrates a production representation mismatch, not
that the resulting image is visibly worse on real footage. No stabilization defaults or production fitting
behavior changed. The candidate composition module's stale mesh/row evaluation-order note was corrected.

Verification. The focused audit/candidate/shape tests passed **62 tests**; the composition/plan/render group
passed **281 tests**; `.claude/verify.sh` passed **1,536 tests**. CodeGraph was synced and current; its affected
test heuristic found no tests, so importing tests were run directly. The composition candidate can express the
full composed map, but real-footage quality and runtime impact remain unmeasured.

## Held-out rendered source_highpass composition candidate on `00177.MTS` t090 10 s (2026-10-16)

This is the next E02 composition-candidate gate after the `00176.MTS` t060 10 s source-chart high-pass
medium-confidence tie: a **distinct source** at the longer available held-out window. Distinct local
`00177.MTS` (SHA-256 `8c5d98d15448ad0557537ba9d02a28fc426185b4e32bdf1be3eb4964fdbea714`, different from
`00176.MTS`) at the 90–100 s window, whose 10 s baseline capture
`out/bench/current_source_00177_t090_10s_baseline/manifest.json` (SHA-256
`b72992740c74067421ca6592df1c0abd471710ba46bfe2d0c3ba40b2b1e926a5`) is `status=success` with captured
`plan_r0.npz`/`motion_r0.npz`, 600 deinterlaced fields -> 300 outputs at 1920x1080 / `30000/1001`, weights
`84c1f730d4620ef4dd0a2eeeef97db7a729fc4190ee05cdd0169030fc92fccc1`, `config_digest`
`8e0378b967beaed72f14d572ad1dd6d0b480034c1b5a094c2d90c7562258a2fe`, captured at code revision
`afc16d40c744a8d4b54207c5ee6ab3bf9d68216f` (clean). The capture manifest's `metrics.reliability.reliable` is
**false** ("output motion fit fell back to identity on 5 frame(s)"), so the baseline's own evaluator numbers are
not reliable evidence; the captured motion has 1 fit fallback. Render, mask build, review and evaluation ran at
HEAD `c6afd7f71ea896101e7ce474de5cb8cb62de3fc0`; **no source code, default or production path changed.**

Only the `source_highpass` candidate was rendered (new, never-overwritten dir
`out/bench/e02_source_highpass_00177_t090_10s_heldout/`):

```bash
.venv/bin/python -m bench.real_composition_render out/bench/current_source_00177_t090_10s_baseline/manifest.json \
    --mode source_highpass --out-video out/bench/e02_source_highpass_00177_t090_10s_heldout/composition_source_highpass_r0.mp4 \
    --lock on --lock-chunk 30 --lock-pause 12
```

Candidate MP4 SHA-256 `81951a40e9595b59e781b54a7ba1110587145688e49166e38c93cccf87dabe96`, 300 frames,
1920x1080, CFR-verified; 11 shared-lock releases, 120.08 s aggregate pause. The dense-grid approximation to the
exact composition was checked on 315 sub-pixel points over 5 representative frames: **max 0.11828 px, mean
0.000913 px, p95 0.000805 px** (the max is at source frame 300; frame 0 is exact to ~5e-13 px). The candidate's
own center-out-of-bounds coverage (`invalid_support`, same `render_from_src` convention as production) is
pooled **1.375 %** of output pixels (max frame 6.07 %); the baseline `mask_r0.mp4` is 1.155 % mean / 6.28 %
max. The border regression is concentrated at the edges: **bottom-4-row mean 53.4 % candidate vs 37.7 %
baseline; right-4-column 45.9 % vs 34.9 %**; over output frames 120–270 (≈4–9 s) full-frame invalid is 0.94 %
candidate vs 0.38 % baseline, and the candidate exceeds the baseline on **174/300** frames (per-frame
correlation 0.93).

Matched numerical metrics used the existing evaluator at the manifest's evaluation settings (`--iters 12
--downscale 2 --vf fps=30000/1001 --tracks`). To keep the comparison truly matched, the candidate's validity
mask was rebuilt from the same candidate field with `warp.valid_mask` (no image render) and **verified 0/300
mismatches** against the render report's per-frame invalid counts (`candidate_mask_r0.mp4`). The baseline
masked run reproduces the manifest exactly (jitter 2.502795 px, wobble 0.478335 px, stability 0.896392, 5
fallback fits). Input metrics are identical in all runs (jitter 4.725118 px, wobble 0.749449 px).

| pair | output jitter px | output wobble px | Liu stability (min) | fit fallbacks |
| --- | ---: | ---: | ---: | ---: |
| baseline, own mask | 2.502795 | 0.478335 | 0.896392 | 5 |
| **source_highpass, own mask** | **2.841923** | **0.199645** | **0.805056** | 4 |
| baseline, no mask (matched control) | 2.489112 | 0.584231 | 0.902086 | 5 |
| source_highpass, no mask (matched control) | 2.849094 | 0.227934 | 0.772198 | 4 |

Relative to the baseline, the candidate **raises output jitter by +13.55 %** (+0.339128 px; +14.46 % without
masks), **lowers wobble by 58.26 %** (−0.278690 px; −60.99 % without masks) and **lowers the Liu stability
minimum by 0.091336** (the min axis is rotation). The independent KLT evaluator is **not applicable** on this
window: path metrics are unavailable because input tracking failed on 37 pairs, the baseline output on 40 pairs
and the candidate masked output on 80 pairs (failed pairs are never bridged or treated as identity).

Blinded position-reversed review (640x360 panels; `bench.real_pair_review --candidate-video`, then
`bench.gemini_blind_review` via the Vertex API, `gemini-3.8-flash`, gcloud user login). Original clip
(baseline = left/A, candidate = right/B) was judged **A = baseline, medium confidence**, citing "jagged black
boundary deformation along the lower and right margins" on the candidate panel during 00:04–00:09 (consistent
with the measured coverage regression). The position-reversed clip (candidate left/A, baseline right/B) was a
**medium-confidence tie**. The position-bias audit records one complete pair, one non-tie and no method
agreement (`side_following_pairs = 0`, `method_agreement_pairs = 0`). So the visual pair is **inconsistent and
uninformative**: it does not confirm a consistent candidate regression, but it also gives no candidate benefit.
Reports/clips are `out/bench/e02_source_highpass_00177_t090_10s_heldout/{composition_source_highpass_r0.json,
baseline_vs_source_highpass{,_posrev}.mp4,gemini_source_highpass_review.json,
gemini_source_highpass_position_bias.json,e02_report.json}`; all ignored.

Reading. On this distinct held-out window the source-chart high-pass candidate does **not** reproduce the
`00176` tie: it shows a real **jitter regression (~+14 %)**, a measured **border-coverage regression** and a
lower Liu stability minimum, alongside a large wobble reduction. The single non-tie visual judgment preferred
the baseline and named the candidate's border artifacts, but the reversal was a tie, so the visual evidence is
not decisive. This is **one source, one 10 s window, one review model**; the baseline manifest's reliability is
false (5 output fallback frames) and the independent KLT estimator could not score either run. It is not a
quality improvement and does not justify any production integration or default change.

Next decision. Do **not** integrate or select `source_highpass`. Treat `00176` t060 and `00177` t090 as
reserved evidence and do not tune thresholds on them. Next, attribute the candidate's extra border invalids by
composition component on this window (the existing border-component audit) to test whether the high-pass chart
inherently enlarges the invalid support, and only then consider a border-aware or coverage-constrained variant;
re-test any variant on a further distinct source/window with the matched masked evaluator and a position-
reversed review. No production wiring, default change or mode selection follows from this entry.

## E02 source_highpass invalid-support attribution on `00177.MTS` t090 10 s (2026-10-16)

Read-only CPU/NumPy attribution of the extra border invalids recorded in the held-out E02 entry above. A
local, ignored script `out/bench/e02_source_highpass_00177_t090_10s_heldout/mask_attribution.py` reconstructed
the per-pixel center-out-of-bounds (`invalid support`) mask of three output->source fields on the **identical**
300 output indices of `out/bench/current_source_00177_t090_10s_baseline/`: the production plan
(`stabilize._source_coords_ts_scales_np` + `_valid_mask_np`), the exact composed candidate (`highpass=False`),
and `source_highpass`, the latter two through the renderer's `ForwardComposedMap` (dense-grid interpolated
forward composition). No source footage was decoded for rendering, no video was re-rendered, no heavy lock was
acquired, and no production default or artifact changed.

Verification gates (all passed, otherwise the comparison would stop): the reconstructed output indices equal the
render report's `pairing.frame_indices` (300 frames, 59.94->29.97, step 2); the reconstructed production mask is
**pixel-exact against `mask_r0.mp4`** (0 differing pixels over 300 x 2,073,600); and the reconstructed
`source_highpass` per-frame invalid counts match `composition_source_highpass_r0.json` exactly (0/300
mismatches; 8,556,066 total).

Pooled over 300 output frames (622,080,000 pixels):

| field | invalid pixels | pooled invalid |
| --- | ---: | ---: |
| production baseline | 7,185,290 | 1.155 % |
| exact composition (`highpass=False`) | 49,447,376 | 7.949 % |
| source_highpass | 8,556,066 | 1.375 % |

Paired per-pixel transitions (valid -> invalid / invalid -> valid / net invalid):

| pair | valid -> invalid | invalid -> valid | net invalid | net pooled |
| --- | ---: | ---: | ---: | ---: |
| baseline -> exact | 43,973,048 | 1,710,962 | **+42,262,086** | +6.79 pp |
| exact -> source_highpass | 2,089,386 | 42,980,696 | **-40,891,310** | -6.57 pp |
| baseline -> source_highpass | 2,922,938 | 1,552,162 | **+1,370,776** | +0.22 pp |

The three nets add exactly (`baseline -> exact` + `exact -> source_highpass` = `baseline -> source_highpass`).
On the baseline -> source_highpass pair the net regression is concentrated at the fixed 4 px edge strips
(overlapping corners counted per strip, the renderer's convention): bottom valid->invalid pooled **21.5 %**
(invalid->valid 5.8 %), right 15.4 % (4.4 %), left 5.3 % (19.4 %), top 2.7 % (8.7 %); pooled strip invalid
fractions move bottom 37.7 % -> 53.4 % and right 34.9 % -> 45.9 %, matching the renderer report. The per-frame
net flips at about output frame 120: frames 0-120 have fewer candidate invalids (nets -219,002 and -390,384),
frames 120-300 more (+668,174, +685,046, +626,942), and `source_highpass > baseline` on 174/300 frames.

Reading. On this window the high-pass chart does **not** inherently enlarge invalid support. Relative to the
exact composition it removes 42,980,696 pixels from the invalid set (net -6.57 pp), cancelling **97.7 %** of the
exact composition's 43,973,048 newly-invalid pixels. The raw exact composition is what blows coverage up
(1.155 % -> 7.949 %); the temporal high-pass is precisely its coverage control. The candidate's residual net
regression versus production (1,370,776 px, +0.22 pp pooled, +19.1 % relative) is a small second-order
difference between the composed-then-high-passed source chart and production's fixed-grid smoothed residual
plan, concentrated at the bottom/right edges and after frame ~120. This is a geometry/coverage attribution, not
perceptual quality; baseline -> exact also conflates fixed-grid-vs-composed residual placement with
smoothed-vs-raw residuals, and the exact field uses the same dense-grid interpolation approximation as the
renderer. It is one reserved 10 s window of one source and nothing is tuned on it.

Next decision. Do **not** treat "the high-pass chart inherently enlarges support" as the cause, and do **not**
integrate or select `source_highpass` on this evidence. If a composed variant is revisited, target the
bottom/right border residual placement (e.g. a border-aware or coverage-constrained correction) and re-test on a
further distinct source/window with the matched masked evaluator and a position-reversed review. Script and
report are ignored under `out/bench/e02_source_highpass_00177_t090_10s_heldout/` (`mask_attribution.py`,
`mask_attribution.json`); no production wiring, default change or mode selection follows.

## E02 source_highpass held-out test on `STREAM_00013.MTS` t032–042 (2026-10-16)

This is the third-source check after `STREAM_00016.MTS` t120–130 proved too static. The proxy screen shortlisted
`STREAM_00013.MTS`; baseline-only RAFT/KLT anchors at t032 and t039 measured 1.854 and 2.218 px/frame,
respectively, with zero fit fallbacks and both evaluators agreeing that the baseline reduces jitter. A t048
anchor measured 29.574 px/frame and KLT was unreliable, so the adjacent t039–049 window was rejected. DeepSeek
via Pi independently recommended t032–042: sustained, trackable motion, two reliable checkpoints, and no fast
pan at t042 onward. This selection was made before evaluating the candidate.

The 10 s baseline capture `out/bench/e02_baseline_s13_t032_10s/manifest.json` succeeded at revision
`4a9f36af391ad2031cbdd2c7b08c14663856c680`, with config digest
`feef17a4cf8d3bf85c900c599465e8df800f022a6a7ac8cd4ab0fb9e7d78f9ab`, 300 output frames, 600 interlaced
analysis fields, zero output fit fallbacks, and reliable RAFT/KLT metrics. Input motion was 2.457 px/frame;
masked baseline output jitter was 0.582 px RAFT and 0.724 px KLT. The candidate was rendered by the unchanged
`source_highpass` composition path to
`out/bench/e02_source_highpass_s13_t032_10s/composition_source_highpass_r0.mp4`.

Geometry and pairing gates passed. The dense-grid approximation differed from direct exact composition by at
most 0.00314 px over 315 points on five frames (mean 0.000295 px, p95 0.001061 px). The candidate validity
mask was reconstructed from the candidate field and matched the render report on all 300 frames. Baseline,
candidate, and masks have identical 300-frame PTS at 30000/1001 fps. Candidate pooled invalid support was
1.833% (max frame 6.13%), slightly below baseline's 1.919% mean (6.68% max); the coverage change is small.

Matched numerical comparison used 12-iteration, half-resolution RAFT plus independent KLT. Baseline masked
metrics below come from the successful capture manifest; baseline no-mask and both candidate measurements use
`bench.evaluate` at the same cadence and settings. Every run had zero fit fallbacks and 299 supported frames.

| pair | RAFT jitter px | RAFT wobble px | Liu stability | KLT jitter px | KLT stability |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline, own mask | 0.581889 | 0.205542 | 0.904685 | 0.724355 | 0.906185 |
| source_highpass, own mask | 0.597238 | 0.220403 | 0.896251 | 0.944486 | 0.808050 |
| baseline, no mask | 0.597527 | 0.209258 | 0.906597 | 0.722939 | 0.905916 |
| source_highpass, no mask | 0.591276 | 0.222406 | 0.896191 | 0.945822 | 0.818298 |

With masks, the candidate raises RAFT jitter **2.6%**, KLT jitter **30.4%**, and wobble **7.2%**, while lowering
Liu stability by 0.00843. Without masks, RAFT jitter improves only 1.0%, but KLT jitter worsens **30.8%** and
wobble worsens 6.3%. Both evaluators still classify the candidate as improving over raw input; that does not
mean it improves over the baseline. The similar KLT regression with and without masks makes invalid-pixel
masking an unlikely explanation. This points to an appearance-domain or sparse-feature-stage interaction,
but does not identify the cause.

Gemini 3.8 Flash via the Vertex API reviewed the position-reversed, blinded baseline/candidate pair. Both
orientations were medium-confidence ties, citing nearly identical motion and low-light/projection limitations.
The deterministic position-bias audit found one complete pair, no non-tie, and no method agreement; this is
consistent with no visible difference at 640x360, not evidence of equivalence.

Reading. This is a reliable, motionful third-source comparison, and it does **not** justify selecting or
integrating `source_highpass`: RAFT is effectively flat while KLT regresses materially, and Gemini sees no
clear advantage. Keep the candidate experimental. Do not tune on `STREAM_00013.MTS` t032–042.

Next decision. The first diagnostic should separate KLT feature detection from KLT template tracking on a
pre-declared, disjoint calibration window that passes the same baseline reliability and motion gates. Use a
2x2 detection-image/tracking-image experiment (baseline luminance or high-pass for each stage), with the
baseline/baseline and high-pass/high-pass cells reproducing the observed endpoints. Track KLT jitter, feature
locations and responses, per-feature trajectory noise, plus RAFT as a negative control. Choose the next
candidate only from the predeclared window; use t032–042 once as confirmation, never to tune. The test can tell
whether the effect follows the detected feature set, the tracking appearance, or their interaction before any
production change is considered. Reports, comparison clips, and evaluator files remain ignored under
`out/bench/e02_{anchor_screen,baseline_s13_t032_10s,source_highpass_s13_t032_10s}/`.

## KLT detector/tracker-domain audit (2026-10-16)

The purpose is to locate the KLT portion of the `source_highpass` regression from E02, without changing the
stabilizer. `mlxstab.tracks.pair_similarity_domains` now exposes an explicit detection image and two LK images;
the existing `pair_similarity` wrapper still sends luminance to both stages. `bench.track_domain_audit` runs
the 2x2 luminance/high-pass factorial, records detection/tracking/inlier counts and failures, respects the same
invalid-pixel masks across cells, and computes contrasts on the longest contiguous segment supported by all
four cells. LK uses uint8 only (verified against the installed OpenCV 4.13, which rejects float32). High-pass
uses a signed midpoint encoding with one fixed scale per run; `--hp-scale` reuses the calibration scale. This
is an audit path, not a production option.

The scale was calibrated on decoded `STREAM_00013.MTS` t029–031. A baseline-only 1 s anchor at t029 had
2.385 px/frame motion, zero KLT fallbacks, and RAFT/KLT agreement on direction. Over the 2 s calibration cache,
the four cells shared 57 consecutive pairs; the last two luminance-detection pairs fell back. On those 57 pairs,
the KLT tracking-image main effect was -0.0086 px (high-pass minus luminance), interaction -0.0087 px, and
detection-image main effect -0.0165 px. This was raw-input calibration, not independent output validation.

The first masked output diagnostic decoded the same first 2 s of the E02 baseline and `source_highpass` outputs
at identical 30000/1001 cadence and used each output's aligned invalid mask. All four cells supported all 59
pairs. The frozen t029 high-pass scale and all KLT parameters were the same for both output sequences:

| detector / tracker | baseline jitter px | source_highpass jitter px | candidate minus baseline |
| --- | ---: | ---: | ---: |
| luminance / luminance (LL) | 0.343187 | 0.374797 | +0.031610 (+9.2%) |
| luminance / high-pass (LH) | 0.340667 | 0.378368 | +0.037701 (+11.1%) |
| high-pass / luminance (HL) | 0.345238 | 0.332412 | -0.012826 (-3.7%) |
| high-pass / high-pass (HH) | 0.346004 | 0.338344 | -0.007660 (-2.2%) |

On this one 2 s output window, changing the LK image had a small effect (0.0048 px within the candidate);
changing the detection representation was larger. The candidate-versus-baseline KLT regression appeared in
both luminance-detection cells and disappeared in both high-pass-detection cells. However, high-pass detection
hit the 500-corner cap in both sequences, while luminance detected only about 286 corners in baseline and 279
in candidate. The detector effect is therefore confounded by candidate count and feature selection. This does
not show that high-pass improves stabilization, and it does not generalize beyond this one output window.

Next experiment: use a fresh set of disjoint windows, freeze all tracking and representation parameters, and
cross detector representation with an equalized corner count (primary comparison at a count both detectors
can supply, with the 500-cap cell as a secondary density comparison). Aggregate paired contrasts by window,
not by pretending adjacent pairs are independent. Keep the current t032 result as a diagnostic lead only; do
not tune a production default from it. Test commands passed: `tests/test_tracks.py` and
`tests/test_track_domain_audit.py` (run before commit).

## Follow-up: localize the E02 KLT regression (2026-10-16)

The equal-count KLT factorial was extended from the first two seconds to the full E02 t032–042 output, using
K=250 detected corners in every cell and the same aligned per-output masks. All 299 pairs were supported, with
zero tracking fallbacks. The candidate regressed in every detector/tracker cell: LL jitter rose 31.0%
(0.7258 -> 0.9508 px), LH 28.9% (0.7282 -> 0.9384), HL 26.0% (0.7260 -> 0.9149), and HH 20.9%
(0.7186 -> 0.8689). Equalizing feature count therefore removes the earlier cap/count confound but does not
remove the full-window regression. The high-pass detector/tracker representation softens it somewhat; it does
not reverse it.

Per-pair loss localizes most of the increase to output seconds 7–9 (source t039–041). In the LL cell, pairs
210–239 added an average 1.061 px of jitter contribution and every pair in that block was worse; pairs 240–269
added 0.473 px on average. The first 210 pairs were near zero or mixed. Track counts remained similar and there
were no fallbacks, though candidate inlier fraction was modestly lower in the event block (0.873 vs 0.903).
The equal-count result is one clip/window and does not establish a general effect.

A focused, unmasked `bench.evaluate` run on the same 60-frame t039–041 cuts found RAFT jitter 0.5437 ->
0.6377 px (+17.3%) and KLT jitter 0.6411 -> 1.4097 px (+119.9%), with zero fallbacks. This supports a real
local increase in measured frame-to-frame motion in the rendered candidate; KLT amplifies that increase much
more than RAFT. It does not by itself tell us whether viewers would prefer either rendition or which component
of the composition change causes the motion.

Gemini 3.8 Flash reviewed the isolated seconds 7–9 in both left/right orientations and returned a tie at medium
and low confidence, describing similar visible motion. That agrees with “no reliably visible winner at this
review scale,” not with equivalence. Independent fresh source anchors produced mixed or near-zero one-second
KLT deltas (STREAM_00024.MTS t571 and STREAM_00030.MTS t087, tested at K=250 and K=500), so the E02 event has
not generalized to other windows.

Decision: keep `source_highpass` experimental; make no default or integration change. The investigation has
now narrowed the next question to the localized t039–041 rendered-motion increase and its cause. Next, compare
the baseline and candidate source-coordinate maps over that fixed event at equal output probes, then repeat the
same predeclared motion and blinded-review protocol on a disjoint qualifying window. If the map comparison does
not account for the measured increase, audit rendered-image resampling and evaluator sensitivity. Do not tune
on the already-inspected t039–041 event. Detailed ignored artifacts and per-pair data are under
`out/bench/e02_source_highpass_s13_t032_10s/`, `out/bench/e02_anchor_s13_t571_1s/`, and
`out/bench/e02_anchor_s30_t087_1s/`.

## Source-map and feature-location decomposition of the E02 event (2026-10-16)

Follow-up probes use the exact captured plan/motion and renderer cadence for `STREAM_00013.MTS` t032–042. The
analysis stream has 600 deinterlaced frames at 59.94006 fps; the encoded video has 300 frames at 29.97003 fps.
The renderer's rational source-index mapping advances two analysis frames per output frame. This avoids comparing
plan-frame derivatives at 59.94 Hz with output-image metrics at 29.97 Hz.

The renderer-equivalent dense `source_highpass` map and production `source_coords` were compared on a shared 9x5
interior output grid. At source-fixed points, all 45 Newton inversions converged and remained visible over output
frames 150–299 (baseline inverse residual p95 0.000061 px; candidate below 4e-13 px). Applying the exact
`mlxstab.metrics.jitter` Gaussian high-pass (sigma 15 output frames) to those source-fixed trajectories gives
1.6660 px for baseline and 1.5930 px for the candidate (-4.4%). The result does not support a noisier coherent
camera path in the candidate. It is sparse and does not measure local texture-weighted deformation.

A 3x3 KLT audit on the same 60 rendered event frames found candidate jitter higher in each of the five tiles
with reliable tracks: top-center +15%, middle-left +22%, middle-center +29%, bottom-left +13%, and bottom-center
+70%. The other four edge tiles already had many KLT fallbacks in both clips and are not reliable comparisons.
The five supported tiles had modestly fewer candidate tracks (about 4–9%); inlier fractions stayed similar or
rose slightly. Per-tile source-fixed geometry jitter was flat or lower in the regions with the clearest KLT
regression, including bottom-center (1.723 -> 1.494 px). A pairwise local-affine map-flow audit also found
mostly flat/lower affine jitter; its non-affine map-flow residual increased modestly in some middle tiles but
not in the bottom-center tile with the largest KLT increase. These map grids therefore do not by themselves
explain the KLT pattern.

An instrumented KLT run on those 60 frames reproduced the saved evaluator result (0.6325 vs 0.6411 px baseline;
1.4518 vs 1.4097 px candidate), with zero fallbacks. For each detected feature, the renderer-equivalent map
predicted its next output position by solving `M_(t+1)(q) = M_t(p)`. The map-predicted global similarity jitter
was effectively unchanged (2.1353 -> 2.1302 px), as was the fitted residual transform jitter (2.5205 -> 2.5281).
However, the spatial non-similarity residual of the map at actual KLT feature locations rose from 0.406 to
0.606 px RMS (+49%); the high-passed temporal variation of that per-pair RMS rose from 0.119 to 0.229 (+93%).
This supports a time-varying local-warp interaction at tracked features, though the residual is not independent
ground truth and the audit uses the features selected in each rendered condition.

The fitted residual transform composed with the map-predicted transform reproduces observed KLT jitter almost
exactly: 0.6332 px vs 0.6325 baseline and 1.4562 px vs 1.4518 candidate. The high-passed geometry and residual
velocity vectors were strongly opposed in baseline (cosine -0.971) and less opposed in the candidate (-0.770).
That phase change explains how similar component magnitudes can combine into very different observed jitter.
It is a decomposition of the measured estimator, not proof that the residual is an independent physical signal.
Other track-quality measures were close: median forward-backward error 0.00974 -> 0.00971 px, p90 0.0481 ->
0.0505 px; median patch ZNCC 0.9945 -> 0.9947; median structure-tensor conditioning 2.370 -> 2.417. These do
not show a gross tracking failure, but do not exclude subtler local localization effects.

Decision: the evidence now weighs against a globally rougher candidate path. The leading explanation is an
interaction between the changed composition map's local, time-varying non-similarity and feature-based motion
estimation; do not label it an appearance-only filter effect. Keep `source_highpass` experimental and make no
default change. Next, repeat the feature-location map/track decomposition on a predeclared, disjoint window that
passes the same baseline motion and reliability gates. Check whether local map residual variation and the
geometry/residual phase shift recur in the same image regions as KLT jitter. If they do not, instrument feature
localization and local image resampling before considering any map regularization. Ignored diagnostics are under
`out/bench/e02_source_highpass_s13_t032_10s/` (`geometry_map_dense_probe.json`,
`source_fixed_geometry_probe.json`, `klt_tile_audit.json`, `local_map_flow_audit.json`,
`klt_map_decomposition.json`).

## Pre-registered disjoint-window replication: `STREAM_00024.MTS` t109–119 (2026-10-16)

This window was selected before running the candidate. Baseline-only anchors at t109 and t114 both passed the
reliability gate with zero fit fallbacks and 29/29 supported output pairs. Their input motion was 0.452 and
0.823 px/frame, respectively. The full 10 s baseline capture succeeded at
`out/bench/e02_baseline_s24_t109_10s/`, at code revision `e55336452fdeb98c51d1184d2933010104bcaef1` and config
digest `feef17a4cf8d3bf85c900c599465e8df800f022a6a7ac8cd4ab0fb9e7d78f9ab`. It contains 600 deinterlaced fields,
300 aligned outputs at 30000/1001 fps, captured plan/motion, and zero fit fallbacks. The reliability gate passed:
299/299 supported pairs, zero fallbacks, and 0.559 px/frame mean input RAFT motion. Masked baseline output jitter
is 0.228 px RAFT and 1.521 px KLT; raw input jitter is 0.538 px RAFT and 1.993 px KLT. The candidate render
is now running from this captured plan; no candidate metrics have been inspected.

The baseline stabilization took 1059 s and evaluation 553 s on this M5. This cost is recorded because the
paired render/evaluation has the same full-window scope; avoid additional full-window repeats unless the result
is inconclusive or needed to verify a failure.

DeepSeek Flash via Pi reviewed the protocol before candidate results existed. It endorsed the disjoint paired
window as a reproducibility check, while warning that one new interval cannot establish generality, baseline
reliability alone does not guarantee an informative event, and Gemini should remain secondary to quantitative
metrics. It also advised freezing the contrast and thresholds before the paired candidate run.

Frozen comparison: use full-window masked KLT and RAFT jitter on the same 300 output frames at 30000/1001 fps,
with each rendition's aligned validity mask. The primary contrast is
`delta_KLT = KLT_candidate - KLT_baseline` in px; the supporting KLT-specific contrast is
`delta_KLT - delta_RAFT` in px. A reliable baseline must have zero fit fallbacks, at least 270/299 supported
pairs for both evaluators, and mean input motion >= 0.4 px/frame. A candidate is not compared if baseline or
candidate alignment/reliability fails.

Decision thresholds are anchored to the original held-out KLT increase of 0.220 px (0.724 -> 0.944 px):

- **Replicates the material KLT-specific increase:** `delta_KLT >= 0.110 px` and
  `delta_KLT - delta_RAFT >= 0.110 px`, with both captures reliable and aligned. Then run the same local-map / KLT
  feature-location decomposition on this disjoint window. A blind Gemini comparison is a secondary perceptual
  check and cannot override these measurements.
- **Does not replicate at the pre-set material scale:** both absolute contrasts are <= 0.055 px, with both
  captures reliable and aligned. Do not make a mechanism claim from this window; classify the original as
  sequence-specific or unreproduced pending additional anchors.
- **Inconclusive:** either contrast falls between the thresholds, the KLT and RAFT contrasts conflict, or any
  reliability/alignment gate fails. Do not choose a favorable subwindow after seeing results; select another
  anchor without candidate screening or revise the protocol before a new paired run.

This single disjoint window tests replication only, not generality or causality. Any rolling 2 s breakdown,
Gemini preference, or local feature/map analysis will be labeled secondary or exploratory. The original
t032–042 effect remains the positive reference; its event window was data-selected, so it is not used to tune
the thresholds.

## Event-severity gate and exploratory outlier audit (2026-10-16)

The disjoint t109–119 full-window no-replication result does not by itself test whether that window contained an
event comparable to the original t039–041 excursion. To check that gate without RAFT, stabilization, or a new
render, a single LL K=250 KLT pass was run on the existing S24 baseline/candidate videos and their own aligned
masks. It used the original factorial's frozen detector/tracker settings, seed, masks, and jitter sigma. Both
clips had 299 supported pairs and zero fallbacks. LL jitter was 1.5047 px baseline and 1.4908 px candidate.
Across all 240 possible 60-pair blocks, the largest candidate-minus-baseline mean per-pair jitter contribution
was +0.0899 px (pairs 221–280), with only 34/60 pairs positive. No block reached the fixed comparable-event
gate of +0.519 px and at least 46/60 positive pairs (the minimum event mean and positive-pair count across the
four original K=250 cells). Thus S24 is a valid full-window no-replication, but it did not contain an event of
the original excursion's severity; it does not refute an event-specific effect.

The existing full-window t032–042 factorial confirms that the original effect is highly concentrated: mean
candidate-minus-baseline KLT jitter contribution in pairs 210–269 was +0.519 to +0.771 px across the four cells,
versus +0.053 to +0.089 px in pairs 0–209 and 270–298; 46–54 of the 60 event pairs worsened in each cell. In
the separate 59-pair event map decomposition, the per-pair LL contribution delta had Spearman rho +0.499 with
the local map non-similarity RMS delta (+0.200 px mean), and rho -0.595 with inlier-fraction change (-0.033
mean). Associations with track count (-0.091), FB error p90 (-0.154), patch ZNCC (-0.235), tensor lambda2
(+0.073), and tensor condition (+0.070) were weak. Candidate invalid-mask coverage was 0.216 percentage points
worse in the event, while it improved in the pre and post stretches. Downsampled event image proxies changed
little (mean luminance -0.82/255, high-pass RMS -0.03, Laplacian variance -5). On S24, the same proxies were
also near flat while KLT remained near flat. The two event correlations may be readouts of one degraded
correspondence process; they are post-hoc associations, not independent evidence or causal proof.

Gemini 3.8 Flash's position-blinded visual calls on S24 were mixed: it preferred the candidate once and called
a tie in the reversed presentation, both at medium confidence. This is not a consistent perceptual winner.

Decision: preserve the pre-registered S24 no-replication label and keep `source_highpass` experimental. The
S24 window is out-of-event, and the original event associations do not justify a code change. The only useful
next mechanism test is a bounded offline replay on the existing t039–041 rendered clips, with a fixed split of
the 60 pairs and negative-control pairs before/after. The saved decomposition JSON contains per-pair summaries
but not feature correspondences, so any support/map-residual reweighting needs one deterministic KLT extraction
from those existing renders first. No new RAFT pass, stabilization, or render is needed. Continue only if that
held-out manipulation reduces at least half of the event excess while leaving the negative controls unchanged;
otherwise close this mechanism as unproven without tuning a production path.

Ignored S24 mask/frame/LL audit artifacts are under `out/bench/e02_source_highpass_s24_t109_10s/`.

## Localized sampler-support candidate replay on `00176.MTS` t060–070 (2026-10-16)

A research-only local border-support correction was replayed from the successful baseline capture at
`out/bench/current_source_t060_10s_baseline/` without rerunning RAFT. It blends the complete source-coordinate map
toward the global-only map near a temporal union of full-kernel-support failures. The rendered candidate and
mask are under `out/bench/local_support_t060_10s_r8_t2/`; the paired RAFT/KLT evaluation is
`evaluation.json`. The 300 output frames are cadence aligned at 30000/1001 fps. Raw-input RAFT numbers exactly
reproduce the baseline manifest, confirming the matched evaluation settings.

On this 10 s window, full-kernel-support invalid pixels fell from **2.640%** before correction to **2.350%**
after correction (1,807,861 net pixels). The candidate repaired 1,828,570 pixels and introduced 20,709 new
invalid pixels. Masked RAFT jitter was **0.51917 -> 0.51902 px** and wobble **0.05365 -> 0.05352 px** relative
to the existing baseline: effectively unchanged quality metrics. On raw input to candidate, RAFT and KLT both
reported reduced jitter (1.58734 -> 0.51902 px and 1.56631 -> 0.52454 px), but this run did not compute KLT
for the baseline output, so it does not establish a KLT difference between baseline and candidate. This is one
window and the support gain is not a demonstrated perceptual improvement.

The audit exposed a reporting bug: the class where both the complete and global-only endpoint maps were
unsupported was named “unrecoverable,” implying the blended map must also be unsupported. In fact, interpolation
between two unsupported endpoints can yield a supported result. The endpoint class contains 14,595,859 pixels;
the candidate repaired 207 more pixels than the “complete unsupported / global supported” class alone. The
report now calls this class `both_endpoint_maps_invalid` and reports final post-blend support separately. The
saved report was migrated to schema 2 labels without changing the previously measured counts or video/mask.
DeepSeek Flash via Pi reviewed this distinction; a regression test demonstrates an unsupported-to-supported
interpolation between endpoint maps.

Decision: keep the support correction research-only. It improves the sampler-support count on this window but
does not move measured stabilization quality. For the earlier source_highpass event analysis, the t039–041
window was selected after its metric increase was observed. Any split-based reweighting or replay on that same
event is exploratory mechanism analysis, not a held-out confirmation. Freeze any formula and control windows
before viewing a new candidate outcome; confirmation requires a new qualifying event selected from baseline-only
evidence. Do not tune production behavior on either already-inspected window.

## Pre-registered Stage 0 residual ablations on `00176.MTS` t060–070 (2026-10-16)

This is a development reproduction of the previously reported tree deformation / floaty-motion concern, not a
held-out test: this source window has already been visually reviewed. The fixed baseline is the successful
`out/bench/current_source_t060_10s_baseline/manifest.json` capture (source SHA-256
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`, t060–070, 600 deinterlaced fields,
300 outputs at 30000/1001 fps, config digest
`8e0378b967beaed72f14d572ad1dd6d0b480034c1b5a094c2d90c7562258a2fe`). Its baseline output is aligned and
reliable with zero fit fallbacks, RAFT jitter 0.51917 px and wobble 0.05365 px.

Before inspecting any new ablation output, run the existing `global-only`, `mesh-only` and `row-only` cases on
the exact same source window, weights and baseline configuration, with `--eval-tracks` and the same output and
evaluation frame rates. The existing combined baseline is the fourth arm. Each case differs only by disabling
the named residual channel(s); global motion is refit as defined by the case. Compare masked RAFT and KLT
jitter/wobble, fit/support failures, full-warp shape and coverage, and a blinded full-window review with panel
positions reversed. Require exact 300-frame CFR alignment and zero evaluator fit fallbacks before interpreting
quality metrics. If reliability or alignment fails, label that arm unscored. No parameter or threshold tuning
is allowed from this window. A single window/reviewer can establish a reproduction or a local explanation only;
any general quality claim requires a new source/window chosen without candidate outcomes.

Decision: run this matched ablation before further local-support or source_highpass work. Use the residual arms
to determine whether the reported deformation is present in the combined output and which fitted channel changes
it; do not select a new default from this diagnostic comparison alone.

Interim results (2026-10-16): the `global-only` and `mesh-only` 10 s arms both completed with 300-frame aligned
CFR output, reliable RAFT metrics, 299/299 supported output pairs and zero fit fallbacks. The combined baseline
has RAFT jitter/wobble **0.51917/0.05365 px** and KLT jitter **0.52383 px** (baseline KLT was measured
separately from its saved output and aligned mask, without another RAFT pass). Global-only measured
**0.51805/0.07810 px** RAFT jitter/wobble and **0.54210 px** KLT jitter. Mesh-only (global + mesh, rows
disabled) measured **0.51648/0.06323 px** and **0.52542 px** KLT jitter. On this window mesh-only is close to
the combined baseline for jitter but has higher wobble; global-only retains similar RAFT jitter while worsening
both wobble and KLT jitter. At this interim checkpoint, the row-only arm and blinded visual review were still
pending; both were completed later in this entry.

Final numerical arm (2026-10-16): `row-only` also completed with aligned 300-frame CFR output, reliable RAFT
metrics, 299/299 supported pairs and zero RAFT/KLT fit fallbacks. It measured **0.53613 px RAFT jitter,
0.05091 px wobble and 0.53842 px KLT jitter**. Across all four reliable fits, `mesh-only` has the lowest RAFT
jitter (0.51648 px), `row-only` the lowest wobble (0.05091 px), and the combined baseline the lowest KLT
jitter (0.52383 px). Relative to global-only, mesh-only improves RAFT jitter by 0.00157 px, wobble by 0.01487
px and KLT jitter by 0.01668 px. Relative to mesh-only, adding rows in the combined fit raises RAFT jitter by
0.00269 px while reducing wobble by 0.00959 px and KLT jitter by 0.00159 px. Row-only has higher RAFT and KLT
jitter than the combined fit, with slightly lower wobble. These are paired point estimates from one already
reviewed development window; adjacent frame pairs are not independent samples, and no significance or generality
claim follows. The blinded comparison is recorded below; causal attribution of the visible tree/floaty-motion
artifact remains open. Manifests and full-resolution rendered arms remain in the ignored
`out/bench/stage0_ablation_t060_10s_{global_only,mesh_only,row_only}/` directories.

Blinded visual review (2026-10-17): Gemini 3.8 Flash reviewed each full-window pair at 640x360, then
reviewed it again with panel positions reversed. Mesh-only and row-only were ties in both presentations.
Global-only was preferred once, then called a tie after reversal; treat this as an order-sensitive tie, not a
candidate win. Gemini noted that the low resolution and brief clip made fine trajectory differences hard to
distinguish. This does not establish perceptual equivalence or localize the reported tree deformation.

Decision: no residual arm has a reliable overall advantage on this window, and no default changes follow from
this reproduction. The pairwise results are consistent with the conflicting point metrics: mesh-only has the
lowest RAFT jitter, row-only the lowest wobble, and the combined fit the lowest KLT jitter. DeepSeek Flash via Pi
reviewed the outcome and agreed this is a null screen, not a causal test: reliable support/fits establish
measurement coverage, not perceptual validity; the single window cannot establish equivalence or generality.
If identifying the tree artifact remains necessary, the next useful test is a separately pre-registered,
position-reversed review using native-resolution, motion-magnified tree-canopy crops. Treat it as exploratory
because this development window and its ablations have already been inspected; do not use it to select a new
default. Review JSON and paired clips are under the ignored
`out/bench/stage0_ablation_t060_10s_review/` directory.

## Pre-registered S13 fixed-grid composition screen (2026-10-16)

Use the already captured `out/bench/e02_baseline_s13_t032_10s/manifest.json`, `plan_r0.npz` and
`motion_r0.npz`; do not run RAFT, decode footage, render, or change production behavior. First run
`bench.real_motion_field_audit` to screen all captured pairs for the maximum absolute fitted image-plane
rotation. Proceed to composition comparison only if at least one pair exceeds **0.5 degrees**. If not, report
that this capture does not qualify and stop without interpreting the composition difference as a rotation or
parallax test.

For a qualifying capture, run `bench.real_composition_audit` with `--every 1 --samples 401`. The predeclared
primary comparison is production `source_coords` versus `exact_composition`; the two high-pass charts are
secondary descriptive outputs. Report max and mean point displacement across the four fixed probes and all
300 output frames. Label a real-window counterexample to fixed-grid accumulation only if the rotation gate
passes and the primary maximum displacement exceeds **1 px**, or the mean exceeds **0.5 px**. Also report
probe-line deviation and source-bounds coverage; neither may be replaced by a composite score. Crossing the
displacement threshold is a real-window non-equivalence screen only; attribute it to fixed-grid residual
placement only after matching the temporal filtering. This sparse probe audit cannot establish a fold elsewhere
in the full warp or a perceptual improvement. Results from this already reviewed source are diagnostic and do
not authorize a default change.

## Baseline-only rotation-window selection (2026-10-16)

The S13 t032–042 capture failed the preceding rotation gate (maximum absolute rotation 0.108 degrees), so its
composition output will not be interpreted as a rotation/parallax test. For a follow-up stress-window screen,
select the strongest existing reliable one-second baseline anchor at a window without a candidate render:
`STREAM_00013.MTS` t048–049, anchor `out/bench/e02_anchor_s13_t048_1s/`. It has 29/29 supported pairs, zero
fallbacks and 29.574 px/frame input motion, the highest input-motion value in the available reliable anchor
inventory. The anchor itself is already inspected baseline-only data; any follow-up on its nested t048–058
window is exploratory and cannot count as independent or held-out validation.

First run `bench.real_motion_field_audit` on the anchor. Continue to a new 10-second baseline-only capture at
t048–058 only if the RAFT-derived motion remains reliable and at least one pair's absolute fitted image-plane
rotation exceeds **0.5 degrees**. If this fails, stop and report that the available saved anchors did not yield a qualifying
window; do not relax the threshold based on the result. If the 10-second baseline succeeds, freeze its plan and
motion and run the already specified `--every 1 --samples 401` production-versus-exact-composition audit, with
the same **>1 px maximum or >0.5 px mean** primary displacement criterion. Do not render a candidate or change
production behavior.

Screen results (2026-10-16): the full S13 t032–042 baseline has 599 reliable pairs, zero fallbacks, and a
maximum absolute fitted rotation of **0.1081 degrees**, so it fails the preregistered gate. The selected t048–049
anchor has 59 RAFT-supported pairs, zero RAFT fit fallbacks, and maximum absolute fitted rotation
**1.3677 degrees** (2 pairs exceed 0.5 degrees), so it passes the motion gate. Its benchmark RAFT reliability
flag is true; independent KLT is unavailable because 4 pairs failed. This is acceptable for the geometric
motion/composition audit only, and cannot support a stabilization-quality claim. The rotation-screen reports
are ignored under `out/bench/real_motion_field_audit/`.

The new t048–058 baseline capture completed successfully with the same config digest as S13 t032–042. It
decoded 600 interlaced fields into 300 CFR outputs at 30000/1001 fps; video and mask are aligned, with
299/299 supported RAFT pairs and zero fit fallbacks. Input-to-output RAFT jitter was **10.390 -> 5.761 px**;
wobble was **0.496 -> 0.451 px**. The full-window motion audit also passes the **0.5 degree** rotation gate:
599 fits, zero fallbacks, maximum absolute fitted rotation **1.3677 degrees**, with 2 pairs above the threshold.
KLT path metrics are unavailable (4 input and 5 output failed pairs), as anticipated from the one-second anchor;
this window is used only for the fitted-map geometry check. Its saved plan and motion are under
`out/bench/stage0_composition_t048_10s_baseline/`.

Composition screen result: `bench.real_composition_audit` ran with `--every 1 --samples 401`, so the report
contains all 600 deinterlaced-field plan indices. The output-rate subset (even plan indices 0–598, 300 frames)
also crosses the predeclared map-difference screen. Production versus unfiltered exact composition has **276.27
px maximum / 63.10 px mean** point displacement across the four probes on that subset; exact composition has
**153.30 px maximum / 49.08 px mean** mapped-line deviation and **86.53% minimum** probe coverage. The
source-highpass candidate differs from production by **177.55 / 13.77 px** maximum / mean displacement, with
**57.60 px maximum** line deviation; the reference-highpass candidate differs by **184.02 / 13.79 px**, with
**60.37 px maximum** line deviation. All four probes stayed finite, but no dense fold audit was run.

Interpretation boundary: this is a large real-window map discrepancy, not yet an isolated proof that fixed-grid
placement is the cause or that exact composition is better. The base `Plan.G` is shared, but the current exact
mode composes raw residuals while production uses separately time-smoothed cumulative mesh and row fields;
their filtering and sampling charts differ. KLT is unavailable and sparse line probes do not evaluate visual
quality. The 600-field sweep exceeds the 300 encoded-frame scope originally stated; the output-rate summary above
uses even indices, while the all-field report remains exploratory. No default or production change follows.

Exploratory same-base smoothing ablation (pre-registered before its result): using the same saved t048–058
Plan.G and captured motion, form raw cumulative mesh/row fields with no temporal smoother and compare
`source_coords` against `candidate_source_coords(..., highpass=False)` at identical four probes for the 300
output-rate frames (even indices 0–598). This removes the production-vs-candidate temporal-filter mismatch
while retaining the same virtual path. Reuse the **>1 px maximum or >0.5 px mean** displacement screen only as
a local geometry criterion: a mismatch then shows that identity smoothing does not eliminate the difference
between fixed-grid accumulation and source-space composition on these fitted maps; a match indicates that
filtering likely explains the prior screen. Either result is exploratory on an already inspected window and
cannot select a default or establish perceptual quality. The audit report is under
`out/bench/real_composition_audit/s13_t048_10s.json`.

Result (2026-10-18): evaluated production with raw cumulative mesh/row residuals and the captured `Plan.G`
unchanged against the exact composed map on a 65 × 37 uniform image-space lattice (2,405 points) for all 300
encoded-frame indices. The residual temporal smoothers were disabled on the production map; exact composition
has no residual smoother. The maximum displacement was **221.52 px**, the mean of per-frame maxima was
**143.06 px**, and the mean of per-frame spatial means was **54.90 px**. The predeclared geometric mismatch
screen is crossed by a wide margin, and the result shows that temporal residual smoothing alone does not explain
the sparse-probe discrepancy. This remains a same-window geometry diagnosis, not proof that exact composition
improves perceived stability: the comparison uses sparse samples and does not score the rendered video. The
dense report is `out/bench/real_composition_audit/s13_t048_10s_same_base_dense.json`. The exact-composition
candidate rendered successfully at the captured 300-frame 1920 × 1080 CFR cadence. Its own support report shows
6.392% mean invalid pixels and 16.451% at the worst frame. Gemini 3.8 Flash reviewed the 960 × 540-per-panel
baseline/candidate comparison and its panel reversal, preferring the baseline at medium confidence in both
orders; it cited rubber-sheet warping and curved border distortion in the candidate. The position-bias audit
confirmed that the preference followed the baseline across physical sides. The candidate is rejected for this
window; the result does not show that all composed residuals fail, but unfiltered exact composition is not a
production improvement here. A matched evaluator replay used the same raw source window, automatic bwdif
deinterlacing, `fps=30000/1001`, RAFT 12 iterations at downscale 2, and no output mask for either arm. Baseline
versus candidate RAFT jitter was **5.827 → 5.624 px** (3.5% lower) and wobble **0.485 → 0.418 px** (13.8%
lower), with zero RAFT fit fallbacks in both. However, KLT path metrics were unavailable: the raw input had 4
failed pairs and each output had 5, so the evaluator cannot establish an independent motion improvement. The
small RAFT gain does not outweigh the repeatable medium-confidence Gemini baseline preference and the
candidate's 6.392% mean invalid support. No production change follows. The render, paired review, judgments and
unmasked evaluator reports are under `out/bench/stage0_composition_t048_10s_baseline/`.

## Shared-pipeline MLX versus PyTorch MPS flow comparison (2026-10-17)

Added `bench.bench_stabilizer`, a paired command that substitutes only the RAFT flow backend while keeping the
planner, renderer, media I/O and MLX evaluator shared. The PyTorch adapter uses the upstream RAFT model on MPS,
reuses frame features across adjacent pairs and matches the production bidirectional batch and warm-start path.
Before running, it converts both checkpoint files to the same tensor layout and requires exact element-wise
equality; the successful default check matched all 154 tensors. Each run records separate checkpoint hashes and
the normalized tensor digest. Failed or cadence-mismatched pairs do not produce a Markdown comparison table.
This measures a backend swap inside mlxstab, not an independently implemented PyTorch stabilizer.

The full 540p synthetic comparison used one shared 30-frame 960×544 source, 12 RAFT iterations at full analysis
resolution, 12-iteration evaluation at 2× downscale, and three timed cached-model repeats after each cold pass.
Both runs succeeded with matching source hash, config digest, 30 output frames and encoded PTS. There were no
fallback frames. Median warm stabilization time was **15.441 s for MLX** and **19.275 s for PyTorch MPS** (about
**19.9% lower** for MLX on this run); end-to-end stabilization including model construction was **15.448 s** and
**19.612 s**. The shared MLX evaluator measured output jitter **0.07423 / 0.07333 px** and wobble **0.03115 /
0.03275 px** (MLX / PyTorch). Mean border fractions were **0.01722 / 0.01713**. Fitted plans differed: maximum
absolute delta was **1.944** in global homographies, **1.253 px** in mesh correction and **1.206 px** in row
correction. The run report is ignored under `out/bench/stabilizer_backend_540p_20261017/`.

This is one synthetic sequence. The stability metrics use the same MLX RAFT evaluator for both outputs; KLT and
blind visual review were not enabled, and plan differences are not a quality ranking. These numbers establish
that the full shared pipeline comparison runs and provide one scoped runtime result, not general performance or
perceptual superiority. Next, run the paired backends on real clips with KLT enabled and blinded visual review
before making a stabilizer-quality claim. No production default changed. Verification: `.claude/verify.sh` passed
**1589 tests in 57.60 s**.

## Real camcorder backend comparison (2026-10-17)

Both backends ran on the same decoded one-second window of `00176.MTS` at t800–801 (source SHA-256
`e4b1f80d…d9fbb3`), with the historical baseline flow settings, matched config digest, 30 encoded frames at
30000/1001 fps, and exact equality across all 154 normalized checkpoint tensors. The interlaced input was
deinterlaced to 60 fields for analysis. There were no planner fallbacks. MLX's one cached warm repeat took
**50.645 s** versus **66.456 s** for PyTorch MPS; these are single-run timings on this machine, not a general
performance estimate. RAFT jitter was **0.04580 / 0.04606 px** and wobble **0.02944 / 0.03000 px** (MLX / MPS).
Independent KLT jitter was **0.11875 / 0.11952 px**; both evaluators agreed that each output improved over the
input. The per-frame fitted plans were close, with maximum deltas of **0.0150** in global homographies,
**0.0726 px** in mesh correction and **0.0105 px** in row correction. Full results are in
`out/bench/stabilizer_backend_00176_t800_20261017/comparison.json`.

Gemini 3.8 Flash reviewed a blinded backend A/B clip and its position-reversed control. It selected the
PyTorch candidate in the original, then selected the MLX baseline after reversal; it chose screen side B both
times. The position-bias audit therefore flags that the preference followed screen position as the method
mapping changed. Treat Gemini's visual result as inconclusive. This is one previously used development window,
one warm repeat, and one short low-resolution visual review. It does not establish a quality winner or broad
speed result, and no production default changed. Next evidence should come from multiple separately selected
real windows, with repeated timing and counterbalanced blind reviews; keep the shared planner, renderer and
evaluator fixed.

## Preregistered multi-stream backend challenge panel (2026-10-17)

Before paired backend outputs are generated, freeze these eight existing reliable one-second baseline anchors,
one from each raw stream: `STREAM_00013.MTS` t048; `STREAM_00016.MTS` t120; `STREAM_00017.MTS` t021;
`STREAM_00023.MTS` t004; `STREAM_00024.MTS` t571; `STREAM_00025.MTS` t025; `STREAM_00030.MTS` t087;
`STREAM_00037.MTS` t081. The panel was chosen from existing baseline-only anchors to span motion levels;
the baseline input-motion values range from 0.129 to 29.574 px/frame, with the S13 t048 window an extreme
stress point. These clips were previously inspected for baseline reliability, so this is a fixed challenge
panel, not a random or held-out sample of the full camcorder corpus. The eight exact source hashes are recorded
in the benchmark output manifests.

Run every window through the shared planner, renderer and evaluator at the same historical settings
(20 flow iterations, half-resolution flow, 12 evaluation iterations at 2× downscale, three timed warm repeats,
KLT enabled, seed 0). Alternate `--backend-order mlx-first` and `--backend-order torch-first` across the frozen
window list; each manifest comparison records the order used. Require exact equality of all converted checkpoint
tensors and matched source/config, frame count, and encoded cadence. Keep all eight windows in the primary result,
including any weak or failed quality measurements. The anchor is the unit of analysis; frames, tracks and warm repeats are not independent
samples. Compare paired per-anchor MLX/MPS warm-median runtime ratios, and paired log ratios for each output
jitter metric. Use a one-sided paired t confidence bound (df=7); call MLX faster on this panel only if the upper
95% bound for runtime ratio is below 1.0, and non-inferior on RAFT and KLT jitter only if each upper bound is
below the predeclared 1.05 relative margin. Report all per-anchor values and a sensitivity summary without
S13 t048, while keeping the all-eight result primary.

Create blinded baseline/candidate review pairs for all eight windows and include each position-reversed control.
Treat visual judgments as usable only if the per-window position audit shows method agreement after reversal;
otherwise record them as position-following/inconclusive. The previous Gemini call followed screen side and is
not evidence. No algorithm or production defaults change from this challenge-panel experiment. Even a positive
result supports only this fixed panel and backend swap, not corpus-wide performance or a full independent
PyTorch stabilizer comparison.

Challenge-panel result (2026-10-17): all eight pairs succeeded with exact equality across 154 converted
checkpoint tensors, matching config and encoded cadence, reliable RAFT metrics, and zero planner fallbacks.
Each warm runtime is the median of three repeats. Ratios below are MLX / PyTorch MPS; KLT is absent where
contiguous-path metrics could not be formed.

| Frozen window | Backend order | Warm median seconds (MLX / MPS) | RAFT output jitter ratio | KLT output jitter ratio |
|---|---|---:|---:|---:|
| S13 t048 | MLX first | 50.79 / 66.66 | 0.979 | — |
| S16 t120 | MPS first | 54.44 / 66.34 | 1.074 | 1.000 |
| S17 t021 | MLX first | 53.88 / 69.22 | 0.968 | 1.031 |
| S23 t004 | MPS first | 55.39 / 67.93 | 0.921 | 0.996 |
| S24 t571 | MLX first | 53.88 / 69.81 | 1.147 | 0.948 |
| S25 t025 | MPS first | 55.04 / 67.99 | 1.080 | 1.011 |
| S30 t087 | MLX first | 54.10 / 69.17 | 1.060 | 1.004 |
| S37 t081 | MPS first | 54.73 / 67.63 | 1.037 | — |

The paired warm-runtime ratio mean was **0.7936**, with one-sided 95% upper bound **0.8087** (df=7), passing
the predeclared faster-on-this-panel criterion. RAFT jitter's paired geometric mean ratio was **1.0310**;
its upper bound **1.0813** exceeds the **1.05** non-inferiority limit, so the quality gate failed. Both backends
had higher RAFT jitter than their input on S25 t025 and S37 t081. KLT path metrics were available on six
anchors only: the available-case geometric mean ratio was **0.9979**, upper bound **1.0210** (df=5), but the
predeclared all-eight KLT gate is incomplete. KLT path metrics were unavailable because four input/output fits
failed on S13 t048 and two failed on S37 t081. Do not use the available-case estimate as a full-panel pass.

Gemini reviewed 16 blinded clips (eight original pairs and eight position reversals). Every non-tie judgment
selected screen side B; in all five pairs with non-tie judgments in both orders, the mapped preference changed
with panel position and none agreed on a method. The visual endpoint is invalid for comparing these backends.
The machine-readable per-anchor analysis is `out/bench/backend_panel8_20261017/panel_analysis.json`; raw paired
manifests, blind review clips and audit are in the same ignored output directory. Conclusion is limited to this
fixed, previously inspected challenge panel: MLX passed the runtime criterion, RAFT jitter non-inferiority did
not pass, KLT coverage is incomplete, and Gemini supplied no usable method preference. No overall quality
winner or production default follows.

The benchmark CLI records and honors the selected `--backend-order`; this run alternated it across windows.
Verification after that harness change: `.claude/verify.sh` passed **1591 tests in 57.70 s**.

## Exploratory fitted-motion localization (2026-10-17)

The challenge-panel runs saved plans but not fitted-motion NPZs. The paired CLI gained an opt-in `--dump-motion`
capture and records that setting in its comparison JSON and replayable run commands. The exploratory rerun of
four already-reported windows (S13 t048, S24 t571, S25 t025 and S37 t081) captured fitted transforms and
showed backend differences, especially in the tails for S13/S24. It also confirmed exact frozen-plan replay.
These windows were selected after the fixed-panel results because of high-motion plan divergence or RAFT/KLT
disagreement; no quality claim, gate, or prior result changes. The fitted-motion NPZ cannot attribute those
differences to raw flow prediction alone.

To close that gap, `--dump-flow-samples` now saves each backend's restored forward and backward vectors at the
exact grid-cell centres consumed by `motion.observe()`, plus frame-index pairs, cycle diagnostics and the
candidate-valid mask after any source/destination mask rejection. The numeric-only `mlxstab.flow_samples/1`
NPZ is a compact observation-boundary capture before robust fitting, not full-resolution dense flow and not
ground truth. Its option is part of experiment identity and the paired command. DeepSeek Flash via Pi reviewed
the capture boundary and specifically recommended retaining the validity mask; it is included.

The four exploratory windows were rerun with the original config and alternating backend order. All paired
manifests passed source hash, resolved-config, cadence and exact checkpoint tensor checks; each saved 59 pairs
on the common 135×240 fit grid. Among samples accepted by both backends, forward/backward EPE and fit/validity
results were:

| Window | Forward median / p95 EPE (px) | Backward median / p95 EPE (px) | Candidate-mask agreement | Median / max fitted-H corner delta (px) |
|---|---:|---:|---:|---:|
| S13 t048 | 0.0217 / 0.329 | 0.0221 / 0.396 | 96.99% | 0.033 / 26.77 |
| S24 t571 | 0.0058 / 0.050 | 0.0060 / 0.056 | 99.39% | 0.001 / 11.07 |
| S25 t025 | 0.0048 / 0.035 | 0.0047 / 0.033 | 99.68% | 0.003 / 2.31 |
| S37 t081 | 0.0026 / 0.032 | 0.0029 / 0.039 | 99.52% | 0.002 / 0.07 |

Per-pair common-valid flow EPE and fitted-H corner transfer differences were positively rank-correlated in every
window (Spearman rho 0.39–0.83). Most samples agree closely, with the practical differences concentrated in
tails and a few fitted pairs. An offline replay from the saved forward vectors, candidate-valid masks and
cycle diagnostics exactly reproduced `H`, mesh, rows and fallback decisions for all 472 backend/pair fits.
Therefore the observed split enters at the flow/validity boundary and the shared fit can amplify it on a few
pairs; this does not establish which flow is more accurate or explain perceived quality. The windows were
selected after the earlier challenge, the common-valid subset is selected by both estimated flows, and there
is no ground truth. Machine-readable results and captures are in ignored
`out/bench/backend_panel8_flow_samples_20261017/`.

Follow-up diagnostics on S13 t048 narrowed the numerical question, but did not identify a winning backend or
production fix. With `warm_start=false`, cross-backend sampled-flow EPE and candidate-mask disagreement fell
relative to the warm-start run (forward median/mean EPE 0.0325/0.4387 to 0.0197/0.1396 px; backward
0.0338/0.3755 to 0.0202/0.1645 px; mask agreement 96.99% to 98.98%), while output jitter stayed essentially
unchanged: MLX 21.265 to 21.607 px and Torch MPS 21.725 to 21.490 px. This shows warm state contributes to
backend divergence, but disabling it did not improve stabilization on this clip.

A teacher-forced initialization replay over the first 28 pairs reduced some of the self-warm divergence but
did not erase it: common-sample forward median/mean EPE was 0.0304/0.228 px from zero init, 0.0417/0.691
with self-warm, and 0.0370/0.471 with the previous pair's zero-init state; backward EPE was
0.0318/0.257, 0.0436/0.435, and 0.0391/0.470 px. Preprocessing was ruled out for this sample: all 60
normalized, padded network input tensors were bitwise identical across implementations. In an easy and a hard
pair replay, framework boundary differences and update-block numerical differences were similar in scale; the
hard pair's first-update discrepancy was about 0.05–0.06 source pixels. The hard pair's cross-backend EPE
started around 0.06 px after one update and its p95 grew through later recurrent updates. These are diagnostic
observations on selected pairs, not evidence that either flow is more accurate.

An 8/12/20-iteration paired sweep also failed to support lowering the default on this clip. Input jitter was
27.397 px throughout. At 20 iterations, output jitter was 21.265 px (MLX) and 21.725 px (Torch MPS); at 12,
21.704 and 21.602 px; at 8, 21.851 and 21.970 px. The backend ordering changes with iteration count, and the
shorter runs did not beat the 20-iteration result consistently. Wobble also moved in different directions.
There is no ground truth here, so these output metrics cannot establish stabilization accuracy.

The next decision-relevant step is a blinded, multi-clip quality evaluation with independently annotated
static landmarks or trusted camera-motion references, reporting stability, crop/warp artifacts and valid
coverage alongside runtime. Until that evaluation exists, keep the current defaults: these experiments explain
where backend differences can arise, but they do not establish a quality improvement. Captures and detailed
JSON analyses remain under ignored `out/bench/`.

Verification after adding the opt-in capture: `.claude/verify.sh` passed **1591 tests in 56.71 s**.

## Localized border-support candidate: rendered checks (2026-10-17)

The existing research-only candidate (`bench.local_support_render`, fixed `radius_px=8`, `temporal_radius=2`) was
rendered against three captured baselines without rerunning RAFT: `00176.MTS` at 60–70 s and 800–810 s, and the
distinct `00177.MTS` source at 90–100 s. Each render preserved the original 300-frame output cadence. Across all
three, full-kernel invalid support decreased, but that geometric proxy did not predict all motion metrics:

| Window | Kernel-support invalid fraction, baseline → candidate | Repaired / regressed pixels | Encoded center-invalid fraction, baseline → candidate | RAFT output jitter, baseline → candidate | KLT output jitter |
|---|---:|---:|---:|---:|---:|
| 00176 t060–070 | 2.640% → 2.350% | 1,828,570 / 20,709 | 2.237% → 1.993% | Not rerun | 0.52383 → 0.52454 px |
| 00176 t800–810 | 1.951% → 1.862% | 558,533 / 4,904 | 1.555% → 1.493% | 0.06321 → 0.06272 px | 0.18540 → 0.18347 px |
| 00177 t090–100 | 1.625% → 1.257% | 2,321,433 / 28,427 | 1.155% → 0.865% | 2.50279 → 3.07907 px | unavailable for both |

For the known t060 lower-edge frame (output 44, plan frame 88), the candidate recovered 1,664 center-valid pixels
with no newly invalid pixels. On t060, KLT had no failed pairs in either render; on t800 it also had full path
support. On the distinct 00177 window, KLT path metrics were unavailable on both renders (43 baseline and 48
candidate failed pairs). The paired RAFT evaluation on 00177 also had incomplete reliability: baseline/candidate
output fallback counts were 5/4, so the jitter increase cannot be treated as a clean quality estimate. Still,
this is an adverse guardrail result, not a pass. Gemini 3.8 Flash via the Vertex API reviewed blinded baseline vs
candidate clips at 640×360 with the panel positions reversed; all three windows were ties in both orientations.
This is no evidence of a visible improvement.

Interpretation: keep the candidate research-only. It improves measured sampler support on these windows, but
visual benefit is unresolved, one independent source has a large RAFT jitter increase with missing KLT path
metrics, and the candidate has not been evaluated across a broad held-out set. Do not integrate it into the
production warp or change defaults.

The unchanged candidate was then rendered on all eight previously frozen `backend_panel8_20261017` anchors.
Baseline/candidate metrics use the same source, MLX plan, output cadence, RAFT weights and evaluator; candidate
RAFT evaluation used 12 iterations at 2x downscale, while KLT used the same seeded sparse tracker. Streams are
the units of analysis, not pixels or frames:

| Stream | Kernel-support invalid % | Center-mask invalid % | Support pixels repaired / regressed | RAFT jitter ratio | KLT jitter ratio |
|---|---:|---:|---:|---:|---:|
| S13 t048 | 7.836 → 7.527 | 7.494 → 7.221 | 206,436 / 14,242 | 1.015 | unavailable |
| S16 t120 | 0.866 → 0.686 | 0.254 → 0.133 | 113,113 / 1,058 | 1.054 | 1.000 |
| S17 t021 | 0.864 → 0.762 | 0.405 → 0.356 | 64,106 / 449 | 1.090 | 1.121 |
| S23 t004 | 0.906 → 0.774 | 0.470 → 0.390 | 84,145 / 2,009 | 1.085 | 1.010 |
| S24 t571 | 3.927 → 3.746 | 3.506 → 3.419 | 119,796 / 7,120 | 1.034 | 1.005 |
| S25 t025 | 1.421 → 1.126 | 0.908 → 0.728 | 190,852 / 6,926 | 0.940 | 1.042 |
| S30 t087 | 1.803 → 1.680 | 1.378 → 1.308 | 81,996 / 5,000 | 1.015 | 0.995 |
| S37 t081 | 0.778 → 0.639 | 0.168 → 0.093 | 90,222 / 3,500 | 0.915 | unavailable |

Kernel-support invalid fraction decreased on all eight streams. All eight candidate RAFT measurements had zero fallback. The paired RAFT jitter geometric-mean ratio was
**1.0168** (candidate / baseline), with a one-sided 95% upper bound of **1.0608** (df=7). This exceeds the
**1.05 reference margin** used in the earlier backend challenge; that margin was not registered specifically for
this local-support test, so treat the comparison as exploratory rather than a formal gate. KLT path metrics were
available on six streams only: geometric-mean ratio **1.0278**, one-sided 95% upper bound **1.0669** (df=5),
with S17 at **1.121**. KLT was unavailable on S13 (4 baseline / 6 candidate failed pairs) and S37 (2 / 2).
KLT is incomplete and the adverse stream result is retained. The candidate's complete
machine-readable per-stream metrics are in ignored `out/bench/local_support_panel8_20261017/panel_eval.json`.

Gemini reviewed 16 blinded baseline/candidate clips, each with a position-reversed control. It returned six
candidate preferences and two ties in original position, then two baseline preferences and six ties after
reversal. The bias audit found one pair non-tied in both orders; its mapped preference flipped from candidate to
baseline, following screen side B. There were zero method-agreement pairs. Treat the visual review as
position-confounded and inconclusive.

Conclusion: the fixed candidate reliably repairs some sampler and center-mask support, but it does not show a
quality improvement and its motion metrics worsen on several streams. Keep it research-only; do not integrate
this map correction or change defaults. The 8-stream panel was previously inspected for backend work and is not
a newly sampled held-out corpus. The next border experiment should preserve the estimated source-coordinate map
and compare an explicit crop/zoom policy with border fill on new source windows; evaluate support, motion,
shape and visible artifacts separately.

## Minimal crop support screen (2026-10-17)

On the same development-only `00176.MTS` t0613 one-second window, I replayed the retained baseline plan and
swept output-side center zoom against exact production source coordinates and Lanczos-3 support. The first
tested zoom with full support on all 30 encoded output frames was **1.088** (increments of 0.002); 1.086 still
had a minimum support fraction of 0.999827. At 1.088, the output-side crop removes about **15.5% of frame
area**. This is a frame-specific support threshold, not a general per-clip policy.

Matched production runs used the same source hash, window, config and evaluator, with KLT enabled for both. The
baseline and 1.088 candidate both completed without fit fallbacks and retained 29/29 supported evaluation
frames. Full-support Lanczos-3 coverage changed from **0.97610 mean / 0.93883 minimum** to **1.0 / 1.0**.
RAFT jitter/wobble changed from **0.061504 / 0.030419 px** to **0.064008 / 0.032556 px** (**+4.1% / +7.0%**);
independent KLT jitter changed from **0.080803** to **0.078133 px** (**−3.3%**). The motion estimators therefore
do not show a common regression direction. Gemini 3.8 Flash via the Vertex API preferred the candidate in the
original pair and baseline in the position reversal; the audit found one side-following pair and zero method
agreement. Visual review is inconclusive.

Decision: this resolves that a minimal full-support zoom can reduce the measured RAFT jitter penalty relative
to fixed 1.10 on this one window, but it is still **not a selected policy**: the tuning and evaluation used the
same single window, the crop is substantial, and KLT/RAFT and Gemini do not give a consistent quality verdict.
Do not tune on the reserved S16/S25 windows. To pursue borders, prioritize preserved-framing fill against
source-coordinate maps on new source windows; revisit adaptive cropping only after a crop-area cap and a
development/held-out split are declared.

## Same-coordinate temporal fill screen (2026-10-17)

The existing `bench.temporal_border_fill_pilot` was run at radius 3 on the matched baseline output for the same
development-only t0613 window. Its bounded donor search filled **441,453 / 1,226,609 invalid pixels (36.0%)**,
equal to **0.71% of all output pixels**; 785,156 invalid pixels remained unresolved. The candidate kept the
baseline 1920x1080, 30-frame, 30000/1001 cadence. The implementation changes only invalid pixels and reads only
original neighboring frames/masks, as covered by its focused unit tests.

Gemini 3.8 Flash via Vertex preferred the filled candidate in the original blind pair, then returned a tie in
the position reversal. The audit therefore found no two-order method agreement. This is inconclusive one-model,
one-second visual evidence; it does not establish improved stabilization. Most border pixels remain unfilled at
this short radius, and a larger radius can increase temporal ghosting. Keep the pilot opt-in and research-only;
do not wire it into the production default.

Implementation direction: the nearest-valid same-output-coordinate rule is a useful baseline but cannot
distinguish a static background from a moving object or disocclusion. Before promoting temporal completion,
prototype donor alignment/confidence using temporal motion correspondences, reject uncertain or occluded donors,
and compare against this baseline on new source windows. Preserve the target frame's valid pixels and original
coordinate provenance, and report filled/unresolved area and motion/ghosting separately.

Research implementation. The pilot now has an opt-in max-bilateral-delta gate. It accepts a fill only when
both equidistant temporal donors are valid and their maximum per-channel RGB difference is within the supplied
threshold; a disagreeing nearest bilateral pair remains unresolved instead of falling back to a more distant
donor. The default policy is unchanged. On this same development window, radius 3 with thresholds 16 and 64
filled only **122 (0.010%)** and **496 (0.040%)** of invalid pixels, versus **36.0%** for the ungated pilot.
Those gated candidates were not sent for visual review because their filled area is negligible. This color
consistency test is not motion alignment and is not a selected setting; it shows that simple symmetric RGB
agreement is too conservative here to preserve much coverage.

## Crop plus scalar border backoff rejection and correlation-payload preflight (2026-10-17)

### Fixed 1.02 crop with global path backoff

The development-only `00176.MTS` t0613 captured plan was rendered with a fixed `crop_zoom=1.02` and one
clip-wide correction scale chosen to satisfy the existing 0.5% sampled border limit. The selected path scale was
0.255. The render kept mean center-invalid coverage at 0.497% (maximum 1.481%) and full-support Lanczos-3 at
99.01% mean / 98.14% minimum; crop area loss was 3.88%.

Matched evaluation used the raw source window, 12 RAFT iterations at 2x downscale, 30000/1001 fps, and the
independent seeded KLT estimator. Raw jitter was 1.1431 px RAFT / 1.1542 px KLT. The crop-plus-backoff output
measured 0.8490 / 0.8584 px, respectively, with RAFT wobble 0.03764 px. The ordinary stabilized baseline on
this window measured 0.06150 px RAFT jitter, 0.03042 px wobble and 0.08080 px KLT jitter. The backoff candidate
does improve on raw, but gives up most of the existing stabilization to preserve coverage. Reject this
crop-plus-scalar-backoff policy; no crop or path-scale default changes follow. This is a development-window
screen and does not establish a held-out crop policy.

### Opt-in bidirectional correlation-payload estimate and flow-scale resolver

`mlxstab.raft.model.estimate_corr_pyramid_bytes` now computes the stored payload for both RAFT correlation
pyramids from source dimensions, integer flow downscale, four-level floor pooling and correlation dtype.
`select_flow_downscale` chooses the highest flow resolution in a supplied candidate set that fits a byte budget.
The estimate was checked against actual small MLX `CorrBlock` pyramid arrays for both fp32 and fp16; their
summed `nbytes` matched exactly.

The stabilizer API and CLI now accept opt-in `flow_downscale="auto"` with a required positive
`max_flow_corr_bytes` budget. The auto choices are 1, 2 and 4; a fixed scale can also be rejected before RAFT
construction if its stored-payload estimate exceeds the supplied budget. `bench.runner` records the requested
configuration separately from the resolved scale in the manifest and experiment identity. Defaults remain
unchanged, and the memory scope is included in reports because this estimate excludes feature maps, activations,
temporaries and allocator/cache overhead; it is **not peak process memory or an OOM guarantee**.

For the implemented four-level shape, the estimate is 11,119,680,000 bytes at 1920x1080 fp32, 178,350,336,000
bytes at 3840x2160 fp32, 11,119,680,000 bytes at 4K with flow downscale 2, and 706,982,400 bytes at 4K with
downscale 4. These are payload calculations, not measured process-memory values and not evidence that the
selected flow resolution preserves output quality.

The initial 4K execution smoke used a generated 3-frame 3840x2160/30 fps source, one RAFT iteration, no mesh or
row residuals, `--flow-downscale auto --max-flow-corr-gib 8`, and the full-resolution renderer. Auto resolved to
4; the report recorded the 706,982,400-byte payload estimate; the encoded output verified as 3 frames at
3840x2160 and 30 fps with zero fit fallbacks. `/usr/bin/time -l` reported 717,176,832 bytes maximum resident
set, 4,514,875,456 bytes peak memory footprint and zero swaps. This is a short synthetic execution smoke, not a
quality comparison, a long-video test or an OOM guarantee on other workloads/devices.

Focused verification: `tests/test_raft.py`, `tests/test_stabilize.py` and `tests/test_bench_runner.py` passed
**153 tests**. The complete `.claude/verify.sh` passed **1,653 tests in 58.05 s**. CodeGraph sync/affected checks
are recorded with this implementation; its graph heuristic found no linked test files, so the focused tests were
run directly. The next gate is to measure peak MLX/process memory on additional safe reference sizes, then compare
flow-downscale quality against full-resolution analysis on known-motion and real windows before advertising an
automatic 4K preset.

## RAFT flow-downscale quality screen (2026-10-17)

The matched `00176.MTS` 1080i comparison used two non-overlapping one-second windows, at 60 s and 800 s. For
each window, scale 2 and scale 4 used the same source hash, 60 deinterlaced frames at 59.940 fps, baseline
planner and renderer settings, 20 analysis iterations, output cadence, 12-iteration scale-2 RAFT evaluator,
and seeded KLT evaluator. Both scales completed with aligned timelines, 59 supported motion pairs, zero fit
fallbacks, and reliable RAFT/KLT agreement that stabilization reduced measured jitter. These are two scenes
from one source recording, not independent source clips or held-out validation.

| Window | Flow scale | Stabilization (s) | Output RAFT jitter (px) | Output RAFT wobble (px) | Output KLT jitter (px) | Mean / max border fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t060 | 2 | 50.74 | 0.02028 | 0.02441 | 0.04580 | 0.00498 / 0.01920 |
| t060 | 4 | 19.94 | 0.04674 | 0.03839 | 0.06557 | 0.00568 / 0.02179 |
| t800 | 2 | 49.80 | 0.02739 | 0.01968 | 0.06915 | 0.00613 / 0.01257 |
| t800 | 4 | 19.81 | 0.05484 | 0.03831 | 0.08860 | 0.00692 / 0.01315 |

Scale 4 was about **2.5x faster** than scale 2 in both windows, while its output RAFT jitter was **130% higher**
at t060 and **100% higher** at t800. RAFT wobble was 57% and 95% higher; KLT jitter was 43% and 28% higher.
Scale 4 still improved jitter relative to raw in both windows. The absolute scale-4-minus-scale-2 residuals
were 0.0265/0.0275 px for RAFT jitter, 0.0140/0.0186 px for wobble, and 0.0198/0.0194 px for KLT jitter
(t060/t800). The relative increases are large because the scale-2 residuals are small; these single-run
summaries provide no variance estimate or confidence interval. Mean and maximum border fractions also rose
slightly at scale 4 in both windows.

Gemini 3.8 Flash via the Vertex API reviewed the scale-2/scale-4 pair and its position reversal for each
window. At t060 it called the original a tie and preferred scale 2 in the reversal; at t800 it preferred
scale 4 in the original and called the reversal a tie. The position-bias audit found **zero two-order method
agreements across two complete pairs**. Treat the visual review as inconclusive; it does not overturn or confirm
the motion metrics.

Two scale-1 attempts on t060 were stopped before output: an exploratory three-second run remained in analysis
for roughly eight minutes, and a one-second run was cancelled after **648 s (10m48s)** while still in
stabilization. They provide no quality result. At 1080p the estimator reports an 11,119,680,000-byte stored
bidirectional correlation-pyramid payload at scale 1, versus 706,982,400 bytes at scale 2 and 44,031,360 bytes
at scale 4. The payload estimate is not peak memory.

Decision: these two windows reject a claim of **scale-4 quality parity with scale 2** for this source, while
showing a substantial runtime gain. Do not treat this as a general scale-4 quality ranking: the absolute
differences are small, there is no run variance, the scenes are from one source, and both are 1080i rather than
4K. Keep `flow_downscale=auto` opt-in, report its resolved scale, and do not advertise a production 4K quality
preset. Next compare scales 2 and 4 on a distinct source and longer real windows; separately validate quality
on actual 4K footage before setting an acceptance threshold or changing auto-selection behavior. Manifests,
blinded pairs, Gemini responses and the positional-bias audit remain under ignored `out/bench/`.

### Distinct-source ten-second follow-up on `00177.MTS` (2026-10-17)

The next matched screen used `00177.MTS` at 50–60 s, a separate 10 s low-light dance scene with no observed cut.
Both runs used native 59.940 fps output, the same baseline configuration except stabilization `flow_downscale`,
20 stabilization iterations, and a 12-iteration RAFT scale-2 evaluator. Each output had 600 aligned frames and
599 supported consecutive pairs. The source SHA-256 is
`8c5d98d15448ad0557537ba9d02a28fc426185b4e32bdf1be3eb4964fdbea714`. Scale 2 took 537.7 s to stabilize;
scale 4 took 213.6 s (2.52x faster).

| Flow scale | Output RAFT jitter (px) | Output RAFT wobble (px) | Fit fallbacks | Mean / max invalid-mask fraction |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.15611 | 0.05096 | 0 / 599 | 0.00637 / 0.03033 |
| 4 | 0.89237 | 0.07570 | 0 / 599 | 0.02293 / 0.09388 |

The input measured 0.26891 px jitter and 0.05352 px wobble. On this window, scale 2 reduced jitter by 42%; scale
4's jitter was 3.32x the raw input and 5.72x the scale-2 output. Scale 4 wobble was 41% above the raw input and
49% above scale 2. The independent KLT metric was unavailable: two input pairs and two scale-2 / three scale-4
output pairs failed track support, so it cannot validate the RAFT direction here.

To localize the difference without changing metric definitions, `bench/flow_scale_localization.py` re-evaluated
the two saved outputs with the same fixed RAFT evaluator. It first checked manifest provenance, then confirmed
whole-window jitter, wobble and fallback counts against the original metrics (jitter absolute error below
4e-16 px; wobble identical; fallback counts identical). The 599 fitted pairs were split into ten contiguous
approximately one-second blocks **after** applying the existing 15-frame high-pass across the full clip; there
was no per-block refit or edge-specific smoothing. Scale-4 jitter exceeded scale 2 in all ten blocks:

| Window (s) | Scale 2 jitter (px) | Scale 4 jitter (px) | Scale 2 wobble (px) | Scale 4 wobble (px) | Scale 2 / 4 FB-valid fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–1 | 0.201 | 1.957 | 0.074 | 0.121 | 0.871 / 0.827 |
| 1–2 | 0.318 | 2.228 | 0.081 | 0.155 | 0.893 / 0.832 |
| 2–3 | 0.120 | 0.501 | 0.032 | 0.042 | 0.960 / 0.941 |
| 3–4 | 0.084 | 0.530 | 0.031 | 0.036 | 0.964 / 0.942 |
| 4–5 | 0.085 | 0.526 | 0.031 | 0.033 | 0.970 / 0.962 |
| 5–6 | 0.093 | 0.436 | 0.034 | 0.037 | 0.958 / 0.960 |
| 6–7 | 0.108 | 0.762 | 0.039 | 0.049 | 0.948 / 0.938 |
| 7–8 | 0.187 | 0.884 | 0.063 | 0.109 | 0.908 / 0.901 |
| 8–9 | 0.224 | 0.603 | 0.077 | 0.095 | 0.952 / 0.932 |
| 9–10 | 0.141 | 0.514 | 0.048 | 0.081 | 0.961 / 0.947 |

This is a clip-wide regression with a particularly large first two seconds, not a single isolated failure. Both
scales had zero fit fallbacks throughout, so fallback count would not detect it. The lower scale-4
forward/backward-valid fraction in most blocks is diagnostic context only, not a calibrated error measure. Gemini
3.8 Flash's original-order judgment was a tie and the position-reversed judgment preferred scale 2; one complete
pair is insufficient to claim a visual preference, so the visual review remains inconclusive. This single
window localizes a failure but does not establish its prevalence across recordings. The machine-readable blocks,
replay verification and video artifacts are retained under ignored `out/bench/downscale_quality_00177_t050_`*.

Decision: treat scale 4 as a runtime/memory option with an observed quality risk, not as a generally acceptable
quality/runtime tradeoff. Do not implement a gate based on fallback count. Before choosing a gate, test a
candidate independent signal (such as flow consistency or scale-2/scale-4 disagreement) on additional scenes
and measure whether it flags actual output regressions without rejecting stable scale-4 windows. Keep the auto
resolver's current opt-in behavior and do not claim 4K quality parity.

### Input-side scale-4 fit-confidence screen (2026-10-17)

To check whether a signal already available during scale-4 analysis might flag the observed regression, an
offline replay captured per-pair fit and forward/backward diagnostics on the raw inputs. The new
`bench/flow_scale_fit_diagnostics.py` reports one-second RANSAC inlier-fraction quantiles, fallback events,
valid sample counts, mover fraction, and cycle-consistency summaries. It does not change stabilization or define
a threshold. The source frame decode, RAFT settings, config and source hash come from the matched manifests;
downstream quality deltas are recorded from the existing fixed scale-2 evaluator results.

| Window | Inlier fraction p10 | Minimum inlier fraction | Mean FB-valid fraction | Fallback pairs | Output Δ jitter (px) | Output Δ wobble (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00176 t060, 1 s | 0.950 | 0.933 | 0.991 | 0 | +0.0265 | +0.0140 |
| 00176 t800, 1 s | 1.000 | 1.000 | 0.999 | 0 | +0.0275 | +0.0186 |

On `00177.MTS` t050–060, scale-4 inlier-fraction p10 was 0.831 and 0.915 in the first two seconds, where
the scale-4-minus-scale-2 output jitter differences were +1.756 and +1.910 px. Later blocks had p10 at or above
0.967 and smaller jitter differences of +0.343 to +0.698 px. Minimum inlier fraction also dipped in some later
blocks, so the minimum alone was not a clean separator. All 10 blocks had zero fallback pairs. The two
`00176.MTS` windows had much higher inlier fractions and smaller penalties. The FB-valid fraction was also much
lower throughout the `00177` window (0.806–0.913) than in the `00176` windows, so inlier p10 has not been
separated from general texture, occlusion, or motion difficulty. Ten adjacent blocks are correlated and are one
clip, not 10 independent validations.

Decision: retain inlier fraction p10 as an exploratory covariate, not a quality gate. The evidence says it may
identify the two worst seconds in this clip, but does not show that it uniquely predicts scale-4 harm or that
any cutoff generalizes. The next useful check is a controlled synthetic scale-2/scale-4 comparison with known
camera motion and varied texture/blur, plus additional held-out real windows if new footage becomes available.
Compare the fit signal to known-truth stabilization error while stratifying by input difficulty; report the full
series and do not tune a threshold on these three windows. Diagnostic artifacts remain under ignored
`out/bench/*fit_diagnostics.json`.

### Known-translation scale-4 flow sweep (2026-10-17)

Following DeepSeek's recommendation, `bench/flow_scale_synthetic_screen.py` tested a paired flow-estimation
screen before attempting full stabilization. It generated 1080p static textured scenes with exact horizontal
translation, then evaluated RAFT at scales 2 and 4 on the same sequences. Factors were three texture seeds,
four appearances (sharp, 1.5 px blur, 4 px blur, and low contrast), and shifts of 4, 8, 16, and 32 source pixels
per pair. Each condition had 30 frames and 29 fitted pairs. The primary endpoint was dense-flow endpoint error
against the exact translation, excluding a 64 px image border.

| Shift (px/pair) | Median scale-2 EPE (px) | Median scale-4 EPE (px) | Median scale-4 minus scale-2 EPE (px) | Scale-4 inlier p10 range |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.091 | 0.211 | +0.118 | 1.000–1.000 |
| 8 | 0.070 | 0.244 | +0.156 | 1.000–1.000 |
| 16 | 0.083 | 0.183 | +0.098 | 1.000–1.000 |
| 32 | 0.068 | 0.151 | +0.085 | 1.000–1.000 |

Scale 4 had higher mean EPE than scale 2 in all 48 condition/seed/displacement cells. However, scale-4 inlier
p10 was exactly 1.0 in all cells, and forward/backward validity stayed approximately 0.996. The inlier statistic
therefore cannot flag even the measured scale-4 flow error under these known-translation conditions. A
texture-seed-clustered fixed-effect fit estimated the scale-4-minus-scale-2 EPE change at -0.00204 px per extra
pixel of displacement (cluster bootstrap interval -0.00249 to -0.00161). This is descriptive for this synthetic
generator only; the lower relative penalty at larger shifts is not a real-scene claim. No p-value or threshold
was used. The original single-shift screen and full 48-cell result are retained under ignored `out/bench/`.

Decision: drop inlier-fraction p10 as a scale-selection gate candidate. Its association with the two worst real
seconds did not replicate in the controlled flow screen, and the signal had no dynamic range. This does not
invalidate the original real-footage quality result: scale 4 still had higher flow error in every synthetic
cell. The first full-stabilization path controls are recorded below; they confirm a scale-4 high-frequency
penalty in one far-only scene but do not yet test parallax or moving subjects. Keep the fit statistic diagnostic
only. Do not change auto selection or claim a calibrated confidence measure.

### Full-stabilization synthetic path controls (2026-10-17)

`bench/flow_scale_synthetic_path_audit.py` maps a fixed grid of output pixels through each saved plan's full
`source_coords` field, then through the synthetic source's exact camera transform, and compares those world
coordinates with the intended camera path. It requires matched run manifests, plans, cadence, and source hash;
the paired runs differed only in stabilization flow downscale and used the same fixed scale-2 output evaluator.
The audit separates global bias, temporal path residual, linear drift, and high/low-frequency velocity error.
The audit is a geometric synthetic metric, not a perceptual score.

On a 60-frame 1080p far-layer-only seed-13 scene with rolling shutter enabled (`rho=0.8`), scale 2 / scale 4
high-pass path-velocity error RMS was 0.445 / 0.503 px per frame (mean absolute error 0.166 / 0.270); the
fixed evaluator reported jitter 0.078 / 0.201 px. In a matched global-shutter control (`rho=0`), the
corresponding RMS values were 0.066 / 0.201 px per frame (mean absolute error 0.052 / 0.164), and evaluator
jitter was 0.029 / 0.173 px. Thus scale 4 was worse on these high-frequency endpoints in both versions of
this one generated scene. Both scales also accumulated roughly 59 px of linear path drift over the clip in
each control, while raw input drift was only a few pixels. That shared drift is a low-frequency path-fidelity
warning, but it does not identify the smoother as its cause: target semantics, trajectory mapping, integration
bias, and boundary effects remain possible. Fit-inlier p10 was 1.0 and there were no fallback pairs, so those
signals again did not expose the scale-4 difference.

This is one seed and one far-only scene, without parallax or a moving subject; neither result generalizes to
real footage or identifies an acceptable scale. The exact-motion follow-up in `bench/path_smoother_gain_audit.py`
isolates the planner smoother from optical flow. With exact static and signed constant-pan Fits, `gauss` with
`sigma=0` preserved the path exactly and static-camera drift was zero. The production L1 settings
(`budget=0.04`, `fidelity=0.1`) held both +0.5 and -0.5 px/frame ramps stationary across 60 frames, producing
29.5 px endpoint error despite exact motion inputs. This reproduces the direction and order of magnitude of the
earlier full-stabilization drift without flow, so the smoother is a direct cause for linear pan suppression in
that control.

An exact-Fit L1 fidelity screen with a 0.5 px/frame intended pan plus 6 px-std low-frequency camera jitter,
five noise seeds and both pan directions exposed a tradeoff. Median path gain at fidelity 0.1 was zero in both
directions; at fidelity 1 it was 0.42 / 0.54, at fidelity 3 it was 0.88 / 0.85, and at fidelity 10 it was
0.96 / 0.98 (positive / negative pan). Median high-pass velocity error at fidelity 3 was 0.027 / 0.033
px/frame versus 5.00 px/frame in the raw path; at fidelity 10 it rose to 0.167 / 0.169. This is planner-only
evidence from five synthetic noise traces, not rendered quality or a default-selection result. The gain is
directionally symmetric in the medians, but individual seeds vary; path gain alone also cannot rank visual
quality.

The matched full RAFT scale-2 rerun at fidelity 3 used the same 60-frame global-shutter far-only seed-13 source,
analysis/evaluation settings and renderer as the existing fidelity-0.1 run; only `Config.fidelity` changed.
There were zero fit fallbacks in either run. The exact camera-path audit improved total path RMSE from 18.06 to
3.93 px, and clip x-drift error from -59.84 to -5.14 px. Fixed-evaluator jitter rose from 0.0294 to 0.1977 px
(6.7x), while raw-input jitter was 6.72 px; both settings reduced jitter substantially relative to raw. Wobble
was 0.01183 / 0.01271 px. Stabilization time was 49.92 / 49.39 s (fidelity 0.1 / 3), so the path-fidelity
change had no material runtime cost in this run.

Gemini 3.8 Flash via Vertex reviewed the paired outputs in original and position-reversed order. It returned
ties in both, explicitly saying the noise-texture scene lacked recognizable structure for visual stabilization
assessment. Treat that review as inconclusive, not as evidence that the outputs are equivalent. The exact-Fit
five-seed screen and this rendered run agree that higher fidelity can preserve intended pan while retaining
strong jitter reduction, but this is still one rendered source and one pan profile.

The follow-up on natural footage used `00177.MTS` at 50–60 s, the same 600-frame low-light dance window as the
earlier scale screen. Scale 2, source hash, cadence, evaluation settings, and every other config field matched;
only L1 fidelity changed from 0.1 to 3. Both runs had zero flow-fit fallbacks. Raw jitter/wobble were
0.26891 / 0.05352 px; fidelity 0.1 output was 0.15611 / 0.05096 px, and fidelity 3 was 0.17096 / 0.05343 px.
Fidelity 3 therefore raised jitter 9.5% and wobble 4.8% relative to 0.1, while still reducing jitter 36% from
raw and leaving wobble essentially at raw. KLT path metrics remained unavailable in both outputs (two failed
pairs each). Stabilization time was 537.7 s / 531.3 s (0.1 / 3), with no material time difference.

Gemini 3.8 Flash via Vertex reviewed the position-balanced blind A/B pair and called tie in both orders. It
reported no visible stabilization difference and cited darkness, blur, and rolling-shutter distortion as
limitations. Treat this as inconclusive on subtle differences. The natural clip provides no measured benefit
for fidelity 3; the synthetic exact-path result remains the only evidence that it preserves known pan better.
Do not promote fidelity 3 or change the production default. Keep 0.1 provisional, not validated. The next gate
is a multi-clip paired fidelity sweep with clips that have reliable independent tracking, plus exact-motion
tests for multiple pan speeds and sinusoidal intentional motion. Evaluate path retention together with jitter,
wobble, crop, and distortion; use frame-block bootstrap intervals and blind visual review as secondary evidence.
Keep scale selection unchanged and treat scale 4 as an observed quality-risk option. Reports, comparisons and
Gemini responses remain under ignored `out/bench/`.

### Path-retention follow-up: sinusoidal intent and Gaussian smoothing (2026-10-17)

The exact-motion audit now includes known sinusoidal camera paths. It fits output amplitude and phase at the
injected frequency after trimming boundary transients; phase is marked unidentifiable when the output
amplitude is effectively zero. This avoids treating arbitrary phase from a fully suppressed signal as
evidence. Across 480-frame controls with a 20 px sinusoid and 6 px smooth path jitter, L1 fidelity 0.1
suppressed the tested 60-, 120-, and 240-frame components (gain 0). Fidelity 3 retained gains of 0.87,
0.98, and 0.91 respectively, but had already performed poorly on natural KLT-supported jitter metrics in
the prior tests. Thus the same scalar knob does not resolve the conflict between suppressing wobble and
preserving deliberate movement.

The existing Gaussian path smoother was screened as an alternative. On exact controls, sigma 10 retained
about 0.59 / 0.91 / 0.93 amplitude at periods 60 / 120 / 240 frames, with high-frequency path error lower
than narrow kernels. On the two matched 1-second KLT-supported windows from `00176.MTS` (t060 and t800),
sigma 10 remained between the default L1 fidelity 0.1 and the less-stable L1 fidelity 2–3 candidates on
measured jitter, but was substantially worse than the default: RAFT jitter was 0.0638 / 0.1641 px versus
0.0203 / 0.0274 px for L1 0.1. Wobble was 0.0621 / 0.0277 px versus 0.0244 / 0.0197 px. KLT jitter was
0.0718 / 0.1889 px versus 0.0458 / 0.0692 px. All runs had complete KLT support. These short windows do
not provide ground truth for which low-frequency motion should be preserved.

Gemini 3.8 Flash via Vertex reviewed each Gaussian/default pair in both panel orders. For both windows it
preferred the Gaussian candidate in the original order, then preferred the default when panel positions
were reversed. That is a clear position-order inconsistency, so the visual review is inconclusive despite
describing less high-frequency jitter in each original-order call. It does not override the fixed metrics.
Do not change the default or promote Gaussian sigma 10. Keep it as a measured comparison point. The next
useful experiment is a frequency-aware path objective (separate intentional low-frequency motion from
high-frequency disturbance), first against exact multiscale motion profiles and then on disjoint natural
windows with reliable independent tracking. Compare at matched high-frequency suppression and report path
retention, jitter, wobble, crop, distortion, and uncertainty together; do not select a smoother from a single
aggregate score.

### Frequency-selective path candidate: exact controls and real-window rejection (2026-10-17)

An opt-in `butter` smoother was added for research comparison. It applies a fourth-order Butterworth low-pass
in both time directions to each path column; the realized magnitude response is therefore an effective
8th-order rolloff, with -6 dB at the configured single-pass -3 dB cutoff period. This is offline and has
boundary transients, so the exact-path audit trims two cutoff periods at each end and requires two complete
sinusoid cycles. It is not an intent estimator, is not exposed as the application default, and leaves L1
fidelity 0.1 unchanged.

The 99-case exact-motion screen swept three seeds, pan amplitudes 5/20/40 px, periods 60/120/240 frames,
static camera shake with temporal correlation 0.8 or 60 frames, and L1, Gaussian sigma 10, and Butterworth
cutoff period 45. Butterworth retained roughly 0.90x / 1.01x / 0.98x amplitude at periods 60/120/240 for
20 px pans; gains varied modestly with seed and amplitude. Gaussian retained about 0.58x / 0.88x / 0.95x.
On static fast-shake controls, median output path RMSE was 3.45 px for Butterworth and 3.01 px for Gaussian.
When static shake was slow (60-frame temporal correlation), both retained substantial motion (median output
path RMSE about 11.7 px). This is the expected counterexample to treating low frequency as intent. Reports
are in ignored `out/bench/path_frequency_candidate_screen.json` and
`out/bench/path_frequency_multiseed_screen.json`; the single-seed cutoff sweep is exploratory, not a tuned
or confirmatory result.

The matched real renders changed only smoother mode/parameter (`butter`, period 45). Both were complete,
with 60 aligned output/mask frames and zero flow-fit fallbacks. On the KLT-supported 00176.MTS t060 window,
RAFT/KLT jitter was 0.1025/0.1019 px and wobble 0.0664 px. On t800 it was 0.1865/0.2118 px and wobble
0.0278 px. In the same windows, L1 fidelity 0.1 measured 0.0203/0.0458 and 0.0274/0.0692 px jitter;
Gaussian sigma 10 measured 0.0638/0.0718 and 0.1641/0.1889 px. Butterworth therefore did not improve the
real jitter tradeoff over Gaussian and was much worse than L1, even though its mean border fractions were
slightly lower. Do not promote this smoother or change defaults.

Gemini 3.8 Flash via Vertex reviewed the Butterworth/default pairs in both panel orders. It preferred the
candidate in both t800 orders, but its t060 preference flipped with panel position. Treat the visual result as
mixed and secondary to the measured real-window jitter regression. DeepSeek Flash via Pi advised that the
exact controls need intent labels, pan-amplitude/SNR variation, cutoff sweeps across the intended band, fair
budget matching, and boundary treatment. These checks were added; DeepSeek then recommended testing whether
measured-path outliers and robust trajectory estimation explain the gap between sinusoidal controls and real
footage.

Next, replay independent KLT-estimated real trajectories through the same planner with L1, Gaussian, and
Butterworth, then perturb known trajectories with controlled outliers. Measure tracking residual kurtosis and
outlier fraction against path retention, jitter and crop. If those cues explain the smoother gap, test the
hypothesis on held-out windows from 00177/00178; otherwise return to a motion-intent cue that combines
subject-relative scene motion and uncertainty. Frequency remains a useful descriptor, not a synonym for
intent.

### Fixed-trajectory replay and controlled outlier screen (2026-10-17)

`bench/path_trajectory_replay.py` now reuses one `Motion` estimate for all smoother arms. Its synthetic arm
starts from known static or sinusoidal camera motion, injects pair-fit translation errors at predeclared rates,
and runs the resulting transforms through production `plan()`. The 33-case screen uses three seeds, static
paths at 0/2/5/10% outlier pairs and 12/36 px errors, plus 120-frame intended sinusoids with 6 px jitter and
0/2/5% 24 px fit outliers. On static paths with injected outliers, median output error ratios (smoothed path
RMSE / corrupted path RMSE) at 2/5/10% were 0.70/0.54/0.54 for L1 fidelity 0.1, 0.96/0.99/0.98 for
Gaussian sigma 10, and 0.97/0.99/0.98 for Butterworth period 45. This confirms that the L1 path objective
rejects these large isolated errors better. When an intended sinusoid is present, the same L1 robustness can
also remove wanted movement; the known-path reference and outlier-error ratio must therefore be read together.

The real arm decodes one window, tracks it once with KLT, and replays the same resulting `Motion` through all
three planners. On `00176.MTS` t060/t800, KLT fallback fractions were zero and mean inlier fractions were
0.921/0.997. The diagnostic temporal residual outlier fraction was zero in both, with excess kurtosis
-0.18/-1.04. That statistic is exploratory and uncalibrated, but these windows do not support the hypothesis
that isolated KLT trajectory outliers explain the Butterworth regression. Relative to the raw KLT path, the
planner correction RMSE was L1 7.81/11.47 px, Gaussian 1.24/3.47 px, and Butterworth 1.29/3.68 px. This
shows the candidate smoothers produce similar KLT paths and L1 produces a much stronger correction; it is
descriptive only, not a quality ranking or ground-truth measurement. Both replay artifacts are under ignored
`out/bench/path_trajectory_real_t060.json` and `path_trajectory_real_t800.json`; the synthetic report is
`out/bench/path_trajectory_synthetic_outlier_screen.json`.

DeepSeek Flash via Pi recommends the next causal screen: a paired-seed synthetic factorial that varies
subject-relative motion, depth/parallax separation, and trajectory predictability independently while holding
camera shake fixed. Use known camera/subject/depth truth; score camera-motion retention, shake rejection, and
outlier response separately. Existing `mlxstab.synth` already provides layered parallax, an independently
moving subject, and smooth camera motion, so first extend it with controlled predictability/visibility arms and
compare estimator signals before adding a learned intent classifier. Keep real 00176/00177 clips held out for
subsequent external validation; the current replay does not justify any smoother or production-default change.

### Synthetic intent-factor controls added (2026-10-17)

The procedural scene generator now supports a paired predictable (`smooth`) versus correlated stochastic camera-intent
trajectory, with an independent RNG stream so camera shake remains identical for a seed across intent modes. The
existing smooth mode stays the default and retains its historical camera formula. `render()` also exposes subject
world-speed and validates three positive layer motion factors; its existing arguments, defaults and return shape are
unchanged. These are stimulus controls only; no factorial flow/stabilization result has been run yet. DeepSeek
recommended holding the far-layer factor at 1.0 while varying only mid/near factors so parallax contrast is not
confounded with global image speed. Next expose these controls through the benchmark runner and provenance manifest,
then run paired seeds over intent predictability, depth separation and subject absent/static/moving. Keep all cells at
the same camera-shake seed and record exact intended/actual paths.

### Runner integration smoke test (2026-10-17)

`bench/run_case.py` now exposes the synthetic intent, layer factors, subject mode/speed, shake amplitude,
rotation shake and temporal correlation; `bench/runner.py` validates and records these settings along with the
exact intended and generated camera paths and shake RMS. Synthetic manifest windows are explicit, allowing the
existing `bench.real_pair_review` and `bench.gemini_blind_review` path to review generated sources after hash
verification. Focused validation passed 107 tests across runner, source-review and synthetic-control tests.

Two seed-17 smoke runs completed at 24 frames and 160x96, with layer factors far/mid/near 1/.85/.7, a moving
subject, and matching shake settings. They differed only in smooth versus stochastic camera-intent mode. Both
source windows and truth arrays are recorded in their manifests under ignored
`out/bench/synth_intent_factorial_seed17/*_windowfix/`. Raw-versus-stabilized, position-reversed Gemini review
clips were built and submitted. This tiny pair checks plumbing only: it is not a multi-seed factorial, does not
compare stabilization methods, and cannot establish whether visual content cues improve stabilization. Record
Gemini's per-clip judgments when returned, but keep objective metrics and known camera truth primary.

DeepSeek's design review flags the current `smooth`/`stochastic` label as insufficient: intent regimes need
matched RMS velocity/acceleration and predeclared frequency bands, otherwise the stochastic arm may simply be
more shake-like or energetic. It recommends a 2x2x3 paired-seed screen: path regime (smooth/stochastic), scene
layers (far-only/far+mid-near parallax), and subject (absent/world-static/world-moving). Hold a distinct shake
trace fixed across all cells for each seed; use identical intent realizations for layer/subject comparisons,
but matched energy rather than an identical path across intent regimes. Keep the far layer rigid and camera-locked
as ground-truth camera reference. Use 8 seeds as a screening floor (12 preferred), at least 64 frames and
320x192, with a separate small resolution/length sensitivity check. Add clean and hard calibration anchors for
intent-only, shake-only/static-camera, both, and neither.

Measure intended-path gain/correlation/RMSE and endpoint drift in the intent band; shake RMS/PSD attenuation in
the shake band; per-layer/subject distortion, non-rigidity, fold-over and common-valid-mask crop; and predeclared
failure events with all-run as well as complete-case reporting. Plot retention and rejection jointly and include
identity/freeze reference baselines rather than collapsing results into one score. Freeze estimator config before
the factorial. This screen is not yet implemented. The Gemini API report is partial (two of four panel-order
judgments); the next request stalled beyond two minutes and was interrupted, leaving the resumable partial JSON
under ignored `out/bench/synth_intent_factorial_seed17/gemini_review.json`. Both predictable-scene judgments
were ties, citing no clearly discernible motion/difference at this resolution and duration. Treat those as
evidence that the smoke stimuli are too weak for visual adjudication, not evidence that stabilization has no effect.

### Matched intent and subject-transform controls (2026-10-17)

DeepSeek's construction review recommends a shared Fourier envelope with phase-only variation: equal spectral
support and amplitudes, with one random phase per frequency shared by x/y/rotation. This preserves the cross-spectrum
as well as marginal power; independently randomizing camera axes would create a different joint motion process.
`synth.camera(..., intent_profile="matched-spectrum")` now implements that paired control while the historical
smooth path remains the default. The paired path includes a shared x drift and uses deterministic coherent phases
for `smooth`, seeded randomized phases for `stochastic`. A `static` intent mode provides zero-motion calibration
clips. Tests verify equal FFT magnitudes, cross-spectra and spectral derivative RMS of the periodic component,
distinct waveforms, seed reproducibility and identical shake. The matched profile requires at least 16 frames.

This phase-only pair is specifically a **phase-coherence** contrast. The common non-periodic x drift is identical
across arms; compare the periodic component when measuring matched displacement/velocity/acceleration energy.
Because it preserves second-order statistics, a linear time-invariant/linear-MSE predictor should perform
similarly on both arms; this is a useful null, not proof of general predictability. Test phase-sensitive or
nonlinear cues separately before calling the contrast a general camera-intent predictor. DeepSeek also recommends
not combining px and degrees without an image-space rotation scale.

The renderer's subject had previously been overlaid in image coordinates, so the supposedly world-static subject
did not respond to camera translation. It now uses the same inverse camera transform as the near scene layer.
Focused synth, runner and stabilizer tests passed (79 passed at that checkpoint). A 96-frame 320x192 seed-17 matched-spectrum pair
completed with identical shake and zero output fallbacks. Whole-frame jitter diagnostics were input/output
1.76/0.16 px for coherent phase and 1.94/0.33 px for randomized phase; these are not a method comparison or
content-understanding result because whole-frame flow mixes camera, parallax and subject motion. Longer clips
are under ignored `out/bench/synth_matched_seed17/`; blinded review artifacts and the completed Gemini result
are in that ignored run directory. Gemini preferred raw over stabilized in all four original/reversed panel
judgments, citing shifting black edge gaps in the stabilized panel while noting slightly lower interior movement.
This is consistent across panel order but is only a visual tradeoff judgment on a short synthetic clip; it
reinforces that crop/border coverage must be reported with motion retention and shake rejection.

### Truth-aware synthetic stabilization measures (2026-10-17)

The renderer can now return a per-frame visible-layer ID map. The runner saves it as a compressed, hash-verified
`synthetic_truth.npz` sidecar in each synthetic run directory; scene generation guarantees that the near layer
intersects the camera window while preserving randomized additional blobs. `bench.synth_intent_metrics` loads a
successful manifest and saved plan, composes the production output-to-input map with the exact row-time input
camera-to-world transform, then robustly fits output-to-world similarity using only known visible-layer samples.
It reports the effective far-layer camera path, band-limited intent gain/correlation/RMSE, shake-band attenuation,
map-fit distortion, visible support, input-valid/crop fraction and unavailable frames. Translation and rotation
remain separate units. It refuses plan/truth dimension mismatches and checks the truth sidecar hash.

The current truth-aware 24-frame integration smoke (`out/bench/synth_metrics_smoke_v4/`) recovered all 24 frames
for far, mid, near and subject layers. Far visible support averaged 44.5%, output-map fit RMSE 0.75 px, and
input-map valid coverage 93.3%. The report correctly flags that the run is below the recommended 64-frame
minimum and raises the shake-band boundary to keep it disjoint from intent harmonics. These short-run metric
values are plumbing checks only, not evidence for selecting or tuning stabilization. A smaller, frame-relative
near anchor also restored the full-suite no-fallback flow fixture while keeping near-layer visibility.

The next experiment is the predeclared sequential 2x2x3 screen with at least eight paired seeds, calibration
anchors, frozen configuration, and explicit failure rows. Keep path retention, shake rejection, layer distortion,
crop, and failure rate separate. Include a small resolution/frame-length sensitivity check after the main screen.
The 96-frame smoke and Gemini border judgment do not substitute for the factorial.

### Factorial execution harness (2026-10-17)

`bench.synth_intent_factorial` now builds the fixed 12-cell intent x layer-motion x subject design plus per-seed
intent-only, shake-only and neither anchors. It pairs scene, intent and shake seeds across cells, freezes baseline
stabilizer/evaluator settings, hashes relevant source files into the experiment identity, and runs cells
sequentially. Each cell writes a standard manifest, production plan, truth sidecar, and separate truth-aware metric
report. Atomic `screen.json` checkpoints retain failed rows and resume only matching successful provenance; an
interrupted metrics step reuses the already successful run. The dry-run resolved 15 cells per seed / 120 runs for
the default eight seeds. The one-seed pilot (`out/bench/synth_intent_pilot_seed17/`) completed all 15 cells with
zero failures, all 96 frames available for every layer, far-layer visibility about 0.63–0.69, input-valid
coverage above 0.94, and camera-fit RMSE below 1.1 px. In this seed, stochastic-path cases retained more injected
rotational intent (gain 0.25–0.33 versus 0.00–0.07 for smooth cases). Its horizontal shake residual was -12.3 dB
for smooth versus -1.4 dB for stochastic. The metric is `20*log10(output residual RMS / input shake RMS)`, so
negative values mean attenuation and more-negative values mean stronger rejection. The full screen completed:
120/120 cells succeeded across seeds 0–7, with no missing or duplicate main cells and no failed runs. The
stochastic-minus-smooth x-intent RMSE contrast is +1.066 px (7/8 seeds positive; seed-level t interval
[0.048, 2.085], percentile-bootstrap interval [0.205, 1.715]). This is a modest synthetic screening signal; the
t interval is close to zero, and no multiplicity adjustment was predeclared, so do not call it confirmatory.
Y-intent RMSE is +0.184 px [-0.148, 0.517] and rotation-intent RMSE is +0.016 deg [-0.088, 0.121]. Shake
residual dB is +3.915 x [-3.771, 11.602], +1.128 y [-2.947, 5.203], and +0.373 rotation [-2.201, 2.947];
lower values mean stronger rejection. All shake intervals cross zero. The tested phase-by-layer, phase-by-subject
and three-way interaction intervals also cross zero. Across 96 main cells, all far-layer poses are available;
visibility ranges 0.446–0.716, input-valid coverage 0.938–0.959 and pose-fit RMSE 0.493–1.805 px. The analysis
JSON is `out/bench/synth_intent_factorial_8seed/analysis.json`. Calibration anchors behave distinctly:
shake-only removes the injected shake, while intent-only retains the shared drift and low-frequency pan.

`bench.synth_intent_analysis` treats seed as the experimental unit, averages the six main cells per intent mode,
and summarizes paired stochastic-minus-smooth contrasts with per-seed signs and a seed-level Student-t interval;
the percentile bootstrap interval is retained as a sensitivity check because it is coarse at small n.
It excludes calibration anchors and reports layer/subject strata, each cell's phase contrast, and phase-by-layer,
phase-by-subject and three-way difference-in-differences. Lower intent RMSE is better; lower (more negative)
shake residual dB means stronger rejection. DeepSeek code review caught a count-only completeness check that could
have allowed a duplicate cell to stand in for a missing one; the analyzer now validates the exact designed cell
set per mode and lists configured seeds with no main cells. A regression test covers duplicate cells. The
synthetic screen does not justify an intent-aware model change yet. DeepSeek recommends real footage next. Reuse
the established 10-second, fidelity-0.1 baseline on `00176.MTS` at t060 and t800 as two paired raw/stabilized
windows. Measure horizontal and vertical residual flow, drift, peak displacement, jerk, crop/frame retention,
and repeat/encode variability separately. Blind and randomize raw-versus-stabilized pairs for Gemini, with
separate judgments of horizontal drift, vertical drift, shake and artifacts; treat these two windows from one
source as exploratory proxy evidence, not independent clips or camera ground truth. Continue only if any
x-specific effect exceeds repeat variability, is visible in both windows and is not explained by crop/encoding.
Do not change the model from the synthetic result alone. The small resolution/frame-length sensitivity check
also remains open.

### Real-footage gate revalidation and disposition (2026-10-17)

The planned current-source runs and blind reviews already exist; do not rerun them. Both
`out/bench/current_source_t060_10s_baseline/` and
`out/bench/current_source_t800_10s_baseline/` are successful 10-second runs on source SHA
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`, with matching config digest
`8e0378b967beaed72f14d572ad1dd6d0b480034c1b5a094c2d90c7562258a2fe`, 300 aligned output frames, reliable
metrics and no fit fallbacks. Their paired review clip hashes match the completed Gemini records. Both
position-reversal audits report mapped-method agreement and zero side-following.

At t060, RAFT jitter changed **1.5873 → 0.5192 px**, reported frame motion **2.1710 → 1.0483 px**, and
x/y/rotation stability **[0.929, 0.843, 0.876] → [0.956, 0.934, 0.926]**; crop support was 97.76%. At t800,
jitter changed **1.0361 → 0.0632 px**, frame motion **1.1195 → 0.0993 px**, and x/y/rotation stability
**[0.920, 0.784, 0.856] → [0.950, 0.722, 0.824]**; crop support was 98.44%. These are single-run flow-based
measurements, not camera-path ground truth. Reduced jitter alone does not distinguish shake removal from
intentional-motion suppression; the axis scores do not move uniformly across windows.

Gemini 3.8 Flash preferred raw at medium confidence in both panel orders for both windows, citing black edge
gaps/wedges on the stabilized output. The t060 and t800 audits show that this preference followed the mapped
content, not screen position. This is one automated reviewer on compressed, low-resolution comparisons from
two windows of the same source. It is exploratory evidence of a perceptual border tradeoff, not a general
quality result. It does not test or validate content-aware intent preservation.

A redundant t060 baseline rerun was interrupted after about 104 minutes; an exploratory one-second run with
flow downscale 4 and 8 iterations then completed in 55 seconds. The short run only demonstrated that the
reduced-cost path works and is not quality evidence. No new full baseline or Gemini call is needed. The same-
source real-footage gate is complete; the next useful gate is an independently selected source/window with
separate measurements of high-frequency shake attenuation and low-frequency path retention, plus edge/crop
artifacts and blinded review. The small resolution/frame-length sensitivity check remains open, and no model
or production default change follows from these results.

### Preregistered unseen-segment replication on `00177.MTS` (2026-10-17)

The two-window `00176.MTS` screen cannot validate intent preservation because it has no camera-path ground truth.
To test whether the jitter/artifact tradeoff repeats on a different recording, use the existing `00177.MTS`
source (SHA-256 `8c5d98d15448ad0557537ba9d02a28fc426185b4e32bdf1be3eb4964fdbea714`) at one previously unused
ten-second segment. Candidate integer-second starts were `5..40`, `65..80`, and `100..105`, excluding all
previously measured windows; select uniformly with `random.Random(20261017)`. The frozen draw is **22–32 s**.
This is an unexamined segment from a second source file, not a new camera or a source-level holdout: other
windows in `00177.MTS` have been studied.

Use the existing fidelity-0.1 baseline config (digest
`8e0378b967beaed72f14d572ad1dd6d0b480034c1b5a094c2d90c7562258a2fe`), output and evaluation at `30000/1001`
fps, one run and no tuning. Keep the result row even if RAFT/KLT reliability fails. Before interpretation require
exact cadence alignment, no fit fallback, and adequate independent KLT support; report RAFT and KLT jitter and
wobble separately, per-axis path stability, crop/edge support, and frame-motion retention. Then build a
source-hash-verified raw/stabilized pair and its position reversal and review both with the fixed blinded Gemini
protocol. Treat agreement as descriptive only. This screen tests whether the measured tradeoff recurs on a new
segment; it cannot identify intended camera motion or validate content-aware intent. Do not change the model or
defaults from it.

Outcome (2026-10-17): the frozen `00177.MTS` t022–032 s run completed at
`out/bench/intent_real_00177_t022_baseline/`. Source hash, config digest, 300-frame CFR output and mask
alignment all matched the preregistration. It is **unscorable**: the RAFT evaluator fell back on 3 input and 2
output fits, and the independent KLT input had 11 failed fits with no jitter estimate. The manifest sets
`metrics.reliability.reliable=false` and `evaluator_agreement.reliable=false`. Its flow-based point estimates
are not quality evidence. Following the frozen gate, no Gemini pair was built or submitted for this segment.
Stabilization took 1,385 s; evaluation took 693 s; total wall time was 2,083 s, including 672 s of configured
shared-lock pauses. DeepSeek Flash via Pi confirmed that the failure invalidates the sequential visual-review
step; do not rehabilitate it with subjective comments. This is a retained failure row, not proof that the
stabilizer worsens quality. The planned real-footage intent gate remains open pending a different source with
reliable independent tracking or camera-path ground truth.

Supplementary positive-control visual QC (2026-10-17): the already-studied `00177.MTS` t050–060 scale-2
baseline was paired against its raw input and submitted with a position reversal. It is not holdout evidence and
does not change the failed t022–032 gate. Gemini returned medium-confidence ties in both orders, citing severe
low-light blur/noise; the original-order call noted subtle bottom-border warping in the stabilized opening
frame. The position-bias audit paired the two ties correctly, but zero non-tie calls mean side bias cannot be
estimated. The pair, API response and audit are in
`out/bench/intent_real_00177_t050_positive_control/`. RAFT jitter on this previously explored window was
0.2689 → 0.1561 px with zero RAFT fallbacks, but KLT had 2 failed input and 2 failed output pairs and no
jitter estimates. Treat this as a protocol/limitations check only; it does not validate the real-footage
intent hypothesis or establish a perceptual benefit.

### Explicit shot-boundary handling (2026-10-17)

DeepSeek Flash via Pi implemented an opt-in `--cut-frames N[,N...]` path for caller-supplied frame indices that
start new shots. The pair spanning each cut is skipped, the RAFT temporal state is reset, and global-path,
mesh/row residual accumulation and smoothing restart from the new shot. The default remains a single continuous
shot; indices must align with `--analysis-stride`. This gives prediction and future depth/scene cues a clean
state boundary, but does not automatically detect cuts or infer content.

Cut metadata is saved in motion schema `/5` and plan schema `/3`; the audit readers continue to load earlier
schemas. Motion `/4` reason codes remain limited to 0–3; `/5` adds code 4 for `sequence_boundary`. Audit summaries
count these boundary placeholders separately from failed geometric fits and verify they match the cut indices and
analysis stride.

Focused coverage includes skipped flow/reset calls, independent per-shot path and residual planning, fallback
accounting, invalid/misaligned cut validation, CLI wiring, and old/new schema round-trips. The focused set passed
**341 tests** and the full suite passed **1,724 tests** after the final audit-accounting changes. This is a lifecycle
correctness capability, not evidence of better perceptual quality or an integrated prediction/depth/intent
model.

### Low-frequency L1 path-fidelity real-footage check (2026-10-18)

Using cached analysis for `00176.MTS` (SHA-256
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`), replanned and rendered the previously
selected t060–070 s and t800–810 s windows with only `l1_low_freq_fidelity=5.0` and sigma 12 changed. The
reconstruction check reproduced the baseline global path, mesh, rows, and raw/path quads exactly (max absolute
difference 0); both candidate renders retained the 300-frame cadence and had no fit fallbacks. This isolates the
planner setting without repeating the expensive optical-flow analysis.

The candidate reduces border loss (mean border fraction 0.718% vs 2.24% at t060, and 0.373% vs the baseline
1.56% at t800), but it retains more camera path and consequently gives back stabilization. KLT-only estimates
changed from 0.524 to 0.835 px jitter and 1.039 to 1.756 px frame motion at t060; at t800 they changed from
0.185 to 0.412 px jitter and 0.239 to 0.626 px frame motion. RAFT measurements also show candidate jitter
above baseline at both windows (0.848 vs 0.519 px; 0.384 vs 0.063 px), while remaining below raw-input jitter
(1.587 and 1.036 px). These are correlated windows from one source and flow-based proxies without camera-path
ground truth; increased retained motion is not itself proof of better intent preservation.

Gemini 3.8 Flash comparisons were baseline-vs-candidate on identical output frames, with panel position reversed.
The initial review was mixed (ties in original order; opposite mapped preferences on the reversals). A second
640x360-panel pass on the same windows returned three ties and one baseline preference for visible border/edge
behavior. Both position-bias audits have zero pairs with non-tie agreement across the two orders. The second
pass is a resolution-sensitivity check by the same model on the same source windows, not independent evidence.
The clips, raw judgments and audits are retained under `out/bench/low_freq_l1_fidelity_00176/` and
`out/bench/gemini_review/low_freq_l1_00176/` (ignored experiment artifacts). The feature remains opt-in; no
quality win is established and no default change is justified.

### Mask-selected matched-framing review (2026-10-18)

Added an opt-in `bench.real_pair_review --matched-crop-area-loss-cap 0.04` mode. It verifies and streams the
baseline run's validity-mask video, selects one centered aspect-preserving rectangle maximizing mean valid
support over the whole window within the area-loss cap, and applies that exact rectangle to both raw and
stabilized panels. The review manifest records the rectangle, cap, selected support and mask hash. The default
uncropped workflow and production renderer are unchanged. Position-reversal identities include the crop
rectangle so uncropped and cropped judgments cannot be accidentally joined as one pair.

The first screen reused reliable, previously selected ten-second baselines from two distinct sources:
`STREAM_00013.MTS` t032–042 (SHA-256
`594d11e5789f2cf318dfe67a575169ed45dcc3bd6d974a96d0a4f7653f69e39e`) and `STREAM_00024.MTS` t109–119
(SHA-256 `8885a7890de8a35069cd1866d26cdbcd9b70279dd98307a8d407606216d8a56e`). Both had 300 aligned frames,
zero fit fallbacks, reliable RAFT metrics and independent KLT agreement. The selected shared crop was
`x=19,y=10,width=1882,height=1059`, removing 3.885% of image area. Average mask support rose from 98.085% to
99.234% on S13 and from 98.851% to 99.708% on S24. This measures available pixels only; the crop does not
change the underlying stabilization.

Gemini 3.8 Flash reviewed uncropped and matched-crop pairs in both panel orders (eight judgments total). S13
received ties in all four calls. S24 received ties in three calls and one original-order raw preference; its
position reversal was a tie. The audit found four correctly paired conditions and zero pairs with repeatable
non-tie method agreement. This review therefore does not show that cropping changes perceived quality; the
sample remains a small automated screen. No crop policy or stabilization default changes follow. The complete
review and audit remain in ignored `out/bench/matched_crop_visual_review_gemini*.json` artifacts. Next, apply
the same frozen crop rule to the existing `00176.MTS` t060/t800 baselines whose earlier uncropped Gemini calls
flagged borders, to directly test that framing confound; treat those same-source windows as paired diagnostics,
not independent validation.

Follow-up (2026-10-18): the frozen rule selected the same 3.885%-area crop for both `00176.MTS` t060–070 and
t800–810, raising mean mask support to 98.972% and 99.576%, respectively. The earlier uncropped reviews had
preferred raw at medium confidence in both panel orders for each window. On the matched-crop reviews, Gemini
returned medium-confidence ties in both orders for both windows; the position audit paired both reversals and
found zero non-tie method agreements. This directly supports shifting borders as a major contributor to the
earlier raw preference on these two windows. It does not prove that borders are the only quality issue, eliminate
all invalid pixels, or generalize beyond two windows from one source. No production crop behavior or default
changes follow; a crop policy still needs evaluation of motion, retained field of view, and failure cases on
independent sources.

### Fixed crop-zoom replay at the matched-crop scale (2026-10-18)

To distinguish a review-only crop from a real output change, replayed the cached `00176.MTS` t060/t800 motion
fits with only `crop_zoom=1.0202`, approximately matching the `1882x1059` review crop. Each baseline plan was
reconstructed exactly before planning; candidate `G` equaled baseline `G @ Z` (max absolute difference 0),
mesh/row corrections and measured camera path were unchanged, both renders had zero fit fallbacks, and all
600 processed frames produced 300 aligned `30000/1001` outputs with baseline-identical output PTS.

At t060, mean invalid mask area fell **2.236% → 1.018%** (worst frame **5.804% → 4.346%**); at t800 it fell
**1.556% → 0.415%** (worst **4.106% → 2.172%**). The corresponding RAFT jitter changed **0.51917 →
0.52825 px** at t060 (**+1.75%**) and **0.06321 → 0.06484 px** at t800 (**+2.58%**). RAFT wobble was
**0.05365 → 0.05352 px** and **0.03431 → 0.03442 px**, respectively. Independent KLT-only jitter changed
**0.52383 → 0.53300 px** (**+1.75%**) and **0.18540 → 0.18687 px** (**+0.79%**), with no fit failures.
These are two windows from one source, and both evaluators measure image-space motion rather than camera-intent
truth.

Gemini 3.8 Flash compared baseline and crop-zoom output with reversed panel order. It preferred the candidate in
one call for each window and returned ties for both reversals. The position audit found **zero two-order non-tie
agreements**. The visible result is therefore suggestive but inconclusive. `crop_zoom=1.0202` remains an
experimental operating point; default `1.0` is unchanged. Next verify the frozen ratio on the reliable S13/S24
windows from distinct source files, measuring the same crop support and independent KLT changes before considering
any preset or runtime policy.

Follow-up on independent sources (2026-10-18): the same fixed crop-zoom replay was applied to the existing S13
t032–042 and S24 t109–119 10-second windows. Both plans preserved their baseline camera paths and residual fields,
had zero fit fallbacks, and produced aligned 300-frame outputs. Mean invalid mask area fell **1.919% → 0.767%**
on S13 and **1.149% → 0.284%** on S24 (worst-frame area **6.677% → 4.866%** and **3.925% → 2.028%**).
RAFT jitter rose **1.57%** and **1.69%**, respectively; KLT-only jitter rose **2.74%** and **1.79%**. RAFT
wobble improved **6.39%** on S24 and worsened **1.10%** on S13. Gemini 3.8 Flash returned medium-confidence
ties in both panel orders for both windows; the position audit found zero two-order non-tie agreements. The
fixed zoom consistently improves edge support at a small measured motion cost, but the visual review does not
show a perceived quality difference on these clips. This remains a four-window screen across three source files,
not held-out population evidence; `crop_zoom=1.0` remains the default. Full reports are in ignored
`out/bench/cropzoom_102_crosssource/`.

### Real-pixel depth-rank camera-path kill gate (2026-10-18)

Added `bench/depth_layer_real_render_review.py`, an isolated renderer for the existing E13 pooled and
relative-depth-rank affine corrections. It backward-warps cached RGB samples with explicit bilinear support,
writes matched pooled/rank outputs and validity masks, and delegates blinded A/B clip creation plus panel
reversal to `bench.real_pair_review`. It changes no production path. CodeGraph was synced; its affected-test
query found no linked tests, so the five new sampler/input-validation tests were run directly and passed.

The first pixel gate used the existing two-second `STREAM_00025.MTS` cache (source SHA-256
`d6744f20554b80cf93ac7e7a1310ce16bc1d567a7409eaf96cc6d272b2b97ca0`) and the frozen E13 `bins_4__sigma_1`
variant. The 30 depth-sampled frames were rendered at their effective **15000/1001 fps** cadence. Mean
bilinear-valid support was **99.488%** for the pooled path and **99.512%, 99.495%, 99.355%, and 99.009%** for
rank 0–3. Gemini 3.8 Flash reviewed four pooled-versus-rank comparisons and their position reversals: **all
eight calls were ties**, with no repeatable non-tie method preference.

The independent KLT check could not score continuous path jitter: input and pooled outputs each had **4 failed
pairs of 29**; rank 0 had 4, ranks 1 and 2 had 5 each, and rank 3 had 7. Per the evaluator contract, jitter
remains unavailable for every sequence. This is an inconclusive, single-window screen, not evidence of
equivalence or quality improvement. Together with the five earlier E13 geometric windows showing no
consistent rank-policy advantage, DeepSeek Flash via Pi recommended stopping the **rank-selected global path**
branch rather than extending its sweep. Reopen it only for a pre-registered targeted case with valid independent
tracking and a decision-relevant predicted benefit. Reports, renders, KLT failures and blinded reviews remain
under ignored `out/bench/depth_layer_real_render_review_stream00025_t05/`; no depth policy or default changes.

### E25 depth-gated residual mesh prototype and real-footage screen (2026-10-18)

Added a default-off production-path hook for depth-aware local residual mesh fitting. `Config.mesh_depth_strength`
remains `0.0` by default. When enabled, `analyze`/`stabilize` accept a caller-provided `depth_fn(source_frame_index)`;
relative depth changes only the local residual mesh regression, fill and smoothing. Global RANSAC/mover handling,
row correction, planning and rendering remain on their existing paths. The callback is an API for supplying aligned
maps, not a shipped depth estimator or CLI adapter. Depth inference and its cost are therefore not included in
these render comparisons.

Focused synthetic tests verify depth-off byte parity, boundary separation, source-frame indexing and that the
mesh change reaches planned/rendered pixels. The independently rerun full suite passed: **1826 passed, 18 warnings**.
CodeGraph is synced. The implementation was committed and pushed as `0c23e7e`.

Ran one matched 24-iteration RAFT comparison per arm on two 120-frame, two-second windows, using the same
`flow_downscale=2` and half-rate `analysis_stride=2`. The candidate used `mesh_depth_strength=1.0`; its every-frame
relative depth was linearly interpolated from every-fourth-frame Depth Anything V2 Base maps. No arm had fit
fallbacks and every frame rendered.

| source window | baseline / depth analysis (s) | baseline / depth mean border | baseline / depth worst border | Gemini (original + panel reversal) |
|---|---:|---:|---:|---|
| `00176.MTS` t060–062 | 49.249 / 51.727 | 1.5215% / 1.5408% | 3.5185% / 3.6060% | tie / tie |
| `STREAM_00016.MTS` t130–132 | 50.954 / 50.494 | 0.1468% / 0.1559% | 0.4063% / 0.4077% | tie / tie |

The `00176.MTS` source container's current hash differs from the older cache metadata, so those old depth maps were
not used until a fresh decode proved the 120 RGB frames byte-identical to the cached frame content digest. The maps
were then rebound to the fresh frame-cache hash and validated against it. `STREAM_00016.MTS` source and depth-cache
hashes already matched. This is still only a two-window, two-second screen with one timing run per arm. Gemini's
`STREAM_00016` judgment explicitly notes that the clip is very dark and too short for fine motion comparison. There
is no repeatable visible win, and border loss is slightly higher with depth in both windows. Keep this API
experimental and default-off; do not add a bundled estimator, CLI option, or extra inference cost on this evidence.
Revisit only under E25's longer parallax-heavy footage and independent-tracking gate. Render pairs, Gemini judgments,
and provenance stay in ignored `out/bench/depth_mesh_real_*/` artifacts.

E25 follow-up (2026-10-18): longer real-window gate on `00176.MTS` t060–070

To test whether the two-second screen had missed a real parallax benefit, rebuilt a fresh ten-second cache from the
current `00176.MTS` (SHA-256
`e4b1f80d54dfa9089741486be297c6fe74d754fe127ec04dfe5a7edcf8d9fbb3`), inferred Depth Anything V2 Base maps on
MPS every four frames (150 maps), and ran the default and depth-gated production paths over the same 600 decoded
frames. Both arms used RAFT-things, 24 iterations, downscale 2, half-rate analysis (300 pairs), and rendered
600 frames at 60000/1001 fps. `mesh_depth_strength=1.0` was the only candidate change. The matched 7% interior
edge mask and deterministic seed were used for all independent KLT sequences.

| measure | baseline | depth-gated |
|---|---:|---:|
| analysis / planning / rendering (s) | 281.03 / 30.40 / 22.00 | 291.51 / 28.59 / 19.13 |
| total throughput (frames/s) | 1.799 | 1.769 |
| fit fallbacks | 0 | 0 |
| mean / worst border loss | 2.2385% / 5.7360% | 2.3002% / 5.7754% |
| independent KLT jitter (px) | 0.262393 | 0.262348 |
| independent KLT frame motion (px) | 0.612270 | 0.613012 |
| KLT failed pairs | 0/599 | 0/599 |

The raw-input KLT jitter was 0.664856 px; both outputs reduced it by about 60.5%. The candidate's output/input
jitter ratio was 0.394593 versus 0.394661 for baseline, a negligible difference; its frame-motion proxy was
slightly worse. This is evidence of no practical KLT gain, not equivalence or ground truth. Gemini 3.8 Flash gave
a medium-confidence tie in the original panel order and a medium-confidence baseline preference in the reversal;
the pair has no repeatable non-tie method preference. In the reversal it noted dynamic bottom-edge deformation
on candidate panel A, a single-model observation without repeat confirmation. Border loss also increased slightly,
and this one timing run measured about 1.7% lower total throughput with depth.

E25's improvement gate is **not met**. Keep the default-off API as a research hook for now, stop tuning this
strength, and do not add a bundled depth estimator or inference/runtime cost. The results, full-resolution outputs,
KLT report, Gemini judgments and reversal audit remain in ignored
`out/bench/depth_mesh_real_00176_t60_10s/`. Continue the plan at the Rust compatibility spike (E20) rather than
extend this depth branch without a new evidence-backed target.

### E20 follow-up (2026-10-18): checkpoint-backed RAFT encoder parity

Expanded `rust/raft-compat` from the correlation-lookup prototype to both RAFT `BasicEncoder` paths. The Rust
implementation now loads the converted `fnet` and `cnet` parameters by their exact names and shapes, rejects
missing or unconsumed tensors within each encoder prefix, and runs the six residual blocks, stride-2 projections,
instance normalization for `fnet`, checkpoint-stat batch normalization for evaluation-mode `cnet`, and final
1x1 projection. It does not yet port the correlation pyramid or recurrent update block.

The reproducible fixture uses a deterministic normalized 32x40 frame pair. It runs `fnet` at the production
batch shape `[2,32,40,3]` and `cnet` on the first frame at `[1,32,40,3]`; it stores all 10,240 `fnet` output
values at `[2,4,5,256]` and all 5,120 `cnet` output values at `[1,4,5,256]`. It records the source checkpoint SHA-256
`84c1f730d4620ef4dd0a2eeeef97db7a729fc4190ee05cdd0169030fc92fccc1`, Python MLX `0.32.3`, Rust `mlx-rs 0.32.0`,
and its MLX `0.32.2` target. Regenerating the fixture produced the same SHA-256. The independent Rust comparison
reported **zero maximum absolute and relative error** for both encoders on this fixture. The earlier
checkpoint-backed `cnet.conv1` comparison and encoder loader rejection cases also pass. All **8 Rust tests pass**;
`cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` pass. CodeGraph is synced (its affected-test
query does not discover Rust tests, which were run directly).

This closes only the encoder portion of E20. It does not establish parity for correlation lookup inside the
network, compiled recurrent updates, bidirectional final flow, end-to-end timing/copies/memory, cancellation,
packaging, or sustained ownership. The encoder fixture crosses the current MLX `0.32.2`/`0.32.3` version gap,
but the broader migration gate still requires an explicitly pinned pairing. Keep the migration decision open and
continue with correlation-pyramid construction and the update block before judging Rust for production.

### E20 follow-up (2026-10-18): correlation pyramid parity

Added a Rust `CorrBlock` that forms the all-pairs correlation from paired NHWC features, divides by `sqrt(C)`,
builds floor-sized 2x2 average-pooled levels (dropping odd trailing rows and columns), and samples each level with
the already-tested RAFT XY-order, bilinear, zero-padded lookup. The checkpoint-independent fixture exercises
fractional and out-of-frame coordinates on deterministic 7x9x5 feature maps, compares every pyramid value and
lookup output, and covers float32 and float16 storage. Its Python generator uses `mlxstab.raft.model.CorrBlock`;
two consecutive generations produced the same SHA-256
`87b06c7de783fcd1e104175ada38ac30a7eefaa4497c85fdd63344185d51f1e0`.

The exact pyramid values matched the Python fixture in both dtypes; worst normalized lookup error was
`7.14e-8` for float32 and zero for float16. All **11 Rust tests pass**, and `cargo fmt --check` and strict clippy
pass. CodeGraph was synced; its affected-file query did not discover Rust tests, which were run directly. This
closes correlation construction and lookup parity for the small fixture only. It is still not a compiled or
device-side production path, and the recurrent update block, bidirectional RAFT flow, end-to-end parity, and
runtime/memory evidence remain open. Continue with the update block and integrate the full flow before judging
whether Rust improves the project.

### E20 follow-up (2026-10-18): checkpoint-backed update-block parity

Added the checkpoint-backed Rust `BasicUpdateBlock`: motion encoder, horizontal then vertical separable GRU,
flow-delta head, and quarter-scaled 576-channel convex-upsample mask head. The loader checks exact parameter
names and shapes, rejects missing and unexpected update-block keys, and validates the production tensor shapes
`net/inp=[1,H,W,128]`, `corr=[1,H,W,324]`, and `flow=[1,H,W,2]`.

The fixture generator runs the repository's Python MLX `BasicUpdateBlock` on deterministic 3x4 inputs with the
checked-in RAFT checkpoint. It records the inputs, updated hidden tensor, flow delta, and mask with checkpoint and
runtime version metadata; two generations produced identical SHA-256
`e95fdcf8219aa60a843f9b62bba3a788dbb2f6f39a200bff2123889efbe5d88c`. Rust matched Python with **zero maximum
absolute error** for all three outputs and for the complete `forward` entry point. All **14 Rust tests pass**;
formatting and strict clippy pass. CodeGraph was synced, and the changed-file query did not find Rust tests, which
were run directly.

The update block is now individually parity-tested but is not yet connected to the encoders and correlation
pyramid in an iterative bidirectional flow call. Compiled execution, final-flow parity, precision edge cases,
complete checkpoint schema validation, cancellation/recovery, and sustained end-to-end timing and memory remain
open. Continue to one integrated RAFT flow test before judging Rust for production.

### E20 follow-up (2026-10-18): integrated bidirectional flow parity

Connected the checkpoint-backed frame encoders, separate forward/backward correlation pyramids, batch-2 update
block, and convex mask upsampler behind Rust `Raft::flow_bidir`. The loop accepts an iteration count, reuses each
frame's features, starts both directions from a shared XY grid, and keeps coordinate/flow updates in float32.
This follows Python `RAFT.frame_features` and `RAFT.flow_bidir`; correlation lookup remains host-materialized,
and this API is not compiled.

The checkpoint-backed fixture uses deterministic 64x64 RGB inputs and one refinement iteration on an 8x8 feature
grid. It stores both inputs and all four outputs (forward/backward low-resolution and full-resolution flow),
along with checkpoint and runtime metadata. Two generations produced the same SHA-256
`6382f3f5c8db471c6a54b7d1ce92376044ba3d19e5ab9bc673ddf9af9221c770`. Rust matched Python MLX with **zero maximum
absolute error** for all four outputs. All **17 Rust tests pass**; formatting, strict clippy, and `git diff --check`
pass. CodeGraph was synced; its affected-file query did not find Rust tests, which were run directly.

This establishes one-iteration fp32 parity on the checked-in setup, where Rust `mlx-rs 0.32.0` targets MLX 0.32.2
and Python MLX is 0.32.3. Additional iteration counts and warm starts are not fixture-tested; the current Rust API
does not expose warm starts or fp16 network/correlation paths. Full checkpoint-schema validation, exact version
pairing, compiled updates, runtime/copy/memory measurements, cancellation and recovery remain untested. Continue
with multi-iteration parity and a measured complete-call comparison before considering a production migration.

### E20 follow-up (2026-10-18): 24-iteration flow parity

Extended the same deterministic bidirectional fixture to include both the one-iteration regression and the
production 24-iteration setting. Two consecutive Python generator runs produced the same SHA-256
`b48a360ba0cbcd0805beda5eed4ae439e969a15fcb436887162a4dd6feb7fa17`. The one-iteration Rust outputs remain
exact. At 24 iterations, maximum absolute errors are `1.0884e-4` pixels (low forward), `7.0375e-4` (full
forward), `8.9097e-4` (low backward), and `2.8856e-3` (full backward). The observed divergence grows over
recurrent refinement; it remains below 0.003 pixels on this fixture, but is not exact parity.

All **17 Rust tests pass**, along with formatting, strict clippy, and `git diff --check`. CodeGraph is synced;
its affected-file query still does not identify Rust tests. These checks establish only the deterministic
64x64 fp32 fixture. The binding/MLX version gap remains (`mlx-rs 0.32.0` targets 0.32.2; Python is 0.32.3),
and the host-lookup loop has no runtime or memory measurement. Warm starts, other iteration counts, fp16 paths,
compiled updates, cancellation/recovery, and repeated complete-call performance remain open.

### E20 follow-up (2026-10-18): production-length bidirectional flow parity

Extended the integrated Rust/Python flow fixture to a `cases` schema so one checkpoint-backed call can verify
both a one-iteration regression and a production-length loop. The fixture is now schema
`mlxstab.raft_compat.flow_bidir/2`: it records the deterministic 64x64 images once, then a `cases` array with
entries for **1 and 24 iterations**. Each case stores the forward/backward low-resolution flow and both
full-resolution flows after the mask head and convex 8x upsampling; the top level also records
`verified_iters: [1, 24]`, the checkpoint SHA-256, Python/Rust MLX versions and the correlation configuration.
The 1-iteration case is byte-for-byte unchanged from the previous fixture and remains the regression;
24 iterations matches the production Python stabilizer's analysis path. Two consecutive regenerations produced
the same fixture SHA-256 `b48a360ba0cbcd0805beda5eed4ae439e969a15fcb436887162a4dd6feb7fa17`.

The Rust test now loads the model once, runs each fixture case at its recorded iteration count, and compares all
four outputs. Maximum absolute errors are:

| iterations | low forward | full forward | low backward | full backward |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 |
| 24 | 1.0883808e-4 | 7.0375204e-4 | 8.9097023e-4 | 2.8855205e-3 |

The 1-iteration case keeps the original `1e-4` px gate. The 24-iteration case uses a `5e-3` px gate because
small per-step differences accumulate over the longer loop; every measured output remains under 3e-3 px on a
flow whose values are on the order of one pixel. All **17 Rust tests pass**; `cargo fmt --check`, strict
`cargo clippy --all-targets -- -D warnings` and `git diff --check` pass. CodeGraph was synced and its
affected-file query again found no Rust test links, so the tests were run directly.

This verifies the integrated fp32 loop at **1 and 24 iterations** only, on the checked-in setup where Rust
`mlx-rs 0.32.0` targets MLX 0.32.2 and Python MLX is 0.32.3. It is a parity result, not a runtime or speed
measurement. Warm starts, fp16 network/correlation paths, other iteration counts, full checkpoint-schema
validation, exact version pairing, compiled updates, cancellation/recovery and sustained memory behavior
remain untested; the production-migration decision stays open.

### E20 follow-up (2026-10-18): matched end-to-end baseline

Added a matched benchmark input generator and Rust release harness. All three paths use byte-identical seeded
RGB inputs, the same checkpoint, a 24-iteration bidirectional call, one warmup and repeated timed calls. The
Rust harness materializes all four outputs inside each timed call; model loading and construction are excluded.
At 360x640 (45x80 feature grid), seven measured calls produced these medians:

| backend | median per pair | process peak RSS |
|---|---:|---:|
| Rust release / MLX 0.32.2 | 0.707 s | 235 MB |
| Python MLX 0.32.3 | 0.431 s | 177 MB |
| PyTorch MPS | 0.531 s | 491 MB |

On this host and workload, Rust is **1.64x slower than Python MLX and 1.33x slower than PyTorch MPS**. Its
process peak RSS is 58.9 MB above MLX and 255 MB below PyTorch. This is a single-host, short-run baseline;
process RSS includes interpreter/runtime and model overhead and does not report accelerator allocations.
The current paths also differ slightly: the PyTorch reference computes the upsampling mask each iteration,
while Rust and Python MLX compute it only for the final iteration. The most obvious optimization target is
Rust's host-side correlation lookup, which materializes correlation data and loops on the CPU at every lookup.
This measurement is evidence to optimize and remeasure, not evidence that the Rust path is faster or ready
to replace either existing backend.

### E20 follow-up (2026-10-18): device-side correlation lookup

Replaced repeated host copies and the Rust sampling loop with a cached MLX Metal kernel ported from Python
`_LOOK_SRC`. Correlation and coordinate arrays stay on-device. The kernel preserves the tested XY offset order,
bilinear interpolation, zero padding, float32 output, float16/float32 correlation storage, and non-contiguous
input behavior. Its opaque handle is cached per thread alongside MLX's thread-local stream, avoiding unsafe
`Send` or `Sync` assumptions. The small correlation fixture and integrated flow fixture at 1 and 24 iterations
still pass; all **17 Rust tests pass**, and formatting, strict clippy, and `git diff --check` pass. CodeGraph
was synced; its affected-file query found no linked test files, so Rust tests were run directly.

The three backends were rerun on the same 360x640 pair with one warmup and seven timed 24-iteration calls:

| backend | median per pair | process peak RSS |
|---|---:|---:|
| Rust release / MLX 0.32.2 | 0.490 s | 91 MB |
| Python MLX 0.32.3 | 0.453 s | 176 MB |
| PyTorch MPS | 0.574 s | 422 MB |

In this run Rust is **8% slower than Python MLX and 15% faster than PyTorch MPS**. Against the pre-kernel
Rust baseline, latency fell from **0.707 s to 0.490 s** (31% faster) and process peak RSS fell from **235 MB
to 91 MB**. These are short measurements from one host; RSS is not accelerator memory. The update block
remains uncompiled, so this result does not justify replacing the Python backend. The next experiment is
compiling or fusing the recurrent update path, then repeating the same three-backend benchmark.

### E20 follow-up (2026-10-18): rejected GRU gate-convolution fusion

Tested merging each separable GRU step's update (`z`) and reset (`r`) convolutions, which share the same input,
by concatenating their checkpoint output channels and splitting the result before sigmoid. The combined
checkpoint tensors were bit-identical to the concatenated source weights, direct output error versus the
original two-convolution path was below `1e-6`, and the full 1/24-iteration fixtures passed. However, the
matched Rust release median regressed from **0.490 s to 0.567 s** on 360x640 (seven calls), roughly 16% slower.
The implementation was removed; no code from this experiment remains in the runtime. This rejects that fusion
for the current MLX convolution backend. The next optimization should reduce recurrent graph or kernel overhead
without increasing convolution cost, then be accepted only after parity and benchmark gates.
