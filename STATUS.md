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

Remaining Stage 0 work: rerun the current defaults and the unfinished camcorder configurations; reproduce the reported tree deformation and floaty motion with global-only, mesh-only, row-only and combined ablations; audit motion-field accumulation under rotation and parallax. Real-footage baselines and the deformation diagnosis remain outstanding, so Stage 0 is not complete. Fixture parity (fp32/fp16) does not establish full-clip quality of the approximations.

## What works

- `mlxstab/warp.py`: Metal sampler (bilinear / Catmull-Rom / Lanczos-3, zeros/border, align_corners). `grid_sample` max abs error vs torch 1.2e-7 .. 2.4e-7. Forward only.
- `mlxstab/raft/`: RAFT-things in MLX + `.pth` -> safetensors converter. Final flow EPE vs PyTorch at 12 iterations: 2.2e-3 px (reference demo frames), 2.4e-3 px (synthetic); batch-2 bidirectional path: 2.2e-3 px fp32, 2.7e-3 px fp16. Gate is < 0.01 px for the fp32 path.
- `mlxstab/stabilize.py`, `python -m mlxstab`: no-crop stabilizer (similarity global fit by RANSAC with forward-backward masking and mover exclusion, residual mesh + rolling-shutter row bands, L1-optimal paths with border budget, roll levelling, one Lanczos resample, border mask video). Interlaced sources deinterlaced first (bwdif send_field).
- `.claude/verify.sh`: 99 tests pass locally, offline; weight-dependent tests skip when weights are unavailable.

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
.claude/verify.sh                                              # green check (99 tests)
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
Stage 0 resume: the runner is built and reviewed; the 1 s real smoke passed. Next is the baseline 10 s
windows (`--case baseline --start 60/800 --duration 10`, shared lock), then the isolated `fp16-network`,
`fp16-corr`, `combined-fp16`, `warm-8` and `stride-2` cases on the same windows before any quality
conclusion. Use `--lock on --lock-chunk 30 --lock-pause 12` for long jobs. Setup, weights and conversion are
in the README; private-footage outputs stay under this repo's ignored `out/`, earlier film-repo outputs are
under renders/v5_proto/mlxstab_*.
