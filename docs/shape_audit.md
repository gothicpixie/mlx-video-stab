# E02 rigid-line shape audit (CPU-only, analytic)

`bench/shape_audit.py` is a bounded, CPU-only geometry experiment for experiment **E02** ("what causes
deformation?"). It measures whether the current fixed-grid residual map
(`composition_audit.naive_from_plan`) bends a straight line more than the Stage 1 composed candidate
(`composition_candidate.candidate_source_coords(..., highpass=False)`). It is analytic scope only: **no media,
FFmpeg, MLX/Metal kernels, production wiring, or changed defaults**, and it makes no real-footage quality
claim. Run it with:

```bash
python -m bench.shape_audit --out-dir out/shape_audit       # writes shape_audit.json
```

## Metric

`sample_line(p0, p1, n)` returns `n` output-space points on a segment. `best_fit_line(pts)` centres the mapped
points, takes the dominant SVD/PCA direction as the line, and measures the absolute perpendicular residual.
It reports:

| field | meaning |
| --- | --- |
| `dev_max_px`, `dev_p95_px`, `dev_rms_px` | perpendicular deviation from the best-fit line (px) |
| `span_px` | along-line extent of the mapped points (px) |
| `dev_max_over_span`, `dev_p95_over_span` | deviation normalised by the span (scale-free) |
| `nonfinite_frac`, `ok`, `reasons` | rejection accounting |

Bad shapes are **rejected, not hidden**: `ok=False` plus a reason for non-finite coordinates, fewer than two
points, a failed SVD, or a collapsed span (`span <= 1e-9` px). In those cases the deviation summaries are
`None`, so a collapsed map cannot score as perfectly straight. `map_line_deviation(map_fn, p0, p1, n)`
composes the two steps.

## Controls (identity/projective straight; bowed detected)

The diagonal test line is `(5,5) -> (W-6,H-6)` with `W,H = 64,48`, `n = 401`; a bowed control adds a 6 px
sine bow perpendicular to the line.

| control | `dev_max_px` | expected |
| --- | --- | --- |
| identity | ~8.9e-15 | straight |
| single projective homography | ~1.8e-14 | straight |
| bowed field (6 px amplitude) | ~3.26 | detected |
| pipeline: projective globals, zero residuals, fixed grid | ~1.4e-14 | straight |
| pipeline: projective globals, zero residuals, candidate | ~2.0e-14 | straight |

The last two route a projective-global, zero-residual fixture through the real plan/`naive_from_plan` and the
candidate, so the metric is connected to the pipeline, not only to hand-written maps.

## Two-step analytic fixture

Each true per-pair motion is a full projective homography `P_k(p) = p / (1 + a_k x + b_k y)` with

```
P0: a=7.5e-3,  b=-4.5e-3
P1: a=-6.0e-3, b=6.0e-3
```

reported through the fit interface as `Fit.H = I` plus the mesh vertex samples of `P_k(p) - p` (48x36 cells).
Because each pair motion is a homography, the true composition `P1 o P0` is a homography and maps lines to
lines. The **ground truth is derived independently** by composing the two matrices and applying them to the
sampled line — it never touches the plan. The fixture is deliberately horizontal through mid-frame
(`(4,24) -> (59,24)`) so it samples the perspective gradient along `x` maximally.

The candidate composes the pair maps at their source positions, so it approximates `P1 o P0`; the fixed grid
adds the cumulative residual at the current global coordinate, `x + (P0-Id)(x) + (P1-Id)(x)`, which is not
that composition and bends the line.

## Measured results (fixture, frame 2, `n = 401`)

| map | `dev_max_px` | `dev_p95_px` | `dev_max_over_span` | `span_px` | pointwise `max |map - truth|` px |
| --- | --- | --- | --- | --- | --- |
| ground truth `P1 o P0` | 1.3e-14 | 9.6e-15 | ~0 | 48.66 | 0 |
| candidate (composed) | 0.001814 | 0.001332 | 3.7e-05 | 48.66 | 0.005059 |
| fixed grid (`naive_from_plan`) | 1.028842 | 0.832250 | 0.01837 | 56.00 | 8.023751 |

The candidate's self-fit deviation (~0.0018 px) is bounded by its measured mesh discretization error versus
the true composition (~0.0051 px), and is **~567x** smaller than the fixed grid's (~1.03 px); the fixed grid
is also ~8.0 px from the true line pointwise versus ~0.005 px for the candidate. The candidate maps a line to a
line to within mesh discretization; the fixed grid measurably bends it. This is a mechanism check on the
residual-composition path, consistent with the `composition_audit` counterexample.

## Temporal shape variation

The fixture also records `dev_max_px` at frames 0, 1, 2 and the maximum step-to-step change, so the audit
measures temporal shape variation and not only final-frame bending. Measured (`n = 401`):

| map | frame 0 | frame 1 | frame 2 | max step |
| --- | --- | --- | --- | --- |
| ground truth (matrix product `P1 @ P0`) | 3.6e-15 | 1.4e-14 | 1.1e-14 | 1.1e-14 |
| candidate (composed) | 1.1e-14 | 4.45e-05 | 0.001814 | 0.001770 |
| fixed grid (sum) | 1.1e-14 | 4.45e-05 | 1.028842 | 1.028798 |

A single pair motion is a homography, so at frame 1 both maps stay straight; the fixed grid bends only once
the second pair residual is summed in (frame 2), a step of ~1.03 px. The candidate's step (~0.0018 px) is
bounded by its measured mesh discretization (~0.0051 px).

## Tolerances

Test tolerances are taken from the measured discretization, not chosen generously: the candidate's self-fit
deviation must stay below `0.01` px **and** no larger than its own pointwise discretization error
(`candidate_vs_ground_truth_max_px`); its temporal step must be `<=` that same measured mesh discretization;
while the fixed grid must exceed `0.5` px and `100x` the candidate for both final-frame bending and temporal
step. The ground-truth control must stay below `1e-9` px. If the fixture ever fails to separate the two maps,
or any measured line is invalid, the report sets `conclusion.separation = "unknown"` (and
`candidate_straighter_than_fixed_grid = null`) with reasons and the tests fail rather than claim an improvement;
the JSON then records the discrepancy so the obstacle can be documented.

## Limitations

- Analytic 2D, CPU-only. Two projective pairs (T=3), one fixture parameterisation, one line orientation.
- **This does not guarantee that a real scene stays straight.** It is a controlled analytic mechanism check,
  not evidence about real footage, coverage, or perception, and it is not evidence for changing defaults.
- The mesh (`48x36`) discretizes the projective residual; the reported candidate deviation is mesh-limited.
- `naive_from_plan` is the independent NumPy replication used by `composition_audit`; it is not the MLX
  `source_coords` path (which agrees with it to MLX float32, per the audit). No production code is touched.

---

# Captured-plan geometry audit (real plans)

`bench/plan_shape_audit.py` is a **separate** audit for captured plans. The analytic audit above builds known
fixtures and compares the fixed-grid residual sum against the composed candidate. This real-plan audit loads a
plan that was already produced and dumped by the stabilizer (`mlxstab.stabilize.save_plan`) and probes the
**production** map `mlxstab.stabilize.source_coords` directly. The analytic results above are unchanged and
were not re-derived; the two audits share only the straightness metric.

```bash
python -m bench.plan_shape_audit out/plan.npz --every 30 --samples 401        # JSON to stdout
python -m bench.plan_shape_audit out/plan.npz --out out/plan_shape_audit.json # JSON to a file
```

- `PLAN` (positional): a `save_plan` NPZ. `--every` (default `30`) selects frames; the **final frame is always
  included even if the stride misses it**. `--samples` (default `401`) is the number of points sampled on each
  output-space segment. With `--out` the JSON is written there (plus a one-line summary to stdout); otherwise
  the JSON report is printed to stdout.

## What it measures

For each selected frame it probes four straight **output-space** segments — `horizontal`, `vertical`,
`diagonal_down` (top-left → bottom-right) and `diagonal_up` (top-right → bottom-left) — and reports, per
segment: the input line endpoints, the input-line straightness, the straightness of the **source-coordinate
mapped** line (production `source_coords`), and the **source-bounds coverage** of the mapped samples. Each
segment is evaluated on an MLX grid shaped `[1, N]` (one row of `N` samples), so the real mesh/row correction
kernel runs at `N` points and no full `H×W` frame is ever built or rendered.

Straightness is the existing [`best_fit_line`/`map_line_deviation`](#metric) metric (perpendicular deviation
from the best-fit line in px). Coverage is the fraction of mapped samples inside `[-0.5, W-0.5] ×
[-0.5, H-0.5]` (the `warp.valid_mask` pixel-edge convention), with `x/y` minima and maxima and a
`fully_in_bounds` flag. The report records the exact frame indices and a per-segment `summary`
(`mapped_dev_max_px_max`, `coverage_frac_min`, `invalid_lines`).

## Payload safety

The NPZ is opened with `np.load(..., allow_pickle=False)` and only plain numeric/unicode arrays are accepted.
Before any geometry runs the audit validates `PLAN_SCHEMA`, the required keys
(`schema, width, height, frames, fps, G, mesh_corr, row_corr, raw_quads, path_quads, fallbacks`), the frame
dimensions, every array shape (`G [T,3,3]`, `mesh_corr [T,gr+1,gc+1,2]`, `row_corr [T,nr,2]`,
`raw_quads/path_quads [T,4,2]`) and finiteness, then reconstructs `mlxstab.stabilize.Plan`. Malformed payloads
raise `PlanPayloadError` with a stable code (`unreadable`, `missing_keys`, `bad_schema`, `bad_scalar`,
`bad_dims`, `bad_shape`, `nonfinite`); the CLI turns that into a JSON-safe error object
(`{"schema": ..., "ok": false, "error": {"code", "message"}}`) and exits `2` instead of printing a traceback.
No payload is ever unpickled, and no production code or default is changed.

## Real-plan audit limitations

- **Warp-geometry proxy, not detection or perception.** It measures how production `source_coords` maps four
  straight output-space segments onto the source frame. It is **not** object/tree detection, scene
  understanding, or a perceptual/viewer-quality evaluation.
- **No media, no rendering.** Nothing is decoded and no frames are materialised; the audit evaluates the
  existing MLX kernel at `N` sparse points per segment, so it does not exercise sampling/media handling.
- **Four segments per frame.** It probes one horizontal, one vertical and two diagonal segments (default
  `401` samples each) at a strided selection of frames, not the whole field, and not real scene content.
- **Mesh/row resolution.** Corrections are bilinear over the mesh/row bands and sampled at the probe points;
  a bend below that discretisation is not resolved.
- **Coverage is geometric only.** The bounds fraction says whether mapped samples land inside the source
  image; it says nothing about what those samples contain.
- A `save_plan` NPZ is required input; this audit never runs the stabilizer and never writes a plan.

---

# Matched multi-plan comparison (E02)

`bench/plan_shape_audit.py --compare ...` compares two or more captured plans that are **proven to come from the
same source window, preprocessing, weights and resolved stabilization config** (apart from the deliberate
`mesh`/`rs_rows` residual-channel ablation), so their curvatures and coverages are directly comparable. The
single-plan behaviour above is unchanged; `--compare` is a separate mode.

```bash
python -m bench.plan_shape_audit --compare \
  out/bench/baseline/plan_r0.npz out/bench/global_only/plan_r0.npz \
  out/bench/mesh_only/plan_r0.npz out/bench/row_only/plan_r0.npz \
  --every 30 --samples 401                          # JSON to stdout
python -m bench.plan_shape_audit --compare A.npz B.npz --out out/plan_shape_comparison.json
```

- `--compare` takes **two or more** `save_plan` NPZ paths and is mutually exclusive with the single positional
  `PLAN`. Each plan is loaded through the same strict loader (`load_plan_npz`) and audited with the **identical**
  `--every`/`--samples`, so frames and probes line up one-to-one.
- **Sibling benchmark manifests.** For a plan at `RUN_DIR/plan_r0.npz`, the manifest written by `bench.run_case`
  is read from `RUN_DIR/manifest.json`. Every compared manifest must record the **same** `source.sha256`,
  `source.window` (`start_s`, `duration_s`), `source.preprocessing`, `weights.sha256` and a resolved `config`
  object.
- **Captured-plan proof.** Every compared manifest must also prove that *this* run captured the supplied plan
  file, so an orphaned plan cannot borrow another run's provenance. `experiment_identity.capture.dump_plan` must
  be exactly `true`, and `outputs.plan` must be a nonempty path whose **basename** matches the supplied plan
  file (e.g. `plan_r0.npz`). Only the basename is compared, and it is **never** resolved against the current
  working directory: the plan is guaranteed to be a sibling of its manifest, so an `outputs.plan` recorded
  relative to the run directory or to the caller's CWD (or recorded bare) is accepted. A missing/non-object
  `outputs`/`experiment_identity.capture` or a missing/non-string/empty `outputs.plan` is `manifest_malformed`;
  a `capture.dump_plan` that is not `true` is refused as `manifest_plan_not_captured`; a basename that does not
  match the supplied plan is refused as `mismatched_plan_output`.
- **Config equality (residual-channel E02).** The resolved `config` is canonicalized (JSON-compatible:
  tuples/lists and scalars normalized, sorted keys) and every key/value must be **exactly equal** across the
  compared plans **except** `mesh` and `rs_rows`, the two intended residual-channel ablation variables. A
  different seed, iteration count, smoothing, dtype, fidelity or any other field is a different experiment and
  is refused (`mismatched_config`) rather than silently compared. Matching source/window/weights alone is not
  enough: two plans captured with the same source but different stabilizer settings are not comparable.
- In addition the payloads must agree **exactly** on dimensions, frame count and fps.
- Case names (`case.name`) must be **unique**. `baseline` is the reference when present, otherwise the first path.
- If a manifest is absent or malformed (including a missing/non-object `config`, or a missing/non-object
  `outputs`/`experiment_identity.capture`), if the captured-plan proof fails (`capture.dump_plan` not `true`, or
  an `outputs.plan` basename that does not match the supplied plan), if provenance/window/
  preprocessing/weights/config mismatch, or if dimensions/frame count/fps differ, the CLI prints a JSON error and
  exits `2`. It **never** compares plans whose windows are not proven identical.

## Comparison report contract

Distinct schema `mlxstab.plan_shape_comparison/1` with `ok: true` on success:

| field | meaning |
| --- | --- |
| `provenance` | the matched identities (source sha256, window, preprocessing, weights sha256, the shared reference `config`) plus a per-plan identity record (`case`, `plan_path`, `manifest_path`, `run_id`, `capture_dump_plan`, `output_plan`, the manifest fields) and a `capture` block (`checked_fields`, `dump_plan = true`, `output_plans` keyed by case) proving the supplied plan was captured |
| `equality` | which manifest fields, config keys and plan fields (`dimensions`, `frame_count`, `fps`) were required equal, with the shared values; `checked_capture_fields` lists the per-manifest capture fields (`outputs.plan`, `experiment_identity.capture.dump_plan`) that were checked, `checked_config_keys` lists every resolved config key that was compared and `allowed_config_diff_keys` is the `mesh`/`rs_rows` allowance |
| `selection` | the shared `every`, `samples`, `frames`, `final_frame_included` |
| `cases[]` | per case: `plan` meta, `is_reference`, the single-plan `summary`, the absolute `per_frame` records, and `per_segment` arrays aligned to `selection.frames` |
| `cases[].per_segment[SEG]` | `dev_max_px`, `dev_p95_px` (absolute curvature per frame), `coverage_frac` (absolute source-bounds coverage per frame), and `delta_vs_reference` (element-wise case − reference for each) |
| `not_a_winner_score` | explicit statement that curvature and coverage are **not** collapsed into a single winner/quality score |

On refusal the CLI emits `{"schema": "mlxstab.plan_shape_comparison/1", "ok": false, "plans": [...],
"error": {"code", "message"}}` and exits `2`. Stable codes include `manifest_missing`, `manifest_unreadable`,
`manifest_malformed`, `mismatched_source`, `mismatched_window`, `mismatched_preprocessing`, `mismatched_weights`,
`mismatched_config` (a resolved-config key other than `mesh`/`rs_rows` differs), `manifest_plan_not_captured`
(`experiment_identity.capture.dump_plan` is not `true`), `mismatched_plan_output` (the `outputs.plan` basename
does not match the supplied plan file), `mismatched_dimensions`,
`mismatched_frame_count`, `mismatched_fps`, `duplicate_case`, `too_few_plans`, and `plan_<code>` for a plan that
fails the strict loader.

## Comparison limits

- Same warp-geometry proxy caveats as the single-plan audit: **no media, no rendering**, four segments per frame
  at the shared selection, bilinear mesh/row resolution, and coverage that is geometric only.
- Matching is only as strong as the manifests: it proves the recorded source sha256/window/preprocessing/weights
  sha256 agree and that the resolved config is identical apart from `mesh`/`rs_rows`, and nothing about the media
  content itself.
- The captured-plan proof ties each manifest to the supplied plan file **by basename only** (the plan is a
  sibling of its manifest by construction), so it rules out borrowing another run's `manifest.json`, but it does
  not by itself detect a plan file that was swapped in while keeping the same `plan_r0.npz` name.
- Every resolved config key outside `mesh`/`rs_rows` (seed, iterations, smoothing, dtype, fidelity, ...) must be
  identical; a comparison across different stabilizer settings is refused, not rescaled or averaged.
- All compared plans must share the exact dimensions, frame count and fps; plans captured at a different cadence
  or resolution must be re-captured, not rescaled here.
- Curvature and coverage are reported side by side (per case, segment and frame) with deltas against the
  reference; there is deliberately no combined/winner metric, and no real-footage quality claim is made.

---

# Real plan+motion composition audit (E02)

`bench/real_composition_audit.py` is the **real-motion** companion to the captured-plan audit. The plan audit
probes production `source_coords` on captured plans; this audit additionally loads the **captured fitted
motion** (`mlxstab.stabilize.save_motion_npz` / `bench/run_case.py --dump-motion`) from the same run and
compares production's actual output->source map against the Stage 1 composed candidate
(`composition_candidate.candidate_source_coords`) in three modes:

| mode | candidate call | meaning |
| --- | --- | --- |
| `exact_composition` | `highpass=False` | the exact composition `Phi_t` of the fitted per-pair maps |
| `source_highpass` | `highpass=True`, `smoothing_basis="source"` | candidate with the source-chart temporal high-pass |
| `reference_highpass` | `highpass=True`, `smoothing_basis="reference"` | candidate with the reference-chart temporal high-pass |

It uses the **same four sparse output-space line probes and the same strided frame selection** as
`bench.plan_shape_audit` (final frame always included). Both high-pass charts are computed with one shared
temporal smoothing per basis (`precompute_smoothed`), never re-smoothed per frame.

```bash
python -m bench.real_composition_audit \
  out/bench/e02_t060_baseline_1s/plan_r0.npz out/bench/e02_t060_baseline_1s/motion_r0.npz \
  --every 30 --samples 401                     # JSON to stdout
python -m bench.real_composition_audit PLAN.npz MOTION.npz --out out/real_composition_audit.json
```

## Strict input and provenance

Both NPZs are opened with `np.load(..., allow_pickle=False)`; only plain numeric/unicode arrays are accepted.
The plan is validated through `plan_shape_audit.load_plan_npz`; the motion has its own strict
schema/shape/finiteness validation (`MOTION_SCHEMA` = `mlxstab.motion/4`, required keys, `H [n_fits,3,3]`,
`mesh [n_fits,gr+1,gc+1,2]`, `rows [n_fits,nr,2]`, finite `inlier_frac`/`mover_frac`, the per-fit gate
diagnostics `n_valid_samples`/`n_ransac_inliers`/`fallback_reason_code` (the integer code, mapped back to a
readable `fallback_reason` label) with the `fit_meta_available` flag, the
observation-coverage support masses `mesh_support_mass [n_fits,gr+1,gc+1]` / `row_support_mass [n_fits,nr]`,
and the forward-backward cycle-consistency maps `cycle_err_rms`/`cycle_inconsistency_frac`/`cycle_sample_mass`
with the per-fit `cycle_map_available` flag; legacy `mlxstab.motion/3` (no fit metadata), `mlxstab.motion/2`
(support, no cycle maps) and
`mlxstab.motion/1` payloads still load, missing fields as `None`). Before any geometry the
audit cross-checks the sibling benchmark manifest:

- `experiment_identity.capture.dump_plan` and `.dump_motion` must both be exactly `true`;
- `outputs.plan`/`outputs.motion` must be nonempty paths whose **basenames** match the supplied plan/motion
  files (never resolved against the CWD; both artifacts must be siblings of the manifest);
- the resolved `config` must reconstruct a `mlxstab.stabilize.Config`, and its `mesh`/`rs_rows` must agree with
  the captured plan/motion array shapes.

It then reconstructs `Plan`, `Motion`, `Fit` and `Config` and enforces the candidate's composition scope:
`stride == 1`, matching dimensions/fps, and `len(plan.G) == len(motion.fits) + 1 == motion.n_frames`. Anything
else is refused with a stable code (`manifest_missing`, `manifest_unreadable`, `manifest_malformed`,
`manifest_dump_plan_not_captured`, `manifest_dump_motion_not_captured`, `mismatched_plan_output`,
`mismatched_motion_output`, `artifacts_not_siblings`, `unsupported_stride`, `mismatched_lengths`,
`inconsistent_plan_motion`, `inconsistent_config`, `bad_config`, `plan_<code>`, and the motion loader's
`unreadable`/`missing_keys`/`bad_schema`/`bad_scalar`/`bad_dims`/`bad_shape`/`nonfinite`). The CLI emits a
JSON-safe error object and exits `2`.

## Report contract

Distinct schema `mlxstab.real_composition_audit/1` with `ok: true` on success. For every selected frame and
each of the four probes it records, for production and for each candidate mode:

- best-fit-line deviation of the mapped probe (`mapped.dev_max_px`/`dev_p95_px`/span, via the shared
  `shape_audit.best_fit_line` metric) and the source-bounds coverage (`coverage`);
- for candidate modes, the pointwise displacement from the production map
  (`displacement_from_production_px.max_px`/`mean_px`).

`summary` carries the per-segment/per-mode series maxima and minima; `selection` records the exact frame
indices. There is deliberately **no composite/winner score**: `no_composite_winner_score` states that the
curvature, coverage and displacement series are reported side by side and never collapsed into a ranking.

## Scope and honesty

This is an experiment on **real fitted motions**. It is **not** object/tree detection, scene understanding, a
perceptual quality evaluation, or evidence that a tree looks straight to a viewer, and it is **not** a
production integration: `composition_candidate` is not imported by `stabilize`/`source_coords`, and no
production code, default or output is changed. It probes four straight output-space segments per selected
frame (not the full field or real scene content), samples mesh/row corrections bilinearly at the probe points,
and decodes no media and renders no frames. The open decision on the temporal-smoothing chart is unchanged:
both charts are reported and neither is selected.

## Next composition-candidate gate

The recorded t060/t800 E02 plan-audit results (see `STATUS.md`) show that residual fields introduce measurable
warp-geometry curvature and that global-only is near-straight but worse in output jitter/wobble. The next gate
is to run this real plan+motion audit on those same captured windows (capturing motion for new runs with
`--dump-motion`) and review, per window, the candidate modes' curvature/coverage/displacement against
production alongside longer and held-out windows with visual review; only then is production integration
considered. Nothing here changes defaults or selects a chart.

## Rendered companion: dense-map approximation

The rendered companion `bench/real_composition_render.py` turns the same three candidate maps into pixels with
the production sampler. Re-deriving the exact full-history composition `Phi_t` for every output frame is O(T^2)
over the clip length and was impractical for the 300-output capture, so the renderer forward-propagates `Phi_t`
**once per motion fit** on a **padded world-pixel lattice** with `composition_candidate.forward_apply`, then
bilinearly interpolates it at `V_t(x) = inv(Psi_t) @ plan.G[t](x)`. The lattice is the output pixel-center
rectangle's `V_t` image over every requested source frame (the four output pixel-center corners through `V_t`),
grown by a one-pixel interpolation guard and stored in world pixel coordinates (a negative origin is allowed);
this keeps every query inside the lattice even when `V_t(x)` leaves the frame (at frame 0 the current capture
reaches `y ~ 1087` for `H=1080`), so no border clamp inflates the error. A projective horizon/sign change of
`V_t` and a non-finite mapping are refused. This is a **dense-grid interpolation approximation to the
continuous exact composition, not mathematically exact**: it is exact on lattice nodes and interpolated between
them, and building every frame is O(T) instead of O(T^2). The provenance JSON names it
(`approximation.name = dense_grid_interpolated_forward_composition`), records the grid, grid shape,
`grid_origin`, `padding`, the raw corner query bounds, interpolation and reference-grid resolution, and compares
the dense map directly against the exact `composition_candidate.candidate_source_coords` on a sparse inset grid
of sub-pixel output points at five representative frames, recording per-frame and aggregate max/mean/p95 pixel
error. The geometry-only audit above is unchanged and still evaluates the exact composition pointwise on its
sparse probes; only the *rendered* companion uses the dense-map propagation.
