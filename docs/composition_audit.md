# Residual composition audit (Stage 0, E02 preparation)

`bench/composition_audit.py` is a bounded analytic geometry experiment with exact ground truth. It asks a
single question: **when the stabilizer accumulates per-pair mesh/row residuals on the fixed output grid, is
that the same as transporting and composing the residuals through the camera motion?** It introduces no
production change, no default change, no media, and no private data. It replaces the reviewed root
counterexample (three 64x48 frames, two identical 10 degree rotations, constant `u=[8,0]` residual) with a
reproducible fixture set and specifies the Stage 1 fix and its decision gate.

The reviewed counterexample is reproduced: production `source_coords` at frame 2 differs from the exact
composition by **mean 1.39449147 px / max 1.39449561 px**, matching `||u - R_linear u|| = 1.39449188 px`.
That error is a genuine vector-transport error, not a numerical tolerance: the global part of the map is
composed correctly, but the residual *vector* is added in the wrong frame.

## Conventions and signs

Coordinates are pixel centres on integers; a `W x H` frame spans `[-0.5, W-0.5] x [-0.5, H-0.5]`
(`geometry.corners`). `x` is an **output** (virtual-camera) coordinate. `p` is a **source-frame** coordinate.

Sign convention: `rotation_about_clockwise(W, H, deg)` rotates visual content **clockwise** about the image
centre (a positive angle moves a point right of centre downward). With the image y axis pointing down this
is the `[[cos,-sin],[sin,cos]]` matrix, the inverse sign of `geometry.rotate_quad` (visual
counter-clockwise).

Per analysed pair `k` the forward frame-to-frame motion is

```
F_k(p) = H_k(p) + r_k(p) + row_k(p_y)          frame k -> frame k+1
```

- `H_k` maps frame `k` to frame `k+1` (`motion.Fit.H`), i.e. a point at `p` in frame `k` appears at `H_k(p)`
  in frame `k+1`.
- `r_k` is a residual **displacement in the destination frame `k+1`**, parameterised by the *source*
  position `p` in frame `k`. This is how `motion.fit_motion` measures it: the observed destination is
  `H(pts) + resid`, so `resid` is a vector added after the global map, living in `k+1`.
- `row_k` is the x-independent rolling-shutter row residual, also a `k+1` displacement, parameterised by the
  frame-`k` row.
- `r_k`/`row_k` are discrete vertex/band fields, bilinearly interpolated with border clamping exactly like
  `warp.sample`. Vertex `(i,j)` sits at `x=-0.5+i*W/gc`, `y=-0.5+j*H/gr`; row band `j` at
  `y=-0.5+(j+0.5)*H/nr`.

The virtual camera is **locked** to the first analysed frame (the experiment pins every smoothed path series
to its first sample). The output frame therefore *is* reference frame 0, so the output-to-source map is the
forward composition

```
Phi_0(x) = x,        Phi_{k+1}(x) = F_k(Phi_k(x))
```

This is the **oracle** (ground truth). `plan.G[t]` is only the global part `G_t = H_{t-1} ... H_0` of it.

The **current fixed-grid sum** (production) evaluates the cumulative residual at the global source
coordinate instead of at the correct preimage:

```
naive_t(x) = G_t(x) + (sum_{k<t} r_k)(G_t(x))   [rows sampled after the mesh shift]
```

Writing `Phi_k(x) = G_k(x) + delta_k(x)` gives the exact affine transport recursion with
`A_k = H_k[:2,:2]`:

```
delta_0 = 0,     delta_{k+1}(x) = A_k delta_k(x) + r_k(G_k(x) + delta_k(x))
```

`compose_oracle` evaluates the composition directly; `transported_field` evaluates this recursion. For
similarity/affine globals the two agree to machine precision (tested), so the displacement representation is
an exact oracle, not an approximation. `naive_from_plan` is an independent NumPy replication of
`source_coords` using the plan's own arrays, and `production_field` calls the real MLX/Metal path.

Residual frame summary: `r_k` lives in frame `k+1`; the cumulative residual the production path wants to add
at frame `t` must first be carried through `A_{k+1} ... A_{t-1}`, which is exactly what the recursion does.
`A_k^{-1}` is needed only for the reverse (source-to-output / visibility) direction.

## Fixtures

All fixtures are deterministic and built in memory. `mesh=(gc,gr)` and `rs_rows` are chosen per fixture.

| fixture | globals | residual | what it isolates |
| --- | --- | --- | --- |
| `rigid_rotation_control` | two 10 deg rotations | none | composition control (addition trivially valid) |
| `translation_only_residual` | identity | constant `u=[8,0]` on both pairs | `A_k=I` + constant => addition exact |
| `constant_residual_rotation` | two 10 deg rotations | constant `u=[8,0]` on pair 0 | pure vector transport |
| `spatial_ramp_translation` | 6 px x translation | linear `r(p)=diag(.02,.01) p` | residual resampling/location error |
| `parallax_two_layer` | 6 deg rotation + background translation | supported two-layer differential | transport + location + support boundary |
| `rows_mesh_order` | identity | constant mesh y-shift 3 px + linear row field | mesh/row evaluation order |
| `rotation_translation_border` | 10 deg rotation + 20 px x translation | constant `u=[8,0]` on pair 0 | transport error with large border exclusion |

The parallax fixture states its own physical convention explicitly: `displacement_px = (f_px / Z) *
translation`, with `f_px = 100`, `Z_bg = 5.0`, `Z_fg = 1.25` in arbitrary units, so `f_px/Z` is inverse depth.
The differential on the foreground is `(w_fg - w_bg)*T = 3 px`. This is deliberately **not** a metric-depth
label and does not touch `synth.render`'s existing defaults (`depths=(1, .55, .25)`), whose values multiply
camera translation directly and are translation/disparity factors, not physical depths.

## Measured results

Production-enabled run (actual `source_coords`), frame 2 unless noted. `naive_vs_oracle` is the
fixed-grid-sum error against the composed ground truth; `production_vs_naive` is MLX float32 vs the NumPy
replication (a numerical-agreement check, not a quality metric).

| fixture | `naive_vs_oracle` mean / max (px) | `production_vs_naive` max (px) |
| --- | --- | --- |
| `rigid_rotation_control` | 3.0e-14 / 6.6e-14 | 6.9e-06 |
| `translation_only_residual` | 1.9e-14 / 3.2e-14 | 3.2e-14 |
| `constant_residual_rotation` | 1.39449 / 1.39449 | 7.8e-06 |
| `spatial_ramp_translation` | 0.31052 / 0.36003 | 3.8e-06 |
| `parallax_two_layer` | 0.09988 / 0.78964 | 1.2e-05 |
| `rows_mesh_order` (frame 1) | 0.05000 / 0.06000 | 1.7e-06 |
| `rotation_translation_border` | 1.39449 / 1.39449 | 8.6e-06 |

Frame 1 of `constant_residual_rotation` is exact (constant residual: the sampling position is irrelevant);
frame 2 is wrong by exactly `||u - R_linear u||`. `spatial_ramp_translation` frame 1 is wrong by the
analytically derived vector field `L T` and frame 2 by `3 L T - L^2 x` (the tests assert the vectors, not
just a scalar). `parallax_two_layer` frame-2 error is localised: mean **0.343 px on the foreground** vs
**0.039 px on the background**. `rows_mesh_order` is wrong by `0.06 px` because production samples the row
field at the mesh-shifted row (`row(y+3) - row(y) = 3*slope`) while the model evaluates rows at the source
row.

Support/visibility is reported, never silently dropped. `oracle_in_bounds_frac` falls to `0.807` for
`constant_residual_rotation`, `0.765` for `spatial_ramp_translation`, and `0.233` for
`rotation_translation_border` at frame 2. For the border fixture, **76.7%** of output pixels have no source
at frame 2 under the oracle composition. The transport error remains on the supported subset
(`1.39449 px`), so this is not fixed by cropping the border. Error counts are always over all output pixels,
with a separate in-bounds figure.

The affine transport recursion reproduces the composition to `<= 3.2e-14 px` on every fixture, including the
spatially varying ones, so the frame-consistency of the displacement representation is exact for
similarity/affine globals (a projective `H` would require the full composed map).

## When simple addition is valid

The naive sum is exact only when both conditions hold:

1. **No linear transport:** `A_k = I` for every pair (pure-translation globals). A rotation or scale rotates
   the residual vector between frames.
2. **Position-invariant residual:** the residual is constant, or the evaluation point does not change the
   value (e.g. `H_k` leaves the residual's argument fixed). Otherwise the naive sum samples the field at
   `G_t(x)` instead of the true preimage.

The controls pin both: `rigid_rotation_control` and `translation_only_residual` are exact to `< 1e-9 px`;
`constant_residual_rotation` and `spatial_ramp_translation` violate (1) and (2) respectively and are wrong by
the closed-form amounts. If only (1) holds but the field varies (translation globals + ramp), the error is
`r(x+T) - r(x)`, i.e. the location error. If only (2) holds but the globals rotate, the error is the vector
transport `(I - A) delta`.

## Inverse convergence

The locked-camera audit needs no inversion: the oracle is already output-to-source. Inversion is only needed
for the reverse direction (source-to-output / visibility). `inverse_audit()` records:

- **well-conditioned residual + similarity:** the damped fixed point converges in 8 iterations to
  `< 1e-6 px`.
- **invertible similarity with scale 2.5:** the fixed-point iteration has eigenvalue `|1 - 2.5| > 1` and
  **diverges** (not converged after 50 iterations). Border clamping can rescue a diverging iterate outside
  the frame, so this control deliberately uses a scaled similarity with no residual; a pure edge-clamped
  fold can otherwise converge to a spurious preimage.
- **orientation-reversing residual** (`1 - 2 = -1` Jacobian): `negative_det_frac = 1.0`. Such a map is not a
  valid warp and must be rejected, not inverted.

Conclusion: a Stage 1 implementation must use closed-form affine/similarity inverses (exact), reserve
iterative inversion for genuinely nonlinear maps, and explicitly detect non-convergence and `det <= 0`
fold-over instead of returning a coordinate.

## Stage 1 implementation specification and decision gate

**Specification.** Replace the fixed-grid cumulative sum with a frame-consistent composition:

1. Keep residuals in their measured frame (`r_k` is a frame-`k+1` displacement parameterised by frame `k`).
2. Compose forward: `Phi_{k+1} = F_k ∘ Phi_k`, or equivalently transport the displacement
   `delta_{k+1} = A_k delta_k + r_k(G_k + delta_k)` for affine globals. Evaluate each residual at its own
   preimage, not at `G_t(x)`.
3. Preserve the fit model's mesh/row order (`H`, then residual `r_k(p)`, then `row_k(p_y)`) and document the
   evaluation point; the current `source_coords` mesh-then-row order is a separate, measurable discrepancy.
4. Recompute visibility and border support from the composed map; keep `valid_mask` semantics. Do not
   assume the oracle is a drop-in fix: it changes cover support and can map pixels out of frame.
5. For general projective `H`, compose the full homography rather than only `A_k`; use closed-form affine
   inverses and reject/fall back on non-convergent or `det <= 0` maps.
6. Keep a global-only fallback for weak or degenerate motion fits.

**Decision gate.**

- Analytic equivalence: `transported_vs_oracle` and any new implementation match the composed oracle to
  `< 0.05 px` on all fixtures; controls stay `< 1e-9 px`.
- Reproduce the counterexample: `constant_residual_rotation` frame 2 drops from `1.39449 px` to `< 0.05 px`,
  and the closed-form `||u - R_linear u||` is matched, not merely reduced.
- No regression in support: out-of-frame fractions and border masks are reported and no worse than the
  current path for the same virtual camera; no `det <= 0` fold-over is introduced.
- Then proceed to the E02 real-footage ablation (global-only, mesh-only, row-only, combined) using this
  convention, under Stage 0 measurement, before any new geometry model.

This audit does **not** establish that a real motion fit assigns a uniform translation to residuals, and it
does not by itself explain the reported tree deformation. It isolates one concrete mechanism in the plan /
`source_coords` path and fixes the semantics to test next.

## Reproduction

Pure geometry (no GPU, no lock):

```bash
.venv/bin/python -m bench.composition_audit --out-dir out/composition_audit
```

With the real `source_coords` path (MLX/Metal; must hold the shared heavy lock):

```bash
python - <<'PY'
from bench.heavylock import HeavyLock
import subprocess, sys
with HeavyLock("mlxstab composition audit"):
    sys.exit(subprocess.call([".venv/bin/python", "-m", "bench.composition_audit",
                              "--out-dir", "out/composition_audit", "--production"]))
PY
```

Outputs (under ignored `out/`, no private media): `fixtures.npz` (per-pair homographies and residual
vertex fields) and `results.json` (JSON-safe, `allow_nan=False`), plus the same data returned by
`run_audit()`. The committed source of truth is the module and tests; the JSON is regenerable evidence.

Focused tests: `tests/test_composition_audit.py` (15 tests) covers the closed-form oracle, the affine
transport recursion, the pinned plan's global composition, the addition-valid controls, the transport
counterexample, the location error, the two-layer localisation, mesh/row order, support/visibility, the
inverse convergence/failure modes, JSON safety, and the actual production path. Several tests assert that
the naive result is *wrong* and that the oracle matches the independent closed form, rather than freezing the
current output as desired.

## Limitations

- Analytic 2D fixtures with similarity/affine globals; a projective `H` is composed by the oracle but not
  isolated as its own fixture.
- Residual fields are discrete mesh/row vertex fields, so a layer boundary is one cell wide and is excluded
  from exact-vector assertions; the audit reports regional means rather than sub-cell truth.
- The border/support fractions are those of the locked virtual camera in each fixture, not of a real
  planner's crop.
- No resampling fidelity, occlusion, motion fit, or perceptual claim is made. `synth.render`'s `depths` are
  not metric depths and are deliberately left unchanged.
- The inverse controls demonstrate failure modes of a fixed-point inverse; they are not a complete
  invertibility analysis of a production warp.
