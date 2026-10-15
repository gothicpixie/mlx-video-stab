# Complete-warp shape and coverage diagnostics

`mlxstab.warpmetrics` measures the **complete** source-coordinate field that rendering actually uses:
global homography + residual mesh + rolling-shutter row correction. It is separate from the historical
`distortion` score, which is unchanged and only looks at the global homography (see *Compatibility* below).

This is a diagnostics/measurement module. It does not change rendering, the border mask, or the selection of
frames, and it adds no quality claims by itself.

## Conventions

**Sampling.** The field is `src[y, x] = (sx, sy)` mapping an *output* (virtual) pixel centre `(x, y)` to a
source-image coordinate, evaluated on the integer output grid (source pixels, pixel centres on integers).
This is exactly the field built by `stabilize.source_coords`.

**Derivatives.** Local shape uses **forward differences between adjacent pixel centres**. Each 1x1 output
cell with corners `(x,y) -> (x+1,y) -> (x+1,y+1) -> (x,y+1)` is a bilinear patch, so its **four corner
Jacobians** are evaluated (the top/bottom x-edge differences and the left/right y-edge differences combined in
all four pairings):

```
J_TL = [ sx(x+1,y)-sx(x,y)   sx(x,y+1)-sx(x,y) ]   ...and likewise at TR, BL, BR
       [ sy(x+1,y)-sy(x,y)   sy(x,y+1)-sy(x,y) ]
```

Derivatives are in source pixels per output pixel (pixel spacing 1, not normalised further). Forward
differences are used deliberately: a centred difference cancels an alternating (checkerboard) fold, and a
single top-left Jacobian misses a fold, non-finite corner or concave cell at the opposite corner. The signed
determinant of a corner Jacobian is the oriented area there: positive = orientation preserved, negative =
folded, near zero = collapsed. A bilinear patch's determinant is affine (the `uv` terms cancel), so its signed
extrema lie on the corners and the four corners bound the signed determinant. `det_negative_frac` flags any
corner fold, including a mixed-sign (concave) cell whose interior necessarily reaches zero; `det_collapse_frac`
is a **corner**-collapse fraction and does not by itself bound interior collapse of a mixed-sign cell.

**Coverage.** `center_valid` is the existing pixel-edge convention from `warp.valid_mask`:
`-0.5 <= sx <= W-0.5` and `-0.5 <= sy <= H-0.5`. `full_support` additionally requires every interpolation tap
with non-zero weight to lie inside the source pixel array `[0, W-1] x [0, H-1]`. A tap is treated as zero when
`|weight| <= tol` (default `1e-6`), so an exact identity warp is fully supported even though the nominal
footprint of an edge pixel extends past the image. Coverage fractions are always over **all** output pixels;
invalid coverage is never removed from the denominator.

## Metrics

| metric | unit | perfect | meaning |
| --- | --- | --- | --- |
| `shape.anisotropy_*` | ratio | 1 | per cell `max` over its four corner Jacobians of `s1/s2` (corners with `s2 <= s2_tol` dropped); a local stretch/squash |
| `shape.stretch_*` | px/px | 1 (context) | per cell `max` corner `s1`, largest principal stretch |
| `shape.shear_deg_*` | degrees | 0 | per cell `max` corner deviation of the mapped axes from a right angle; separates shear from anisotropic scale |
| `shape.det_mean` | area | 1 | mean signed determinant over the four corners of usable cells |
| `shape.det_min`, `det_abs_min` | area | 1, 1 | extreme corner signed determinant / `abs(det)` over usable cells (`<0` fold, `~0` collapse) |
| `shape.det_negative_frac` | fraction | 0 | usable cells with any corner `det < -det_tol` (orientation reversed; includes reflections and folds) |
| `shape.det_collapse_frac` | fraction | 0 | usable cells with any corner `abs(det) <= det_tol` (**corner** collapse; a mixed-sign cell is caught by `det_negative_frac`) |
| `shape.nonfinite_frac` / `nonfinite_cell_frac` | fraction | 0 | sampled positions / cells containing a non-finite coordinate |
| `shape.cells` / `nonfinite_cells` / `overflow_cells` / `usable_cells` | count | – / 0 / 0 / – | cell partition: any non-finite corner / all corners finite but `det` or a singular value overflowed / fully usable (sums to `cells`) |
| `coverage.center_valid_frac` | fraction | 1 | mapped point inside the pixel-edge extent (equals `cropping_ratio`) |
| `coverage.full_support_<kernel>_frac` | fraction | 1 | all non-zero taps of that kernel inside the source array |
| `coverage.partial_support_<kernel>_frac` | fraction | 0 | centre-valid but not full support (border loss) |
| `coverage.nonfinite_frac` | fraction | 0 | non-finite mapped positions |
| `coverage.pixels` / `nonfinite_pixels` | count | – / 0 | output pixels / non-finite mapped positions |

`*` summaries are `mean`, `min`, `max`, `p50`, `p95`. Within one frame they are computed over that frame's
**usable cells** (corner-aggregated as noted above). A cell's anisotropy is the max over its corners with
`s2 > s2_tol`; collapsed corners are dropped but the other corners of the same cell are kept, and a cell with
no usable corner contributes nothing. Non-finite/overflow cells are excluded from all summaries. Across
frames each scalar leaf (including the per-frame `p50`/`p95`) is aggregated to `{"mean","min","max"}`: these
are **means/minima/maxima of per-frame values, not pooled quantiles**. `None` leaves are ignored and a leaf
with no finite values is `null`. Large (but representable) finite means and quantiles are reduced after
scaling, so they stay finite rather than becoming `inf`.

Notes:

- **Reflections and collapses do not score perfect.** A mirror has `anisotropy = 1` but
  `det_negative_frac = 1`; a collapse has `det_collapse_frac > 0`, `det_abs_min = 0`, and undefined `s1/s2`
  (collapsed corners are excluded from the anisotropy ratio, which is documented rather than reported as `inf`).
- **An identity-global + bent row/mesh field is detected.** The global-only `distortion` stays ~1.0 while
  `anisotropy_p95` (bent row) or `shear_deg_p95` (cross-term mesh) rises; the tests pin both.
- **Degenerate dimensions and non-finite mappings are explicit.** `shape.degenerate` is set and cell metrics
  are `None` for `H < 2` or `W < 2`; coverage is still reported for a single row/column and `None` for an
  empty image. No `NaN`/`Inf` is ever emitted (all values are JSON-safe).
- **Overflow is explicit, not a perfect score.** Finite coordinates large enough to overflow the determinant
  or singular-value arithmetic mark the cell `overflow_cells`; those cells are excluded from summaries, and
  if no usable cell remains the summaries are `None`. Singular values use a scale-normalised 2x2 SVD with
  `s2 = |det| / s1`, avoiding the cancellation in `sqrt((E - disc)/2)` on high-shear matrices.

## Enabling and report schema

- Python API: `stabilize.Config(warp_diag=True)` and the `stabilize.warp_diagnostics(plan, cfg, W, H)` helper.
- CLI: `python -m mlxstab in.mp4 out.mp4 --report report.json --warp-diag`.
- Rendering reuses the source field it already computed (`render_frame(..., diagnose=True)`); it does not
  build the warp twice. `warp_diagnostics` recomputes it for callers that only want measurements.

The report keeps the historical fields and adds an explicitly scoped block; when `warp_diag` is off the new
field is `null`:

```json
{
  "distortion_scope": "global_homography_only",
  "distortion_min": 0.99,
  "distortion_mean": 1.0,
  "warp_diag": {
    "shape.anisotropy_p95": {"mean": 1.02, "min": 1.01, "max": 1.31},
    "shape.det_negative_frac": {"mean": 0.0, "min": 0.0, "max": 0.0},
    "shape.det_collapse_frac": {"mean": 0.0, "min": 0.0, "max": 0.0},
    "coverage.center_valid_frac": {"mean": 0.983, "min": 0.97, "max": 1.0},
    "coverage.full_support_lanczos3_frac": {"mean": 0.91, "min": 0.88, "max": 0.94}
  }
}
```

`distortion_min` / `distortion_mean` are unchanged Liu-2013 global-homography scores (`None` for a
degenerate/non-finite global mapping). `distortion_scope` labels them so they are not confused with the
complete-warp diagnostics.

## Measured overhead

On a synthetic 1920x1080 field (16x9 mesh, 18 row bands, all three kernels, one frame, min of 3 runs on the
development M-series Mac): render-only **~17 ms**, render + diagnostics **~755 ms**, diagnostics alone
**~677 ms** (`_coverage` ~432 ms, `_shape` ~245 ms). The cost is why complete-warp diagnostics are opt-in:
they are suitable for analysis/regression reports, not for ordinary rendering.

## Limitations

- These are geometric measurements of the warp field. They say nothing about resampling fidelity, ghosting,
  temporal flicker, or perceptual quality.
- Shape uses the four corner Jacobians of each bilinear cell; sub-cell structure is bounded (the bilinear
  determinant's extrema are at the corners) but is not sampled more finely than cell resolution.
- The `|weight| <= tol` rule and the `det`/`s2` tolerances are analysis choices, not calibrated quality
  thresholds. No pass/fail gate is baked into the report.
- RAFT precision fixture parity (`tests/test_raft.py`) validates model outputs on demo frames only. It does
  **not** establish the full-clip quality of fp16 approximations.
- Real-footage baselines and the reported tree-deformation / floaty-motion diagnosis are still outstanding
  Stage 0 work.
