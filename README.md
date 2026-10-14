# mlxstab

Video stabilization on Apple Silicon with [MLX](https://github.com/ml-explore/mlx): a from-scratch MLX port of
[RAFT](https://github.com/princeton-vl/RAFT) optical flow, a Metal image-resampling kernel, and a quality-first, no-crop
stabilizer that renders to the full original frame and hands you a mask of the pixels that had no source (to be filled by a
video inpainter in a later round).

Status: round 1 (RAFT, warp kernel, stabilizer, metrics, benchmarks). Not yet published.

## What is in it

| module | what |
|---|---|
| `mlxstab/warp.py` | Bilinear / Catmull-Rom / Lanczos-3 sampler as an `mx.fast.metal_kernel`; `grid_sample` look-alike (NHWC), zeros/border padding, `align_corners` both ways. Forward only. |
| `mlxstab/raft/` | RAFT-things (full model): feature + context encoders, all-pairs correlation pyramid + lookup, ConvGRU update block, convex upsampling; `.pth` -> safetensors converter. |
| `mlxstab/stabilize.py` | The stabilizer (below). CLI: `python -m mlxstab`. |
| `mlxstab/metrics.py` | Stability, jitter, rolling-shutter wobble, cropping, distortion. |
| `mlxstab/synth.py` | Procedural test clip with parallax layers, a moving subject, jitter and rolling shutter (CC0 by construction). |

### Stabilizer design (quality first, speed second)

1. **Interlaced sources** are detected with ffprobe (`field_order`) and deinterlaced first with `bwdif=send_field` (59.94i -> 59.94p),
   before any flow.
2. **Flow**: RAFT forward and backward per consecutive pair, 24 iterations by default, warm-started from the previous pair's flow,
   full resolution by default (`--flow-downscale 2` trades accuracy for time).
3. **Motion model, spatially varying**: a robust global similarity (affine / homography selectable) fitted by RANSAC on flow samples
   with forward-backward-inconsistent samples masked out, plus a *residual mesh* (16x9 cells by default, robust kernel regression of
   the flow left over after the global model) and per-row-band residuals for the **rolling shutter**. A frame that is too low-texture
   falls back to "no global motion" and zero mesh (reported as `fallback_frames`).
4. **Subject vs camera**: samples on independently moving things (coherent blobs whose flow disagrees with the dominant motion, found
   after the first fit) are excluded from the camera fit and from the mesh.
5. **Camera path**: the global path is accumulated in corner space and decomposed into translation, rotation, log-scale and a small
   non-similar residual. Each is smoothed with an **L1-optimal path** (Grundmann et al. 2011: minimise |velocity|, |acceleration| and
   |jerk| in L1, so the result is made of static, constant-velocity and constant-acceleration segments, i.e. a tripod or a dolly)
   solved as an LP (HiGHS), with a box constraint `|path - raw| <= budget` (the **border budget**) and a small L1 pull to the raw path
   so the border stays small. Mesh and row residual paths get the same treatment. `--smooth gauss --sigma N` selects a Gaussian instead.
6. **Roll levelling**: an optional absolute roll series (degrees, positive = counter-clockwise lean) is folded into the target path
   before smoothing, so levelling and stabilization are one transform.
7. **One resample**: stabilization, mesh, rolling-shutter correction, roll and optional zoom are composed into a single source-coordinate
   field and the original frame is resampled once with Lanczos-3 (bilinear is used only inside the flow network).
8. **Border mask** (first-class output): for each frame, the pixels whose source position falls outside the original frame
   (`--mask mask.mp4`, white = no source, lossless).

```
python -m mlxstab in.mp4 out.mp4 --mask mask.mp4 [--sigma 15] [--roll roll.json] [--budget 0.04] [--smooth l1|gauss]
                                  [--mesh 16x9|0x0] [--rs-rows 18] [--iters 24] [--flow-downscale 1|2|4] [--start S --duration D] [--fps F]
```
`roll.json`: `[deg per frame]`, or `[[t_seconds, deg], ...]`, or `{"series": ...}`.

## Setup

```
uv venv .venv && uv pip install -p .venv/bin/python mlx numpy scipy safetensors     # runtime
uv pip install -p .venv/bin/python torch pytest pillow opencv-python                # to run the tests / benchmarks
git clone https://github.com/princeton-vl/RAFT vendor/reference/RAFT               # reference code (BSD-3), not committed
# download the RAFT release models.zip (see the RAFT README), extract raft-things.pth into vendor/weights/things/, then
python -I -m mlxstab.raft.convert vendor/weights/things/raft-things.pth vendor/weights/conv/raft-things.safetensors
.claude/verify.sh        # fast, offline, no weights needed (weight tests skip when absent)
```
Downloaded files are untrusted data: the converter loads the pickle with `torch.load(weights_only=True)`, and everything runs with `python -I`.

## Parity with PyTorch

RAFT-things at 12 iterations, MLX (fp32, Apple GPU) vs the unmodified reference (CPU):

| input | final flow EPE vs torch (px) |
|---|---|
| synthetic textured shift, 160x192 | 2.4e-3 |
| reference demo frames (frame_0016 -> frame_0017, 432x1024 crop) | 2.2e-3 |

Per-stage relative max error (error / max(1, max|ref|)): feature/context encoders 1e-5 .. 4e-2 (real weights, large frames: activation
magnitudes grow), correlation volume and lookup ~7e-4, update block ~1e-5..1e-2, convex upsampling ~1e-2 relative to flows of tens of
px. `grid_sample`: max abs error 1.2e-7 .. 2.4e-7 vs `torch.nn.functional.grid_sample` (all four padding / align_corners combinations).
Note: the reference produces NaN when the coarsest correlation level is 1 px wide (inputs under about 128 px per side); this port does not.

## Benchmarks

RAFT-things, 12 iterations, 1 pair, fp32, M-series Mac with 24 GB (median of 3; CPU at 1080p: 2). The torch backends run the reference code.

BENCH_TABLE

Stabilizer throughput (whole pipeline incl. flow in both directions, planning, Lanczos render and x264 encode):

STAB_TABLE

## Metrics

Measured on the videos themselves with the same RAFT-based analysis for input and output (`bench/evaluate.py`):
stability and cropping/distortion as in Liu et al. 2013 (*Bundled camera paths*), residual jitter in px, and a rolling-shutter wobble
measure (row-wise residual motion after removing its linear trend).

METRICS_TABLE

## Credits and licenses

- Code in this repository: MIT (see `LICENSE`).
- **RAFT**: Zachary Teed and Jia Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020
  ([arXiv:2003.12039](https://arxiv.org/abs/2003.12039)); reference implementation https://github.com/princeton-vl/RAFT, BSD 3-Clause.
  This port follows that architecture; weights are not distributed here, fetch them from the RAFT release. The weights were trained on
  FlyingChairs and FlyingThings3D; check their dataset terms for your use.
- L1 optimal camera paths: Grundmann, Kwatra, Essa, *Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths*, CVPR 2011.
- Metrics: Liu, Yuan, Tan, Sun, *Bundled Camera Paths for Video Stabilization*, SIGGRAPH 2013. Mesh residuals follow the spirit of
  MeshFlow (Liu et al. 2016).

## Roadmap

- **Round 2: inpainting** the border mask with [ProPainter](https://github.com/sczhou/ProPainter). Its weights are released under the
  S-Lab license (non-commercial), so users fetch them themselves; nothing from ProPainter is vendored or distributed here. The mask
  output is already in the format an inpainter wants. ProPainter needs deformable convolution, which MLX does not have: expect a custom
  Metal kernel (or sampling-based reformulation) for it.
- Backward pass for the warp kernel; fp16 correlation; tiled/streamed correlation so full-res 1080p/4K flow fits comfortably.
- Chunked, streaming L1 solve for hour-long clips; per-vertex L1 (currently solved per mesh vertex, windowed).
- Roll estimation (this round takes the roll series as input).
