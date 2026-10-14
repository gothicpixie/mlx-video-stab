# mlxstab

Video stabilization on Apple Silicon with [MLX](https://github.com/ml-explore/mlx): a from-scratch MLX port of
[RAFT](https://github.com/princeton-vl/RAFT) optical flow, a Metal image-resampling kernel, and a quality-first, no-crop
stabilizer that renders to the full original frame and hands you a mask of the pixels that had no source (to be filled by a
video inpainter in a later round).

Status: round 1 (RAFT, warp kernel, stabilizer, metrics, benchmarks). Not yet published.

See the [development plan](PLAN.md) for the roadmap, experiment register, acceptance criteria, and Rust migration decision.

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

| resolution | MLX (GPU) | PyTorch MPS | PyTorch CPU |
|---|---|---|---|
| 960x544 (540p padded to 8) | 0.45 s | 0.39 s | 1.58 s |
| 1920x1088 (1080p padded to 8) | 2.05 s | 2.47 s | 11.53 s |

MLX is on par with PyTorch-MPS at 540p (0.45 s vs 0.39 s), about 20% faster at 1080p (2.05 s vs 2.47 s) and 5 to 6x faster than CPU. Speed was not optimised (fp32 everywhere, correlation volume materialised in full).

Stabilizer throughput (whole pipeline incl. flow in both directions, planning, Lanczos render and x264 encode):

| clip (synthetic) | flow | flow analysis | render + encode | whole pipeline |
|---|---|---|---|---|
| 640x360, 60 frames | 1/1 res, 12 iters | 2.01 fps | 110.4 fps | 1.89 fps |
| 1920x1080, 30 frames | 1/2 res, 20 iters | 0.37 fps | 15.8 fps | 0.36 fps |
| 1920x1080, 12 frames | 1/1 res, 12 iters | 0.14 fps | 6.4 fps | 0.14 fps |

The flow analysis (RAFT forward + backward per pair, RANSAC, mesh fit) is the cost; planning (LP) takes seconds and rendering with Lanczos-3 is real-time-ish. A 10 s 1080i camcorder clip (600 field-frames after bwdif) at 1/2-resolution flow and 20 iterations took about 26 minutes end to end (0.39 fps). Throughput is deliberately secondary to quality in this round.

## Metrics

Measured on the videos themselves with the same RAFT-based analysis for input and output (`bench/evaluate.py`):
stability and cropping/distortion as in Liu et al. 2013 (*Bundled camera paths*), residual jitter in px, and a rolling-shutter wobble
measure (row-wise residual motion after removing its linear trend).

**Synthetic clip** (procedural, 640x360, 150 frames, parallax layers + moving subject + jitter + rolling shutter; RAFT 12 iterations; fidelity 1.0 as it was then):

| | stability | jitter px | wobble px |
|---|---|---|---|
| input | 0.051 | 3.43 | 0.181 |
| mlxstab | 0.917 | 0.12 | 0.062 |

**Real handheld 1080i camcorder footage** (two private 10 s windows; bwdif to 59.94p, outputs taken at 29.97 fps; stabilizer flow at half resolution, 20 iterations; metrics measured with quarter-resolution flow, in 1080p pixels). The baseline is the OpenCV stabilizer of the author's own film pipeline (LK tracking at 640 px, similarity path, Gaussian sigma 15 frames, 6% crop-in, reflected borders):

| clip | | stability | jitter px | wobble px |
|---|---|---|---|---|
| A | input | 0.841 | 1.43 | 0.061 |
| A | OpenCV baseline (6% crop) | 0.756 | 0.86 | 0.064 |
| A | mlxstab (fidelity 0.1) | 0.942 | 0.46 | 0.054 |
| B | input | 0.496 | 4.97 | 0.081 |
| B | OpenCV baseline (6% crop) | 0.692 | 3.22 | 0.091 |
| B | mlxstab (fidelity 1.0, superseded) | 0.594 | 2.61 | 0.063 |

mlxstab on clip A: mean border 1.7% of pixels (max 8.4% in one frame), cropping ratio 0.983, distortion score (min over frames) 1.000; the baseline crops 6% of every frame. Clip B was measured before the fidelity default was tuned (1.0 leaks residual jitter into the path) and has not been re-run: there mlxstab had lower jitter and wobble than the baseline but a lower stability score.

How to read these. *Stability* (Liu et al.) is the share of path energy in the lowest frequency bins. It is scale-free, so an almost perfectly static path scores badly because the little energy left sits at the knots of the piecewise path; read it together with *jitter*, and treat it as noisy on short windows. Ablation on the planned paths of clip A: fidelity 0.1 gave stability 0.94 and high-pass jitter 1.5 px, Gaussian sigma 15 gave 0.87 and 2.3, fidelity 1.0 gave 0.87 and 3.8, and pure L1 (fidelity 0) gave a near-static path with stability 0.39. On a 4 s excerpt, the row-wise (rolling-shutter) plus mesh residuals lowered the wobble measure from 0.035 (global path only) to 0.030, against 0.037 for the input, and a lower fidelity lowered it further (0.023): the mesh/row stage helps modestly, the global L1 path is what moves jitter. Real public-domain footage was not downloaded in this round (it needs approval), so the synthetic clip is the reproducible public test.

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
