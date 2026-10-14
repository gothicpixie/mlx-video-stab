# mlxstab status (parked after round 1b)

## What works

- `mlxstab/warp.py`: Metal sampler (bilinear / Catmull-Rom / Lanczos-3, zeros/border, align_corners). `grid_sample` max abs error vs torch 1.2e-7 .. 2.4e-7. Forward only.
- `mlxstab/raft/`: RAFT-things in MLX + `.pth` -> safetensors converter. Final flow EPE vs PyTorch at 12 iterations: 2.2e-3 px (reference demo frames), 2.4e-3 px (synthetic); batch-2 bidirectional path: 2.2e-3 px fp32, 2.7e-3 px fp16. Gate is < 0.01 px for the fp32 path.
- `mlxstab/stabilize.py`, `python -m mlxstab`: no-crop stabilizer (similarity global fit by RANSAC with forward-backward masking and mover exclusion, residual mesh + rolling-shutter row bands, L1-optimal paths with border budget, roll levelling, one Lanczos resample, border mask video). Interlaced sources deinterlaced first (bwdif send_field).
- `.claude/verify.sh`: 15 tests, offline, no weights needed (weight tests skip), green.

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
.claude/verify.sh                                              # green check
.venv/bin/python -I bench/profile.py raft 1088 1920 12         # per-stage RAFT profile
.venv/bin/python -I bench/profile.py raftfast 1088 1920 12 float16 float16
.venv/bin/python -I bench/warm_iters.py ref && .venv/bin/python -I bench/warm_iters.py mlx
.venv/bin/python -I bench/bench_raft.py {mlx|mps|cpu} 544 960 12 3 {pair|bidir|bidir16}
python -m mlxstab in.mp4 out.mp4 --mask mask.mp4 --iters 20 --flow-downscale 2     # exact path
```
Config fields for the fast path: `flow_dtype="float16"`, `warm_iters=8`, `analysis_stride=2`. To finish the gate, run clips A and B (10 s windows) with `base`, `fast`, `fast2` and compare jitter, wobble and border with the table above, taking the film's heavy lock per chunk (about 30 frames, 12 s gap). Setup, weights and conversion are in the README. Outputs of earlier runs are in the film repo under renders/v5_proto/mlxstab_*.
