# mlxstab launch copy

Drafts for sharing the repository when it is ready. These describe the current experimental implementation and make no performance claims.

## GitHub description

Experimental no-crop video stabilization for Apple Silicon. MLX + RAFT optical flow, L1 camera paths, Metal resampling, and explicit border masks.

## Suggested GitHub topics

`video-stabilization` · `apple-silicon` · `mlx` · `optical-flow` · `raft` · `metal` · `computer-vision` · `video-processing` · `python`

## Short pitch

**Steady motion. Full frame.**

mlxstab is an experimental, offline video stabilizer for Apple Silicon. It combines MLX optical flow, L1 camera paths, and Metal resampling to render at the original frame dimensions. Export a border mask to identify pixels with no source coverage; filling those gaps is future work.

## X or Bluesky launch post

mlxstab: experimental video stabilization for Apple Silicon, built with MLX + RAFT. Full-frame output and an explicit border mask—missing pixels stay marked, not filled.

https://github.com/gothicpixie/mlx-video-stab

## Square-card feature post

Keep the frame. Know the gaps.

mlxstab renders stabilized video at the original frame dimensions, without a crop-in by default. Add `--mask` to get a separate video marking pixels that have no source coverage. Those gaps remain unfilled, ready for a later compositing or inpainting step.

Experimental, offline, and built for Apple Silicon with MLX.

https://github.com/gothicpixie/mlx-video-stab

## LinkedIn launch post

Camera shake is only part of the stabilization problem. Moving a frame also exposes edges where there are no source pixels.

mlxstab is an experimental video stabilizer for Apple Silicon that makes those gaps explicit. It renders at the original frame dimensions and can export a border mask for a later compositing or inpainting step.

The pipeline uses a from-scratch MLX port of RAFT optical flow, robust camera-motion fitting, L1 camera paths, and a custom Metal sampler. The final transform combines the corrections into one resample of the original frame.

This is an offline research tool. Border filling is future work, and shape distortion and floaty motion still need investigation. The repository includes the implementation, tests, evaluation tools, and development plan.

If you work on optical flow, stabilization, or MLX, feedback on shape preservation and reproducible evaluation would be useful.

https://github.com/gothicpixie/mlx-video-stab

## Technical community announcement

**Title:** mlxstab: Experimental full-frame video stabilization with MLX on Apple Silicon

mlxstab is an offline video-stabilization experiment for Apple Silicon. It renders at the original frame dimensions and can write a separate mask for pixels with no source coverage. There is no crop-in by default, and missing borders remain unfilled.

The implementation includes:

- A from-scratch MLX port of RAFT, with forward/backward flow and numerical parity checks against the PyTorch reference.
- Robust global motion fitting, moving-subject exclusion, residual mesh correction, and rolling-shutter row bands.
- L1 camera-path smoothing with a configurable border budget.
- One final resample from the original image through a custom Metal kernel, using Lanczos-3 by default.

After following the repository setup and converting the separately downloaded RAFT weights, run:

```sh
python -m mlxstab input.mp4 stabilized.mp4 --mask border-mask.mp4
```

The mask is white where no source pixel is available. It provides a way to inspect missing coverage or pass it to a later fill stage. Inpainting is not implemented yet.

The project is experimental. Shape deformation and floaty motion have been reported; real-footage quality checks for the faster configurations are incomplete. The current encoder produces a video-only output, so audio needs a separate workflow. Historical measurements and open issues are documented in `STATUS.md`.

Useful feedback would include reproducible short clips, settings, and observations about straight lines, moving subjects, parallax, and intended camera movement. The repository code is MIT-licensed; RAFT weights are fetched separately.

Code and setup: https://github.com/gothicpixie/mlx-video-stab

## Feature blurbs

**Original frame dimensions.** Render to the full canvas without a default crop-in. Any exposed area with no source coverage stays visible and can be exported as a mask.

**Motion from optical flow.** Forward/backward RAFT flow feeds a robust camera fit, with independently moving subjects excluded from that fit.

**A path you can tune.** L1 smoothing and a border budget control how far the planned camera path can move from the measured path.

**One final sample.** Compose stabilization and spatial corrections into one source-coordinate field, then resample the original image once.

## Calls to action

- Explore the code and setup.
- Try a short clip and report what bends, drifts, or jitters.
- Help improve shape preservation and reproducible evaluation.
