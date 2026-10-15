# Stage 0 benchmark runner

`bench/run_case.py` runs one **explicit** stabilization case over a synthetic clip or a local video window
and writes one atomic, JSON-safe manifest with enough provenance to invalidate a comparison. It is Stage 0
measurement infrastructure, not a quality claim: it does not compute a single score, does not change
stabilization defaults, and records failures and missing support explicitly.

## Cases

Each case changes one variable relative to the historical practical preset
(`flow_downscale=2`, `flow_iters=20`, `fidelity=0.1`, `analysis_stride=1`, fp32, warm start on, output and
metrics at `30000/1001`; **not** the literal current `Config` defaults of full-res flow / 24 iterations).
`combined-fp16` deliberately changes two variables and must be read alongside the two isolated fp16 cases.

| case | changed variable | meaning |
| --- | --- | --- |
| `baseline` | none | historical practical preset |
| `fp16-network` | `flow_dtype=float16` | fp16 encoders/update block, fp32 correlation |
| `fp16-corr` | `corr_dtype=float16` | fp32 network, fp16 correlation pyramid |
| `combined-fp16` | `flow_dtype` + `corr_dtype` | fp16 network and fp16 correlation pyramid (two variables) |
| `warm-8` | `warm_iters=8` | 8 iterations once warm-started (cold pair still 20) |
| `stride-2` | `analysis_stride=2` | analyse every second frame, interpolate the path |

`--set KEY=VALUE` overrides a single `Config` field; `--config-json` accepts a JSON object. Unknown fields
are rejected. `--seed` sets both synthetic clip generation and the per-clip fitting RNG (`--set seed=`
overrides the fitting seed). Config identity is canonical JSON (sorted keys, tuples as lists) under SHA-256,
so any field change invalidates a manifest.

## Usage

Public synthetic smoke (no private footage, heavy lock off, small):

```bash
python -I bench/run_case.py --case baseline --source synthetic --duration 1 --synth-frames 30
```

Local file source (footage stays local; keep outputs in this repo's ignored `out/`):

```bash
python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS \
    --start 60 --duration 1 --diag-every 15 \
    --out-dir out/bench/smoke_t060 --log out/bench/smoke_t060.log
```

Full 10 s window, shared heavy lock held for the run:

```bash
python -I bench/run_case.py --case baseline --source /path/to/local/clip.MTS \
    --start 60 --duration 10 --lock on --out-dir out/bench/baseline_t060 --log out/bench/baseline_t060.log
```

List cases: `python -I bench/run_case.py --list-cases`. Inspect a case without running:
`python -I bench/run_case.py --dry-run --case warm-8`.

Useful options:

- `--start`, `--duration` – precise source window.
- `--eval-iters`, `--eval-downscale`, `--eval-flow-dtype`, `--eval-corr-dtype` – evaluation flow.
- `--output-fps VALUE` / `--eval-vf VALUE` – override the case output cadence (rational, or `native`) and the
  evaluation `fps` filter. The resolved values are recorded in `case` and folded into `experiment_identity`.
- `--overwrite` – reuse an existing run directory. By default an existing `manifest.json` is protected and the
  CLI returns a structured nonzero failure instead of clobbering evidence. Default run directories carry a
  microsecond stamp and are unique.
- `--selection-note TEXT` – provenance note (e.g. the 60/800 s windows were inferred from historical output
  filenames, not a recovered original manifest). Started in `source.selection_note` and the identity.
- `--repeats N` – one cold run plus `N-1` warm runs in the same process; each repeat writes distinct
  `stabilized_r{i}.mp4` / `mask_r{i}.mp4` / `report_r{i}.json` and records its model/temporal state and wall
  time. `rendered_frames` (writes) and the separately measured `encoded_frames` are kept distinct. The
  retained output for evaluation is repeat 0 (`outputs.retained_repeat`).
- `--diag-every K` – sample complete-warp diagnostics every `K` rendered frames. Mutually exclusive with
  `Config.warp_diag` (all frames): the runner rejects the combination during validation.
- `--lock {auto,on,off}`, `--lock-chunk N`, `--lock-pause S` – shared heavy lock. `auto` uses the lock for
  file sources and not for synthetic. A `chunk` releases the lock, yields for `--lock-pause` seconds (the
  shared-project etiquette is ~30 frames with a ~12 s gap) and re-acquires; acquisition wait and pause are
  accounted separately from processing. The default holds the lock for the whole run and always releases in
  a `finally`; cancellation writes a `cancelled` manifest before re-raising.

## Cadence

The historical preset renders at the source processing rate and encodes the output at `30000/1001` using an
explicit FFmpeg `fps` filter (`-vf fps=... -r ... -fps_mode cfr`). Relying on output `-r` alone was wrong: it
produced 32 frames for a 1 s 60 fps clip at 30 fps. Conversion is validated by
`tests/test_media_cadence.py` for 60→30 and 60000/1001→30000/1001, including mask alignment. `stabilize`
reports `rendered_frames` (writes) and the runner measures the true `outputs.encoded_frames` by probing the
finished file; the two are never conflated.

Metrics use the same `fps=RATE` filter on input, output and mask. Their complete decoded frame counts and
normalized frame-index timelines must agree. Separately, every encoded video and mask PTS is checked against
the requested zero-origin CFR clock, with a 10-microsecond tolerance for ffprobe's decimal rounding. This
detects shifted clocks, missing timestamps, duplicate/interior timing errors and extra mask tails. The
encoded rate may differ from evaluation: 60 fps video/mask can correctly be evaluated at 30 fps. Output
and evaluation rates must be positive; arbitrary filter chains are not supported by `--eval-vf`.

The source decoder does not expose original frame PTS. The benchmark records that limitation and uses
normalized frame-index time for evaluation; it does not validate original VFR timing or absolute source
timestamps. Raw container `start_time`/`time_base` are retained as provenance.

The `Config` field `analysis_stride` controls analysis cadence separately from output cadence.

## Diagnostics

Complete-warp diagnostics are opt-in and expensive. `--diag-every K` samples the **actual full-window plan of
the retained repeat** during rendering (no extra RAFT inference) at exactly the render indices where
`t % K == 0`. Each sampled frame is rendered once without diagnostics and the `warpmetrics.diagnose` call is
timed separately; the manifest stores the exact frame indices, per-frame diagnostic dictionaries, aggregates
and the separated overhead. `Config.warp_diag=True` (all-frame diagnostics) is mutually exclusive with
`--diag-every` and the runner rejects the combination. `render` minus the diagnostic total is labelled
**observed non-diagnostic interval time**, not counterfactual throughput, because encoding overlaps with
rendering. Per-sample maxima are labelled as such and are not whole-video guarantees.

## Manifest

Each run writes `<out-dir>/manifest.json` (atomic temp-write + `os.replace`). It contains:

- `schema`, `created`, `status` (`success`/`failure`), `error` (`type`, `message`, `stage`).
- `case` (name, changed variable, output/eval cadence), full resolved `config`, `config_digest`, the expanded
  `experiment_identity` (source + window + preprocessing, weights, config, output/eval cadence, evaluation
  parameters, seed policy, diagnostic selection, working-tree code digests) and its `content_key`.
- `source` (sha256, ffprobe metadata, window, preprocessing, decoded frame count/fps/duration) and `weights` sha256.
- `code` (git revision + sha256 of `mlxstab/**`, `bench/**`, `pyproject.toml`, plus a relevant-dirty flag that
  ignores unrelated docs), `evaluation` (eval flow precision/iters/downscale, per-clip reset, metric config)
  and `metrics.definitions_version` bound to the metric source hashes, and `seed_policy`.
- `environment` (Python/MLX/Torch/NumPy/SciPy/FFmpeg/ffprobe) and `hardware`.
- `timing` (provenance, model construction, repeat-0 wall, end-to-end stabilization including model load and
  encoder close, end-to-end with evaluation, and a `lock_scope` note that acquisition waits/pauses are inside
  `total_wall_s` and not subtracted from stabilization) and `runtime_s` (analysis/plan/render stages,
  evaluation, diagnostics total, observed non-diagnostic render time, total wall) plus per-repeat
  `model_state`/`temporal_state`/`rendered_frames`/`encoded_frames`/wall time and distinct output paths.
- `outputs` (`encoded_frames`, `encoded_pts`, `mask_pts`, cadence) and `source.selection_note`.
- `memory` (process peak RSS and MLX peak, each with scope, else `null` with a reason), `metrics` (input/
  output, cadence with validated timelines, reliability + support stats and reasons), `diagnostics`, `lock`
  (backend and acquire/pause accounting), and `unavailable` reasons for anything not measured.

Unavailable measurements are written as `null` with an explanation, never as zero. Serialization uses
`allow_nan=False`; non-finite values are converted to `null` and never emitted as `NaN`/`Infinity`.

## Reliability

`reliability()` marks a comparison unusable (never "stable") when cadence is misaligned, when required
metrics are missing or non-finite, when explicit support counters (`support_frames`, `inlier_frac_mean`,
`fallback_frames`) are absent or show no adequate support, or when any motion fit fell back to an identity
transform. Failed fits can otherwise score as perfectly stable footage. The manifest keeps the input and
output metric dicts and support statistics separate; there is no aggregate score.

## Locking and private data

For file sources the runner reuses the anniversary project's shared lock (`media/.heavy.lock`) by importing
its `tools/heavy.py`; if it is unavailable a local lock under this repo's ignored `out/` is used and recorded
as `lock.backend`. Lock paths and private source paths appear only in ignored `out/` manifests, never in
tracked docs. No credentials or arbitrary environment variables are recorded; only the optional
`MLXSTAB_FILM_TOOLS` path override is read.

## Tests

`tests/test_bench_runner.py` covers configuration/experiment identity, change labelling against the fully
resolved config, state isolation, manifest hashing/invalidation, atomic and JSON-safe writes, failure and
cancellation reporting, overwrite protection, run-dir uniqueness, cadence/timeline and mask-tail mismatch,
support flags, rendered-vs-encoded counts, diagnostics sampling policy and the CLI failure paths, all with
cheap model/media fakes. `tests/test_media_cadence.py` adds real (CPU-only) ffmpeg regressions and a fake-flow
end-to-end 60-render/30-encode case that needs no RAFT.
