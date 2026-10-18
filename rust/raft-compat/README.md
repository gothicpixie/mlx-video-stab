# RAFT Rust/MLX compatibility spike

This crate starts E20 from `PLAN.md`. It pins the community `mlx-rs` binding to
0.32.0 and exercises parts of the Python RAFT contract: correlation lookup,
the all-pairs correlation pyramid, loading the checked-in converted RAFT
safetensors file, the checkpoint-backed `cnet.conv1` layer, the complete
`fnet` and `cnet` `BasicEncoder` stacks, the checkpoint-backed RAFT
`BasicUpdateBlock`, and a Python-shaped bidirectional flow loop.

`CorrBlock` builds level-zero correlation from paired NHWC features as
`(f1 @ f2^T) / sqrt(C)`. Each coarser level uses floor-sized 2x2 averages,
dropping any odd trailing row or column exactly like Python's `_pool2`.
`CorrBlock::lookup` flattens an `[1,H,W,2]` coordinate grid, samples every
level through the tested `lookup_level` XY-order/zero-padding contract, and
returns `[1,H,W,levels*(2*radius+1)^2]` float32 values.

`BasicUpdateBlock` loads the production update-block tensors by exact names and
shapes, then runs the motion encoder, horizontal and vertical GRU passes, flow
delta head, and quarter-scaled 576-channel convex-upsample mask head. Its
`core` and complete `forward` interfaces are checked independently against the
Python MLX block using checkpoint-backed fixture data. `net`, `inp`, `corr` and
`flow` accept a positive batch dimension so the forward and backward directions
can share one batch-2 update, matching Python's `_iterate`.

`Raft` combines the encoders and update block behind a Python-shaped API:

- `Raft::frame_features` normalizes a `[1,H,W,3]` 0..255 image, runs `fnet` and
  `cnet` once, and splits the context tensor into `tanh(net)` and `relu(inp)`.
- `Raft::flow_bidir` / `flow_bidir_features` build separate forward and backward
  correlation pyramids, initialize one shared XY coordinate grid, then for each
  of `iters` iterations look up both directions, run one batch-2
  `update_block.core`, and add the float32 delta to the coordinates. The final
  mask head runs once and both directions are convex-upsampled by 8x.

Inputs must be positive multiples of 8 in height and width. The 64x64 fixture
uses an 8x8 feature grid, so all four correlation levels remain non-empty.

`RaftEncoders` loads both encoders from the converted checkpoint by their exact
safetensors names. It validates every expected shape and rejects unconsumed keys
under the encoder prefix. `fnet` uses parameter-free instance norm and `cnet`
uses the checkpoint's evaluation-mode batch-norm running statistics. Both
include the six residual blocks, all three stride-2 downsampling stages (the
first 7x7 convolution plus two downsample projections), and the final 1x1
projection. Inputs are normalized NHWC floats with height and width
already padded to multiples of 8, matching what `RAFT.encode` passes to the
Python encoders.

Five goldens cover deterministic inputs:

- `tests/fixtures/cnet_conv1_reference.json` for the first context convolution.
- `tests/fixtures/basic_encoder_reference.json` for all values on a normalized
  32x40 frame pair. `fnet` runs with the production pair batch (`[2,4,5,256]`);
  `cnet` runs on the first frame (`[1,4,5,256]`). It records the Python MLX
  version, Rust `mlx-rs`/MLX versions, and checkpoint SHA-256.
- `tests/fixtures/corr_block_reference.json` for every correlation-pyramid value
  and every lookup value on deterministic 7x9x5 feature maps, in both float32
  and float16 correlation storage. It is checkpoint-independent, uses three
  levels with radius two, and records the Python MLX/Rust version metadata
  without a checkpoint hash.
- `tests/fixtures/update_block_reference.json` for the updated hidden state,
  flow delta, and mask on deterministic 3x4 inputs. It records all inputs and
  outputs plus the checkpoint SHA-256 and Python/Rust MLX versions.
- `tests/fixtures/flow_bidir_reference.json` for the complete bidirectional flow
  path on two deterministic 64x64 images at **1 and 24** refinement iterations.
  It records both images once, then a `cases` array whose entries carry the
  iteration count and the forward/backward low-resolution flows and both
  full-resolution flows after mask-head convex upsampling. It also records the
  checkpoint SHA-256, Python MLX version, Rust `mlx-rs`/MLX versions, the
  verified iteration counts, and the production correlation configuration.
  The 1-iteration case is preserved unchanged as a regression, and two
  consecutive regenerations produced the same fixture SHA-256
  `b48a360ba0cbcd0805beda5eed4ae439e969a15fcb436887162a4dd6feb7fa17`.

Regenerate them with:

```sh
.venv/bin/python rust/raft-compat/tools/generate_cnet_conv1_reference.py
.venv/bin/python rust/raft-compat/tools/generate_basic_encoder_reference.py
.venv/bin/python rust/raft-compat/tools/generate_corr_block_reference.py
.venv/bin/python rust/raft-compat/tools/generate_update_block_reference.py
.venv/bin/python rust/raft-compat/tools/generate_flow_bidir_reference.py
```

Correlation lookup runs through a cached MLX Metal kernel, with a narrow
`mlx-sys` FFI wrapper; correlation and coordinates stay on-device. `CorrBlock`
construction and pooling use ordinary MLX operations. The bidirectional loop
and update block are still uncompiled, so this remains a parity spike rather
than a production inference path:

- The fixture covers 1 and 24 shared iterations in fp32; 24 matches the
  production Python stabilization path. The loop accepts any iteration count,
  but other counts have no checked-in golden. Zero iterations is supported by
  construction but not separately tested.
- Initial flow (`init_f`/`init_b` in Python) is not exposed; both directions
  start from zero flow.
- fp16 encoder/correlation variants and the Python compile path are not ported.
- Synchronization, cancellation, error recovery, and sustained-memory behavior
  still need measurement. Exact MLX version pairing remains open:
  `mlx-rs` 0.32.0 targets MLX 0.32.2 while this checkout's Python environment
  has MLX 0.32.3.

E20 still requires an exact version pairing, strict validation of the complete
RAFT parameter schema, comparison at iteration counts beyond 1 and 24,
cancellation and error-recovery tests, and repeated end-to-end runtime and
memory measurements before any migration decision.

## Matched runtime harness

`examples/bench_flow_bidir.rs` measures the current Rust bidirectional RAFT
path against the existing Python MLX and PyTorch MPS runs. It reads the two raw
float32 images produced by `bench/make_raft_bench_images.py`, loads the same
converted checkpoint outside the timing region, performs one warmup pass, then
runs repeated 24-iteration `Raft::flow_bidir` calls. Each timed call evaluates
all four returned arrays inside the clock. The Rust output is one compact JSON
line containing run metadata and per-repeat wall times; it does not infer speed
or memory.

The shared pair is exactly the one `bench/bench_raft.py` constructs with
`np.random.default_rng(0)` and `np.roll(a, (3, -5), (0, 1))`. At 256x320 the
feature grid is 32x40 and the production correlation configuration
(4 levels, radius 4, fp32) is exercised. Run from the repository root,
sequentially, one backend per process:

```sh
# 1. Write byte-identical inputs for all three backends.
.venv/bin/python -I bench/make_raft_bench_images.py \
    --height 256 --width 320 --out-dir out/raft-bench/pair-256x320

# 2. Build the Rust release example once.
cargo build --release --manifest-path rust/raft-compat/Cargo.toml \
    --example bench_flow_bidir

# 3. Rust/MLX. /usr/bin/time -l reports process lifetime peak RSS on stderr.
/usr/bin/time -l rust/raft-compat/target/release/examples/bench_flow_bidir \
    --first out/raft-bench/pair-256x320/first.f32 \
    --second out/raft-bench/pair-256x320/second.f32 \
    --checkpoint vendor/weights/conv/raft-things.safetensors \
    --height 256 --width 320 --iters 24 --reps 5 --warmup 1 \
    > out/raft-bench/rust.json 2> out/raft-bench/rust.time.txt

# 4. Python MLX, same H/W/iters/reps. bench_raft.py performs one warmup run
#    before its measured repeats, matching --warmup 1 above.
/usr/bin/time -l .venv/bin/python -I bench/bench_raft.py mlx 256 320 24 5 bidir \
    > out/raft-bench/mlx.json 2> out/raft-bench/mlx.time.txt

# 5. PyTorch MPS, same H/W/iters/reps.
/usr/bin/time -l .venv/bin/python -I bench/bench_raft.py mps 256 320 24 5 bidir \
    > out/raft-bench/mps.json 2> out/raft-bench/mps.time.txt
```

`bench_raft.py` creates and loads its model before the warmup and repeats, so
model loading is excluded there as well. The Rust `timing_scope` field records
that file I/O, checkpoint loading, model construction and warmup are outside
the measured calls. Peak RSS from `/usr/bin/time -l` is the lifetime peak for
each whole process (interpreter, model load and run); compare it only with the
same metric from the other commands. Extract it after the runs with:

```sh
grep 'maximum resident set size' out/raft-bench/*.time.txt
```

The Rust path here is the current compatibility prototype (device-side
correlation lookup, uncompiled update block), not a production runtime. The
PyTorch reference computes the upsampling mask on every iteration while the
MLX paths compute it once; the harness therefore measures each implementation
as it currently exists, not an identical graph. Keep the generated inputs and
outputs under the ignored `out/` directory.

The pre-kernel baseline at 360x640 (feature grid 45x80), fp32 and 24 iterations
used one warmup and seven measured calls:

| backend | median per pair | process peak RSS |
|---|---:|---:|
| Rust release / MLX 0.32.2 | 0.707 s | 235 MB |
| Python MLX 0.32.3 | 0.431 s | 177 MB |
| PyTorch MPS | 0.531 s | 491 MB |

Rust measured 1.64x slower than Python MLX and 1.33x slower than PyTorch MPS.
Its process peak RSS was 58.9 MB above MLX and 255 MB below PyTorch. These are
short-run results on one host, not a broad performance claim. RSS does not
capture accelerator allocations. This baseline exposed host-side correlation
materialization as the first optimization target; the follow-up result below
measures the device-kernel change.

After moving correlation lookup to the cached MLX Metal kernel, all three
backends were rerun on the same 360x640 pair, with one warmup and seven measured
calls per backend:

| backend | median per pair | process peak RSS |
|---|---:|---:|
| Rust release / MLX 0.32.2 | 0.490 s | 91 MB |
| Python MLX 0.32.3 | 0.453 s | 176 MB |
| PyTorch MPS | 0.574 s | 422 MB |

Rust is 8% slower than Python MLX and 15% faster than PyTorch MPS in this run.
Compared with its pre-kernel 0.707 s baseline, Rust is 31% faster and uses
144 MB less process peak RSS. These short measurements support the targeted
optimization, but do not establish a general speed or memory win; the
uncompiled update block remains in the critical path, and process RSS does not
report accelerator allocations.

Run on Apple Silicon with Rust and Xcode installed:

```sh
cargo test --manifest-path rust/raft-compat/Cargo.toml
```

`mlx-rs` 0.32.0 currently targets MLX 0.32.2; the Python environment in this
checkout has MLX 0.32.3. Encoder, update-block, and bidirectional fixture
outputs at 1 and 24 iterations match on this setup: max absolute error is zero
for all four 1-iteration outputs, while the 24-iteration errors are
`1.09e-4` (low forward), `7.04e-4` (full forward), `8.91e-4` (low backward)
and `2.89e-3` (full backward) pixels. That is not full-model parity across
versions or iteration counts. E20 still requires an exact version pairing,
strict validation of the complete RAFT parameter schema, comparisons at
iteration counts beyond 1 and 24, cancellation/error-recovery tests, and
repeated end-to-end runtime and memory measurements before any migration
decision.
