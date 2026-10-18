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
- `tests/fixtures/flow_bidir_reference.json` for the complete one-iteration
  bidirectional flow path on two deterministic 64x64 images. It records both
  images, the forward/backward low-resolution flows, and both full-resolution
  flows after mask-head convex upsampling, plus the checkpoint SHA-256, Python
  MLX version, Rust `mlx-rs`/MLX versions, and the production correlation
  configuration.

Regenerate them with:

```sh
.venv/bin/python rust/raft-compat/tools/generate_cnet_conv1_reference.py
.venv/bin/python rust/raft-compat/tools/generate_basic_encoder_reference.py
.venv/bin/python rust/raft-compat/tools/generate_corr_block_reference.py
.venv/bin/python rust/raft-compat/tools/generate_update_block_reference.py
.venv/bin/python rust/raft-compat/tools/generate_flow_bidir_reference.py
```

The lookup is deliberately a host-side correctness reference: it materializes
MLX inputs into contiguous f32 slices and builds a new MLX output array.
`CorrBlock` construction and pooling use ordinary MLX operations on-device.
The bidirectional loop is also uncompiled, so it does not demonstrate Rust
runtime, memory, or copy behavior. The crate is a parity spike, not a
production inference path:

- The fixture covers one shared iteration in fp32. The loop accepts any
  iteration count, but additional iteration counts have no checked-in golden
  yet. Zero iterations is supported by construction but not separately tested.
- Initial flow (`init_f`/`init_b` in Python) is not exposed; both directions
  start from zero flow.
- fp16 encoder/correlation variants and the Python compile path are not ported.
- Lookup, synchronization, cancellation, error recovery, and sustained-memory
  behavior still require measurements. Exact MLX version pairing remains open:
  `mlx-rs` 0.32.0 targets MLX 0.32.2 while this checkout's Python environment
  has MLX 0.32.3.

E20 still requires an exact version pairing, strict validation of the complete
RAFT parameter schema, comparison at more iteration counts, cancellation and
error-recovery tests, and repeated end-to-end runtime and memory measurements
before any migration decision.

Run on Apple Silicon with Rust and Xcode installed:

```sh
cargo test --manifest-path rust/raft-compat/Cargo.toml
```

`mlx-rs` 0.32.0 currently targets MLX 0.32.2; the Python environment in this
checkout has MLX 0.32.3. Encoder, update-block, and one-iteration bidirectional
fixture outputs match exactly on this setup, but that is not full-model parity
across versions or iteration counts. E20 still requires an exact version
pairing, strict validation of the complete RAFT parameter schema, comparisons
at additional iteration counts, cancellation/error-recovery tests, and repeated
end-to-end runtime and memory measurements before any migration decision.
