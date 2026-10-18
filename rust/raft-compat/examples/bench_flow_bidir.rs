//! Matched runtime harness for the Rust/MLX bidirectional RAFT path.
//!
//! This example reads the raw float32 image pair produced by
//! `bench/make_raft_bench_images.py`, loads the converted RAFT safetensors
//! checkpoint outside the timing region, warms up, and then times repeated
//! 24-iteration `Raft::flow_bidir` calls. Every timed call evaluates all four
//! returned arrays before stopping the clock, so lazy MLX work is included.
//!
//! Run from the repository root:
//!
//! ```sh
//! cargo build --release --manifest-path rust/raft-compat/Cargo.toml \
//!     --example bench_flow_bidir
//! /usr/bin/time -l rust/raft-compat/target/release/examples/bench_flow_bidir \
//!     --first out/raft-bench/pair-256x320/first.f32 \
//!     --second out/raft-bench/pair-256x320/second.f32 \
//!     --checkpoint vendor/weights/conv/raft-things.safetensors \
//!     --height 256 --width 320 --iters 24 --reps 5 --warmup 1 \
//!     --compiled-update 1
//! ```
//!
//! `--compiled-update` selects the update-block core mode: `1` (the default) uses the cached
//! compiled core, and `0` uses the direct `BasicUpdateBlock::core` reference. The selected mode is
//! recorded in the JSON output.
//!
//! The program prints one compact JSON line. Peak process RSS is deliberately
//! not inferred here; wrap the executable with `/usr/bin/time -l` as shown.
//! This is a measurement harness, not a production path.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::Array;
use mlxstab_raft_compat::Raft;
use sha2::{Digest, Sha256};

const DEFAULT_CHECKPOINT: &str = "vendor/weights/conv/raft-things.safetensors";

#[derive(Debug)]
struct Args {
    first: PathBuf,
    second: PathBuf,
    checkpoint: PathBuf,
    height: usize,
    width: usize,
    iters: usize,
    reps: usize,
    warmup: usize,
    compiled_update: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            first: PathBuf::new(),
            second: PathBuf::new(),
            checkpoint: PathBuf::from(DEFAULT_CHECKPOINT),
            height: 256,
            width: 320,
            iters: 24,
            reps: 5,
            warmup: 1,
            compiled_update: true,
        };
        let mut argv = std::env::args().skip(1);
        while let Some(arg) = argv.next() {
            let mut value = || argv.next().ok_or_else(|| format!("{arg} requires a value"));
            match arg.as_str() {
                "--first" => args.first = PathBuf::from(value()?),
                "--second" => args.second = PathBuf::from(value()?),
                "--checkpoint" => args.checkpoint = PathBuf::from(value()?),
                "--height" => args.height = parse_usize("--height", &value()?)?,
                "--width" => args.width = parse_usize("--width", &value()?)?,
                "--iters" => args.iters = parse_usize("--iters", &value()?)?,
                "--reps" => args.reps = parse_usize("--reps", &value()?)?,
                "--warmup" => args.warmup = parse_usize("--warmup", &value()?)?,
                "--compiled-update" => {
                    args.compiled_update = parse_bool01("--compiled-update", &value()?)?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: bench_flow_bidir --first PATH --second PATH \
                         [--checkpoint PATH] [--height N] [--width N] \
                         [--iters N] [--reps N] [--warmup N] \
                         [--compiled-update 0|1]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        if args.first.as_os_str().is_empty() || args.second.as_os_str().is_empty() {
            return Err("--first and --second are required".into());
        }
        if args.height == 0 || args.width == 0 {
            return Err("height and width must be positive".into());
        }
        if !args.height.is_multiple_of(8) || !args.width.is_multiple_of(8) {
            return Err("height and width must be multiples of 8".into());
        }
        if args.reps == 0 {
            return Err("reps must be at least 1".into());
        }
        Ok(args)
    }
}

fn parse_usize(flag: &str, text: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|e| format!("{flag}: invalid integer {text:?}: {e}"))
}

fn parse_bool01(flag: &str, text: &str) -> Result<bool, String> {
    match text {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{flag}: expected 0 or 1, got {text:?}")),
    }
}

fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok((format!("{:x}", Sha256::digest(&bytes)), bytes.len() as u64))
}

fn read_raw_image(path: &Path, height: usize, width: usize) -> Result<Array, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let expected_bytes = height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "image dimensions overflow usize".to_string())?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{} has {} bytes, expected {} for {height}x{width}x3 float32",
            path.display(),
            bytes.len(),
            expected_bytes
        ));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let height = i32::try_from(height).map_err(|_| "height exceeds i32".to_string())?;
    let width = i32::try_from(width).map_err(|_| "width exceeds i32".to_string())?;
    Ok(Array::from_slice(&values, &[1, height, width, 3]))
}

/// Run one bidirectional pair and materialize all four outputs inside the clock.
fn timed_once(
    model: &mut Raft,
    first: &Array,
    second: &Array,
    iters: usize,
) -> Result<f64, String> {
    let started = Instant::now();
    let flow = model
        .flow_bidir(first, second, iters)
        .map_err(|e| e.to_string())?;
    mlx_rs::transforms::eval([
        &flow.low_forward,
        &flow.up_forward,
        &flow.low_backward,
        &flow.up_backward,
    ])
    .map_err(|e| e.to_string())?;
    Ok(started.elapsed().as_secs_f64())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn summary(values: &[f64]) -> serde_json::Value {
    let sorted = {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted
    };
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    serde_json::json!({
        "min_s": sorted[0],
        "median_s": median(values),
        "mean_s": mean,
        "max_s": sorted[sorted.len() - 1],
    })
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let (first_hash, first_bytes) = sha256_file(&args.first)?;
    let (second_hash, second_bytes) = sha256_file(&args.second)?;
    let (checkpoint_hash, checkpoint_bytes) = sha256_file(&args.checkpoint)?;

    let first = read_raw_image(&args.first, args.height, args.width)?;
    let second = read_raw_image(&args.second, args.height, args.width)?;

    // Model construction and checkpoint loading are outside both warmup and
    // measured timing, matching bench_raft.py's placement.
    let mut model = Raft::load(&args.checkpoint)?;
    model.set_compiled_update(args.compiled_update);

    let mut warmup_s = Vec::with_capacity(args.warmup);
    for _ in 0..args.warmup {
        warmup_s.push(timed_once(&mut model, &first, &second, args.iters)?);
    }

    let mut run_s = Vec::with_capacity(args.reps);
    for _ in 0..args.reps {
        run_s.push(timed_once(&mut model, &first, &second, args.iters)?);
    }

    let payload = serde_json::json!({
        "schema": "mlxstab.raft_compat.bench_flow_bidir/1",
        "backend": "rust-mlx",
        "mode": "bidir",
        "height": args.height,
        "width": args.width,
        "feature_grid": [args.height / 8, args.width / 8],
        "iters": args.iters,
        "reps": args.reps,
        "warmup": args.warmup,
        "precision": {
            "network": "float32",
            "correlation": "float32",
            "coordinates": "float32",
        },
        "correlation": {"levels": 4, "radius": 4},
        "images": {
            "first": {
                "path": args.first.display().to_string(),
                "bytes": first_bytes,
                "sha256": first_hash,
            },
            "second": {
                "path": args.second.display().to_string(),
                "bytes": second_bytes,
                "sha256": second_hash,
            },
        },
        "checkpoint": {
            "path": args.checkpoint.display().to_string(),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_hash,
        },
        "runtime": {
            "crate": env!("CARGO_PKG_NAME"),
            "crate_version": env!("CARGO_PKG_VERSION"),
            "mlx_rs": "0.32.0 (pinned in rust/raft-compat/Cargo.toml)",
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        },
        "compile": {
            "update_block_compiled": model.compiled_update(),
            "update_block_mode": if model.compiled_update() {
                "compiled"
            } else {
                "direct"
            },
            "correlation_lookup": "cached MLX Metal kernel",
        },
        "timing_scope": (
            "wall time around one Raft::flow_bidir call plus one eval of all four \
             outputs; checkpoint load, image I/O, model construction and warmup excluded"
        ),
        "warmup_s": warmup_s,
        "run_s": run_s,
        "summary_s": summary(&run_s),
        "peak_rss_note": (
            "not measured in-process; wrap the executable with /usr/bin/time -l \
             and record maximum resident set size"
        ),
        "argv": std::env::args().collect::<Vec<_>>(),
    });
    println!(
        "{}",
        serde_json::to_string(&payload).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bench_flow_bidir: {error}");
        std::process::exit(2);
    }
}
