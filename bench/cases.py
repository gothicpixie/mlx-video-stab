"""Explicit, single-variable benchmark cases for the Stage 0 measurement runner.

The historical *practical* preset (used by the saved camcorder reports) is fixed here: half-resolution
flow, 20 iterations, fidelity 0.1, output at 30000/1001 fps, and metrics evaluated at 30000/1001 fps. It is
deliberately **not** the literal current ``Config`` default (full-resolution flow, 24 iterations). Each
named case below changes one variable relative to that preset so a comparison measures one thing; the two
intentional multi-field cases are ``combined-fp16`` (two precision fields) and ``global-only`` (two residual
channels), and each must be read alongside its isolated counterpart. The residual-channel cases are practical
configuration ablations, **not** fixed-plan causal isolation: disabling a channel makes the fit re-estimate the
remaining motion fields, so the comparison measures the configuration's output rather than the isolated effect
of a frozen plan:

- ``baseline``         the historical preset unchanged (full mesh + rows)
- ``fp16-network``     baseline + fp16 network/encoders (correlation stays fp32)
- ``fp16-corr``        baseline + fp16 correlation pyramid (network stays fp32)
- ``combined-fp16``    baseline + fp16 network **and** fp16 correlation (two variables)
- ``warm-8``           baseline + 8 warm-start iterations (cold first pair still uses 20)
- ``stride-2``         baseline + analysis every second frame
- ``global-only``      baseline with both residual channels disabled (``mesh=(0,0)``, ``rs_rows=0``)
- ``mesh-only``        baseline with rolling-shutter rows disabled (``rs_rows=0``)
- ``row-only``         baseline with the residual mesh disabled (``mesh=(0,0)``)

Config identity is canonical JSON of the resolved config (sorted keys, tuples as lists) under SHA-256, so a
manifest can be invalidated when any field changes.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

# Practical historical preset defined by the task (see STATUS.md / bench/README.md).
# Only fields that differ from Config() defaults are listed; resolve_case() merges over Config.
HISTORICAL_PRESET = {
    "flow_iters": 20,
    "flow_downscale": 2,
    "fidelity": 0.1,
    "analysis_stride": 1,
    "flow_dtype": "float32",
    "corr_dtype": "float32",
    "warm_start": True,
    "warm_iters": 0,
}

# The historical metrics and output were analysed at 30000/1001 fps.
HISTORICAL_EVAL_VF = "fps=30000/1001"
HISTORICAL_OUTPUT_FPS = "30000/1001"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    config: dict
    eval_vf: str | None
    output_fps: str | None
    description: str
    changed: str  # the single variable changed relative to baseline ("" for baseline)

    def resolved_config(self):
        """Full Config kwargs: Config() defaults overlaid with the case dict (imports mlx lazily)."""
        from mlxstab.stabilize import Config
        from dataclasses import asdict
        base = asdict(Config())
        base.update(self.config)
        return base


def _spec(name, config, description, changed, eval_vf=HISTORICAL_EVAL_VF, output_fps=HISTORICAL_OUTPUT_FPS):
    return CaseSpec(name=name, config=dict(config), eval_vf=eval_vf, output_fps=output_fps,
                    description=description, changed=changed)


_CASES = {
    "baseline": _spec(
        "baseline", HISTORICAL_PRESET,
        "Historical practical preset: half-res flow, 20 iters, fidelity 0.1, metrics at 30000/1001 fps.",
        ""),
    "fp16-network": _spec(
        "fp16-network", {**HISTORICAL_PRESET, "flow_dtype": "float16"},
        "Baseline with fp16 RAFT encoders/update block; correlation volume stays fp32.",
        "flow_dtype"),
    "fp16-corr": _spec(
        "fp16-corr", {**HISTORICAL_PRESET, "corr_dtype": "float16"},
        "Baseline with an fp16 correlation pyramid; network stays fp32.",
        "corr_dtype"),
    "combined-fp16": _spec(
        "combined-fp16", {**HISTORICAL_PRESET, "flow_dtype": "float16", "corr_dtype": "float16"},
        "Both fp16 network and fp16 correlation pyramid (two variables changed).",
        "flow_dtype+corr_dtype"),
    "warm-8": _spec(
        "warm-8", {**HISTORICAL_PRESET, "warm_iters": 8},
        "Baseline with 8 iterations once warm-started (cold first pair still uses 20).",
        "warm_iters"),
    "stride-2": _spec(
        "stride-2", {**HISTORICAL_PRESET, "analysis_stride": 2},
        "Baseline analysing every second frame and interpolating the path.",
        "analysis_stride"),
    "global-only": _spec(
        "global-only", {**HISTORICAL_PRESET, "mesh": (0, 0), "rs_rows": 0},
        "Historical preset with both residual channels disabled (mesh=(0,0), rs_rows=0); the global model is "
        "refit.",
        "mesh+rs_rows"),
    "mesh-only": _spec(
        "mesh-only", {**HISTORICAL_PRESET, "rs_rows": 0},
        "Historical preset with rolling-shutter rows disabled (rs_rows=0); global + mesh are refit.",
        "rs_rows"),
    "row-only": _spec(
        "row-only", {**HISTORICAL_PRESET, "mesh": (0, 0)},
        "Historical preset with the residual mesh disabled (mesh=(0,0)); global + rows are refit.",
        "mesh"),
}

case_names = tuple(_CASES)


def get_case(name: str) -> CaseSpec:
    try:
        return _CASES[name]
    except KeyError:
        raise KeyError(f"unknown case {name!r}; known: {', '.join(case_names)}") from None


def resolve_case(name: str, overrides: dict | None = None) -> CaseSpec:
    """Return a CaseSpec, optionally with validated Config field overrides applied on top.

    The ``changed`` label is derived by comparing the fully resolved config (Config defaults overlaid with
    overrides) against the resolved historical preset, so a no-op override such as ``seed=0`` (already the
    Config default) is not reported as a change.
    """
    case = get_case(name)
    cfg = dict(case.config)
    overrides = overrides or {}
    if overrides:
        allowed = _config_fields()
        bad = sorted(set(overrides) - allowed)
        if bad:
            raise KeyError(f"unknown Config field(s) {bad}; allowed: {sorted(allowed)}")
        cfg.update(overrides)
    if cfg == case.config and not overrides:
        return case
    baseline = _resolved_baseline()
    resolved = dict(baseline)
    resolved.update(cfg)
    diff = sorted(k for k in resolved if resolved[k] != baseline[k])
    changed = ",".join(diff) if diff else case.changed
    return CaseSpec(name=case.name, config=cfg, eval_vf=case.eval_vf, output_fps=case.output_fps,
                    description=case.description, changed=changed)


def _resolved_baseline() -> dict:
    from mlxstab.stabilize import Config
    from dataclasses import asdict
    base = asdict(Config())
    base.update(HISTORICAL_PRESET)
    return base


def _config_fields() -> set[str]:
    from mlxstab.stabilize import Config
    from dataclasses import fields
    return {f.name for f in fields(Config)}


# --------------------------------------------------------------------------------------
# Config identity (canonical JSON + SHA-256)
# --------------------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)                      # canonical JSON is finite-only; config has no NaN
    return str(value)


def canonical_config(config: dict) -> str:
    """Stable canonical JSON for a Config kwargs dict (sorted keys, no whitespace, finite floats only)."""
    return json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_digest(config: dict) -> str:
    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()


def historical_preset() -> dict:
    return deepcopy(HISTORICAL_PRESET)
