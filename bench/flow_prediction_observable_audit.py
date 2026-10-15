"""Post-hoc audit of candidate-observable convergence signals in E04 flow-prediction manifests.

This is an **offline evaluator only**. It reads one or more successful ``mlxstab.flow_prediction_real/1``
manifests that carry an enabled budget scan, and for the candidate-observable signals already recorded there
(``flow_update_low.mean`` and ``photometric_low.resid_textured.median``) measures how each signal co-varies
with the oracle ``epe_low.mean`` *within each trajectory*, where a trajectory is one method/direction/target
pair and the variation is across the scan iterations. It does not run RAFT, decode media, render frames,
change any default, or import production stabilization code.

Association only, not accuracy and not a stopping rule
-----------------------------------------------------
The audit computes a dependency-free Spearman rank correlation with average ties between the signal series and
the oracle-EPE series of one trajectory. A high correlation means only that the candidate-observable signal
and the oracle-EPE curve move together monotonically across that trajectory's iterations; it says nothing
about absolute flow accuracy, and the oracle EPE is itself the distance to a cold high-iteration RAFT
reference (a model-derived reference, not ground truth). This module does **not** search, fit or select a
threshold, does **not** recommend a deployable stopping rule, and makes no speedup, online or
production-readiness claim. Any correlation here is a signal diagnosis, never a stop-policy validation.

Cycle-consistency signals
--------------------------
Besides the budget-scan signals, the audit also accepts a cold bidirectional ``cycle_scan`` manifest (both
candidate directions refined together from zero; same trajectory/scan shape). The cycle signal
``cycle_low.resid.mean`` is computed only from the two same-iteration candidate fields and never from the
reference; it is a forward-backward **cycle-consistency / visibility** diagnostic, not true-flow accuracy.
An explicit ``--signal cycle_low.resid.mean`` selects it, so the default report stays the two budget signals.
The ``coverage`` summary covers both kinds from one shape: budget rows report the photometric/update fractions
and host costs, cycle rows report the ``cycle_low.valid_frac`` and ``cycle_check_s`` statistics, and fields for
a kind that is absent from the audited rows stay ``null``.

Dependence caveat for the cross-sectional view
----------------------------------------------
``cross_sectional_by_iteration`` pools the same trajectories at each iteration index. Those samples are
strongly dependent: repeated directions, methods and targets reuse the same source pairs and share the same
cold reference, and adjacent iterations are serially correlated. That table is descriptive only and must not
be read as a set of independent estimates or used to pick a threshold.

Payload safety and determinism
------------------------------
Manifests are parsed as JSON only. Reports are passed through ``json_safe`` and serialized with
``allow_nan=False`` (non-finite values become ``null``). Every list is emitted in a stable order and
``--out`` writes atomically, so two runs over the same manifest bytes produce byte-identical JSON.

Run with::

    python -m bench.flow_prediction_observable_audit MANIFEST.json [MANIFEST.json ...] [--out file.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from .manifest import json_safe, write_manifest

SCHEMA = "mlxstab.flow_prediction_observable_audit/1"
MANIFEST_SCHEMA = "mlxstab.flow_prediction_real/1"

#: Candidate-observable signal name -> dotted path into a per-iteration scan row.
SIGNALS = {
    "flow_update_low.mean": ("flow_update_low", "mean"),
    "photometric_low.resid_textured.median": ("photometric_low", "resid_textured", "median"),
    # Cold-only bidirectional cycle-consistency pilot (``cycle_scan`` blocks): forward-backward cycle residual
    # from the two same-iteration candidate fields, never the reference. Audited with an explicit
    # ``--signal cycle_low.resid.mean`` so the default budget-scan report is unchanged.
    "cycle_low.resid.mean": ("cycle_low", "resid", "mean"),
}
DEFAULT_SIGNALS = ("flow_update_low.mean", "photometric_low.resid_textured.median")

#: Stable statuses for a trajectory / an individual signal-vs-EPE series. ``ok`` is the only one that
#: contributes a correlation to the aggregates; the rest are explicit skips.
STATUS_OK = "ok"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
STATUS_NO_SCAN_ROWS = "no_scan_rows"
STATUS_NULL_SIGNAL = "null_signal"
STATUS_NULL_EPE = "null_epe"
STATUS_INSUFFICIENT_POINTS = "insufficient_points"
STATUS_CONSTANT_SIGNAL = "constant_signal"
STATUS_CONSTANT_EPE = "constant_epe"

CAVEATS = [
    "Association only: the correlation between a candidate-observable signal and the oracle EPE curve is a "
    "signal diagnosis, not flow accuracy and not validation of any stopping rule.",
    "The oracle is the cold high-iteration RAFT reference, a model-derived reference rather than ground truth; "
    "EPE is the distance to that reference and cannot establish absolute flow accuracy.",
    "Per-trajectory correlations vary across a single pair's iterations and are temporally autocorrelated; "
    "cross_sectional_by_iteration additionally reuses the same pairs, methods and targets at each iteration, "
    "so its samples are dependent and it is descriptive only.",
    "No threshold is searched, fitted or selected and no deployable stop rule is recommended; nothing here "
    "claims a speedup, online use or production readiness.",
    "Trajectories whose initialization fell back with insufficient_history are excluded from the correlations "
    "and counted explicitly; scan rows whose signal or oracle EPE is null or non-finite are excluded from "
    "pairing and counted explicitly.",
    "photometric_low maps a feature cell to the top-left preprocessed pixel of its 8x8 upsampled block and "
    "does not reconstruct RAFT's learned convex-upsampling sub-cell weighting, so an alignment-dependent "
    "residual floor is possible; flat regions carry no photometric signal and texture_frac reports the "
    "observed share, so a low valid_frac/texture_frac means the signal is only weakly observed.",
    "cycle_low.resid.mean is a candidate-observable forward-backward cycle-consistency/visibility residual "
    "computed from the two same-iteration cold candidate fields (a ``cycle_scan`` manifest); it never uses "
    "the cold reference. An inconsistent cycle can also mean wrong flow, blur or independent motion, a "
    "consistent cycle does not prove visibility or correspondence correctness, and a low cycle residual is "
    "not true-flow accuracy. No threshold or stop rule is selected.",
]


class ObservableAuditError(ValueError):
    """A manifest was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


# --------------------------------------------------------------------------------------
# small numeric helpers (dependency-free)
# --------------------------------------------------------------------------------------

def _is_finite_number(value) -> bool:
    """True for a real, finite int/float (bools excluded). ``None``, NaN and +/-inf are not finite numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _dig(row, path):
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_signal(row, signal):
    """Return the finite float for ``signal`` in one scan row, or ``None`` if missing/null/non-finite."""
    value = _dig(row, SIGNALS[signal])
    return float(value) if _is_finite_number(value) else None


def average_ranks(values):
    """Ranks ``values`` (1-based) with average ranks for ties, returned in the input order.

    Ties are compared by exact equality, which is what the JSON numeric payload provides. The result is a
    list of floats; it does not assume the input is sorted and never mutates it.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation of two equal-length sequences (average ties), or ``None`` if undefined.

    Returns ``None`` for fewer than two samples or when either sequence is constant (zero rank variance), the
    two cases where a correlation is mathematically undefined rather than a property of the data. Uses only the
    standard library.
    """
    value, _reason = spearman_with_reason(x, y)
    return value


def spearman_with_reason(x, y):
    """Like :func:`spearman` but also returns a stable reason string when the value is ``None``."""
    x = list(x)
    y = list(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return None, STATUS_INSUFFICIENT_POINTS
    rx = average_ranks(x)
    ry = average_ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0.0 and syy == 0.0:
        # Both constant: report the signal axis first so the reason stays deterministic.
        return None, STATUS_CONSTANT_SIGNAL
    if sxx == 0.0:
        return None, STATUS_CONSTANT_SIGNAL
    if syy == 0.0:
        return None, STATUS_CONSTANT_EPE
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return sxy / (math.sqrt(sxx) * math.sqrt(syy)), None


def _aggregate(spearman_values):
    """Median/mean/positive-fraction over the non-null correlations (``None`` values are explicit skips)."""
    valid = [float(v) for v in spearman_values if v is not None]
    if not valid:
        return {"n": 0, "median": None, "mean": None, "positive_fraction": None}
    return {
        "n": len(valid),
        "median": float(statistics.median(valid)),
        "mean": float(statistics.mean(valid)),
        "positive_fraction": float(sum(1 for v in valid if v > 0.0) / len(valid)),
    }


# --------------------------------------------------------------------------------------
# manifest loading and validation
# --------------------------------------------------------------------------------------

def _require(cond, code, message):
    if not cond:
        raise ObservableAuditError(code, message)


def _scan_source(doc):
    """Return ``(kind, scan_block, results)``; the cycle block wins when it is enabled, else the budget block.

    A cycle-only manifest keeps its per-pair trajectories under ``cycle_scan["results"]`` (also
    ``direction``/``method``/``per_pair``/``scan`` shaped), so the same trajectory walker and signal machinery
    apply. Budget manifests keep using ``doc["results"]`` exactly as before.
    """
    cycle = doc.get("cycle_scan")
    if isinstance(cycle, dict) and cycle.get("enabled") is True:
        return "cycle_scan", cycle, cycle.get("results")
    return "budget_scan", doc.get("budget_scan") or {}, doc.get("results")


def load_manifest(path):
    """Read and structurally validate one successful, scan-enabled E04 manifest.

    Accepts a budget-scan manifest or the cold bidirectional ``cycle_scan`` pilot. Returns
    ``(raw_bytes, document)``. Refuses with :class:`ObservableAuditError` (stable ``code``) when the file
    cannot be read/parsed, is not a successful ``mlxstab.flow_prediction_real/1`` run, has no enabled scan,
    or lacks the ``results[*][*].per_pair`` trajectory structure.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ObservableAuditError("not_found", f"cannot read manifest {path}: {exc}") from exc
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservableAuditError("not_json", f"manifest {path} is not valid JSON: {exc}") from exc
    _require(isinstance(doc, dict), "not_object", f"manifest {path} is not a JSON object")
    _require(doc.get("schema") == MANIFEST_SCHEMA, "bad_schema",
             f"manifest {path} schema {doc.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    _require(doc.get("status") == "success", "not_success",
             f"manifest {path} status {doc.get('status')!r} is not 'success'")
    kind, scan, results = _scan_source(doc)
    _require(isinstance(scan, dict) and scan.get("enabled") is True, "budget_scan_disabled",
             f"manifest {path} has no enabled budget or cycle scan")
    _require(isinstance(results, dict) and results, "missing_results",
             f"manifest {path} ({kind}) has no results block")
    for direction, methods in results.items():
        _require(isinstance(methods, dict) and methods, "missing_methods",
                 f"manifest {path} results[{direction!r}] has no methods")
        for method, entry in methods.items():
            _require(isinstance(entry, dict) and isinstance(entry.get("per_pair"), list),
                     "missing_per_pair",
                     f"manifest {path} results[{direction!r}][{method!r}] has no per_pair list")
    return raw, doc


def _window_summary(source):
    window = source.get("window") if isinstance(source, dict) else None
    if isinstance(window, dict):
        return {key: window.get(key) for key in sorted(window)}
    return None


def _provenance(doc, path, raw):
    """Record what identifies the audited manifest and its source, without inventing missing fields."""
    source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    weights = doc.get("weights") if isinstance(doc.get("weights"), dict) else {}
    config = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    code = doc.get("code") if isinstance(doc.get("code"), dict) else {}
    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    cycle = config.get("cycle") if isinstance(config.get("cycle"), dict) else {}
    return {
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "schema": doc.get("schema"),
        "run_id": doc.get("run_id"),
        "created": doc.get("created"),
        "status": doc.get("status"),
        "source_sha256": source.get("sha256"),
        "frames_sha256": source.get("frames_sha256"),
        "window": _window_summary(source),
        "fps": source.get("fps"),
        "weights_sha256": weights.get("sha256"),
        "config_digest": doc.get("config_digest"),
        "config": {
            "iters": config.get("iters"),
            "ref_iters": config.get("ref_iters"),
            "downscale": config.get("downscale"),
            "methods": config.get("methods"),
            "directions": config.get("directions"),
            "budget": {
                "max_iters": budget.get("max_iters"),
                "thresholds": budget.get("thresholds"),
                "scan": budget.get("scan"),
            },
            "cycle": {
                "max_iters": cycle.get("max_iters"),
                "scan": cycle.get("scan"),
            },
        },
        "code": {"revision": code.get("revision"), "dirty_relevant": code.get("dirty_relevant")},
    }


# --------------------------------------------------------------------------------------
# per-manifest audit
# --------------------------------------------------------------------------------------

def _trajectory_record(direction, method, pair, signal):
    """Correlate one trajectory's signal series against its oracle-EPE series across the scan iterations."""
    scan = pair.get("scan")
    reason = pair.get("reason")
    record = {
        "direction": direction,
        "method": method,
        "target": pair.get("target"),
        "status": None,
        "n": 0,
        "spearman": None,
        "n_null_signal": 0,
        "n_null_epe": 0,
    }
    if reason == "insufficient_history":
        record["status"] = STATUS_INSUFFICIENT_HISTORY
        return record, []
    if not isinstance(scan, list) or not scan:
        record["status"] = STATUS_NO_SCAN_ROWS
        return record, []

    signal_series = []
    epe_series = []
    observations = []  # (iter, signal, epe) for the cross-sectional view
    for row in scan:
        value = extract_signal(row, signal)
        epe = _dig(row, ("epe_low", "mean"))
        epe = float(epe) if _is_finite_number(epe) else None
        if value is None:
            record["n_null_signal"] += 1
        if epe is None:
            record["n_null_epe"] += 1
        if value is not None and epe is not None:
            signal_series.append(value)
            epe_series.append(epe)
            observations.append((row.get("iter"), value, epe))
    record["n"] = len(signal_series)

    n_rows = len(scan)
    if n_rows > 0 and record["n_null_signal"] == n_rows:
        record["status"] = STATUS_NULL_SIGNAL
    elif n_rows > 0 and record["n_null_epe"] == n_rows:
        record["status"] = STATUS_NULL_EPE
    elif record["n"] < 2:
        record["status"] = STATUS_INSUFFICIENT_POINTS
    else:
        value, why = spearman_with_reason(signal_series, epe_series)
        record["spearman"] = value
        record["status"] = STATUS_OK if value is not None else why
    return record, observations


def _subset_aggregate(records, key):
    """Group trajectory records by a key and aggregate the non-null correlations in each group."""
    groups = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return {name: _aggregate([r["spearman"] for r in grouped]) for name, grouped in sorted(groups.items())}


def _coverage(usable_rows):
    """Median/mean valid- and texture-fractions and the per-iteration host check costs over usable rows.

    Budget-scan rows carry the photometric ``valid_frac``/``texture_frac`` and the update-magnitude/photometric
    host costs; cycle-scan rows carry ``cycle_low.valid_frac`` and ``cycle_check_s``. Fields for a kind that is
    absent from these rows stay ``null``, so one report shape covers both scan kinds without inventing values.
    """
    def _median(seq):
        return float(statistics.median(seq)) if seq else None

    valid = [float(r["photometric_low"]["valid_frac"]) for r in usable_rows
             if isinstance(r.get("photometric_low"), dict)
             and _is_finite_number(r["photometric_low"].get("valid_frac"))]
    texture = [float(r["photometric_low"]["texture_frac"]) for r in usable_rows
               if isinstance(r.get("photometric_low"), dict)
               and _is_finite_number(r["photometric_low"].get("texture_frac"))]
    signal_cost = [float(r["update_mag_check_s"]) for r in usable_rows
                   if _is_finite_number(r.get("update_mag_check_s"))]
    photo_cost = [float(r["photometric_check_s"]) for r in usable_rows
                  if _is_finite_number(r.get("photometric_check_s"))]
    cycle_valid = [float(r["cycle_low"]["valid_frac"]) for r in usable_rows
                   if isinstance(r.get("cycle_low"), dict)
                   and _is_finite_number(r["cycle_low"].get("valid_frac"))]
    cycle_cost = [float(r["cycle_check_s"]) for r in usable_rows
                  if _is_finite_number(r.get("cycle_check_s"))]
    return {
        "scan_rows": len(usable_rows),
        "valid_frac_median": _median(valid),
        "valid_frac_min": min(valid) if valid else None,
        "texture_frac_median": _median(texture),
        "texture_frac_min": min(texture) if texture else None,
        "signal_check_s_mean": (sum(signal_cost) / len(signal_cost)) if signal_cost else None,
        "photometric_check_s_mean": (sum(photo_cost) / len(photo_cost)) if photo_cost else None,
        "photometric_check_s_total": float(sum(photo_cost)) if photo_cost else None,
        "cycle_valid_frac_median": _median(cycle_valid),
        "cycle_valid_frac_min": min(cycle_valid) if cycle_valid else None,
        "cycle_check_s_mean": (sum(cycle_cost) / len(cycle_cost)) if cycle_cost else None,
        "cycle_check_s_total": float(sum(cycle_cost)) if cycle_cost else None,
    }


def _cross_sectional(observations):
    """Pool (iter, signal, epe) samples by iteration and correlate across trajectories, descriptive only."""
    by_iter = {}
    for it, value, epe in observations:
        by_iter.setdefault(it, ([], []))
        by_iter[it][0].append(value)
        by_iter[it][1].append(epe)
    rows = []
    for it in sorted(by_iter, key=lambda v: (v is None, v)):
        xs, ys = by_iter[it]
        value, why = spearman_with_reason(xs, ys)
        rows.append({
            "iter": it,
            "n": len(xs),
            "spearman": value,
            "status": STATUS_OK if value is not None else why,
        })
    return rows


def audit_manifest(path, signals=DEFAULT_SIGNALS):
    """Audit one manifest and return a JSON-safe per-manifest case. Refuses malformed input via an exception."""
    raw, doc = load_manifest(path)
    kind, scan_block, results = _scan_source(doc)
    directions = sorted(results)

    trajectories_total = 0
    insufficient_history = 0
    usable_rows = []
    per_signal_records = {signal: [] for signal in signals}
    per_signal_observations = {signal: [] for signal in signals}

    for direction in directions:
        for method in sorted(results[direction]):
            pairs = results[direction][method]["per_pair"]
            for pair in pairs:
                _require(isinstance(pair, dict), "missing_per_pair",
                         f"manifest {path} {direction}/{method} has a non-object per_pair row")
                trajectories_total += 1
                if pair.get("reason") == "insufficient_history":
                    insufficient_history += 1
                is_usable = pair.get("reason") != "insufficient_history"
                if is_usable and isinstance(pair.get("scan"), list):
                    usable_rows.extend(pair["scan"])
                for signal in signals:
                    record, observations = _trajectory_record(direction, method, pair, signal)
                    per_signal_records[signal].append(record)
                    if is_usable:
                        per_signal_observations[signal].extend(observations)

    signals_block = {}
    for signal in signals:
        records = per_signal_records[signal]
        status_counts = {}
        for record in records:
            status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
        aggregate = _aggregate([r["spearman"] for r in records])
        signals_block[signal] = {
            "trajectories_total": len(records),
            "trajectories_usable": sum(1 for r in records if r["status"] != STATUS_INSUFFICIENT_HISTORY),
            "trajectories_valid": sum(1 for r in records if r["status"] == STATUS_OK),
            "status_counts": dict(sorted(status_counts.items())),
            "n_null_signal_samples": sum(r["n_null_signal"] for r in records),
            "n_null_epe_samples": sum(r["n_null_epe"] for r in records),
            "spearman_median": aggregate["median"],
            "spearman_mean": aggregate["mean"],
            "positive_fraction": aggregate["positive_fraction"],
            "by_method": _subset_aggregate(records, "method"),
            "by_direction": _subset_aggregate(records, "direction"),
            "cross_sectional_by_iteration": _cross_sectional(per_signal_observations[signal]),
            "trajectories": sorted(
                records, key=lambda r: (r["direction"], r["method"], r["target"] if r["target"] is not None else -1)),
        }

    return {
        "manifest": str(path),
        "ok": True,
        "provenance": _provenance(doc, path, raw),
        "scan": {
            "kind": kind,
            "enabled": scan_block.get("enabled"),
            "max_iters": scan_block.get("max_iters"),
            "history_iters": scan_block.get("history_iters"),
            "thresholds": scan_block.get("thresholds"),
            "oracle": scan_block.get("oracle"),
        },
        # Backwards-compatible alias for the pre-cycle reports that always carried a ``budget_scan`` block.
        "budget_scan": {
            "kind": kind,
            "enabled": scan_block.get("enabled"),
            "max_iters": scan_block.get("max_iters"),
            "history_iters": scan_block.get("history_iters"),
            "thresholds": scan_block.get("thresholds"),
            "oracle": scan_block.get("oracle"),
        },
        "coverage": {
            "trajectories_total": trajectories_total,
            "trajectories_insufficient_history": insufficient_history,
            "trajectories_usable": trajectories_total - insufficient_history,
            **_coverage(usable_rows),
        },
        "signals": signals_block,
    }


def _error_case(path, exc):
    return {"manifest": str(path), "ok": False, "error": {"code": exc.code, "message": str(exc)}}


def run_observable_audit(manifest_paths, signals=DEFAULT_SIGNALS):
    """Audit one or more manifests. A refused manifest becomes an ``ok=False`` case; the rest still run."""
    for signal in signals:
        if signal not in SIGNALS:
            raise ValueError(f"unknown signal {signal!r}; expected one of {tuple(SIGNALS)}")
    cases = []
    for path in manifest_paths:
        try:
            cases.append(audit_manifest(path, signals=signals))
        except ObservableAuditError as exc:
            cases.append(_error_case(path, exc))
    return {
        "schema": SCHEMA,
        "ok": all(case["ok"] for case in cases),
        "signals": list(signals),
        "caveats": list(CAVEATS),
        "cases": cases,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _emit(payload, out):
    if out:
        write_manifest(out, payload)  # atomic: serialize, fsync a temp file, os.replace
    else:
        print(json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline post-hoc audit of candidate-observable convergence signals recorded in "
                    "mlxstab.flow_prediction_real/1 budget-scan manifests. Association only: no threshold is "
                    "searched or selected and no stopping rule is recommended.")
    parser.add_argument("manifests", nargs="+", metavar="MANIFEST.json",
                        help="one or more successful flow_prediction_real manifests with an enabled budget scan")
    parser.add_argument("--signal", action="append", choices=sorted(SIGNALS), default=None,
                        help="signal to audit (repeatable; default: both known signals)")
    parser.add_argument("--out", default=None, help="write the JSON report here atomically (default: stdout)")
    args = parser.parse_args(argv)

    signals = tuple(args.signal) if args.signal else DEFAULT_SIGNALS
    report = run_observable_audit(args.manifests, signals=signals)
    _emit(report, args.out)
    if args.out:
        print(f"observable audit cases={len(report['cases'])} ok={report['ok']}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
