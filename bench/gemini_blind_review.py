"""Research-only Vertex Gemini visual review of the blinded A/B review clips.

What this does
--------------
``bench.multiscale_real_review`` writes blinded two-panel A/B MP4s plus a ``review.json`` report that hides
which side is the multiscale-retention candidate and which is its nearest measured-crop L1 counterpart. This
module sends each of those clips to Vertex AI ``generateContent`` (the anniversary-video precedent: project
``phx-ai-billing``, model ``gemini-3.8-flash``, the user's own ``gcloud`` login, inline ``video/mp4`` bytes,
structured JSON out) and records the reviewer's **per-clip** side preference, then maps it back to the hidden
method using the report's own ``blind_mapping``.

Privacy and blinding
--------------------
The request body carries **only** the clip bytes (inline base64 ``video/mp4``) and one fixed review brief.
No filename, source path, sha256, window, run id, candidate name, grid value or blind mapping is ever placed
in the prompt or anywhere else in the request. The access token is fetched per invocation with
``gcloud auth print-access-token`` and is never printed or stored. Response bodies and tokens are never echoed
into errors; a failed request reports only its HTTP status.

This is a research aid, not a decision
--------------------------------------
The report stores, per clip, the raw A/B/tie side preference and the mapped method preference, with the clip
sha256, the run/window provenance from the source ``review.json``, the model and the prompt/schema versions.
It computes **no** winner, ranking, aggregate or quality score and selects no method.

Resume and reliability
----------------------
``--out`` is read back before a run and each stored judgment is reused only when its clip sha256, model,
prompt version and response-schema version all match. The result document is written atomically after every
completed request, so an interrupted run resumes without re-paying for finished clips. Transient HTTP
``429``/``500``/``503`` and URL/timeout failures are retried with bounded exponential backoff.

Usage::

    python -m bench.gemini_blind_review REVIEW.json [REVIEW.json ...] --out OUT.json [--limit-bytes N]

Network, auth and the CLI live in functions only: importing this module performs no I/O, and nothing runs
unless :func:`main` is called under ``if __name__ == "__main__"``.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath, PureWindowsPath

from .manifest import sha256_file

# --------------------------------------------------------------------------------------
# fixed identity / versions
# --------------------------------------------------------------------------------------

OUTPUT_SCHEMA = "mlxstab.gemini_blind_review/1"
INPUT_SCHEMA = "mlxstab.multiscale_real_review/1"

PROJECT = "phx-ai-billing"
LOCATION = "global"
MODEL = "gemini-3.8-flash"

PROMPT_VERSION = "mlxstab.gemini_blind_review.prompt/1"
RESPONSE_SCHEMA_VERSION = "mlxstab.gemini_blind_review.response/1"

VIDEO_MIME = "video/mp4"
#: Inline payload cap; Vertex inline requests are bounded, and the review clips are a few hundred KB.
MAX_INLINE_BYTES = 15 * 1024 * 1024
REQUEST_TIMEOUT_S = 600.0
RETRY_ATTEMPTS = 5
BACKOFF_BASE_S = 15.0
RETRY_STATUS = (429, 500, 503)

PREFERENCES = ("A", "B", "tie")
CONFIDENCES = ("low", "medium", "high")
PANEL_METHODS = ("multiscale", "l1")

PROVIDER = "Vertex AI REST, authenticated via gcloud user login"

NO_QUALITY_SCORE_NOTE = (
    "Per-clip blind judgments only. No aggregate winner, ranking, total or quality score is computed and no "
    "method is selected; every stored preference is one clip's reviewer call mapped back through that clip's "
    "own hidden blind_mapping."
)

#: The only instructions and context the reviewer receives, alongside the video bytes. It is fixed: it never
#: varies with the clip and it names no method, filename, path, hash, window, run or candidate.
REVIEW_BRIEF = """You are a careful independent visual reviewer comparing two video stabilization results. Each attached video is a single 2-panel blind A/B comparison. The panels show the same source frames synchronized side by side, with A on the left and B on the right. Judge only visible motion stability and image integrity: residual shake or jitter, unwanted drift or float, distortion or warping, crop or border changes, and detail preservation. Do not infer which method or settings produced either panel, and do not guess details you cannot see. If the difference is not clearly visible, answer tie. For each clip return a concise structured judgment: which side appears steadier (A, B, or tie), your confidence (low, medium, or high), the concrete visible evidence for the call or an explicit statement that no reliable difference is visible, any apparent artifacts, and the limitations of your judgment. These are short, low-resolution, translation-only research renders; account for that limit. Never declare an overall method winner."""

#: Vertex structured-output schema for one clip judgment.
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "preference": {"type": "STRING", "enum": ["A", "B", "tie"]},
        "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "visible_evidence": {"type": "STRING"},
        "artifacts": {"type": "STRING"},
        "limitations": {"type": "STRING"},
    },
    "required": ["preference", "confidence", "visible_evidence", "artifacts", "limitations"],
}


#: Stable error code for any path the tool refuses to trust: a pair clip that is not a plain ``.mp4``
#: basename, an ``output.dir`` that is absolute, nested, ``.`` or ``..``, or a result ``--out`` that would
#: overwrite an input ``review.json`` or clip.
UNSAFE_PATH = "unsafe_path"


class GeminiReviewError(ValueError):
    """A review could not be produced. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise GeminiReviewError(code, message)


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------
# request construction and mapping (pure, no I/O)
# --------------------------------------------------------------------------------------

def vertex_url(project=PROJECT, location=LOCATION, model=MODEL):
    return (f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent")


def build_request(video_bytes, *, brief=REVIEW_BRIEF, mime=VIDEO_MIME, schema=None):
    """Build the Vertex ``generateContent`` body from clip bytes and the fixed brief.

    The returned body contains exactly two parts: the inline video and the fixed brief as text. It never
    carries a filename, path, hash, window, run id, candidate name, grid value or blind mapping.
    """
    schema = GEMINI_RESPONSE_SCHEMA if schema is None else schema
    encoded = base64.b64encode(video_bytes).decode("ascii")
    return {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": mime, "data": encoded}},
            {"text": brief},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.2,
        },
    }


def validate_blind_mapping(mapping):
    """Validate the A-left / B-right hidden mapping and return it unchanged."""
    _require(isinstance(mapping, dict), "blind_mapping_invalid", "blind_mapping must be a JSON object")
    left, right = mapping.get("left_panel"), mapping.get("right_panel")
    _require(left in PANEL_METHODS and right in PANEL_METHODS and left != right, "blind_mapping_invalid",
             "blind_mapping.left_panel/right_panel must be distinct members of "
             f"{list(PANEL_METHODS)}")
    labels = mapping.get("video_labels")
    _require(isinstance(labels, dict) and labels.get("left") == "A" and labels.get("right") == "B",
             "blind_mapping_invalid",
             "blind_mapping.video_labels must be exactly {left: 'A', right: 'B'}")
    return mapping


def map_preference(blind_mapping, side):
    """Map a raw A/B/tie reviewer call to the hidden method (``multiscale``/``l1``/``tie``)."""
    mapping = validate_blind_mapping(blind_mapping)
    if side == "tie":
        return "tie"
    if side == "A":
        return mapping["left_panel"]
    if side == "B":
        return mapping["right_panel"]
    raise GeminiReviewError("preference_invalid", "preference must be one of 'A', 'B' or 'tie'")


# --------------------------------------------------------------------------------------
# review.json loading and per-pair validation
# --------------------------------------------------------------------------------------

def _plain_path_component(value):
    """True when ``value`` is one relative path component: no separators, drive or ``.``/``..``."""
    if not isinstance(value, str) or not value.strip() or value in (".", ".."):
        return False
    if "/" in value or "\\" in value:
        return False
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return False
    if PureWindowsPath(value).drive:
        return False
    return PurePosixPath(value).name == value


def _plain_mp4_basename(value):
    """True when ``value`` is a plain ``.mp4`` basename with no directory components or traversal."""
    return _plain_path_component(value) and value.lower().endswith(".mp4")


def load_review(review_path):
    """Read and validate one ``multiscale_real_review.py`` report; return the parsed document.

    Refuses pair clip values that are not plain ``.mp4`` basenames (no absolute paths, nesting, ``.`` or
    ``..``) and an ``output.dir`` that is absolute, nested, ``.`` or ``..``, so neither can escape the
    review directory. The documented same-directory and one-child copied-report layouts remain valid.
    """
    review_path = Path(review_path)
    _require(review_path.is_file(), "review_missing", f"no review file at {review_path.name!r}")
    try:
        document = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        raise GeminiReviewError("review_unreadable", f"cannot read/parse review file {review_path.name!r}")
    _require(isinstance(document, dict), "review_malformed", "review top level must be a JSON object")
    _require(document.get("schema") == INPUT_SCHEMA, "review_schema_unsupported",
             f"review schema is {document.get('schema')!r}, expected {INPUT_SCHEMA!r}")
    _require(document.get("ok") is True, "review_not_ok",
             "review report is not a successful ('ok' is true) run")
    output = document.get("output")
    _require(isinstance(output, dict) and isinstance(output.get("dir"), str) and output["dir"].strip(),
             "review_malformed", "review output.dir is missing or empty")
    _require(_plain_path_component(output["dir"]), UNSAFE_PATH,
             "review output.dir must be a plain relative directory name (no absolute path, nesting, '.' or "
             "'..')")
    pairs = document.get("pairs")
    _require(isinstance(pairs, list) and pairs, "review_malformed", "review pairs must be a non-empty list")
    provenance = document.get("provenance")
    _require(isinstance(provenance, dict), "review_malformed", "review provenance object is missing")
    for field in ("run", "run_id", "source_sha256"):
        _require(isinstance(provenance.get(field), str) and provenance[field].strip(), "review_malformed",
                 f"review provenance.{field} is missing or empty")
    window = provenance.get("source_window")
    _require(isinstance(window, dict) and _finite(window.get("start_s")) and _finite(window.get("duration_s")),
             "review_malformed", "review provenance.source_window.start_s/duration_s must be finite")
    for index, pair in enumerate(pairs):
        _require(isinstance(pair, dict), "pair_malformed", f"review pair {index} must be a JSON object")
        _require(isinstance(pair.get("clip"), str) and pair["clip"].strip(), "pair_malformed",
                 f"review pair {index} clip is missing or empty")
        _require(_plain_mp4_basename(pair["clip"]), UNSAFE_PATH,
                 f"review pair {index} clip must be a plain .mp4 basename")
        validate_blind_mapping(pair.get("blind_mapping"))
    return document


def _finite(value):
    try:
        return value is not None and float("-inf") < float(value) < float("inf")
    except (TypeError, ValueError):
        return False


def resolve_clip_path(review_path, review, clip):
    """Resolve a clip relative to the review file using ``output.dir`` then a sibling fallback.

    ``multiscale_real_review`` writes ``review.json`` inside the clip directory recorded as ``output.dir``;
    those clips are siblings. A report copied one level up keeps its ``output.dir`` and nests the clips. The
    resolver accepts both: it prefers ``<review dir>/<output.dir>/<clip>`` and falls back to
    ``<review dir>/<clip>``.

    Every existing candidate is canonicalised before it is trusted: a clip that resolves (through a symlink
    or a symlinked directory) outside the directory holding ``review.json`` is refused with
    :data:`UNSAFE_PATH`. Valid sibling and one-child copied-report layouts stay inside that directory.
    """
    review_path = Path(review_path)
    base = review_path.parent
    review_dir = base.resolve()
    out_dir = review.get("output", {}).get("dir")
    candidates = []
    if isinstance(out_dir, str) and out_dir.strip() and out_dir != base.name:
        candidates.append(base / out_dir / clip)
    candidates.append(base / clip)
    for candidate in candidates:
        if candidate.is_file():
            _require(candidate.resolve().is_relative_to(review_dir), UNSAFE_PATH,
                     f"review clip {clip!r} resolves outside the directory holding {review_path.name!r}")
            return candidate
    return candidates[0]


def validate_clip(clip_path, limit_bytes=MAX_INLINE_BYTES):
    """Validate that a clip exists and fits the inline payload cap; return its size in bytes."""
    path = Path(clip_path)
    _require(path.is_file(), "clip_missing", f"no review clip at {path.name!r}")
    size = path.stat().st_size
    _require(size > 0, "clip_empty", f"review clip {path.name!r} is empty")
    _require(size <= int(limit_bytes), "clip_too_large",
             f"review clip {path.name!r} is {size} bytes, over the {int(limit_bytes)}-byte inline cap")
    return size


def collect_tasks(review_paths, *, limit_bytes=MAX_INLINE_BYTES):
    """Validate every review file and clip up front and return one task dict per pair."""
    tasks = []
    for review_path in review_paths:
        review_path = Path(review_path)
        review = load_review(review_path)
        provenance = review["provenance"]
        for index, pair in enumerate(review["pairs"]):
            clip = pair["clip"]
            clip_path = resolve_clip_path(review_path, review, clip)
            size = validate_clip(clip_path, limit_bytes=limit_bytes)
            tasks.append({
                "review_path": str(review_path),
                "review_schema": review["schema"],
                "pair_index": pair.get("pair_index", index),
                "clip": clip,
                "clip_path": clip_path,
                "clip_bytes": int(size),
                "clip_sha256": sha256_file(clip_path),
                "blind_mapping": validate_blind_mapping(pair["blind_mapping"]),
                "run": provenance["run"],
                "run_id": provenance["run_id"],
                "source_sha256": provenance["source_sha256"],
                "source_window": {"start_s": float(provenance["source_window"]["start_s"]),
                                  "duration_s": float(provenance["source_window"]["duration_s"])},
                "case": provenance.get("case"),
            })
    return tasks


# --------------------------------------------------------------------------------------
# Vertex access (network lives here and nowhere else)
# --------------------------------------------------------------------------------------

def access_token(runner=None):
    """Fetch the user's ``gcloud`` access token. The token is returned but never printed or stored."""
    runner = runner or subprocess.run
    try:
        proc = runner(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        raise GeminiReviewError("auth_failed", "could not run gcloud; run `gcloud auth login` yourself")
    token = (proc.stdout or "").strip()
    if proc.returncode != 0 or not token:
        raise GeminiReviewError("auth_failed", "no gcloud login; run `gcloud auth login` yourself")
    return token


def call_vertex(body, *, token, model=MODEL, project=PROJECT, location=LOCATION, opener=None, sleep=None,
                attempts=RETRY_ATTEMPTS, backoff_base=BACKOFF_BASE_S, timeout=REQUEST_TIMEOUT_S):
    """POST one ``generateContent`` request, retrying transient failures with bounded exponential backoff.

    HTTP ``429``/``500``/``503`` and URL/timeout failures are retried up to ``attempts`` times; anything else
    is raised as :class:`GeminiReviewError`. Error messages carry the HTTP status only, never the response
    body or the token.
    """
    opener = opener or urllib.request.urlopen
    sleep = sleep or time.sleep
    request_bytes = json.dumps(body).encode("utf-8")
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    url = vertex_url(project, location, model)
    for attempt in range(int(attempts)):
        request = urllib.request.Request(url, request_bytes, headers)
        try:
            with opener(request, timeout=timeout) as response:
                raw = response.read()
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise GeminiReviewError("response_unparseable", "Vertex HTTP response was not valid JSON")
        except urllib.error.HTTPError as exc:
            status = int(getattr(exc, "code", 0))
            if status in RETRY_STATUS and attempt < int(attempts) - 1:
                sleep(float(backoff_base) * (2 ** attempt))
                continue
            raise GeminiReviewError(f"http_{status}", f"Vertex request failed with HTTP status {status}")
        except OSError as exc:  # URLError, TimeoutError, ConnectionError, socket errors
            if attempt < int(attempts) - 1:
                sleep(float(backoff_base) * (2 ** attempt))
                continue
            raise GeminiReviewError("network_error", f"Vertex request failed: {type(exc).__name__}")
    raise GeminiReviewError("retry_exhausted", "Vertex request exhausted its retries")


# --------------------------------------------------------------------------------------
# response parsing (sanitized)
# --------------------------------------------------------------------------------------

def validate_judgment(judgment):
    """Validate one structured reviewer judgment; return a normalized copy."""
    _require(isinstance(judgment, dict), "response_invalid", "reviewer judgment was not a JSON object")
    preference = judgment.get("preference")
    _require(preference in PREFERENCES, "response_invalid", "reviewer preference is not a recognized value")
    confidence = judgment.get("confidence")
    _require(confidence in CONFIDENCES, "response_invalid", "reviewer confidence is not a recognized value")
    normalized = {"preference": preference, "confidence": confidence}
    for field in ("visible_evidence", "artifacts", "limitations"):
        value = judgment.get(field)
        _require(isinstance(value, str), "response_invalid", f"reviewer {field} must be a string")
        normalized[field] = value
    return normalized


def parse_response(response):
    """Extract and validate the structured judgment from a Vertex ``generateContent`` response."""
    _require(isinstance(response, dict), "response_malformed", "Vertex response was not a JSON object")
    candidates = response.get("candidates")
    _require(isinstance(candidates, list) and candidates, "response_no_candidates",
             "Vertex response carried no candidates")
    first = candidates[0]
    try:
        text = first["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise GeminiReviewError("response_malformed", "Vertex candidate carried no text part")
    _require(isinstance(text, str) and text.strip(), "response_malformed", "Vertex candidate text is empty")
    try:
        judgment = json.loads(text)
    except ValueError:
        raise GeminiReviewError("response_unparseable", "Vertex candidate text was not valid JSON")
    return validate_judgment(judgment)


# --------------------------------------------------------------------------------------
# atomic output + resume cache
# --------------------------------------------------------------------------------------

def write_json_atomic(path, payload):
    """Serialize first, then fsync a temp file and ``os.replace`` it into place.

    A serialization error leaves the destination untouched (no temp file is created); a write or replace
    error unlinks the temp file and leaves the prior destination intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(path)


def _load_existing(out_path):
    path = Path(out_path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise GeminiReviewError("resume_unreadable", f"cannot read existing result {path.name!r}")
    if not text.strip():
        return None
    try:
        document = json.loads(text)
    except ValueError:
        raise GeminiReviewError("resume_unreadable", f"existing result {path.name!r} is not valid JSON")
    _require(isinstance(document, dict) and document.get("schema") == OUTPUT_SCHEMA,
             "resume_schema_mismatch",
             f"existing result schema is {document.get('schema')!r}, expected {OUTPUT_SCHEMA!r}")
    return document


def _cache_key(clip_sha256, model, prompt_version, response_schema_version):
    return (clip_sha256, model, prompt_version, response_schema_version)


def _cache_from_document(document):
    cache = {}
    if not isinstance(document, dict):
        return cache
    for record in document.get("judgments", []):
        if not isinstance(record, dict):
            continue
        key = _cache_key(record.get("clip_sha256"), record.get("model"),
                         record.get("prompt_version"), record.get("response_schema_version"))
        if all(part is not None for part in key):
            cache[key] = record
    return cache


def _raw_from_record(record):
    return {
        "preference": record.get("raw_side_preference"),
        "confidence": record.get("confidence"),
        "visible_evidence": record.get("visible_evidence"),
        "artifacts": record.get("artifacts"),
        "limitations": record.get("limitations"),
    }


def _judgment(task, raw, timestamp, model):
    return {
        "review_file": Path(task["review_path"]).name,
        "review_schema": task["review_schema"],
        "pair_index": task["pair_index"],
        "clip": task["clip"],
        "clip_sha256": task["clip_sha256"],
        "clip_bytes": task["clip_bytes"],
        "blind_mapping": dict(task["blind_mapping"]),
        "raw_side_preference": raw["preference"],
        "mapped_preference": map_preference(task["blind_mapping"], raw["preference"]),
        "confidence": raw["confidence"],
        "visible_evidence": raw["visible_evidence"],
        "artifacts": raw["artifacts"],
        "limitations": raw["limitations"],
        "run": task["run"],
        "run_id": task["run_id"],
        "source_sha256": task["source_sha256"],
        "source_window": dict(task["source_window"]),
        "case": task["case"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "timestamp": timestamp,
    }


def _base_document(model, project, location, now):
    return {
        "schema": OUTPUT_SCHEMA,
        "complete": False,
        "provider": PROVIDER,
        "no_quality_score_note": NO_QUALITY_SCORE_NOTE,
        "model": model,
        "project": project,
        "location": location,
        "prompt": {"version": PROMPT_VERSION, "brief": REVIEW_BRIEF},
        "response_schema": {"version": RESPONSE_SCHEMA_VERSION, "schema": GEMINI_RESPONSE_SCHEMA},
        "started_at": now(),
        "judgments": [],
    }


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def run_review_set(review_paths, out_path, *, token=None, opener=None, sleep=None, model=MODEL,
                   project=PROJECT, location=LOCATION, limit_bytes=MAX_INLINE_BYTES, now=None,
                   write=True, resume=True):
    """Review every clip in the given ``review.json`` files and return the result document.

    All clips are validated before any request is made. An existing ``out_path`` is read back and reused only
    when a judgment's clip sha256, model, prompt version and response-schema version all match. The document
    is written atomically after every completed request; a fully finished run sets ``complete`` true.
    """
    now_fn = now or _utcnow
    tasks = collect_tasks(review_paths, limit_bytes=limit_bytes)
    protected = {Path(path).resolve() for path in review_paths}
    protected |= {Path(task["review_path"]).resolve() for task in tasks}
    protected |= {Path(task["clip_path"]).resolve() for task in tasks}
    _require(Path(out_path).resolve() not in protected, UNSAFE_PATH,
             "result --out would overwrite an input review.json or clip")
    cache = _cache_from_document(_load_existing(out_path)) if resume else {}
    document = _base_document(model, project, location, now_fn)
    resolved_token = token
    for task in tasks:
        key = _cache_key(task["clip_sha256"], model, PROMPT_VERSION, RESPONSE_SCHEMA_VERSION)
        cached = cache.get(key)
        if cached is not None:
            judgment = _judgment(task, _raw_from_record(cached), cached.get("timestamp") or now_fn(), model)
        else:
            if resolved_token is None:
                resolved_token = access_token()
            body = build_request(Path(task["clip_path"]).read_bytes())
            raw = parse_response(call_vertex(body, token=resolved_token, model=model, project=project,
                                             location=location, opener=opener, sleep=sleep))
            judgment = _judgment(task, raw, now_fn(), model)
            cache[key] = judgment
        document["judgments"].append(judgment)
        if write:
            write_json_atomic(out_path, document)
    document["complete"] = True
    document["finished_at"] = now_fn()
    if write:
        write_json_atomic(out_path, document)
    return document


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="bench/gemini_blind_review.py",
        description="Research-only Vertex Gemini review of blinded multiscale A/B clips. Sends only the clip "
                    "bytes plus a fixed brief, maps the A/B/tie call back through each review.json's hidden "
                    "blind_mapping, and stores per-clip judgments with no aggregate winner or score.")
    parser.add_argument("reviews", nargs="+", metavar="REVIEW.json",
                        help="one or more multiscale_real_review.py review.json reports")
    parser.add_argument("--out", required=True, help="path to the result JSON (written atomically)")
    parser.add_argument("--model", default=MODEL, help="Vertex model id (default: %(default)s)")
    parser.add_argument("--limit-bytes", type=int, default=MAX_INLINE_BYTES,
                        help="maximum inline clip size in bytes (default: %(default)s)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore and overwrite any existing result JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = run_review_set(args.reviews, args.out, model=args.model, limit_bytes=args.limit_bytes,
                                  resume=not args.no_resume)
    except GeminiReviewError as exc:
        print(json.dumps({"schema": OUTPUT_SCHEMA, "ok": False,
                          "error": {"code": exc.code, "message": str(exc)}}, sort_keys=True), file=sys.stderr)
        return 2
    print(f"schema={document['schema']} complete={document['complete']} "
          f"judgments={len(document['judgments'])} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_SCHEMA",
    "INPUT_SCHEMA",
    "PROJECT",
    "LOCATION",
    "MODEL",
    "PROMPT_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "VIDEO_MIME",
    "MAX_INLINE_BYTES",
    "REQUEST_TIMEOUT_S",
    "RETRY_ATTEMPTS",
    "BACKOFF_BASE_S",
    "RETRY_STATUS",
    "PREFERENCES",
    "CONFIDENCES",
    "PANEL_METHODS",
    "PROVIDER",
    "NO_QUALITY_SCORE_NOTE",
    "REVIEW_BRIEF",
    "GEMINI_RESPONSE_SCHEMA",
    "UNSAFE_PATH",
    "GeminiReviewError",
    "vertex_url",
    "build_request",
    "validate_blind_mapping",
    "map_preference",
    "load_review",
    "resolve_clip_path",
    "validate_clip",
    "collect_tasks",
    "access_token",
    "call_vertex",
    "validate_judgment",
    "parse_response",
    "write_json_atomic",
    "run_review_set",
    "build_parser",
    "main",
]
