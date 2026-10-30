"""Local web UI served by ``http.server`` — the Streamlit replacement.

Why not Streamlit: it dragged in ~291 MB of dependencies (pyarrow, pandas,
pydeck, pillow, altair) for a page that needs a file input, a few sliders and a
progress bar. This module serves one static HTML file and a small JSON API from
the standard library, on top of the unchanged :mod:`whisper_pipeline` engine.

Design notes that matter for behaviour:

* **One job at a time.** The engine holds a model (up to ~764 MB resident) and
  saturates its intra-op threads; a second concurrent job would only halve both.
* **Transcription runs in a thread**, the browser polls ``/api/job`` every
  ~600 ms. Long files therefore survive a page reload, and no request is held
  open for minutes.
* **Uploads stream to disk** in 1 MiB chunks — a 4 GB video never sits in RAM.
* **Cancel is cooperative**: the flag is checked in the progress callback, so it
  takes effect at the next decoded block rather than after the whole file.
* **Bound to 127.0.0.1** and POSTs require an ``X-Whisper`` header, so a random
  web page cannot drive the API through the browser (a cross-origin request with
  a custom header needs a preflight, which this server never grants).

Run it with ``python -m whisper_pipeline.web`` (or ``python app.py``).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import traceback
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import fetch, models
from .audio import AudioExtractionError, MediaInfo, probe
from .batch import BatchItem, BatchOutcome, expand_inputs, parse_path_text, run_batch
from .engine import ModelLoadError, Transcriber, VadOptions
from .export import FORMATS, bundle_zip, filename, payloads

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "python-whisper-uploads"

#: Formats offered by the UI. The CLI default stays narrower (txt,srt).
UI_FORMATS: tuple[str, ...] = FORMATS

#: Refuse anything larger than this, so a stray drag of a disk image fails fast.
MAX_UPLOAD_BYTES = int(os.environ.get("WHISPER_MAX_UPLOAD_GB", "16")) * 1000**3

#: How many finished jobs keep their results downloadable. Each holds the full
#: segment tuples, so this is a memory ceiling, not a history feature.
KEEP_JOBS = 3

_MIME = {
    ".txt": "text/plain; charset=utf-8",
    ".srt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".zip": "application/zip",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class JobCancelled(RuntimeError):
    """Raised inside the progress callback to stop the running job."""


class BadRequest(ValueError):
    """A 400 with a message the browser can show verbatim."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def human_size(num_bytes: int) -> str:
    if num_bytes < 1_000_000:
        return f"{num_bytes / 1e3:.0f} kB"
    if num_bytes < 1_000_000_000:
        return f"{num_bytes / 1e6:.1f} MB"
    return f"{num_bytes / 1e9:.2f} GB"


def safe_name(raw: str) -> str:
    """Strip any directory part from an uploaded name and forbid separators."""
    name = Path(str(raw).replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ()]", "_", name).strip(" .")
    return name or "upload.bin"


def probe_entry(path: Path) -> dict[str, Any]:
    """Describe one queued file; never raises."""
    entry: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "folder": str(path.parent),
        "size": path.stat().st_size if path.exists() else 0,
    }
    entry["size_label"] = human_size(entry["size"])
    try:
        info: MediaInfo = probe(path)
    except AudioExtractionError as exc:
        entry["error"] = str(exc)
        return entry
    entry.update(
        {
            "duration": info.duration,
            "duration_label": info.duration_label,
            "audio_tracks": len(info.audio_streams),
            "default_track": info.stream().label,
            "has_video": bool(info.video_streams),
            "tracks": [
                {
                    "index": stream.index,
                    "label": stream.label,
                    "language": stream.language,
                    "channels": stream.channels,
                    "is_default": stream.is_default,
                }
                for stream in info.audio_streams
            ],
        }
    )
    return entry


def _as_int(payload: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{key} must be a whole number") from exc
    return max(lo, min(hi, value))


def _as_float(payload: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{key} must be a number") from exc
    return max(lo, min(hi, value))


def _as_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


# --------------------------------------------------------------------------- #
# Settings: one validated snapshot of what the UI sent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Settings:
    """Everything a job needs, already clamped to safe ranges."""

    model_key: str
    device: str
    threads: int
    vad: VadOptions | None
    max_chars: int
    max_cue: float
    formats: tuple[str, ...]
    stream_index: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Settings:
        model_key = str(payload.get("model") or models.default_model().key)
        installed = {spec.key for spec in models.available_models()}
        if model_key not in installed:
            raise BadRequest(f"model {model_key!r} is not installed on disk")

        device = "cuda" if str(payload.get("device", "cpu")).lower() == "cuda" else "cpu"
        if device == "cuda" and not models.supports_gpu():
            device = "cpu"  # the engine reports the same fallback, but say why up front

        threads = _as_int(payload, "threads", 2, 1, max(1, os.cpu_count() or 2))
        max_chars = _as_int(payload, "max_chars", 84, 20, 200)
        max_cue = _as_float(payload, "max_cue", 6.0, 1.0, 20.0)

        vad: VadOptions | None = None
        if _as_bool(payload, "use_vad", True):
            vad = VadOptions(
                model_path=models.vad_model_path(),
                threshold=_as_float(payload, "vad_threshold", 0.5, 0.05, 0.95),
                min_speech_duration=_as_float(payload, "vad_min_speech", 0.1, 0.05, 5.0),
                min_silence_duration=_as_float(payload, "vad_min_silence", 0.5, 0.05, 5.0),
                max_speech_duration=_as_float(payload, "vad_max_speech", 20.0, 1.0, 60.0),
            )

        requested = payload.get("stream_index")
        stream_index = (
            None if requested in (None, "", "auto") else _as_int(payload, "stream_index", 0, 0, 64)
        )

        formats = tuple(fmt for fmt in payload.get("formats", UI_FORMATS) if isinstance(fmt, str))
        formats = tuple(fmt for fmt in formats if fmt in UI_FORMATS) or UI_FORMATS
        return cls(model_key, device, threads, vad, max_chars, max_cue, formats, stream_index)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_key,
            "device": self.device,
            "threads": self.threads,
            "use_vad": self.vad is not None,
            "stream_index": self.stream_index,
            "formats": list(self.formats),
        }


# --------------------------------------------------------------------------- #
# Model cache: one live model, like Streamlit's cache_resource(max_entries=1)
# --------------------------------------------------------------------------- #
class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, str, int] | None = None
        self._transcriber: Transcriber | None = None

    def get(self, settings: Settings) -> Transcriber:
        key = (settings.model_key, settings.device, settings.threads)
        with self._lock:
            if self._key == key and self._transcriber is not None:
                return self._transcriber
            spec = models.get_model(settings.model_key)
            transcriber = Transcriber(spec, device=settings.device, num_threads=settings.threads)
            self._key, self._transcriber = key, transcriber
            return transcriber


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class JobFile:
    path: Path
    status: str = "queued"  # queued | running | ok | skipped | failed | cancelled
    label: str = ""
    error: str | None = None
    segments: int | None = None
    words: int | None = None
    duration_label: str | None = None
    rtf: float | None = None


@dataclass(slots=True)
class Job:
    id: str
    kind: str  # "transcribe" | "download"
    settings: Settings | None = None
    files: list[JobFile] = field(default_factory=list)
    status: str = "running"  # running | done | failed | cancelled
    progress: float = 0.0
    message: str = "starting…"
    error: str | None = None
    note: str | None = None
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    outcome: BatchOutcome | None = None
    cancel_requested: bool = False

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def snapshot(self) -> dict[str, Any]:
        """The JSON the browser polls for."""
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "note": self.note,
            "elapsed": round(self.elapsed, 1),
            "cancel_requested": self.cancel_requested,
            "settings": self.settings.describe() if self.settings else None,
            "files": [
                {
                    "index": index,
                    "name": item.path.name,
                    "path": str(item.path),
                    "status": item.status,
                    "label": item.label,
                    "error": item.error,
                    "segments": item.segments,
                    "words": item.words,
                    "duration": item.duration_label,
                    "rtf": item.rtf,
                }
                for index, item in enumerate(self.files)
            ],
        }
        outcome = self.outcome
        if outcome is not None:
            data["summary"] = {
                "ok": len(outcome.ok),
                "skipped": len(outcome.skipped),
                "failed": len(outcome.failed),
                "total": len(outcome.items),
                "audio_seconds": outcome.audio_seconds,
                "wall_seconds": round(outcome.wall_seconds, 1),
                "rtf": outcome.rtf,
                "words": outcome.word_count,
                "problems": outcome.problems,
            }
            data["results"] = [
                {
                    "index": index,
                    "name": item.path.name,
                    "status": item.status,
                    "segments": item.result.segment_count if item.result else None,
                    "words": item.result.word_count if item.result else None,
                    "duration": item.result.media.duration_label if item.result else None,
                    "rtf": item.result.rtf if item.result else None,
                    "formats": list(UI_FORMATS) if item.result else [],
                    "error": item.error,
                }
                for index, item in enumerate(outcome.items)
            ]
        return data


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #
class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_cache = ModelCache()
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._counter = 0

    # -- jobs ------------------------------------------------------------- #
    def _next_id(self) -> str:
        self._counter += 1
        return f"{int(time.time())}-{self._counter}"

    def start_job(self, job: Job, runner, *args, **kwargs) -> Job:
        """Register ``job`` and run ``runner(job, *args)`` on a worker thread."""
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            while len(self.order) > KEEP_JOBS:
                self.jobs.pop(self.order.pop(0), None)
        thread = threading.Thread(
            target=runner,
            args=(job, *args),
            kwargs=kwargs,
            name=f"job-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def active(self) -> Job | None:
        with self.lock:
            for job_id in reversed(self.order):
                job = self.jobs.get(job_id)
                if job is not None and job.status == "running":
                    return job
        return None

    def job(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def latest(self) -> Job | None:
        with self.lock:
            for job_id in reversed(self.order):
                job = self.jobs.get(job_id)
                if job is not None and job.kind == "transcribe":
                    return job
        return None

    # -- runners ---------------------------------------------------------- #
    def run_transcription(self, job: Job, paths: list[Path]) -> None:
        settings = job.settings
        assert settings is not None
        try:
            transcriber = self.model_cache.get(settings)
            job.note = transcriber.fallback_note

            def on_progress(fraction: float, message: str) -> None:
                job.progress = max(job.progress, min(1.0, fraction))
                job.message = message
                if job.cancel_requested:
                    raise JobCancelled

            def on_start(item: BatchItem, index: int, total: int) -> None:
                if index - 1 < len(job.files):
                    job.files[index - 1].status = "running"
                job.message = f"File {index} of {total}: {item.name}"

            def on_item(item: BatchItem) -> None:
                entry = next((f for f in job.files if f.path == item.path), None)
                if entry is None:
                    return
                entry.status = item.status
                entry.label = item.label
                entry.error = item.error
                if item.result is not None:
                    entry.segments = item.result.segment_count
                    entry.words = item.result.word_count
                    entry.duration_label = item.result.media.duration_label
                    entry.rtf = item.result.rtf

            outcome = run_batch(
                paths,
                transcriber,
                vad=settings.vad,
                formats=settings.formats,
                stream_index=settings.stream_index,
                max_chars=settings.max_chars,
                max_cue=settings.max_cue,
                on_progress=on_progress,
                on_start=on_start,
                on_item=on_item,
            )
            job.outcome = outcome
            job.progress = 1.0
            job.status = "done"
            counts = f"{len(outcome.ok)} of {len(outcome.items)} file(s) transcribed"
            if outcome.skipped:
                counts += f", {len(outcome.skipped)} skipped"
            job.message = counts
        except JobCancelled:
            job.status = "cancelled"
            job.message = "cancelled"
            for entry in job.files:
                if entry.status in {"queued", "running"}:
                    entry.status = "cancelled"
        except (ModelLoadError, AudioExtractionError, OSError) as exc:
            job.status, job.error = "failed", str(exc)
            job.message = "failed"
        except Exception as exc:  # noqa: BLE001 - a crash must reach the browser
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.message = "failed"
            traceback.print_exc()
        finally:
            job.finished = time.monotonic()

    def run_download(self, job: Job, model_key: str) -> None:
        try:
            assets = fetch.pending_for(model_key)

            def on_progress(fraction: float, message: str) -> None:
                if job.cancel_requested:
                    raise JobCancelled
                job.progress = max(job.progress, min(1.0, fraction))
                job.message = message

            if not assets:
                job.message = "already on disk"
            else:
                fetch.download(assets, progress=on_progress)
            job.progress = 1.0
            job.status = "done"
            job.message = "model ready"
        except JobCancelled:
            job.status, job.message = "cancelled", "cancelled"
        except Exception as exc:  # noqa: BLE001
            job.status, job.error, job.message = "failed", str(exc), "download failed"
        finally:
            job.finished = time.monotonic()


STATE = AppState()


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "python-whisper"
    protocol_version = "HTTP/1.1"

    # -- plumbing --------------------------------------------------------- #
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib name
        if self.path.startswith(("/api/job", "/api/state")):
            return  # polling would drown the console
        super().log_message(fmt, *args)

    def _send(self, status: int, payload: bytes, ctype: str, extra: dict[str, str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise BadRequest("empty request body")
        if length > 4 * 1024 * 1024:
            raise BadRequest("request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BadRequest(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise BadRequest("expected a JSON object")
        return payload

    def _check_origin(self) -> None:
        """Cheap CSRF/DNS-rebinding guard for state-changing requests."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in {"127.0.0.1", "localhost", "::1", "[::1]", ""}:
            raise BadRequest(
                f"unexpected Host header: {host!r}",
            )
        if self.headers.get("X-Whisper") != "1":
            raise BadRequest("missing X-Whisper header")

    # -- GET -------------------------------------------------------------- #
    def do_HEAD(self) -> None:  # noqa: N802 - stdlib name
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if route in {"/", "/index.html"}:
                return self._serve_static("index.html")
            if route == "/api/state":
                return self._json(self.state_payload())
            if route == "/api/job":
                return self._json(self.job_payload(query))
            if route.startswith("/download/"):
                return self._serve_download(route, query)
            if route == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
        except BadRequest as exc:
            return self._error(str(exc), 400)
        except FileNotFoundError as exc:
            return self._error(str(exc), 404)
        return self._error(f"no route for {route}", 404)

    # -- POST ------------------------------------------------------------- #
    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if route == "/api/upload":
                self._check_origin()
                return self._json(self.handle_upload(query))
            self._check_origin()
            if route == "/api/paths":
                return self._json(self.handle_paths())
            if route == "/api/transcribe":
                return self._json(self.handle_transcribe())
            if route == "/api/cancel":
                return self._json(self.handle_cancel())
            if route == "/api/download-model":
                return self._json(self.handle_download_model())
        except BadRequest as exc:
            return self._error(str(exc), 400)
        except (KeyError, ValueError) as exc:
            return self._error(str(exc), 400)
        return self._error(f"no route for {route}", 404)

    # -- state ------------------------------------------------------------ #
    def state_payload(self) -> dict[str, Any]:
        installed = models.available_models()
        active = STATE.active()
        latest = STATE.latest()
        return {
            "providers": list(models.available_providers()),
            "gpu": models.supports_gpu(),
            "cpu_count": os.cpu_count() or 2,
            "models": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "description": spec.description,
                    "detail": spec.describe(),
                    "languages": spec.languages,
                }
                for spec in installed
            ],
            "downloads": [
                {
                    "key": key,
                    "label": models.CATALOG[key].label if key in models.CATALOG else key,
                    "assets": [*fetch.MODEL_DOWNLOADS[key], "vad"],
                    "mb": sum(
                        fetch.ASSET_SIZES.get(a, 0.0) for a in (*fetch.MODEL_DOWNLOADS[key], "vad")
                    ),
                }
                for key in fetch.MODEL_DOWNLOADS
            ],
            "formats": list(UI_FORMATS),
            "defaults": {
                "model": installed[0].key if installed else None,
                "device": "cpu",
                "threads": min(2, os.cpu_count() or 2),
                "use_vad": True,
                "vad_threshold": 0.5,
                "vad_min_speech": 0.1,
                "vad_min_silence": 0.5,
                "vad_max_speech": 20.0,
                "max_chars": 84,
                "max_cue": 6.0,
            },
            "upload_dir": str(UPLOAD_DIR),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "job": (active or latest).snapshot() if (active or latest) else None,
            "busy": active is not None,
        }

    def job_payload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        job_id = (query.get("id") or [""])[0]
        job = STATE.job(job_id) if job_id else (STATE.active() or STATE.latest())
        if job is None:
            raise BadRequest("no such job")
        data = job.snapshot()
        data["busy"] = STATE.active() is not None
        return data

    # -- uploads ---------------------------------------------------------- #
    def handle_upload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        name = safe_name((query.get("name") or ["upload.bin"])[0])
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise BadRequest("empty upload")
        if length > MAX_UPLOAD_BYTES:
            raise BadRequest(f"file too large: {human_size(length)}")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / name
        counter = 2
        while target.exists() and target.stat().st_size != length:
            target = UPLOAD_DIR / f"{Path(name).stem} ({counter}){Path(name).suffix}"
            counter += 1

        partial = target.with_name(target.name + ".part")
        remaining = length
        with partial.open("wb") as sink:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                sink.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            partial.unlink(missing_ok=True)
            raise BadRequest("upload truncated")
        partial.replace(target)

        entry = probe_entry(target)
        entry["uploaded"] = True
        return entry

    def handle_paths(self) -> dict[str, Any]:
        payload = self._body()
        text = payload.get("text")
        raw: list[str] = []
        if isinstance(text, str) and text.strip():
            raw.extend(parse_path_text(text))
        given = payload.get("paths")
        if isinstance(given, list):
            raw.extend(str(item) for item in given if str(item).strip())
        if not raw:
            raise BadRequest("no paths given")

        files, problems = expand_inputs(raw)
        return {
            "entries": [probe_entry(path) for path in files],
            "problems": problems,
        }

    # -- transcription ---------------------------------------------------- #
    def handle_transcribe(self) -> dict[str, Any]:
        if STATE.active() is not None:
            raise BadRequest("a job is already running")
        payload = self._body()

        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise BadRequest("no files to transcribe")
        files, problems = expand_inputs([str(item) for item in raw_paths])
        if not files:
            raise BadRequest("; ".join(problems) or "no readable files")
        if any(not path.is_file() for path in files):
            raise BadRequest("every path must be an existing file")

        settings = Settings.from_payload(payload)
        job = Job(
            id=STATE._next_id(),
            kind="transcribe",
            settings=settings,
            files=[JobFile(path=path) for path in files],
            message=f"queued {len(files)} file(s)",
        )
        STATE.start_job(job, STATE.run_transcription, files)
        data = job.snapshot()
        data["problems"] = problems
        return data

    def handle_cancel(self) -> dict[str, Any]:
        payload = self._body()
        job_id = str(payload.get("id") or "")
        job = STATE.job(job_id) if job_id else STATE.active()
        if job is None or job.status != "running":
            raise BadRequest("no running job")
        job.cancel_requested = True
        job.message = "cancelling…"
        return job.snapshot()

    # -- model download --------------------------------------------------- #
    def handle_download_model(self) -> dict[str, Any]:
        if STATE.active() is not None:
            raise BadRequest("a job is already running")
        payload = self._body()
        model_key = str(payload.get("model") or "")
        if model_key not in fetch.MODEL_DOWNLOADS:
            raise BadRequest(f"no download configured for {model_key!r}")
        job = Job(
            id=STATE._next_id(),
            kind="download",
            message=f"downloading {model_key}",
        )
        STATE.start_job(job, STATE.run_download, model_key)
        return job.snapshot()

    # -- static and downloads --------------------------------------------- #
    def _serve_static(self, name: str) -> None:
        target = (STATIC_DIR / name).resolve()
        if not target.is_file() or STATIC_DIR not in target.parents:
            raise FileNotFoundError(f"missing static asset {name}")
        ctype = _MIME.get(
            target.suffix, mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )
        self._send(200, target.read_bytes(), ctype)

    def _serve_download(self, route: str, query: dict[str, list[str]]) -> None:
        parts = [part for part in route.split("/") if part]
        # /download/<job id>/zip  or  /download/<job id>/<index>/<fmt>
        if len(parts) not in {3, 4}:
            raise BadRequest("bad download path")
        job = STATE.job(parts[1])
        if job is None or job.outcome is None:
            raise FileNotFoundError("that result is no longer in memory")
        limit = int((query.get("limit") or ["0"])[0] or 0)

        if len(parts) == 3 and parts[2] == "zip":
            ok = job.outcome.ok
            if not ok:
                raise FileNotFoundError("nothing to zip")
            settings = job.settings
            payload = bundle_zip(
                [item.result for item in ok],
                settings.formats if settings else UI_FORMATS,
                max_chars=settings.max_chars if settings else 84,
                max_cue=settings.max_cue if settings else 6.0,
            )
            name = "transcripts.zip"
            return self._send(
                200,
                payload,
                "application/zip",
                {"Content-Disposition": f'attachment; filename="{name}"'},
            )

        if len(parts) != 4:
            raise BadRequest("bad download path")
        index, fmt = int(parts[2]), parts[3]
        items = job.outcome.items
        if index < 0 or index >= len(items):
            raise FileNotFoundError("no such result index")
        item = items[index]
        if item.result is None:
            raise FileNotFoundError(item.error or "that file produced no result")
        if fmt not in UI_FORMATS:
            raise BadRequest(f"unknown format {fmt!r}")

        settings = job.settings
        text = payloads(
            item.result,
            (fmt,),
            max_chars=settings.max_chars if settings else 84,
            max_cue=settings.max_cue if settings else 6.0,
        )[fmt]
        if limit > 0 and len(text) > limit:
            text = text[:limit] + "\n…"
        name = filename(Path(item.result.media.path).stem, fmt)
        ctype = _MIME.get(f".{fmt}", "text/plain; charset=utf-8")
        disposition = "inline" if limit else "attachment"
        self._send(
            200,
            text.encode("utf-8"),
            ctype,
            {"Content-Disposition": f'{disposition}; filename="{name}"'},
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def serve(host: str = "127.0.0.1", port: int = 8501, open_browser: bool = True) -> None:
    """Start the server, trying a few ports if the first is taken."""
    if not any(models.available_models()):
        print(
            "No speech model on disk yet — the page will offer a one-time download "
            f"into {fetch.model_dir()}"
        )

    httpd: ThreadingHTTPServer | None = None
    for candidate in range(port, port + 12):
        try:
            httpd = ThreadingHTTPServer((host, candidate), Handler)
        except OSError:
            continue
        httpd.daemon_threads = True
        port = candidate
        break
    if httpd is None:
        raise SystemExit(f"could not bind {host}:{port}-{port + 11} — is another copy running?")

    url = f"http://{host}:{port}/"
    print(f"python-whisper UI on {url}")
    print("  local only · Ctrl+C to stop")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local transcription web UI.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: localhost only)"
    )
    parser.add_argument("--port", type=int, default=8501, help="first port to try (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
