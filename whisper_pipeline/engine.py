"""Speech recognition engine: VAD segmentation plus sherpa-onnx decoding.

Two segmenters are provided:

* :class:`VadSegmenter` — Silero VAD cuts the stream on silence, so every decoded
  utterance is a complete phrase and timestamps land on speech boundaries.
* :class:`FixedWindowSegmenter` — blind fixed-length windows, used when VAD is
  disabled. Cheaper, but it slices words in half at window edges.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import ort_runtime
from .models import (
    FAMILY_NEMO_CTC,
    FAMILY_PARAFORMER,
    FAMILY_TRANSDUCER,
    FAMILY_WHISPER,
    ModelSpec,
    available_providers,
    vad_model_path,
)
from .types import Word

# On Windows, ``onnxruntime.dll`` is resolved by base name and the Windows ML copy
# in System32 outranks the sherpa-onnx package directory, so the bundled runtime
# has to be mapped first.  Nothing may import sherpa_onnx before this call.
ort_runtime.prepare()

import sherpa_onnx as so  # noqa: E402

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_VAD_WINDOW = 512  # Silero VAD requires 512-sample windows at 16 kHz


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be instantiated."""


@dataclass(frozen=True, slots=True)
class VadOptions:
    """Tuning for silero-vad voice activity detection."""

    model_path: Path
    threshold: float = 0.5
    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.5
    max_speech_duration: float = 20.0

    @classmethod
    def default(cls) -> VadOptions:
        return cls(model_path=vad_model_path())

    def validate(self) -> None:
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"VAD model missing: {self.model_path} — run scripts/download_models.py"
            )


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    """A contiguous slice of speech located at an absolute position."""

    samples: np.ndarray
    start_sample: int
    sample_rate: int = SAMPLE_RATE

    @property
    def start(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end(self) -> float:
        return (self.start_sample + self.samples.size) / self.sample_rate


@dataclass(frozen=True, slots=True)
class Utterance:
    """Recogniser output for a single chunk, with words in absolute seconds."""

    text: str
    words: tuple[Word, ...]
    seconds: float


class Segmenter(Protocol):
    """Streaming splitter of audio blocks into recognisable chunks."""

    def feed(self, block: np.ndarray, offset: int) -> list[SpeechChunk]: ...

    def finish(self) -> list[SpeechChunk]: ...


class VadSegmenter:
    """Silero VAD driven segmentation over a stream of audio blocks."""

    def __init__(self, options: VadOptions, sample_rate: int = SAMPLE_RATE) -> None:
        options.validate()
        self.options = options
        self.sample_rate = sample_rate
        self._carry = np.empty(0, dtype=np.float32)
        self._first_offset: int | None = None

        silero = so.SileroVadModelConfig()
        silero.model = str(options.model_path)
        silero.threshold = float(options.threshold)
        silero.min_speech_duration = float(options.min_speech_duration)
        silero.min_silence_duration = float(options.min_silence_duration)
        silero.max_speech_duration = float(options.max_speech_duration)
        silero.window_size = _VAD_WINDOW

        config = so.VadModelConfig()
        config.silero_vad = silero
        config.sample_rate = sample_rate
        config.num_threads = 1
        config.provider = "cpu"  # a 0.6 MB model gains nothing from a GPU
        self._vad = so.VoiceActivityDetector(config, buffer_size_in_seconds=60)

    def feed(self, block: np.ndarray, offset: int) -> list[SpeechChunk]:
        if self._first_offset is None:
            self._first_offset = offset
        return self._push(block)

    def finish(self) -> list[SpeechChunk]:
        # The final utterance is still open inside the VAD; feed silence so it
        # closes naturally, then flush whatever is still buffered.
        silence = np.zeros(
            int((self.options.min_silence_duration + 1.0) * self.sample_rate),
            dtype=np.float32,
        )
        chunks = self._push(silence)
        try:
            self._vad.flush()
        except Exception:  # noqa: BLE001 - flush is best-effort across versions
            log.debug("VAD flush unavailable", exc_info=True)
        chunks.extend(self._drain())
        return chunks

    def _push(self, samples: np.ndarray) -> list[SpeechChunk]:
        buffer = np.concatenate([self._carry, samples]) if self._carry.size else samples
        whole = (buffer.size // _VAD_WINDOW) * _VAD_WINDOW
        if whole:
            for start in range(0, whole, _VAD_WINDOW):
                self._vad.accept_waveform(buffer[start : start + _VAD_WINDOW])
            self._carry = buffer[whole:].copy()
        else:
            self._carry = buffer.copy()
        return self._drain()

    def _drain(self) -> list[SpeechChunk]:
        chunks: list[SpeechChunk] = []
        base = self._first_offset or 0
        while not self._vad.empty():
            segment = self._vad.front
            samples = np.asarray(segment.samples, dtype=np.float32)
            start = int(segment.start)
            # ``start`` counts samples fed since construction; the stream may
            # begin at a non-zero offset when the caller seeks into a file.
            if start < 0 or samples.size == 0:
                log.debug("dropping degenerate VAD segment (start=%s, n=%s)", start, samples.size)
            else:
                chunks.append(SpeechChunk(samples=samples, start_sample=base + start))
            self._vad.pop()
        return chunks


class FixedWindowSegmenter:
    """Blind fixed-length windowing, for when VAD is switched off."""

    def __init__(self, window_seconds: float = 30.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.window = max(1, int(window_seconds * sample_rate))
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._started = False

    def feed(self, block: np.ndarray, offset: int) -> list[SpeechChunk]:
        if not self._started:
            self._buffer_start = offset
            self._started = True
        self._buffer = np.concatenate([self._buffer, block]) if self._buffer.size else block.copy()
        return self._pop_ready()

    def finish(self) -> list[SpeechChunk]:
        chunks: list[SpeechChunk] = []
        if self._buffer.size > self.sample_rate // 10:
            chunks.append(SpeechChunk(self._buffer.copy(), self._buffer_start, self.sample_rate))
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start += chunks[0].samples.size if chunks else 0
        return chunks

    def _pop_ready(self) -> list[SpeechChunk]:
        chunks: list[SpeechChunk] = []
        while self._buffer.size >= self.window:
            chunks.append(
                SpeechChunk(
                    self._buffer[: self.window].copy(), self._buffer_start, self.sample_rate
                )
            )
            self._buffer_start += self.window
            self._buffer = self._buffer[self.window :]
        return chunks


_BPE_MARKS = ("\u2581", "\u0120")  # sentencepiece "▁" and GPT-2 "Ġ"


def word_timings(
    tokens: list[str],
    timestamps: list[float],
    start: float,
    end: float,
) -> tuple[Word, ...]:
    """Regroup sub-word tokens into timed words.

    sherpa-onnx emits sub-word tokens (``" R"``, ``"ul"``, ``"es"``) with a start
    time each. Word boundaries are the tokens carrying a ``▁``/``Ġ`` marker; the
    end of a word is the start of the next one, which is close enough for
    subtitles.
    """
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return ()

    pieces: list[tuple[str, float]] = []
    current: list[str] = []
    current_start: float | None = None

    for token, stamp in zip(tokens, timestamps, strict=True):
        text = token
        for mark in _BPE_MARKS:
            text = text.replace(mark, " ")
        begins_word = text.startswith(" ") and bool(text.strip())
        if begins_word and current:
            pieces.append(
                ("".join(current).strip(), current_start if current_start is not None else stamp)
            )
            current, current_start = [], None
        if current_start is None:
            current_start = float(stamp)
        current.append(text)

    if current:
        pieces.append(
            ("".join(current).strip(), current_start if current_start is not None else start)
        )

    words: list[Word] = []
    for position, (text, relative_start) in enumerate(pieces):
        if not text:
            continue
        relative_end = pieces[position + 1][1] if position + 1 < len(pieces) else (end - start)
        words.append(
            Word(
                text=text,
                start=round(start + relative_start, 3),
                end=round(start + max(relative_start, relative_end), 3),
            )
        )
    return tuple(words)


def _create_recognizer(
    spec: ModelSpec, *, provider: str, num_threads: int, debug: bool = False
) -> so.OfflineRecognizer:
    """Instantiate a recogniser for ``spec`` on the requested provider."""
    missing = spec.missing_files()
    if missing:
        raise ModelLoadError(
            f"model {spec.key!r} is incomplete — missing {missing} in {spec.directory}"
        )

    common = {
        "tokens": str(spec.tokens),
        "num_threads": num_threads,
        "provider": provider,
        "debug": debug,
    }

    if spec.family == FAMILY_TRANSDUCER:
        kwargs = dict(
            encoder=str(spec.encoder),
            decoder=str(spec.decoder),
            joiner=str(spec.joiner),
            **common,
        )
        if spec.model_type:
            kwargs["model_type"] = spec.model_type
        return so.OfflineRecognizer.from_transducer(**kwargs)

    if spec.family == FAMILY_WHISPER:
        return so.OfflineRecognizer.from_whisper(
            encoder=str(spec.encoder),
            decoder=str(spec.decoder),
            language=spec.language,
            task=spec.task,
            **common,
        )

    if spec.family == FAMILY_PARAFORMER:
        return so.OfflineRecognizer.from_paraformer(
            paraformer=str(spec.encoder),
            **common,
        )

    if spec.family == FAMILY_NEMO_CTC:
        return so.OfflineRecognizer.from_nemo_ctc(
            model=str(spec.encoder),
            **common,
        )

    raise ModelLoadError(f"unsupported model family {spec.family!r}")


def resolve_threads(value: int | None) -> int:
    """Map ``None``/``0`` to the machine's core count, otherwise clamp to >= 1."""
    if value is None or value <= 0:
        return max(1, os.cpu_count() or 1)
    return int(value)


class Transcriber:
    """A loaded model plus the locking needed to decode safely from a web app."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str = "cpu",
        num_threads: int | None = None,
        debug: bool = False,
    ) -> None:
        self.spec = spec
        self.device = device
        self.num_threads = resolve_threads(num_threads)
        self.debug = debug
        self._lock = threading.Lock()
        self.recognizer, self.provider, self.fallback_note, self.load_seconds = self._load()

    def _load(self) -> tuple[so.OfflineRecognizer, str, str | None, float]:
        """Load the model, degrading gracefully from GPU to CPU when needed."""
        supported = available_providers()
        requested = self.device
        wants_gpu = requested in ("cuda", "gpu", "tensorrt", "cuda:0")
        providers: list[str] = []

        if wants_gpu:
            gpu_provider = "cuda" if "cuda" in supported else requested
            if gpu_provider in supported:
                providers.append(gpu_provider)
            providers.append("cpu")
        else:
            providers.append("cpu")

        errors: list[str] = []
        fallback_note: str | None = None
        for index, provider in enumerate(providers):
            started = time.perf_counter()
            try:
                recognizer = _create_recognizer(
                    self.spec,
                    provider=provider,
                    num_threads=self.num_threads,
                    debug=self.debug,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
                errors.append(f"{provider}: {exc}")
                log.warning("provider %s failed for %s: %s", provider, self.spec.key, exc)
                continue
            elapsed = time.perf_counter() - started
            if wants_gpu and provider == "cpu":
                if index == 0:
                    reason = (
                        f"this sherpa-onnx build only offers {', '.join(supported)} "
                        f"execution providers"
                    )
                else:
                    reason = "the CUDA provider failed to initialise"
                fallback_note = (
                    f"GPU was requested but {reason}; running on CPU instead. "
                    f"See README 'GPU setup'."
                )
            return recognizer, provider, fallback_note, elapsed

        joined = " | ".join(errors)
        if "API version" in joined:
            joined += " — " + ort_runtime.mismatch_hint()
        raise ModelLoadError(f"could not load {self.spec.label}: " + joined)

    @property
    def label(self) -> str:
        return self.spec.label

    def transcribe(self, samples: np.ndarray, *, start: float, end: float) -> Utterance:
        """Decode one chunk of 16 kHz mono float32 audio."""
        if samples.size == 0:
            return Utterance("", (), 0.0)
        with self._lock:
            started = time.perf_counter()
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self.recognizer.decode_stream(stream)
            result = stream.result
            elapsed = time.perf_counter() - started

        text = (getattr(result, "text", "") or "").strip()
        tokens = list(getattr(result, "tokens", []) or [])
        stamps = [float(t) for t in (getattr(result, "timestamps", []) or [])]
        return Utterance(
            text=text,
            words=word_timings(tokens, stamps, start, end),
            seconds=elapsed,
        )
