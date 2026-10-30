"""End-to-end orchestration: media file in, timed segments out.

The whole loop is streaming: one block at a time is decoded, pushed through the
segmenter, and recognised as soon as a chunk closes. Memory therefore stays flat
(≈1.3 MB for a 20 s utterance) no matter how long the input is.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import audio as audio_io
from .audio import AudioStreamInfo, MediaInfo
from .engine import FixedWindowSegmenter, Transcriber, VadOptions, VadSegmenter
from .models import ModelSpec
from .prose import Sentence, build_sentences, to_prose
from .subtitles import to_json, to_srt, to_txt, to_vtt
from .types import Segment

ProgressFn = Callable[[float, str], None]
SegmentFn = Callable[[Segment], None]


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Everything produced by one transcription run."""

    media: MediaInfo
    stream: AudioStreamInfo
    segments: tuple[Segment, ...]
    model_label: str
    provider: str
    device_note: str | None
    model_load_seconds: float
    inference_seconds: float
    wall_seconds: float
    audio_seconds: float
    vad_enabled: bool

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text).strip()

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def word_count(self) -> int:
        return sum(segment.word_count for segment in self.segments)

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        """Utterances regrouped into sentences that ignore VAD boundaries."""
        return tuple(build_sentences(self.segments))

    @property
    def prose(self) -> str:
        """Sentence-level transcript: the VAD cuts merged back into prose."""
        return to_prose(self.segments)

    @property
    def rtf(self) -> float:
        """Real-time factor: processing seconds per second of audio."""
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else 0.0

    def metadata(self) -> dict[str, object]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": self.media.path,
            "container": self.media.container_format,
            "duration_seconds": round(self.audio_seconds, 2),
            "audio_codec": self.stream.codec,
            "source_sample_rate": self.stream.sample_rate,
            "source_channels": self.stream.channels,
            "stream_index": self.stream.index,
            "model": self.model_label,
            "provider": self.provider,
            "vad": self.vad_enabled,
            "model_load_seconds": round(self.model_load_seconds, 3),
            "inference_seconds": round(self.inference_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "real_time_factor": round(self.rtf, 4),
        }

    def to_srt(self, *, max_chars: int = 84, max_duration: float = 6.0) -> str:
        return to_srt(self.segments, max_chars=max_chars, max_duration=max_duration)

    def to_vtt(self, *, max_chars: int = 84, max_duration: float = 6.0) -> str:
        return to_vtt(self.segments, max_chars=max_chars, max_duration=max_duration)

    def to_txt(self, *, timestamps: bool = False) -> str:
        return to_txt(self.segments, timestamps=timestamps)

    def to_json(self) -> str:
        return to_json(self.segments, metadata=self.metadata())


def _to_segment(index: int, chunk, utterance) -> Segment:
    return Segment(
        index=index,
        start=round(chunk.start, 3),
        end=round(chunk.end, 3),
        text=utterance.text,
        words=utterance.words,
    )


def transcribe_file(
    path: str | Path,
    transcriber: Transcriber,
    *,
    vad: VadOptions | None = None,
    window_seconds: float = 30.0,
    stream_index: int | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    on_progress: ProgressFn | None = None,
    on_segment: SegmentFn | None = None,
) -> TranscriptionResult:
    """Transcribe a media file.

    Args:
        path: Any container PyAV can decode (mp4, mkv, webm, m4a, mp3, wav…).
        transcriber: A loaded :class:`~whisper_pipeline.engine.Transcriber`.
        vad: VAD settings, or ``None`` to disable VAD and use fixed windows.
        window_seconds: Window length used when ``vad`` is ``None``.
        stream_index: Which audio track to decode; ``None`` picks the default.
        start_seconds: Skip into the media before transcribing.
        duration_seconds: Stop after this much audio.
        on_progress: ``(fraction, message)`` callback for UI feedback.
        on_segment: Called as each segment is recognised.
    """
    media = audio_io.probe(path)
    stream = media.stream(stream_index)

    segmenter = VadSegmenter(vad) if vad is not None else FixedWindowSegmenter(window_seconds)

    def report(fraction: float, message: str) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, fraction)), message)

    segments: list[Segment] = []
    inference_seconds = 0.0
    audio_seconds = duration_seconds or (max(0.0, (stream.duration or 0.0) - start_seconds) or None)
    started = time.perf_counter()

    def handle(chunks: list) -> None:
        nonlocal inference_seconds
        for chunk in chunks:
            utterance = transcriber.transcribe(chunk.samples, start=chunk.start, end=chunk.end)
            inference_seconds += utterance.seconds
            if not utterance.text.strip():
                continue
            segment = _to_segment(len(segments) + 1, chunk, utterance)
            segments.append(segment)
            if on_segment:
                on_segment(segment)

    def decode_progress(fraction: float) -> None:
        span = audio_seconds or 0.0
        decoded = fraction * span
        report(
            fraction * 0.98,
            f"Decoding and recognising · {audio_io.format_duration(decoded)}"
            + (f" / {audio_io.format_duration(span)}" if span else ""),
        )

    report(0.0, "Decoding audio…")
    for offset, block in audio_io.iter_blocks(
        path,
        stream_index=stream.index,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        progress=decode_progress,
    ):
        handle(segmenter.feed(block, offset))

    report(0.98, "Flushing final utterance…")
    handle(segmenter.finish())

    wall = time.perf_counter() - started
    if audio_seconds is None:
        audio_seconds = segments[-1].end - start_seconds if segments else 0.0

    report(1.0, "Done")
    return TranscriptionResult(
        media=media,
        stream=stream,
        segments=tuple(segments),
        model_label=transcriber.label,
        provider=transcriber.provider,
        device_note=transcriber.fallback_note,
        model_load_seconds=transcriber.load_seconds,
        inference_seconds=inference_seconds,
        wall_seconds=wall,
        audio_seconds=float(audio_seconds or 0.0),
        vad_enabled=vad is not None,
    )


def transcribe_array(
    samples: np.ndarray,
    transcriber: Transcriber,
    *,
    vad: VadOptions | None = None,
    sample_rate: int = audio_io.SAMPLE_RATE,
) -> list[Segment]:
    """Transcribe an in-memory waveform (useful for tests and notebooks)."""
    segmenter = VadSegmenter(vad) if vad is not None else FixedWindowSegmenter(30.0, sample_rate)
    segments: list[Segment] = []
    for chunk in segmenter.feed(samples.astype(np.float32, copy=False), 0):
        utterance = transcriber.transcribe(chunk.samples, start=chunk.start, end=chunk.end)
        if utterance.text.strip():
            segments.append(_to_segment(len(segments) + 1, chunk, utterance))
    for chunk in segmenter.finish():
        utterance = transcriber.transcribe(chunk.samples, start=chunk.start, end=chunk.end)
        if utterance.text.strip():
            segments.append(_to_segment(len(segments) + 1, chunk, utterance))
    return segments


__all__ = ["TranscriptionResult", "transcribe_array", "transcribe_file", "ModelSpec"]
