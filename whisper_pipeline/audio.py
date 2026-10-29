"""Audio extraction built on PyAV — no external ffmpeg binary required.

PyAV vendors FFmpeg as shared libraries inside its own wheels, so mp4/mkv/webm/
m4a/mp3/flac decode with nothing installed on the host.

Everything is resampled to 16 kHz mono float32, which is the single format every
downstream consumer (sherpa-onnx, whisper.cpp) wants.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BLOCK_SECONDS = 1.0

# PyAV renamed its exception hierarchy across versions; keep both spellings working.
_DECODE_ERRORS: tuple[type[BaseException], ...] = tuple(
    exc for exc in (getattr(av, "FFmpegError", None), OSError) if exc is not None
)


class AudioExtractionError(RuntimeError):
    """Raised when a file cannot be opened or holds no usable audio stream."""


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """Metadata for one audio stream inside a container."""

    index: int
    codec: str
    sample_rate: int
    channels: int
    layout: str
    bit_rate: int | None
    language: str | None
    title: str | None
    duration: float | None
    is_default: bool

    @property
    def label(self) -> str:
        bits = [
            f"track #{self.index}",
            self.codec,
            f"{self.sample_rate} Hz",
            self.layout or f"{self.channels} ch",
        ]
        if self.title:
            bits.insert(0, self.title)
        return " · ".join(bits)

    @property
    def channel_label(self) -> str:
        return self.layout or f"{self.channels} channel(s)"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Everything the UI needs to describe a file before decoding it."""

    path: str
    container_format: str
    duration: float | None
    size_bytes: int
    audio_streams: tuple[AudioStreamInfo, ...]
    video_streams: int
    subtitle_streams: int

    def default_stream_index(self) -> int:
        """Pick the track a user most likely means.

        Prefers the container's default disposition, then the track with the
        most channels, then simply the first one.
        """
        if not self.audio_streams:
            raise AudioExtractionError("file has no audio stream")
        ranked = sorted(
            self.audio_streams,
            key=lambda s: (s.is_default, s.channels, s.bit_rate or 0),
            reverse=True,
        )
        return ranked[0].index

    def stream(self, index: int | None = None) -> AudioStreamInfo:
        if index is None:
            index = self.default_stream_index()
        for stream in self.audio_streams:
            if stream.index == index:
                return stream
        raise AudioExtractionError(
            f"stream {index} is not an audio stream "
            f"(audio tracks: {[s.index for s in self.audio_streams]})"
        )

    @property
    def duration_label(self) -> str:
        return format_duration(self.duration)


def format_duration(seconds: float | None) -> str:
    """Render seconds as ``H:MM:SS`` (or ``--`` when unknown)."""
    if seconds is None or seconds <= 0:
        return "--:--"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _seconds(value: object, time_base: object | None = None) -> float | None:
    """Convert an ffmpeg duration/timestamp in stream time base to seconds."""
    if value is None:
        return None
    try:
        seconds = float(value) * (float(time_base) if time_base is not None else 1.0)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _is_default(stream: av.stream.Stream) -> bool:
    """True when the container marks this stream as the default track."""
    disposition = getattr(stream, "disposition", None)
    if disposition is None:
        return False
    try:
        return bool(int(disposition) & 0x1)  # AV_DISPOSITION_DEFAULT
    except (TypeError, ValueError):
        return False


def probe(path: str | Path) -> MediaInfo:
    """Inspect a container without decoding its audio.

    Costs a few milliseconds, which makes it cheap enough to run on every
    Streamlit rerun and lets the UI warn about missing audio tracks before a
    user waits through a failed decode.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise AudioExtractionError(f"file not found: {resolved}")

    try:
        with av.open(str(resolved)) as container:
            container_duration = container.duration / 1_000_000 if container.duration else None
            audio: list[AudioStreamInfo] = []
            video_count = subtitle_count = 0

            for stream in container.streams:
                if stream.type == "video":
                    video_count += 1
                elif stream.type == "subtitle":
                    subtitle_count += 1
                elif stream.type != "audio":
                    continue
                else:
                    ctx = stream.codec_context
                    audio.append(
                        AudioStreamInfo(
                            index=stream.index,
                            codec=ctx.name or "unknown",
                            sample_rate=int(getattr(ctx, "sample_rate", 0) or 0),
                            channels=int(getattr(ctx, "channels", 0) or 0),
                            layout=_layout_name(stream, int(getattr(ctx, "channels", 0) or 0)),
                            bit_rate=stream.bit_rate,
                            language=(stream.metadata or {}).get("language"),
                            title=(stream.metadata or {}).get("title")
                            or (stream.metadata or {}).get("handler_name"),
                            duration=_seconds(stream.duration, stream.time_base)
                            or container_duration,
                            is_default=_is_default(stream),
                        )
                    )

            return MediaInfo(
                path=str(resolved),
                container_format=container.format.name or "unknown",
                duration=container_duration,
                size_bytes=resolved.stat().st_size,
                audio_streams=tuple(audio),
                video_streams=video_count,
                subtitle_streams=subtitle_count,
            )
    except _DECODE_ERRORS as exc:  # pragma: no cover - depends on input file
        raise AudioExtractionError(f"cannot read {resolved.name}: {exc}") from exc


_COMMON_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


def _layout_name(stream: av.stream.Stream, channels: int) -> str:
    """Human-readable channel layout.

    PyAV renders ``str(layout)`` as ``<av.AudioLayout 'stereo'>`` and
    ``layout.name`` as ``"1 channels"`` for mono streams, so normalise both.
    """
    try:
        name = stream.layout.name or ""
    except (AttributeError, ValueError, TypeError):
        name = ""
    if name and not name.endswith("channels"):
        return name
    return _COMMON_LAYOUTS.get(channels, name or (f"{channels} ch" if channels else ""))


def _select_stream(
    container: av.container.InputContainer, stream_index: int | None
) -> av.stream.Stream:
    """Resolve the audio stream to decode, honouring an explicit request."""
    candidates = [s for s in container.streams if s.type == "audio"]
    if not candidates:
        raise AudioExtractionError("file has no audio stream")
    if stream_index is None:
        default = [s for s in candidates if _is_default(s)]
        return (default or candidates)[0]
    for stream in candidates:
        if stream.index == stream_index:
            return stream
    raise AudioExtractionError(
        f"stream {stream_index} is not an audio stream (available: {[s.index for s in candidates]})"
    )


def iter_blocks(
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    block_seconds: float = BLOCK_SECONDS,
    stream_index: int | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_sample, samples)`` blocks of 16 kHz mono float32 audio.

    Streaming is what keeps memory flat: a 20 s utterance costs 1.3 MB, whereas
    materialising a 27-minute file costs 106 MB. Each yielded block is a private
    copy, so callers may retain it freely.

    ``start_sample`` is the absolute sample index in the source timeline, which
    lets callers reconstruct timestamps without tracking offsets themselves.
    """
    resolved = Path(path)
    block_samples = max(1, int(block_seconds * sample_rate))
    total_seconds: float | None = None
    decoded = 0
    last_report = 0.0
    carry = np.empty(0, dtype=np.float32)

    try:
        container = av.open(str(resolved))
    except _DECODE_ERRORS as exc:
        raise AudioExtractionError(f"cannot open {resolved.name}: {exc}") from exc

    with container:
        stream = _select_stream(container, stream_index)

        span_start = _seconds(stream.duration, stream.time_base)
        if span_start is None and container.duration:
            span_start = container.duration / 1_000_000
        total_seconds = span_start
        if duration_seconds is not None:
            total_seconds = duration_seconds
        elif total_seconds is not None:
            total_seconds = max(0.0, total_seconds - start_seconds)

        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate
        )

        skip_samples = int(start_seconds * sample_rate)
        limit_samples = (
            int(duration_seconds * sample_rate) if duration_seconds is not None else None
        )
        # Timeline position of the first sample we will yield.
        position = skip_samples

        def emit(frames: list[np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
            nonlocal carry, decoded, last_report, position
            if not frames:
                return
            chunk = np.concatenate(frames) if len(frames) > 1 else frames[0]
            if skip_samples > decoded + chunk.size:
                decoded += chunk.size
                return
            if skip_samples > decoded:
                chunk = chunk[skip_samples - decoded :]
                decoded = skip_samples
                position = skip_samples
            if limit_samples is not None and position + chunk.size > skip_samples + limit_samples:
                chunk = chunk[: max(0, skip_samples + limit_samples - position)]
                if chunk.size == 0:
                    return
            carry = np.concatenate([carry, chunk])
            decoded += chunk.size

            while carry.size >= block_samples:
                yield position, carry[:block_samples].copy()
                position += block_samples
                carry = carry[block_samples:]

            if progress and total_seconds:
                elapsed = (position + carry.size) / sample_rate
                if elapsed - last_report >= 0.25:
                    last_report = elapsed
                    progress(min(1.0, elapsed / total_seconds))

        if start_seconds > 0:
            try:
                container.seek(int(start_seconds * 1_000_000), backward=True, stream=stream)
            except Exception:  # noqa: BLE001 - seeking is an optimisation, not a requirement
                log.debug("seek unsupported on %s; decoding from the start", resolved.name)

        try:
            for frame in container.decode(stream):
                frames = [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
                yield from emit(frames)
                if limit_samples is not None and position >= skip_samples + limit_samples:
                    break
        except _DECODE_ERRORS as exc:  # pragma: no cover - depends on input file
            raise AudioExtractionError(f"decode failed on {resolved.name}: {exc}") from exc

        tail = [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
        yield from emit(tail)

    if carry.size:
        yield position, carry.copy()
    if progress:
        progress(1.0)


def decode_audio(
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    stream_index: int | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> np.ndarray:
    """Decode a whole file into one float32 array (convenience for short clips)."""
    blocks = [
        block
        for _, block in iter_blocks(
            path,
            sample_rate=sample_rate,
            stream_index=stream_index,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
    ]
    if not blocks:
        raise AudioExtractionError("no audio samples decoded")
    return np.concatenate(blocks).astype(np.float32, copy=False)
