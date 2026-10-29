"""Subtitle and transcript serialisation.

SRT/VTT cues are built from word timings rather than whole VAD segments, so a
30-second utterance becomes readable 1-2 line cues instead of a wall of text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .types import Segment, Word

_SENTENCE_END = (".", "!", "?", "\u3002", "\uff1f", "\uff01")


@dataclass(frozen=True, slots=True)
class Cue:
    """One subtitle entry."""

    start: float
    end: float
    text: str


def format_timestamp(seconds: float, *, vtt: bool = False) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (WebVTT)."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _ends_sentence(word: Word) -> bool:
    text = word.text.strip()
    return bool(text) and text.endswith(_SENTENCE_END)


def build_cues(
    segments: Sequence[Segment],
    *,
    max_chars: int = 84,
    max_duration: float = 6.0,
    sentence_gap: float = 0.6,
) -> list[Cue]:
    """Greedily pack words into subtitle cues.

    A cue is closed when adding the next word would exceed ``max_chars``, exceed
    ``max_duration``, follow a pause longer than ``sentence_gap``, or start a new
    sentence after the previous word ended one.
    """
    cues: list[Cue] = []
    for segment in segments:
        if not segment.words:
            cues.extend(_split_without_timings(segment, max_chars, max_duration))
            continue

        bucket: list[Word] = []
        for word in segment.words:
            if bucket:
                used = sum(len(w.text) + 1 for w in bucket) - 1
                too_long = used + 1 + len(word.text) > max_chars
                too_slow = bucket[-1].end - bucket[0].start > max_duration
                pause = word.start - bucket[-1].end > sentence_gap
                if too_long or too_slow or pause or _ends_sentence(bucket[-1]):
                    cues.append(_as_cue(bucket, segment))
                    bucket = []
            bucket.append(word)
        if bucket:
            cues.append(_as_cue(bucket, segment))
    return cues


def _as_cue(words: Sequence[Word], segment: Segment) -> Cue:
    text = " ".join(w.text for w in words).strip()
    start = words[0].start if words else segment.start
    end = words[-1].end if words else segment.end
    return Cue(start=start, end=max(start + 0.2, end), text=text)


def _split_without_timings(segment: Segment, max_chars: int, max_duration: float) -> Iterator[Cue]:
    """Fallback cue split for backends that emit no word timings.

    The text is distributed proportionally across the segment's duration, which
    is approximate but keeps subtitles aligned to the right neighbourhood.
    """
    words = segment.text.split()
    if not words:
        return
    span = max(segment.duration, 0.2)
    per_word = span / len(words)
    bucket: list[str] = []
    bucket_start_index = 0

    def emit(start_index: int, count: int) -> Cue:
        start = segment.start + start_index * per_word
        end = min(segment.end, start + max(count * per_word, 0.2))
        return Cue(start=start, end=end, text=" ".join(bucket))

    for index, word in enumerate(words):
        used = sum(len(w) + 1 for w in bucket) - 1 if bucket else 0
        longest = (index - bucket_start_index + 1) * per_word
        if bucket and (used + 1 + len(word) > max_chars or longest > max_duration):
            yield emit(bucket_start_index, len(bucket))
            bucket, bucket_start_index = [], index
        bucket.append(word)
    if bucket:
        yield emit(bucket_start_index, len(bucket))


def to_srt(
    segments: Sequence[Segment],
    *,
    max_chars: int = 84,
    max_duration: float = 6.0,
) -> str:
    """Render SRT subtitles."""
    cues = build_cues(segments, max_chars=max_chars, max_duration=max_duration)
    blocks = [
        f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(
    segments: Sequence[Segment],
    *,
    max_chars: int = 84,
    max_duration: float = 6.0,
) -> str:
    """Render WebVTT subtitles."""
    cues = build_cues(segments, max_chars=max_chars, max_duration=max_duration)
    blocks = [
        f"{format_timestamp(cue.start, vtt=True)} --> "
        f"{format_timestamp(cue.end, vtt=True)}\n{cue.text}"
        for cue in cues
    ]
    body = "\n\n".join(blocks)
    return f"WEBVTT\n\n{body}\n" if blocks else "WEBVTT\n"


def to_txt(
    segments: Sequence[Segment],
    *,
    timestamps: bool = False,
    paragraph_gap: float = 2.0,
    strip_paragraphs: bool = True,
) -> str:
    """Render a plain-text transcript.

    Segment breaks longer than ``paragraph_gap`` become blank lines, which mirrors
    how people actually read transcripts. Set ``strip_paragraphs`` to keep one
    line per segment instead.
    """
    lines: list[str] = []
    previous_end: float | None = None
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if timestamps:
            stamp = format_timestamp(segment.start).replace(",", ".")
            text = f"[{stamp}] {text}"
        if (
            previous_end is not None
            and not strip_paragraphs
            and segment.start - previous_end > paragraph_gap
        ):
            lines.append("")
        lines.append(text)
        previous_end = segment.end
    return "\n".join(lines) + ("\n" if lines else "")


def to_json(
    segments: Sequence[Segment],
    *,
    metadata: dict[str, object] | None = None,
    indent: int = 2,
) -> str:
    """Render a machine-readable transcript with word-level timings."""
    payload: dict[str, object] = dict(metadata or {})
    payload["segments"] = [
        {
            "index": segment.index,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text,
            "words": [
                {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in segment.words
            ],
        }
        for segment in segments
    ]
    payload["text"] = "\n".join(s.text for s in segments if s.text).strip()
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def guess_extension_format(name: str) -> str:
    """Pick an export format from a file name."""
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
    return suffix if suffix in {"txt", "srt", "vtt", "json"} else "txt"


def iter_words(segments: Iterable[Segment]) -> Iterator[Word]:
    """Flatten every word across segments."""
    for segment in segments:
        yield from segment.words
