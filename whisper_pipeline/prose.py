"""Sentence-level post-processing: turn VAD segments into continuous prose.

VAD boundaries are chosen for decoding, not for reading. A single spoken sentence
arrives as two segments whenever the speaker pauses mid-clause, and one segment
often carries several sentences. Merging ``Segment.text`` strings therefore needs
faked timings; regrouping the timed words does not.

This module flattens every word across every segment and rebuilds sentences from
them, so the ``sentences`` export keeps real start/end times while reading as
prose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import Segment, Word

#: Characters that end a sentence, Latin and CJK.
TERMINAL = (".", "!", "?", "\u2026", "\u3002", "\uff1f", "\uff01")
_CLOSERS = "\"')]\u201d\u2019"
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "fig", "no", "e.g", "i.e", "cf"}
)
_LONE_I = re.compile(r"(?<![\w'\u2019])i(?![\w'\u2019])", re.IGNORECASE)
_I_CLITIC = re.compile(r"(?<![\w'\u2019])i(?=['\u2019](?:m|ve|ll|d|re|s)\b)", re.IGNORECASE)
_TEXT_SPLIT = re.compile(r"(?<=[.!?\u2026\u3002\uff1f\uff01])\s+")


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence assembled from the word stream, in absolute seconds."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _is_cjk(char: str) -> bool:
    return "\u3040" <= char <= "\u9fff" or "\uff00" <= char <= "\uffef"


def _join(pieces: Sequence[str]) -> str:
    """Join words, inserting a space unless a CJK boundary makes that wrong."""
    out = ""
    for piece in pieces:
        if out and not (_is_cjk(out[-1]) or _is_cjk(piece[0])):
            out += " "
        out += piece
    return out


def _closes_sentence(word: Word) -> bool:
    """True when ``word`` carries sentence-final punctuation.

    Casing cannot arbitrate here: Parakeet transcribes in a lower case alphabet,
    so "end. next" is normal even at a real sentence boundary. Only a known
    abbreviation ("Dr.", "e.g.") suppresses the break.
    """
    text = word.text.strip()
    if not text:
        return False
    core = text.rstrip(_CLOSERS)
    if not core.endswith(TERMINAL):
        return False
    stem = core.rstrip("".join(TERMINAL)).strip().lower()
    return stem not in _ABBREVIATIONS


def _capitalise(text: str) -> str:
    """Upper-case the first letter and standalone first-person pronouns."""
    for index, char in enumerate(text):
        if char.isalpha():
            text = f"{text[:index]}{char.upper()}{text[index + 1 :]}"
            break
    text = _LONE_I.sub("I", text)
    return _I_CLITIC.sub("I", text)


def _from_text_only(segments: Sequence[Segment], *, capitalise: bool) -> list[Sentence]:
    """Fallback for backends without word timings: split text, time it evenly."""
    sentences: list[Sentence] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        parts = [part for part in _TEXT_SPLIT.split(text) if part.strip()]
        if not parts:
            continue
        total = sum(len(part) for part in parts)
        cursor = segment.start
        for part in parts:
            share = segment.duration * len(part) / total if total else segment.duration
            body = _capitalise(part.strip()) if capitalise else part.strip()
            sentences.append(
                Sentence(
                    start=round(cursor, 3),
                    end=round(min(segment.end, cursor + share), 3),
                    text=body,
                )
            )
            cursor += share
    return sentences


def build_sentences(
    segments: Sequence[Segment],
    *,
    sentence_gap: float = 1.2,
    max_words: int = 45,
    max_duration: float = 25.0,
    capitalise: bool = True,
    dedupe_seams: bool = True,
) -> list[Sentence]:
    """Regroup timed words into sentences.

    A sentence is closed by terminal punctuation, by a pause longer than
    ``sentence_gap`` seconds, or — as a runaway guard for models that emit no
    punctuation — by ``max_words`` / ``max_duration``.

    Args:
        segments: VAD segments in playback order.
        sentence_gap: Pause length that definitively ends a sentence.
        max_words: Hard ceiling per sentence.
        max_duration: Hard ceiling per sentence, in seconds.
        capitalise: Upper-case sentence starts and first-person ``i``.
        dedupe_seams: Drop a word repeated across a segment boundary, which is a
            common artefact when two neighbouring decodes overlap slightly.
    """
    words: list[tuple[Word, int]] = [
        (word, index)
        for index, segment in enumerate(segments)
        for word in segment.words
        if word.text.strip()
    ]
    if not words:
        return _from_text_only(segments, capitalise=capitalise)

    sentences: list[Sentence] = []
    bucket: list[Word] = []

    def flush() -> None:
        if not bucket:
            return
        text = _join([word.text.strip() for word in bucket])
        sentences.append(
            Sentence(
                start=round(bucket[0].start, 3),
                end=round(bucket[-1].end, 3),
                text=_capitalise(text) if capitalise else text,
                words=tuple(bucket),
            )
        )
        bucket.clear()

    for position, (word, segment_index) in enumerate(words):
        if bucket:
            previous, previous_segment = bucket[-1], words[position - 1][1]
            gap = word.start - previous.end
            repeated = dedupe_seams and (
                segment_index != previous_segment
                and gap < 0.25
                and word.text.strip().lower() == previous.text.strip().lower()
            )
            if repeated:
                continue
            boundary = (
                gap > sentence_gap
                or _closes_sentence(previous)
                or len(bucket) >= max_words
                or previous.end - bucket[0].start > max_duration
            )
            if boundary:
                flush()
        bucket.append(word)

    flush()
    return sentences


def to_prose(
    segments: Sequence[Segment],
    *,
    paragraph_gap: float = 2.0,
    **sentence_kwargs: Any,
) -> str:
    """Render the sentence stream as readable prose.

    Pauses longer than ``paragraph_gap`` become blank lines, matching how people
    read transcripts. Keyword arguments are forwarded to :func:`build_sentences`.
    """
    sentences = build_sentences(segments, **sentence_kwargs)
    if not sentences:
        return ""
    parts: list[str] = []
    previous_end: float | None = None
    for sentence in sentences:
        if previous_end is not None:
            parts.append("\n\n" if sentence.start - previous_end > paragraph_gap else " ")
        parts.append(sentence.text)
        previous_end = sentence.end
    return "".join(parts).strip() + "\n"


__all__ = ["Sentence", "build_sentences", "to_prose"]
