"""Shared value types for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Word:
    """A single timed word, in absolute seconds from the start of the media."""

    text: str
    start: float
    end: float

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text


@dataclass(frozen=True, slots=True)
class Segment:
    """One decoded utterance, in absolute seconds from the start of the media."""

    index: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        if self.words:
            return len(self.words)
        return len(self.text.split())
