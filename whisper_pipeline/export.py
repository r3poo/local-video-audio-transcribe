"""Serialise transcription results to files or a zip bundle.

One place decides what a format is called and what it contains, so the CLI, the
batch runner and the Streamlit app cannot drift apart. ``txt``/``srt``/``vtt``/
``json`` keep the names they have always had; only ``sentences`` adds a suffix.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from .pipeline import TranscriptionResult

FORMATS: tuple[str, ...] = ("txt", "sentences", "srt", "vtt", "json")
DEFAULT_FORMATS: tuple[str, ...] = ("txt", "srt")

_SENTENCE_SUFFIX = "_transcript_sentences.txt"


def filename(stem: str, fmt: str) -> str:
    """File name for one format, given a media file stem."""
    if fmt == "sentences":
        return f"{stem}{_SENTENCE_SUFFIX}"
    return f"{stem}.{fmt}"


def payloads(
    result: TranscriptionResult,
    formats: Sequence[str],
    *,
    max_chars: int = 84,
    max_cue: float = 6.0,
) -> dict[str, str]:
    """Rendered text for each requested format, in the requested order."""
    renderers = {
        "txt": result.to_txt,
        "sentences": lambda: result.prose,
        "srt": lambda: result.to_srt(max_chars=max_chars, max_duration=max_cue),
        "vtt": lambda: result.to_vtt(max_chars=max_chars, max_duration=max_cue),
        "json": result.to_json,
    }
    unknown = [fmt for fmt in formats if fmt not in renderers]
    if unknown:
        raise ValueError(f"unknown format(s): {', '.join(unknown)} (choose from {FORMATS})")
    return {fmt: renderers[fmt]() for fmt in formats}


def output_paths(
    result: TranscriptionResult, outdir: str | Path, formats: Sequence[str]
) -> list[Path]:
    """Where :func:`write_outputs` would place the files for ``result``."""
    return paths_for_stem(Path(result.media.path).stem, outdir, formats)


def paths_for_stem(stem: str, outdir: str | Path, formats: Sequence[str]) -> list[Path]:
    """Same as :func:`output_paths` without needing a finished transcription."""
    return [Path(outdir) / filename(stem, fmt) for fmt in formats]


def write_outputs(
    result: TranscriptionResult,
    outdir: str | Path,
    formats: Sequence[str],
    max_chars: int = 84,
    max_cue: float = 6.0,
) -> list[Path]:
    """Write every requested format into ``outdir`` and return the paths."""
    target_dir = Path(outdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt, payload in payloads(result, formats, max_chars=max_chars, max_cue=max_cue).items():
        target = target_dir / filename(Path(result.media.path).stem, fmt)
        target.write_text(payload, encoding="utf-8")
        written.append(target)
    return written


def existing_outputs(
    stem: str, outdir: str | Path, formats: Sequence[str]
) -> tuple[list[Path], bool]:
    """Paths for ``stem`` and whether *every* one already exists non-empty."""
    targets = paths_for_stem(stem, outdir, formats)
    complete = all(target.is_file() and target.stat().st_size > 0 for target in targets)
    return targets, complete


def _unique(name: str, taken: set[str]) -> str:
    """``talk.txt`` -> ``talk (2).txt`` when two inputs share a stem."""
    if name not in taken:
        return name
    stem, suffix = name[: -len(".txt")], ".txt"
    if name.endswith(".json"):
        stem, suffix = name[: -len(".json")], ".json"
    elif name.endswith(".srt"):
        stem, suffix = name[: -len(".srt")], ".srt"
    elif name.endswith(".vtt"):
        stem, suffix = name[: -len(".vtt")], ".vtt"
    counter = 2
    while f"{stem} ({counter}){suffix}" in taken:
        counter += 1
    return f"{stem} ({counter}){suffix}"


def bundle_zip(
    results: Iterable[TranscriptionResult],
    formats: Sequence[str],
    *,
    max_chars: int = 84,
    max_cue: float = 6.0,
) -> bytes:
    """Every format for every result, flat, in one in-memory archive."""
    buffer = io.BytesIO()
    taken: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            stem = Path(result.media.path).stem
            for fmt, payload in payloads(
                result, formats, max_chars=max_chars, max_cue=max_cue
            ).items():
                name = _unique(filename(stem, fmt), taken)
                taken.add(name)
                archive.writestr(name, payload)
    return buffer.getvalue()
