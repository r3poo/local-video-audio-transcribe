"""Batch orchestration: many files in, one queue of outcomes out.

Batch is deliberately sequential. One ``Transcriber`` holds the model (up to
764 MB resident) and sherpa-onnx already saturates the intra-op threads, so
running two files at once buys throughput only on machines that cannot feed the
CPU — while doubling memory and making progress reporting lie.

The runner owns three things the single-file path does not need: expanding
directory arguments, resuming (``skip_existing``) and isolating failures so one
corrupt file cannot end a 200-file queue.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .audio import AudioExtractionError
from .engine import ModelLoadError, Transcriber, VadOptions
from .export import DEFAULT_FORMATS, existing_outputs, payloads, write_outputs
from .pipeline import TranscriptionResult, transcribe_file

#: Container extensions the walker accepts when an argument is a directory.
#: Anything PyAV can decode is fine; this list only decides what a directory
#: scan picks up, so it stays close to what people actually keep on disk.
MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".avi",
        ".flac",
        ".m2ts",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".oga",
        ".ogg",
        ".ogv",
        ".opus",
        ".ts",
        ".wav",
        ".webm",
        ".wma",
        ".wmv",
    }
)

ProgressFn = Callable[[float, str], None]
ItemFn = Callable[["BatchItem"], None]
StartFn = Callable[["BatchItem", int, int], None]


@dataclass(slots=True)
class BatchItem:
    """One queued file and what happened to it."""

    path: Path
    status: str = "pending"  # "ok" | "skipped" | "failed"
    result: TranscriptionResult | None = None
    outputs: tuple[Path, ...] = ()
    error: str | None = None
    seconds: float = 0.0

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        """One-line verdict for tables and logs."""
        if self.status == "ok" and self.result is not None:
            return (
                f"ok · {self.result.segment_count} segments, "
                f"{self.result.word_count} words, {self.result.media.duration_label}"
            )
        if self.status == "skipped":
            return "skipped · outputs already on disk"
        return f"failed · {self.error or 'unknown error'}"


@dataclass(slots=True)
class BatchOutcome:
    """Everything a batch produced, including the paths it refused to queue."""

    items: list[BatchItem] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> list[BatchItem]:
        return [item for item in self.items if item.status == "ok"]

    @property
    def skipped(self) -> list[BatchItem]:
        return [item for item in self.items if item.status == "skipped"]

    @property
    def failed(self) -> list[BatchItem]:
        return [item for item in self.items if item.status == "failed"]

    @property
    def audio_seconds(self) -> float:
        return sum(item.result.audio_seconds for item in self.ok if item.result)

    @property
    def wall_seconds(self) -> float:
        return sum(item.seconds for item in self.items)

    @property
    def rtf(self) -> float:
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def word_count(self) -> int:
        return sum(item.result.word_count for item in self.ok if item.result)

    def summary(self) -> str:
        """Counts plus throughput, in one line suitable for a terminal."""
        counts = f"{len(self.ok)} ok · {len(self.skipped)} skipped · {len(self.failed)} failed"
        if not self.ok:
            return counts
        return (
            f"{counts} · {self.audio_seconds / 60:.1f} min audio in "
            f"{self.wall_seconds:.0f} s (RTF {self.rtf:.2f})"
        )


def _is_hidden(relative: Path) -> bool:
    """Skip dot-files and macOS ``._`` shadows picked up by a directory walk."""
    return any(part.startswith(".") for part in relative.parts)


#: Opening quote -> matching closer, for the styles that reach a paste box.
_QUOTES = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}


def clean_path(raw: str) -> str:
    """Strip whitespace and surrounding quotes from one pasted path.

    ``"C:\\Videos\\My clip.mp4"``, ``'talk.mp4'`` and the curly variants that
    office software produces all resolve to the bare path. Nested pairs
    (``"'x'"``) are unwrapped too; a lopsided quote is left alone, because
    guessing which side was meant is worse than reporting the file as missing.
    """
    text = raw.strip().lstrip("\ufeff").strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        if _QUOTES.get(text[0]) == text[-1] and text[1:-1].strip():
            text = text[1:-1].strip()
            changed = True
    return text


def _tokenize(line: str) -> list[str]:
    """Split one line into paths, keeping quoted runs (and their spaces) whole."""
    tokens: list[str] = []
    buffer: list[str] = []
    closing: str | None = None
    for char in line:
        if closing is not None:
            if char == closing:
                closing = None
            else:
                buffer.append(char)
        elif char in _QUOTES and not buffer:
            closing = _QUOTES[char]
        elif char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
        else:
            buffer.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return [token for token in tokens if token.strip()]


def parse_path_text(text: str) -> list[str]:
    """Extract paths from what a user pasted into a multi-line box.

    One path per line, but each line may also be quoted — including several
    quoted paths on one line, which is exactly what Windows Explorer's *Copy as
    path* produces for a multi-selection. When a line contains no quotes it is
    taken whole, so an unquoted path with spaces still works. Ambiguity is
    settled by the filesystem: if the tokens exist, they are separate files;
    otherwise the line is treated as one path and the caller reports it missing.
    """
    entries: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff").strip()
        if not line:
            continue
        tokens = _tokenize(line)
        if len(tokens) <= 1:
            entries.extend(clean_path(token) for token in tokens or [line])
        elif any(Path(clean_path(token)).expanduser().exists() for token in tokens):
            entries.extend(clean_path(token) for token in tokens)
        elif Path(clean_path(line)).expanduser().exists():
            entries.append(clean_path(line))
        else:
            entries.extend(clean_path(token) for token in tokens)
    return [entry for entry in entries if entry]


def expand_inputs(paths: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """Resolve files and directories into an ordered, de-duplicated file list.

    Directories are walked recursively and filtered to :data:`MEDIA_EXTENSIONS`;
    files are taken as given (an unusual container still works when named
    explicitly). Order is preserved, so ``batch/ *.mkv`` keeps the shell's order,
    and a file named twice — or reachable through two directories — is queued
    once.

    Returns the files plus a list of human-readable problems for arguments that
    do not exist.
    """
    files: list[Path] = []
    problems: list[str] = []
    seen: set[Path] = set()

    def add(candidate: Path) -> None:
        key = candidate.resolve()
        if key not in seen:
            seen.add(key)
            files.append(key)

    for raw in paths:
        text = clean_path(str(raw))
        if not text:
            continue
        candidate = Path(text).expanduser()
        if candidate.is_dir():
            for child in sorted(candidate.rglob("*")):
                if not child.is_file() or child.suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                if _is_hidden(child.relative_to(candidate)):
                    continue
                add(child)
        elif candidate.is_file():
            add(candidate)
        else:
            problems.append(f"no such file or directory: {candidate}")

    return files, problems


def run_batch(
    inputs: Sequence[str | Path],
    transcriber: Transcriber,
    *,
    vad: VadOptions | None = None,
    outdir: str | Path | None = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    skip_existing: bool = False,
    write_files: bool = True,
    stream_index: int | None = None,
    max_chars: int = 84,
    max_cue: float = 6.0,
    on_progress: ProgressFn | None = None,
    on_start: StartFn | None = None,
    on_item: ItemFn | None = None,
) -> BatchOutcome:
    """Transcribe every input in turn, collecting outcomes instead of raising.

    ``on_progress`` receives one *overall* fraction for the whole queue, so a
    progress bar keeps climbing across files instead of restarting at zero.
    ``on_start(item, index, total)`` fires before each file and
    ``on_item(item)`` after it, which is where callers print or render.

    ``write_files=False`` keeps every result in memory and writes nothing to
    disk — the browser UI downloads what it needs instead of dropping files
    next to the sources (its checkbox turns the writes back on).
    """
    files, problems = expand_inputs(inputs)
    outcome = BatchOutcome(problems=problems)
    total = len(files)

    for index, path in enumerate(files, start=1):
        item = BatchItem(path=path)
        if on_start is not None:
            on_start(item, index, total)

        target_dir = Path(outdir) if outdir is not None else path.parent
        if write_files and skip_existing:
            targets, complete = existing_outputs(path.stem, target_dir, formats)
            if complete:
                item.status = "skipped"
                item.outputs = tuple(targets)
                outcome.items.append(item)
                if on_item is not None:
                    on_item(item)
                continue

        def forward(fraction: float, message: str, _done: int = index - 1) -> None:
            if on_progress is None:
                return
            overall = (_done + fraction) / total if total else fraction
            on_progress(min(1.0, overall), message)

        started = time.monotonic()
        try:
            item.result = transcribe_file(
                path,
                transcriber,
                vad=vad,
                stream_index=stream_index,
                on_progress=forward,
            )
            if write_files:
                item.outputs = tuple(
                    write_outputs(
                        item.result,
                        target_dir,
                        formats,
                        max_chars=max_chars,
                        max_cue=max_cue,
                    )
                )
            else:
                # Render once on the way out so a bad format fails here, not in
                # the browser: the payloads are the download endpoints' source.
                payloads(item.result, formats, max_chars=max_chars, max_cue=max_cue)
            item.status = "ok"
        except (AudioExtractionError, ModelLoadError, OSError) as exc:
            item.status, item.error = "failed", str(exc)

        item.seconds = time.monotonic() - started
        outcome.items.append(item)
        if on_item is not None:
            on_item(item)

    return outcome
