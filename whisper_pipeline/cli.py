"""Command line front end.

Installed as the ``python-whisper`` command by ``pyproject.toml``:

    python-whisper clip.mp4 -f txt,srt,vtt
    python-whisper clips/*.mkv -m parakeet-tdt-0.6b-v2 --threads 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import (
    AudioExtractionError,
    ModelLoadError,
    Transcriber,
    VadOptions,
    available_models,
    default_model,
    get_model,
    probe,
    transcribe_file,
)
from .models import vad_model_path

VALID_FORMATS = ("txt", "srt", "vtt", "json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python-whisper",
        description="Transcribe video/audio locally with sherpa-onnx (CPU or CUDA).",
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="media files to transcribe")
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="model key (e.g. parakeet-tdt-0.6b-v2, whisper-base-en). "
        "Defaults to the best installed model.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="compute device (cuda needs a CUDA-enabled sherpa-onnx build)",
    )
    parser.add_argument("-t", "--threads", type=int, default=0, help="CPU threads, 0 = auto")
    parser.add_argument("--no-vad", action="store_true", help="disable silence-based segmentation")
    parser.add_argument(
        "--max-segment",
        type=float,
        default=20.0,
        help="hard ceiling per decoded chunk in seconds (default 20); shorter is "
        "faster on CPU because transformer attention grows quadratically",
    )
    parser.add_argument(
        "--stream",
        type=int,
        default=None,
        help="audio stream index to decode (default: the container's default track)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="where to write transcripts (default: next to each input file)",
    )
    parser.add_argument(
        "-f",
        "--formats",
        default="txt,srt",
        help=f"comma separated output formats, any of: {', '.join(VALID_FORMATS)}",
    )
    parser.add_argument("--max-chars", type=int, default=84, help="subtitle cue line budget")
    parser.add_argument(
        "--max-cue-seconds", type=float, default=6.0, help="subtitle cue duration ceiling"
    )
    parser.add_argument("--list-models", action="store_true", help="show installed models and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    return parser


def print_models() -> int:
    specs = available_models()
    if not specs:
        print("No models installed. Run scripts/download_models.py first.", file=sys.stderr)
        return 1
    width = max(len(spec.key) for spec in specs)
    for spec in specs:
        print(f"{spec.key:<{width}}  {spec.describe()}")
        print(f"{'':<{width}}  {spec.description}")
    return 0


def parse_formats(raw: str) -> list[str]:
    wanted = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in VALID_FORMATS]
    if unknown:
        raise SystemExit(f"unknown format(s): {', '.join(unknown)} (choose from {VALID_FORMATS})")
    return wanted


def write_outputs(
    result, outdir: Path, formats: list[str], max_chars: int, max_cue: float
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.media.path).stem
    writers = {
        "txt": (result.to_txt(), "txt"),
        "srt": (result.to_srt(max_chars=max_chars, max_duration=max_cue), "srt"),
        "vtt": (result.to_vtt(max_chars=max_chars, max_duration=max_cue), "vtt"),
        "json": (result.to_json(), "json"),
    }
    written = []
    for fmt in formats:
        payload, extension = writers[fmt]
        target = outdir / f"{stem}.{extension}"
        target.write_text(payload, encoding="utf-8")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        return print_models()

    if not args.inputs:
        print("Nothing to do: pass one or more media files (or --list-models).", file=sys.stderr)
        return 2

    formats = parse_formats(args.formats)

    try:
        spec = get_model(args.model) if args.model else default_model()
    except (KeyError, FileNotFoundError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    if not spec.is_available():
        print(
            f"error: model '{spec.key}' is incomplete in {spec.directory} "
            f"(missing: {', '.join(spec.missing_files())})",
            file=sys.stderr,
        )
        return 1

    threads = args.threads or None
    vad = (
        None
        if args.no_vad
        else VadOptions(model_path=vad_model_path(), max_speech_duration=args.max_segment)
    )

    exit_code = 0
    try:
        transcriber = Transcriber(spec, device=args.device, num_threads=threads)
    except ModelLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if transcriber.fallback_note and not args.quiet:
        print(f"note: {transcriber.fallback_note}", file=sys.stderr)

    throttle = {"percent": -10, "at": 0.0}

    def on_progress(fraction: float, message: str) -> None:
        """Redraw one progress line, at most every 5% or 1.5 seconds."""
        if args.quiet:
            return
        percent = int(fraction * 100)
        now = time.monotonic()
        if percent < 100 and percent - throttle["percent"] < 5 and now - throttle["at"] < 1.5:
            return
        throttle["percent"], throttle["at"] = percent, now
        print(f"\r  {fraction * 100:5.1f}%  {message:<56}", end="", file=sys.stderr)

    for path in args.inputs:
        if not path.is_file():
            print(f"error: no such file: {path}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            info = probe(path)
            if not args.quiet:
                print(
                    f"{path.name}: {info.duration_label}, "
                    f"{len(info.audio_streams)} audio track(s), {info.stream().label}",
                    file=sys.stderr,
                )
            result = transcribe_file(
                path,
                transcriber,
                vad=vad,
                stream_index=args.stream,
                on_progress=on_progress,
            )
        except (AudioExtractionError, ModelLoadError) as exc:
            print(f"\nerror: {path.name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        if not args.quiet:
            print(file=sys.stderr)

        outdir = args.outdir or path.parent
        written = write_outputs(result, outdir, formats, args.max_chars, args.max_cue_seconds)

        print(
            f"{path.name}: {result.segment_count} segments, {result.word_count} words, "
            f"{result.audio_seconds:.1f}s audio in {result.wall_seconds:.1f}s "
            f"(RTF {result.rtf:.2f})"
        )
        for target in written:
            print(f"  wrote {target}")
        if not args.quiet and result.segments:
            preview = result.to_txt()
            snippet = preview[:300] + ("…" if len(preview) > 300 else "")
            print(f"  text: {snippet}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
