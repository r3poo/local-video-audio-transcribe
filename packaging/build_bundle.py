#!/usr/bin/env python3
"""Assemble a self-contained "unzip and run" bundle for the host platform.

The bundle is a folder, not a single executable — deliberately. Freezing
Streamlit, PyAV's FFmpeg DLLs and onnxruntime into a PyInstaller exe is a long
tail of hidden-import archaeology, and `--onefile` would unpack ~730 MB into a
temp directory on every launch. A portable CPython plus the same wheels
`uv sync` installs has none of that fragility and still needs no setup: unzip,
double-click ``Start.bat``, transcribe.

    uv run --no-project python packaging/build_bundle.py            # host OS
    uv run --no-project python packaging/build_bundle.py --models all
    uv run --no-project python packaging/build_bundle.py --no-zip --out /tmp/x

Requirements: ``uv`` on PATH. Python 3.12 is the default target because every
pinned wheel (av, numpy, pyarrow, onnxruntime, sherpa-onnx) publishes a
cp312 win_amd64 build.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = "3.12"
DEFAULT_NAME = "whisper-portable"

#: Copied into the bundle verbatim.
APP_FILES = ("app.py", "README.md", "LICENSE", "LICENCE")
APP_DIRS = ("whisper_pipeline",)

#: Never worth their bytes in a distribution archive.
PRUNE_DIRS = ("__pycache__", "tests", "test")
PRUNE_SUFFIXES = (".pyc", ".pyo", ".pdb", ".lib", ".a")


def read_dependencies() -> list[str]:
    """Runtime dependencies from pyproject.toml, so pins live in exactly one place."""
    text = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 on a Windows box
        block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
        if not block:
            raise SystemExit("cannot find [project].dependencies in pyproject.toml") from None
        return re.findall(r'"([^"]+)"', block.group(1))
    return list(tomllib.loads(text)["project"]["dependencies"])


def run(command: list[str], *, quiet: bool = False) -> None:
    shown = " ".join(command)
    print(f"  $ {shown[:160]}{'…' if len(shown) > 160 else ''}", flush=True)
    result = subprocess.run(
        command,
        cwd=PROJECT,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if quiet and result.stdout:
            print(result.stdout[-4000:], file=sys.stderr)
        raise SystemExit(f"command failed ({result.returncode}): {shown}")


def find_interpreter(runtime: Path) -> Path:
    """Locate the interpreter inside a ``uv python install`` tree."""
    names = ("python.exe",) if os.name == "nt" else ("python3", "python")
    for name in names:
        matches = sorted(runtime.rglob(name))
        executables = [path for path in matches if path.is_file() and os.access(path, os.X_OK)]
        if os.name == "nt":
            executables = [path for path in matches if path.is_file()]
        if executables:
            # Shallowest match wins: <runtime>/cpython-3.12.15-…/python.exe on Windows,
            # <runtime>/cpython-3.12.15-…/bin/python3 on POSIX.
            return min(executables, key=lambda path: len(path.parts))
    raise SystemExit(f"no interpreter found under {runtime}")


def launcher_scripts(python_native: str, python_posix: str) -> dict[str, str]:
    """Entry points written next to the bundle, keyed by filename.

    ``uv python install`` unpacks the interpreter into a version-named folder, so the
    launchers must point at the path discovered at build time rather than at
    ``runtime\\python.exe`` (which does not exist).
    """
    header = '@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\nset PYTHONUTF8=1\r\n'
    exe = f'"%~dp0{python_native}"'
    streamlit = (
        f'{exe} -m streamlit run "%~dp0app.py" '
        "--server.maxUploadSize=4096 --browser.gatherUsageStats=false %*\r\n"
    )
    return {
        "Start.bat": (
            header
            + "title Local Transcriber\r\n"
            + streamlit
            + "echo.\r\necho The app stopped. Press any key to close.\r\n"
            + "pause >nul\r\n"
        ),
        "Transcribe.bat": (
            header
            + 'rem Command-line transcription, e.g.  Transcribe.bat "C:\\clips" -f txt\r\n'
            + 'if "%~1"=="" goto usage\r\n'
            + f"{exe} -m whisper_pipeline.cli %*\r\n"
            + "goto end\r\n"
            + ":usage\r\n"
            + "echo Usage: Transcribe.bat MEDIA [MEDIA...] [options]\r\n"
            + 'echo Example: Transcribe.bat "C:\\clips" -f txt,sentences -o transcripts\r\n'
            + "echo Run without arguments for the full help: Transcribe.bat --help\r\n"
            + ":end\r\n"
            + "pause >nul\r\n"
        ),
        "start.sh": (
            '#!/bin/sh\ncd "$(dirname "$0")"\nexport PYTHONUTF8=1\n'
            f"exec ./{python_posix} -m streamlit run app.py "
            '--server.maxUploadSize=4096 --browser.gatherUsageStats=false "$@"\n'
        ),
        "transcribe.sh": (
            '#!/bin/sh\ncd "$(dirname "$0")"\nexport PYTHONUTF8=1\n'
            f'exec ./{python_posix} -m whisper_pipeline.cli "$@"\n'
        ),
    }


def readme_text(zip_name: str) -> str:
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"""Local Transcriber — portable build
==================================

No installation. Unzip anywhere you can write (Desktop, C:\\Tools, a USB stick)
and double-click Start.bat. Your browser opens the transcription page.

First run: the app has no speech model yet, so the sidebar offers a one-time
download (~483 MB). It needs internet once and lands in the models folder
beside this file; every run after that is fully offline.

  Start.bat              the web UI (the usual entry point)
  Transcribe.bat         command line, e.g.  Transcribe.bat ..\\clips -f txt
  app.py                 the Streamlit page
  whisper_pipeline\\      the engine
  runtime\\               self-contained Python {py}, do not edit
  models\\                downloaded models live here

Notes
-----
* CPU only. This build has no CUDA runtime; GPU use needs the CUDA wheel and a
  matching NVIDIA driver — see the project README.
* Windows may show "Windows protected your PC" because the launcher is not
  code-signed: More info -> Run anyway. Some antivirus tools flag any freshly
  built unsigned launcher for a day or two; the bundle contains nothing but
  CPython, PyAV, onnxruntime and Streamlit.
* Uninstall = delete this folder.
* Delete models\\ to reclaim disk space, or move it to a bigger drive and run
  the app from there.

Built by packaging/build_bundle.py — {zip_name}
"""


def copy_app(bundle: Path) -> None:
    for name in APP_FILES:
        source = PROJECT / name
        if source.is_file():
            shutil.copy2(source, bundle / name)
    for name in APP_DIRS:
        source = PROJECT / name
        if not source.is_dir():
            raise SystemExit(f"missing source directory: {source}")
        shutil.copytree(
            source,
            bundle / name,
            ignore=shutil.ignore_patterns(*PRUNE_DIRS, "__pycache__", "*.pyc"),
            dirs_exist_ok=True,
        )


def copy_models(bundle: Path, wanted: list[str]) -> list[str]:
    """Copy selected model directories out of the development ``models/`` folder."""
    models_dir = bundle / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if not wanted:
        return []
    available = {path.name: path for path in (PROJECT / "models").iterdir() if path.is_dir()}
    copied: list[str] = []
    for name in wanted:
        matches = [key for key in available if name == "all" or name in key]
        if not matches:
            raise SystemExit(f"no model directory matching {name!r} in {PROJECT / 'models'}")
        for match in matches:
            print(f"  copying model {match}")
            shutil.copytree(available[match], models_dir / match)
            copied.append(match)
    vad = PROJECT / "models" / "silero_vad.onnx"
    if vad.is_file():
        shutil.copy2(vad, models_dir / "silero_vad.onnx")
    return copied


def prune(bundle: Path) -> int:
    """Delete caches and test suites; returns the number of removed entries."""
    removed = 0
    for path in sorted(bundle.rglob("*"), reverse=True):
        if path.is_dir() and path.name in PRUNE_DIRS:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        elif path.is_file() and path.suffix in PRUNE_SUFFIXES:
            path.unlink(missing_ok=True)
            removed += 1
    # The stdlib test suite is another ~30 MB of nothing.
    for path in bundle.glob("runtime/**/lib/python3*/test"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def folder_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def make_zip(bundle: Path, archive_base: Path) -> Path:
    """Zip the bundle with its own folder as the archive root."""
    target = shutil.make_archive(
        str(archive_base), "zip", root_dir=bundle.parent, base_dir=bundle.name
    )
    return Path(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python", default=DEFAULT_PYTHON, help=f"runtime version (default {DEFAULT_PYTHON})"
    )
    parser.add_argument(
        "--out", default=str(PROJECT / "dist"), help="output folder (default dist/)"
    )
    parser.add_argument(
        "--name", default=DEFAULT_NAME, help=f"bundle folder name (default {DEFAULT_NAME})"
    )
    parser.add_argument(
        "--models",
        default="none",
        help="'none' (download in the app), 'all', or comma-separated names from models/",
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="leave the folder, skip archive creation"
    )
    parser.add_argument("--keep", action="store_true", help="reuse an existing bundle folder")
    args = parser.parse_args(argv)

    if not shutil.which("uv"):
        raise SystemExit(
            "uv is required (https://docs.astral.sh/uv/) — install with:\n"
            "  winget install --id=astral-sh.uv -e"
        )

    out = Path(args.out).expanduser().resolve()
    bundle = out / args.name
    platform_tag = f"{sys.platform}-{os.uname().machine if hasattr(os, 'uname') else 'x86_64'}"
    zip_name = f"{args.name}-{platform_tag}"
    started = time.monotonic()

    print(f"Building {bundle} (python {args.python})")
    if bundle.exists():
        if args.keep:
            print("  reusing existing folder (--keep)")
        else:
            shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)

    runtime = bundle / "runtime"
    print("[1/6] interpreter")
    run(["uv", "python", "install", args.python, "--install-dir", str(runtime)], quiet=True)
    python = find_interpreter(runtime)
    print(f"  {python.relative_to(bundle)}")
    # uv marks its interpreter trees PEP 668 "externally managed", which blocks both
    # pip and `uv pip install --python`. Drop the marker so the build can install the
    # wheels and so owners of the bundle can pip-install extras into it later.
    for marker in runtime.glob("**/[Ll]ib/python3*/EXTERNALLY-MANAGED"):
        marker.unlink(missing_ok=True)

    print("[2/6] wheels from pyproject.toml")
    # Use the bundled interpreter's own pip: `uv pip install --python` would need
    # --break-system-packages semantics and pip's version is pinned by the runtime.
    requirements = out / ".requirements.bundle.txt"
    requirements.write_text("\n".join(read_dependencies()) + "\n", encoding="utf-8")
    try:
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--root-user-action=ignore",
                "--break-system-packages",
                "-r",
                str(requirements),
            ]
        )
    finally:
        requirements.unlink(missing_ok=True)

    print("[3/6] application files")
    copy_app(bundle)

    print("[4/6] models")
    wanted = [part for part in args.models.split(",") if part and part != "none"]
    copied_models = copy_models(bundle, wanted)
    summary = "bundled: " + ", ".join(copied_models) if copied_models else "none"
    print(f"  {summary} — the app offers a one-time download when models/ is empty")

    print("[5/6] launchers")
    python_posix = python.relative_to(bundle).as_posix()
    python_native = str(python.relative_to(bundle)).replace("/", "\\")
    for name, body in launcher_scripts(python_native, python_posix).items():
        target = bundle / name
        target.write_bytes(body.encode("utf-8"))
        if target.suffix == ".sh":
            target.chmod(0o755)
        print(f"  {name}")
    (bundle / "README-FIRST.txt").write_text(readme_text(zip_name), encoding="utf-8")

    print("[6/6] prune")
    print(f"  removed {prune(bundle)} cache/test entries")

    size_mb = folder_size(bundle) / 1e6
    print(f"\nBundle: {bundle}  ({size_mb:.0f} MB, {time.monotonic() - started:.0f}s)")
    if args.no_zip:
        print("Skipped the archive (--no-zip).")
        return 0

    print("Zipping…", flush=True)
    archive = make_zip(bundle, out / zip_name)
    print(f"Archive: {archive}  ({archive.stat().st_size / 1e6:.0f} MB)")
    print("\nDistribute that zip. Users unzip it and double-click Start.bat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
