#!/usr/bin/env python3
"""Download the default models into ./models.

    python scripts/download_models.py                 # everything
    python scripts/download_models.py --only parakeet-v2
    python scripts/download_models.py --dest /data/models --force

Only the Python standard library is used, so this runs before (or without) the
virtualenv being set up.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# name -> (remote file, is_tarball, path that proves it is already installed)
ASSETS: dict[str, tuple[str, bool, str]] = {
    "parakeet-v2": (
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        True,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx",
    ),
    "vad": ("silero_vad.onnx", False, "silero_vad.onnx"),
}

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "models"


def report(message: str) -> None:
    print(message, flush=True)


def download(url: str, target: Path) -> None:
    """Stream a URL to disk, showing progress on one line."""
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if total:
                sys.stderr.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                sys.stderr.flush()
    if total:
        sys.stderr.write("\n")


def install(name: str, dest: Path, force: bool) -> bool:
    """Fetch one asset. Returns False when it was already present."""
    remote, is_tarball, marker = ASSETS[name]
    if (dest / marker).exists() and not force:
        report(f"[skip] {name}: already installed")
        return False

    url = f"{RELEASE}/{remote}"
    report(f"[get ] {name}: {url}")
    if not is_tarball:
        dest.mkdir(parents=True, exist_ok=True)
        download(url, dest / marker)
        return True

    with tempfile.TemporaryDirectory(dir=dest if dest.exists() else None) as tmpdir:
        archive = Path(tmpdir) / remote
        download(url, archive)
        report(f"[unpack] {name}")
        with tarfile.open(archive, "r:bz2") as tar:
            # filter="data" hardens against path traversal (Python >= 3.12);
            # older interpreters fall back to the unchecked extractall.
            try:
                tar.extractall(dest, filter="data")
            except TypeError:
                tar.extractall(dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the default speech models into ./models.",
        epilog="Example: python scripts/download_models.py --only parakeet-v2 --force",
    )
    parser.add_argument(
        "--only",
        choices=sorted(ASSETS),
        nargs="+",
        help="install just these assets (default: all)",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="models directory")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    wanted = args.only or list(ASSETS)
    failures = 0
    for name in wanted:
        try:
            install(name, args.dest, args.force)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            report(f"[fail] {name}: {exc}")

    if failures:
        report(f"\n{failures} asset(s) failed. Check your network and retry.")
        return 1
    report(f"\nModels ready in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
