"""Fetch model assets into the local ``models/`` directory. Standard library only.

Kept free of package-level imports so it works three ways: from the web UI
(frozen bundle or venv), from a bare interpreter before the venv exists, and from
``scripts/download_models.py``, which re-exports the same tables.

Assets come from the k2-fsa release page as tar.bz2 archives whose top-level
directory is already the model directory, so extraction is just "unpack here".
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

#: Called with (fraction in 0..1, human-readable status).
Progress = Callable[[float, str], None]

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

#: download key -> (remote filename, is a tarball, path inside models/ proving install)
ASSETS: dict[str, tuple[str, bool, str]] = {
    "parakeet-v2": (
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        True,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx",
    ),
    "vad": ("silero_vad.onnx", False, "silero_vad.onnx"),
}

#: catalogue model key -> the downloads it needs. The VAD asset is always added:
#: it is 632 kB and voice-activity detection is on by default.
MODEL_DOWNLOADS: dict[str, tuple[str, ...]] = {
    "parakeet-tdt-0.6b-v2": ("parakeet-v2",),
}

#: Archive sizes in megabytes, shown before a download starts (from the release page).
#: Unpacked the folder is bigger: ~662 MB.
ASSET_SIZES = {"parakeet-v2": 482.5, "vad": 0.6}


def model_dir() -> Path:
    """The ``models/`` directory that sits beside the package or the bundle."""
    from .models import model_root

    return model_root()


def _target(asset: str, dest: Path | None) -> Path:
    root = Path(dest) if dest is not None else model_dir()
    return root / ASSETS[asset][2]


def is_installed(asset: str, dest: Path | None = None) -> bool:
    """True when the file that proves an asset is unpacked exists."""
    return _target(asset, dest).exists()


def pending_for(model_key: str, dest: Path | None = None) -> list[str]:
    """Which assets still have to be downloaded for ``model_key`` (empty if ready)."""
    if model_key not in MODEL_DOWNLOADS:
        raise KeyError(f"no download is configured for {model_key!r}")
    needed = (*MODEL_DOWNLOADS[model_key], "vad")
    return [asset for asset in needed if not is_installed(asset, dest)]


def _download_file(url: str, target: Path, progress: Progress, base: float, span: float) -> None:
    """Stream one URL to disk, reporting ``base + span * fraction`` as it goes."""
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https release URL
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last = -1.0
        with target.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    fraction = min(done / total, 1.0)
                    if fraction - last >= 0.01:
                        last = fraction
                        progress(base + span * fraction, f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")


def _extract(archive: Path, dest: Path) -> None:
    """Unpack a model tarball, refusing members that escape ``dest``."""
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(dest, filter="data")  # Python 3.12+: blocks traversal
        except TypeError:  # Python 3.10/3.11 have no filter argument
            tar.extractall(dest)


def download(
    assets: str | Iterable[str],
    dest: Path | None = None,
    progress: Progress | None = None,
) -> list[Path]:
    """Download and unpack ``assets`` (a key or an iterable of keys).

    Returns the proof-of-install paths. Raises ``RuntimeError`` with the URL when
    the network, the release, or the disk refuses, so the caller can show it
    verbatim instead of a bare traceback.
    """
    keys = [assets] if isinstance(assets, str) else list(assets)
    unknown = [key for key in keys if key not in ASSETS]
    if unknown:
        raise KeyError(f"unknown asset(s) {unknown}; known: {sorted(ASSETS)}")

    root = Path(dest) if dest is not None else model_dir()
    root.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda _fraction, _message: None)

    weights = [ASSET_SIZES.get(key, 100.0) for key in keys]
    total_weight = sum(weights) or 1.0
    written: list[Path] = []
    base = 0.0

    for key, weight in zip(keys, weights, strict=True):
        filename, is_archive, _probe = ASSETS[key]
        span = weight / total_weight
        url = f"{RELEASE}/{filename}"
        report(base, f"downloading {filename}")
        with tempfile.TemporaryDirectory(dir=root, prefix=".fetch-") as staging:
            payload = Path(staging) / filename
            try:
                _download_file(url, payload, report, base, span)
            except OSError as exc:  # URLError, HTTPError, ENOSPC all land here
                raise RuntimeError(f"download failed for {url}: {exc}") from exc
            report(base + span, f"installing {filename}")
            if is_archive:
                _extract(payload, root)
            else:
                shutil.move(str(payload), str(root / filename))
        written.append(_target(key, dest))
        base += span

    report(1.0, "done")
    return written
