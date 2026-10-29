"""Bind sherpa-onnx to the ONNX Runtime it was actually compiled against.

The ``sherpa-onnx`` wheel holds only the Python bindings (``_sherpa_onnx*.pyd``
on Windows, ``.so``/``.dylib`` elsewhere); the runtime itself ships in the
companion ``sherpa-onnx-core`` wheel under ``sherpa_onnx/lib/``.  Windows
resolves a DLL import by *base name*, and ``C:\\Windows\\System32\\onnxruntime.dll``
— the copy that Windows ML installs, ONNX Runtime 1.17.x — is searched before
the package directory, so the bindings can end up bound to a runtime years
older than the one they were built for:

    The requested API version [28] is not available, only API versions
    [1, 17] are supported in this build. Current ORT Version is: 1.17.1

Loading the bundled runtime by absolute path *before* ``import sherpa_onnx``
registers it in the process under that base name, so the bindings' implicit
import resolves to it instead.  :func:`prepare` does that and is called right
before the first ``import sherpa_onnx`` (see ``whisper_pipeline/engine.py``).

:func:`describe` reports which runtime the process is really using, so
``python scripts/check_runtime.py`` can explain a mismatch instead of leaving
the ONNX Runtime message to speak for itself.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

#: Base name of the ONNX Runtime shared library, per platform.
RUNTIME_NAMES = {
    "win32": "onnxruntime.dll",
    "darwin": "libonnxruntime.dylib",
}

#: Distributions that ship an ``onnxruntime`` of their own and can therefore
#: shadow the bundled one.  Harmless on their own; fatal if imported first.
FOREIGN_RUNTIME_DISTS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")

#: DLLs to preload on Windows, in dependency order.  ``onnxruntime.dll`` must
#: come first: it is the one whose base name is contested.
_WINDOWS_PRELOAD = ("onnxruntime.dll", "sherpa-onnx-c-api.dll", "sherpa-onnx-cxx-api.dll")

# Handles returned by ``os.add_dll_directory`` / ``ctypes.WinDLL`` close when
# garbage collected, which would undo the preload, so they are kept alive here.
_kept_alive: list[object] = []


def runtime_name() -> str:
    """File name of the ONNX Runtime shared library on this platform."""
    return RUNTIME_NAMES.get(sys.platform, "libonnxruntime.so")


def runtime_dir() -> Path | None:
    """Path of ``site-packages/sherpa_onnx/lib``, located without importing it.

    ``find_spec`` does not execute the module, so this is safe to call before
    ``import sherpa_onnx`` — which is the whole point: it has to happen first.
    """
    try:
        spec = importlib.util.find_spec("sherpa_onnx")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent / "lib"


def bundled_runtime() -> Path | None:
    """The ONNX Runtime that ``sherpa-onnx-core`` installed, if present."""
    directory = runtime_dir()
    if directory is None:
        return None
    candidate = directory / runtime_name()
    return candidate if candidate.is_file() else None


def prepare() -> None:
    """Preload the bundled ONNX Runtime so sherpa-onnx binds to it (Windows).

    A no-op elsewhere: ELF and Mach-O resolve this through RPATH, and no
    system-wide copy of the library competes for the name.
    """
    if sys.platform != "win32" or _kept_alive:
        return
    directory = runtime_dir()
    if directory is None or not directory.is_dir():
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        _kept_alive.append(add_dll_directory(str(directory)))
    for name in _WINDOWS_PRELOAD:
        path = directory / name
        if not path.is_file():
            continue
        try:
            _kept_alive.append(ctypes.WinDLL(str(path)))
        except OSError:
            # A missing dependency, or the module is already loaded under this
            # base name.  Either way there is nothing left to do here and
            # describe() reports what actually happened.
            continue


def loaded_runtime() -> Path | None:
    """Path of the ONNX Runtime the process has mapped, if it can be told."""
    try:
        return _loaded_windows_runtime() if sys.platform == "win32" else _loaded_posix_runtime()
    except OSError:  # pragma: no cover - platform probing is best effort
        return None


def _loaded_windows_runtime() -> Path | None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    # HMODULE must not be truncated to int, hence the explicit restype above.
    handle = kernel32.GetModuleHandleW(runtime_name())
    if not handle:
        return None
    kernel32.GetModuleFileNameW.restype = ctypes.c_uint
    kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    buffer = ctypes.create_unicode_buffer(32768)
    if kernel32.GetModuleFileNameW(handle, buffer, len(buffer)) == 0:
        return None
    return Path(buffer.value)


def _loaded_posix_runtime() -> Path | None:
    maps = Path("/proc/self/maps")
    try:
        text = maps.read_text(encoding="utf-8", errors="replace")
    except OSError:  # not Linux, or /proc is hidden
        return None
    wanted = (runtime_name(), "libonnxruntime.so")
    for line in text.splitlines():
        # "<addr> <perms> <offset> <dev> <inode> <pathname>"
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        path = parts[5].removesuffix(" (deleted)")
        if any(name in path for name in wanted):
            return Path(path)
    return None


def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _same_file(first: Path, second: Path) -> bool:
    """True when both paths are the same runtime, however it was installed.

    ``sherpa-onnx-core`` drops an identical ``onnxruntime.dll`` next to
    ``python.exe`` as well as into ``sherpa_onnx/lib``, so the mapped path is
    not always the bundled one even when nothing is wrong.
    """
    try:
        if first.resolve() == second.resolve():
            return True
        return first.stat().st_size == second.stat().st_size
    except OSError:
        return False


def describe() -> dict[str, object]:
    """Everything needed to explain a runtime mismatch.

    Returns a dict with ``facts`` (label -> printable value), ``problems``
    (fatal), ``notes`` (suspicious but working) and ``ok``.  Importing
    sherpa-onnx is allowed to fail here — that is itself a finding.
    """
    directory = runtime_dir()
    bundled = bundled_runtime()
    loaded = loaded_runtime()
    bindings = _dist_version("sherpa-onnx")
    core = _dist_version("sherpa-onnx-core")
    foreign = {name: _dist_version(name) for name in FOREIGN_RUNTIME_DISTS}
    foreign = {name: found for name, found in foreign.items() if found}

    ort_version: str | None = None
    providers: tuple[str, ...] = ()
    problems: list[str] = []
    notes: list[str] = []

    try:
        import sherpa_onnx

        ort_version = getattr(sherpa_onnx, "onnxruntime_version", None)
    except Exception as exc:  # noqa: BLE001 - an import failure *is* a diagnosis
        problems.append(f"importing sherpa_onnx failed — {type(exc).__name__}: {exc}")
    try:
        from .models import available_providers

        providers = available_providers()
    except Exception:  # noqa: BLE001 - provider probing is a nicety, not a verdict
        providers = ()

    if directory is None:
        problems.append("sherpa-onnx is not installed in this interpreter")
    elif bundled is None:
        problems.append(
            f"{runtime_name()} is missing from {directory} — the sherpa-onnx-core "
            "package is not installed (or has the wrong version)"
        )
    if loaded is not None and bundled is not None and not _same_file(loaded, bundled):
        problems.append(
            f"the bindings loaded {loaded} (ORT {ort_version or 'unknown'}), "
            f"not the bundled {bundled}"
        )
    if bindings and core and bindings != core:
        problems.append(
            f"sherpa-onnx {bindings} and sherpa-onnx-core {core} are different "
            "releases; the bindings and their runtime must match"
        )
    if foreign:
        installed = ", ".join(f"{name} {found}" for name, found in foreign.items())
        notes.append(
            f"{installed} also provides {runtime_name()}; it only wins if it is "
            "imported before sherpa-onnx"
        )
    if loaded is None:
        notes.append(f"{runtime_name()} is not loaded yet — nothing has been decoded")

    return {
        "facts": {
            "python": f"{sys.version.split()[0]} ({sys.platform})",
            "sherpa-onnx": bindings or "not installed",
            "sherpa-onnx-core": core or "MISSING",
            "other onnxruntime dists": ", ".join(f"{n} {v}" for n, v in foreign.items()) or "none",
            "bundled runtime": str(bundled) if bundled else "MISSING",
            "loaded runtime": str(loaded) if loaded else "not loaded",
            "ORT version in use": ort_version or "unknown",
            "providers": ", ".join(providers) or "unknown",
        },
        "problems": problems,
        "notes": notes,
        "ok": not problems,
        "bundled": bundled,
        "foreign": foreign,
    }


def mismatch_hint() -> str:
    """One-paragraph remedy for the ONNX Runtime API-version abort.

    Appended to :class:`~whisper_pipeline.engine.ModelLoadError` so the user
    sees the cause rather than the raw ONNX Runtime message.
    """
    bundled = bundled_runtime()
    loaded = loaded_runtime()
    loaded_version = None
    try:
        import sherpa_onnx

        loaded_version = getattr(sherpa_onnx, "onnxruntime_version", None)
    except Exception:  # noqa: BLE001 - the hint must never mask the real error
        pass

    if bundled is None:
        return (
            f"the sherpa-onnx bindings cannot find the {runtime_name()} they were "
            "built against: sherpa-onnx-core is missing. Install it with "
            "`pip install sherpa-onnx-core==<same version as sherpa-onnx>`, then run "
            "`python scripts/check_runtime.py`"
        )

    if sys.platform == "win32":
        where = f" ({loaded})" if loaded else ""
        return (
            "the sherpa-onnx bindings and the ONNX Runtime they load come from "
            f"different releases: bindings expect a newer API than the {loaded_version or 'old'}"
            f" runtime in use{where}. Remove any onnxruntime / onnxruntime-gpu package "
            f"from this environment and make sure C:\\Windows\\System32\\{runtime_name()} "
            "(Windows ML) is not imported first — run `python scripts/check_runtime.py`"
        )

    where = f" ({loaded})" if loaded else ""
    return (
        "the sherpa-onnx bindings and the ONNX Runtime they load come from different "
        f"releases: bindings expect a newer API than the {loaded_version or 'old'} runtime "
        f"in use{where}. Install the engine as one matched set — "
        f"{runtime_name()} must come from sherpa-onnx-core, not from onnxruntime[-gpu] "
        "(README: GPU setup)"
    )
