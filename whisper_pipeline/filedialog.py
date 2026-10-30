"""Open a native file dialog on the *server* machine and return absolute paths.

The web page cannot open a file dialog on the machine that runs the server, and a
browser file picker would stream the bytes over HTTP into a temp directory. When the
server and the user sit at the same computer — the only supported deployment — the
dialog belongs to the server process: the user picks files, Python receives paths, and
no media is copied anywhere.

Each platform's picker therefore runs as a **short-lived child process** rather than
in-process: ``powershell`` + WinForms on Windows, ``osascript`` on macOS, ``zenity`` or
``kdialog`` on Linux, and a ``tkinter`` subprocess only if nothing better exists. A
child process keeps toolkit main-thread rules (Cocoa) out of the HTTP server's way, so
this is safe to call from a request thread, and a hung dialog is killable by timeout.

Environment overrides, both for headless machines and for tests:

``WHISPER_NO_DIALOG=1``      pretend no toolkit is available (UI falls back to paths)
``WHISPER_DIALOG_STUB``      newline-separated paths to return instead of showing one
``WHISPER_DIALOG_CANCEL=1``  with the stub, behave as if the user cancelled
"""

from __future__ import annotations

import locale
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

#: Extensions offered in the dialog filter, mirroring what the pipeline can decode.
MEDIA_GLOBS: tuple[str, ...] = (
    "*.mp4",
    "*.mkv",
    "*.mov",
    "*.webm",
    "*.avi",
    "*.mpg",
    "*.mpeg",
    "*.ts",
    "*.m2ts",
    "*.m4a",
    "*.mp3",
    "*.wav",
    "*.flac",
    "*.ogg",
    "*.opus",
    "*.aac",
    "*.wma",
)

#: PowerShell dialog bodies. Kept as templates (not f-strings) because Windows
#: Forms/WinForms properties are case-sensitive and braces are literal PowerShell.
_PS_FOLDER = """
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = '{title}'
if ($dlg.ShowDialog($owner) -eq 'OK') { Write-Output $dlg.SelectedPath }
"""

_PS_FILES = """
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.Multiselect = $true
$dlg.Title = '{title}'
$dlg.Filter = 'Media files|{patterns}|All files|*.*'
if ($dlg.ShowDialog($owner) -eq 'OK') { $dlg.FileNames | ForEach-Object { Write-Output $_ } }
"""

#: Long enough for someone to browse a folder, short enough that a dialog that never
#: appears (service, wrong session) does not wedge the server.
DIALOG_TIMEOUT = 600.0

_KINDS = ("files", "folder")
_lock = threading.Lock()
_cache: dict[str, str | None] = {}  # "provider" once resolved


class DialogUnavailable(RuntimeError):
    """No native dialog toolkit on this machine (headless, or a bare interpreter)."""


class DialogBusy(RuntimeError):
    """Another dialog is already open — only one can own the screen."""


class DialogFailed(RuntimeError):
    """The picker ran but could not be used; message says why."""


# --------------------------------------------------------------------------- #
# Provider detection
# --------------------------------------------------------------------------- #
def _windows_powershell() -> str | None:
    for name in ("powershell", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _linux_dialog_tool() -> str | None:
    for name in ("zenity", "kdialog"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _tkinter_works() -> bool:
    """True when this interpreter has tkinter *and* something to draw on."""
    if os.name not in {"nt", "darwin"} and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("tkinter") is not None
    except (ImportError, ValueError):  # a broken tcl install, oddly packaged builds
        return False


def provider() -> str | None:
    """Which picker this machine would use, or ``None``. Resolved once and cached."""
    if "provider" in _cache:
        return _cache["provider"]

    resolved: str | None
    if os.environ.get("WHISPER_NO_DIALOG", "").strip() not in {"", "0"}:
        resolved = None
    elif os.environ.get("WHISPER_DIALOG_STUB"):
        resolved = "stub"
    elif os.name == "nt":
        resolved = "powershell" if _windows_powershell() else None
    elif sys.platform == "darwin":
        resolved = "osascript" if shutil.which("osascript") else None
    else:
        resolved = _linux_dialog_tool()

    if resolved is None and _tkinter_works():
        resolved = "tkinter"
    _cache["provider"] = resolved
    return resolved


def describe() -> dict[str, object]:
    """What the page needs to know to show, hide, or explain the picker."""
    found = provider()
    reason = ""
    if found is None:
        reason = (
            "no file dialog is available here — use the paths box "
            "(install zenity on Linux, or start the app on a desktop session)"
        )
    return {"provider": found, "available": found is not None, "reason": reason}


# --------------------------------------------------------------------------- #
# Child processes
# --------------------------------------------------------------------------- #
def _run(argv: list[str], timeout: float) -> tuple[int, str, str]:
    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user interpolation
            argv,
            capture_output=True,
            timeout=timeout,
            creationflags=creationflags,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise DialogFailed(f"the file dialog did not close within {int(timeout)} s") from None
    except OSError as exc:  # missing toolkit, no DISPLAY, exec denied
        raise DialogUnavailable(str(exc)) from None
    return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)


def _decode(raw: bytes) -> str:
    """Decode child output, preferring UTF-8 and degrading to the console codepage.

    PowerShell is told to emit UTF-8, but a process without a usable console (a
    service, a detached session) may ignore that and emit the ANSI/OEM codepage
    instead, so fall back until some strict decode succeeds.
    """
    if not raw:
        return ""
    candidates = ["utf-8", locale.getpreferredencoding(False), "cp1252", "latin-1"]
    for encoding in candidates:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _split(text: str) -> list[Path]:
    return [Path(line.strip()) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Per-platform pickers
# --------------------------------------------------------------------------- #
def _powershell_script(kind: str, title: str) -> str:
    quoted = title.replace("'", "''")
    if kind == "folder":
        body = _PS_FOLDER.replace("{title}", quoted)
    else:
        body = _PS_FILES.replace("{title}", quoted).replace("{patterns}", ";".join(MEDIA_GLOBS))
    return (
        "$ErrorActionPreference = 'Stop'\n"
        # UTF-8 before anything is printed, so accented paths survive the console;
        # a detached process has no console, which must not kill the script.
        "try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }\n"
        "Add-Type -AssemblyName System.Windows.Forms\n"
        "Add-Type -AssemblyName System.Drawing\n"
        # A hidden topmost owner window makes the dialog appear in front of the browser
        # instead of behind it, which otherwise looks like nothing happened.
        "$owner = New-Object System.Windows.Forms.Form\n"
        "$owner.TopMost = $true\n"
        "$owner.ShowInTaskbar = $false\n"
        "$owner.FormBorderStyle = 'None'\n"
        "$owner.Size = New-Object System.Drawing.Size(1, 1)\n"
        "$owner.StartPosition = 'CenterScreen'\n"
        "$owner.Show()\n"
        "$owner.Activate()\n"
        "try {\n" + body + "} finally { $owner.Close() }\n"
    )


def _macos_script(kind: str, title: str) -> list[str]:
    quoted = title.replace('"', '\\"')
    if kind == "folder":
        return [
            "osascript",
            "-e",
            f'set theFolder to choose folder with prompt "{quoted}"',
            "-e",
            "return POSIX path of theFolder",
        ]
    return [
        "osascript",
        "-e",
        f'set theFiles to choose file with prompt "{quoted}" with multiple selections allowed',
        "-e",
        'set out to ""',
        "-e",
        "repeat with f in theFiles",
        "-e",
        "set out to out & POSIX path of f & linefeed",
        "-e",
        "end repeat",
        "-e",
        "return out",
    ]


def _zenity_argv(kind: str, title: str) -> list[str]:
    argv = ["zenity", "--file-selection", f"--title={title}"]
    if kind == "folder":
        return [*argv, "--directory"]
    return [
        *argv,
        "--multiple",
        "--separator=\n",
        f"--file-filter=Media files | {' '.join(MEDIA_GLOBS)}",
        "--file-filter=All files | *",
    ]


def _kdialog_argv(kind: str, title: str) -> list[str]:
    if kind == "folder":
        return ["kdialog", "--title", title, "--getexistingdirectory", str(Path.home())]
    return [
        "kdialog",
        "--title",
        title,
        "--getopenfilename",
        str(Path.home()),
        " ".join(MEDIA_GLOBS),
        "--multiple",
        "--separate-output",
    ]


_TK_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog

kind = sys.argv[1]
title = sys.argv[2]
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
if kind == "folder":
    picked = filedialog.askdirectory(title=title, mustexist=True)
    paths = [picked] if picked else []
else:
    patterns = sys.argv[3]
    paths = list(
        filedialog.askopenfilenames(
            title=title,
            filetypes=[("Media files", patterns), ("All files", "*")],
        )
    )
sys.stdout.write("\\n".join(paths))
"""


def _pick_tkinter(kind: str, title: str, timeout: float) -> list[Path]:
    argv = [sys.executable, "-c", _TK_SCRIPT, kind, title, " ".join(MEDIA_GLOBS)]
    code, text, err = _run(argv, timeout)
    if code != 0 and not text.strip():
        raise DialogFailed(err.strip() or "the tkinter dialog could not be opened")
    return _split(text)


def _pick_with(provider_name: str, kind: str, title: str, timeout: float) -> list[Path]:
    if provider_name == "powershell":
        argv = [
            _windows_powershell() or "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-Command",
            _powershell_script(kind, title),
        ]
        code, text, err = _run(argv, timeout)
        if code != 0 and not text.strip():
            raise DialogFailed(err.strip() or "PowerShell could not show the dialog")
        return _split(text)

    if provider_name == "osascript":
        code, text, err = _run(_macos_script(kind, title), timeout)
        if not text.strip():
            if code != 0 and "cancel" not in err.lower():  # -128 is "user cancelled"
                raise DialogFailed(err.strip() or "osascript could not show the dialog")
            return []
        return _split(text)

    if provider_name == "zenity":
        code, text, err = _run(_zenity_argv(kind, title), timeout)
        if code == 1 and not text.strip():
            return []
        if code != 0 and not text.strip():
            raise DialogFailed(err.strip() or "zenity could not show the dialog")
        return _split(text)

    if provider_name == "kdialog":
        code, text, err = _run(_kdialog_argv(kind, title), timeout)
        if code == 1 and not text.strip():
            return []
        if code != 0 and not text.strip():
            raise DialogFailed(err.strip() or "kdialog could not show the dialog")
        return _split(text)

    if provider_name == "tkinter":
        return _pick_tkinter(kind, title, timeout)

    raise DialogUnavailable(f"no picker for {provider_name!r}")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def pick(kind: str = "files", *, title: str = "", timeout: float = DIALOG_TIMEOUT) -> list[Path]:
    """Show the native dialog and return the chosen paths (``[]`` when cancelled).

    Raises :class:`DialogUnavailable` when the machine has no usable picker and
    :class:`DialogBusy` when one is already open, so the caller can answer with a
    specific message instead of looking like a hang.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    label = title or (
        "Choose a folder of media" if kind == "folder" else "Choose video or audio files"
    )

    # Test hooks: headless CI has no desktop, so the smoke test drives the dialog
    # outcomes from the environment instead of clicking a real window.
    if os.environ.get("WHISPER_DIALOG_CANCEL", "").strip() not in {"", "0"}:
        return []
    stub = os.environ.get("WHISPER_DIALOG_STUB")
    if stub:
        return [Path(line.strip()) for line in stub.splitlines() if line.strip()]

    name = provider()
    if name is None:
        raise DialogUnavailable(describe()["reason"] or "no file dialog available")

    if not _lock.acquire(blocking=False):
        raise DialogBusy("a file dialog is already open — finish or cancel it first")
    try:
        return _pick_with(name, kind, label, timeout)
    finally:
        _lock.release()
