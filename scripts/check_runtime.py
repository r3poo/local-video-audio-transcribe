#!/usr/bin/env python3
"""Show which ONNX Runtime sherpa-onnx actually loaded, and what to do if it bit.

    python scripts/check_runtime.py

Exit code 0 when the bindings and their runtime are a matched set, 1 when not.
Mostly useful on Windows, where ``C:\\Windows\\System32\\onnxruntime.dll`` (the
ONNX Runtime 1.17.x that Windows ML installs) can outrank the runtime bundled in
``sherpa-onnx-core`` and produce:

    The requested API version [28] is not available, only API versions [1, 17]
    are supported in this build. Current ORT Version is: 1.17.1

Only the standard library plus this project is used, so it runs before a broken
environment is repaired.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


#: Remedies keyed by the failure they address; printed only when it applies.
REMEDIES = {
    "core missing": (
        "The bindings need the runtime from sherpa-onnx-core:\n"
        "    pip install sherpa-onnx-core==<same version as sherpa-onnx>\n"
        "    uv pip install sherpa-onnx-core   # uv resolves the matching version"
    ),
    "foreign": (
        "An onnxruntime package in this environment can claim the name first:\n"
        "    pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml"
    ),
    "system": (
        "Windows ML ships an old onnxruntime.dll in System32. whisper_pipeline\n"
        "preloads the bundled one, so this only bites code that imports ONNX\n"
        "Runtime before it — keep onnxruntime imports inside the pipeline, or\n"
        "drop the package from this environment."
    ),
}


def load_runtime_module() -> tuple[object, str | None]:
    """Import ``whisper_pipeline.ort_runtime``, even when the package is broken.

    ``import whisper_pipeline`` pulls in sherpa-onnx, which is exactly what fails
    when the runtime is missing — so fall back to loading the single module from
    its file and report why the normal import did not work.
    """
    try:
        from whisper_pipeline import ort_runtime

        return ort_runtime, None
    except Exception as exc:  # noqa: BLE001 - being broken is why this script runs
        spec = importlib.util.spec_from_file_location(
            "_ort_runtime", PROJECT / "whisper_pipeline" / "ort_runtime.py"
        )
        if spec is None or spec.loader is None:  # pragma: no cover - package is gone
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ort_runtime, failure = load_runtime_module()
    if failure is not None:
        print(f"note: `import whisper_pipeline` failed — {failure}")
        print("      diagnosing the engine directly\n")

    # Same preload the pipeline performs, so the report describes the state the
    # app runs in rather than the state before the fix.
    ort_runtime.prepare()

    report = ort_runtime.describe()
    facts = report["facts"]
    width = max(len(label) for label in facts)

    print("ONNX Runtime used by sherpa-onnx\n")
    for label, value in facts.items():
        print(f"  {label:<{width}}  {value}")

    problems = report["problems"]
    notes = report["notes"]
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")

    if not problems:
        print("\nOK: the bindings and their ONNX Runtime are a matched set.")
        return 0

    print("\nFAIL: sherpa-onnx cannot decode with this runtime.")
    for problem in problems:
        print(f"  - {problem}")

    print("\nRemedies:")
    if report["bundled"] is None:
        print(f"  {REMEDIES['core missing']}\n")
    if report["foreign"]:
        print(f"  {REMEDIES['foreign']}\n")
    if sys.platform == "win32":
        print(f"  {REMEDIES['system']}\n")
    print("Re-run this script after each fix; it prints the runtime actually mapped.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
