#!/usr/bin/env python3
"""Backwards-compatible entry point.

The CLI itself lives in ``whisper_pipeline/cli.py`` so that ``pyproject.toml`` can
expose it as the ``python-whisper`` command. This shim keeps the documented
``python cli.py …`` invocation working from a checkout.
"""

from __future__ import annotations

import sys

from whisper_pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
