"""Entry point for the local web UI.

The UI lives in :mod:`whisper_pipeline.web` (stdlib ``http.server``, no
Streamlit). This shim keeps ``python app.py`` working, and is what the portable
bundle's launcher calls.

    python app.py               # http://127.0.0.1:8501, opens a browser
    python app.py --no-browser --port 9000
"""

from whisper_pipeline.web import main

if __name__ == "__main__":
    raise SystemExit(main())
