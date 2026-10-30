"""Streamlit front end for the local transcription pipeline.

Run with:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from whisper_pipeline import (
    AudioExtractionError,
    MediaInfo,
    ModelLoadError,
    Transcriber,
    TranscriptionResult,
    VadOptions,
    available_providers,
    models,
    probe,
    transcribe_file,
)
from whisper_pipeline.models import supports_gpu

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "python-whisper-uploads"

SUPPORTED_TYPES = [
    "mp4",
    "mkv",
    "mov",
    "webm",
    "avi",
    "ts",
    "m4a",
    "mp3",
    "wav",
    "flac",
    "ogg",
    "opus",
    "aac",
]

st.set_page_config(
    page_title="Local Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources and small helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(max_entries=1, show_spinner=False)
def load_transcriber(model_key: str, device: str, threads: int) -> Transcriber:
    """Load a model once per (model, device, threads) combination.

    ``max_entries=1`` is deliberate: the largest model here is 764 MB resident and
    the default cache would keep every model the user ever selected alive at once.
    """
    spec = models.get_model(model_key)
    return Transcriber(spec, device=device, num_threads=threads)


def sample_media() -> Path | None:
    """Find the bundled test clip in the parent workspace, if present."""
    candidates = sorted(WORKSPACE_DIR.glob("*.mp4"))
    return candidates[0] if candidates else None


def save_upload(uploaded) -> Path:
    """Persist an upload so PyAV can seek within it."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / Path(uploaded.name).name
    if not target.exists() or target.stat().st_size != uploaded.size:
        target.write_bytes(uploaded.getbuffer())
    return target


def human_size(num_bytes: int) -> str:
    if num_bytes < 1_000_000:
        return f"{num_bytes / 1e3:.0f} kB"
    if num_bytes < 1_000_000_000:
        return f"{num_bytes / 1e6:.1f} MB"
    return f"{num_bytes / 1e9:.2f} GB"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def build_sidebar() -> dict:
    st.sidebar.title("🎙️ Local Transcriber")
    st.sidebar.caption("sherpa-onnx + PyAV · no ffmpeg binary, no cloud calls")

    installed = models.available_models()
    if not installed:
        st.sidebar.error("No models found in `models/` — run `scripts/download_models.py`.")
        st.stop()

    labels = {spec.key: f"{spec.label} — {spec.describe()}" for spec in installed}
    model_key = st.sidebar.selectbox(
        "Model",
        options=list(labels),
        format_func=lambda key: labels[key],
        help="Parakeet leads on English accuracy; Whisper base.en is a useful baseline.",
    )
    st.sidebar.caption(models.get_model(model_key).description)

    gpu_ready = supports_gpu()
    device = st.sidebar.radio(
        "Compute",
        options=["cpu"] + (["cuda"] if gpu_ready else []),
        format_func=lambda value: "CPU" if value == "cpu" else "GPU (CUDA)",
        horizontal=True,
    )
    if not gpu_ready:
        st.sidebar.caption(
            f"GPU unavailable in this build (providers: {', '.join(available_providers())}). "
            "The GPU code path is implemented; it needs a CUDA-enabled sherpa-onnx "
            "build — see README › GPU setup."
        )

    max_threads = max(1, os.cpu_count() or 2)
    threads = st.sidebar.slider(
        "CPU threads",
        1,
        max_threads,
        min(2, max_threads),
        help="sherpa-onnx intra-op threads. Scales roughly linearly until you "
        "exhaust physical cores.",
    )

    st.sidebar.divider()
    use_vad = st.sidebar.toggle(
        "Voice activity detection",
        value=True,
        help="Cuts on silence so each utterance is decoded whole. Turning this off "
        "uses blind fixed windows, which clips words at window edges.",
    )

    with st.sidebar.expander("VAD tuning", expanded=False):
        threshold = st.slider(
            "Speech threshold",
            0.1,
            0.9,
            0.5,
            0.05,
            help="Higher is stricter: fewer false positives on noise.",
        )
        min_speech = st.slider("Min speech (s)", 0.1, 2.0, 0.1, 0.05)
        min_silence = st.slider("Min silence (s)", 0.1, 2.0, 0.5, 0.05)
        max_speech = st.slider(
            "Max segment (s)", 5.0, 30.0, 20.0, 1.0, help="Hard ceiling per decoded chunk."
        )

    with st.sidebar.expander("Subtitle output", expanded=False):
        max_chars = st.slider("Max characters per cue", 40, 120, 84, 4)
        max_cue_seconds = st.slider("Max cue duration (s)", 2.0, 12.0, 6.0, 0.5)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Providers: `{', '.join(available_providers())}`  \nModels on disk: `{len(installed)}`"
    )

    return {
        "model_key": model_key,
        "device": device,
        "threads": threads,
        "vad": (
            VadOptions(
                model_path=models.vad_model_path(),
                threshold=threshold,
                min_speech_duration=min_speech,
                min_silence_duration=min_silence,
                max_speech_duration=max_speech,
            )
            if use_vad
            else None
        ),
        "max_chars": max_chars,
        "max_cue_seconds": max_cue_seconds,
    }


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def pick_media() -> Path | None:
    """Resolve the file to transcribe from an upload or a server-side path."""
    upload_tab, path_tab = st.tabs(["Upload file", "Local path"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Drop a video or audio file",
            type=SUPPORTED_TYPES,
            help="Any container PyAV can decode: mp4, mkv, webm, m4a, mp3, wav…",
        )
    with path_tab:
        example = sample_media()
        typed = st.text_input(
            "Path on this machine",
            value=str(example) if example else "",
            placeholder="/path/to/recording.mp4",
        )
        if example is not None:
            st.caption(
                f"Pre-filled with a sample clip found in the workspace `{example.name}` — "
                "clear the field to start from an upload instead."
            )

    if uploaded is not None:
        return save_upload(uploaded)
    if typed.strip():
        candidate = Path(typed.strip()).expanduser()
        if not candidate.is_file():
            st.error(f"No such file: {candidate}")
            return None
        return candidate
    return None


def show_media_summary(path: Path) -> MediaInfo | None:
    """Probe the container and render a compact summary."""
    try:
        info = probe(path)
    except AudioExtractionError as exc:
        st.error(f"Cannot read this file: {exc}")
        return None

    if not info.audio_streams:
        st.error("This container has no audio stream.")
        return None

    columns = st.columns(5)
    columns[0].metric("Container", info.container_format.split(",")[0])
    columns[1].metric("Duration", info.duration_label)
    columns[2].metric("Audio tracks", len(info.audio_streams))
    columns[3].metric("Size", human_size(info.size_bytes))
    columns[4].metric("Video", "yes" if info.video_streams else "no")

    st.caption(f"Default track: {info.stream().label}")
    return info


def stream_selector(info: MediaInfo) -> int:
    """Choose an audio track when the container carries several."""
    default = info.default_stream_index()
    if len(info.audio_streams) < 2:
        return default

    labels = {}
    for stream in info.audio_streams:
        language = f" · {stream.language}" if stream.language not in (None, "und") else ""
        labels[stream.index] = f"{stream.label}{language}"

    options = list(labels)
    return st.selectbox(
        "Audio track",
        options=options,
        index=options.index(default),
        format_func=lambda index: labels[index],
        help="Films often carry commentary and 5.1 mixes alongside the main track.",
    )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def show_results(result: TranscriptionResult, max_chars: int, max_cue_seconds: float) -> None:
    st.divider()
    metrics = st.columns(5)
    metrics[0].metric("Audio", f"{result.audio_seconds / 60:.1f} min")
    metrics[1].metric("Elapsed", f"{result.wall_seconds:.1f} s")
    metrics[2].metric(
        "Real-time factor",
        f"{result.rtf:.2f}×",
        help="Seconds of compute per second of audio. Lower is better.",
    )
    metrics[3].metric("Segments", result.segment_count)
    metrics[4].metric("Words", result.word_count)

    if result.device_note:
        st.warning(result.device_note)

    srt = result.to_srt(max_chars=max_chars, max_duration=max_cue_seconds)
    vtt = result.to_vtt(max_chars=max_chars, max_duration=max_cue_seconds)
    txt = result.to_txt()
    sentences_txt = result.prose
    payload = result.to_json()

    tab_text, tab_segments, tab_srt, tab_json = st.tabs(
        ["Transcript", "Segments", "Subtitles", "JSON"]
    )
    with tab_text:
        st.text_area("Plain text", txt, height=380, label_visibility="collapsed")
    with tab_segments:
        st.dataframe(
            [
                {
                    "#": segment.index,
                    "start": segment.start,
                    "end": segment.end,
                    "secs": round(segment.duration, 2),
                    "words": segment.word_count,
                    "text": segment.text,
                }
                for segment in result.segments
            ],
            hide_index=True,
        )
    with tab_srt:
        st.code(srt, language=None)
    with tab_json:
        st.json(json.loads(payload))

    stem = Path(result.media.path).stem
    downloads = st.columns(5)
    downloads[0].download_button(
        "⬇ Transcript .txt", txt, file_name=f"{stem}.txt", mime="text/plain"
    )
    downloads[1].download_button(
        "⬇ Sentences .txt",
        sentences_txt,
        file_name=f"{stem}_transcript_sentences.txt",
        mime="text/plain",
    )
    downloads[2].download_button(
        "⬇ Subtitles .srt", srt, file_name=f"{stem}.srt", mime="text/plain"
    )
    downloads[3].download_button("⬇ Subtitles .vtt", vtt, file_name=f"{stem}.vtt", mime="text/vtt")
    downloads[4].download_button(
        "⬇ Transcript .json", payload, file_name=f"{stem}.json", mime="application/json"
    )

    with st.expander("Run details"):
        st.json(result.metadata())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    settings = build_sidebar()
    st.title("Transcribe video and audio locally")

    path = pick_media()
    if path is None:
        st.info(
            "Pick a file to begin. Everything runs on this machine — audio is decoded "
            "by PyAV (bundled FFmpeg libraries, no binary to install) and recognised "
            "by sherpa-onnx."
        )
        return

    info = show_media_summary(path)
    if info is None:
        return

    stream_index = stream_selector(info)

    if st.button("Transcribe", type="primary"):
        bar = st.progress(0.0, text="Loading model…")
        live = st.empty()

        try:
            with st.spinner("Loading model into memory…"):
                transcriber = load_transcriber(
                    settings["model_key"], settings["device"], settings["threads"]
                )
            if transcriber.fallback_note:
                st.warning(transcriber.fallback_note)

            def on_progress(fraction: float, message: str) -> None:
                bar.progress(min(1.0, fraction), text=message)

            seen = {"count": 0}

            def on_segment(segment) -> None:
                seen["count"] += 1
                snippet = segment.text[:70] + ("…" if len(segment.text) > 70 else "")
                live.caption(f"Recognised {seen['count']} segments · latest: “{snippet}”")

            result = transcribe_file(
                path,
                transcriber,
                vad=settings["vad"],
                stream_index=stream_index,
                on_progress=on_progress,
                on_segment=on_segment,
            )
        except (ModelLoadError, AudioExtractionError) as exc:
            bar.empty()
            st.error(str(exc))
            return

        bar.progress(1.0, text="Done")
        live.empty()
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if isinstance(result, TranscriptionResult) and Path(result.media.path) == path:
        show_results(result, settings["max_chars"], settings["max_cue_seconds"])


if __name__ == "__main__":
    main()
