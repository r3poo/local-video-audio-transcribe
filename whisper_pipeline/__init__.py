"""Local, CPU-friendly transcription pipeline.

Public surface:

    from whisper_pipeline import (
        TranscriptionResult, Transcriber, VadOptions, transcribe_file, probe,
    )
"""

from .audio import (
    AudioExtractionError,
    AudioStreamInfo,
    MediaInfo,
    decode_audio,
    iter_blocks,
    probe,
)
from .engine import ModelLoadError, Transcriber, VadOptions
from .models import (
    CATALOG,
    ModelSpec,
    available_models,
    available_providers,
    default_model,
    get_model,
    model_root,
    supports_gpu,
)
from .pipeline import TranscriptionResult, transcribe_file
from .subtitles import to_json, to_srt, to_txt, to_vtt
from .types import Segment, Word

__all__ = [
    "AudioExtractionError",
    "AudioStreamInfo",
    "CATALOG",
    "MediaInfo",
    "ModelLoadError",
    "ModelSpec",
    "Segment",
    "Transcriber",
    "TranscriptionResult",
    "VadOptions",
    "Word",
    "available_models",
    "available_providers",
    "decode_audio",
    "default_model",
    "get_model",
    "iter_blocks",
    "model_root",
    "probe",
    "supports_gpu",
    "to_json",
    "to_srt",
    "to_txt",
    "to_vtt",
    "transcribe_file",
]

__version__ = "1.0.0"
