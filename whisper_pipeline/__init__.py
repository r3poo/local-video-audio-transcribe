"""Local, CPU-friendly transcription pipeline.

Public surface:

    from whisper_pipeline import (
        TranscriptionResult, Transcriber, VadOptions, transcribe_file, probe,
    )
"""

from . import fetch
from .audio import (
    AudioExtractionError,
    AudioStreamInfo,
    MediaInfo,
    decode_audio,
    iter_blocks,
    probe,
)
from .batch import (
    MEDIA_EXTENSIONS,
    BatchItem,
    BatchOutcome,
    clean_path,
    expand_inputs,
    parse_path_text,
    run_batch,
)
from .engine import ModelLoadError, Transcriber, VadOptions
from .export import DEFAULT_FORMATS, FORMATS, bundle_zip, write_outputs
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
from .prose import Sentence, build_sentences, to_prose
from .subtitles import to_json, to_srt, to_txt, to_vtt
from .types import Segment, Word

__all__ = [
    "AudioExtractionError",
    "AudioStreamInfo",
    "BatchItem",
    "BatchOutcome",
    "CATALOG",
    "DEFAULT_FORMATS",
    "FORMATS",
    "MEDIA_EXTENSIONS",
    "MediaInfo",
    "ModelLoadError",
    "ModelSpec",
    "Segment",
    "Sentence",
    "Transcriber",
    "TranscriptionResult",
    "VadOptions",
    "Word",
    "available_models",
    "available_providers",
    "build_sentences",
    "bundle_zip",
    "clean_path",
    "decode_audio",
    "default_model",
    "expand_inputs",
    "fetch",
    "get_model",
    "iter_blocks",
    "model_root",
    "parse_path_text",
    "probe",
    "run_batch",
    "supports_gpu",
    "to_json",
    "to_prose",
    "to_srt",
    "to_txt",
    "to_vtt",
    "transcribe_file",
    "write_outputs",
]

__version__ = "1.0.0"
