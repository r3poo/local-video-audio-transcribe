"""Model catalogue and runtime capability detection.

A model is described declaratively so the UI, the CLI and the pipeline all agree
on what exists on disk, what is missing, and how to load it.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Model families map onto the factory methods of ``sherpa_onnx.OfflineRecognizer``.
FAMILY_TRANSDUCER = "transducer"
FAMILY_WHISPER = "whisper"
FAMILY_PARAFORMER = "paraformer"
FAMILY_NEMO_CTC = "nemo_ctc"


def model_root() -> Path:
    """Directory holding all local models (``<project>/models``)."""
    return Path(__file__).resolve().parent.parent / "models"


def vad_model_path() -> Path:
    """Path to the Silero VAD ONNX model."""
    return model_root() / "silero_vad.onnx"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A locally installed speech-recognition model."""

    key: str
    label: str
    family: str
    directory: Path
    description: str
    languages: str = "English"
    size_label: str = ""
    model_type: str = ""
    language: str = ""
    task: str = "transcribe"
    feature_dim: int = 80
    encoder_name: str = "encoder.int8.onnx"
    decoder_name: str = "decoder.int8.onnx"
    joiner_name: str = ""
    tokens_name: str = "tokens.txt"

    @property
    def encoder(self) -> Path:
        return self.directory / self.encoder_name

    @property
    def decoder(self) -> Path:
        return self.directory / self.decoder_name

    @property
    def joiner(self) -> Path | None:
        return self.directory / self.joiner_name if self.joiner_name else None

    @property
    def tokens(self) -> Path:
        return self.directory / self.tokens_name

    @property
    def required_files(self) -> tuple[Path, ...]:
        files = [self.encoder, self.decoder, self.tokens]
        if self.joiner is not None:
            files.append(self.joiner)
        return tuple(files)

    def missing_files(self) -> list[str]:
        return [f.name for f in self.required_files if not f.is_file()]

    def is_available(self) -> bool:
        return not self.missing_files()

    def describe(self) -> str:
        bits = [self.languages]
        if self.size_label:
            bits.append(self.size_label)
        return " · ".join(bits)


CATALOG: dict[str, ModelSpec] = {
    "parakeet-tdt-0.6b-v2": ModelSpec(
        key="parakeet-tdt-0.6b-v2",
        label="Parakeet TDT 0.6B v2 (int8)",
        family=FAMILY_TRANSDUCER,
        directory=model_root() / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        description=(
            "NVIDIA NeMo transducer. Best English accuracy available here, with native "
            "punctuation, capitalisation and word timestamps."
        ),
        languages="English",
        size_label="~662 MB",
        model_type="nemo_transducer",
        joiner_name="joiner.int8.onnx",
    ),
}


def available_models() -> list[ModelSpec]:
    """Catalogue entries whose files are all present, best-first."""
    return [spec for spec in CATALOG.values() if spec.is_available()]


def get_model(key: str) -> ModelSpec:
    """Look up a catalogue entry by key."""
    try:
        return CATALOG[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; available: {sorted(CATALOG)}") from None


def default_model() -> ModelSpec:
    """The best model that is actually installed."""
    available = available_models()
    if not available:
        raise FileNotFoundError(
            f"no models found under {model_root()} — download one in the app's settings, "
            "or run scripts/download_models.py"
        )
    return available[0]


def spec_from_directory(directory: str | Path, key: str | None = None) -> ModelSpec:
    """Build a spec for any sherpa-onnx model directory by sniffing file names.

    This is the escape hatch for multilingual Whisper, GigaAM, SenseVoice, and
    anything else a user drops into ``models/`` without touching this catalogue.
    """
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)

    onnx = sorted(path.glob("*.onnx"))
    if not onnx:
        raise FileNotFoundError(f"no .onnx files in {path}")

    def pick(*needles: str) -> str:
        for needle in needles:
            for candidate in onnx:
                if needle in candidate.name.lower():
                    return candidate.name
        return ""

    tokens = pick("tokens.txt")
    if not tokens:
        token_candidates = sorted(path.glob("*tokens.txt"))
        if not token_candidates:
            raise FileNotFoundError(f"no tokens.txt in {path}")
        tokens = token_candidates[0].name

    encoder = pick("encoder")
    decoder = pick("decoder")
    joiner = pick("joiner")
    name = path.name.lower()

    if joiner:
        family = FAMILY_TRANSDUCER
        model_type = "nemo_transducer" if ("parakeet" in name or "nemo" in name) else ""
    elif encoder and decoder:
        family = FAMILY_WHISPER
        model_type = ""
    elif pick("model"):
        family = FAMILY_PARAFORMER
        model_type = ""
    else:
        raise ValueError(
            f"cannot infer model family from files in {path}: {[f.name for f in onnx]}"
        )

    return ModelSpec(
        key=key or path.name,
        label=path.name,
        family=family,
        directory=path,
        description="Custom model directory",
        languages="auto" if family == FAMILY_WHISPER else "unknown",
        size_label="",
        model_type=model_type,
        language="" if family == FAMILY_WHISPER else "",
        encoder_name=encoder,
        decoder_name=decoder,
        joiner_name=joiner,
        tokens_name=tokens,
    )


@functools.lru_cache(maxsize=1)
def available_providers() -> tuple[str, ...]:
    """Execution providers this sherpa-onnx build can actually serve.

    pip wheels are CPU-only builds. A GPU build (source-compiled with
    ``SHERPA_ONNX_ENABLE_GPU=ON``) ships ``libonnxruntime_providers_cuda``; that
    file is the honest signal, because asking a CPU build for ``provider="cuda"``
    can abort the process rather than raise.
    """
    import sherpa_onnx

    providers = ["cpu"]
    package_root = Path(sherpa_onnx.__file__).resolve().parent
    patterns = {
        "cuda": ("*providers_cuda*",),
        "tensorrt": ("*providers_tensorrt*",),
        "coreml": ("*providers_coreml*", "*coreml*"),
    }
    for provider, globs in patterns.items():
        for pattern in globs:
            try:
                next(package_root.rglob(pattern))
            except StopIteration:
                continue
            providers.append(provider)
            break
    log.debug("sherpa-onnx providers: %s", providers)
    return tuple(providers)


def supports_gpu() -> bool:
    """True when this sherpa-onnx build has a GPU execution provider."""
    return any(p != "cpu" for p in available_providers())
