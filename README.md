# python-whisper

Local video/audio transcription with a Streamlit UI and a CLI. Audio is demuxed and
decoded by **PyAV** (bundled FFmpeg libraries — no system `ffmpeg` to install) and
recognised by **sherpa-onnx** (ONNX Runtime). Runs on CPU out of the box; the same
code drives a CUDA GPU when one is available.

```
video/audio file ─▶ PyAV demux ─▶ 16 kHz mono float32 ─▶ Silero VAD ─▶ sherpa-onnx ─▶ segments ─▶ txt / sentences / srt / vtt / json
```

## Features

- **No external binaries.** PyAV ships FFmpeg as Python wheels, so `pip install` is
  the whole setup on Linux, macOS and Windows.
- **Two model families wired up.** NVIDIA Parakeet TDT 0.6B v2 (best English
  accuracy, native punctuation + word timestamps) and Whisper base.en (smaller,
  useful baseline).
- **Speaker-aware subtitle timing.** VAD splits on silence so each utterance is
  decoded as a whole; subtitles are then re-split into readable cues at word
  boundaries, targeting a character budget and a max cue duration.
- **CPU or GPU.** One flag (`device="cuda"`). If the installed sherpa-onnx build
  has no CUDA provider, the app says so and falls back to CPU instead of crashing.
- **Multiple audio tracks.** Picks the container's default track; the UI lets you
  switch when a film carries commentary and 5.1 mixes.
- **Batch processing.** Several files in the UI, or a folder on the command line: one
  model load, sequential decoding, per-file status, a resumable queue
  (`--skip-existing`) and a single zip of every transcript.
- **Streaming decode.** Audio is processed in blocks, so a 2-hour film never sits
  in RAM as one giant array.

## Requirements

- Python 3.10 – 3.13 (tested on 3.13.5, Linux; `dataclass(slots=True)` sets the floor at 3.10)
- ~1.2 GB disk for models, ~2 GB RAM while transcribing with Parakeet
- No GPU required

## Install

**With uv** — `pyproject.toml` is the source of truth, `uv.lock` pins exact versions:

```bash
cd python-whisper
uv sync                                    # creates .venv, installs deps + dev tools
uv run python scripts/download_models.py   # ~1.1 GB, one time
uv run streamlit run app.py
```

**With plain pip and a venv:**

```bash
cd python-whisper
python3 -m venv .venv                      # Windows: py -3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # or: pip install -e .
python scripts/download_models.py          # ~1.1 GB, one time
.venv/bin/streamlit run app.py
```

Both routes install the `python-whisper` command (`uv run python-whisper …` or
`.venv/bin/python-whisper …`). A checkout also still works via `python cli.py …`,
which is a thin shim onto `whisper_pipeline/cli.py`.

`python scripts/check_runtime.py` prints which ONNX Runtime the install actually
uses — the one check worth running whenever sherpa-onnx refuses to load a model.

A built wheel contains the library and the CLI; the Streamlit UI (`app.py`) stays in
the checkout, where `uv run streamlit run app.py` picks up the same environment.

The downloader writes into `models/` and skips anything already present
(`--force` re-fetches, `--only vad` fetches a single asset).

## Models

| Key | Files | Quality | RTF (CPU, 2 threads) |
| --- | --- | --- | --- |
| `parakeet-tdt-0.6b-v2` | ~660 MB | Best English; punctuation, casing, digits | 0.07 |
| `whisper-base.en-int8` | ~160 MB | Lowercase, unpunctuated; more errors | 0.10 |
| `whisper-base.en-fp32` | ~225 MB | Same weights, full precision | 0.12 |
| `silero_vad.onnx` | ~2 MB | Segmentation only | — |

RTF = seconds of compute per second of audio, measured on the same 60 s clip of
spoken English on a 2-core slice of an AMD Ryzen 7 5800X. Lower is better: 0.07
means a 1-hour recording transcribes in about 4 minutes.

End-to-end check on the bundled 27 min 34 s lecture video (Parakeet, CPU, 2
threads): 119.7 s wall time, RTF 0.07, 319 utterances, 3 330 words, 419 subtitle
cues — none over the 84-character budget, no overlaps, no empty or duplicate cues.

Same clip, both models, for calibration:

```
Parakeet : "Hey students, Teacher Elric here. … apply that knowledge at the word and phrasal levels …"
Whisper  : "hey students, teacher albert here. … apply that knowledge at the word and freezer levels …"
```

Both models miss the odd domain term ("Grammar MCQ" comes back as "grammar and
CQ" in one place in the full lecture); Parakeet keeps casing and punctuation, which
matters for readable subtitles.

List what is installed with `python cli.py --list-models`.

Any other sherpa-onnx model directory dropped into `models/` can be loaded with
`whisper_pipeline.models.spec_from_directory()` helper (auto-detects transducer,
Whisper, Paraformer and NeMo-CTC layouts), which is how you would add multilingual
Whisper.

## Web UI

```bash
.venv/bin/streamlit run app.py          # then open http://localhost:8501
```

- Pick a model, device and thread count in the sidebar.
- Upload one file or several (the uploader takes a multi-selection), or drop a list of
  paths into the `Local paths` box — one per line, directories welcome. More than one
  file switches the page into batch mode: a queue table, one progress bar for the whole
  queue, then batch totals, a per-file status table and a **zip** with every transcript.
- Review the transcript, the segment table, the generated subtitles and the JSON
  payload; download any of them. The **Sentences .txt** button writes
  `NAME_transcript_sentences.txt`: the VAD cuts merged back into whole sentences.
- In a batch, `Inspect one transcript` re-opens that single-file view — same tabs, same
  downloads — for any file that succeeded. A failed file is reported in the table and
  never stops the queue.

The model is cached per (model, device, threads) with `max_entries=1`, so switching
models frees the previous one instead of holding every model in RAM.

## CLI

```bash
# one file, subtitles + text next to the source
python-whisper lecture.mp4                       # or: uv run python-whisper lecture.mp4

# a directory is expanded recursively to media files
python-whisper talks/ -f txt,sentences -o transcripts

# explicit list, all formats, explicit model
python-whisper talks/*.mkv -m parakeet-tdt-0.6b-v2 \
    -f txt,sentences,srt,vtt,json -o transcripts --threads 8

# resume a queue: skip anything whose requested outputs already exist
python-whisper talks/ --skip-existing

# a specific audio track, no VAD
python-whisper film.mkv --stream 2 --no-vad
```

Several inputs run as a **batch**: one model load, files decoded one after another, and
one summary line at the end (`3 ok · 0 skipped · 1 failed · 12.4 min audio in 95 s
(RTF 0.13)`). A corrupt file is reported and skipped — it never ends the queue — and the
exit code is `1` when anything failed. Batch work is sequential on purpose: the model is
up to 764 MB resident and sherpa-onnx already saturates the intra-op threads, so a second
worker would double memory to buy throughput the CPU cannot use.

| Flag | Purpose |
| --- | --- |
| `-m/--model` | model key (default: best installed) |
| `-d/--device` | `cpu` (default) or `cuda` |
| `-t/--threads` | CPU threads, `0` = all cores |
| `-f/--formats` | any of `txt,sentences,srt,vtt,json` |
| `-o/--outdir` | output directory (default: beside each input) |
| `--skip-existing` | skip files whose requested outputs are already on disk |
| `--stream` | audio track index to decode |
| `--no-vad` | blind 30 s windows instead of silence-aware segmentation |
| `--max-segment` | hard ceiling per decoded chunk (default 20 s) |
| `--max-chars` / `--max-cue-seconds` | subtitle line budget / cue duration ceiling |
| `--list-models` | show the installed model catalogue |

## GPU setup

The wheels at `k2-fsa.github.io` are named after the ONNX Runtime they bundle:

```
sherpa_onnx-1.13.8+cuda12.cudnn9.onnxruntime1.28.2-cp313-cp313-linux_x86_64.whl
                 └─ CUDA 12 ┘└ cuDNN 9 ┘└── ORT 1.28.2 ──┘
```

The **`+cuda` wheels with no `onnxruntimeX.Y.Z` in the name** expect an externally
installed `onnxruntime[-gpu]` and link against whatever it provides — that is the
usual cause of

```
The requested API version [28] is not available, only API versions [1, 17] are supported
in this build. Current ORT Version is: 1.17.1
```

The bindings are compiled against one ORT and load a different one. Install the tagged
wheel instead; it ships its own runtime, so nothing else may provide one.

**uv, ad hoc** (redo it after any `uv sync`, which restores the CPU engine):

```bash
uv pip uninstall sherpa-onnx sherpa-onnx-core onnxruntime onnxruntime-gpu
uv pip install "https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.13.8/sherpa_onnx-1.13.8+cuda12.cudnn9.onnxruntime1.28.2-cp313-cp313-linux_x86_64.whl"
```

**uv, declarative** (survives `uv sync`) — add to `pyproject.toml`, then `uv lock && uv sync`:

```toml
[tool.uv.sources]
sherpa-onnx = { url = "https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/1.13.8/sherpa_onnx-1.13.8+cuda12.cudnn9.onnxruntime1.28.2-cp313-cp313-linux_x86_64.whl" }
```

**pip:** see `requirements-gpu.txt`.

Then verify — both lines must agree, `1.28.2` here is the tag in the filename:

```bash
uv run python -c "import sherpa_onnx as so; \
  from whisper_pipeline.models import available_providers as p; print(p(), so.onnxruntime_version)"
# ('cpu', 'cuda') 1.28.2      ← good
# ('cpu',) 1.28.2             ← CPU wheel: no CUDA provider, still consistent
# ('cpu',) 1.17.1             ← mismatched set, this is the error above
```

Pick the wheel whose `cp3XX` tag matches your interpreter and whose `cudaXX` matches the
driver/CUDA runtime you have; nothing else in the environment should supply
`libonnxruntime.so`.

Then pick **GPU (CUDA)** in the sidebar, or pass `-d cuda` on the CLI.

Notes:

- CUDA wheels are built for Linux, and need a matching CUDA runtime + cuDNN (e.g. an
  `nvidia/cuda:12.x` base image). Check the `k2-fsa` index for the Python versions
  currently published before committing to this route.
- `sherpa-onnx-core` (the PyPI package that carries the CPU `libonnxruntime.so`) must be
  removed for the CUDA wheel — otherwise both distributions ship a file at the same
  path in `sherpa_onnx/lib/`.
- `supports_gpu()` checks for the `libonnxruntime_providers_cuda` shared object
  rather than asking ONNX Runtime, because requesting `provider="cuda"` from a CPU
  build can abort the process instead of raising.
- If a GPU load fails, the transcriber retries on CPU and reports why
  (`result.device_note` in the UI, a `note:` line in the CLI).

## Output formats

| Format | Shape |
| --- | --- |
| `txt` | Plain transcript, one line per utterance |
| `sentences` | Prose: VAD cuts merged back into whole sentences, `NAME_transcript_sentences.txt` |
| `srt` | Numbered cues, `HH:MM:SS,mmm`, sentence-aware splitting |
| `vtt` | Same cues, `HH:MM:SS.mmm`, `WEBVTT` header |
| `json` | Full fidelity: media metadata, provider, timings, segments and word timestamps |

Cue splitting works on word timestamps when the model provides them: cues break at
sentence ends first, then at the character budget (default 84), never splitting a
word. Models without word timings fall back to proportional timing.

The `sentences` export is the prose counterpart of `txt`: segments are discarded as
timeline artefacts and the timed words are regrouped into sentences, so a sentence
the VAD cut in half comes out whole. A sentence ends at terminal punctuation, at a
pause longer than 1.2 s, or — for models that emit no punctuation — at 45 words /
25 s. Parakeet transcribes in a lower case alphabet, so sentence-initial capitals
and first-person `I` are restored; pauses over 2 s become paragraph breaks.

## The pipeline

| Module | Responsibility |
| --- | --- |
| `whisper_pipeline/audio.py` | PyAV probing, block-wise decode + resample to 16 kHz mono |
| `whisper_pipeline/models.py` | Declarative model catalogue, disk introspection, provider detection |
| `whisper_pipeline/engine.py` | `Transcriber` (model + lock), Silero VAD segmenter, fixed-window segmenter, word timing |
| `whisper_pipeline/ort_runtime.py` | Locates `sherpa_onnx/lib`, preloads the bundled ONNX Runtime on Windows, reports which one is loaded |
| `whisper_pipeline/pipeline.py` | Orchestration: decode → segment → recognise → `TranscriptionResult` |
| `whisper_pipeline/subtitles.py` | SRT/VTT/TXT/JSON rendering and cue splitting |
| `app.py` / `cli.py` | Streamlit front end / command line front end |

`Transcriber.transcribe()` takes a lock, because Streamlit serves sessions from
multiple threads and a single recogniser cannot be used concurrently.

## Tuning

- **VAD threshold** (0.1 – 0.9, default 0.5): raise it for noisy recordings, lower
  it when quiet speech is being dropped.
- **Min silence** (default 0.5 s): raise it to merge short pauses into one cue,
  lower it to split faster dialogue.
- **Min speech** (default 0.1 s): raise it to drop clicks, breaths and other
  sub-100 ms blips before they reach the decoder, lower it to keep clipped
  one-word interjections.
- **Max segment** (default 20 s): hard ceiling per decode call.
- **Threads**: sherpa-onnx scales nearly linearly until physical cores are
  exhausted; hyperthreads add little.

## Troubleshooting

Start with `python scripts/check_runtime.py`: it prints the bindings version, the
runtime bundled with `sherpa-onnx-core`, the runtime the process has actually
mapped, and the ORT version in use.

| Symptom | Cause / fix |
| --- | --- |
| `No models found in models/` | Run `python scripts/download_models.py` |
| GPU option missing, providers `('cpu',)` | CPU-only wheel — see *GPU setup* |
| `GPU was requested but … running on CPU` | CUDA provider present but failed to initialise (driver/cuDNN mismatch) |
| `The requested API version [28] is not available, only [1, 17] …` (Linux) | Two engine releases in one venv: bindings want ORT 1.28.2, the loaded `libonnxruntime.so` is 1.17.1. Install one tagged CUDA wheel and remove `onnxruntime[-gpu]` and `sherpa-onnx-core` — see *GPU setup* |
| Same message on Windows, `Current ORT Version is: 1.17.1` | `sherpa_onnx/lib/onnxruntime.dll` is missing (`sherpa-onnx-core` absent) or `C:\Windows\System32\onnxruntime.dll` (Windows ML) / an `onnxruntime` package was loaded first. The app maps the bundled DLL at import, so nothing else in the process may import ONNX Runtime earlier. Install `sherpa-onnx-core==<same version as sherpa-onnx>`, `pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml`, then re-run the check script |
| `ImportError: libonnxruntime.so: cannot open shared object file` (Linux) or `ImportError: DLL load failed while importing _sherpa_onnx` (Windows) | `sherpa-onnx-core` missing or pruned — `pip install sherpa-onnx-core==<same version as sherpa-onnx>` |
| Garbled or missing words at chunk edges | VAD disabled — remove `--no-vad` |
| Much slower than RTF 0.1 | Other work on the box, or `--threads` set to 1 |

## Limitations

- Parakeet TDT 0.6B v2 is **English only**; the catalogue ships no multilingual
  model. Whisper base.en is English too — add a multilingual model directory if you
  need one.
- No speaker diarisation; segments are utterances, not speakers.
- Word timestamps come from the model. Whisper in sherpa-onnx returns
  cross-attention-aligned timings, which are coarser than Parakeet's transducer
  alignment.
- Licensing is yours to check: sherpa-onnx is Apache-2.0, Parakeet weights are
  CC-BY-4.0, Whisper weights are MIT, PyAV's bundled FFmpeg is LGPL/GPL depending
  on the wheel build.

## Layout

```
python-whisper/
├── pyproject.toml             project metadata, deps, `python-whisper` entry point
├── uv.lock                    pinned resolution (uv sync)
├── app.py                     Streamlit UI
├── cli.py                     shim → whisper_pipeline/cli.py
├── whisper_pipeline/          the library + CLI
│   ├── audio.py               PyAV probe + 16 kHz decode
│   ├── batch.py               queue: expand inputs, transcribe, collect outcomes
│   ├── engine.py              sherpa-onnx transcriber, VAD, chunker
│   ├── export.py              format naming, file writing, zip bundle
│   ├── models.py              model catalogue, download metadata, provider probe
│   ├── prose.py               sentence rebuilding (VAD cuts -> prose)
│   ├── subtitles.py           cue building, srt/vtt/txt/json
│   └── types.py               MediaInfo, Segment, Word
├── scripts/download_models.py model fetcher (stdlib only)
├── models/                    downloaded weights (git-ignored)
├── out/                       sample output (git-ignored)
├── requirements.txt           CPU install (plain pip)
└── requirements-gpu.txt       CUDA override (plain pip)
```
