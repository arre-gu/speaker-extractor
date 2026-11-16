# Audio speaker extractor

This is a standalone uv project using Python 3.12 and WhisperX 3.8.6. It has its
own `pyproject.toml`, `uv.lock`, tests, and `.venv`, with no imports or dependencies
on the parent training project. The entire directory can be used independently.
The lock pins transitive dependencies as well as the direct dependencies.

Run all commands below from this directory in PowerShell:

```powershell
cd audio-speaker-extractor
```

## Setup

Install [FFmpeg](https://ffmpeg.org/download.html), including its shared libraries,
and put its `bin` directory on `PATH` before starting Python. Verify with
`ffmpeg -version`. TorchCodec needs the shared FFmpeg libraries, not only a static
`ffmpeg.exe`; see [TorchCodec's installation guidance](https://github.com/pytorch/torchcodec#installing-torchcodec).

`uv sync` installs Python dependencies, not the FFmpeg executable. If you see
`[WinError 2]` while loading every recording, verify `ffmpeg -version` in the
same terminal. After changing Windows PATH, restart the terminal or app that
launches the script. The extractor checks for FFmpeg before loading models.

For NVIDIA GPUs, install a compatible NVIDIA driver and the
[CUDA 12.8 toolkit](https://developer.nvidia.com/cuda-12-8-1-download-archive),
plus the cuDNN 9 libraries required by CTranslate2. Make their DLLs available on
`PATH`. See [WhisperX setup](https://github.com/m-bain/whisperX/tree/v3.8.6#setup-)
and [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu).
This tool's CUDA environment is separate from the training project's CUDA 13 setup.

```powershell
uv sync --extra cuda --locked
```

Alternatively, install the CPU environment:

```powershell
uv sync --extra cpu --locked
```

Accept the access terms for
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
using your Hugging Face account. Add a read token from that account to `.env`
beside `extract_main_speaker.py`:

```dotenv
HF_TOKEN=your_hugging_face_read_token
```

The script automatically loads this file regardless of the current working
directory. An existing `HF_TOKEN` environment variable takes precedence. `.env`
is ignored by Git; do not commit your token.

Alternatively, set the token in the current PowerShell session without including
it in shell history:

```powershell
$env:HF_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Hugging Face read token' -AsSecureString)).Password
```

The first run downloads ASR, VAD, alignment, and diarization models. Subsequent
runs use their normal library caches. Audio processing runs locally.

## Preview and export

Start with a few representative recordings and preview the export decisions:

```powershell
uv run --extra cuda --locked python extract_main_speaker.py "data/raw_recordings" "data/main_speaker" --dry-run
```

Dry-run performs full transcription, alignment, diarization, and filtering. It
prints every proposed filename, the selected speaker, candidate utterance count,
would-save clip count, total would-save duration, rejection counts, and batch
totals including failures. It creates no output directory, clips, or manifests.
It still takes full inference time and may download models into library caches.

Remove `--dry-run` to write clips and one `<AGE_GENDER_ID>_manifest.json` per
recording:

```powershell
uv run --extra cuda --locked python extract_main_speaker.py "data/raw_recordings" "data/main_speaker"
```

For CPU operation, replace `--extra cuda` with `--extra cpu` and append
`--device cpu`. Device `auto` uses CUDA when available and CPU otherwise;
compute type `auto` uses float16 on CUDA and int8 on CPU. It does not silently
retry on CPU after a CUDA/DLL error. Reduce `--batch-size` first if GPU memory
is insufficient, or choose a smaller `--model`.

Add `--recursive` for nested input directories. Language is detected separately
for each recording; pass a code such as `--language sv` or `--language en` to
override detection. Unsupported alignment languages stop the batch rather than
falling back to coarse transcript timestamps.

## Filenames and selection

For example:

```text
RespCam_Anonymous 29-04-26 16h18m_Scene camera 1_(40,FEMALE,1010).wav
    -> 40_FEMALE_1010_1.wav
    -> 40_FEMALE_1010_3.wav
```

Metadata must appear in a trailing `(AGE,GENDER,ID)` group. Spaces around fields
are allowed, gender is uppercased, and leading ID zeros are preserved. File
extensions are case-insensitive. X is the main speaker's chronological utterance
number in that recording, starting at 1 **before filtering**, so missing numbers
represent rejected utterances.

An utterance joins adjacent aligned words from the same speaker until a pause
exceeds 0.7 seconds, a speaker changes, another speaker intervenes, or an
untimed/unassigned word occurs. Selection counts these utterances before clip
filters. Ties use total utterance duration, earliest appearance, then speaker
label. Speaker labels are local to each recording. Untimed/unassigned words are
excluded and recorded in the manifest; silence produces zero clips.

All output clips share a flat directory. The script validates the complete batch
before loading models, rejecting malformed filenames, duplicate `AGE_GENDER_ID`
prefixes, and any existing output entry with a matching prefix. Dry-run applies
the same checks. Multiple recordings of the same metadata cannot be combined
in one invocation under this naming scheme. Use separate invocations with separate
output directories for those recordings. Clips are never overwritten.

Clips retain the original sample rate, channels, and level, using 32-bit PCM WAV.
WhisperX's resampled mono audio is used only for inference. Inputs exceeding the
PCM amplitude range are rejected rather than clipped. No denoising, separation,
normalization, or automatic long-utterance splitting is applied.

## Tuning the filters

These starting values preserve natural speech; calibrate them by listening to
your recordings. All durations are seconds.

| Flag | Default | Effect |
|---|---:|---|
| `--pause-threshold` | 0.7 | Longer gaps split utterances; also affects main-speaker counts. |
| `--min-duration` | 0.5 | Reject shorter unpadded utterances; lower for brief replies. |
| `--max-duration` | Off | Reject longer unpadded utterances without splitting them. |
| `--min-words` | 1 | Raise to exclude fragments, losing some valid replies. |
| `--padding` | 0.1 | Add boundary context; clamp to recording edges and other speakers. |
| `--vad-onset` | 0.5 | Higher requires stronger evidence of speech and may lose quiet speech. |
| `--vad-offset` | 0.363 | Controls speech endings; raising it may cut quiet word endings. |
| `--min-rms-dbfs` | Off | Reject low-level padded clips; e.g. `-45`. Not an SNR estimate. |
| `--min-speech-ratio` | Off | Minimum union of diarization speech intervals / padded clip duration, from 0 to 1. |
| `--min-speakers`, `--max-speakers` | Automatic | Constrain the diarization speaker count when known. |
| `--model` | `large-v3` | ASR accuracy/memory/speed tradeoff. |
| `--batch-size` | 4 | Lower to reduce GPU memory use. |
| `--device` | `auto` | `cuda` or `cpu` can be selected explicitly. |
| `--compute-type` | `auto` | Override with `float16`, `float32`, or `int8`. |

Duration limits exclude padding. RMS and speech coverage include padding. Speech
ratio uses diarization activity as a speech-coverage proxy, not ASR confidence.
RMS averages sample energy across channels. Silence is represented by a null RMS
dBFS value in JSON and fails an enabled RMS threshold. VAD uses Pyannote;
`--vad-offset` must not exceed `--vad-onset`.

Any detected other-speaker activity inside an utterance rejects the whole
utterance, including speech inside pauses between its words. Boundary padding
cannot extend into another detected speaker. This policy is mandatory.
Rejection counts can overlap because a clip can fail several filters.

Diarization is imperfect and can miss quiet background voices or confuse
speakers. It does not remove simultaneous voices, environmental noise, or music
from retained audio. Listen to clips from a small first batch before producing
the complete dataset. Strong RMS or word-count filters can bias the dataset
against quiet speech, hesitations, and unusual speech patterns.

## Failures, manifests, and verification

Each manifest records settings and resolved runtime, source, detected language,
speaker counts, all grouped utterances, nonexclusive diarization intervals,
excluded words, proposed clips, sample boundaries, quality measurements,
rejection reasons, and which clips were saved. Individual corrupt recordings
are reported and processing continues with exit status 1. Authentication and
model-loading failures stop the batch. A failed write can leave partial output;
inspect the manifest and use a fresh output directory when retrying.

Run focused synthetic tests and syntax checks without private audio:

```powershell
uv run --extra cpu --locked python -m pytest -q
uv run --extra cpu --locked python -m py_compile extract_main_speaker.py tests/test_extract_main_speaker.py
```

The tests mock inference. They validate extraction behavior and dry-run parity,
not the accuracy of WhisperX on your recordings.
