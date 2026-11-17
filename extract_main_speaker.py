"""Extract non-overlapping, pause-separated utterances from every speaker."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np
import soundfile as sf
from dotenv import load_dotenv


class BatchError(RuntimeError):
    """An error that makes further recordings unsafe or impossible to process."""


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    words: list[str]


def parse_prefix(path: Path) -> str:
    """Extract AGE_GENDER_ID without losing leading zeros in the ID."""
    match = re.search(
        r"\(\s*(\d+)\s*,\s*([A-Za-z]+)\s*,\s*(\d+)\s*\)$", path.stem
    )
    if match is None:
        raise ValueError(f"Missing trailing (AGE,GENDER,ID): {path.name}")
    age, gender, speaker_id = match.groups()
    return f"{int(age)}_{gender.upper()}_{speaker_id}"


def preflight(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """Validate the complete input set and output namespace before inference."""
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("Output path must be a directory")
    paths = args.input_dir.rglob("*") if args.recursive else args.input_dir.iterdir()
    files = sorted(
        (p for p in paths if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: (str(p).casefold(), str(p)),
    )
    if not files:
        raise ValueError("No WAV files found")
    existing = (
        [p.name.casefold() for p in args.output_dir.iterdir()]
        if args.output_dir.exists() else []
    )
    errors = []
    seen = {}
    recordings = []
    for path in files:
        try:
            prefix = parse_prefix(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if prefix in seen:
            errors.append(f"Duplicate metadata {prefix}: {seen[prefix]} and {path}")
        seen[prefix] = path
        if any(name.startswith(prefix.casefold() + "_") for name in existing):
            errors.append(f"Existing output uses prefix {prefix}")
        recordings.append((path, prefix))
    if errors:
        raise ValueError("Preflight failed:\n" + "\n".join(errors))
    return recordings


def coverage(start: float, end: float, turns: list[Turn]) -> float:
    """Measure the union of speech intervals inside a time range."""
    intervals = sorted(
        (max(start, t.start), min(end, t.end))
        for t in turns if t.start < end and t.end > start
    )
    total = 0.0
    cursor = start
    for left, right in intervals:
        total += max(0.0, right - max(cursor, left))
        cursor = max(cursor, right)
    return total


def build_utterances(
    words: list[dict[str, Any]], turns: list[Turn], pause_threshold: float
) -> tuple[list[Utterance], list[dict[str, Any]]]:
    """Group aligned words; never infer a speaker from a nearby nonoverlapping turn."""
    utterances = []
    excluded = []
    current = None
    for word in words:
        start, end = word.get("start"), word.get("end")
        text = word.get("word", "").strip()
        if (start is None or end is None or not text
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start):
            # Nonfinite timestamps must not make the JSON manifest invalid.
            excluded.append({
                "word": text, "reason": "untimed_or_invalid_word",
                "start": start if start is not None and math.isfinite(start) else None,
                "end": end if end is not None and math.isfinite(end) else None,
            })
            current = None
            continue
        speakers = {t.speaker for t in turns if t.start < end and t.end > start}
        if not speakers:
            excluded.append({"word": word, "reason": "unassigned_word"})
            current = None
            continue
        # Union coverage avoids double-counting duplicated turns. Ties are stable.
        speaker = min(speakers, key=lambda s: (
            -coverage(start, end, [t for t in turns if t.speaker == s]), s
        ))
        intervening = current is not None and any(
            t.speaker != speaker and t.start < start and t.end > current.end
            for t in turns
        )
        if (current is None or current.speaker != speaker or intervening
                or start - current.end > pause_threshold or start < current.start):
            current = Utterance(float(start), float(end), speaker, [text])
            utterances.append(current)
        else:
            current.end = max(current.end, float(end))
            current.words.append(text)
    return utterances, excluded


def plan_clips(
    utterances: list[Utterance], turns: list[Turn], prefix: str,
    audio: np.ndarray, sample_rate: int, args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Make all export decisions once, shared by real and dry runs."""
    decisions = []
    duration = len(audio) / sample_rate
    selected = sorted(utterances, key=lambda u: (u.start, u.end, u.speaker))
    for ordinal, utterance in enumerate(selected, 1):
        reasons = []
        others = [t for t in turns if t.speaker != utterance.speaker]
        if any(t.start < utterance.end and t.end > utterance.start for t in others):
            reasons.append("other_speaker_overlap")
        if utterance.start >= duration or utterance.end > duration:
            reasons.append("timestamp_out_of_bounds")
        start = max(0.0, utterance.start - args.padding)
        end = min(duration, utterance.end + args.padding)
        for turn in others:
            if turn.end <= utterance.start:
                start = max(start, turn.end)
            if turn.start >= utterance.end:
                end = min(end, turn.start)
        # Round inward, so sampling cannot introduce another speaker at a boundary.
        first = min(len(audio), max(0, math.ceil(start * sample_rate)))
        last = min(len(audio), max(first, math.floor(end * sample_rate)))
        clip_duration = (last - first) / sample_rate
        speech_duration = utterance.end - utterance.start
        if speech_duration < args.min_duration or last <= first:
            reasons.append("too_short")
        if args.max_duration is not None and speech_duration > args.max_duration:
            reasons.append("too_long")
        if len(utterance.words) < args.min_words:
            reasons.append("too_few_words")
        clip = audio[first:last]
        rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
        rms_dbfs = 20 * math.log10(rms) if rms > 0 else None
        speech_ratio = (
            coverage(first / sample_rate, last / sample_rate, turns) / clip_duration
            if clip_duration else 0.0
        )
        if args.min_rms_dbfs is not None and (
            rms_dbfs is None or rms_dbfs < args.min_rms_dbfs
        ):
            reasons.append("below_rms_threshold")
        if args.min_speech_ratio is not None and speech_ratio < args.min_speech_ratio:
            reasons.append("low_speech_ratio")
        decisions.append({
            "ordinal": ordinal, "filename": f"{prefix}_{ordinal}.wav",
            "speaker": utterance.speaker, "text": " ".join(utterance.words),
            "speech_start": utterance.start, "speech_end": utterance.end,
            "start_sample": first, "end_sample": last,
            "start": first / sample_rate, "end": last / sample_rate,
            "duration": clip_duration, "rms_dbfs": rms_dbfs,
            "speech_ratio": speech_ratio, "rejection_reasons": reasons,
            "eligible": not reasons, "saved": False,
        })
    return decisions


class WhisperXBackend:
    """Lazy optional dependency loading keeps CLI help and unit tests lightweight."""

    def __init__(self, args: argparse.Namespace) -> None:
        if shutil.which("ffmpeg") is None:
            raise BatchError(
                "FFmpeg executable not found on PATH. Install a shared FFmpeg build, "
                "add its bin directory to PATH, then restart your terminal/app. "
                "Verify with: ffmpeg -version. uv sync does not install FFmpeg."
            )
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise BatchError(
                "Set HF_TOKEN in the project's .env or environment and accept "
                "the diarization model's access terms"
            )
        try:
            import torch
            import whisperx
            from whisperx.diarize import DiarizationPipeline

            self.whisperx = whisperx
            self.args = args
            if args.device == "auto":
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            if args.compute_type == "auto":
                args.compute_type = "float16" if args.device == "cuda" else "int8"
            self.asr = whisperx.load_model(
                args.model, args.device, compute_type=args.compute_type,
                language=args.language, vad_method="pyannote",
                vad_options={"vad_onset": args.vad_onset, "vad_offset": args.vad_offset},
            )
            self.diarizer = DiarizationPipeline(token=token, device=args.device)
            self.align_language = None
            self.aligner = None
            self.align_metadata = None
        except Exception as exc:
            raise BatchError(f"WhisperX/model initialization failed: {exc}") from exc

    def analyze(self, path: Path) -> tuple[str, list[dict[str, Any]], list[Turn]]:
        """Transcribe, force-align, and retain nonexclusive diarization intervals."""
        wx, args = self.whisperx, self.args
        try:
            audio = wx.load_audio(str(path))
        except FileNotFoundError as exc:
            raise BatchError(
                "WhisperX could not launch FFmpeg to decode the recording. "
                "Check that ffmpeg -version works in this terminal."
            ) from exc
        result = self.asr.transcribe(audio, batch_size=args.batch_size)
        language = result["language"]
        if not result["segments"]:
            return language, [], []
        if self.align_language != language:
            # Keep only one language's aligner resident at a time.
            self.aligner = None
            try:
                self.aligner, self.align_metadata = wx.load_align_model(
                    language_code=language, device=args.device
                )
            except Exception as exc:
                raise BatchError(f"Cannot load alignment model for {language}: {exc}") from exc
            self.align_language = language
        aligned = wx.align(
            result["segments"], self.aligner, self.align_metadata,
            audio, args.device, return_char_alignments=False,
        )
        if not aligned.get("word_segments"):
            raise ValueError("Speech was transcribed but no aligned words were returned")
        diarization = self.diarizer(
            audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers
        )
        turns = [Turn(float(row.start), float(row.end), str(row.speaker))
                 for row in diarization.itertuples(index=False)]
        return language, aligned["word_segments"], turns


def safe_error(exc: Exception) -> str:
    """Avoid exposing the authentication token in library error messages."""
    message = str(exc)
    token = os.environ.get("HF_TOKEN")
    return message.replace(token, "[REDACTED]") if token else message


def is_auth_error(exc: Exception) -> bool:
    """Find authentication failures even when wrapped by a library."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def run(args: argparse.Namespace, backend_factory: Any = WhisperXBackend) -> int:
    """Run a batch, optionally injecting a backend for data-free verification."""
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)
    recordings = preflight(args)
    backend = backend_factory(args)
    print(f"Runtime: device={args.device}, compute_type={args.compute_type}, "
          f"model={args.model}, batch_size={args.batch_size}")
    totals = Counter()
    fatal = None
    for path, prefix in recordings:
        record = {"source": str(path.resolve()), "prefix": prefix,
                  "settings": {k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items()},
                  "status": "ok", "clips": []}
        try:
            audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Recording is empty or contains nonfinite samples")
            if np.max(np.abs(audio)) > 1:
                raise ValueError("Audio exceeds PCM range; refusing to clip its signal")
            language, words, turns = backend.analyze(path)
            utterances, excluded = build_utterances(words, turns, args.pause_threshold)
            clips = plan_clips(utterances, turns, prefix, audio, sample_rate, args)
            record.update({
                "language": language, "sample_rate": sample_rate,
                "channels": audio.shape[1], "speaker_selection": "all",
                "speaker_counts": dict(Counter(u.speaker for u in utterances)),
                "utterances": [asdict(u) for u in utterances],
                "diarization": [asdict(t) for t in turns],
                "excluded_words": excluded, "clips": clips,
            })
            if not args.dry_run:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for clip in clips:
                    if clip["eligible"]:
                        # Exclusive creation protects against collisions after preflight.
                        with (args.output_dir / clip["filename"]).open("xb") as stream:
                            sf.write(stream, audio[clip["start_sample"]:clip["end_sample"]],
                                     sample_rate, format="WAV", subtype="PCM_32")
                        clip["saved"] = True
        except Exception as exc:
            record.update(status="failed", error=safe_error(exc))
            totals["failed"] += 1
            print(f"FAILED {path.name}: {safe_error(exc)}", file=sys.stderr)
            if isinstance(exc, BatchError) or is_auth_error(exc):
                fatal = exc
        clips = record["clips"]
        eligible = [c for c in clips if c["eligible"]]
        saved = [c for c in clips if c["saved"]]
        rejected = Counter(reason for c in clips for reason in c["rejection_reasons"])
        print(f"{path.name}: speakers={record.get('speaker_counts', {})}, "
              f"candidates={len(clips)}, would_save={len(eligible)}, "
              f"duration={sum(c['duration'] for c in eligible):.3f}s, "
              f"saved={len(saved)}, rejections={dict(rejected)}")
        for clip in eligible:
            print(f"  {'WOULD SAVE' if args.dry_run else 'SAVED' if clip['saved'] else 'NOT SAVED'} "
                  f"{clip['filename']} (speaker={clip['speaker']})")
        if not args.dry_run:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                manifest = args.output_dir / f"{prefix}_manifest.json"
                with manifest.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            except Exception as exc:
                if record["status"] != "failed":
                    totals["failed"] += 1
                print(f"FAILED to save manifest for {path.name}: {safe_error(exc)}",
                      file=sys.stderr)
                fatal = exc
        totals["processed"] += 1
        totals["would_save"] += len(eligible)
        totals["saved"] += len(saved)
        totals["duration_ms"] += round(sum(c["duration"] for c in eligible) * 1000)
        if fatal is not None:
            break
    print(f"TOTAL: processed={totals['processed']}/{len(recordings)}, "
          f"failed={totals['failed']}, would_save={totals['would_save']}, "
          f"saved={totals['saved']}, duration={totals['duration_ms'] / 1000:.3f}s, "
          f"unprocessed={len(recordings) - totals['processed']}")
    return 1 if totals["failed"] else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate extraction settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help=(
        "Run full inference and report proposed clips without writing results. "
        "Still incurs full inference time and may download models into caches."
    ))
    parser.add_argument("--language", help="Language code; default: detect per recording")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compute-type", choices=["auto", "float16", "float32", "int8"], default="auto")
    parser.add_argument("--pause-threshold", type=float, default=0.7)
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum unpadded utterance duration in seconds (default: 0.5)")
    parser.add_argument("--max-duration", type=float,
                        help="Maximum unpadded duration; longer utterances are rejected")
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--vad-onset", type=float, default=0.5)
    parser.add_argument("--vad-offset", type=float, default=0.363)
    parser.add_argument("--min-rms-dbfs", type=float)
    parser.add_argument("--min-speech-ratio", type=float,
                        help="Minimum diarization speech coverage / padded clip duration")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    for name in ("pause_threshold", "min_duration", "padding"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("batch_size", "min_words", "min_speakers", "max_speakers"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("vad_onset", "vad_offset", "min_speech_ratio"):
        if getattr(args, name) is not None and not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.vad_offset > args.vad_onset:
        parser.error("--vad-offset must not exceed --vad-onset")
    if args.max_duration is not None and (
        args.max_duration <= 0 or args.max_duration < args.min_duration
    ):
        parser.error("--max-duration must be positive and at least --min-duration")
    if (args.min_speakers is not None and args.max_speakers is not None
            and args.min_speakers > args.max_speakers):
        parser.error("--min-speakers must not exceed --max-speakers")
    if args.device == "cpu" and args.compute_type == "float16":
        parser.error("Use float32 or int8 compute on CPU")
    return args


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {safe_error(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
