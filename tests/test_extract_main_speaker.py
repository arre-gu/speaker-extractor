"""Dataset-free checks for extraction decisions, audio export, and dry-run parity."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from extract_main_speaker import (
    BatchError, Turn, Utterance, WhisperXBackend, build_utterances, coverage,
    parse_args, parse_prefix, plan_clips, preflight, run,
)


@pytest.fixture(autouse=True)
def isolate_project_env(tmp_path, monkeypatch):
    # Tests must never read the user's real token file.
    monkeypatch.setattr("extract_main_speaker.__file__", str(tmp_path / "extract_main_speaker.py"))


def settings(tmp_path, *flags):
    return parse_args([str(tmp_path / "in"), str(tmp_path / "out"), *flags])


def word(start, end, text="hello"):
    return {"start": start, "end": end, "word": text}


def recording(tmp_path, name="camera_(40,FEMALE,0010).wav"):
    source = tmp_path / "in"
    source.mkdir(exist_ok=True)
    # Different channel signals verify that export never silently downmixes.
    audio = np.column_stack((np.linspace(-0.8, 0.8, 48000), np.full(48000, 0.1)))
    path = source / name
    sf.write(path, audio, 8000, subtype="PCM_16")
    return path


def test_filename_metadata():
    assert parse_prefix(Path("RespCam_Anonymous 29-04-26 16h18m_Scene camera 1_(40,FEMALE,1010).wav")) == "40_FEMALE_1010"
    assert parse_prefix(Path("camera_( 40 , female , 0010 ).WAV")) == "40_FEMALE_0010"
    with pytest.raises(ValueError, match="trailing"):
        parse_prefix(Path("no_metadata.wav"))


def test_preflight_discovery_and_collisions(tmp_path):
    recording(tmp_path, "b_(40,FEMALE,0010).WAV")
    nested = tmp_path / "in" / "nested"
    nested.mkdir()
    sf.write(nested / "a_(41,MALE,0001).wav", np.zeros(10), 8000)
    args = settings(tmp_path)
    assert len(preflight(args)) == 1
    args.recursive = True
    assert len(preflight(args)) == 2
    args.output_dir.mkdir()
    (args.output_dir / "40_FEMALE_0010_1.WAV").touch()
    with pytest.raises(ValueError, match="Existing output"):
        preflight(args)


def test_preflight_validates_whole_batch_before_models(tmp_path):
    recording(tmp_path)
    recording(tmp_path, "other_(40,female,0010).wav")
    recording(tmp_path, "bad.wav")
    backend = Mock()
    with pytest.raises(ValueError, match="Duplicate metadata") as error:
        run(settings(tmp_path), backend)
    assert "Missing trailing" in str(error.value)
    backend.assert_not_called()


def test_utterance_splits_and_unassigned_words():
    turns = [Turn(0, 4, "A"), Turn(1.1, 1.2, "B"), Turn(4, 5, "B")]
    words = [word(0.1, 0.5), word(0.6, 1), word(1.3, 1.5),
             word(2.5, 3), {"word": "untimed"}, word(3.1, 3.5),
             word(4, 4.5), word(6, 7), word(7, 7)]
    utterances, excluded = build_utterances(words, turns, 0.7)
    assert [(u.speaker, len(u.words)) for u in utterances] == [
        ("A", 2), ("A", 1), ("A", 1), ("A", 1), ("B", 1)
    ]
    assert [e["reason"] for e in excluded] == [
        "untimed_or_invalid_word", "unassigned_word", "untimed_or_invalid_word"
    ]


def test_every_speaker_exported_despite_dominant_tts(tmp_path):
    utterances = [Utterance(4, 5, "PERSON", ["reply"]),
                  Utterance(0, 1, "TTS", ["prompt"]),
                  Utterance(2, 3, "TTS", ["prompt"])]
    turns = [Turn(u.start, u.end, u.speaker) for u in utterances]
    clips = plan_clips(utterances, turns, "p", np.ones((6000, 1)),
                       1000, settings(tmp_path))
    assert [c["speaker"] for c in clips] == ["TTS", "TTS", "PERSON"]
    assert [c["filename"] for c in clips] == ["p_1.wav", "p_2.wav", "p_3.wav"]
    assert all(c["eligible"] for c in clips)
    assert plan_clips([], [], "p", np.ones((100, 1)), 1000, settings(tmp_path)) == []


def test_overlap_rejected_for_each_speaker(tmp_path):
    utterances = [Utterance(0, 1, "A", ["a"]), Utterance(0.5, 1.5, "B", ["b"]),
                  Utterance(2, 3, "B", ["clean"]), Utterance(3, 4, "A", ["clean"])]
    turns = [Turn(u.start, u.end, u.speaker) for u in utterances]
    clips = plan_clips(utterances, turns, "p", np.ones((5000, 1)),
                       1000, settings(tmp_path))
    assert [c["ordinal"] for c in clips if c["eligible"]] == [3, 4]
    assert clips[0]["rejection_reasons"] == ["other_speaker_overlap"]
    assert clips[1]["rejection_reasons"] == ["other_speaker_overlap"]
    assert clips[2]["end_sample"] == 3000
    assert clips[3]["start_sample"] == 3000


def test_overlap_numbering_padding_and_quality_filters(tmp_path):
    args = settings(tmp_path, "--min-rms-dbfs", "-60", "--min-speech-ratio", "0.5")
    utterances = [Utterance(0.05, 0.8, "A", ["one"]),
                  Utterance(1.1, 2, "A", ["two"]),
                  Utterance(3.1, 3.2, "A", ["three"]),
                  Utterance(4, 5, "A", ["four"])]
    turns = [Turn(0, 5, "A"), Turn(1.5, 1.6, "B"), Turn(0.85, 0.95, "B")]
    clips = plan_clips(utterances, turns, "40_FEMALE_0010",
                             np.full((6000, 2), 0.1), 1000, args)
    assert [c["ordinal"] for c in clips if c["eligible"]] == [1, 4]
    assert clips[0]["start_sample"] == 0
    assert clips[0]["end_sample"] == 850
    assert clips[1]["rejection_reasons"] == ["other_speaker_overlap"]
    assert clips[2]["rejection_reasons"] == ["too_short"]
    args.min_words = 2
    args.max_duration = 0.6
    args.min_rms_dbfs = -10
    args.min_speech_ratio = 0.9
    clips = plan_clips([utterances[0]], [Turn(0.05, 0.15, "A")], "p",
                          np.zeros((1000, 1)), 1000, args)
    assert set(clips[0]["rejection_reasons"]) == {
        "too_long", "too_few_words", "below_rms_threshold", "low_speech_ratio"
    }


def test_coverage_unions_overlapping_intervals():
    assert coverage(0, 3, [Turn(0, 2, "A"), Turn(1, 3, "B")]) == 3


class FakeBackend:
    def __init__(self, args):
        pass

    def analyze(self, path):
        return "en", [word(0.2, 1), word(2, 2.2), word(4, 5)], [Turn(0, 3, "A"), Turn(4, 6, "B")]


@pytest.mark.parametrize("existing", [None, "existing-test-token"])
def test_project_dotenv_loaded_before_backend(tmp_path, monkeypatch, existing):
    recording(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("HF_TOKEN=file-test-token\n", encoding="utf-8")
    monkeypatch.setattr("extract_main_speaker.__file__", str(project / "extract_main_speaker.py"))
    monkeypatch.chdir(tmp_path)
    if existing is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", existing)

    class Backend(FakeBackend):
        def __init__(self, args):
            import os

            assert os.environ["HF_TOKEN"] == (existing or "file-test-token")

    assert run(settings(tmp_path, "--dry-run"), Backend) == 0


def test_dry_run_matches_export_without_writes(tmp_path, capsys):
    path = recording(tmp_path)
    args = settings(tmp_path, "--dry-run")
    before = set(tmp_path.rglob("*"))
    assert run(args, FakeBackend) == 0
    dry_output = capsys.readouterr().out
    assert set(tmp_path.rglob("*")) == before
    assert "would_save=2" in dry_output
    assert "saved=0" in dry_output
    args.dry_run = False
    assert run(args, FakeBackend) == 0
    manifest = json.loads((args.output_dir / "40_FEMALE_0010_manifest.json").read_text())
    assert manifest["speaker_selection"] == "all"
    assert "selected_speaker" not in manifest
    assert manifest["speaker_counts"] == {"A": 1, "B": 1}
    assert [c["speaker"] for c in manifest["clips"] if c["eligible"]] == ["A", "B"]
    assert manifest["language"] == "en"
    original, rate = sf.read(path, always_2d=True)
    for clip in manifest["clips"]:
        output = args.output_dir / clip["filename"]
        if not clip["eligible"]:
            assert not output.exists()
            continue
        assert f"WOULD SAVE {clip['filename']}" in dry_output
        assert clip["saved"]
        actual, actual_rate = sf.read(output, always_2d=True)
        assert actual_rate == rate
        np.testing.assert_array_equal(actual, original[clip["start_sample"]:clip["end_sample"]])
    with pytest.raises(ValueError, match="Existing output"):
        run(args, FakeBackend)


def test_recording_failure_continues_and_empty_speech(tmp_path, capsys):
    recording(tmp_path, "a_(40,FEMALE,0010).wav")
    recording(tmp_path, "b_(41,MALE,0011).wav")

    class Backend(FakeBackend):
        def analyze(self, path):
            if path.name.startswith("a"):
                raise ValueError("corrupt recording")
            return "en", [], []

    assert run(settings(tmp_path, "--dry-run"), Backend) == 1
    output = capsys.readouterr().out
    assert "processed=2/2" in output
    assert "failed=1" in output
    assert "would_save=0" in output


def test_model_failure_stops_batch_and_redacts_token(tmp_path, monkeypatch, capsys):
    recording(tmp_path)
    recording(tmp_path, "b_(41,MALE,0011).wav")
    monkeypatch.setenv("HF_TOKEN", "test-secret")

    class Backend(FakeBackend):
        def analyze(self, path):
            raise BatchError("alignment failed test-secret")

    assert run(settings(tmp_path, "--dry-run"), Backend) == 1
    output = capsys.readouterr()
    assert "test-secret" not in output.err
    assert "[REDACTED]" in output.err
    assert "unprocessed=1" in output.out


def test_backend_alignment_and_empty_speech(tmp_path):
    # Exercise the real adapter without importing or downloading WhisperX.
    backend = WhisperXBackend.__new__(WhisperXBackend)
    backend.args = settings(tmp_path)
    backend.args.device = "cpu"
    backend.whisperx = Mock()
    backend.asr = Mock()
    backend.align_language = None
    backend.aligner = None
    backend.align_metadata = None
    backend.asr.transcribe.return_value = {"language": "sv", "segments": []}
    assert backend.analyze(Path("fake.wav")) == ("sv", [], [])
    backend.whisperx.load_align_model.assert_not_called()
    backend.asr.transcribe.return_value = {
        "language": "sv", "segments": [{"text": "hej", "start": 0, "end": 1}]
    }
    backend.whisperx.load_align_model.side_effect = ValueError("unsupported language")
    with pytest.raises(BatchError, match="alignment model for sv"):
        backend.analyze(Path("fake.wav"))
    backend.whisperx.load_align_model.side_effect = None
    backend.whisperx.load_align_model.return_value = (Mock(), {})
    backend.whisperx.align.return_value = {"word_segments": []}
    with pytest.raises(ValueError, match="no aligned words"):
        backend.analyze(Path("fake.wav"))
    backend.whisperx.align.return_value = {"word_segments": [word(0, 1)]}
    backend.diarizer = Mock()
    backend.diarizer.return_value.itertuples.return_value = [
        SimpleNamespace(start=0, end=1, speaker="A"),
        SimpleNamespace(start=0.5, end=1, speaker="B"),
    ]
    language, words, turns = backend.analyze(Path("fake.wav"))
    assert language == "sv"
    assert words == [word(0, 1)]
    assert turns == [Turn(0, 1, "A"), Turn(0.5, 1, "B")]
    # Repeated language reuses its aligner, including after an alignment error.
    assert backend.whisperx.load_align_model.call_count == 2
    backend.diarizer.assert_called_once_with(
        backend.whisperx.load_audio.return_value, min_speakers=None, max_speakers=None
    )


def test_missing_ffmpeg_fails_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr("extract_main_speaker.shutil.which", lambda name: None)
    with pytest.raises(BatchError, match="FFmpeg executable not found on PATH"):
        WhisperXBackend(settings(tmp_path))


def test_ffmpeg_launch_failure_stops_batch(tmp_path):
    backend = WhisperXBackend.__new__(WhisperXBackend)
    backend.args = settings(tmp_path)
    backend.whisperx = Mock()
    backend.whisperx.load_audio.side_effect = FileNotFoundError("[WinError 2]")
    with pytest.raises(BatchError, match="could not launch FFmpeg"):
        backend.analyze(Path("fake.wav"))


def test_nonfinite_word_timestamp_is_manifest_safe(tmp_path):
    recording(tmp_path)

    class Backend(FakeBackend):
        def analyze(self, path):
            return "en", [word(float("nan"), 1)], [Turn(0, 1, "A")]

    args = settings(tmp_path)
    assert run(args, Backend) == 0
    manifest = json.loads((args.output_dir / "40_FEMALE_0010_manifest.json").read_text())
    assert manifest["excluded_words"][0]["start"] is None
    assert manifest["clips"] == []


def test_padding_clamped_to_end_and_sample_grid(tmp_path):
    args = settings(tmp_path)
    clips = plan_clips(
        [Utterance(0.31, 1, "A", ["hello"])],
        [Turn(0.31, 1, "A"), Turn(0.1, 0.2555, "B")],
        "p", np.ones((1000, 1)), 1000, args,
    )
    assert clips[0]["start_sample"] == 256
    assert clips[0]["end_sample"] == 1000
    assert clips[0]["eligible"]


def test_natural_pause_merged_but_speaker_change_preserved(tmp_path):
    args = settings(tmp_path)
    assert args.pause_threshold == 1.2
    words = [word(0, 0.5), word(1.5, 2), word(2.05, 3)]
    turns = [Turn(0, 2, "A"), Turn(2.05, 3, "B")]
    utterances, _ = build_utterances(words, turns, args.pause_threshold)
    assert [(u.speaker, len(u.words)) for u in utterances] == [("A", 2), ("B", 1)]
    shorter, _ = build_utterances(words, turns, 0.7)
    assert len(shorter) == 3


def test_later_response_survives_short_leading_overlap(tmp_path):
    args = settings(tmp_path)
    utterances = [Utterance(0, 1.2, "TTS", ["prompt"]),
                  Utterance(1, 2, "HUMAN", ["reply"])]
    turns = [Turn(0, 1.2, "TTS"), Turn(1, 2, "HUMAN")]
    clips = plan_clips(utterances, turns, "p", np.ones((3000, 1)), 1000, args)
    assert clips[0]["rejection_reasons"] == ["other_speaker_overlap"]
    assert clips[1]["eligible"]
    assert clips[1]["start_sample"] == 1200
    assert clips[1]["trimmed_leading_seconds"] == pytest.approx(0.2)
    assert clips[1]["retained_speech_start"] == pytest.approx(1.2)
    args.handoff_trim = 0
    strict = plan_clips(utterances, turns, "p", np.ones((3000, 1)), 1000, args)
    assert not strict[1]["eligible"]


@pytest.mark.parametrize("tts_start, tts_end, human_start, human_end", [
    (0, 1.5, 1, 2),       # more leading overlap than allowed
    (1, 1.2, 0, 2),       # internal interruption by a later voice
    (1, 1.2, 1, 2),       # simultaneous starts: no later speaker
    (0, 2, 1, 1.1),       # entire reply overlaps
])
def test_handoff_never_keeps_overlapping_audio(
    tmp_path, tts_start, tts_end, human_start, human_end
):
    turns = [Turn(tts_start, tts_end, "TTS"), Turn(human_start, human_end, "HUMAN")]
    clips = plan_clips([Utterance(human_start, human_end, "HUMAN", ["reply"])],
                       turns, "p", np.ones((3000, 1)), 1000, settings(tmp_path))
    assert "other_speaker_overlap" in clips[0]["rejection_reasons"]


def test_short_replies_use_retained_duration_threshold(tmp_path):
    utterances = [Utterance(1, 1.5, "HUMAN", ["yes"])]
    turns = [Turn(0, 1.2, "TTS"), Turn(1, 1.5, "HUMAN")]
    args = settings(tmp_path)
    clips = plan_clips(utterances, turns, "p", np.ones((2000, 1)), 1000, args)
    assert clips[0]["rejection_reasons"] == ["too_short"]
    args.min_duration = 0.2
    clips = plan_clips(utterances, turns, "p", np.ones((2000, 1)), 1000, args)
    assert clips[0]["eligible"]


@pytest.mark.parametrize("flags", [
    ["--padding", "-1"], ["--pause-threshold", "nan"],
    ["--min-speakers", "3", "--max-speakers", "2"],
    ["--min-speech-ratio", "1.1"], ["--max-duration", "0"],
    ["--vad-offset", "0.8"], ["--device", "cpu", "--compute-type", "float16"],
    ["--handoff-trim", "-0.1"],
])
def test_invalid_settings(tmp_path, flags):
    with pytest.raises(SystemExit):
        settings(tmp_path, *flags)
