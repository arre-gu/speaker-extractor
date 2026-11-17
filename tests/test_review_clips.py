"""Review persistence, non-destructive exports, and Windows playback conversion."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import numpy as np
import pytest
import soundfile as sf

from review_clips import ClipReviewer, ReviewStore, prepare_playback


@pytest.fixture
def clips(tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    for ordinal in (1, 10, 2):
        sf.write(folder / f"p_{ordinal}.wav", np.full((800, 2), 0.25), 8000, subtype="PCM_32")
    (folder / "p_manifest.json").write_text(json.dumps({
        "prefix": "p", "source": "recording.wav", "clips": [
            {"filename": "p_1.wav", "speaker": "TTS", "text": "prompt", "duration": 0.1},
            {"filename": "p_2.wav", "speaker": "PERSON", "text": "reply", "duration": 0.1},
        ]}), encoding="utf-8")
    return folder


def test_progress_resume_undo_and_originals_untouched(clips):
    original = {p.name: p.read_bytes() for p in clips.glob("*.wav")}
    store = ReviewStore(clips)
    assert [p.name for p in store.files] == ["p_1.wav", "p_2.wav", "p_10.wav"]
    assert store.metadata["p_2.wav"]["speaker"] == "PERSON"
    store.mark("p_1.wav", "reject")
    store.mark("p_2.wav", "keep")
    assert ReviewStore(clips).decisions == {"p_1.wav": "reject", "p_2.wav": "keep"}
    assert store.undo() == "p_2.wav"
    assert ReviewStore(clips).decisions == {"p_1.wav": "reject"}
    store.mark("p_1.wav", "keep")
    store.undo()
    assert store.decisions["p_1.wav"] == "reject"
    assert {p.name: p.read_bytes() for p in clips.glob("*.wav")} == original


def test_export_only_kept_never_overwrites(clips, tmp_path):
    store = ReviewStore(clips)
    store.mark("p_2.wav", "keep")
    store.mark("p_1.wav", "reject")
    output = tmp_path / "kept"
    assert store.export_kept(output) == 1
    assert [p.name for p in output.iterdir()] == ["p_2.wav"]
    assert (output / "p_2.wav").read_bytes() == (clips / "p_2.wav").read_bytes()
    with pytest.raises(FileExistsError):
        store.export_kept(output)
    with pytest.raises(ValueError, match="separate"):
        store.export_kept(clips)


def test_corrupt_decisions_are_not_overwritten(clips):
    state = clips / ".review.json"
    state.write_text('{"version": 1, "decisions": {"p_1.wav": "bad"}}', encoding="utf-8")
    before = state.read_bytes()
    with pytest.raises(ValueError, match="invalid"):
        ReviewStore(clips)
    assert state.read_bytes() == before


def test_save_failure_rolls_back_memory(clips, monkeypatch):
    store = ReviewStore(clips)
    monkeypatch.setattr(store, "save", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        store.mark("p_1.wav", "keep")
    assert store.decisions == {}
    assert store.history == []


@pytest.mark.parametrize("subtype", ["PCM_U8", "PCM_16", "PCM_24", "PCM_32"])
def test_playback_conversion_preserves_source_and_timing(tmp_path, subtype):
    source = tmp_path / "source.wav"
    target = tmp_path / "playback.wav"
    samples = np.column_stack((np.linspace(-0.75, 0.75, 800), np.full(800, 0.25)))
    sf.write(source, samples, 8000, subtype=subtype)
    before = source.read_bytes()
    assert prepare_playback(source, target) == pytest.approx(0.1)
    with wave.open(str(target)) as stream:
        assert stream.getsampwidth() == 2
        assert stream.getnchannels() == 2
        assert stream.getnframes() == 800
    actual, rate = sf.read(target, always_2d=True)
    expected, _ = sf.read(source, always_2d=True)
    np.testing.assert_allclose(actual, expected, atol=1 / 32768)
    assert rate == 8000
    assert source.read_bytes() == before


def test_hidden_reviewer_navigation_and_decisions(clips, monkeypatch):
    import tkinter as tk

    monkeypatch.setitem(sys.modules, "winsound", SimpleNamespace(PlaySound=Mock()))
    root = tk.Tk()
    root.withdraw()
    app = ClipReviewer(root)
    app.autoplay.set(False)
    try:
        app.open_folder(clips)
        root.update_idletasks()
        assert app.current.name == "p_1.wav"
        app.mark("reject")
        assert app.current.name == "p_2.wav"
        app.mark("keep")
        assert app.current.name == "p_10.wav"
        app.undo()
        assert app.current.name == "p_2.wav"
        assert "p_2.wav" not in app.store.decisions
        app.speaker_filter.set("p / TTS")
        app.refresh()
        assert [p.name for p in app.visible] == ["p_1.wav"]
        assert app.transcript.get("1.0", "end").strip() == "prompt"
    finally:
        app.close()
