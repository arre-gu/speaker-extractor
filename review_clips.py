"""Windows clip reviewer: listen, mark keep/reject, and copy kept WAVs safely.

Uses only Python's standard library. Original clips are never moved or deleted.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable
import wave


class ReviewStore:
    """Persist decisions atomically beside the clips and resume on next launch."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder.resolve()
        self.path = self.folder / ".review.json"
        self.files = sorted(
            (p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
            key=lambda p: [int(s) if s.isdigit() else s.casefold()
                           for s in re.split(r"(\d+)", p.name)],
        )
        self.decisions = {}
        self.history = []
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            decisions = state.get("decisions")
            if (state.get("version") != 1 or not isinstance(decisions, dict)
                    or any(v not in ("keep", "reject") for v in decisions.values())):
                raise ValueError("Unsupported or invalid .review.json; existing decisions were not changed")
            self.decisions = decisions
        self.metadata = {}
        for manifest in sorted(self.folder.glob("*_manifest.json")):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for clip in data.get("clips", []):
                name = clip.get("filename", "")
                if name and Path(name).name == name:
                    self.metadata[name] = {**clip, "source": data.get("source", ""),
                                           "prefix": data.get("prefix", "")}

    def save(self) -> None:
        """Replace only the review state, never the recordings."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.folder,
                prefix=".review-", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump({"version": 1, "decisions": self.decisions}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def mark(self, name: str, decision: str | None) -> None:
        """Save an explicit decision, retaining an undo entry after successful save."""
        if decision not in (None, "keep", "reject"):
            raise ValueError("Decision must be keep, reject, or None")
        old = self.decisions.get(name)
        self._set(name, decision)
        try:
            self.save()
        except Exception:
            self._set(name, old)
            raise
        self.history.append((name, old))

    def _set(self, name: str, decision: str | None) -> None:
        if decision is None:
            self.decisions.pop(name, None)
        else:
            self.decisions[name] = decision

    def undo(self) -> str | None:
        """Undo the latest decision made during this session."""
        if not self.history:
            return None
        name, previous = self.history.pop()
        current = self.decisions.get(name)
        self._set(name, previous)
        try:
            self.save()
        except Exception:
            self._set(name, current)
            self.history.append((name, previous))
            raise
        return name

    def export_kept(self, destination: Path) -> int:
        """Copy kept originals with exclusive creation; refuse any collision."""
        kept = [p for p in self.files if self.decisions.get(p.name) == "keep"]
        if destination.resolve() == self.folder:
            raise ValueError("Choose a separate export folder")
        collisions = [p.name for p in kept if (destination / p.name).exists()]
        if collisions:
            raise FileExistsError(f"Export would overwrite existing clips: {', '.join(collisions[:5])}")
        destination.mkdir(parents=True, exist_ok=True)
        for path in kept:
            with path.open("rb") as source, (destination / path.name).open("xb") as target:
                shutil.copyfileobj(source, target)
        return len(kept)


def prepare_playback(source: Path, destination: Path) -> float:
    """Make a temporary PCM16 WAV for reliable Windows playback of PCM32 exports."""
    with wave.open(str(source), "rb") as stream:
        width = stream.getsampwidth()
        rate, channels = stream.getframerate(), stream.getnchannels()
        raw = stream.readframes(stream.getnframes())
    if width not in (1, 2, 3, 4):
        raise ValueError("Playback supports uncompressed 8/16/24/32-bit PCM WAV")
    if width != 2:
        samples = array("h", (
            ((raw[i] - 128) << 8) if width == 1 else
            (int.from_bytes(raw[i:i + width], "little", signed=True) >> (8 * (width - 2)))
            for i in range(0, len(raw), width)
        ))
        if sys.byteorder != "little":
            samples.byteswap()
        raw = samples.tobytes()
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(raw)
    return len(raw) / (2 * channels * rate)


class ClipReviewer:
    """Single-window review UI with shortcuts, speaker filters, and saved progress."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.store = None
        self.visible = []
        self.current = None
        self.playing = False
        self.playback_timer = None
        self.temporary = tempfile.TemporaryDirectory(prefix="clip-review-")
        self.playback = Path(self.temporary.name) / "playback.wav"
        root.title("Speech clip reviewer")
        root.geometry("1040x680")
        root.minsize(800, 500)
        root.protocol("WM_DELETE_WINDOW", self.close)
        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open folder", command=self.choose_folder).pack(side="left")
        ttk.Button(toolbar, text="Copy kept clips…", command=self.export).pack(side="left", padx=8)
        self.autoplay = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Autoplay", variable=self.autoplay).pack(side="left")
        self.status_filter = ttk.Combobox(toolbar, values=["Pending", "All", "Keep", "Reject"],
                                          state="readonly", width=10)
        self.status_filter.set("Pending")
        self.status_filter.pack(side="right")
        self.speaker_filter = ttk.Combobox(toolbar, values=["All speakers"], state="readonly", width=32)
        self.speaker_filter.set("All speakers")
        self.speaker_filter.pack(side="right", padx=8)
        for box in (self.status_filter, self.speaker_filter):
            box.bind("<<ComboboxSelected>>", lambda event: self.refresh())
        self.progress = ttk.Label(root, text="Open an extraction output folder", padding=(10, 4))
        self.progress.pack(fill="x")
        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=8)
        left = ttk.Frame(body)
        self.listbox = tk.Listbox(left, exportselection=False, width=42)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.select)
        body.add(left, weight=2)
        right = ttk.Frame(body, padding=10)
        self.details = ttk.Label(right, text="", wraplength=510, justify="left")
        self.details.pack(fill="x", pady=(0, 14))
        self.transcript = tk.Text(right, wrap="word", height=12, state="disabled")
        self.transcript.pack(fill="both", expand=True)
        ttk.Label(right, text="Filename demographics describe the recording, not every speaker.",
                  wraplength=500).pack(fill="x", pady=10)
        body.add(right, weight=3)
        buttons = ttk.Frame(root, padding=10)
        buttons.pack(fill="x")
        for label, command in [
            ("◀ Previous", lambda: self.navigate(-1)), ("Play / Stop [Space]", self.toggle_play),
            ("Keep [K]", lambda: self.mark("keep")), ("Reject [R]", lambda: self.mark("reject")),
            ("Skip →", lambda: self.navigate(1)), ("Undo [U]", self.undo),
        ]:
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=3)
        for key, command in [("<space>", self.toggle_play), ("k", lambda: self.mark("keep")),
                             ("r", lambda: self.mark("reject")), ("u", self.undo),
                             ("<Left>", lambda: self.navigate(-1)),
                             ("<Right>", lambda: self.navigate(1))]:
            root.bind(key, lambda event, action=command: self.shortcut(action))

    def shortcut(self, action: Callable[[], None]) -> str:
        action()
        return "break"

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self.root, title="Choose extracted WAV folder")
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder: Path) -> None:
        try:
            store = ReviewStore(folder)
            if not store.files:
                raise ValueError("No WAV files in this folder")
            self.stop()
            self.store = store
            groups = sorted({self.group(p) for p in store.files})
            self.speaker_filter.configure(values=["All speakers", *groups])
            self.speaker_filter.set("All speakers")
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Cannot open folder", str(exc), parent=self.root)

    def group(self, path: Path) -> str:
        meta = self.store.metadata.get(path.name, {})
        return f"{meta.get('prefix', 'Unknown recording')} / {meta.get('speaker', 'Unknown speaker')}"

    def refresh(self, preferred: str | None = None) -> None:
        if self.store is None:
            return
        self.stop()
        status = self.status_filter.get().lower()
        group = self.speaker_filter.get()
        self.visible = [p for p in self.store.files
                        if (status == "all" or self.store.decisions.get(p.name, "pending") == status)
                        and (group == "All speakers" or self.group(p) == group)]
        self.listbox.delete(0, "end")
        for path in self.visible:
            self.listbox.insert("end", f"[{self.store.decisions.get(path.name, 'pending')}] {path.name}")
        counts = Counter(self.store.decisions.get(p.name, "pending") for p in self.store.files)
        self.progress.configure(text=(f"{self.store.folder} — {len(self.store.files)} clips | "
                                      f"Keep {counts['keep']} | Reject {counts['reject']} | "
                                      f"Pending {counts['pending']} | Showing {len(self.visible)}"))
        self.current = None
        if self.visible:
            index = next((i for i, p in enumerate(self.visible) if p.name == preferred), 0)
            self.listbox.selection_set(index)
            self.listbox.see(index)
        self.select()

    def select(self, event: object = None) -> None:
        self.stop()
        selection = self.listbox.curselection()
        self.current = self.visible[selection[0]] if selection else None
        meta = self.store.metadata.get(self.current.name, {}) if self.current else {}
        self.details.configure(text=(
            f"{self.current.name}\n{self.group(self.current)}\n"
            f"Source: {Path(meta.get('source', '')).name}\n"
            f"Duration: {meta.get('duration', '?')} s"
        ) if self.current else "No clips in this view. Change the filter to inspect reviewed clips.")
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("1.0", meta.get("text", "Transcript unavailable"))
        self.transcript.configure(state="disabled")
        if self.current and self.autoplay.get():
            self.toggle_play()

    def stop(self) -> None:
        if self.playback_timer is not None:
            self.root.after_cancel(self.playback_timer)
            self.playback_timer = None
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, 0)
        self.playing = False

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
            return
        if self.current is None:
            return
        try:
            import winsound

            duration = prepare_playback(self.current, self.playback)
            winsound.PlaySound(str(self.playback), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            self.playing = True
            self.playback_timer = self.root.after(max(1, round(duration * 1000)), self.playback_finished)
        except Exception as exc:
            messagebox.showerror("Playback failed", str(exc), parent=self.root)

    def playback_finished(self) -> None:
        self.playing = False
        self.playback_timer = None

    def navigate(self, delta: int) -> None:
        if not self.visible:
            return
        selection = self.listbox.curselection()
        index = max(0, min(len(self.visible) - 1, (selection[0] if selection else 0) + delta))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.select()

    def mark(self, decision: str) -> None:
        if self.current is None:
            return
        index = self.visible.index(self.current)
        next_name = self.visible[index + 1].name if index + 1 < len(self.visible) else None
        try:
            self.store.mark(self.current.name, decision)
            self.refresh(next_name)
        except Exception as exc:
            messagebox.showerror("Could not save decision", str(exc), parent=self.root)

    def undo(self) -> None:
        if self.store is None:
            return
        try:
            name = self.store.undo()
            if name:
                self.status_filter.set("All")
                self.speaker_filter.set("All speakers")
                self.refresh(name)
        except Exception as exc:
            messagebox.showerror("Could not undo", str(exc), parent=self.root)

    def export(self) -> None:
        if self.store is None:
            return
        destination = filedialog.askdirectory(parent=self.root, title="Copy kept clips to a separate folder")
        if destination:
            try:
                count = self.store.export_kept(Path(destination))
                messagebox.showinfo("Export complete", f"Copied {count} kept clips. Originals are unchanged.", parent=self.root)
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc), parent=self.root)

    def close(self) -> None:
        self.stop()
        self.temporary.cleanup()
        self.root.destroy()


def main() -> None:
    """Launch the Windows reviewer with an optional extraction folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This reviewer uses Windows audio playback (winsound)")
    root = tk.Tk()
    app = ClipReviewer(root)
    if args.folder:
        root.after(0, lambda: app.open_folder(args.folder))
    root.mainloop()


if __name__ == "__main__":
    main()
