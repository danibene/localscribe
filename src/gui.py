import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from localscribe.skeleton import transcribe_audio


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LocalScribe")
        self.geometry("900x600")

        self._worker_thread: Optional[threading.Thread] = None

        top = tk.Frame(self)
        top.pack(fill="x", padx=12, pady=12)

        self.btn_browse = tk.Button(top, text="Select audio file…", command=self.on_browse)
        self.btn_browse.pack(side="left")

        self.btn_save_as = tk.Button(
            top,
            text="Save transcription as…",
            command=self.on_save_as,
            state="disabled",
        )
        self.btn_save_as.pack(side="left", padx=(8, 0))

        self.lbl_status = tk.Label(top, text="Idle")
        self.lbl_status.pack(side="left", padx=(12, 0))

        self.txt = tk.Text(self, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._last_transcript: Optional[str] = None
        self._last_source_path: Optional[Path] = None

    def _set_busy(self, busy: bool, status: str) -> None:
        self.lbl_status.configure(text=status)
        self.btn_browse.configure(state="disabled" if busy else "normal")

    def on_browse(self) -> None:
        # Common audio/video types supported by ffmpeg/whisper.
        file_path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[
                ("Audio/Video", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.wma *.mp4 *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        src = Path(file_path)
        if not src.exists():
            messagebox.showerror("LocalScribe", f"File not found:\n{src}")
            return

        # Clear UI and run transcription in a background thread so the window doesn't freeze.
        self.txt.delete("1.0", "end")
        self._last_transcript = None
        self._last_source_path = src
        self.btn_save_as.configure(state="disabled")

        self._set_busy(True, f"Transcribing: {src.name} …")

        def worker() -> None:
            try:
                result = transcribe_audio(str(src))
                text = (result or {}).get("text", "")
                self.after(0, lambda: self._on_done(text))
            except Exception as e:
                self.after(0, lambda: self._on_error(e))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _on_done(self, text: str) -> None:
        self._set_busy(False, "Done")
        self._last_transcript = text

        self.txt.insert("1.0", text)
        self.btn_save_as.configure(state="normal")

        # Auto-save next to the audio file (nice default), and tell the user where it went.
        if self._last_source_path is not None:
            out_path = self._last_source_path.with_suffix(self._last_source_path.suffix + ".txt")
            try:
                out_path.write_text(text, encoding="utf-8")
                self.lbl_status.configure(text=f"Saved: {out_path.name}")
            except Exception:
                # If autosave fails, user can still use Save As.
                self.lbl_status.configure(text="Done (autosave failed; use Save As)")

    def _on_error(self, err: Exception) -> None:
        self._set_busy(False, "Error")
        messagebox.showerror(
            "LocalScribe",
            "Transcription failed.\n\n"
            "Most common causes:\n"
            "• ffmpeg missing or not found\n"
            "• unsupported/corrupted media file\n"
            "• model download blocked by network/SSL\n\n"
            f"Error:\n{err}",
        )

    def on_save_as(self) -> None:
        if not self._last_transcript:
            return

        initial_name = "transcription.txt"
        if self._last_source_path is not None:
            initial_name = self._last_source_path.stem + ".txt"

        out_file = filedialog.asksaveasfilename(
            title="Save transcription",
            defaultextension=".txt",
            initialfile=initial_name,
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not out_file:
            return

        try:
            Path(out_file).write_text(self._last_transcript, encoding="utf-8")
            self.lbl_status.configure(text=f"Saved: {Path(out_file).name}")
        except Exception as e:
            messagebox.showerror("LocalScribe", f"Could not save file:\n{e}")


def scribe() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    scribe()
