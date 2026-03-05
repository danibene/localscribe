import platform
import threading
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from localscribe.skeleton import transcribe_audio_safe


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LocalScribe")
        self.geometry("900x600")

        self._worker_thread: Optional[threading.Thread] = None
        self._last_transcript: Optional[str] = None
        self._last_source_path: Optional[Path] = None

        self._transcribe_start_ts: Optional[float] = None
        self._tick_job: Optional[str] = None

        top = tk.Frame(self)
        top.pack(fill="x", padx=12, pady=12)

        self.btn_browse = tk.Button(top, text="Select audio/video file…", command=self.on_browse)
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

    def _set_busy(self, busy: bool, status: str) -> None:
        self.lbl_status.configure(text=status)
        self.btn_browse.configure(state="disabled" if busy else "normal")

    def _start_tick(self) -> None:
        import time

        self._transcribe_start_ts = time.time()

        def tick():
            import time
            if self._transcribe_start_ts is None:
                return
            elapsed = int(time.time() - self._transcribe_start_ts)
            # keep whatever base status we had, but append elapsed
            base = self.lbl_status.cget("text").split("  |  ")[0]
            self.lbl_status.configure(text=f"{base}  |  {elapsed}s")
            self._tick_job = self.after(1000, tick)

        tick()

    def _stop_tick(self) -> None:
        self._transcribe_start_ts = None
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except Exception:
                pass
            self._tick_job = None

    def _write_debug_log(self, src: Path, text: str) -> Optional[Path]:
        try:
            log_path = src.with_suffix(src.suffix + ".localscribe.gui.log.txt")
            log_path.write_text(text, encoding="utf-8")
            return log_path
        except Exception:
            return None

    def _show_traceback_dialog(self, title: str, message: str, tb_text: str) -> None:
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("820x520")
        win.transient(self)
        win.grab_set()

        lbl = tk.Label(win, text=message, justify="left", anchor="w")
        lbl.pack(fill="x", padx=12, pady=(12, 6))

        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        txt = tk.Text(frame, wrap="none")
        txt.insert("1.0", tb_text)
        txt.configure(state="disabled")
        txt.pack(side="left", fill="both", expand=True)

        yscroll = tk.Scrollbar(frame, orient="vertical", command=txt.yview)
        yscroll.pack(side="right", fill="y")
        txt.configure(yscrollcommand=yscroll.set)

        bottom = tk.Frame(win)
        bottom.pack(fill="x", padx=12, pady=(0, 12))

        def copy_to_clipboard() -> None:
            self.clipboard_clear()
            self.clipboard_append(tb_text)
            self.update()
            messagebox.showinfo("LocalScribe", "Copied to clipboard.")

        tk.Button(bottom, text="Copy traceback", command=copy_to_clipboard).pack(side="left")
        tk.Button(bottom, text="Close", command=win.destroy).pack(side="right")

    def on_browse(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose an audio/video file",
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

        self.txt.delete("1.0", "end")
        self._last_transcript = None
        self._last_source_path = src
        self.btn_save_as.configure(state="disabled")

        self._set_busy(True, f"Transcribing: {src.name}")
        self._start_tick()

        def worker() -> None:
            try:
                # hard timeout: 5 minutes. Adjust if you want.
                result = transcribe_audio_safe(str(src), model_name="base", timeout_s=300)
                text = (result or {}).get("text", "")
                log_path = (result or {}).get("_localscribe_log_path", None)

                if not text.strip():
                    raise RuntimeError(
                        "Transcription returned empty text.\n"
                        f"Subprocess log: {log_path}"
                    )

                self.after(0, lambda: self._on_done(text, log_path))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error(e, tb))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _on_done(self, text: str, subproc_log_path: Optional[str]) -> None:
        self._stop_tick()
        self._set_busy(False, "Done")
        self._last_transcript = text

        self.txt.insert("1.0", text)
        self.btn_save_as.configure(state="normal")

        # Auto-save next to the source file
        if self._last_source_path is not None:
            out_path = self._last_source_path.with_suffix(self._last_source_path.suffix + ".txt")
            try:
                out_path.write_text(text, encoding="utf-8")
                if subproc_log_path:
                    self.lbl_status.configure(text=f"Saved: {out_path.name}  |  log: {Path(subproc_log_path).name}")
                else:
                    self.lbl_status.configure(text=f"Saved: {out_path.name}")
            except Exception:
                self.lbl_status.configure(text="Done (autosave failed; use Save As)")

    def _on_error(self, err: Exception, tb_text: str) -> None:
        self._stop_tick()
        self._set_busy(False, "Error")

        src = self._last_source_path
        where = f"\n\nFile: {src}" if src else ""

        env_info = (
            f"Time: {datetime.now().isoformat()}\n"
            f"Platform: {platform.platform()}\n"
            f"Python: {platform.python_version()}\n"
        )

        log_note = ""
        if src is not None:
            log_text = env_info + "\n" + tb_text
            log_path = self._write_debug_log(src, log_text)
            if log_path is not None:
                log_note = f"\n\nGUI debug log saved to:\n{log_path}"

        msg = (
            "Transcription failed.\n\n"
            f"{where}\n\n"
            f"Error:\n{repr(err)}\n"
            f"{log_note}\n\n"
            f"Environment:\n{env_info}"
        )

        messagebox.showerror("LocalScribe", msg)
        self._show_traceback_dialog("LocalScribe - Traceback", "Full traceback:", tb_text)

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