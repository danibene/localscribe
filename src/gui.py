import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from localscribe.skeleton import transcribe_audio


class LocalScribeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Audio Transcription App")
        self.root.geometry("760x520")

        self.progress_queue = queue.Queue()
        self.worker_thread = None
        self.selected_file = None

        self.file_var = tk.StringVar(value="No file selected")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.IntVar(value=0)
        self.chunk_seconds_var = tk.IntVar(value=60)
        self.model_var = tk.StringVar(value="base")

        self._build_ui()
        self.root.after(100, self._poll_progress_queue)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        file_row = ttk.Frame(outer)
        file_row.pack(fill="x", pady=(0, 10))

        ttk.Button(file_row, text="Browse", command=self.browse_file).pack(side="left")
        ttk.Label(file_row, textvariable=self.file_var).pack(
            side="left", padx=(10, 0), fill="x", expand=True
        )

        options_row = ttk.Frame(outer)
        options_row.pack(fill="x", pady=(0, 10))

        ttk.Label(options_row, text="Chunk size (seconds)").pack(side="left")
        ttk.Spinbox(
            options_row,
            from_=15,
            to=600,
            increment=15,
            textvariable=self.chunk_seconds_var,
            width=8,
        ).pack(side="left", padx=(8, 20))

        ttk.Label(options_row, text="Model").pack(side="left")
        ttk.Combobox(
            options_row,
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=(8, 0))

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=(0, 10))

        self.start_button = ttk.Button(
            button_row, text="Start transcription", command=self.start_transcription
        )
        self.start_button.pack(side="left")

        self.progress = ttk.Progressbar(
            outer,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress.pack(fill="x", pady=(0, 8))

        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(0, 10))

        ttk.Label(outer, text="Log").pack(anchor="w")
        self.log_text = tk.Text(outer, height=20, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Audio and video files",
                    "*.mp3 *.wav *.m4a *.mp4 *.aac *.flac *.ogg *.webm *.mov *.mkv",
                ),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            self.selected_file = file_path
            self.file_var.set(file_path)
            self._append_log(f"Selected file: {file_path}")

    def start_transcription(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo(
                "Transcription running", "A transcription is already in progress."
            )
            return

        if not self.selected_file:
            messagebox.showwarning("No file selected", "Please choose a file first.")
            return

        file_path = self.selected_file
        output_path = str(Path(file_path).with_suffix(".transcription.txt"))
        chunk_seconds = int(self.chunk_seconds_var.get())
        model_name = self.model_var.get().strip()

        self.progress_var.set(0)
        self.status_var.set("Starting transcription...")
        self.start_button.configure(state="disabled")
        self._append_log("")
        self._append_log(f"Starting transcription for: {file_path}")
        self._append_log(f"Chunk size: {chunk_seconds} seconds | Model: {model_name}")

        self.worker_thread = threading.Thread(
            target=self._run_transcription,
            args=(file_path, output_path, chunk_seconds, model_name),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_transcription(
        self, file_path: str, output_path: str, chunk_seconds: int, model_name: str
    ):
        def progress_callback(update: dict):
            self.progress_queue.put(("progress", update))

        try:
            transcribe_audio(
                file_path=file_path,
                output_path=output_path,
                progress_callback=progress_callback,
                chunk_seconds=chunk_seconds,
                model_name=model_name,
            )
            self.progress_queue.put(("finished", {"output_path": output_path}))
        except Exception as exc:
            self.progress_queue.put(("error", {"message": str(exc)}))

    def _poll_progress_queue(self):
        try:
            while True:
                kind, payload = self.progress_queue.get_nowait()
                if kind == "progress":
                    self._handle_progress(payload)
                elif kind == "finished":
                    self.progress_var.set(100)
                    self.status_var.set(f"Done. Saved to {payload['output_path']}")
                    self._append_log(
                        f"Finished. Transcript saved to: {payload['output_path']}"
                    )
                    self.start_button.configure(state="normal")
                elif kind == "error":
                    self.status_var.set("Failed")
                    self._append_log(f"Error: {payload['message']}")
                    self.start_button.configure(state="normal")
                    messagebox.showerror("Transcription failed", payload["message"])
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_progress_queue)

    def _handle_progress(self, update: dict):
        message = update.get("message", "")
        progress = update.get("progress")
        if progress is not None:
            self.progress_var.set(progress)
        if message:
            self.status_var.set(message)
            self._append_log(message)

    def _append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def scribe():
    root = tk.Tk()
    LocalScribeApp(root)
    root.mainloop()


if __name__ == "__main__":
    scribe()
