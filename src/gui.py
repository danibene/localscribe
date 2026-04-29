from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from localscribe.skeleton import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERLAP_SECONDS,
    default_output_text_path,
    download_model,
    transcribe_audio,
)


class LocalScribeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LocalScribe")
        self.is_busy = False
        self._last_auto_output_path = ""

        self.file_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.model_name_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.chunk_seconds_var = tk.StringVar(value=str(DEFAULT_CHUNK_SECONDS))
        self.overlap_seconds_var = tk.StringVar(value=str(DEFAULT_OVERLAP_SECONDS))
        self.status_var = tk.StringVar(value="Select an audio file to start")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.file_path_var.trace_add("write", self._sync_default_output_path)

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Audio file").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.file_path_var, width=60).grid(
            row=1, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(frame, text="Browse", command=self.browse_file).grid(
            row=1, column=1, sticky="ew"
        )

        ttk.Label(frame, text="Output text file").grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(frame, textvariable=self.output_path_var, width=60).grid(
            row=3, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(frame, text="Save as", command=self.browse_output_path).grid(
            row=3, column=1, sticky="ew"
        )

        ttk.Label(frame, text="Model").grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.model_name_var, width=20).grid(
            row=5, column=0, sticky="w"
        )

        ttk.Label(frame, text="Chunk seconds").grid(
            row=6, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(frame, textvariable=self.chunk_seconds_var, width=20).grid(
            row=7, column=0, sticky="w"
        )

        ttk.Label(frame, text="Overlap seconds").grid(
            row=8, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(frame, textvariable=self.overlap_seconds_var, width=20).grid(
            row=9, column=0, sticky="w"
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        ttk.Button(
            buttons, text="Download model", command=self.download_selected_model
        ).pack(side="left")
        ttk.Button(
            buttons, text="Transcribe", command=self.transcribe_selected_file
        ).pack(side="left", padx=(8, 0))

        ttk.Progressbar(frame, variable=self.progress_var, maximum=100).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=(16, 0)
        )
        ttk.Label(
            frame, textvariable=self.status_var, wraplength=500, justify="left"
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(8, 0))

        frame.columnconfigure(0, weight=1)

    def _default_output_path_for(self, file_path: str) -> str:
        if not file_path.strip():
            return ""
        try:
            return str(default_output_text_path(file_path))
        except Exception:
            return ""

    def _sync_default_output_path(self, *_args) -> None:
        current_file_path = self.file_path_var.get().strip()
        current_output_path = self.output_path_var.get().strip()
        next_default_output_path = self._default_output_path_for(current_file_path)
        if not next_default_output_path:
            return
        if (
            not current_output_path
            or current_output_path == self._last_auto_output_path
        ):
            self.output_path_var.set(next_default_output_path)
            self._last_auto_output_path = next_default_output_path

    def browse_file(self) -> None:
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)

    def browse_output_path(self) -> None:
        default_output_path = (
            self.output_path_var.get().strip()
            or self._default_output_path_for(self.file_path_var.get().strip())
        )
        dialog_kwargs: dict[str, str | list[tuple[str, str]]] = {
            "defaultextension": ".txt",
            "filetypes": [("Text files", "*.txt"), ("All files", "*.*")],
        }
        if default_output_path:
            default_path = Path(default_output_path).expanduser()
            dialog_kwargs["initialdir"] = str(default_path.parent)
            dialog_kwargs["initialfile"] = default_path.name

        output_path = filedialog.asksaveasfilename(**dialog_kwargs)
        if output_path:
            self.output_path_var.set(output_path)

    def _progress_callback(self, fraction: float, message: str) -> None:
        def apply_update() -> None:
            self.progress_var.set(max(0.0, min(100.0, fraction * 100.0)))
            self.status_var.set(message)

        self.root.after(0, apply_update)

    def _run_background(self, target, *, success_message: str | None) -> None:
        if self.is_busy:
            return
        self.is_busy = True
        self.progress_var.set(0.0)

        def worker() -> None:
            try:
                target()
                if success_message:
                    self.root.after(0, lambda: self.status_var.set(success_message))
            except Exception as exc:  # pragma: no cover - GUI exception path
                exception_message = str(exc)
                self.root.after(
                    0, lambda: messagebox.showerror("LocalScribe", exception_message)
                )
                self.root.after(
                    0, lambda: self.status_var.set(f"Error: {exception_message}")
                )
            finally:
                self.root.after(0, self._mark_idle)

        threading.Thread(target=worker, daemon=True).start()

    def _mark_idle(self) -> None:
        self.is_busy = False

    def download_selected_model(self) -> None:
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        self._run_background(
            lambda: download_model(
                model_name=model_name, progress_callback=self._progress_callback
            ),
            success_message=f"Model '{model_name}' is ready",
        )

    def transcribe_selected_file(self) -> None:
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showerror("LocalScribe", "Please choose an audio file first")
            return

        output_path = (
            self.output_path_var.get().strip()
            or self._default_output_path_for(file_path)
        )
        chunk_seconds = int(
            self.chunk_seconds_var.get().strip() or DEFAULT_CHUNK_SECONDS
        )
        overlap_seconds = int(
            self.overlap_seconds_var.get().strip() or DEFAULT_OVERLAP_SECONDS
        )
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME

        def job() -> None:
            result = transcribe_audio(
                file_path=file_path,
                output_path=output_path,
                model_name=model_name,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                progress_callback=self._progress_callback,
            )
            output_text_path = Path(result["output_text_path"])
            output_json_path = Path(result["output_json_path"])
            output_segments_path = Path(result["output_segments_path"])
            self.root.after(
                0,
                lambda: self.status_var.set(
                    "Saved transcript files to "
                    f"{output_text_path}, {output_json_path}, "
                    f"and {output_segments_path}"
                ),
            )

        self._run_background(job, success_message=None)


def scribe() -> None:
    root = tk.Tk()
    LocalScribeApp(root)
    root.mainloop()


if __name__ == "__main__":
    scribe()
