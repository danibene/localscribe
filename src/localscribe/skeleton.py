"""
localscribe.skeleton

Core library + CLI entry points for localscribe.
"""

import argparse
import logging
import os
import ssl
import sys
from pathlib import Path

import whisper
from whisper import Whisper

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Runtime resource helpers
# ------------------------------------------------------------

def _resource_root() -> Path:
    """
    Return the directory where bundled resources live.

    In PyInstaller onefile builds this is sys._MEIPASS.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)

    return Path(__file__).resolve().parent


def _ffmpeg_path() -> Path | None:
    """
    Return bundled ffmpeg path if present.
    """
    root = _resource_root()

    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    candidates = [
        root / exe,
        root / "ffmpeg" / exe,
        root / "bin" / exe,
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def _model_root() -> Path:
    return _resource_root() / "whisper_models"


def _configure_runtime() -> None:
    """
    Configure runtime environment for bundled execution.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    ffmpeg = _ffmpeg_path()

    if ffmpeg:
        # Force Whisper to use this ffmpeg
        os.environ["PATH"] = str(ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = str(ffmpeg)


# ------------------------------------------------------------
# Whisper helpers
# ------------------------------------------------------------

def download_model(model_name: str = "base") -> None:
    """
    Download Whisper model.
    """
    _configure_runtime()

    model_dir = _model_root()
    model_dir.mkdir(parents=True, exist_ok=True)

    whisper.load_model(model_name, download_root=str(model_dir))


def load_model(model_name: str = "base") -> Whisper:
    """
    Load Whisper model.
    """
    _configure_runtime()

    model_dir = _model_root()

    if not model_dir.exists():
        raise RuntimeError(
            f"Whisper model directory not found: {model_dir}"
        )

    model = whisper.load_model(model_name, download_root=str(model_dir))

    if model is None:
        raise RuntimeError("Failed to load Whisper model")

    return model


def transcribe_audio(file_path: str, model_name: str = "base") -> dict:
    """
    Transcribe audio file.
    """
    _configure_runtime()

    file = Path(file_path)

    if not file.exists():
        raise RuntimeError(f"Audio file not found: {file}")

    model = load_model(model_name)

    result = model.transcribe(str(file))

    if not result:
        raise RuntimeError("Whisper returned empty transcription result")

    return result


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(args):
    parser = argparse.ArgumentParser(description="Local Whisper transcription")

    parser.add_argument("file_path", nargs="?", help="Audio file")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    if not args.file_path:
        print("No file provided")
        sys.exit(1)

    try:
        result = transcribe_audio(args.file_path)

        text = result["text"]

        out = Path("transcription.txt")
        out.write_text(text, encoding="utf-8")

        print(f"Saved transcription to {out}")

    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()