"""
localscribe.skeleton

Core library + CLI entry points for localscribe.
"""

from __future__ import annotations

import argparse
import logging
import os
import ssl
import sys
from pathlib import Path
from typing import Optional

import whisper
from whisper import Whisper

from localscribe import __version__

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Runtime resource helpers
# ------------------------------------------------------------

def _resource_root() -> Path:
    """
    Return the directory where bundled resources live.

    In PyInstaller onefile builds this is sys._MEIPASS.
    In normal execution we use this module directory.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _ffmpeg_candidates() -> list[Path]:
    """
    Candidate locations where PyInstaller might place ffmpeg.
    """
    root = _resource_root()
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    return [
        root / "ffmpeg" / exe,  # preferred layout when we bundle to "ffmpeg/"
        root / exe,             # sometimes ends up at the root of _MEIPASS
        root / "bin" / exe,     # alternative layout
    ]


def _find_bundled_ffmpeg() -> Optional[Path]:
    for p in _ffmpeg_candidates():
        try:
            if p.exists():
                return p
        except OSError:
            # In rare cases (permissions/encoding), treat as not found.
            continue
    return None


def _model_root() -> Path:
    """
    Directory containing Whisper model weights.

    We expect PyInstaller to bundle this folder as "whisper_models"
    at runtime (_MEIPASS/whisper_models).
    """
    return _resource_root() / "whisper_models"


def _configure_runtime() -> None:
    """
    Configure runtime environment for bundled execution.

    - Ensure SSL context won't fail in locked-down environments
    - Ensure bundled ffmpeg is discoverable by subprocess/Whisper
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    ffmpeg = _find_bundled_ffmpeg()
    if ffmpeg is None:
        # Make the failure explicit and informative.
        tried = ", ".join(str(p) for p in _ffmpeg_candidates())
        raise RuntimeError(
            "Bundled ffmpeg not found. Whisper needs ffmpeg to decode audio.\n"
            f"Resource root: {_resource_root()}\n"
            f"Searched: {tried}\n"
            "Fix: ensure the PyInstaller build bundles ffmpeg.exe into either:\n"
            "  - ffmpeg/ffmpeg.exe (preferred)\n"
            "  - ffmpeg.exe (resource root)\n"
        )

    # Prepend directory so `ffmpeg` resolves in subprocess calls.
    ffmpeg_dir = str(ffmpeg.parent)
    current_path = os.environ.get("PATH", "")
    if not current_path.startswith(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

    # Some libraries also honor this.
    os.environ["FFMPEG_BINARY"] = str(ffmpeg)


# ------------------------------------------------------------
# Whisper helpers
# ------------------------------------------------------------

def download_model(model_name: str = "base") -> None:
    """
    Download Whisper model into the local model directory.

    Safe to call multiple times.
    """
    # Note: download step may not need ffmpeg, but it doesn't hurt.
    ssl._create_default_https_context = ssl._create_unverified_context

    model_dir = _model_root()
    model_dir.mkdir(parents=True, exist_ok=True)

    whisper.load_model(model_name, download_root=str(model_dir))


def load_model(model_name: str = "base") -> Whisper:
    """
    Load Whisper model from the bundled model directory.
    """
    _configure_runtime()

    model_dir = _model_root()
    if not model_dir.exists():
        raise RuntimeError(
            "Whisper model directory not found.\n"
            f"Expected: {model_dir}\n"
            "Fix: bundle the downloaded model folder into the PyInstaller build "
            "as 'whisper_models'."
        )

    model = whisper.load_model(model_name, download_root=str(model_dir))
    if model is None:
        raise RuntimeError("Whisper returned None while loading the model.")
    return model


def transcribe_audio(file_path: str, model_name: str = "base") -> dict:
    """
    Transcribe an audio file and return the Whisper result dict.
    """
    _configure_runtime()

    src = Path(file_path)
    if not src.exists():
        raise RuntimeError(f"Input file not found: {src}")

    model = load_model(model_name)
    result = model.transcribe(str(src))

    if result is None:
        raise RuntimeError("Whisper returned None for transcription result.")
    if "text" not in result:
        raise RuntimeError(f"Unexpected Whisper result keys: {list(result.keys())}")

    return result


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(args):
    parser = argparse.ArgumentParser(description="Local Whisper transcription")

    parser.add_argument(
        "--version",
        action="version",
        version=f"localscribe {__version__}",
    )

    parser.add_argument("file_path", nargs="?", help="Path to audio file")
    return parser.parse_args(args)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main(args):
    setup_logging()
    ns = parse_args(args)

    if not ns.file_path:
        print("No file provided")
        sys.exit(1)

    try:
        res = transcribe_audio(ns.file_path)
        text = res.get("text", "")
        out = Path("transcription.txt")
        out.write_text(text, encoding="utf-8")
        print(f"Saved transcription to {out}")
        sys.exit(0)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()