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

from localscribe import __version__

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Runtime resource paths
# ------------------------------------------------------------

def _resource_root() -> Path:
    """
    Root directory for bundled resources.

    In PyInstaller onefile mode resources are extracted into
    sys._MEIPASS. In normal execution we use the module directory.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _model_root() -> Path:
    """Directory containing Whisper model weights."""
    return _resource_root() / "whisper_models"


def _ensure_bundled_ffmpeg_on_path() -> None:
    """
    Ensure the bundled ffmpeg executable is discoverable.
    """
    root = _resource_root()
    ffmpeg_dir = root / "ffmpeg"

    if not ffmpeg_dir.exists():
        return

    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = ffmpeg_dir / exe

    if not ffmpeg_path.exists():
        return

    path = os.environ.get("PATH", "")
    ffmpeg_dir_str = str(ffmpeg_dir)

    if ffmpeg_dir_str not in path:
        os.environ["PATH"] = ffmpeg_dir_str + os.pathsep + path


# ------------------------------------------------------------
# Whisper model helpers
# ------------------------------------------------------------

def download_model(model_name: str = "base") -> None:
    """
    Download Whisper model into the local model directory.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    model_dir = _model_root()
    model_dir.mkdir(parents=True, exist_ok=True)

    whisper.load_model(model_name, download_root=str(model_dir))


def load_model(model_name: str = "base") -> Whisper:
    """
    Load Whisper model from bundled directory.
    """
    _ensure_bundled_ffmpeg_on_path()

    ssl._create_default_https_context = ssl._create_unverified_context

    model_dir = _model_root()

    if not model_dir.exists():
        raise RuntimeError(
            f"Whisper model directory not found: {model_dir}"
        )

    model = whisper.load_model(model_name, download_root=str(model_dir))

    if model is None:
        raise RuntimeError("Whisper returned None while loading model")

    return model


def transcribe_audio(file_path: str, model_name: str = "base") -> dict:
    """
    Transcribe an audio file.
    """
    _ensure_bundled_ffmpeg_on_path()

    file = Path(file_path)

    if not file.exists():
        raise RuntimeError(f"Input file not found: {file}")

    model = load_model(model_name)

    result = model.transcribe(str(file))

    if result is None:
        raise RuntimeError("Whisper transcription returned None")

    return result


# ------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Local audio transcription using Whisper"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"localscribe {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const=logging.INFO,
    )

    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        action="store_const",
        const=logging.DEBUG,
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        help="Audio file path",
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    logging.basicConfig(
        level=loglevel or logging.WARNING,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ------------------------------------------------------------
# CLI entry points
# ------------------------------------------------------------

def download_model_cli():
    try:
        download_model()
        print("Model downloaded successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Model download failed: {e}")
        sys.exit(1)


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    if not args.file_path:
        print("No audio file provided")
        sys.exit(1)

    try:
        result = transcribe_audio(args.file_path)

        text = result["text"]

        output = Path("transcription.txt")
        output.write_text(text, encoding="utf-8")

        print(f"Saved transcription to {output}")

    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()