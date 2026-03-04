"""
localscribe.skeleton

Core library + CLI entry points for localscribe.
"""

import argparse
import logging
import ssl
import sys
from pathlib import Path

import whisper
from whisper import Whisper

from localscribe import __version__

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ==========================================================
# --------------------- Library API ------------------------
# ==========================================================

def download_model(model_name: str = "base") -> None:
    """
    Download and cache a Whisper model locally.

    This function is safe to call multiple times.
    """
    # Avoid SSL verification issues in some environments
    ssl._create_default_https_context = ssl._create_unverified_context
    whisper.load_model(model_name)


def load_model(model_name: str = "base") -> Whisper:
    """
    Load and return a Whisper model instance.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    return whisper.load_model(model_name)


def transcribe_audio(file_path: str, model_name: str = "base") -> dict:
    """
    Transcribe an audio file and return the transcription result.
    """
    model = load_model(model_name)
    result = model.transcribe(file_path)
    return result


# ==========================================================
# ----------------------- CLI Logic ------------------------
# ==========================================================

def parse_args(args):
    parser = argparse.ArgumentParser(description="Local audio transcription using Whisper")

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
        help="Set log level to INFO",
    )

    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        action="store_const",
        const=logging.DEBUG,
        help="Set log level to DEBUG",
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to the audio file to transcribe",
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel or logging.WARNING,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ==========================================================
# ------------------ CLI Entry Points ----------------------
# ==========================================================

def download_model_cli():
    """
    Console script entry point for downloading the Whisper model.
    Must NOT return a model object.
    """
    try:
        download_model()
        print("Whisper model downloaded successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


def main(args):
    """
    CLI entry for transcription.
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    if not args.file_path:
        print("Error: You must provide an audio file path.")
        sys.exit(1)

    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        _logger.info("Starting transcription...")
        result = transcribe_audio(str(file_path))

        output_file = Path("transcription.txt")
        output_file.write_text(result["text"])

        print(f"Transcription saved to {output_file}")
        sys.exit(0)

    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)


def run():
    """
    Entry point for main CLI command.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
