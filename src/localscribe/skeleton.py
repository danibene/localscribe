"""
localscribe.skeleton

Core library + CLI entry points for localscribe.
"""

from __future__ import annotations

import argparse
import logging
import os
import ssl
import subprocess
import sys
import tempfile
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
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _ffmpeg_candidates() -> list[Path]:
    root = _resource_root()
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    return [
        root / "ffmpeg" / exe,
        root / exe,
        root / "bin" / exe,
    ]


def _find_bundled_ffmpeg() -> Optional[Path]:
    for p in _ffmpeg_candidates():
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def _model_root() -> Path:
    return _resource_root() / "whisper_models"


def _configure_runtime() -> Path:
    """
    Configure runtime environment for bundled execution and return ffmpeg path.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    # Force CPU and avoid pathological thread explosions.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    ffmpeg = _find_bundled_ffmpeg()
    if ffmpeg is None:
        tried = ", ".join(str(p) for p in _ffmpeg_candidates())
        raise RuntimeError(
            "Bundled ffmpeg not found. Whisper needs ffmpeg to decode audio.\n"
            f"Resource root: {_resource_root()}\n"
            f"Searched: {tried}\n"
            "Fix: ensure the PyInstaller build bundles ffmpeg.exe into either:\n"
            "  - ffmpeg/ffmpeg.exe (preferred)\n"
            "  - ffmpeg.exe (resource root)\n"
        )

    ffmpeg_dir = str(ffmpeg.parent)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
    os.environ["FFMPEG_BINARY"] = str(ffmpeg)

    return ffmpeg


# ------------------------------------------------------------
# Audio pre-processing (avoid ffmpeg hangs inside whisper)
# ------------------------------------------------------------

def _convert_to_wav_16k_mono(
    src: Path,
    ffmpeg: Path,
    timeout_s: int = 60,
) -> Path:
    """
    Convert any media file to 16kHz mono WAV using ffmpeg.

    Returns a path to a temporary WAV file.
    """
    if not src.exists():
        raise RuntimeError(f"Input file not found: {src}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="localscribe_"))
    out_wav = tmp_dir / (src.stem + ".wav")

    # -vn: ignore video stream
    # -ac 1: mono
    # -ar 16000: 16kHz (what whisper expects)
    # -f wav: consistent container
    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_wav),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout_s}s while decoding:\n{src}\n\n"
            "This usually indicates a problematic codec/container or a hung ffmpeg process."
        )

    if proc.returncode != 0 or not out_wav.exists():
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            "ffmpeg failed to decode the input file.\n\n"
            f"File: {src}\n"
            f"Return code: {proc.returncode}\n"
            f"stderr:\n{stderr or '(no stderr)'}"
        )

    return out_wav


# ------------------------------------------------------------
# Whisper helpers
# ------------------------------------------------------------

def download_model(model_name: str = "base") -> None:
    ssl._create_default_https_context = ssl._create_unverified_context
    model_dir = _model_root()
    model_dir.mkdir(parents=True, exist_ok=True)
    whisper.load_model(model_name, download_root=str(model_dir))


def load_model(model_name: str = "base") -> Whisper:
    _configure_runtime()

    model_dir = _model_root()
    if not model_dir.exists():
        raise RuntimeError(
            "Whisper model directory not found.\n"
            f"Expected: {model_dir}\n"
            "Fix: bundle the downloaded model folder into the PyInstaller build as 'whisper_models'."
        )

    model = whisper.load_model(model_name, download_root=str(model_dir))
    if model is None:
        raise RuntimeError("Whisper returned None while loading the model.")
    return model


def transcribe_audio(file_path: str, model_name: str = "base") -> dict:
    ffmpeg = _configure_runtime()

    src = Path(file_path)
    if not src.exists():
        raise RuntimeError(f"Input file not found: {src}")

    # Convert first, with a timeout, so we don't hang forever inside whisper/audio.py
    wav_path = _convert_to_wav_16k_mono(src, ffmpeg=ffmpeg, timeout_s=60)

    model = load_model(model_name)

    # You can optionally add: fp16=False to avoid GPU expectations
    result = model.transcribe(str(wav_path), fp16=False)

    if result is None or "text" not in result:
        raise RuntimeError("Unexpected Whisper transcription result.")

    return result


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(args):
    parser = argparse.ArgumentParser(description="Local Whisper transcription")
    parser.add_argument("--version", action="version", version=f"localscribe {__version__}")
    parser.add_argument("file_path", nargs="?", help="Path to audio/video file")
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
        out = Path("transcription.txt")
        out.write_text(res.get("text", ""), encoding="utf-8")
        print(f"Saved transcription to {out}")
        sys.exit(0)
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()