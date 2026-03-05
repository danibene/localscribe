"""
localscribe.skeleton

Core library + CLI entry points for localscribe.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

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
        root / "ffmpeg" / exe,  # preferred
        root / exe,             # sometimes ends up at root
        root / "bin" / exe,     # alternative
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


def _configure_runtime(progress: Optional[Callable[[str], None]] = None) -> Path:
    """
    Configure runtime for bundled execution and return ffmpeg path.

    - SSL unverified context (avoids some locked-down SSL issues)
    - ffmpeg discovery + PATH prepending
    - conservative CPU threading defaults (but not forced to 1)
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    # Do NOT hard-force OMP=1; that can make even short clips feel "stuck".
    # Cap threads to something reasonable; user can override with LOCALSCRIBE_THREADS.
    threads = os.environ.get("LOCALSCRIBE_THREADS", "").strip()
    if not threads:
        # conservative default: 4 threads if available
        os.environ.setdefault("LOCALSCRIBE_THREADS", "4")

    # Avoid GPU surprises
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Apply thread cap to common backends
    os.environ.setdefault("OMP_NUM_THREADS", os.environ["LOCALSCRIBE_THREADS"])
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["LOCALSCRIBE_THREADS"])

    # Also tell torch directly if available
    try:
        import torch  # type: ignore
        n = int(os.environ["LOCALSCRIBE_THREADS"])
        if n > 0:
            torch.set_num_threads(n)
            torch.set_num_interop_threads(1)
    except Exception:
        pass

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

    # Some libraries honor this, but Whisper mostly uses PATH
    os.environ["FFMPEG_BINARY"] = str(ffmpeg)

    if progress:
        progress(f"Using ffmpeg: {ffmpeg}")

    return ffmpeg


# ------------------------------------------------------------
# Audio pre-processing (deterministic decoding)
# ------------------------------------------------------------

def _convert_to_wav_16k_mono(
    src: Path,
    ffmpeg: Path,
    timeout_s: int = 90,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert any media file to 16kHz mono WAV using ffmpeg, with a timeout.

    Returns a path to a temporary WAV file.
    """
    if not src.exists():
        raise RuntimeError(f"Input file not found: {src}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="localscribe_"))
    out_wav = tmp_dir / (src.stem + ".wav")

    if progress:
        progress("Decoding media with ffmpeg (to 16kHz mono WAV)...")

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

    if progress:
        progress(f"Decoded OK: {out_wav.name}")

    return out_wav


# ------------------------------------------------------------
# Whisper helpers
# ------------------------------------------------------------

def download_model(model_name: str = "base") -> None:
    ssl._create_default_https_context = ssl._create_unverified_context
    model_dir = _model_root()
    model_dir.mkdir(parents=True, exist_ok=True)
    whisper.load_model(model_name, download_root=str(model_dir))


def load_model(model_name: str = "base", progress: Optional[Callable[[str], None]] = None) -> Whisper:
    _configure_runtime(progress=progress)

    model_dir = _model_root()
    if not model_dir.exists():
        raise RuntimeError(
            "Whisper model directory not found.\n"
            f"Expected: {model_dir}\n"
            "Fix: bundle the downloaded model folder into the PyInstaller build as 'whisper_models'."
        )

    if progress:
        progress(f"Loading Whisper model '{model_name}'...")

    t0 = time.time()
    model = whisper.load_model(model_name, download_root=str(model_dir))
    dt = time.time() - t0

    if model is None:
        raise RuntimeError("Whisper returned None while loading the model.")

    if progress:
        progress(f"Model loaded in {dt:.1f}s")

    return model


def transcribe_audio_inprocess(
    file_path: str,
    model_name: str = "base",
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    In-process transcription (can hang if torch/ffmpeg stalls).
    Prefer transcribe_audio_safe() from the GUI.
    """
    ffmpeg = _configure_runtime(progress=progress)

    src = Path(file_path)
    wav_path = _convert_to_wav_16k_mono(src, ffmpeg=ffmpeg, timeout_s=90, progress=progress)

    model = load_model(model_name, progress=progress)

    if progress:
        progress("Transcribing with Whisper...")

    t0 = time.time()
    result = model.transcribe(str(wav_path), fp16=False)
    dt = time.time() - t0

    if result is None or "text" not in result:
        raise RuntimeError("Unexpected Whisper transcription result.")

    if progress:
        progress(f"Transcription finished in {dt:.1f}s")

    return result


def transcribe_audio_safe(
    file_path: str,
    model_name: str = "base",
    timeout_s: int = 300,
) -> dict:
    """
    Robust transcription that runs in a separate process with a hard timeout.

    This prevents the GUI from getting stuck forever if torch/ffmpeg hangs.
    """
    # output json goes to a temp file; the child writes it
    out_json = Path(tempfile.mkstemp(prefix="localscribe_out_", suffix=".json")[1])

    # capture logs to a temp file too
    out_log = Path(tempfile.mkstemp(prefix="localscribe_subproc_", suffix=".log.txt")[1])

    cmd = [
        sys.executable,
        "--_internal-transcribe",
        "--input",
        str(Path(file_path).resolve()),
        "--output",
        str(out_json),
        "--model",
        model_name,
    ]

    with out_log.open("w", encoding="utf-8") as logf:
        try:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=logf,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Transcription timed out after {timeout_s}s.\n\n"
                f"Subprocess log saved to:\n{out_log}"
            )

    if proc.returncode != 0:
        raise RuntimeError(
            "Transcription subprocess failed.\n\n"
            f"Subprocess log saved to:\n{out_log}"
        )

    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(
            "Could not read transcription output JSON.\n\n"
            f"Error: {e}\n"
            f"Subprocess log saved to:\n{out_log}"
        )

    # attach log path for debugging in the GUI
    data["_localscribe_log_path"] = str(out_log)

    return data


def _internal_transcribe_main(inp: Path, outp: Path, model_name: str) -> int:
    """
    Internal entry point used by transcribe_audio_safe().
    Writes JSON to outp.
    """
    try:
        def p(msg: str) -> None:
            print(msg, flush=True)

        p("=== LocalScribe internal transcription ===")
        p(f"resource_root: {_resource_root()}")
        ffmpeg = _configure_runtime(progress=p)
        p(f"ffmpeg: {ffmpeg}")
        p(f"model_root: {_model_root()}")

        t0 = time.time()
        result = transcribe_audio_inprocess(str(inp), model_name=model_name, progress=p)
        dt = time.time() - t0

        payload = {
            "text": result.get("text", ""),
            "language": result.get("language", None),
            "elapsed_s": dt,
        }
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        p("Wrote output JSON.")
        return 0
    except Exception as e:
        err_payload = {
            "error": repr(e),
        }
        try:
            outp.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print("ERROR:", repr(e), flush=True)
        return 1


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Local Whisper transcription")

    parser.add_argument("--version", action="version", version=f"localscribe {__version__}")

    # Internal mode used by the GUI to enforce timeouts
    parser.add_argument("--_internal-transcribe", action="store_true", default=False)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="base")

    # Regular CLI usage
    parser.add_argument("file_path", nargs="?", default=None, help="Path to audio/video file")

    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main(argv):
    setup_logging()
    ns = parse_args(argv)

    # Internal mode: used by GUI subprocess wrapper
    if ns._internal_transcribe:
        if not ns.input or not ns.output:
            print("Missing --input/--output for internal mode.", flush=True)
            return 2
        return _internal_transcribe_main(Path(ns.input), Path(ns.output), ns.model)

    # Normal CLI
    if not ns.file_path:
        print("No file provided", flush=True)
        return 1

    try:
        res = transcribe_audio_inprocess(ns.file_path, model_name=ns.model, progress=lambda m: print(m, flush=True))
        out = Path("transcription.txt")
        out.write_text(res.get("text", ""), encoding="utf-8")
        print(f"Saved transcription to {out}", flush=True)
        return 0
    except Exception as e:
        print(f"Transcription failed: {e}", flush=True)
        return 1


def run():
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()