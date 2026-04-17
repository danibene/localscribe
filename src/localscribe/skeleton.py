import json
import logging
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

try:
    from localscribe import __version__
except Exception:
    __version__ = "0.0.0"

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[dict], None]]


class CliUsageError(ValueError):
    pass


HELP_TEXT = """LocalScribe

Usage
  localscribe <file_path> [--output PATH] [--chunk-seconds N] [--model-name NAME] [-v|-vv]
  localscribe --download-model [--model-name NAME] [-v|-vv]
  localscribe --help
  localscribe --version

Examples
  localscribe audio.mp3
  localscribe audio.mp3 --output my_transcript.txt --chunk-seconds 30 --model-name small
  localscribe --download-model --model-name base
"""


def _emit(progress_callback: ProgressCallback, **payload):
    if progress_callback is not None:
        progress_callback(payload)


_whisper_module = None


def _get_whisper_module():
    global _whisper_module
    if _whisper_module is None:
        import whisper

        _whisper_module = whisper
    return _whisper_module


# ---- Python API ----


def download_model(
    model_name: str = "base", progress_callback: ProgressCallback = None
):
    """Download/load a Whisper model and return it.

    Keeping this function preserves the previous public API while allowing the
    GUI and CLI to call the same backend.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    _emit(
        progress_callback,
        stage="loading_model",
        message=f"Loading Whisper model '{model_name}'",
        progress=0,
        model_name=model_name,
    )
    whisper = _get_whisper_module()
    model = whisper.load_model(model_name)
    _emit(
        progress_callback,
        stage="model_loaded",
        message=f"Model '{model_name}' loaded",
        progress=100,
        model_name=model_name,
    )
    return model


def get_model(progress_callback: ProgressCallback = None, model_name: str = "base"):
    """Backward-compatible alias for older code paths."""
    return download_model(model_name=model_name, progress_callback=progress_callback)


def _require_binary(binary_name: str):
    if shutil.which(binary_name) is None:
        raise RuntimeError(
            f"Required binary '{binary_name}' was not found in PATH. Install ffmpeg/ffprobe first."
        )


def get_audio_duration_seconds(file_path: str) -> Optional[float]:
    _require_binary("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        return None
    return float(duration)


def chunk_audio_with_ffmpeg(file_path: str, chunk_seconds: int, output_dir: str):
    _require_binary("ffmpeg")
    output_pattern = os.path.join(output_dir, "chunk_%04d.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        output_pattern,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    chunks = sorted(str(p) for p in Path(output_dir).glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("No audio chunks were created.")
    return chunks


def _format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _default_output_path(file_path: str) -> str:
    return str(Path(file_path).with_suffix(".transcription.txt"))


def transcribe_audio(
    file_path: str,
    output_path: Optional[str] = None,
    progress_callback: ProgressCallback = None,
    chunk_seconds: int = 60,
    model_name: str = "base",
):
    if not file_path:
        raise ValueError("file_path is required")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")

    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if output_path is None:
        output_path = _default_output_path(file_path)

    model = download_model(model_name=model_name, progress_callback=progress_callback)

    _emit(
        progress_callback,
        stage="probing_audio",
        message="Reading audio duration",
        progress=8,
    )
    duration = get_audio_duration_seconds(file_path)
    if duration is None or duration <= 0:
        raise RuntimeError("Could not determine audio duration.")

    estimated_chunks = max(1, int((duration + chunk_seconds - 1) // chunk_seconds))
    _emit(
        progress_callback,
        stage="chunking_audio",
        message=f"Splitting audio into about {estimated_chunks} chunk(s)",
        progress=10,
        duration_seconds=duration,
        estimated_chunks=estimated_chunks,
    )

    all_text_parts = []
    all_segments = []

    with tempfile.TemporaryDirectory(prefix="localscribe_chunks_") as temp_dir:
        chunk_paths = chunk_audio_with_ffmpeg(
            file_path, chunk_seconds=chunk_seconds, output_dir=temp_dir
        )
        total_chunks = len(chunk_paths)
        _emit(
            progress_callback,
            stage="chunks_ready",
            message=f"Created {total_chunks} chunk(s)",
            progress=15,
            total_chunks=total_chunks,
        )

        for index, chunk_path in enumerate(chunk_paths, start=1):
            chunk_start = (index - 1) * chunk_seconds
            _emit(
                progress_callback,
                stage="transcribing_chunk",
                message=f"Transcribing chunk {index}/{total_chunks}",
                chunk_index=index,
                total_chunks=total_chunks,
                chunk_start_seconds=chunk_start,
                progress=15 + int(80 * (index - 1) / total_chunks),
            )

            result = model.transcribe(chunk_path, verbose=False)
            chunk_text = (result.get("text") or "").strip()
            if chunk_text:
                all_text_parts.append(chunk_text)

            for segment in result.get("segments", []):
                all_segments.append(
                    {
                        "start": segment["start"] + chunk_start,
                        "end": segment["end"] + chunk_start,
                        "text": segment["text"].strip(),
                    }
                )

            _emit(
                progress_callback,
                stage="chunk_done",
                message=f"Finished chunk {index}/{total_chunks}",
                chunk_index=index,
                total_chunks=total_chunks,
                progress=15 + int(80 * index / total_chunks),
            )

    final_text = "\n\n".join(part for part in all_text_parts if part)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    segments_output_path = str(Path(output_path).with_suffix(".segments.txt"))
    with open(segments_output_path, "w", encoding="utf-8") as f:
        for segment in all_segments:
            start = _format_timestamp(segment["start"])
            end = _format_timestamp(segment["end"])
            f.write(f"[{start} --> {end}] {segment['text']}\n")

    json_output_path = str(Path(output_path).with_suffix(".json"))
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": final_text,
                "segments": all_segments,
                "source_file": file_path,
                "chunk_seconds": chunk_seconds,
                "model_name": model_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _emit(
        progress_callback,
        stage="done",
        message=f"Done. Saved to {output_path}",
        progress=100,
        output_path=output_path,
        segments_output_path=segments_output_path,
        json_output_path=json_output_path,
    )

    return {
        "text": final_text,
        "segments": all_segments,
        "output_path": output_path,
        "segments_output_path": segments_output_path,
        "json_output_path": json_output_path,
    }


# ---- CLI ----


def _print_help():
    print(HELP_TEXT.strip())


def parse_cli_args(args):
    options = {
        "file_path": None,
        "output": None,
        "chunk_seconds": 60,
        "model_name": "base",
        "loglevel": logging.WARNING,
        "download_model_only": False,
    }

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("-h", "--help"):
            options["show_help"] = True
            return options
        if arg == "--version":
            options["show_version"] = True
            return options
        if arg == "-v":
            options["loglevel"] = logging.INFO
            i += 1
            continue
        if arg == "-vv":
            options["loglevel"] = logging.DEBUG
            i += 1
            continue
        if arg == "--download-model":
            options["download_model_only"] = True
            i += 1
            continue
        if arg == "--output":
            if i + 1 >= len(args):
                raise CliUsageError("--output requires a value")
            options["output"] = args[i + 1]
            i += 2
            continue
        if arg == "--chunk-seconds":
            if i + 1 >= len(args):
                raise CliUsageError("--chunk-seconds requires a value")
            try:
                options["chunk_seconds"] = int(args[i + 1])
            except ValueError as exc:
                raise CliUsageError("--chunk-seconds must be an integer") from exc
            i += 2
            continue
        if arg == "--model-name":
            if i + 1 >= len(args):
                raise CliUsageError("--model-name requires a value")
            options["model_name"] = args[i + 1]
            i += 2
            continue
        if arg.startswith("-"):
            raise CliUsageError(f"Unknown option: {arg}")
        if options["file_path"] is not None:
            raise CliUsageError("Only one input file can be provided")
        options["file_path"] = arg
        i += 1

    return options


def setup_logging(loglevel):
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _console_progress(update: dict):
    progress = update.get("progress")
    message = update.get("message", "")
    if progress is None:
        print(message)
    else:
        print(f"[{int(progress):3d}%] {message}")


def main(args):
    try:
        options = parse_cli_args(args)
    except CliUsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        _print_help()
        return 2

    if options.get("show_help"):
        _print_help()
        return 0

    if options.get("show_version"):
        print(f"localscribe {__version__}")
        return 0

    setup_logging(options["loglevel"])

    if options["download_model_only"]:
        download_model(
            model_name=options["model_name"],
            progress_callback=_console_progress,
        )
        return 0

    if not options["file_path"]:
        print(
            "Error: file_path is required unless --download-model is used",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        _print_help()
        return 2

    transcribe_audio(
        file_path=options["file_path"],
        output_path=options["output"],
        progress_callback=_console_progress,
        chunk_seconds=options["chunk_seconds"],
        model_name=options["model_name"],
    )
    _logger.info("Script ends here")
    return 0


def run():
    sys.exit(main(sys.argv[1:]))


def download_model_main(args):
    """Dedicated wrapper for the getwhispermodel entry point."""
    cli_args = list(args)
    if "--download-model" not in cli_args:
        cli_args = ["--download-model"] + cli_args
    return main(cli_args)


def download_model_run():
    sys.exit(download_model_main(sys.argv[1:]))


if __name__ == "__main__":
    run()
