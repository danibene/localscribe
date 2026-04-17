import argparse
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

import whisper
from whisper import Whisper

try:
    from localscribe import __version__
except Exception:
    __version__ = "0.0.0"

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[dict], None]]


def _emit(progress_callback: ProgressCallback, **payload):
    if progress_callback is not None:
        progress_callback(payload)


def get_model(
    progress_callback: ProgressCallback = None, model_name: str = "base"
) -> Whisper:
    ssl._create_default_https_context = ssl._create_unverified_context
    _emit(
        progress_callback,
        stage="loading_model",
        message=f"Loading Whisper model '{model_name}'",
        progress=0,
    )
    model = whisper.load_model(model_name)
    _emit(
        progress_callback,
        stage="model_loaded",
        message=f"Model '{model_name}' loaded",
        progress=5,
    )
    return model


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


def transcribe_audio(
    file_path: str,
    output_path: str = "transcription.txt",
    progress_callback: ProgressCallback = None,
    chunk_seconds: int = 60,
    model_name: str = "base",
):
    model = get_model(progress_callback=progress_callback, model_name=model_name)

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


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with chunked progress updates"
    )
    parser.add_argument("file_path", help="Path to the audio file to transcribe")
    parser.add_argument(
        "--output", default="transcription.txt", help="Output transcript file"
    )
    parser.add_argument(
        "--chunk-seconds", type=int, default=60, help="Chunk size in seconds"
    )
    parser.add_argument("--model-name", default="base", help="Whisper model name")
    parser.add_argument(
        "--version",
        action="version",
        version=f"localscribe {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    def console_progress(update: dict):
        progress = update.get("progress")
        message = update.get("message", "")
        if progress is None:
            print(message)
        else:
            print(f"[{progress:3d}%] {message}")

    transcribe_audio(
        file_path=args.file_path,
        output_path=args.output,
        progress_callback=console_progress,
        chunk_seconds=args.chunk_seconds,
        model_name=args.model_name,
    )
    _logger.info("Script ends here")


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
