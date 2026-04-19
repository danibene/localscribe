from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import URLError

from localscribe import __version__
from localscribe.packaging import get_runtime_ffmpeg_binary, get_runtime_model_dir

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
DEFAULT_MODEL_NAME = "base"
DEFAULT_CHUNK_SECONDS = 60
DEFAULT_OVERLAP_SECONDS = 5


def _import_whisper():
    import whisper

    return whisper


def _emit_progress(
    progress_callback: ProgressCallback | None, fraction: float, message: str
) -> None:
    clipped = max(0.0, min(1.0, float(fraction)))
    if progress_callback is not None:
        progress_callback(clipped, message)


def _stdout_progress(prefix: str = "[localscribe]") -> ProgressCallback:
    def callback(fraction: float, message: str) -> None:
        percent = int(round(max(0.0, min(1.0, fraction)) * 100))
        print(f"{prefix} {percent:3d}% {message}", flush=True)

    return callback


def _ensure_existing_file(file_path: str | os.PathLike[str]) -> Path:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def default_output_text_path(file_path: str | os.PathLike[str]) -> Path:
    source_path = Path(file_path).expanduser()
    if source_path.suffix:
        output_base = source_path.with_suffix("")
    else:
        output_base = source_path
    return output_base.with_suffix(".transcription.txt")


def _import_imageio_ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg


def _ffmpeg_binary() -> str:
    try:
        return str(get_runtime_ffmpeg_binary())
    except Exception:
        override = os.environ.get("FFMPEG_BINARY")
        if override:
            return override
        try:
            imageio_ffmpeg = _import_imageio_ffmpeg()
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"


def _ensure_ffmpeg_on_path() -> str:
    ffmpeg_executable = _ffmpeg_binary()
    ffmpeg_path = Path(ffmpeg_executable)
    if ffmpeg_path.is_file():
        ffmpeg_dir = str(ffmpeg_path.parent)
        current_path = os.environ.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        if ffmpeg_dir not in path_parts:
            os.environ["PATH"] = ffmpeg_dir + (
                os.pathsep + current_path if current_path else ""
            )
    return ffmpeg_executable


def _build_missing_ffmpeg_error(exc: FileNotFoundError) -> RuntimeError:
    attempted = _ffmpeg_binary()
    message = (
        "Could not find FFmpeg. LocalScribe needs FFmpeg to measure audio duration,"
        " export chunks, and let Whisper decode chunk audio during transcription. "
        f"It tried to use '{attempted}'. Install imageio-ffmpeg correctly,"
        " bundle FFmpeg with the app, "
        "or set the FFMPEG_BINARY environment variable to a valid ffmpeg executable. "
        f"Original error: {exc}"
    )
    return RuntimeError(message)


def _parse_ffmpeg_duration_seconds(stderr_text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _probe_duration_with_wave(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (wave.Error, EOFError, OSError):
        return None
    if frame_rate <= 0:
        return None
    return frame_count / float(frame_rate)


def _probe_duration_seconds(file_path: str | os.PathLike[str]) -> float:
    path = _ensure_existing_file(file_path)

    wave_duration = _probe_duration_with_wave(path)
    if wave_duration is not None:
        if wave_duration <= 0:
            raise ValueError(f"Invalid audio duration for {path}: {wave_duration}")
        return float(wave_duration)

    command = [_ensure_ffmpeg_on_path(), "-hide_banner", "-i", str(path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise _build_missing_ffmpeg_error(exc) from exc

    stderr_text = completed.stderr or completed.stdout or ""
    duration = _parse_ffmpeg_duration_seconds(stderr_text)
    if duration is None or duration <= 0:
        raise RuntimeError(
            f"Could not determine audio duration for '{path}'. "
            f"FFmpeg output was:\n{stderr_text}"
        )
    return float(duration)


def _iter_chunk_specs(
    total_duration_seconds: float,
    chunk_seconds: int,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
) -> Iterable[tuple[int, float, float, float, float]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be a positive integer")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be zero or a positive integer")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    stride_seconds = chunk_seconds - overlap_seconds
    if total_duration_seconds <= 0:
        raise ValueError("total_duration_seconds must be positive")

    raw_specs: list[tuple[int, float, float]] = []
    chunk_index = 0
    start = 0.0
    while True:
        end = min(float(total_duration_seconds), start + float(chunk_seconds))
        raw_specs.append((chunk_index, start, end))
        if end >= total_duration_seconds:
            break
        chunk_index += 1
        start += float(stride_seconds)

    half_overlap = float(overlap_seconds) / 2.0
    last_index = len(raw_specs) - 1
    for chunk_index, start, end in raw_specs:
        keep_start = start if chunk_index == 0 else min(end, start + half_overlap)
        keep_end = end if chunk_index == last_index else max(start, end - half_overlap)
        yield chunk_index, start, end, keep_start, keep_end


def _export_audio_chunk(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    start_seconds: float,
    duration_seconds: float,
) -> None:
    command = [
        _ensure_ffmpeg_on_path(),
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise _build_missing_ffmpeg_error(exc) from exc


def _model_filename(model_name: str) -> str:
    return f"{model_name}.pt"


def _is_download_error(exc: Exception) -> bool:
    network_types = (URLError, TimeoutError, socket.timeout, ConnectionError)
    if isinstance(exc, network_types):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in [
            "urlopen error",
            "timed out",
            "connection attempt failed",
            "certificate",
            "ssl",
        ]
    )


def _build_model_download_error(
    model_name: str, model_dir: Path, exc: Exception
) -> RuntimeError:
    model_file = model_dir / _model_filename(model_name)
    message = (
        f"Could not load Whisper model '{model_name}'. "
        f"LocalScribe looked in '{model_file}' and then tried to "
        f"download it, but the download failed. This usually means "
        f"the first model download was blocked by the "
        f"network, proxy, firewall, or an offline machine. "
        f"Connect to the internet once and retry, or place the "
        f"model file at '{model_file}'. Original error: {exc}"
    )
    return RuntimeError(message)


def download_model(
    model_name: str = DEFAULT_MODEL_NAME,
    progress_callback: ProgressCallback | None = None,
):
    _ensure_ffmpeg_on_path()
    model_dir = get_runtime_model_dir()
    model_file = model_dir / _model_filename(model_name)
    if model_file.exists():
        _emit_progress(
            progress_callback,
            0.0,
            f"Loading Whisper model '{model_name}' from {model_file}",
        )
    else:
        _emit_progress(
            progress_callback,
            0.0,
            f"Loading Whisper model '{model_name}' into {model_dir}",
        )
    ssl._create_default_https_context = ssl._create_unverified_context
    whisper = _import_whisper()
    try:
        model = whisper.load_model(model_name, download_root=str(model_dir))
    except Exception as exc:
        if _is_download_error(exc):
            raise _build_model_download_error(model_name, model_dir, exc) from exc
        raise
    _emit_progress(progress_callback, 1.0, f"Model '{model_name}' ready")
    return model


def get_model(
    model_name: str = DEFAULT_MODEL_NAME,
    progress_callback: ProgressCallback | None = None,
):
    return download_model(model_name=model_name, progress_callback=progress_callback)


def _normalized_compare_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in text.split():
        normalized = re.sub(r"(^[^\w']+|[^\w']+$)", "", raw_token).casefold()
        if normalized:
            tokens.append(normalized)
    return tokens


def _remove_repeated_prefix(
    previous_text: str, next_text: str, max_words: int = 12
) -> str:
    previous_words = previous_text.split()
    next_words = next_text.split()
    if not previous_words or not next_words:
        return next_text.strip()

    previous_normalized = _normalized_compare_tokens(previous_text)
    next_normalized = _normalized_compare_tokens(next_text)
    max_overlap = min(max_words, len(previous_normalized), len(next_normalized))
    overlap_words = 0
    for candidate in range(max_overlap, 0, -1):
        if previous_normalized[-candidate:] == next_normalized[:candidate]:
            overlap_words = candidate
            break

    if overlap_words <= 0:
        return next_text.strip()
    if overlap_words >= len(next_words):
        return ""
    return " ".join(next_words[overlap_words:]).strip()


def _normalize_text_parts(text_parts: list[str]) -> str:
    merged_parts: list[str] = []
    for raw_part in text_parts:
        part = raw_part.strip()
        if not part:
            continue
        if not merged_parts:
            merged_parts.append(part)
            continue
        deduplicated = _remove_repeated_prefix(merged_parts[-1], part)
        if deduplicated:
            merged_parts.append(deduplicated)

    joined = " ".join(merged_parts).strip()
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    return joined + ("\n" if joined else "")


def _segment_midpoint_seconds(segment: dict) -> float:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    if end < start:
        end = start
    return start + (end - start) / 2.0


def _segment_in_keep_window(
    segment: dict,
    keep_start: float,
    keep_end: float,
    *,
    is_last_chunk: bool,
) -> bool:
    midpoint = _segment_midpoint_seconds(segment)
    if midpoint < keep_start:
        return False
    if is_last_chunk:
        return midpoint <= keep_end
    return midpoint < keep_end


def _deduplicate_selected_segments(segments: list[dict]) -> list[dict]:
    deduplicated: list[dict] = []
    for segment in sorted(
        segments,
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
        ),
    ):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        if deduplicated:
            previous = deduplicated[-1]
            same_text = _normalized_compare_tokens(text) == _normalized_compare_tokens(
                str(previous.get("text", ""))
            )
            current_start = float(segment.get("start", 0.0))
            previous_end = float(previous.get("end", 0.0))
            if same_text and current_start <= previous_end + 0.25:
                if float(segment.get("end", 0.0)) > previous_end:
                    deduplicated[-1] = segment
                continue
        deduplicated.append(segment)
    return deduplicated


def transcribe_audio(
    file_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    progress_callback: ProgressCallback | None = None,
    language: str | None = None,
    task: str = "transcribe",
):
    source_path = _ensure_existing_file(file_path)
    if output_path is None:
        output_text_path = default_output_text_path(source_path)
        output_base = output_text_path.with_suffix("")
    else:
        output_text_path = Path(output_path).expanduser().resolve()
        output_base = output_text_path.with_suffix("")

    output_json_path = output_base.with_suffix(".transcription.json")
    output_segments_path = output_base.with_suffix(".transcription.segments.txt")

    _emit_progress(
        progress_callback, 0.0, f"Preparing transcription for {source_path.name}"
    )
    total_duration = _probe_duration_seconds(source_path)
    _emit_progress(
        progress_callback, 0.05, f"Audio duration: {total_duration:.1f} seconds"
    )

    model = download_model(model_name=model_name, progress_callback=progress_callback)

    fallback_text_parts: list[str] = []
    all_segments: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="localscribe_") as temp_dir:
        chunk_specs = list(
            _iter_chunk_specs(
                total_duration_seconds=total_duration,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
        )
        total_chunks = len(chunk_specs)
        for chunk_index, start, end, keep_start, keep_end in chunk_specs:
            chunk_duration = end - start
            chunk_path = Path(temp_dir) / f"chunk_{chunk_index:05d}.wav"
            _emit_progress(
                progress_callback,
                0.10 + 0.80 * (chunk_index / max(1, total_chunks)),
                f"Exporting chunk {chunk_index + 1}/{total_chunks}",
            )
            _export_audio_chunk(source_path, chunk_path, start, chunk_duration)

            _emit_progress(
                progress_callback,
                0.10 + 0.80 * (chunk_index / max(1, total_chunks)),
                f"Transcribing chunk {chunk_index + 1}/{total_chunks}",
            )
            try:
                result = model.transcribe(str(chunk_path), language=language, task=task)
            except FileNotFoundError as exc:
                raise _build_missing_ffmpeg_error(exc) from exc

            chunk_text = str(result.get("text", "") or "")
            raw_segments = result.get("segments", []) or []
            selected_segments_for_chunk: list[dict] = []
            for segment in raw_segments:
                adjusted = dict(segment)
                adjusted["start"] = float(adjusted.get("start", 0.0)) + start
                adjusted["end"] = float(adjusted.get("end", 0.0)) + start
                if _segment_in_keep_window(
                    adjusted,
                    keep_start,
                    keep_end,
                    is_last_chunk=(chunk_index == total_chunks - 1),
                ):
                    selected_segments_for_chunk.append(adjusted)

            if selected_segments_for_chunk:
                all_segments.extend(selected_segments_for_chunk)
            elif chunk_text.strip():
                fallback_text_parts.append(chunk_text)

            _emit_progress(
                progress_callback,
                0.10 + 0.80 * ((chunk_index + 1) / max(1, total_chunks)),
                f"Finished chunk {chunk_index + 1}/{total_chunks}",
            )

    all_segments = _deduplicate_selected_segments(all_segments)
    if all_segments:
        full_text = _normalize_text_parts(
            [str(segment.get("text", "")) for segment in all_segments]
        )
    else:
        full_text = _normalize_text_parts(fallback_text_parts)

    structured_result = {
        "source": str(source_path),
        "model_name": model_name,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "task": task,
        "language": language,
        "text": full_text,
        "segments": all_segments,
    }

    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path.write_text(full_text, encoding="utf-8")
    output_json_path.write_text(
        json.dumps(structured_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with output_segments_path.open("w", encoding="utf-8") as handle:
        for segment in all_segments:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            text = str(segment.get("text", "")).strip()
            handle.write(f"[{start:8.2f} -> {end:8.2f}] {text}\n")

    _emit_progress(
        progress_callback, 1.0, f"Done. Saved transcript to {output_text_path}"
    )
    return {
        "text": full_text,
        "segments": all_segments,
        "output_text_path": str(output_text_path),
        "output_json_path": str(output_json_path),
        "output_segments_path": str(output_segments_path),
    }


def parse_args(args: list[str]):
    parser = argparse.ArgumentParser(
        description="Transcribe audio locally with Whisper"
    )
    parser.add_argument(
        "file_path", nargs="?", help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional output text path."
        + " JSON and segment files will use the same base name.",
    )
    parser.add_argument(
        "--model-name", default=DEFAULT_MODEL_NAME, help="Whisper model name to load"
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Chunk duration in seconds for chunked transcription",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=int,
        default=DEFAULT_OVERLAP_SECONDS,
        help="Overlap in seconds between adjacent chunks",
    )
    parser.add_argument(
        "--language", default=None, help="Optional language hint passed to Whisper"
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task to run",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download or preload the specified Whisper model and exit",
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
    namespace = parser.parse_args(args)
    if not namespace.download_model and not namespace.file_path:
        parser.error("the following arguments are required: file_path")
    return namespace


def setup_logging(loglevel: int | None):
    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(args: list[str]):
    namespace = parse_args(args)
    setup_logging(namespace.loglevel)
    progress_callback = _stdout_progress()
    if namespace.download_model:
        download_model(
            model_name=namespace.model_name, progress_callback=progress_callback
        )
        return 0

    result = transcribe_audio(
        file_path=namespace.file_path,
        output_path=namespace.output_path,
        model_name=namespace.model_name,
        chunk_seconds=namespace.chunk_seconds,
        overlap_seconds=namespace.overlap_seconds,
        progress_callback=progress_callback,
        language=namespace.language,
        task=namespace.task,
    )
    _logger.info("Transcription saved to %s", result["output_text_path"])
    return 0


def run():
    return main(sys.argv[1:])


if __name__ == "__main__":
    run()
