from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


ENV_MODEL_DIR = "LOCALSCRIBE_MODEL_DIR"
ENV_FFMPEG_BINARY = "FFMPEG_BINARY"


def get_whisper_assets_dir() -> Path:
    whisper_module = import_module("whisper")
    assets_dir = Path(whisper_module.__file__).resolve().parent / "assets"
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Whisper assets directory not found: {assets_dir}")
    return assets_dir


def get_whisper_cache_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "whisper"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser().resolve() / "whisper"
    return (Path.home() / ".cache" / "whisper").resolve()


def get_packaged_model_dir() -> Path | None:
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "whisper_models")
    executable = getattr(sys, "executable", None)
    if executable:
        candidates.append(Path(executable).resolve().parent / "whisper_models")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def get_runtime_model_dir() -> Path:
    override = os.environ.get(ENV_MODEL_DIR)
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    packaged = get_packaged_model_dir()
    if packaged is not None:
        return packaged

    cache_dir = get_whisper_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_source_ffmpeg_binary() -> Path:
    imageio_ffmpeg = import_module("imageio_ffmpeg")
    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not ffmpeg_path.is_file():
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg_path}")
    return ffmpeg_path


def get_packaged_ffmpeg_binary() -> Path | None:
    candidates: list[Path] = []
    binary_name = get_source_ffmpeg_binary().name
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "ffmpeg" / binary_name)
        candidates.append(Path(meipass) / binary_name)
    executable = getattr(sys, "executable", None)
    if executable:
        root = Path(executable).resolve().parent
        candidates.append(root / "ffmpeg" / binary_name)
        candidates.append(root / binary_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def get_runtime_ffmpeg_binary() -> Path:
    override = os.environ.get(ENV_FFMPEG_BINARY)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FFmpeg executable not found: {path}")
        return path

    packaged = get_packaged_ffmpeg_binary()
    if packaged is not None:
        return packaged

    return get_source_ffmpeg_binary()


def get_pyinstaller_datas() -> list[tuple[str, str]]:
    datas = [(str(get_whisper_assets_dir()), "whisper/assets")]
    cache_dir = get_whisper_cache_dir()
    if cache_dir.is_dir():
        datas.append((str(cache_dir), "whisper_models"))
    return datas


def get_pyinstaller_binaries() -> list[tuple[str, str]]:
    return [(str(get_source_ffmpeg_binary()), "ffmpeg")]
