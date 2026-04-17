from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path


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


def get_pyinstaller_datas() -> list[tuple[str, str]]:
    datas = [(str(get_whisper_assets_dir()), "whisper/assets")]
    cache_dir = get_whisper_cache_dir()
    if cache_dir.is_dir():
        datas.append((str(cache_dir), "whisper_models"))
    return datas
