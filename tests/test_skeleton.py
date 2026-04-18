from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import localscribe.packaging as packaging
import localscribe.skeleton as skeleton
from localscribe.skeleton import main

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"


@pytest.fixture
def sample_audio(tmp_path):
    import wave

    audio_file = tmp_path / "test_audio.wav"
    with wave.open(str(audio_file), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    return audio_file


def test_parse_args_requires_file_path_when_not_downloading():
    with pytest.raises(SystemExit):
        skeleton.parse_args([])


def test_main(sample_audio, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        skeleton,
        "transcribe_audio",
        lambda **kwargs: {"output_text_path": str(tmp_path / "out.txt")},
    )
    rc = main([str(sample_audio)])
    capsys.readouterr()
    assert rc == 0


def test_ensure_ffmpeg_on_path_uses_runtime_binary(monkeypatch, tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg"
    ffmpeg_dir.mkdir()
    ffmpeg_binary = ffmpeg_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    ffmpeg_binary.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(skeleton, "_ffmpeg_binary", lambda: str(ffmpeg_binary))
    monkeypatch.setenv("PATH", "")

    resolved = skeleton._ensure_ffmpeg_on_path()

    assert resolved == str(ffmpeg_binary)
    path_parts = os.environ["PATH"].split(os.pathsep)
    assert str(ffmpeg_dir) == path_parts[0]


def test_build_missing_ffmpeg_error_mentions_attempted_binary(monkeypatch):
    monkeypatch.setattr(skeleton, "_ffmpeg_binary", lambda: "C:/bundle/ffmpeg.exe")
    error = skeleton._build_missing_ffmpeg_error(FileNotFoundError("missing"))
    text = str(error)
    assert "C:/bundle/ffmpeg.exe" in text
    assert "Whisper decode" in text or "Whisper decode chunk audio" in text


def test_get_packaged_ffmpeg_binary_prefers_meipass(monkeypatch, tmp_path):
    source_ffmpeg = tmp_path / "source" / "ffmpeg.exe"
    source_ffmpeg.parent.mkdir(parents=True)
    source_ffmpeg.write_text("stub", encoding="utf-8")
    bundled_ffmpeg = tmp_path / "bundle" / "ffmpeg" / "ffmpeg.exe"
    bundled_ffmpeg.parent.mkdir(parents=True)
    bundled_ffmpeg.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(packaging, "get_source_ffmpeg_binary", lambda: source_ffmpeg)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)

    try:
        assert packaging.get_packaged_ffmpeg_binary() == bundled_ffmpeg.resolve()
    finally:
        if hasattr(sys, "_MEIPASS"):
            monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_get_pyinstaller_binaries_uses_source_ffmpeg(monkeypatch, tmp_path):
    ffmpeg_binary = tmp_path / "ffmpeg.exe"
    ffmpeg_binary.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(packaging, "get_source_ffmpeg_binary", lambda: ffmpeg_binary)

    assert packaging.get_pyinstaller_binaries() == [(str(ffmpeg_binary), "ffmpeg")]


def test_download_model_adds_ffmpeg_to_path(monkeypatch, tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg"
    ffmpeg_dir.mkdir()
    ffmpeg_binary = ffmpeg_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    ffmpeg_binary.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(skeleton, "_ffmpeg_binary", lambda: str(ffmpeg_binary))
    monkeypatch.setattr(skeleton, "get_runtime_model_dir", lambda: tmp_path / "models")

    load_calls = []

    def fake_load_model(model_name, download_root=None):
        load_calls.append((model_name, download_root))
        return SimpleNamespace()

    monkeypatch.setattr(skeleton, "_import_whisper", lambda: SimpleNamespace(load_model=fake_load_model))
    monkeypatch.setenv("PATH", "")

    skeleton.download_model(model_name="base")

    assert load_calls == [("base", str((tmp_path / "models")))]
    assert str(ffmpeg_dir) in os.environ["PATH"].split(os.pathsep)


def test_transcribe_wraps_whisper_missing_ffmpeg(sample_audio, monkeypatch, tmp_path):
    monkeypatch.setattr(skeleton, "_probe_duration_seconds", lambda _path: 1.0)
    monkeypatch.setattr(skeleton, "_export_audio_chunk", lambda *args, **kwargs: None)
    monkeypatch.setattr(skeleton, "download_model", lambda **kwargs: SimpleNamespace(transcribe=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ffmpeg missing"))))

    with pytest.raises(RuntimeError, match="Could not find FFmpeg"):
        skeleton.transcribe_audio(str(sample_audio), output_path=tmp_path / "out.txt")
