from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_default_output_text_path(sample_audio):
    assert (
        skeleton.default_output_text_path(sample_audio).name
        == "test_audio.transcription.txt"
    )


def test_parse_args_requires_file_path_when_not_downloading():
    with pytest.raises(SystemExit):
        skeleton.parse_args([])


def test_parse_args_accepts_overlap_seconds(sample_audio):
    args = skeleton.parse_args([str(sample_audio), "--overlap-seconds", "7"])
    assert args.overlap_seconds == 7


def test_main(sample_audio, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        skeleton,
        "transcribe_audio",
        lambda **kwargs: {"output_text_path": str(tmp_path / "out.txt")},
    )
    rc = main([str(sample_audio)])
    capsys.readouterr()
    assert rc == 0


def test_iter_chunk_specs_with_overlap():
    specs = list(skeleton._iter_chunk_specs(25.0, chunk_seconds=10, overlap_seconds=2))
    assert specs == [
        (0, 0.0, 10.0, 0.0, 9.0),
        (1, 8.0, 18.0, 9.0, 17.0),
        (2, 16.0, 25.0, 17.0, 25.0),
    ]


def test_normalize_text_parts_removes_boundary_repetition():
    merged = skeleton._normalize_text_parts(
        ["hello there common words", "common words continue here"]
    )
    assert merged == "hello there common words continue here\n"


def test_slice_audio_chunk_returns_expected_samples():
    audio = np.arange(0, 16000 * 3, dtype=np.float32)
    sliced = skeleton._slice_audio_chunk(
        audio,
        sample_rate=16000,
        start_seconds=0.5,
        end_seconds=1.5,
    )
    assert sliced.shape == (16000,)
    assert sliced[0] == pytest.approx(8000.0)
    assert sliced[-1] == pytest.approx(23999.0)


def test_decode_audio_to_mono_uses_av_resampler(sample_audio, monkeypatch):
    class FakeResampledFrame:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def to_ndarray(self):
            return self._values

    class FakeResampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def resample(self, frame):
            if frame is None:
                return []
            return [FakeResampledFrame([[0.1, 0.2, 0.3]])]

    class FakeContainer:
        def __init__(self):
            self.streams = [
                SimpleNamespace(type="audio", duration=16000, time_base=1 / 16000)
            ]

        def decode(self, stream):
            yield object()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_av = SimpleNamespace(
        open=lambda path: FakeContainer(),
        audio=SimpleNamespace(
            resampler=SimpleNamespace(
                AudioResampler=lambda **kwargs: FakeResampler(**kwargs)
            )
        ),
    )
    monkeypatch.setattr(skeleton, "_import_av", lambda: fake_av)

    audio, rate = skeleton._decode_audio_to_mono(sample_audio)

    assert rate == 16000
    assert np.allclose(audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))


@pytest.mark.parametrize(
    ("raw_text", "expected_tokens"),
    [
        ("Hello, world!", ["hello", "world"]),
        ("Don't stop", ["don't", "stop"]),
    ],
)
def test_normalized_compare_tokens(raw_text, expected_tokens):
    assert skeleton._normalized_compare_tokens(raw_text) == expected_tokens


def test_download_model_uses_runtime_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    monkeypatch.setattr(skeleton, "get_runtime_model_dir", lambda: model_dir)

    load_calls = []

    def fake_load_model(model_name, download_root=None):
        load_calls.append((model_name, download_root))
        return SimpleNamespace()

    monkeypatch.setattr(
        skeleton, "_import_whisper", lambda: SimpleNamespace(load_model=fake_load_model)
    )

    skeleton.download_model(model_name="base")

    assert load_calls == [("base", str(model_dir))]


def test_transcribe_audio_uses_overlap_and_postprocesses_text(
    sample_audio, monkeypatch, tmp_path
):
    audio = np.zeros(12 * skeleton.WHISPER_SAMPLE_RATE, dtype=np.float32)
    monkeypatch.setattr(
        skeleton,
        "_decode_audio_to_mono",
        lambda *args, **kwargs: (audio, skeleton.WHISPER_SAMPLE_RATE),
    )

    transcribe_calls = []
    results = iter(
        [
            {
                "text": "hello common words",
                "segments": [
                    {"start": 0.0, "end": 4.0, "text": "hello"},
                    {"start": 8.0, "end": 9.4, "text": "common words"},
                ],
            },
            {
                "text": "common words continue here",
                "segments": [
                    {"start": 0.7, "end": 1.6, "text": "common words"},
                    {"start": 2.0, "end": 4.0, "text": "continue here"},
                ],
            },
        ]
    )

    def fake_transcribe(chunk_audio, **kwargs):
        transcribe_calls.append(chunk_audio.shape[0])
        return next(results)

    model = SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setattr(skeleton, "download_model", lambda **kwargs: model)

    result = skeleton.transcribe_audio(
        str(sample_audio),
        output_path=tmp_path / "custom" / "out.txt",
        chunk_seconds=10,
        overlap_seconds=2,
    )

    assert transcribe_calls == [
        10 * skeleton.WHISPER_SAMPLE_RATE,
        4 * skeleton.WHISPER_SAMPLE_RATE,
    ]
    assert result["text"] == "hello common words continue here\n"
    assert Path(result["output_text_path"]).name == "out.txt"
    assert (tmp_path / "custom" / "out.txt").exists()
    assert (tmp_path / "custom" / "out.transcription.json").exists()
    assert (tmp_path / "custom" / "out.transcription.segments.txt").exists()

    with open(result["output_segments_path"], "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    assert len(lines) == 3


def test_packaged_model_dir_prefers_meipass(monkeypatch, tmp_path):
    bundled_models = tmp_path / "bundle" / "whisper_models"
    bundled_models.mkdir(parents=True)
    monkeypatch.setattr(
        packaging.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False
    )
    try:
        assert packaging.get_packaged_model_dir() == bundled_models.resolve()
    finally:
        monkeypatch.delattr(packaging.sys, "_MEIPASS", raising=False)
