import pytest

from localscribe.skeleton import main

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"


@pytest.fixture
def sample_audio(tmp_path):
    """Create a minimal valid WAV file for testing."""
    import wave

    audio_file = tmp_path / "test_audio.wav"

    with wave.open(str(audio_file), "wb") as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)  # 1 second of silence

    return audio_file


def test_main(sample_audio, capsys):
    """Test main with valid audio file."""
    main([str(sample_audio)])
    captured = capsys.readouterr()
    # Check for expected output
    assert "Saved:" in captured.out or "transcription" in captured.out
