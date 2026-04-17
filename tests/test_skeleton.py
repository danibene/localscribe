from localscribe.skeleton import main

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"


def test_main(capsys, tmp_path):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    # Create a temporary test audio file
    test_audio = tmp_path / "test_audio.wav"
    test_audio.write_bytes(b"fake audio data")
    
    main([str(test_audio)])
    captured = capsys.readouterr()
    assert "transcription" in captured.out
