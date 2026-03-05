from unittest.mock import patch
from localscribe.skeleton import main

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    with patch('localscribe.skeleton.transcribe_audio') as mock_transcribe:
        main(["7"])
        captured = capsys.readouterr()
        assert "crazy" in captured.out
        mock_transcribe.assert_called_once_with("7")
