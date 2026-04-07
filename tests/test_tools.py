"""Tests for tools module, primarily the video frame analysis pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil

import pytest

from hf_gaia_agent.tools import (
    _check_binary,
    _download_video,
    _extract_frames,
    _encode_frame_base64,
    extract_youtube_video_id,
)


def test_extract_youtube_video_id_standard() -> None:
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=L1vXCYZAYYM") == "L1vXCYZAYYM"


def test_extract_youtube_video_id_short() -> None:
    assert extract_youtube_video_id("https://youtu.be/L1vXCYZAYYM") == "L1vXCYZAYYM"


def test_extract_youtube_video_id_embed() -> None:
    assert extract_youtube_video_id("https://www.youtube.com/embed/L1vXCYZAYYM") == "L1vXCYZAYYM"


def test_extract_youtube_video_id_shorts() -> None:
    assert extract_youtube_video_id("https://www.youtube.com/shorts/L1vXCYZAYYM") == "L1vXCYZAYYM"


def test_extract_youtube_video_id_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported YouTube URL"):
        extract_youtube_video_id("https://example.com/not-youtube")


def test_check_binary_missing() -> None:
    with pytest.raises(RuntimeError, match="not found on PATH"):
        _check_binary("nonexistent_binary_xyz_123")


def test_check_binary_found() -> None:
    path = _check_binary("python")
    assert path is not None


def test_encode_frame_base64(tmp_path: Path) -> None:
    frame = tmp_path / "test.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0JFIF")
    encoded = _encode_frame_base64(frame)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_download_video_calls_yt_dlp(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/yt-dlp" if name == "yt-dlp" else None)

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"fake-video-content")

    with patch("hf_gaia_agent.tools.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = _download_video("https://www.youtube.com/watch?v=test123", tmp_path)

    assert result == fake_video
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/yt-dlp"
    assert "--no-playlist" in cmd


def test_extract_frames_calls_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff")
    (frames_dir / "frame_0002.jpg").write_bytes(b"\xff\xd8\xff")

    with patch("hf_gaia_agent.tools.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = _extract_frames(video_file, frames_dir)

    assert len(result) == 2
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/ffmpeg"


def test_analyze_youtube_video_tool_integration(tmp_path: Path, monkeypatch) -> None:
    """End-to-end test with all external calls mocked."""
    from hf_gaia_agent.tools import analyze_youtube_video

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    fake_frame = tmp_path / "frame_0001.jpg"
    fake_frame.write_bytes(b"\xff\xd8\xff\xe0sample")

    def fake_download(url, output_dir):
        video = output_dir / "video.mp4"
        video.write_bytes(b"fake")
        return video

    def fake_extract(video_path, output_dir):
        f = output_dir / "frame_0001.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0sample")
        return [f]

    monkeypatch.setattr("hf_gaia_agent.tools._download_video", fake_download)
    monkeypatch.setattr("hf_gaia_agent.tools._extract_frames", fake_extract)

    fake_response = MagicMock()
    fake_response.content = "There are 3 bird species visible simultaneously."

    fake_vision_model = MagicMock()
    fake_vision_model.invoke.return_value = fake_response

    with patch("langchain_openai.ChatOpenAI", return_value=fake_vision_model):
        monkeypatch.setenv("MODEL_PROVIDER", "openai")
        monkeypatch.setenv("MODEL_NAME", "gpt-4.1-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        result = analyze_youtube_video.invoke({
            "url": "https://www.youtube.com/watch?v=L1vXCYZAYYM",
            "question": "How many bird species are on camera simultaneously?",
        })

    assert "3 bird species" in result
