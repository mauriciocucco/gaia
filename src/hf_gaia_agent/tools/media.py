"""Media tools: YouTube transcript, video frame analysis, audio extraction."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ._http import truncate
from ._payloads import StructuredToolResult, TextDocumentPayload
from ._runtime import runtime_workspace
from ._video_analysis import (
    build_video_analysis_prompt,
    build_video_message_content,
    extract_json_object as _extract_json_object,
    extract_max_count_from_payload as _extract_max_count_from_payload,
    is_counting_visual_question as _is_counting_visual_question,
    prepend_audio_transcript,
    select_dense_timestamps_from_payload,
)
from ._web_helpers import extract_youtube_video_id
from .document import _transcribe_audio

logger = logging.getLogger(__name__)

_FRAME_INTERVAL_SECONDS = 5
_MAX_FRAMES = 20
_DENSE_FRAME_INTERVAL_SECONDS = 1


def _check_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' not found on PATH. Install it first.")
    return path


def _download_video(url: str, output_dir: Path) -> Path:
    yt_dlp = _check_binary("yt-dlp")
    output_path = output_dir / "video.%(ext)s"
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--format",
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--output",
        str(output_path),
        "--quiet",
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True, timeout=120)
    videos = list(output_dir.glob("video.*"))
    if not videos:
        raise FileNotFoundError("yt-dlp did not produce an output file.")
    return videos[0]


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    interval_seconds: int = _FRAME_INTERVAL_SECONDS,
    max_frames: int = _MAX_FRAMES,
    prefix: str = "frame",
) -> list[Path]:
    ffmpeg = _check_binary("ffmpeg")
    pattern = output_dir / f"{prefix}_%04d.jpg"
    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds},scale=768:-1",
        "-q:v",
        "2",
        "-frames:v",
        str(max_frames),
        str(pattern),
        "-y",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True, timeout=120)
    return sorted(output_dir.glob(f"{prefix}_*.jpg"))


def _extract_frame_at_timestamp(
    video_path: Path,
    output_dir: Path,
    *,
    timestamp_seconds: int,
    prefix: str = "dense_frame",
) -> Path:
    ffmpeg = _check_binary("ffmpeg")
    frame_path = output_dir / f"{prefix}_{timestamp_seconds:04d}.jpg"
    cmd = [
        ffmpeg,
        "-ss",
        str(max(0, timestamp_seconds)),
        "-i",
        str(video_path),
        "-vf",
        "scale=768:-1",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(frame_path),
        "-y",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True, timeout=120)
    if not frame_path.exists():
        raise FileNotFoundError(
            f"ffmpeg did not produce a dense frame for timestamp {timestamp_seconds}s."
        )
    return frame_path


def _extract_dense_frames(
    video_path: Path,
    output_dir: Path,
    timestamps: list[int],
) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for timestamp in sorted(dict.fromkeys(max(0, int(ts)) for ts in timestamps)):
        frames.append(
            (
                timestamp,
                _extract_frame_at_timestamp(
                    video_path,
                    output_dir,
                    timestamp_seconds=timestamp,
                ),
            )
        )
    return frames


def _extract_audio(video_path: Path, output_dir: Path) -> Path:
    ffmpeg = _check_binary("ffmpeg")
    audio_path = output_dir / "audio.mp3"
    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-q:a",
        "4",
        str(audio_path),
        "-y",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True, timeout=120)
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FileNotFoundError("ffmpeg did not produce an audio file.")
    return audio_path


def _encode_frame_base64(frame_path: Path) -> str:
    from ._video_analysis import encode_frame_base64

    return encode_frame_base64(frame_path)


@tool
def get_youtube_transcript(url: str, languages_csv: str = "en,en-US") -> str:
    """Fetch a YouTube transcript for a video URL using prioritized languages."""
    return get_youtube_transcript_result(
        url=url,
        languages_csv=languages_csv,
    ).text


def get_youtube_transcript_result(
    url: str, languages_csv: str = "en,en-US"
) -> StructuredToolResult:
    """Structured variant of ``get_youtube_transcript`` for internal workflow use."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import CouldNotRetrieveTranscript
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api is not installed. Reinstall project dependencies."
        ) from exc

    video_id = extract_youtube_video_id(url)
    languages = [item.strip() for item in languages_csv.split(",") if item.strip()]
    client = YouTubeTranscriptApi()
    try:
        transcript = client.fetch(video_id, languages=languages or ["en"])
    except CouldNotRetrieveTranscript as exc:
        return StructuredToolResult(text=f"Transcript unavailable for {video_id}: {exc}")

    lines = [f"[{snippet.start:.2f}s] {snippet.text}" for snippet in transcript]
    payload = TextDocumentPayload(
        kind="transcript",
        url=url,
        title=video_id,
        content=truncate("\n".join(lines), max_chars=80000),
    )
    return StructuredToolResult(text=payload.content, payloads=(payload,))


@tool
def analyze_youtube_video(url: str, question: str) -> str:
    """Download a YouTube video, extract frames, and analyze them with a vision model to answer the question."""
    from langchain_core.messages import HumanMessage as HM
    from langchain_openai import ChatOpenAI
    from hf_gaia_agent import tools as tools_module

    video_id = extract_youtube_video_id(url)
    download_video = getattr(tools_module, "_download_video", _download_video)
    extract_frames = getattr(tools_module, "_extract_frames", _extract_frames)
    extract_dense_frames = getattr(
        tools_module,
        "_extract_dense_frames",
        _extract_dense_frames,
    )

    with runtime_workspace("gaia_video_") as tmp_path:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        try:
            video_file = download_video(url, tmp_path)
        except Exception as exc:
            return f"Failed to download video {video_id}: {exc}"

        try:
            frames = extract_frames(video_file, frames_dir)
        except Exception as exc:
            return f"Failed to extract frames from video {video_id}: {exc}"

        if not frames:
            return f"No frames extracted from video {video_id}."

        audio_transcript = ""
        try:
            audio_path = _extract_audio(video_file, tmp_path)
            audio_transcript = _transcribe_audio(audio_path)
        except Exception:
            pass

        counting_mode = _is_counting_visual_question(question)

        provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
        model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": 0,
            "timeout": 120,
        }
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
        elif provider == "huggingface":
            token = os.getenv("HF_TOKEN")
            base_url = os.getenv(
                "OPENAI_BASE_URL",
                "https://router.huggingface.co/v1",
            )
            if token:
                kwargs["api_key"] = token
            kwargs["base_url"] = base_url

        vision_model = ChatOpenAI(**kwargs)
        coarse_frame_items = [
            (index * _FRAME_INTERVAL_SECONDS, frame)
            for index, frame in enumerate(frames)
        ]
        coarse_prompt = prepend_audio_transcript(
            build_video_analysis_prompt(
                question=question,
                video_id=video_id,
                frame_count=len(coarse_frame_items),
                frame_interval_seconds=_FRAME_INTERVAL_SECONDS,
                counting_mode=counting_mode,
                prompt_mode="coarse",
            ),
            audio_transcript,
        )
        coarse_content = build_video_message_content(
            prompt_text=coarse_prompt,
            frame_items=coarse_frame_items,
            detail="high" if counting_mode else "low",
        )
        coarse_response = vision_model.invoke([HM(content=coarse_content)])
        coarse_response_text = str(coarse_response.content)

        if counting_mode:
            coarse_payload = _extract_json_object(coarse_response_text)
            coarse_max = (
                _extract_max_count_from_payload(coarse_payload)
                if coarse_payload is not None
                else None
            )
            if coarse_payload is not None:
                dense_timestamps = select_dense_timestamps_from_payload(coarse_payload)
                if dense_timestamps:
                    dense_dir = tmp_path / "dense_frames"
                    dense_dir.mkdir()
                    try:
                        dense_frames = extract_dense_frames(
                            video_file,
                            dense_dir,
                            dense_timestamps,
                        )
                    except Exception:
                        dense_frames = []
                    if dense_frames:
                        dense_prompt = prepend_audio_transcript(
                            build_video_analysis_prompt(
                                question=question,
                                video_id=video_id,
                                frame_count=len(dense_frames),
                                frame_interval_seconds=_DENSE_FRAME_INTERVAL_SECONDS,
                                counting_mode=True,
                                prompt_mode="verification",
                            ),
                            audio_transcript,
                        )
                        dense_content = build_video_message_content(
                            prompt_text=dense_prompt,
                            frame_items=dense_frames,
                            detail="high",
                        )
                        dense_response = vision_model.invoke([HM(content=dense_content)])
                        dense_payload = _extract_json_object(str(dense_response.content))
                        dense_max = (
                            _extract_max_count_from_payload(dense_payload)
                            if dense_payload is not None
                            else None
                        )
                        if dense_max is not None:
                            if coarse_max is not None:
                                return str(max(coarse_max, dense_max))
                            return str(dense_max)
            if coarse_max is not None:
                return str(coarse_max)
        return truncate(coarse_response_text, max_chars=8000)
