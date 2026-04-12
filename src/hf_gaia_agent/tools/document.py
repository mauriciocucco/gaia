"""Document reading tools — local file content extraction."""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ._http import AUDIO_TIMEOUT, make_client, truncate
from ._parsing import (
    html_to_text,
    read_csv,
    read_json,
    read_pdf,
    read_xlsx,
)
from ._payloads import StructuredToolResult, TextDocumentPayload

logger = logging.getLogger(__name__)

_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav", ".webm"}


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------


def _audio_api_config() -> tuple[str, str, str]:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
        model = os.getenv("AUDIO_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe").strip()
    elif provider == "huggingface":
        api_key = os.getenv("HF_TOKEN", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1").strip()
        model = os.getenv("AUDIO_TRANSCRIPTION_MODEL", "whisper-1").strip()
    else:
        raise RuntimeError(
            f"Audio transcription is not configured for MODEL_PROVIDER '{provider}'."
        )

    if not api_key:
        raise RuntimeError("Missing API key for audio transcription.")
    if not base_url:
        raise RuntimeError("Missing base URL for audio transcription.")
    if not model:
        raise RuntimeError("Missing audio transcription model.")
    return api_key, base_url.rstrip("/"), model


def _transcribe_audio(path: Path) -> str:
    api_key, base_url, model = _audio_api_config()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data: dict[str, Any] = {
        "model": model,
        "response_format": "json",
    }
    language = os.getenv("AUDIO_TRANSCRIPTION_LANGUAGE", "").strip()
    prompt = os.getenv("AUDIO_TRANSCRIPTION_PROMPT", "").strip()
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    with path.open("rb") as handle:
        files = {"file": (path.name, handle, content_type)}
        with make_client(
            timeout=AUDIO_TIMEOUT,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as client:
            response = client.post(f"{base_url}/audio/transcriptions", data=data, files=files)
            response.raise_for_status()

    payload = response.json()
    transcript = str(payload.get("text") or "").strip()
    if not transcript:
        raise RuntimeError(f"Audio transcription returned no text for {path.name}.")
    return truncate(transcript, max_chars=80000)


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def read_file_content(path: str) -> str:
    """Read a local task attachment and return plain text content."""
    return read_file_content_result(path).text


def read_file_content_result(path: str) -> StructuredToolResult:
    """Structured variant of ``read_file_content`` for internal workflow use."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")

    suffix = candidate.suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".csv":
        content = read_csv(candidate)
    elif suffix == ".json":
        content = read_json(candidate)
    elif suffix in {".html", ".htm"}:
        content = html_to_text(candidate.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".pdf":
        content = read_pdf(candidate)
    elif suffix == ".xlsx":
        content = read_xlsx(candidate)
    elif suffix in _AUDIO_SUFFIXES:
        content = _transcribe_audio(candidate)
    else:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    payload = TextDocumentPayload(
        kind="file_text",
        content=truncate(content),
        title=candidate.name,
        metadata={"path": str(candidate)},
    )
    return StructuredToolResult(text=payload.content, payloads=(payload,))


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


@tool
def read_local_file(path: str) -> str:
    """Read a local text, CSV, JSON, HTML, PDF, XLSX, or supported audio file."""
    return read_file_content_result(path).text
