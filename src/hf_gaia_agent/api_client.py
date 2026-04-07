"""Client for the Hugging Face Agents Course GAIA scoring API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import httpx


@dataclass(slots=True)
class Question:
    task_id: str
    question: str
    Level: str | None = None
    file_name: str | None = None


@dataclass(slots=True)
class AnswerPayload:
    task_id: str
    submitted_answer: str


@dataclass(slots=True)
class ScoreResponse:
    username: str
    score: float
    correct_count: int
    total_attempted: int
    message: str
    timestamp: str


class ScoringAPIClient:
    """Thin sync client around the course API."""

    def __init__(
        self,
        base_url: str = "https://agents-course-unit4-scoring.hf.space",
        *,
        timeout: float = 60.0,
        download_dir: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.download_dir = Path(
            download_dir or os.getenv("GAIA_DOWNLOAD_DIR", ".cache/gaia")
        )
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ScoringAPIClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def health(self) -> bool:
        self.list_questions()
        return True

    def list_questions(self) -> list[Question]:
        response = self._client.get("/questions")
        response.raise_for_status()
        payload = response.json()
        return [Question(**item) for item in payload]

    def download_file(self, task_id: str, file_name: str | None = None) -> Path:
        response: httpx.Response | None = None
        last_http_error: httpx.HTTPStatusError | None = None
        for candidate_path in self._candidate_file_paths(task_id, file_name):
            response = self._client.get(candidate_path)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    last_http_error = exc
                    continue
                raise
            break
        else:
            if last_http_error is not None:
                raise last_http_error
            raise RuntimeError(f"Unable to download attachment for task {task_id}.")

        resolved_name = file_name or self._filename_from_response(response, task_id)
        safe_name = resolved_name.replace("/", "_").replace("\\", "_")
        destination = self.download_dir / f"{task_id}__{safe_name}"
        destination.write_bytes(response.content)
        return destination

    @staticmethod
    def _candidate_file_paths(task_id: str, file_name: str | None) -> list[str]:
        candidates = [f"/files/{task_id}"]
        if file_name:
            encoded_name = quote(file_name, safe="")
            candidates.extend(
                [
                    f"/files/{task_id}/{encoded_name}",
                    f"/files/{encoded_name}",
                ]
            )
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def submit_answers(
        self,
        username: str,
        agent_code: str,
        answers: Iterable[AnswerPayload],
    ) -> ScoreResponse:
        payload = {
            "username": username,
            "agent_code": agent_code,
            "answers": [asdict(answer) for answer in answers],
        }
        response = self._client.post("/submit", json=payload)
        response.raise_for_status()
        return ScoreResponse(**response.json())

    @staticmethod
    def _filename_from_response(response: httpx.Response, task_id: str) -> str:
        disposition = response.headers.get("content-disposition", "")
        for part in disposition.split(";"):
            part = part.strip()
            if part.startswith("filename="):
                return part.split("=", 1)[1].strip('"')
        return f"{task_id}.bin"
