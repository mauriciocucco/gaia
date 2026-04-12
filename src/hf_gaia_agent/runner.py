"""Orchestration logic — runs questions and collects results.

Separates *execution* (resolve attachments, invoke agent, collect results)
from *presentation* (CLI output, JSON formatting).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .api_client import Question, ScoringAPIClient
from .graph import GaiaGraphAgent


def resolve_attachment(client: ScoringAPIClient, question: Question) -> Path | None:
    """Download the attachment for *question*, returning its local path or ``None``."""
    if not question.file_name:
        return None
    return client.download_file(question.task_id, question.file_name)


def solve_questions(
    client: ScoringAPIClient,
    agent: GaiaGraphAgent,
    *,
    limit: int | None = None,
    hook: object | None = None,
) -> list[dict[str, Any]]:
    """Solve a batch of questions, returning raw result dicts.

    The agent owns solve lifecycle hooks; ``runner`` only coordinates batch work.
    """
    del hook

    questions = client.list_questions()
    if limit is not None:
        questions = questions[:limit]

    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        attachment_path = None
        attachment_error = None
        try:
            attachment_path = resolve_attachment(client, question)
        except Exception as exc:
            attachment_error = str(exc)

        result = agent.solve(question, local_file_path=attachment_path)
        if attachment_error and not result.get("error"):
            result["error"] = f"Attachment download failed: {attachment_error}"
        result["attachment_path"] = str(attachment_path) if attachment_path else None
        result["attachment_error"] = attachment_error
        result["index"] = index

        results.append(result)

    return results


def solve_question_by_id(
    client: ScoringAPIClient,
    agent: GaiaGraphAgent,
    target: str,
    *,
    hook: object | None = None,
) -> list[dict[str, Any]]:
    """Resolve *target* (task-id prefix or 1-based index) and solve matching questions."""
    del hook
    questions = client.list_questions()
    selected: list[tuple[int, Question]] = []

    if target.isdigit():
        idx = int(target) - 1
        if 0 <= idx < len(questions):
            selected.append((idx + 1, questions[idx]))
    else:
        for i, question in enumerate(questions):
            if question.task_id.startswith(target):
                selected.append((i + 1, question))

    results: list[dict[str, Any]] = []
    for idx, question in selected:
        attachment_path = None
        attachment_error = None
        try:
            attachment_path = resolve_attachment(client, question)
        except Exception as exc:
            attachment_error = str(exc)

        result = agent.solve(question, local_file_path=attachment_path)
        result["attachment_path"] = str(attachment_path) if attachment_path else None
        result["attachment_error"] = attachment_error
        if attachment_error and not result.get("error"):
            result["error"] = f"Attachment download failed: {attachment_error}"
        result["index"] = idx

        results.append(result)

    return results


def write_results(results: list[dict[str, Any]], destination: Path) -> None:
    """Persist results as JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
