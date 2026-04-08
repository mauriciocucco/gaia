"""CLI entrypoint for the GAIA agent."""

from __future__ import annotations

import argparse
import dataclasses
import json
import dataclasses
import os
from pathlib import Path
from typing import Any

from .api_client import AnswerPayload, Question, ScoringAPIClient
from .graph import GaiaGraphAgent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value == "":
            continue
        os.environ[key] = value


def load_runtime_env() -> None:
    _load_dotenv(Path.cwd() / ".env")


def _default_cache_file() -> Path:
    return Path(".cache/gaia/last_run_answers.json")


def _resolve_attachment(client: ScoringAPIClient, question: Question) -> Path | None:
    if not question.file_name:
        return None
    return client.download_file(question.task_id, question.file_name)


def solve_questions(
    client: ScoringAPIClient,
    agent: GaiaGraphAgent,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    questions = client.list_questions()
    if limit is not None:
        questions = questions[:limit]

    results: list[dict[str, Any]] = []
    print(f"Starting evaluation of {len(questions)} questions...")
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] Solving task: {question.task_id}...")
        attachment_path = None
        attachment_error = None
        try:
            attachment_path = _resolve_attachment(client, question)
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


def write_results(results: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _print_results(results: list[dict[str, Any]]) -> None:
    print(json.dumps(results, ensure_ascii=True, indent=2))


def run_command(args: argparse.Namespace) -> int:
    with ScoringAPIClient(base_url=args.api_url) as client:
        client.health()
        agent = GaiaGraphAgent(max_iterations=args.max_iterations)
        results = solve_questions(client, agent, limit=args.limit)

    write_results(results, args.output)
    _print_results(results)
    if args.dry_run:
        print(f"\nDry run complete. Results cached at: {args.output}")
    return 0


def submit_command(args: argparse.Namespace) -> int:
    with ScoringAPIClient(base_url=args.api_url) as client:
        client.health()
        agent = GaiaGraphAgent(max_iterations=args.max_iterations)
        results = solve_questions(client, agent, limit=args.limit)
        write_results(results, args.output)
        answers = [
            AnswerPayload(
                task_id=item["task_id"],
                submitted_answer=item["submitted_answer"],
            )
            for item in results
        ]
        response = client.submit_answers(args.username, args.agent_code_url, answers)

    print(json.dumps(dataclasses.asdict(response), ensure_ascii=True, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph GAIA agent.")
    parser.set_defaults(func=None)
    parser.add_argument(
        "--api-url",
        default=os.getenv("GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"),
        help="Base URL for the scoring API.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("GAIA_MAX_ITERATIONS", "15")),
        help="Maximum agent/tool loop iterations per question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_cache_file(),
        help="Where to write the answers JSON.",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Solve questions without submitting.")
    run_parser.add_argument("--dry-run", action="store_true", help="Do not submit answers.")
    run_parser.add_argument("--limit", type=int, default=None, help="Only solve the first N questions.")
    run_parser.set_defaults(func=run_command)

    submit_parser = subparsers.add_parser("submit", help="Solve questions and submit answers.")
    submit_parser.add_argument("--username", required=True, help="Hugging Face username.")
    submit_parser.add_argument(
        "--agent-code-url",
        required=True,
        help="Public URL to the code repository or Hugging Face Space tree.",
    )
    submit_parser.add_argument("--limit", type=int, default=None, help="Only solve the first N questions.")
    submit_parser.set_defaults(func=submit_command)
    return parser


def main() -> int:
    load_runtime_env()
    parser = build_parser()
    args = parser.parse_args()
    if not args.func:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
