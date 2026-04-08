"""Run one or more GAIA questions by task id (or index) for local debugging."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from src.hf_gaia_agent.cli import load_runtime_env

load_runtime_env()

from src.hf_gaia_agent.api_client import ScoringAPIClient
from src.hf_gaia_agent.graph import GaiaGraphAgent


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_question.py <task_id_prefix_or_index> [task_id2 ...]")
        sys.exit(1)

    with ScoringAPIClient() as client:
        questions = client.list_questions()

    targets = sys.argv[1:]
    selected = []
    for target in targets:
        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(questions):
                selected.append((idx + 1, questions[idx]))
        else:
            for i, question in enumerate(questions):
                if question.task_id.startswith(target):
                    selected.append((i + 1, question))

    if not selected:
        print(f"No questions matched: {targets}")
        sys.exit(1)

    agent = GaiaGraphAgent()
    separator = "-" * 80

    # Patch the tools node once so multi-question reruns do not wrap an
    # already-patched method and recurse indefinitely.
    original_tools_node = GaiaGraphAgent._tools_node

    def verbose_tools_node(self_agent, state):
        last = state["messages"][-1]
        for tool_call in getattr(last, "tool_calls", []):
            args_str = str(tool_call.get("args", {}))[:150]
            print(f"  >> {tool_call['name']}({args_str})", flush=True)
        result = original_tools_node(self_agent, state)
        for message in result.get("messages", []):
            preview = str(getattr(message, "content", ""))[:200].replace("\n", " ")
            name = getattr(message, "name", "?")
            print(f"  << {name} => {preview}", flush=True)
        return result

    agent._tools_node = types.MethodType(verbose_tools_node, agent)
    agent.app = agent._build_graph().compile()

    for idx, question in selected:
        print(f"\n{'=' * 80}")
        print(f"Q{idx} | {question.task_id}")
        print(f"Question: {question.question[:120]}")
        print(f"File: {question.file_name or '(none)'}")
        print(f"{'=' * 80}\n")

        attachment_path = None
        if question.file_name:
            try:
                with ScoringAPIClient() as client:
                    attachment_path = client.download_file(
                        question.task_id, question.file_name
                    )
                print(f"Attachment downloaded: {attachment_path}")
            except Exception as exc:
                print(f"Attachment download failed: {exc}")

        result = agent.solve(question, local_file_path=attachment_path)

        print(f"\n{separator}")
        print(f"ANSWER: {result['submitted_answer']}")
        print(f"Tool trace ({len(result['tool_trace'])} calls):")
        for tool_call in result["tool_trace"]:
            print(f"  * {tool_call[:120]}")
        print(f"Decision trace: {result['decision_trace']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        if result.get("fallback_reason"):
            print(f"Fallback: {result['fallback_reason']}")
        print(separator)

        out = Path(f".cache/gaia/debug_Q{idx}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
