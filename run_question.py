"""Run a single GAIA question by task_id (or index) for local debugging."""

from __future__ import annotations

import json
import sys
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
            for i, q in enumerate(questions):
                if q.task_id.startswith(target):
                    selected.append((i + 1, q))

    if not selected:
        print(f"No questions matched: {targets}")
        sys.exit(1)

    agent = GaiaGraphAgent()

    for idx, question in selected:
        print(f"\n{'='*80}")
        print(f"Q{idx} | {question.task_id}")
        print(f"Question: {question.question[:120]}")
        print(f"File: {question.file_name or '(none)'}")
        print(f"{'='*80}\n")

        # Try to download attachment
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

        # Add verbose callback to see tool calls in real time
        import types
        original_tools_node = agent._tools_node.__func__

        def verbose_tools_node(self_agent, state):
            last = state["messages"][-1]
            for tc in getattr(last, "tool_calls", []):
                args_str = str(tc.get("args", {}))[:150]
                print(f"  >> {tc['name']}({args_str})", flush=True)
            result = original_tools_node(self_agent, state)
            for msg in result.get("messages", []):
                preview = str(getattr(msg, "content", ""))[:200].replace("\n", " ")
                name = getattr(msg, "name", "?")
                print(f"  << {name} => {preview}", flush=True)
            return result

        agent._tools_node = types.MethodType(verbose_tools_node, agent)
        # Rebuild the graph so it picks up the patched method
        agent.app = agent._build_graph().compile()

        result = agent.solve(question, local_file_path=attachment_path)

        print(f"\n{'─'*80}")
        print(f"ANSWER: {result['submitted_answer']}")
        print(f"Tool trace ({len(result['tool_trace'])} calls):")
        for t in result["tool_trace"]:
            print(f"  • {t[:120]}")
        print(f"Decision trace: {result['decision_trace']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        if result.get("fallback_reason"):
            print(f"Fallback: {result['fallback_reason']}")
        print(f"{'─'*80}")

        # Save individual result
        out = Path(f".cache/gaia/debug_Q{idx}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
