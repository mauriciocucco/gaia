"""Run one or more GAIA questions by task id (or index) for local debugging.

Thin wrapper around ``hf-gaia-agent debug-question``.
Usage:  python run_question.py <task_id_prefix_or_index> [task_id2 ...]
"""

from __future__ import annotations

import sys

from hf_gaia_agent.cli import load_runtime_env, main as cli_main

load_runtime_env()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_question.py <task_id_prefix_or_index> [task_id2 ...]")
        sys.exit(1)

    # Rewrite argv so the CLI parser sees: debug-question <targets...>
    sys.argv = [sys.argv[0], "debug-question", *sys.argv[1:]]
    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
