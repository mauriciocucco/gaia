"""Submit cached answers without re-running the agent."""
import dataclasses
import json
import sys
from pathlib import Path

from hf_gaia_agent.api_client import AnswerPayload, ScoringAPIClient
from hf_gaia_agent.cli import load_runtime_env

load_runtime_env()

cache_file = Path(".cache/gaia/last_run_answers.json")
username = sys.argv[1] if len(sys.argv) > 1 else "MauriSC88"
agent_url = sys.argv[2] if len(sys.argv) > 2 else "https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main"

data = json.loads(cache_file.read_text())
answers = [
    AnswerPayload(task_id=item["task_id"], submitted_answer=item["submitted_answer"])
    for item in data
]

with ScoringAPIClient() as client:
    response = client.submit_answers(username, agent_url, answers)

print(json.dumps(dataclasses.asdict(response), indent=2))
