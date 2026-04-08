import os
from hf_gaia_agent.cli import load_runtime_env
from hf_gaia_agent.api_client import ScoringAPIClient
from hf_gaia_agent.graph import GaiaGraphAgent
load_runtime_env()

with ScoringAPIClient() as client:
    questions = client.list_questions()
    q20 = questions[19]  # 0-indexed

print(f"Q: {q20.question}")
agent = GaiaGraphAgent(max_iterations=12)
res = agent.solve(q20)

print("\n--- TRACE ---")
for t in res.get("tool_trace", []):
    print(f"Tool `{t.get('tool_name')}`: {t.get('tool_args')} -> length {len(str(t.get('tool_output')))}")

print("\nFINAL ANSWER:", res["submitted_answer"])
