from hf_gaia_agent.cli import load_runtime_env
from hf_gaia_agent.api_client import ScoringAPIClient
from hf_gaia_agent.graph import GaiaGraphAgent
load_runtime_env()

with ScoringAPIClient() as client:
    questions = client.list_questions()
    q20 = questions[19]

agent = GaiaGraphAgent(max_iterations=12)
res = agent.solve(q20)

for t in res.get("tool_trace", []):
    print("TOOL:", t.get('tool_name'), t.get('tool_args'), "->", len(str(t.get('tool_output'))))

print("SUBMITTED:", res.get("submitted_answer"))
