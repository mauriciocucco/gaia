from hf_gaia_agent.cli import load_runtime_env
from hf_gaia_agent.api_client import ScoringAPIClient
from hf_gaia_agent.graph import GaiaGraphAgent
load_runtime_env()

with ScoringAPIClient() as client:
    questions = client.list_questions()
    q11 = questions[10]

agent = GaiaGraphAgent(max_iterations=10)
res = agent.solve(q11)

print("FINAL ANSWER:", res.get("submitted_answer"))
