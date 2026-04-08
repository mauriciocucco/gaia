from hf_gaia_agent.cli import load_runtime_env
from hf_gaia_agent.api_client import ScoringAPIClient
load_runtime_env()

with ScoringAPIClient() as client:
    questions = client.list_questions()
    q9 = questions[8]

print("Q9:", q9.question)
