from hf_gaia_agent.cli import load_runtime_env
from hf_gaia_agent.api_client import ScoringAPIClient
load_runtime_env()

with ScoringAPIClient() as client:
    questions = client.list_questions()
    q11 = questions[10]

print(q11.question)
