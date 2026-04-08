with open("src/hf_gaia_agent/graph.py", "r", encoding="utf-8") as f:
    text = f.read()
import re
text = re.sub(r'\}, config=\{"recursion_limit": 100\}, config=\{"recursion_limit": 100\}\)', '}, config={"recursion_limit": 50})', text)
with open("src/hf_gaia_agent/graph.py", "w", encoding="utf-8") as f:
    f.write(text)
