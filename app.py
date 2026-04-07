import gradio as gr


README = """
# GAIA LangGraph Agent

This Space hosts the source code for my Hugging Face Agents Course Unit 4 submission.

Run the agent locally with:

- `python -m hf_gaia_agent.cli run --dry-run`
- `python -m hf_gaia_agent.cli submit --username <hf_user> --agent-code-url <this_space_tree_url>`

The public code link for submission is:
`https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main`
"""


with gr.Blocks(title="GAIA LangGraph Agent") as demo:
    gr.Markdown(README)


if __name__ == "__main__":
    demo.launch()
