---
title: GAIA LangGraph Agent
emoji: "🤖"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# hf-gaia-agent

Agente con LangGraph para resolver y enviar el challenge de GAIA de la Unit 4 del curso de agentes de Hugging Face.

## Space público para verificación

Usá esta URL como `agent_code` en el endpoint de submit:

```text
https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main
```

## Comandos locales

```bash
python -m hf_gaia_agent.cli run --dry-run
python -m hf_gaia_agent.cli run --limit 3
python -m hf_gaia_agent.cli submit --username <hf_user> --agent-code-url https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main
python -m hf_gaia_agent.cli graph --format mermaid
```

## Variables de entorno

- `MODEL_PROVIDER`: `openai` o `huggingface`
- `MODEL_NAME`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `HF_TOKEN`
- `GAIA_API_URL`
- `GAIA_DOWNLOAD_DIR`
- `GAIA_MAX_ITERATIONS`

## Notas

- El backend oficial evalúa con `strip().lower()`.
- La respuesta enviada no debe incluir wrappers como `[ANSWER]`.
- El Space existe para exponer el código públicamente; el flujo de resolución y submit puede seguir corriendo en local.
- El flujo actual distingue entre `prompt_reducers` mínimos y rescates source-aware posteriores. Casos como clasificación botánica ya no se resuelven con conocimiento embebido: requieren evidencia fetchada antes de cerrar.

- `metric_row_lookup` tambien puede resolver desde texto lineal de paginas de stats y desde leaderboards rankeados cuando `extract_tables_from_url` no encuentra HTML util pero `fetch_url` si deja evidencia estructurable.
- El fallback de `botanical_classification` valida items del prompt con evidencia fetchada por item y descarta paginas de senal debil basadas solo en titulos, metadata o prompts tipo `is X a vegetable?` para reducir falsos positivos.

## Arquitectura

Hay una descripción más detallada del flujo en `docs/architecture/gaia-langgraph-agent.md`.
