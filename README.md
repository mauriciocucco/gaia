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

## Setup rápido

```bash
# Opción 1 — bootstrap automático (Linux/macOS)
bash scripts/bootstrap.sh

# Opción 2 — manual
python -m venv .venv
source .venv/bin/activate          # o .venv\Scripts\Activate.ps1 en Windows
pip install -e ".[dev]"
```

### Dependencias de sistema

Las tools de video/media necesitan:

| Herramienta | Para qué se usa               | Instalación               |
| ----------- | ----------------------------- | ------------------------- |
| `ffmpeg`    | Extracción de frames de video | `sudo apt install ffmpeg` |
| `yt-dlp`    | Descarga de videos YouTube    | `pip install yt-dlp`      |

## Comandos

```bash
# Resolver preguntas sin enviar
hf-gaia-agent run --dry-run
hf-gaia-agent run --limit 3

# Enviar respuestas
hf-gaia-agent submit \
  --username <hf_user> \
  --agent-code-url https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main

# Debuggear una pregunta individual (por índice 1-based o prefijo de task_id)
hf-gaia-agent debug-question 7

# Exportar el grafo del workflow
hf-gaia-agent graph --format mermaid
hf-gaia-agent graph --format mermaid --output docs/architecture/gaia-graph.mmd

# Análisis de resultados del último run
python analyze_results.py

# Tests
python -m pytest tests/

# Export limpio del working tree actual
powershell -ExecutionPolicy Bypass -File scripts/package_clean.ps1
bash scripts/package_clean.sh
```

## Variables de entorno

| Variable              | Descripción                     | Default                                        |
| --------------------- | ------------------------------- | ---------------------------------------------- |
| `MODEL_PROVIDER`      | `openai` o `huggingface`        | —                                              |
| `MODEL_NAME`          | Nombre del modelo               | —                                              |
| `OPENAI_API_KEY`      | API key de OpenAI               | —                                              |
| `OPENAI_BASE_URL`     | Base URL custom (opcional)      | —                                              |
| `HF_TOKEN`            | Token de Hugging Face           | —                                              |
| `TAVILY_API_KEY`      | Proveedor de búsqueda principal | —                                              |
| `GAIA_API_URL`        | URL de la API de scoring        | `https://agents-course-unit4-scoring.hf.space` |
| `GAIA_DOWNLOAD_DIR`   | Directorio para adjuntos        | `.cache/gaia`                                  |
| `GAIA_MAX_ITERATIONS` | Iteraciones máximas agent↔tools | `15`                                           |

## Estructura del proyecto

```
src/hf_gaia_agent/
├── api_client.py          # Cliente HTTP para la API de scoring (con retry/backoff)
├── cli.py                 # CLI: run, submit, graph, debug-question
├── runner.py              # Orquestación de ejecución (separada de presentación)
├── hooks.py               # Sistema de hooks para observabilidad (reemplaza monkeypatch)
├── normalize.py           # Normalización de respuestas
├── evidence_solver.py     # Orquestador de reducers determinísticos
├── graph/                 # Core del workflow LangGraph
│   ├── workflow.py        #   GaiaGraphAgent + StateGraph
│   ├── state.py           #   AgentState (TypedDict rico)
│   ├── prompts.py         #   System prompt y prompt shaping
│   ├── routing.py         #   Conditional edges + perfilado de preguntas
│   ├── tool_policy.py     #   ToolPolicyEngine: políticas y followups de tools
│   ├── finalizer.py       #   WorkflowFinalizer: arbitraje final de respuesta
│   ├── contracts.py       #   Protocolos WorkflowServices, FinalizationRule, etc.
│   ├── services.py        #   Implementación concreta de WorkflowServices
│   ├── candidate_support.py #  Helpers de ranking y seguimiento de candidatos
│   ├── evidence_support.py  #  Helpers de grounding y respuestas estructuradas
│   ├── nudges.py          #   Sugerencias inyectadas al prompt
│   ├── finalization_rules.py # Reglas benchmark-specific de finalización
│   ├── retry_rules.py     #   Reglas de reintento de respuesta inválida
│   └── answer_policy.py   #   Validación y canonicalización de respuestas
├── fallbacks/             # Registry de fallback resolvers
│   ├── base.py            #   Protocolo FallbackResolver
│   ├── article_to_paper.py
│   ├── text_span.py
│   ├── roster.py
│   ├── botanical.py
│   ├── role_chain.py
│   ├── competition.py
│   └── utils.py           #   Helpers compartidos
├── reducers/              # Extractores determinísticos sobre evidencia
│   ├── base.py            #   Protocolo ReducerResult
│   ├── metric_row.py
│   ├── roster.py
│   ├── text_span.py
│   ├── award.py
│   ├── table_compare.py
│   └── temporal.py
├── source_pipeline/       # Perfilado de preguntas y ranking de fuentes
│   ├── question_classifier.py
│   ├── candidate_ranker.py
│   ├── evidence_normalizer.py
│   ├── source_labels.py
│   ├── _question_classifiers.py # Registro ordenado de clasificadores
│   ├── _question_detectors.py   # Detectores booleanos de tipo de pregunta
│   ├── _question_extractors.py  # Extractores de fechas y entidades
│   └── _models.py               # DTOs: QuestionProfile, SourceCandidate, EvidenceRecord
└── tools/                 # Herramientas del agente
    ├── search.py          #   Brave, DDG, Tavily, Wikipedia
    ├── web.py             #   fetch_url, find_text, extract_tables/links
    ├── document.py        #   Lectura de archivos locales (PDF, XLSX, etc.)
    ├── media.py           #   YouTube, video frames, audio transcription
    └── compute.py         #   calculate, execute_python_code

tests/
├── conftest.py            # Fixtures compartidas
├── test_api_client.py
├── test_normalize.py
├── test_graph.py
├── test_graph_services.py # Tests de WorkflowServices, ToolPolicyEngine y contratos
├── test_evidence_solver.py
├── test_source_pipeline.py
├── test_search_tools.py
├── test_audio_tools.py
├── test_tools.py
└── test_cli.py
```

## Notas técnicas

- El backend oficial evalúa con `strip().lower()`.
- La respuesta enviada no debe incluir wrappers como `[ANSWER]`.
- El Space existe para exponer el código públicamente; el flujo de resolución y submit corre en local.
- `metric_row_lookup` puede resolver desde texto lineal de páginas de stats y desde leaderboards rankeados cuando `extract_tables_from_url` no encuentra HTML útil.
- `award_number` prioriza matches donde el sujeto de la pregunta queda ligado localmente al award. En `article_to_paper`, puede buscar por título exacto del paper cuando el publisher primario bloquea el fetch.
- `analyze_youtube_video` extrae y transcribe el audio (Whisper) antes del análisis de frames.

## Package limpio

Para compartir o evaluar el repo sin secretos ni basura local:

```bash
# Windows / PowerShell
powershell -ExecutionPolicy Bypass -File scripts/package_clean.ps1

# Linux / macOS
bash scripts/package_clean.sh
```

Los scripts excluyen `.git`, `.env`, `.venv`, caches, artefactos de runtime/test y `__pycache__`.

## Arquitectura

Hay una descripción detallada del flujo en `docs/architecture/gaia-langgraph-agent.md`.
