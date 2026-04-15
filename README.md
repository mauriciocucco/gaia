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

La arquitectura actual ya no gira alrededor de un único bloque de `fallbacks`. El agente está separado en capas:

- `graph/`: orquestación del workflow
- `core/recoveries/`: recuperaciones reutilizables
- `skills/`: capacidades generales orientadas a tipos de tarea
- `adapters/`: lógica específica de fuente o ecosistema
- `reducers/`: extractores determinísticos sobre evidencia
- `source_pipeline/`: perfilado de preguntas, ranking de fuentes y normalización de evidencia

## Space público para verificación

Usá esta URL como `agent_code` en el endpoint de submit:

```text
https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main
```

## Setup rápido

```bash
# Opción 1 - bootstrap automático (Linux/macOS)
bash scripts/bootstrap.sh

# Opción 2 - manual
python -m venv .venv
source .venv/bin/activate          # o .venv\Scripts\Activate.ps1 en Windows
pip install -e ".[dev]"
```

### Dependencias de sistema

Las tools de video/media necesitan:

| Herramienta | Para qué se usa | Instalación |
| --- | --- | --- |
| `ffmpeg` | Extracción de frames de video | `sudo apt install ffmpeg` |
| `yt-dlp` | Descarga de videos YouTube | `pip install yt-dlp` |

## Comandos

```bash
# Resolver preguntas sin enviar
hf-gaia-agent run --dry-run
hf-gaia-agent run --limit 3

# Enviar respuestas
hf-gaia-agent submit \
  --username <hf_user> \
  --agent-code-url https://huggingface.co/spaces/MauriSC88/gaia-langgraph-agent/tree/main

# Debuggear una pregunta individual
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

| Variable | Descripción | Default |
| --- | --- | --- |
| `MODEL_PROVIDER` | `openai` o `huggingface` | - |
| `MODEL_NAME` | Nombre del modelo | - |
| `OPENAI_API_KEY` | API key de OpenAI | - |
| `OPENAI_BASE_URL` | Base URL custom | - |
| `HF_TOKEN` | Token de Hugging Face | - |
| `TAVILY_API_KEY` | Proveedor principal de búsqueda | - |
| `GAIA_API_URL` | URL de la API de scoring | `https://agents-course-unit4-scoring.hf.space` |
| `GAIA_DOWNLOAD_DIR` | Directorio para adjuntos | `.cache/gaia` |
| `GAIA_MAX_ITERATIONS` | Iteraciones máximas agent-tools | `15` |
| `GAIA_ENABLE_BENCHMARK_FALLBACKS` | Activa skills benchmark-specific de GAIA | `1` |

## Arquitectura rápida

### 1. Workflow

`GaiaGraphAgent` construye un `StateGraph` con estos nodos principales:

- `prepare_context`
- `agent`
- `tools`
- `resolve_after_tools`
- `retry_invalid_answer`
- `finalize`

La idea no es "dejar que el modelo haga todo". El modelo planifica y lee; el código Python controla:

- qué herramientas se pueden usar
- cómo se rankean fuentes
- cuándo una respuesta está grounding
- cuándo se puede resolver en forma determinística

### 2. Recoveries, Skills y Adapters

La arquitectura nueva separa tres conceptos que antes estaban mezclados:

- `core recoveries`: rescates reutilizables del agente general
  Ejemplos: `article_to_paper`, `text_span`
- `skills`: capacidades de tarea
  Ejemplos: `temporal_ordered_list`, `botanical_gaia`
- `adapters`: integración específica de fuente o ecosistema
  Ejemplos: `FightersAdapter`, `WikipediaRosterAdapter`, `OfficialTeamDirectoryAdapter`

Una regla práctica:

- si algo describe una capacidad general, es `skill`
- si algo sabe de un sitio o un patrón de URLs concreto, es `adapter`
- si algo es recuperación reusable del core, es `core recovery`

### 3. Resolución y Finalización

Después de `tools`, el workflow pasa por `resolve_after_tools`.

Ese nodo ejecuta `run_resolution_pipeline()` en este orden:

1. respuesta estructurada disponible
2. core recoveries
3. skills
4. adapters aplicables

Si esa vía canónica cierra, el flujo finaliza ahí mismo. Si no, vuelve al `agent`.

El `finalizer` conserva una segunda barrera:

1. respuesta estructurada preferida
2. `run_resolution_pipeline()` cuando ya hay evidencia/tool outputs o se agotó el presupuesto
3. reglas finales específicas
4. salvage LLM desde evidencia grounding
5. verificación final

Eso permite resolver temprano cuando ya hay evidencia suficiente y evita que respuestas libres del modelo se cuelen al final en tareas sensibles.

### 4. Política de búsqueda

`tool_policy.py` ya no deduplica búsquedas solo por tokens ordenados. Ahora usa un fingerprint estructurado con:

- familia de acción
- entidades principales
- año o fecha
- dominio o `site:`
- tipo de fuente
- scope

Ejemplo:

- `taisho tamai npb.jp players`
- `site:fighters.co.jp taisho tamai 2023`
- `fighters 2023 roster pitchers taisho tamai`

ya no se consideran automáticamente la misma búsqueda.

Además, el `auto-fetch` solo se dispara cuando el candidato no leído tiene señal fuerte real.

## Estructura del proyecto

```text
src/hf_gaia_agent/
├── adapters/                # Integración específica por fuente o ecosistema
│   ├── base.py
│   └── temporal_roster.py
├── api_client.py
├── core/
│   └── recoveries/          # Recoveries reutilizables del agente general
│       ├── article_to_paper.py
│       ├── text_span.py
│       └── utils.py
├── cli.py
├── botanical_classification.py # Estado canónico y scoring botánico compartido
├── evidence_solver.py       # Orquesta reducers determinísticos
├── graph/                   # Workflow LangGraph y políticas del agente
│   ├── workflow.py
│   ├── tool_policy.py
│   ├── finalizer.py
│   ├── services.py
│   ├── contracts.py
│   ├── routing.py
│   ├── evidence_support.py
│   ├── candidate_support.py
│   ├── finalization_rules.py
│   └── retry_rules.py
├── hooks.py
├── normalize.py
├── reducers/                # Extractores determinísticos sobre evidencia
├── runner.py
├── skills/                  # Capacidades generales y especializaciones GAIA
│   ├── base.py
│   ├── set_classification.py
│   ├── temporal_ordered_list.py
│   └── gaia/
│       ├── botanical_gaia.py
│       ├── competition_gaia.py
│       └── role_chain_gaia.py
├── source_pipeline/         # Perfilado de preguntas y ranking de fuentes
│   ├── _prompt_items.py     # Extracción compartida de listas autocontenidas
└── tools/                   # Tools del agente
```

## Ejemplos rápidos

### Botanical GAIA

Pregunta:

```text
Here's the list I have so far:
broccoli, plums, sweet potatoes
Please alphabetize the vegetables...
```

Flujo:

1. `source_pipeline` clasifica la pregunta como `list_item_classification`
2. `_prompt_items` extrae los ítems del prompt, incluso en variantes cortas autocontenidas
3. `botanical_classification` arma un estado canónico con ítems `include | exclude | unresolved`
4. `botanical_gaia` solo busca para los ítems `unresolved`
5. el sistema solo acepta la respuesta si el cierre canónico quedó completo y coincide con la salida final
6. si la skill aborta, deja breadcrumbs `skill:botanical_gaia:*` en `decision_trace`

### Temporal ordered list

Pregunta:

```text
Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023?
```

Flujo:

1. `source_pipeline` clasifica la pregunta como `temporal_ordered_list`
2. `resolve_after_tools` intenta cerrar por la vía canónica antes de devolverle otro turno al modelo
3. el skill intenta resolver con evidencia temporal ya grounding
4. si no alcanza, los adapters aportan evidencia adicional dentro del mismo pipeline de resolución
5. el reducer `roster_neighbor` arma la respuesta final

## Notas técnicas

- El backend oficial evalúa con `strip().lower()`.
- La respuesta enviada no debe incluir wrappers como `[ANSWER]`.
- El Space existe para exponer el código públicamente; el flujo de resolución y submit corre en local.
- `metric_row_lookup` puede resolver desde texto lineal y desde leaderboards.
- `award_number` prioriza matches donde el sujeto queda ligado localmente al award.
- `analyze_youtube_video` extrae y transcribe audio antes del análisis de frames.

## Package limpio

Para compartir o evaluar el repo sin secretos ni basura local:

```bash
# Windows / PowerShell
powershell -ExecutionPolicy Bypass -File scripts/package_clean.ps1

# Linux / macOS
bash scripts/package_clean.sh
```

Los scripts excluyen `.git`, `.env`, `.venv`, caches, artefactos de runtime/test y `__pycache__`.

## Documentación de arquitectura

La explicación detallada de la arquitectura, términos y flujo real del agente está en:

[`docs/architecture/gaia-langgraph-agent.md`](docs/architecture/gaia-langgraph-agent.md)
