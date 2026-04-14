# GAIA LangGraph Agent

> Documento de estudio para programadores que empiezan con agentes de IA.
> La idea es explicar el código real de este repo con lenguaje claro, ejemplos y un glosario.

## Resumen

Este proyecto implementa un agente para el benchmark GAIA usando LangGraph, pero la idea central no es "dejar que el modelo haga todo".

El LLM se usa como:

- planificador
- lector de evidencia
- ensamblador de respuestas cuando hace falta

El código Python se usa como:

- capa de control
- capa de ranking de fuentes
- capa de grounding
- capa de extracción determinística

La arquitectura actual está organizada en capas explícitas:

1. `graph/`: orquestación del workflow
2. `source_pipeline/`: perfilado de preguntas y ranking de fuentes
3. `reducers/`: extractores determinísticos sobre evidencia
4. `core/recoveries/`: recuperaciones reutilizables
5. `skills/`: capacidades generales orientadas a tipos de tarea
6. `adapters/`: integración específica por sitio, dominio o ecosistema

La separación importante es esta:

- el agente general vive en `graph`, `source_pipeline`, `reducers` y `core/recoveries`
- la lógica muy específica del benchmark ya no debería contaminar el core; vive en `skills/gaia` y en `adapters/`

## Arquitectura general

El flujo visible empieza en la CLI de `cli.py`, que:

- carga configuración
- consulta la API de GAIA con `api_client.py`
- descarga adjuntos cuando existen
- instancia `GaiaGraphAgent`
- llama `solve(question, local_file_path=...)`

La observabilidad se resuelve con `hooks.py`, no con monkeypatching. Eso permite ver qué tools se llaman y con qué resultados sin ensuciar la lógica principal.

`solve()` tiene dos caminos:

1. un `prompt_reducer` rápido para preguntas totalmente contenidas en el prompt
2. el `StateGraph` completo cuando hace falta buscar, leer o validar evidencia

## Qué cambió conceptualmente

Antes, mucha lógica especializada caía en una bolsa llamada `fallbacks`.

Ahora hay tres conceptos distintos:

### Core recovery

Una estrategia reusable del agente general.

Ejemplos:

- `article_to_paper`
- `text_span`

Regla práctica:

- si sirve fuera de GAIA y no depende de un sitio raro, probablemente es core recovery

### Skill

Una capacidad general del agente para resolver una familia de tareas.

Ejemplos:

- `temporal_ordered_list`
- `set_classification`

Regla práctica:

- si describe "qué tipo de problema sé resolver", es una skill

### Adapter

Una integración específica de fuente o ecosistema.

Ejemplos:

- `FightersAdapter`
- `WikipediaRosterAdapter`
- `OfficialTeamDirectoryAdapter`

Regla práctica:

- si sabe de un dominio, una estructura HTML concreta o un patrón de URL, es un adapter

## Mapa de módulos

### `graph/` - Núcleo del workflow

| Módulo | Responsabilidad |
| --- | --- |
| `workflow.py` | `GaiaGraphAgent`, construcción del `StateGraph`, `solve()` |
| `state.py` | `AgentState`: contrato de estado entre nodos |
| `tool_policy.py` | Ejecuta tools y aplica guardrails |
| `finalizer.py` | Orquesta la decisión final |
| `services.py` | Implementa los servicios que consumen workflow y finalizer |
| `contracts.py` | Protocolos chicos para evidence, recoveries, skills y finalización |
| `routing.py` | Helpers de perfilado, hints y decisiones de flujo |
| `candidate_support.py` | Dedupe de búsquedas, ranking auxiliar, selección de candidatos |
| `evidence_support.py` | Recolección de evidencia, grounding y structured answers |
| `finalization_rules.py` | Reglas finales específicas para casos sensibles |
| `retry_rules.py` | Reglas para reintentos de respuestas inválidas |
| `nudges.py` | Sugerencias que se inyectan al prompt |

### `core/recoveries/` - Recuperaciones reutilizables

Acá viven estrategias que no dependen de GAIA como benchmark.

| Recovery | Qué hace |
| --- | --- |
| `article_to_paper` | Va desde un artículo a una fuente primaria para encontrar un award number |
| `text_span` | Busca un fragmento puntual en una página o fetch completo y lo reduce |

### `skills/` - Capacidades generales

| Skill | Qué resuelve |
| --- | --- |
| `temporal_ordered_list` | Preguntas del tipo "quién está antes o después en una lista ordenada válida para una fecha" |
| `set_classification` | Clasificación determinística de ítems de un conjunto |

### `skills/gaia/` - Especializaciones del benchmark

| Skill | Qué hace |
| --- | --- |
| `botanical_gaia` | Clasifica ítems del prompt como vegetable o fruit en sentido botánico |
| `competition_gaia` | Recupera evidencia de competiciones o premios para preguntas de nacionalidad/ganador |
| `role_chain_gaia` | Resuelve preguntas de cadena entidad -> rol con dos pasos |

### `adapters/` - Integración específica por fuente

| Adapter | Qué encapsula |
| --- | --- |
| `FightersAdapter` | Heurísticas concretas de `npb.jp` y `fighters.co.jp` |
| `WikipediaRosterAdapter` | Extracción de tablas roster desde Wikipedia |
| `OfficialTeamDirectoryAdapter` | Lectura de directorios oficiales de jugadores |

### `reducers/` - Extractores determinísticos

> Acá "reducer" no significa Redux. Significa "función que reduce evidencia a una respuesta concreta".

| Reducer | Qué extrae |
| --- | --- |
| `metric_row` | Una fila o líder en una tabla de estadísticas |
| `roster` | Vecino anterior o siguiente en una lista ordenada |
| `text_span` | Un atributo desde un fragmento de texto |
| `award` | Números de subvención o beca |
| `table_compare` | Comparaciones entre celdas |
| `temporal` | Filtros por fecha o temporada |

### `source_pipeline/` - Perfilado y ranking

| Módulo | Responsabilidad |
| --- | --- |
| `question_classifier.py` | Devuelve `QuestionProfile` |
| `_question_classifiers.py` | Registro ordenado de clasificadores |
| `_question_detectors.py` | Detectores booleanos de familia de pregunta |
| `_question_extractors.py` | Fecha, autor, sujeto, text filter |
| `candidate_ranker.py` | Puntúa URLs y snippets |
| `evidence_normalizer.py` | Convierte output de tools en `EvidenceRecord` |
| `_models.py` | `QuestionProfile`, `SourceCandidate`, `EvidenceRecord` |

## El `QuestionProfile`

`QuestionProfile` es la representación estructurada de "qué tipo de pregunta parece ser esto".

Campos importantes:

- `name`: nombre del perfil
- `profile_family`: familia más general
- `expected_domains`: dominios preferidos
- `expected_date`: fecha o temporada esperada
- `subject_name`: sujeto principal
- `text_filter`: hint para tablas, links o spans
- `prompt_items`: ítems extraídos del prompt
- `classification_labels`: etiquetas de inclusión/exclusión
- `ordering_key`: clave de orden esperada
- `scope`: subgrupo relevante, por ejemplo `pitchers`

### Ejemplo 1: clasificación botánica

Pregunta:

```text
Here's the list I have so far:
broccoli, plums, sweet potatoes
Please alphabetize the vegetables...
```

Perfil esperado:

```python
QuestionProfile(
    name="list_item_classification",
    profile_family="list_item_classification",
    prompt_items=("broccoli", "plums", "sweet potatoes"),
    classification_labels={"include": "vegetable", "exclude": "fruit"},
)
```

### Ejemplo 2: roster temporal

Pregunta:

```text
Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023?
```

Perfil esperado:

```python
QuestionProfile(
    name="temporal_ordered_list",
    profile_family="temporal_ordered_list",
    expected_date="as of July 2023",
    subject_name="Taisho Tamai",
    ordering_key="jersey_number",
    scope="pitchers",
)
```

## El `AgentState` y por qué importa

En LangGraph, el estado es el contrato entre nodos. No es solo historial de chat.

Campos importantes:

- `messages`: historial para el modelo y las tools
- `question`, `file_name`, `local_file_path`: contexto base
- `iterations`, `max_iterations`: control de loops
- `tool_trace`: registro legible de tools ejecutadas
- `decision_trace`: decisiones tomadas por el sistema
- `skill_trace`: skills o adapters usados
- `question_profile`: perfil estructurado
- `ranked_candidates`: URLs ordenadas por confianza
- `search_history_fingerprints`: historial de búsquedas con fingerprint estructurado
- `structured_tool_outputs`: outputs tipados de tools
- `evidence_used`, `reducer_used`, `fallback_reason`: explicación del resultado
- `final_answer`, `error`: salida final

Esto convierte al estado en tres cosas a la vez:

- memoria de control
- memoria de calidad
- memoria de auditoría

## Ciclo completo de una pregunta

1. `cli.py` obtiene la `Question`
2. `runner.py` resuelve adjuntos si existen
3. `GaiaGraphAgent.solve()` intenta primero el `prompt_reducer`
4. si no alcanza, arma el estado inicial y ejecuta el `StateGraph`
5. `prepare_context` construye el prompt real
6. `agent` decide si responde o llama tools
7. `tools` ejecuta herramientas bajo `ToolPolicyEngine`
8. si aparece una respuesta estructurada válida, puede cortarse antes
9. si el modelo responde con un no-answer, entra `retry_invalid_answer`
10. `finalize` arbitra la respuesta final

## Cómo decide `finalize`

El orden actual es importante:

1. `preferred structured answer`
2. `core recovery from evidence`
3. `skill execution`
4. `adapter-assisted recovery`
5. reglas finales específicas
6. salvage LLM desde evidencia grounding
7. verificación final
8. error final

La idea es que lo benchmark-specific actúe tarde, no temprano.

## Cómo funciona `tool_policy`

`tool_policy.py` no es un dispatcher tonto. Hace policy enforcement:

- bloquea Python no grounding
- redirige fetches a candidatos mejores
- actualiza `ranked_candidates`
- evita loops de búsqueda
- dispara auto-followups en algunos casos

### Antes

La deduplicación de búsquedas dependía demasiado del set de tokens.

### Ahora

Usa un fingerprint estructurado con:

- `action_family`
- `entities`
- `years`
- `site`
- `source_type`
- `scope`

Eso permite distinguir entre:

- `taisho tamai npb.jp players`
- `site:fighters.co.jp taisho tamai 2023`
- `fighters 2023 roster pitchers taisho tamai`

aunque compartan varias palabras.

### Auto-fetch

El auto-fetch ahora solo ocurre si:

- existe un candidato no leído
- el candidato tiene señal fuerte real
- el score supera el umbral práctico

Si no hay señal fuerte, el sistema no fuerza lectura. En cambio, pide cambio de estrategia.

## Ejemplo detallado: `botanical_gaia`

La skill botánica ya no responde con una lista parcial "más o menos aceptable".

Hace esto:

1. extrae ítems del prompt
2. crea un estado por ítem:
   - `include`
   - `exclude`
   - `unknown`
   - `discarded`
3. intenta clasificar primero desde evidencia ya recolectada
4. solo busca para los ítems `unknown`
5. finaliza únicamente si todos los ítems relevantes están resueltos
6. arma la respuesta final en orden alfabético

### Ejemplo

Si el prompt trae:

```text
broccoli, plums, sweet potatoes
```

y la evidencia grounding dice:

- broccoli -> vegetable
- plums -> fruit
- sweet potatoes -> vegetable

la salida es:

```text
broccoli, sweet potatoes
```

Si `plums` queda ambiguo o sin evidencia fuerte, la skill no debería responder todavía.

## Ejemplo detallado: `temporal_ordered_list` + adapters

La idea de esta skill es separar:

- la lógica del problema
- de la lógica del sitio

La skill general sabe resolver:

- "quién está antes"
- "quién está después"
- "qué vecino tiene este sujeto"

si ya hay evidencia temporal grounding.

Los adapters se ocupan de conseguir esa evidencia cuando la web no es uniforme.

### Caso Fighters

`FightersAdapter` encapsula todo lo que sería feo dejar en el core:

- `npb.jp`
- `fighters.co.jp`
- URLs tipo `/team/player/detail/{year}_{id}.html`
- heurísticas de predicción de IDs

Eso permite que el core siga siendo general aunque exista un adapter hackeado para un benchmark.

## Qué queda en core y qué no

### Sí debería vivir en core

- recoveries reutilizables
- validación de grounding
- reducers
- ranking general
- tool policy general

### No debería vivir en core

- patrones de URL de un club específico
- heurísticas de un benchmark concreto
- wording ultra acoplado a una sola familia de prompts

## Términos clave

### Hook

Función que se llama en momentos importantes del flujo, por ejemplo antes y después de cada tool.

### DTO

Objeto simple que transporta datos.

En este repo:

- `QuestionProfile`
- `SourceCandidate`
- `EvidenceRecord`

### Grounding

Significa que una respuesta está apoyada en evidencia real ya recolectada, no en memoria o intuición del modelo.

### Reducer

Función determinística que toma evidencia estructurada y deriva una respuesta sin llamar al modelo.

### Skill

Capacidad general orientada a un tipo de tarea.

Ejemplo:

- `temporal_ordered_list`

### Adapter

Integración específica con una fuente o ecosistema.

Ejemplo:

- `FightersAdapter`

### Core recovery

Recuperación reusable del agente general, no acoplada a una sola fuente rara.

### Fingerprint de búsqueda

Representación estructurada de una búsqueda para detectar duplicados reales sin bloquear refinamientos válidos.

### Candidate ranking

Proceso de puntuar URLs candidatas según:

- dominio esperado
- fecha
- autor
- hints del tipo de tarea
- penalizaciones por ruido o fuentes débiles

## Glosario

| Término | Significado en este repo |
| --- | --- |
| `adapter` | Lógica específica de fuente o ecosistema |
| `award number` | Número de subvención o beca |
| `backoff` | Reintentos con espera creciente |
| `bucket` | Grupo de candidatos según calidad y estado de lectura |
| `candidate` | URL que podría contener la respuesta |
| `core recovery` | Recuperación reusable del agente general |
| `DTO` | Objeto simple de datos |
| `evidence` | Texto, tabla o dato obtenido de una fuente real |
| `fallback` | Término viejo para rescates; sigue existiendo en compatibilidad, pero ya no es el concepto principal |
| `fingerprint` | Firma estructurada de búsqueda |
| `grounded` | Apoyado en evidencia real |
| `guardrail` | Restricción que evita acciones malas o inútiles |
| `profile` | Clasificación estructurada de una pregunta |
| `prompt reducer` | Camino rápido para resolver desde el prompt sin tools |
| `reducer` | Extractor determinístico de respuesta |
| `registry` | Lista de componentes del mismo tipo que se recorre dinámicamente |
| `skill` | Capacidad general orientada a una familia de tareas |
| `StateGraph` | Grafo de LangGraph con nodos y aristas condicionales |
| `tool trace` | Registro legible de tools ejecutadas |
| `workflow` | Secuencia de pasos que sigue el agente |

## Idea reutilizable

Aunque este repo está hecho para GAIA, la plantilla conceptual sirve para otros agentes:

- `prepare_context` para clasificar y enriquecer el prompt
- `agent` para planear
- `tools` para ejecutar con guardrails
- `source_pipeline` para perfilar preguntas y rankear fuentes
- `reducers` para extraer respuestas determinísticas
- `core recoveries` para rescates reutilizables
- `skills` para capacidades generales
- `adapters` para encapsular hacks inevitables de fuente
- `finalize` para arbitrar la respuesta final

Esa es la parte realmente reusable de esta arquitectura.
