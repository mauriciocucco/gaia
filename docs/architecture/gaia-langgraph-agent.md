# GAIA LangGraph Agent

> Documento de arquitectura del runtime actual.
> El objetivo es explicar como funciona hoy el agente, que responsabilidades tiene cada capa y donde mirar cuando algo falla.

## Resumen

Este repo implementa un agente para GAIA sobre LangGraph, pero no delega toda la resolucion al modelo.

El sistema combina:

- un LLM para planear, decidir que tools usar y producir respuestas cortas cuando la evidencia ya esta cerrada
- codigo Python para perfilar preguntas, rankear fuentes, aplicar guardrails, normalizar evidencia y resolver casos deterministas

La idea central es esta:

- el modelo propone acciones
- las tools traen evidencia
- el runtime intenta cerrar por una via canonica antes de aceptar una respuesta libre

## Estructura del proyecto

Las carpetas principales bajo `src/hf_gaia_agent/` son:

| Carpeta | Rol |
| --- | --- |
| `graph/` | Orquestacion del workflow, policy, retry, finalizacion |
| `source_pipeline/` | Question profiling, candidate ranking, normalizacion de evidencia |
| `reducers/` | Extractores deterministas que reducen evidencia a una respuesta |
| `core/recoveries/` | Estrategias reutilizables para recuperar respuestas desde evidencia o fuentes relacionadas |
| `skills/` | Capacidades orientadas a familias de tareas |
| `skills/gaia/` | Skills especificas del benchmark cuando el caso lo requiere |
| `adapters/` | Integraciones concretas por dominio, sitio o ecosistema |
| `tools/` | Herramientas expuestas al agente y sus variantes estructuradas |

Archivos importantes en la raiz del paquete:

| Archivo | Rol |
| --- | --- |
| `cli.py` | Entrada de linea de comandos |
| `runner.py` | Loop para resolver preguntas desde la API de scoring |
| `api_client.py` | Cliente contra la API de GAIA |
| `hooks.py` | Observabilidad del runtime |
| `botanical_classification.py` | Helpers compartidos para clasificacion botanica |
| `evidence_solver.py` | Resolucion determinista desde `EvidenceRecord` |

## Flujo de alto nivel

El recorrido normal de una pregunta es:

1. `cli.py` carga configuracion y crea `GaiaGraphAgent`.
2. `runner.py` obtiene la `Question` y resuelve adjuntos si hace falta.
3. `GaiaGraphAgent.solve()` intenta primero un camino corto via `prompt_reducer`.
4. Si el prompt no alcanza, construye el estado inicial y ejecuta el `StateGraph`.
5. El nodo `agent` decide entre responder o llamar tools.
6. El nodo `tools` ejecuta herramientas bajo `ToolPolicyEngine`.
7. `resolve_after_tools` intenta cerrar por una via canonica.
8. Si no alcanza, el flujo vuelve al modelo.
9. `finalize` arbitra la salida final y decide si hay respuesta o error.

La forma corta de leer el sistema es:

- `agent -> tools -> resolve_after_tools -> finalize`

con un retorno a `agent` solo cuando la evidencia todavia no alcanza.

## El prompt reducer

No toda pregunta necesita herramientas.

`try_prompt_reducer()` resuelve preguntas completamente contenidas en el prompt, por ejemplo:

- calculos directos
- transformaciones simples
- tareas con lista ya visible y sin necesidad de grounding externo

Si el reducer no puede cerrar con confianza, el agente entra al grafo completo.

## El `QuestionProfile`

`QuestionProfile` es la representacion estructurada del tipo de tarea detectado.

Campos utiles:

- `name`: nombre del perfil
- `profile_family`: familia general
- `expected_domains`: dominios preferidos
- `preferred_tools`: secuencia sugerida de tools
- `expected_date`: fecha o temporada esperada
- `subject_name`: sujeto principal
- `text_filter`: hint para spans, tablas o links
- `prompt_items`: items detectados en el prompt
- `classification_labels`: etiquetas de inclusion y exclusion
- `ordering_key`: clave de orden
- `scope`: subgrupo relevante

Ejemplo botanico:

```python
QuestionProfile(
    name="list_item_classification",
    profile_family="list_item_classification",
    prompt_items=("broccoli", "bell pepper", "sweet potatoes", "fresh basil"),
    classification_labels={"include": "vegetable", "exclude": "fruit"},
    preferred_tools=(
        "search_wikipedia",
        "fetch_wikipedia_page",
        "web_search",
        "fetch_url",
        "find_text_in_url",
    ),
)
```

El profile influye en varias capas:

- el guidance que ve el modelo
- el ranking de candidatos
- las reglas de retry y finalizacion
- las skills y adapters aplicables

## El `AgentState`

El estado no es solo historial de chat. Es el contrato entero del workflow.

Campos importantes:

- `messages`: historial de mensajes y tools
- `question`, `file_name`, `local_file_path`: contexto base
- `iterations`, `max_iterations`: control de loops
- `question_profile`: version serializada del `QuestionProfile`
- `ranked_candidates`: URLs puntuadas
- `search_history_fingerprints`: fingerprints para dedupe de busquedas
- `structured_tool_outputs`: payloads estructurados de tools
- `tool_trace`: log legible de llamadas a tools
- `decision_trace`: breadcrumbs del runtime
- `skill_trace`: skills o adapters que cerraron con exito
- `evidence_used`: evidencia usada en la respuesta final
- `reducer_used`: reducer o skill que cerro
- `recovery_reason`: razon del cierre o del error
- `final_answer`, `error`: salida final

Regla util de observabilidad:

- `decision_trace` registra intentos, abortos y decisiones intermedias
- `skill_trace` se reserva para exito real

## `graph/`: el runtime

La carpeta `graph/` concentra el control del agente.

Piezas principales:

| Modulo | Responsabilidad |
| --- | --- |
| `workflow.py` | Construccion del `StateGraph` y metodo `solve()` |
| `state.py` | Tipo de estado compartido |
| `services.py` | Servicios concretos usados por workflow y finalizer |
| `tool_policy.py` | Ejecuta tools y aplica guardrails |
| `routing.py` | Helpers de perfilado, hints y decisiones de flujo |
| `candidate_support.py` | Dedupe, buckets y seleccion de candidatos |
| `evidence_support.py` | Lectura de evidencia grounding y structured answers |
| `retry_rules.py` | Reglas para invalidar respuestas y pedir otro intento |
| `finalization_rules.py` | Reglas finales especificas para casos sensibles |
| `finalizer.py` | Arbitraje final de la respuesta |
| `prompts.py` y `nudges.py` | Prompt base y sugerencias dinamicas |

## `source_pipeline/`: perfilado, ranking y evidencia

Esta capa decide de que clase parece ser una pregunta y que fuentes merecen atencion.

Subpartes relevantes:

| Modulo | Responsabilidad |
| --- | --- |
| `_question_detectors.py` | Detectores booleanos por familia de pregunta |
| `_question_classifiers.py` | Registro ordenado de perfiles |
| `_question_extractors.py` | Extraccion de fecha, autor, sujeto, etc. |
| `candidate_ranker.py` | Ranking de resultados de busqueda y links |
| `evidence_normalizer.py` | Convierte salida de tools en `EvidenceRecord` |
| `_models.py` | DTOs como `QuestionProfile`, `SourceCandidate`, `EvidenceRecord` |

El ranking considera:

- dominio esperado
- overlap con la pregunta
- pistas del profile
- metadata fuerte del candidato
- penalizaciones por ruido comercial o paginas de bajo valor

Para botanica existe ademas una rama especifica de scoring que:

- prioriza Wikipedia y fuentes ag/extension cuando realmente matchean el item
- premia paginas con senales de clasificacion botanica
- penaliza verticales grotescos como e-commerce, payments, app stores o entertainment

## `reducers/`: cierre determinista

Un reducer toma evidencia estructurada y deriva una respuesta sin volver a preguntarle al modelo.

Ejemplos comunes:

- `metric_row`
- `roster`
- `text_span`
- `award`
- `table_compare`
- `temporal`

La regla practica es:

- si el problema puede resolverse por transformacion determinista de la evidencia ya obtenida, debe vivir en un reducer o en un helper canonico reutilizable

## `core/recoveries/`, `skills/` y `adapters/`

Estas tres capas no hacen lo mismo.

### Core recovery

Una estrategia reusable para recuperar una respuesta cuando ya hay una forma conocida de navegar desde una pista hasta la evidencia correcta.

Ejemplos:

- `article_to_paper`
- `text_span`

### Skill

Una capacidad orientada a una familia de tareas.

Ejemplos:

- `temporal_ordered_list`
- `set_classification`
- `botanical_gaia`

### Adapter

Codigo que sabe hablar con una fuente concreta o una estructura de sitio concreta.

Ejemplos:

- `WikipediaRosterAdapter`
- `OfficialTeamDirectoryAdapter`
- `FightersAdapter`

Resumen rapido:

- `recovery`: como recuperar una respuesta
- `skill`: que tipo de problema se esta resolviendo
- `adapter`: como extraer evidencia de una fuente concreta

## Resolucion canonica post-tools

Despues de cada ronda de tools, el runtime no le da automaticamente otro turno al modelo.

Primero pasa por `resolve_after_tools`, que usa `GraphWorkflowServices.run_resolution_pipeline()`.

El orden actual es:

1. structured answer disponible
2. core recoveries
3. skills
4. adapters aplicables

Esto es importante porque evita este patron malo:

- el modelo ve evidencia parcial
- improvisa una respuesta plausible
- el sistema la acepta sin haber intentado la via determinista

## Finalizacion

`finalize` es la ultima barrera de calidad.

El orden de decision es:

1. structured answer preferido
2. pipeline canonico de resolucion si ya hay evidencia o se agoto el presupuesto
3. reglas finales especificas
4. salvage LLM desde evidencia grounding
5. verificacion final
6. error

Las reglas finales existen para casos donde "respuesta plausible" no alcanza.

## Retry rules

Las retry rules deciden cuando una respuesta del modelo no puede aceptarse todavia.

Ejemplos:

- respuestas vacias o meta
- respuestas no grounding
- respuestas que contradicen un cierre canonico
- tareas sensibles donde todavia faltan items por resolver

En botanica, la regla fuerte es:

- si no hay cierre canonico, la respuesta no se acepta
- si hay cierre canonico pero no coincide con `canonical_answer`, tampoco se acepta

## `tool_policy.py`

`ToolPolicyEngine` no solo ejecuta tools. Tambien impone politica de uso.

Responsabilidades:

- bloquear Python sin grounding
- deduplicar busquedas
- redirigir fetches a mejores candidatos cuando corresponde
- mantener `ranked_candidates`
- decidir auto-followups
- frenar loops de exploracion pobres

El dedupe de busquedas usa fingerprints estructurados, no solo tokens.

Eso permite distinguir refinamientos validos entre queries parecidas pero no equivalentes.

## Caso detallado: `botanical_gaia`

La skill botanica resuelve preguntas del tipo:

```text
fresh basil, broccoli, bell pepper, sweet potatoes

Please alphabetize the vegetables and place each item in a comma separated list.
```

Su contrato es all-or-nothing:

- si todos los items relevantes quedan en `include` o `exclude`, cierra
- si alguno queda `unknown`, no entrega una lista parcial

El flujo interno es:

1. extrae los items del prompt
2. arma un estado canonico compartido con:
   - `included_items`
   - `excluded_items`
   - `unresolved_items`
   - `canonical_answer`
   - `is_closed`
3. intenta clasificar desde evidencia ya presente
4. para cada item sin resolver, prueba primero una via corta de Wikipedia
5. si Wikipedia no cierra rapido, cae a broad web botanico
6. vuelve a construir el estado canonico
7. solo finaliza si `is_closed == True`

La estrategia Wikipedia-first actual es:

- usar `search_wikipedia` para nombres de produce limpios
- elegir el mejor titulo, no cualquier match superficial
- hacer a lo sumo un `fetch_wikipedia_page`
- salir rapido si el match es debil o ambiguo

Ejemplo de salida correcta:

```text
broccoli, fresh basil, sweet potatoes
```

y no:

```text
Bell pepper, Broccoli, Fresh basil, Sweet potatoes
```

porque `bell pepper` queda excluido como fruta botanica.

### Observabilidad botanica

Cuando la skill falla, deja breadcrumbs en `decision_trace`, por ejemplo:

- `skill:botanical_gaia:profile_match`
- `skill:botanical_gaia:items_extracted=4`
- `skill:botanical_gaia:unresolved=sweet potatoes`
- `skill:botanical_gaia:aborted_partial_resolution`

Cuando cierra, deja:

- `skill_trace=["botanical_gaia"]`
- `reducer_used="botanical_classification"`

## Caso detallado: `temporal_ordered_list`

Esta skill separa dos problemas:

- resolver la logica de "antes / despues / vecino"
- conseguir evidencia temporal confiable

La skill general trabaja sobre evidencia ya grounding.
Los adapters se encargan de sitios donde la estructura no es uniforme.

Por eso un adapter como `FightersAdapter` puede contener reglas especificas de una fuente sin contaminar el runtime general.

## Donde mirar cuando algo falla

Si una respuesta salio mal, el orden de inspeccion util suele ser:

1. `question_profile`
   - la pregunta fue perfilada correctamente
2. `tool_trace`
   - el agente uso las tools adecuadas
3. `ranked_candidates`
   - los candidatos utiles subieron y el ruido bajo
4. `decision_trace`
   - hubo abortos, retries o breadcrumbs de skill
5. `evidence_used`
   - la respuesta final se apoyo en evidencia real
6. `reducer_used` y `skill_trace`
   - hubo cierre canonico real o solo salvage

Preguntas concretas que sirven:

- el problema fue de deteccion del profile
- de seleccion de fuentes
- de fetch
- de normalizacion de evidencia
- de reducer
- de policy
- o de finalizacion

## Terminos clave

| Termino | Significado en este repo |
| --- | --- |
| `QuestionProfile` | Clasificacion estructurada de la pregunta |
| `SourceCandidate` | URL candidata con score y razones |
| `EvidenceRecord` | Evidencia normalizada desde una tool |
| `grounding` | Respuesta apoyada en evidencia real ya recolectada |
| `reducer` | Funcion determinista que reduce evidencia a una respuesta |
| `skill` | Capacidad orientada a una familia de tareas |
| `adapter` | Integracion especifica por fuente o ecosistema |
| `core recovery` | Estrategia reusable para recuperar respuesta |
| `tool_trace` | Registro legible de tools ejecutadas |
| `decision_trace` | Breadcrumbs del runtime |
| `canonical answer` | Respuesta producida por la via determinista compartida |
| `search fingerprint` | Firma estructurada para dedupe de busquedas |

## Idea reusable

Aunque este repo esta orientado a GAIA, la plantilla de arquitectura es general:

- perfilar primero
- usar tools con policy
- normalizar evidencia
- intentar cierre determinista temprano
- dejar al modelo como planificador y ensamblador, no como unica fuente de verdad

Esa es la parte mas reusable del proyecto.
