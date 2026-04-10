# GAIA LangGraph Agent

## Resumen

Este proyecto implementa un agente para el benchmark GAIA usando un `StateGraph` de LangGraph relativamente chico, pero reforzado con bastante logica deterministica alrededor del estado, las fuentes y la finalizacion. La idea central no es "dejar que el modelo haga todo", sino usar el LLM como planificador y lector, mientras el codigo Python impone guardrails, ranking de evidencias y reducers estructurados cuando la respuesta puede extraerse sin otra llamada al modelo.

Como caso de estudio de LangGraph, este repo es util porque separa bastante bien tres capas:

1. Orquestacion del grafo en `graph.py`
2. Perfilado de preguntas y normalizacion de evidencia en `source_pipeline.py`
3. Extraccion deterministica de respuestas desde tablas/texto en `evidence_solver.py`

## Arquitectura general

El flujo visible empieza en la CLI de `cli.py`, que:

- carga variables de entorno
- consulta la API de GAIA con `api_client.py`
- descarga adjuntos cuando existen
- instancia `GaiaGraphAgent`
- llama `solve(question, local_file_path=...)`

`solve()` tiene dos caminos:

- Primero intenta resolver por heuristicas puramente Python, antes de entrar al grafo. Esto cubre casos muy autocontenidos, como texto invertido, listas botanicas o tablas que ya vienen completas en el prompt.
- Si no aplica una heuristica, invoca el `StateGraph` compilado y devuelve un resultado enriquecido con `tool_trace`, `decision_trace`, `evidence_used`, `reducer_used` y `fallback_reason`.

Importante: no todo lo "deterministico" de este repo vive en esas heuristicas previas. Tambien hay una segunda capa de logica deterministica post-recoleccion dentro de `finalize`, que intenta rescates orientados a fuentes cuando el modelo no logra cerrar bien una familia de preguntas, pero ya existe suficiente contexto en `ranked_candidates`, `question_profile` o evidencia previa.

La capa de `tools.py` aporta herramientas de lectura y recuperacion: busqueda web, Wikipedia, fetch de paginas, extraccion de tablas, lectura de archivos locales, transcripcion de audio, analisis de YouTube y calculo/ejecucion Python acotada.

## Mapa de modulos

### `graph.py`

Es el corazon del sistema. Define:

- `AgentState`, que extiende `MessagesState`
- el prompt de sistema
- el `StateGraph`
- la politica de ruteo entre nodos
- los guardrails de busqueda, grounding y finalizacion
- rescates source-aware reutilizables para algunas familias de preguntas
- la API publica `solve()`

### `source_pipeline.py`

Convierte resultados de tools en estructuras mas utiles para razonar:

- `QuestionProfile`: clasifica el tipo de pregunta y captura pistas como dominio esperado, fecha, autor o sujeto
- `SourceCandidate`: representa URLs candidatas con score y razones
- `EvidenceRecord`: normaliza evidencia proveniente de tools

Tambien contiene el scoring de candidatos. Esa parte es importante: en este repo el LLM no elige fuentes "a ciegas"; el sistema reordena y penaliza resultados de baja calidad.

### `evidence_solver.py`

Contiene reducers deterministas que intentan responder sin otra llamada al modelo cuando la evidencia ya esta:

- comparacion de tablas
- metric row lookup
- roster neighbor lookup
- filtros temporales
- extraccion de atributos desde spans de texto
- deteccion de award numbers

Esta capa muestra un patron util de LangGraph: dejar que el grafo recolecte evidencia, pero resolver con codigo cuando la forma de la evidencia es reconocible.

Tambien conviene notar una decision de diseño reciente: algunos reducers se endurecieron para no aceptar falsos positivos "plausibles". Por ejemplo, `award_number` ya no toma cualquier palabra despues de "supported by"; exige un identificador con mezcla de letras y digitos para evitar errores tipo `National`.

### `tools.py`

Implementa las herramientas reales. No son wrappers triviales:

- varias tools agregan metadatos de URL o titulo para no perder grounding
- `read_local_file` soporta txt/csv/json/html/pdf/xlsx/audio
- `analyze_youtube_video` combina video, frames, audio y vision model
- `execute_python_code` existe, pero su uso esta fuertemente restringido desde el grafo

### `cli.py`

Expone tres entrypoints:

- `run`
- `submit`
- `graph`

`graph` es nuevo y permite exportar el Mermaid o ASCII del workflow sin depender de credenciales del modelo, usando un stub interno solo para compilar el grafo.

### `api_client.py`

Es una capa delgada sobre la API del curso:

- lista preguntas
- descarga adjuntos
- envia respuestas

## El `AgentState` y por que importa

En LangGraph, el estado es el contrato entre nodos. Aca no solo se guardan mensajes; tambien se conserva contexto operacional para controlar el comportamiento del agente.

Campos importantes:

- `messages`: historial para el modelo y las tools
- `question`, `file_name`, `local_file_path`: contexto base de la tarea
- `iterations`, `max_iterations`: control del loop agente-tools
- `tool_trace`: traza legible de invocaciones
- `decision_trace`: traza de decisiones usada para detectar patrones como busquedas repetidas
- `question_profile`: clasificacion estructurada de la pregunta
- `ranked_candidates`: URLs ordenadas por calidad percibida
- `search_history_normalized`: historial normalizado para bloquear queries semantica o casi duplicadas
- `evidence_used`, `reducer_used`, `fallback_reason`: explicabilidad de la respuesta final
- `final_answer`, `error`: salida final del flujo

La idea reusable es esta: en LangGraph no conviene pensar el estado solo como "chat history". Tambien puede ser memoria de control, memoria de calidad y memoria de auditoria.

## Ciclo completo de una pregunta

1. La CLI obtiene una `Question` y opcionalmente descarga el adjunto.
2. `GaiaGraphAgent.solve()` intenta primero `_try_heuristic_answer()`.
3. Si no hay heuristica, inicializa el estado y ejecuta `self.app.invoke(...)`.
4. `prepare_context` construye el prompt real para el modelo:
   - prompt de sistema
   - contexto del adjunto si existe
   - hints de investigacion
   - `QuestionProfile`
5. `agent` llama al modelo con tools binded.
6. Si el modelo pidio tools, `tools` ejecuta, corrige o redirige esas llamadas.
7. Despues de tools:
   - si ya existe una respuesta estructurada fiable, el flujo puede finalizar
   - si no, vuelve a `agent`
8. Si el modelo responde con algo invalido, entra `retry_invalid_answer` y vuelve a `agent` con una instruccion correctiva.
9. `finalize` decide que respuesta aceptar:
   - respuesta del modelo
   - respuesta de una tool
   - respuesta estructurada desde evidencia
   - rescates source-aware basados en candidatos/evidencia ya recolectada
   - salvage final desde evidencia top-ranked

## Nodos del grafo

### `prepare_context`

Este nodo prepara el terreno. No decide respuestas; decide como presentar el problema.

Lo importante aca es que el prompt no es fijo: se enriquece con:

- lectura previa del adjunto local
- deteccion de URLs del prompt
- hints para YouTube
- deteccion de preguntas autocontenidas
- `QuestionProfile`

Patron reusable: un nodo inicial de "prompt shaping" puede concentrar todo el preprocesamiento y evitar ensuciar el resto del flujo.

### `agent`

Este es el nodo LLM principal.

Responsabilidades:

- toma `messages`
- trunca outputs de tools demasiado largos para proteger el contexto del modelo
- agrega nudges si detecta que ya hay buenas fuentes rankeadas o demasiadas busquedas seguidas
- fuerza `tool_choice="none"` al llegar al limite de iteraciones

Conceptualmente, este nodo representa el "planner/reader", no el "owner absoluto" de la verdad.

### `tools`

Es el nodo mas interesante desde el punto de vista de producto.

No se limita a ejecutar tool calls del modelo; tambien las gobierna:

- detecta busquedas consecutivas excesivas y reemplaza una search por `fetch_url` sobre el mejor candidato no leido
- detecta queries casi duplicadas y bloquea loops de search
- redirige fetches hacia URLs mejor rankeadas
- inyecta `text_filter` derivado del `QuestionProfile`
- bloquea `execute_python_code` si no esta grounded en prompt, adjunto o evidencia previa
- si `extract_tables_from_url` falla en una pregunta de estadisticas, agrega un auto-fallback a `fetch_url`
- parsea search results y actualiza `ranked_candidates`

Patron reusable: en LangGraph, el nodo de tools puede ser un "policy enforcement layer", no solo un dispatcher.

### `retry_invalid_answer`

Es un nodo simple pero valioso. Si el modelo responde con meta-comentarios, disculpas o texto no util, el sistema no finaliza enseguida: reinyecta una instruccion precisa para que reintente usando la evidencia ya reunida.

Tambien tiene una variante especial para preguntas de roster sensibles al tiempo, donde se le recuerda al modelo que no use un roster actual o sin grounding temporal como respuesta final.

Patron reusable: en vez de tratar una respuesta invalida como fallo terminal, insertarla en un loop corto de correccion controlada.

### `finalize`

Es el arbitro final.

Orden de preferencia, simplificado:

1. si ya hay `final_answer` en el estado, la conserva
2. si falta un adjunto requerido, falla salvo que exista una respuesta concreta desde tool
3. si hay un reducer estructurado preferido y temporalmente utilizable, lo privilegia
4. si no alcanza con eso, intenta rescates source-aware por familia de pregunta:
   - `article_to_paper`: busca candidatos externos al publisher original y prueba `find_text_in_url`/`fetch_url` esperando un `award_number`
   - `text_span_lookup`: toma paginas candidatas con buen score, intenta `find_text_in_url` y si falla hace `fetch_url` completo esperando `text_span_attribute`
   - `roster_neighbor_lookup` sensible al tiempo: delega en un registry de resolvers oficiales por ecosistema/fuente
5. si la respuesta del modelo es invalida, intenta fallbacks en cascada:
   - respuesta concreta de tool
   - structured answer desde evidencia
   - salvage LLM usando solo evidencia top-grounded
   - verificacion LLM final usando esa misma evidencia
6. si nada sirve, deja error y `fallback_reason`

Este nodo muestra otra idea fuerte de LangGraph: la finalizacion no tiene por que ser "usar la ultima respuesta del modelo". Puede ser un arbitraje multi-fuente.

Otra observacion importante: estos rescates no son todos igual de generales.

- `article_to_paper` y `text_span_lookup` ya usan helpers relativamente reutilizables (`candidate_urls_from_state`, intentos de `find`/`fetch`, validacion por reducer esperado).
- `roster_neighbor_lookup` ya tiene la interfaz correcta, pero hoy el resolver realmente implementado sigue siendo uno especifico del caso Fighters/NPB. La arquitectura quedo preparada para agregar otros resolvers oficiales sin volver a meter toda la logica en un unico bloque.

## Conditional edges y forma real del workflow

El grafo compilado es:

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    prepare_context(prepare_context)
    agent(agent)
    tools(tools)
    retry_invalid_answer(retry_invalid_answer)
    finalize(finalize)
    __end__([<p>__end__</p>]):::last
    __start__ --> prepare_context;
    agent -.-> finalize;
    agent -.-> retry_invalid_answer;
    agent -.-> tools;
    prepare_context --> agent;
    retry_invalid_answer --> agent;
    tools -.-> agent;
    tools -.-> finalize;
    finalize --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

Comando para regenerarlo:

```bash
python -m hf_gaia_agent.cli graph --format mermaid
```

o escribirlo a archivo:

```bash
python -m hf_gaia_agent.cli graph --format mermaid --output docs/architecture/gaia-graph.mmd
```

La observacion clave es que el grafo es chico a proposito. La complejidad del comportamiento no esta en tener veinte nodos, sino en:

- estado rico
- guards en el nodo de tools
- reducers sobre evidencia
- finalizacion jerarquica

Es un buen recordatorio de que "usar LangGraph" no significa necesariamente construir un DAG enorme.

## LangGraph aplicado: patrones reutilizables

### 1. Grafo pequeno, logica grande

Este repo usa pocos nodos y concentra la sofisticacion en funciones puras de Python. Eso mantiene el flujo mentalmente manejable y hace mas facil testear comportamiento.

### 2. Estado como memoria operativa

`decision_trace`, `ranked_candidates` y `search_history_normalized` no son memoria conversacional; son memoria de control. Esto sirve para romper loops y guiar mejor al agente.

En las versiones mas recientes, `ranked_candidates` paso a ser aun mas importante: no solo sirve para guiar al modelo, sino tambien para rescates deterministas posteriores. Por ejemplo, `finalize` puede reutilizar esos candidatos para probar una pagina externa a un article publisher o una pagina candidata de texto exacto, sin depender de una nueva conjetura del modelo.

### 3. Reducers deterministas despues de las tools

Despues de recolectar evidencia, el sistema intenta resolver por codigo. Esto baja variabilidad y mejora grounding. Es un patron especialmente util cuando la respuesta sale de:

- tablas
- listas
- spans de texto
- fechas
- filas filtradas

### 4. Guardrails antes de ejecutar tools

El repo no confia ciegamente en la tool call del modelo. Intercepta, corrige, redirige o bloquea. Esta capa vale mucho cuando las tools son costosas o propensas a loops.

### 5. Finalizacion como arbitraje

La respuesta final puede venir del modelo, de una tool o de evidencia procesada. Esta separacion entre "explorar" y "cerrar" es muy reusable.

Una extension practica de este patron es: "cerrar" no significa solo leer la ultima evidencia, sino poder aplicar un rescue path especifico pero reusable para una familia de preguntas cuando el modelo ya encontro el terreno correcto y solo le falto el ultimo salto.

## Decisiones no obvias y tradeoffs

### Heuristicas antes del grafo

Ventaja:

- evita costo y latencia cuando el prompt ya contiene todo

Tradeoff:

- obliga a mantener una capa extra de logica puntual

### Ranking de fuentes

Ventaja:

- reduce que el modelo lea la primera URL mediocre que encontro
- favorece dominios esperados, coincidencia temporal y tipos de fuente correctos

Tradeoff:

- requiere heuristicas por dominio/tipo de pregunta

### Grounding temporal en rosters

Ventaja:

- evita respuestas aparentemente plausibles pero temporalmente incorrectas

Tradeoff:

- agrega complejidad especifica de dominio y mas reglas de validacion
- cuando se necesita llegar a una fuente oficial historica, puede requerir resolvers por ecosistema o sitio, no solo scoring generico de URLs

### Rescates source-aware reutilizables

Ventaja:

- capturan familias de error recurrentes del modelo sin hardcodear directamente una respuesta
- reutilizan `QuestionProfile`, `ranked_candidates` y reducers existentes
- permiten cerrar preguntas donde el modelo falla en el "ultimo salto", aunque ya existe evidencia suficiente o casi suficiente

Tradeoff:

- si se llevan demasiado lejos, pueden convertirse en una segunda capa opaca de negocio ad hoc
- conviene mantenerlos como patrones explicitamente nombrados y testeados, no como condiciones dispersas dentro de `finalize`

### Bloqueo de Python no grounded

Ventaja:

- evita que `execute_python_code` se convierta en una herramienta para inventar datasets o reconstruir hechos desde memoria

Tradeoff:

- algunas estrategias potencialmente utiles quedan prohibidas si no estan suficientemente fundamentadas

### Salvage final desde evidencia

Ventaja:

- rescata respuestas cuando el modelo principal fracaso en formato, no en conocimiento

Tradeoff:

- introduce una segunda/tercera oportunidad LLM, con mas complejidad de control

## Que aprender de este proyecto para futuros agentes con LangGraph

Si mañana queres construir algo con LangGraph, lo mas transferible de este repo no es GAIA sino estas decisiones:

- mantener el workflow pequeno y entendible
- invertir en un buen estado, no solo en prompts
- usar profiling para clasificar la tarea antes de actuar
- tratar tools como recursos gobernados por politicas
- resolver deterministamente cuando la evidencia ya tiene forma estructurada
- separar bien exploracion, correccion y finalizacion

Si tu proximo agente trabaja con tickets, documentos, scraping o backoffice, la plantilla conceptual es reutilizable:

- `prepare_context` para perfilar y enriquecer
- `agent` para planear
- `tools` para ejecutar con guardrails
- reducers para convertir evidencia en respuesta
- `finalize` para arbitrar la salida
