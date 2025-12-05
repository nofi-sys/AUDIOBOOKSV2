# RFC 0005 — Alineador paso a paso (visión heurística)

Este documento explica **qué intenta hacer el pipeline de alineación** y **cómo piensa**, usando lenguaje informal. Sirve para debug y para razonar cambios sin perderse en los detalles técnicos.

Las referencias entre paréntesis indican archivo y función, por ejemplo `qc_app.py: _run_alignment_worker`.

---

## 1. Vista general del flujo

Cuando en la GUI apretás “Alinear” con un TXT de guion y un CSV de palabras ASR:

1. **La GUI arma el contexto y llama al motor de alineación.**  
   - `qc_app.py: _run_alignment_worker`
2. **Opcionalmente se recorta el guion** a la parte que realmente se leyó.  
   - `preprocess_clip.py: propose_clip` (antes, en la ventana de recorte).
3. **Se reconstruyen párrafos “limpios” del guion** (títulos, saltos de línea raros, etc.).  
   - `alignment.py: _paragraph_spans`
4. **Se construye un mapa sentwise (por oración) entre guion y ASR.**  
   - `alignment.py: _sentence_spans`, `_align_sentwise`, `_build_rows_sentwise`
5. **Se proyectan esas corridas a nivel párrafo** y se rellenan huecos con filas “solo_ref” / “solo_asr”.  
   - `alignment.py: build_rows_from_words`
6. **Se ajustan bordes para bajar WER sin romper tiempos ni anclas.**  
   - `alignment.py: _rebalance_rows`
7. **Se calcula WER y banderas por fila y se vuelca todo a JSON (y opcionalmente SQLite).**  
   - `alignment.py: _materialize_rows`, `alignment_debugger.py: store_alignment_snapshot`

Al final, `qc_app.py` recibe una lista de filas `[ID, Flag, "", "", WER, tc, Original, ASR]` que es lo que ves en la tabla.

---

## 2. Cómo se prepara el guion (ref)

### 2.1. Lectura y normalización suave

- **Qué entra:** una ruta a TXT o PDF.  
  - `qc_app.py: _run_alignment_worker` llama a `text_utils.read_script`.
- `read_script` intenta:
  - Si es PDF, usar `pdfplumber` para juntar todas las páginas.
  - Si es TXT, probar codificaciones hasta que lea sin explotar.

Heurística: “preferir perder un poco de formato antes que tirar error”. El texto se trae lo más crudo posible; la limpieza fina se hace después.

### 2.2. Reconstrucción de párrafos

- `qc_app.py: _prepare_ref_text` llama a `text_utils.prepare_paragraphs`.
- `prepare_paragraphs`:
  - Junta líneas cortadas por salto de línea “de maquetación”.
  - Respeta saltos reales de párrafo (líneas en blanco, puntos finales).
  - Trata ciertas líneas cortas sin punto como títulos/cabeceras.

Heurística: “si parece encabezado (pocas palabras, muchas mayúsculas, números romanos), córtalo; si parece frase normal envuelta en varias líneas, júntala”.

### 2.3. Párrafos → spans de tokens

- `alignment.py: _paragraph_spans(ref_text)`:
  - Vuelve a tokenizar todo el texto (con `_tokenize` / `_normalize_token`).  
  - Para cada párrafo reconstruido, calcula `(start, end)` de tokens en un flujo plano.

Heurística: “tener un índice lineal de tokens, pero recordar en qué bloque de párrafo cae cada tramo para que la salida respete la maquetación del libro”.

---

## 3. Cómo se prepara el ASR (hyp)

### 3.1. CSV de palabras + tiempos

- `qc_app.py: _run_alignment_worker`:
  - Si el ASR es `.words.csv`, usa `utils.resync_python_v2.load_words_csv`.
  - Sale una lista `csv_words` y otra `csv_tcs` (tiempos de cada palabra).

Heurística: “trabajar a nivel palabra con tiempos reales siempre que se pueda; el texto plain es solo un fallback”.  

### 3.2. Normalización paralela ref/ASR

- En `alignment.py: build_rows_from_words`:
  - Hace `asr_tokens_norm = [_normalize_token_for_align(w) for w in asr_words]`.
  - También genera una versión “alineable” de los tokens de referencia.

`_normalize_token_for_align`:
- Baja a minúsculas, saca tildes y quita puntuación final, manteniendo el “núcleo” de la palabra.

Heurística: “tratar `Guillón`, `Guillon`, `guillon.` como la misma cosa para la similitud; que no se rompa el alineado solo por tildes o puntos”.  

---

## 4. Recorte previo del guion (preprocess_clip)

### 4.1. Dónde se usa

- `qc_app.py: _open_clip_window`:
  - Lee guion (crudo) y ASR (TXT o CSV).
  - Llama a `propose_clip(ref_text, asr_tokens, tcs)`.
  - Muestra una propuesta `[start_par, end_par)` y deja que el usuario confirme el recorte.
  - Al aceptar, guarda un TXT recortado y actualiza `self.v_ref` a ese archivo.

Heurística: “antes de alinear seriamente, intentá que ref y ASR hablen del mismo tramo de mundo; si no estás seguro, pedile ayuda al humano”.

### 4.2. Cómo busca el recorte

`preprocess_clip.py: propose_clip`:

1. Reconstruye párrafos con `prepare_paragraphs` y los tokeniza con `normalize` y `_paragraph_tokens`.
2. Genera patrones de ancla desde el ASR:
   - Primera ventana (~15 palabras) para el **inicio**.
   - Últimas (~30 palabras) para el **final**.
   - De esas ventanas, saca n-gramas 5/4/3 que tengan “palabras de contenido” (números, palabras largas) y no estén llenos de stopwords (`ANCHOR_STOPWORDS`).
3. Para cada patrón, busca el mejor encaje en el guion (`_search_anchor_patterns`):
   - Usa `token_equal` para comparar (tolerante a tildes y equivalencias tipo “19” / “diecinueve”).
   - Prefiere patrones con más contenido y mejor score medio.
4. Si no encuentra buena ancla, recurre a trigramas raros (`text_utils.find_anchor_trigrams`).
5. Si todo eso falla o el rango queda invertido, hace un alineado sentwise rápido (`alignment._align_sentwise`) y toma min/max de los índices de ref como rango.
6. Evalúa la calidad del recorte con un “smell test” de similitud (`_clip_similarity`):
   - Compara una ventana alrededor del inicio de guion recortado vs. inicio del ASR.
   - Lo mismo con el final del guion recortado vs. final del ASR.
   - Si alguno de los dos extremos se parece poco, marca `dubious=True`.

Heurística: “la posición de inicio/final la decide un combo de patrones raros + n-gramas con números, y luego un medidor de ‘esto realmente se parece al ASR o estoy recortando cualquier cosa’”.  
Si es dudoso, se lo dice a la UI y **no hay que confiar ciegamente en el rango sugerido**.

---

## 5. Motor sentwise: cómo se trazan las barandas

La idea principal: **primero alinear por oraciones con ventanas locales**, y recién después respetar párrafos y anclas.

### 5.1. Oraciones en ref

- `alignment.py: _sentence_spans(ref_tokens)`:
  - Toma la secuencia de tokens y la parte en “oraciones” usando un regex de puntos, signos de interrogación y exclamación.
  - Si una “oración” queda enorme, la corta en trozos de hasta ~80 tokens.

Heurística: “cada tramo razonable para alinear debe ser de tamaño moderado: ni una palabra suelta ni 500 palabras seguidas”.  

### 5.2. Matching local de cada oración

- `alignment.py: _match_sent_window(seg, asr, lo, hi)`:
  - Para una oración `seg` y una ventana de ASR `asr[lo:hi]`, aplica `SequenceMatcher` sobre tokens.
  - Recoge los bloques de tokens idénticos en orden y devuelve los índices de ASR donde hay coincidencias.

Heurística: “en vez de un DP global costoso, mirar oraciones y ventanas reducidas y quedarnos solo con los bloques obvios donde las palabras se repiten”.  

### 5.3. Ensamblar pares (i_ref, j_asr)

- `alignment.py: _align_sentwise`:
  - Recorre las oraciones en orden.
  - Para cada `(s, e)`:
    - Define una ventana alrededor de la última posición alineada `last` con un ancho que crece con el tamaño de la oración (`MAX_RATIO`, `MAX_EXTRA`).
    - Llama a `_match_sent_window` en esa ventana.
    - Traduce índices de ventana a pares `(i_ref, j_asr)` y los agrega a `pairs`.
  - Al final pasa por `_filter_monotonic_pairs`:
    - Tira pares fuera de rango.
    - Borra pares que retrocedan en ASR (`j_asr` nunca decrece).

Heurística: “mantené una trayectoria suave, siempre hacia adelante; si una oración no encuentra buenos matches en la vecindad esperada, mejor admitir que no sabemos antes que ‘saltearse’ medio libro”.

---

## 6. Filas sentwise (rows_meta) antes de párrafos

### 6.1. Construir corridas por oración

- `alignment.py: _build_rows_sentwise`:
  - Parte de `pairs` (ref↔asr) ya monotónicos.
  - Construye un mapa `map_h[ref_idx] = asr_idx`.
  - Para cada oración (o chunk) `(s, e)`:
    - Junta índices ASR no consumidos en ese rango (`idx`).
    - Si hay `idx`:
      - Parte los índices en corridas separadas por pausas y gaps (`_split_runs_by_pause`).
      - Elige la corrida más razonable para esa oración (`_choose_best_run`).
      - Inserta, si hace falta, una fila “solo_asr” con el hueco entre `last_h` y `hs`.
      - Inserta una fila normal `{'s': s, 'e': e, 'hs': hs, 'he': he}`.
    - Si no hay `idx`:
      - Inserta una fila marcada `solo_ref` para esa oración (asume que la oración está en guion pero no en ASR).
  - Al final, si quedó ASR sobrante, añade una fila “solo_asr” para el remanente.

Heurística: “cada oración debería tener una corrida de ASR razonable al lado; si hay huecos en ASR, se marcan explícitamente como ‘solo_asr’, y si una oración entera no se encuentra, queda como ‘solo_ref’ en vez de romper índices”.

---

## 7. De oraciones a párrafos + anclas

### 7.1. Anclas por n-gramas

- `alignment.py: _build_ngram_anchors`, `_bidirectional_anchors`, `_prune_outlier_anchors`:
  - Busca n-gramas (tamaño 5→2) que aparezcan tanto en ref como en ASR.
  - Evita reutilizar las mismas posiciones ref/ASR y respeta monotonicidad.
  - Hace una pasada forward y otra reverse para cubrir extremos.
  - Tira anclas con saltos grotescos (`_prune_outlier_anchors`).

Heurística: “identificar hitos locales (fechas, nombres propios, expresiones raras) que actúan como columnas vertebrales; no deben colapsar toda la cola contra el final si la lectura se quedó corta”.  

### 7.2. Párrafos como unidades visibles

- `alignment.py: build_rows_from_words`:
  1. Vuelve a calcular `paragraphs` y `spans` sobre el texto final de referencia.
  2. Obtiene `sentence_spans` y `ref_tokens` de trabajo.
  3. Usa `_align_sentwise` + `_build_rows_sentwise` para generar `sent_rows` (la versión sentwise con “solo_ref/solo_asr” marcados).
  4. Usa anclas n-gram (`anchors`) solamente como **plan B**:
     - Calcula `bounds = _bounds_from_anchors(spans, anchors, n_ref, n_asr)` para tener un rango ASR sugerido por párrafo.

Heurística: “las oraciones (sentwise) mandan; las anclas solo rellenan cuando una zona queda ciega para el sentwise”.  

### 7.3. Ensamblar filas de párrafo

Dentro de `build_rows_from_words`:

1. Se separan:
   - `content_rows` = filas sentwise normales (no solo_asr).
   - `asr_only_runs` = filas “solo_asr” sentwise (intermedias).
2. Se recorre cada párrafo `(s, e)`:
   - Busca `content_rows` que se solapen en referencia con ese párrafo (`_span_overlaps`).
   - Si hay candidatos:
     - Usando la proporción de solapamiento, aproxima un `(hs, he)` razonable para ese párrafo (no simplemente min/max bruto).
     - Antes de usar `hs`, “flushea” cualquier `asr_only_run` que empiece antes de `hs` y aún no se haya emitido.
   - Si no hay candidatos (párrafo sin cobertura sentwise):
     - Usa `bounds[idx]` como fallback de ASR para ese párrafo.
     - Si el rango resulta vacío o más allá del ASR, marca el párrafo como `solo_ref`.
3. Tras recorrer párrafos, descarga los `asr_only_runs` restantes y, si aún queda ASR sin usar, crea una última fila “solo_asr” final.
4. Llama a `_rebalance_rows` para hacer microajustes de bordes entre filas vecinas sin romper pausas ni anclas.
5. Completa textos, flags y WER (`_materialize_rows`) y opcionalmente construye mapeos palabra↔palabra (`_collect_word_alignments`) para depuración fina en SQLite.

Heurística: “la geometría final la dictan los párrafos, pero siempre se apoya primero en la realidad sentwise; solo si sentwise no vio nada se rellena con extrapolación de anclas. Y al final se permite mover 1–3 palabras entre filas si eso baja WER sin cruzar pausas ni fechas importantes”.  

---

## 8. Persistencia para debugging (`.align.db`)

- `alignment_debugger.py: store_alignment_snapshot`:
  - Crea un SQLite con:
    - `anchors` (todas las anclas n-gram detectadas).
    - `paragraphs` (las filas finales: spans ref/asr, textos, WER, flags).
    - `ref_tokens` / `asr_tokens` con metadatos:
      - índice, texto, párrafo/oración, enlaces prev/next, tiempos.
    - `alignments_word` (opcional), con mapeo palabra↔palabra por fila:
      - columnas `ref_idx`, `asr_idx`, `tipo` (match/ins/del/sub), `distancia`, `row_id`.

Heurística: “tener una caja negra: ver después qué spans y mapeos se usaron realmente, dónde se concentran las filas vacías, qué anclas sobrevivieron y cómo se repartió el WER”.  

---

## 9. Cómo lo usa la GUI (`qc_app.py`)

### 9.1. Pipeline del botón “Alinear”

En `qc_app.py: _run_alignment_worker`:

1. Carga y normaliza guion (`read_script` → `_prepare_ref_text`).
2. Según el tipo de ASR:
   - `.csv`: `load_words_csv` → `build_rows_from_words(ref, csv_words, csv_tcs, ...)`.
   - Texto: `build_rows(ref, hyp, ...)` y, si hay `.words.csv` al lado, rehace con `build_rows_from_words`.
3. Opcionalmente guarda markdown normalizado (`paragraphs_to_markdown`) y snapshot SQLite (`.align.db`).
4. Normaliza cada fila a las 9 columnas esperadas por la GUI (`qc_utils.canonical_row`, `_row_from_alignment`).
5. Si ya existía un `.qc.json`, fusiona metadatos antiguos (checks/AI) con las filas nuevas (`qc_utils.merge_qc_metadata`).
6. Escribe un nuevo `.qc.json` y refresca la tabla.

Heurística: “el usuario nunca ve el nivel sentwise ni las anclas; solo ve filas de párrafo con un WER, un tiempo de inicio y un ASR que puede editar. Toda la complejidad interna trabaja para que esas filas sean lo más estables y editables posible.”  

---

## 10. Cómo usar este documento para debug

Sugerido para debugging paso a paso (siguiendo tus recomendaciones):

1. **Casos mínimos con frases cortas:**
   - Construir ref/ASR pequeños y llamar a:
     - `_sentence_spans`, `_align_sentwise`, `_build_rows_sentwise`, `build_rows_from_words`.
   - Ver cómo cambian los pares `(i_ref, j_asr)` y las filas `rows_meta` cuando cambias la longitud de oraciones y pausas.
2. **Ver recorte antes de alinear:**
   - Probar `propose_clip` con distintos ASR (completo, recortado, muy corto) y mirar `start_par`, `end_par`, `sim_start`, `sim_end`, `dubious`.
3. **Inspeccionar `.align.db`:**
   - Usar `alignment_debugger.summarize_alignment` y consultas manuales en SQLite para localizar:
     - Rachas de filas sin ASR (`ref_text <> '' AND asr_text = ''`).
     - Rachas de filas vacías en ambos lados.
     - Distribución de `alignments_word` por `row_id`.
4. **Comparar con el caso “ideal”:**
   - Generar un `.align.db` usando un ref y un ASR exactamente iguales (o casi) y medir cuánto se aleja un caso real de ese comportamiento ideal en cada fase (recorte, sentwise, párrafos, rebalance).

Con esta lectura podés seguir el rastro de un problema: **¿se originó en el recorte? ¿en el sentwise que no vio nada? ¿en la proyección por párrafos? ¿o en el rebalance que movió de más?** Esa es la intención de este RFC.

