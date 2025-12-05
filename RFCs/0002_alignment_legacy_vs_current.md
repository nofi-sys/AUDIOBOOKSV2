# RFC 0002 – Comparación alineador LEGACY vs versión actual

## 1. Contexto

El proyecto tiene dos generaciones de alineador de guion vs ASR:

- **LEGACY/AUDIOBOOKSV2/alignment.py** – alineador original, probado en producción.
- **alignment.py (raíz)** – alineador nuevo con anclas n‑gram y soporte de depuración vía SQLite.

En los ejemplos recientes (`ejemplos/capitulo 4 parte 1 .align.db` / `.qc.json`) la alineación actual falla a partir de un cierto punto: los párrafos 0–70 tienen texto razonable, pero desde el id 71 aparecen filas con `ref_text` y/o `asr_text` vacíos y `tc=0`. La base de datos muestra:

- `paragraphs = 261`, `ref_tokens = 10215`, `asr_tokens = 10208`.
- Primera fila problemática: `id=71, ref_start=2868, ref_end=2903, asr_start=10208, asr_end=10209, len(ref_text)=190, len(asr_text)=0`.
- A partir de ahí, muchas filas tienen `asr_start >= asr_tokens`, por lo que el texto ASR queda vacío y el `tc` se resetea a 0.

Este RFC documenta las diferencias entre ambos alineadores y propone una vía para combinar lo mejor de los dos, manteniendo el nuevo mecanismo de depuración en SQLite y minimizando cambios en la app actual.

---

## 2. Alineador LEGACY (resumen)

Archivo: `LEGACY/AUDIOBOOKSV2/alignment.py`

### 2.1. Tokenización y utilidades

- `_normalize_token`:
  - Minúsculas + eliminación de acentos.
  - Sustituye puntuación por espacios, pero **conserva los signos de fin de frase** para el splitting.
- `_tokenize(text)`:
  - Aplica `_normalize_token` y divide en tokens por espacios.
- `_wer_pct` y `_flag_for_wer`:
  - Calculan WER sobre tokens y devuelven un flag discreto (`✔.`, `¿?`, `✗`) en función del umbral `WARN_WER` y la longitud de la referencia.

### 2.2. Segmentación por oraciones

- `_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")`.
- `_sentence_spans(ref_tokens)`:
  - Reconstruye el texto `ref` a partir de `ref_tokens` y lo corta por signos de cierre de oración.
  - Devuelve spans `(start, end)` sobre el array de tokens.
  - Si no hay signos de cierre pero sí tokens, devuelve un único span `(0, len(ref_tokens))`.

### 2.3. Alineación por ventana local (“sentwise”)

- `_match_sent_window(seg, asr, lo, hi)`:
  - Usa `SequenceMatcher` sobre `seg` (tokens de referencia de una oración) y una ventana local `asr[lo:hi]`.
  - Devuelve índices de ASR donde hay bloques coincidentes.

- `_align_sentwise(ref_tokens, asr_tokens, tcs)`:
  - Recorre las oraciones de referencia (`_sentence_spans`).
  - Para cada oración `(s, e)`:
    - Define una ventana `[lo, win)` alrededor de `last`:
      - `win` crece con la longitud de la oración y un margen.
      - Se corta si hay una pausa larga en los `tcs` (> `PAUSE_SPLIT_SEC`).
    - Llama a `_match_sent_window` para obtener índices `j` de ASR.
    - Traduce cada `j` a un par `(i_ref, j_asr)` aproximado, usando el offset dentro del segmento.
    - Actualiza `last` al final de la última coincidencia para avanzar la ventana.
  - Resultado: una lista de pares `(i_ref, j_asr)` que sirven de “barandas” diagonales, respetando la monotonía y las pausas reales.

### 2.4. Construcción de filas a partir de palabras

`build_rows_from_words(ref, csv_words, csv_tcs)`:

1. Tokeniza referencia y ASR:
   - `ref_tokens = _tokenize(ref)`.
   - `asr_tokens = [_normalize_token(w) for w in csv_words]`.
2. Obtiene pares `(i_ref, j_asr)` con `_align_sentwise`.
3. Para cada oración `(s, e)`:
   - Recoge índices de ASR mapeados para esa oración (filtrando repetidos con `consumed`).
   - Si hay índices:
     - Los agrupa en corridas con `_split_runs_by_pause` (limita gaps de índice y huecos de tiempo).
     - Elige la mejor corrida con `_choose_best_run`, balanceando:
       - Cobertura (#tokens ASR mapeados).
       - Duración vs duracion esperada (`WPS`).
     - Crea una fila para la oración `(s, e)` con `hs, he` obtenidos.
     - Inserta, si corresponde, una fila “solo ASR” entre `last_h` y `hs` (contenido extra del ASR no alineado).
     - Actualiza `last_h = he` y marca `consumed` el tramo `[hs, he)`.
   - Si no hay índices (oración sin matches fiables):
     - Crea una fila con `hs == he == last_h` (tc heredado).
4. Si al final queda ASR sin consumir (`last_h < len(asr_tokens)`):
   - Se añade una fila final “solo ASR” para `[last_h, len(asr_tokens))`.
5. `_recompute_row` calcula `WER`, flag, textos y `tc` (usando `tcs[hs]`).
6. `_rebalance_rows` hace una pasada local, moviendo 1–3 palabras entre filas adyacentes:
   - Solo acepta movimientos que mejoran la suma de WER de las dos filas.
   - No cruza pausas largas (`REBALANCE_MAX_PAUSE`) ni mueve anchors (meses, números).
7. Salida: filas `[ID, flag, "", "", WER, tc, Original, ASR]` (tras la capa de normalización de la GUI).

**Claves del método LEGACY**

- Alineación **local por oración** usando ventanas deslizantes y tiempos (`tcs`).
- Manejo explícito de trozos “solo ASR” y de oraciones sin match (no desaparece el texto de referencia).
- Rebalanceo local que corrige bordes sin romper la monotonía ni anchors sensibles.

---

## 3. Alineador actual (resumen)

Archivo: `alignment.py` (raíz).

### 3.1. Cambios estructurales

- Se añadió soporte para:
  - Reconstrucción de párrafos a partir de TXT con `text_utils.prepare_paragraphs` (heurísticas de títulos/line-breaks).
  - Exportación opcional a Markdown (`paragraphs_to_markdown`).
  - Depuración en SQLite (`alignment_debugger.store_alignment_snapshot`) desde `build_rows_from_words`.
  - Normalización adicional (romanos→arábigos en `_normalize_token`).

- Alineación basada en **anclas de n‑gramas** globales:
  - `_build_ngram_anchors` construye anclas `(ref_idx, asr_idx, size)` para n‑gramas de tamaño 5→2, priorizando la primera ocurrencia en ASR y manteniendo monotonía.
  - `_bidirectional_anchors` combina anclas forward y reverse, y `_prune_outlier_anchors` intenta filtrar outliers con ratios extremos.
  - `_bounds_from_anchors` reparte para cada span de párrafo `(s, e)` un intervalo `(hs, he)` en ASR por interpolación entre anclas y sentinelas `(0,0)` y `(n_ref,n_asr)`.

### 3.2. Segmentación de referencia

- `_paragraph_spans(ref_text)`:
  - Usa `prepare_paragraphs(ref_text)` para reconstruir párrafos.
  - Si solo hay un párrafo, intenta trocear por oraciones y, si no hay puntuación, por bloques fijos de tokens (p.ej. 120).
  - Devuelve:
    - Lista de párrafos (texto).
    - Spans `(s, e)` sobre los tokens.
    - Lista de tokens plana `ref_tokens`.

### 3.3. Construcción desde anclas

`build_rows_from_words(ref, asr_words, tcs, markdown_output=None, debug_db_path=None)`:

1. Obtiene `paragraphs, spans, ref_tokens` con `_paragraph_spans(ref)`.
2. Normaliza `asr_words` → `asr_tokens_norm` con `_normalize_token`.
3. Construye anclas globales:
   - `_bidirectional_anchors` + `_greedy_anchors` como refuerzo + `_prune_outlier_anchors`.
4. Computa `bounds = _bounds_from_anchors(spans, anchors, n_ref, n_asr)`:
   - Cada párrafo `(s, e)` recibe un `(hs, he)` estimado por interpolación.
   - Se ajustan `hs`, `he` para mantener monotonía y no salirse (teóricamente) de `[0, n_asr]`.
5. Crea `rows_meta` con una fila por párrafo: `'s','e','hs','he'`.
6. `_recompute_row` calcula WER/flag/textos/`tc` por fila; `_rebalance_rows` aplica el mismo rebalance legacy.
7. Si `debug_db_path` está activado, vuelca a SQLite: párrafos, tokens, anchors, filas y tiempos.

**Claves del método actual**

- Alineación **global basada en anclas n‑gram**, repartiendo ASR sobre spans de párrafo.
- Depuración estructurada en SQLite y salida en Markdown.
- Reutiliza el mismo módulo de rebalance, pero sin lógica explícita de “solo ASR / solo referencia”.

---

## 4. Análisis del fallo actual

En `ejemplos/capitulo 4 parte 1 .align.db`:

- `ref_tokens = 10215`, `asr_tokens = 10208`.
- Párrafos (`spans`) derivados del texto ORIGINAL ⇒ `261` spans, muchos más que en el TXT ASR procesado.
- Anchors finales en el DB antiguo (antes de los últimos ajustes) incluyen pares muy tardíos con ratios extremos, p.ej. `(ref_idx=2518, asr_idx=10204)`, combinados con el sentinel `(n_ref, n_asr)`. Eso hace que un número grande de párrafos se “compriman” en los últimos tokens del ASR.
- En el DB actual:
  - La primera fila con `asr_start >= asr_tokens` es `id=71`:
    - `id=70`: `asr_start=10207, asr_end=10208` – aún dentro del rango.
    - `id=71`: `asr_start=10208, asr_end=10209` – **fuera de rango** (ASR tiene índices 0..10207).
  - A partir de ahí, los párrafos 71+ comparten `asr_start=10208`/`asr_end=10209`, por lo que `asr_text` queda vacío y el `tc` cae a 0.0 en el volcado.

Conclusiones:

- El método de anclas globales reparte correctamente la parte inicial del texto, pero cuando el guion es significativamente más largo o divergente que el ASR, las anclas finales y la interpolación arrastran los últimos párrafos más allá del final del ASR.
- Falta manejo explícito de:
  - Segmentos de referencia **no cubiertos** por el ASR (deberían ser “solo referencia” con WER=100, pero con texto visible).
  - Segmentos “solo ASR” entre anclas (equivalente a la lógica legacy).

Por eso **en algunos casos funciona (cuando ref≈ASR, p.ej. aligning ref con el TXT ASR)** y **en otros se rompe la cola del texto (cuando ref es el ORIGINAL y hay más contenido o divergencias)**.

---

## 5. Diferencias clave LEGACY vs actual

1. **Unidad de segmentación**:
   - LEGACY: oraciones (`_sentence_spans`) sobre `ref_tokens`.
   - Actual: párrafos (heurísticos) con fallback a bloques de tokens; no se usan directamente las oraciones para alinear.

2. **Estrategia de alineación**:
   - LEGACY:
     - Local por oración, con ventana acotada en ASR, guiada por `last` y pausas en `tcs`.
     - Usa `SequenceMatcher` para aprovechar similitud textual, pero limitada a una ventana viable.
   - Actual:
     - Anclas globales de n‑gramas + interpolación lineal para repartir todo el ASR entre todos los spans.
     - Usa tiempos solo de forma secundaria (p.ej. en el rebalance).

3. **Manejo de zonas sin correspondencia**:
   - LEGACY:
     - Inserta filas “solo ASR” entre corridas de indices ASR no asignados.
     - Para oraciones sin match, crea fila con `hs == he == last_h` pero con `txt_ref` lleno y `txt_asr` vacío.
   - Actual:
     - No distingue explícitamente “solo ASR” / “solo ref”; cualquier span que cae después de agotar `asr_tokens` termina con `(hs, he)` fuera de rango y el texto ASR se pierde.

4. **Uso de tiempos (`tcs`)**:
   - LEGACY: central en `_split_runs_by_pause` y `_choose_best_run`; modela duración esperada de cada oración (`WPS`) y evita saltos temporales antinaturales.
   - Actual: tiempos se usan para rebalances y, de forma indirecta, en funciones heredadas, pero el corazón de la asignación `(s,e)→(hs,he)` se guía por anclas, no por tiempos.

5. **Depuración y persistencia**:
   - LEGACY: sin DB; se apoyaba en tests y logs ad‑hoc.
   - Actual: snapshot estructurado en SQLite (`anchors`, `paragraphs`, `ref_tokens`, `asr_tokens`) + CLI `alignment_debugger`.

---

## 6. Hipótesis sobre el problema actual

1. **Interpolación global vs divergencias locales**  
   Cuando el guion original incluye material que no está leído (o está muy alterado) en el audio, la distribución proporcional basada en anclas tiende a “empujar” los últimos párrafos hacia el final del ASR. Si los anclajes de cola son escasos o poco fiables, una gran cantidad de texto de referencia termina mapeado a pocos tokens ASR o más allá del último índice.

2. **Ausencia de “solo referencia” explícito**  
   El pipeline actual no marca de forma explícita los tramos donde ya no hay ASR disponible. En lugar de generar filas “solo referencia” con texto visible y WER=100, produce pares `(hs, he)` fuera de rango y la lógica de materialización vacía el texto ASR y resetea `tc`. Esto da la sensación de “JSON vacío” a partir de cierto punto.

3. **Pérdida de la geometría temporal**  
   El alineador LEGACY usa `tcs` y una expectativa de duración (`WPS`) para decidir cuál corrida es más plausible por oración. El nuevo alineador confía mucho más en coincidencias textuales de n‑gramas y menos en las pausas/duración, lo que lo vuelve más frágil frente a repeticiones, inserciones largas o saltos de lectura.

---

## 7. Propuesta de combinación “mejor de ambos”

Objetivos:

- Mantener:
  - El esquema de depuración en SQLite (`alignment_debugger`).
  - La reconstrucción de párrafos/Markdown.
  - La API y formato de salida esperados por `qc_app.py` (filas de 8 columnas).

- Recuperar del LEGACY:
  - Alineación local por oración con ventanas guiadas por `tcs`.
  - Manejo explícito de “solo ASR” y “solo referencia”.

### 7.1. Paso 1 – Corrección mínima para no perder texto

Cambios acotados en el alineador actual:

1. **Clamp seguro en `_bounds_from_anchors`**:
   - Asegurarse de que `hs < n_asr` siempre; si la interpolación da `hs >= n_asr`, marcar ese span como “solo referencia”:
     - `hs = he = n_asr - 1` (o usar último índice válido).
     - Materializar con `txt_ref` lleno y `txt_asr` vacío, WER=100, `tc` heredado.
2. **Fila final “solo referencia”**:
   - Si quedan spans de referencia más allá del último anchor/sentinel, consolidarlos explícitamente en una fila o conjunto de filas “solo referencia” con texto visible.

Estos ajustes no arreglan la alineación semántica, pero **evitan que el JSON se vea vacío** y hacen que el contenido de referencia siga presente y audible (al menos con tc correcto hasta el último ASR disponible).

### 7.2. Paso 2 – Reintroducir el núcleo LEGACY como motor principal

Crear una nueva implementación interna, reutilizando código LEGACY:

- `def _build_rows_sentwise(ref: str, asr_words: List[str], tcs: List[float]) -> List[List]:`
  - Copiar/adaptar `_sentence_spans`, `_align_sentwise`, `_split_runs_by_pause`, `_choose_best_run`, `_rebalance_rows` de LEGACY.
  - Generar `rows_meta` con las mismas reglas de “solo ASR” y “solo ref”.
  - Al final, pasar `rows_meta` por el mismo bloque de snapshot SQLite (`alignment_debugger`) que ya existe.

Luego:

- Hacer que `build_rows_from_words` llame **primero** a `_build_rows_sentwise` como alineador base.
- Usar las anclas n‑gram del alineador nuevo solo como:
  - Medida de calidad (p.ej. registrar cuántas anclas se respetan).
  - Pistas opcionales para elegir ventanas iniciales (`approx`) en `_align_sentwise` (sin romper el comportamiento si fallan).

De esta forma:

- Se recupera el comportamiento probado (alineación local tipo LEGACY).
- Se mantiene la nueva infraestructura de depuración y la normalización extra (romanos, párrafos, Markdown).

### 7.3. Paso 3 – Refinos opcionales

- Ajustar `prepare_paragraphs` para imitar más de cerca la lógica de maquetado en LEGACY (detección de títulos, líneas en mayúsculas, etc.), reutilizando funciones de `LEGACY/MAQUETADO/txt2md_mvp/utils.py` cuando tenga sentido.
- Añadir un flag interno o de GUI (no expuesto al usuario final de primera instancia) para alternar entre “alineador legacy” y “alineador n‑gram” durante la fase de pruebas.

---

## 8. Siguientes pasos recomendados

1. **Implementar Paso 1** (clamp + filas “solo ref”) para eliminar inmediatamente la sensación de JSON vacío y estabilizar la UI.
2. **Portar `_build_rows_from_words` LEGACY como `_build_rows_sentwise` en el módulo actual**, y cambiar `build_rows_from_words` para usarlo como motor principal, dejando el código de anclas como capa de depuración/validación.
3. Regenerar la alineación para `ejemplos/capitulo 4 parte 1` usando el nuevo pipeline y validar manualmente:
   - Que el texto original se conserva en todas las filas.
   - Que los tiempos (`tc`) son crecientes y razonables.
   - Que el `.align.db` refleja anchors y spans coherentes.
4. Solo después de esa estabilización, evaluar mejoras más avanzadas (p.ej. uso de spaCy/NER como señal adicional para anclas), siempre como ayuda para la alineación, no como normalización destructiva de texto.

