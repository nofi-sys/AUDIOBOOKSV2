# Especificación del nuevo alineador por anclas (versión para implementación)

- [x] Implementar pipeline de anclas 5-4-3 y DP por segmentos en `alignment.py`.
- [x] Construir filas por párrafo desde `word_alignment` y flags `OK/WARN/BAD/SOLO_REF`.
- [x] Persistir `ref_tokens/asr_tokens/word_alignment/paragraph_rows` en SQLite con el nuevo esquema.
- [x] Agregar herramienta de inspección CLI (`alignment_inspect.py`) y asserts de invariantes básicos.

## 0. Objetivo

Diseñar un alineador textual rígido, basado en:

* anclas de n-gramas (5, 4, 3 palabras) con regla de primera ocurrencia;
* alineación fina palabra a palabra mediante DP (Levenshtein) entre anclas;
* representación persistente en base de datos, separada de la visualización.

El tiempo de audio (`tc`) no debe intervenir en las decisiones de alineación en esta etapa. Se podrá usar más adelante para chequeos o visualización.

---

## 1. Sistema de coordenadas (modelo de datos)

Todo el algoritmo trabaja sobre un sistema de coordenadas palabra a palabra.

Para cada texto diferente (ref y asr) se define:

* una tabla de tokens en base de datos (SQLite);
* cada token tiene un índice entero único dentro de ese texto;
* todas las estructuras de alineación se refieren a tokens por su índice.

### 1.1. Tablas mínimas

```sql
-- Texto original (ref)
CREATE TABLE ref_tokens (
    idx           INTEGER PRIMARY KEY,  -- índice global de palabra
    token_norm    TEXT NOT NULL,       -- forma normalizada
    token_raw     TEXT NOT NULL,       -- forma original
    paragraph_id  INTEGER NOT NULL     -- id de párrafo al que pertenece
);

-- ASR
CREATE TABLE asr_tokens (
    idx           INTEGER PRIMARY KEY,
    token_norm    TEXT NOT NULL,
    token_raw     TEXT NOT NULL,
    tc            REAL                 -- timecode (puede ser NULL)
);

-- Alineación palabra a palabra
CREATE TABLE word_alignment (
    ref_idx   INTEGER,                 -- índice en ref_tokens
    asr_idx   INTEGER,                 -- índice en asr_tokens o NULL
    op        TEXT NOT NULL            -- 'match' | 'sub' | 'ins' | 'del'
);

-- Filas por párrafo (vista agregada)
CREATE TABLE paragraph_rows (
    row_id     INTEGER PRIMARY KEY,
    paragraph_id INTEGER NOT NULL,
    ref_start  INTEGER NOT NULL,
    ref_end    INTEGER NOT NULL,
    asr_start  INTEGER NOT NULL,
    asr_end    INTEGER NOT NULL,
    wer        REAL NOT NULL,
    flag       TEXT NOT NULL           -- 'OK' | 'WARN' | 'BAD' | 'SOLO_REF' | 'SOLO_ASR'
);
```

---

## 2. Normalización y carga de tokens

### 2.1. Preparación de párrafos (solo para etiquetar)

1. A partir del texto ORIGINAL (libro), usar `prepare_paragraphs` (LEGACY) para reconstruir los párrafos.
2. Cada palabra del original se tokeniza y se asigna a un `paragraph_id` según ese resultado.
3. Importante: los párrafos sirven sólo como metadato para reconstruir filas; no condicionan la alineación.

### 2.2. Normalización de tokens

Para ambos textos (ref y asr):

* generar `token_raw` y `token_norm`:

  * pasar a minúsculas;
  * quitar tildes;
  * eliminar puntuación excepto `.`, `?`, `!`;
  * convertir números romanos a arábigos cuando sea posible (`iv` → `4`).

* insertar en las tablas `ref_tokens` y `asr_tokens` con un `idx` secuencial desde 0.

A partir de aquí, el algoritmo trabaja sólo con `idx` y `token_norm`; `token_raw` se usa para reconstruir texto legible.

---

## 3. Extracción de anclas globales (n-gramas 5-4-3)

### 3.1. Índices de n-gramas en ASR

Para cada `n` en `{5, 4, 3}`:

* construir un índice en memoria:

```python
IndexA[n]: dict[tuple(token_norm,...,token_norm) -> list[int]]
```

* para cada posición `j` en `asr_tokens` tal que `j+n-1 < NA`:

  * `ngram = (A[j], ..., A[j+n-1])`;
  * agregar `j` a `IndexA[n][ngram]`.

No filtrar por frecuencia ni por stopwords. Todas las palabras deben poder formar parte de un ancla.

### 3.2. Búsqueda secuencial de anclas en ref

Variables:

```python
anchors = []            # lista de anclas globales
last_ref = 0            # último índice de ref aceptado
last_asr = 0            # último índice de asr aceptado
```

Para `n` en `{5, 4, 3}` en este orden:

1. Recorrer `ref_tokens` de izquierda a derecha:

   ```python
   for i in range(0, NR - n + 1):
       ngram = (R[i], ..., R[i+n-1])
       if ngram not in IndexA[n]:
           continue
       candidates = [j for j in IndexA[n][ngram] if j >= last_asr]
       if not candidates:
           continue
       j = candidates[0]    # PRIMERA ocurrencia válida en ASR
   ```

2. Condiciones para aceptar el ancla `(i, j, n)`:

   * monotonía: `j >= last_asr`;
   * separación mínima (evitar solapamientos absurdos):

     ```
     (i - last_ref) >= n and (j - last_asr) >= n
     ```

3. Si se acepta:

   * añadir a `anchors` un objeto `Anchor(ref_idx=i, asr_idx=j, n=n)`;
   * actualizar `last_ref = i`, `last_asr = j`.

No usar tiempo ni ratio tokens/tiempo en esta fase.

### 3.3. Centinelas y orden

* Al final, ordenar `anchors` por `ref_idx` (y en caso de empate por `asr_idx`).
* Insertar centinelas:

```
Anchor(ref_idx=0,    asr_idx=0,    n=0)           # inicio
Anchor(ref_idx=NR,   asr_idx=NA,   n=0)           # final
```

---

## 4. Segmentos entre anclas y alineación fina

Para cada par de anclas consecutivas `A_k` y `A_{k+1}`:

```
segR = [A_k.ref_idx + A_k.n,    A_{k+1}.ref_idx)   # después del final del n-grama
segA = [A_k.asr_idx + A_k.n,    A_{k+1}.asr_idx)
```

### 4.1. Caso general: ambos segmentos no vacíos

Si `len(segR) > 0` y `len(segA) > 0`:

1. Aplicar un DP clásico de edición sobre `token_norm`:
   * coste `0` para `match` exacto;
   * coste `1` para `sub`, `ins`, `del`.
2. Recuperar el camino óptimo y generar, para este segmento, una secuencia de operaciones:

```
(ref_idx, asr_idx, op)  # op: 'match' | 'sub' | 'ins' | 'del'
```

3. Insertar estas operaciones en `word_alignment`.

Interpretación:
* Tramos consecutivos de `match`/`sub` = anclas locales de tamaño variable.
* Tramos de sólo `ins` o sólo `del` = ruido entre anclas.

No hay filtros por WER en esta etapa; se registra todo el resultado del DP.

### 4.2. Casos borde

* Si `len(segR) > 0` y `len(segA) == 0`:
  * para cada `i` en `segR`: insertar `(ref_idx=i, asr_idx=NULL, op='del')`.

* Si `len(segR) == 0` y `len(segA) > 0`:
  * para cada `j` en `segA`: insertar `(ref_idx=NULL, asr_idx=j, op='ins')`.

---

## 5. Construcción de filas por párrafo

Las filas son una vista agregada del `word_alignment` por párrafo.

Para cada `paragraph_id = p`:

1. Determinar el rango de índices de ref del párrafo:

```
ps = primer idx de ref_tokens con paragraph_id = p
pe = último idx de ref_tokens con paragraph_id = p, +1
```

2. Extraer de `word_alignment` todas las operaciones con `ref_idx` en `[ps, pe)`.

3. Si hay alguna operación con `asr_idx` distinto de `NULL`:
   * `hs = min(asr_idx)` sobre esas ops;
   * `he = max(asr_idx) + 1`.
   * Fila normal:
     * `ref_start = ps`, `ref_end = pe`;
     * `asr_start = hs`, `asr_end = he`;
     * calcular `wer` del párrafo usando las ops del rango;
     * obtener `flag` con `_flag_for_wer(wer)`.

4. Si no hay ninguna operación con `asr_idx` distinto de `NULL`:
   * Fila `solo_ref`:
     * `ref_start = ps`, `ref_end = pe`;
     * `asr_start = asr_end =` `he` de la fila anterior (para mantener monotonía);
     * `wer = 100`, `flag = 'SOLO_REF'`.

5. Ordenar todas las filas por `paragraph_id` y `ref_start`. Garantizar invariantes:
   * `0 <= ref_start < ref_end <= NR`;
   * `0 <= asr_start <= asr_end <= NA`;
   * `asr_start` nunca disminuye de una fila a la siguiente.

6. Insertar las filas en `paragraph_rows`.

---

## 6. Invariantes que deben cumplirse siempre

El código debe verificar (con `assert` o tests) que:

* Todos los índices (`ref_idx`, `asr_idx`) están dentro de rango.
* En `word_alignment`:
  * para un mismo `ref_idx` puede haber como máximo una tupla con `asr_idx != NULL` (`match` o `sub`), el resto deben ser `del`;
  * la alineación es global: todo `ref_idx` aparece al menos una vez (como `match/sub/del`) y todo `asr_idx` aparece al menos una vez (como `match/sub/ins`).
* En `paragraph_rows`:
  * las filas están ordenadas por `paragraph_id`, luego `ref_start`;
  * los spans de ASR son monótonos (`asr_start` no retrocede);
  * no existe ninguna fila vacía (no tiene texto ni en ref ni en asr).

---

## 7. Persistencia y debug

### 7.1. Snapshot de alineación

Al finalizar el proceso de un capítulo:

* guardar en un `.db`:
  * `ref_tokens`
  * `asr_tokens`
  * `word_alignment`
  * `paragraph_rows`

Este `.db` debe poder abrirse sin pasar por la GUI.

### 7.2. Herramienta de inspección (CLI simple)

Implementar un script que permita:

* listar filas:

```
row_id | paragraph_id | ref_start-ref_end | asr_start-asr_end | wer | flag
```

* mostrar, para una fila dada (`row_id`):
  * texto ref reconstruido (`token_raw`);
  * texto asr reconstruido;
  * listado palabra-a-palabra con `op` (`match/sub/ins/del`).

Esto sirve para depurar la alineación y ajustar el algoritmo antes de tocar el JSON o la visualización.

---

## 8. No-hacer (para evitar desvíos)

Mientras se implementa esta versión:

* No usar `tc` ni pausas para aceptar/rechazar anclas.
* No filtrar n-gramas por frecuencia ni por stopwords.
* No introducir heurísticas adicionales de ratio tokens/tiempo.
* No mover límites de párrafos para mejorar WER: los párrafos vienen de `prepare_paragraphs`; las filas se proyectan desde el mapa palabra-a-palabra.

El éxito de esta versión se mide por:

* estabilidad de las anclas globales;
* coherencia del mapa palabra-a-palabra;
* facilidad para detectar visualmente el ruido (ins/del) entre segmentos bien anclados.
