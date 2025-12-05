# Guia rapida de depuracion (recorte + alineacion)

## Problema y objetivo

En el capitulo de ejemplo **"capitulo 4 parte 1"** la alineacion funciona solo en un tramo inicial y luego se degrada: el recorte del texto original no coincide bien con el ASR, aparecen filas solo_ref/solo_asr en bloques grandes y el JSON/tabla que usa la GUI llega a mostrarse vacio o incoherente.

El objetivo concreto de esta guia es:
- Dejar claro **como** se construyen hoy el recorte y la alineacion, paso a paso.
- Indicar **que funciones y archivos** revisar para detectar donde se desincronizan original y ASR.
- Servir como base para que un asesor pueda proponer ajustes al algoritmo (anclas, sentwise, trigramas, heuristicas de parrafo) sin tener que reconstruir el modelo mental desde cero.

## Flujo general
1) **Lectura**: se carga el texto original (ref) y el ASR (palabras + `tc` si vienen de CSV).
2) **Recorte** (`preprocess_clip.propose_clip`):
   - Se reconstruyen parrafos del original (`text_utils.prepare_paragraphs`).
   - Se buscan anclas de inicio y fin en el ref con ventanas 5/4/3 palabras tomadas de los extremos del ASR. Se filtran patrones genericos y los que no coinciden en numeros.
   - Si falta alguna ancla, se usan trigramas poco frecuentes (`text_utils.find_anchor_trigrams`) y como ultimo recurso alineacion sentwise.
   - Se devuelve el rango de parrafos `[start_par, end_par)` y se puede guardar con `save_clip`.
3) **Alineacion** (`alignment.build_rows_from_words`):
   - Normaliza tokens, calcula spans de parrafos y anchors n-gram.
   - Corre sentwise (ventanas por oracion y pausas) para mapear `ref_tokens` -> `asr_tokens`.
   - Construye `rows_meta` marcando filas normales, solo_ref y solo_asr, reequilibra y calcula WER/tc.
   - Opcional: guarda snapshot en SQLite (`alignment_debugger.store_alignment_snapshot`).
4) **Salida UI/JSON**: `qc_app` muestra las filas, y opcionalmente escribe `.align.db`/`.qc.json`.

## Archivos clave a revisar
- `preprocess_clip.py`: heuristica de recorte, funciones `_search_anchor_patterns`, `propose_clip`, `save_clip`.
- `text_utils.py`: `prepare_paragraphs`, `normalize`, `token_equal`, `find_anchor_trigrams`.
- `alignment.py`: `_align_sentwise`, `_bounds_from_anchors`, `_rebalance_rows`, `build_rows_from_words`.
- `qc_app.py`: integracion del recorte (boton "Recortar guion...") y ejecucion de la alineacion.
- Ejemplos para reproducir: `ejemplos/recorte/Capitulo IV ORIGINAL p1.txt`, `ejemplos/recorte/capitulo 4 parte 1 .txt`, `ejemplos/capitulo 4 parte 1 .words.csv`.

## Puntos de fallo habituales
- Anclas extremas no encontradas: revisar patrones generados (tamaños 5/4/3) y si numeros coinciden.
- Trigramas poco frecuentes demasiado laxos: ajustar `ANCHOR_MAX_FREQ` en `text_utils.py`.
- Sentwise que corta mal: revisar `alignment._sentence_spans` y la ventana `MAX_RATIO`/`PAUSE_SPLIT_SEC`.
- Filas vacias: verificar `_bounds_from_anchors` y que `rows_meta` marque solo_ref/solo_asr en lugar de dejar texto vacio.

## Como reproducir y observar
- Recorte rapido en consola (opcional):
  - `python - <<'PY'` con el snippet de `propose_clip` sobre los archivos de `ejemplos/recorte`.
- Alineacion sobre el recorte:
  - Generar ref recortado con `save_clip`, luego `alignment.build_rows_from_words(ref_clip, words, tcs)` (usar `load_words_csv` de `utils/resync_python_v2.py`).
- Snapshots de depuracion:
  - Pasar `debug_db_path="debug.align.db"` a `build_rows_from_words` para inspeccionar con `alignment_debugger.py`.

## Que pedir al asesor
- Sugerir mejoras concretas a: heuristica de anclas extremas, parametros de sentwise (`MAX_RATIO`, `PAUSE_SPLIT_SEC`), y filtrado de trigramas.
- Revisar si conviene endurecer `token_equal` para evitar falsos positivos.
- Validar si la separacion parrafo/oracion de `prepare_paragraphs` esta introduciendo spans demasiado largos o cortos para las ventanas de alineacion.***
