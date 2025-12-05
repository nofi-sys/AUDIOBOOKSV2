# Bitácora de Desarrollo — Check_Audiobook

Este archivo registra decisiones, experimentos, errores y lineamientos para evolucionar la app sin romper funcionalidades existentes.

## Objetivo de la aplicación
- Comparar “Original” (guion) vs “ASR” (transcripción) por frases para control de calidad (QC).
- Permitir escuchar cada clip (inicio=tc de la fila, fin=tc siguiente), marcar filas (OK/mal) y medir WER.
- Alinear automáticamente (DTW) usando texto o lista de palabras con tiempos; aprovechar CSV/JSON word‑level cuando existe.
- Integrar utilidades: transcripción (Whisper/faster‑whisper), revisión con IA, exportar EDL, y colocar marcadores en Audacity.

## Cambio: columna tc en la UI
- Decisión: mostrar `tc` en la UI como timecode `HH:MM:SS.d` (1 decimal) para lectura humana.
- Persistencia: seguir guardando `tc` en el JSON como segundos (string, p.ej. "12.34") para mantener compatibilidad con herramientas y scripts existentes.
- Frontera clara:
  - Mostrar: `_format_tc(val)` → `"HH:MM:SS.d"`
  - Consumir/guardar: `_parse_tc(text)` → `"segundos"` (string)
- Implementación:
  - `qc_app._row_from_alignment` normaliza cualquier fila (6, 8 o 9+ columnas) a 8 columnas y formatea `tc` antes de insertarla en la tabla.
  - `qc_app.save_json` usa `idx_tc = len(row) - 3` para localizar `tc` y persistir en segundos.
  - Reproducción y lógica interna usan `_parse_tc` al leer `tc` de la tabla para no romper nada.

## Lineamientos para evitar regresiones
- Índice de `tc`: siempre es la antepenúltima columna (inmediatamente antes de `Original` y `ASR`). Evita hardcodear `5`; usa `len(row) - 3` si el contexto no garantiza 8 columnas.
- No cambiar el orden de columnas en la UI sin actualizar los adaptadores/normalizadores (`_row_from_alignment`, `save_json`).
- En módulos nuevos, tratar `tc` así:
  - UI: manejar como `HH:MM:SS.d`.
  - Persistencia/IO: segundos como string (compatibilidad con `utils/resync_python_v2`).
- Para merges/movimientos de texto entre filas, recalcular `tc` con `_recompute_tc()` (mantener monotonicidad) y luego guardar JSON.

## Próximas mejoras sugeridas (borrador)
- UX
  - Atajos más consistentes para navegación por “mal” y para reproducir fragmentos.
  - Resaltado de diferencias (diff) entre Original y ASR en el popup de texto.
  - Modo “solo pendientes” (ocultar OK) y filtros rápidos.
- Integración ASR
  - Detectar y ofrecer automáticamente el `*.words.csv`/JSON cuando existe; evitar diálogos redundantes.
  - Cache de transcripciones temporales para reintentos rápidos por fila (retranscribir clip).
- IA
  - Revisiones por lotes con límites de costo y reintentos; bitácora explicativa por fila.
- Robustez
  - Manejar codificaciones y caracteres especiales en textos y rutas.
  - Tests de no-regresión sobre `_row_from_alignment`, `_parse_tc/_format_tc` y guardado/carga JSON.

## Errores/incidencias conocidas
- Codificación de algunos textos UI aparece con glifos no ASCII en ciertos entornos (pendiente de unificar encoding fuente/TTK).

## Decisiones de estilo
- Cambios mínimos y localizados; evitar tocar módulos ajenos si no es necesario.
- Preferir funciones frontera (`_format_tc`, `_parse_tc`, `_row_from_alignment`) para adaptar formatos.

— Última actualización: inicialización de la bitácora y cambio de `tc` a timecode en UI.

## 2025-09-04 – Guided ASR v1 (heavy words)
- Nuevo flujo `--guided` en `transcriber.py`:
  - Pasa pesada `word-level` (modelo `large-v3`) con hotwords del guion.
  - Convierte a CSV (`*.words.csv`) y alinea contra el guion con `build_rows_from_words`.
  - Activa inserción de huecos “solo ASR” (`QC_INSERT_SOLO_ASR=1`) para no perder tokens.
  - Artefactos: `*.guided.txt`, `*.guided.qc.json`.
- Criterio aplicado: usar palabras de la transcripción más sofisticada (heavy) para alinear.
- Objetivo “no perder palabras”: texto plano se arma concatenando todas las palabras de la pasada pesada; los huecos en la alineación generan filas “solo ASR” en QC.

## 2025-11-19 – Anclas y depuración de alineación
- Text normalization: `prepare_paragraphs` reconstruye párrafos desde TXT con heurística (punto + salto corta párrafo, títulos cortos se separan) y `paragraphs_to_markdown` permite exportar versión `.md`.
- Alineación: `build_rows_from_words` ahora usa anclas multi-ngram (5→2) con prioridad a la primera ocurrencia y mapea segmentos completos, fijando el inicio en ASR=0 para capturar “ruido” previo.
- Persistencia: nuevo `alignment_debugger.store_alignment_snapshot` guarda anchors/parrafos/tokens en SQLite; CLI (`python -m alignment_debugger <db>`) imprime resumen rápido. En `qc_app`, hay toggle “Guardar alineación .align.db” (ON por defecto) para emitir `<asr>.align.db`; si se quiere Markdown normalizado, setear `QC_WRITE_MD=1` antes de lanzar la GUI.
- Tests: se ampliaron pruebas de normalización/paragraphs y se añadió cobertura del volcado SQLite. El suite completo falla solo por dependencias opcionales en `LEGACY/MAQUETADO` (spacy/docx/gutenberg_cleaner) y un fixture ausente `ejemplos/repetition_test.qc.json`.
