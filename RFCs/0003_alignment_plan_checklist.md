# RFC 0003 — Plan de implementación alineación (basado en RFC 0002)

Este documento resume el plan de trabajo derivado de `RFCs/0002_alignment_legacy_vs_current.md` y de los pedidos extra del autor, en forma de checklist.

## Referencias

- Diseño y análisis comparativo: `RFCs/0002_alignment_legacy_vs_current.md`
- Código actual de alineación: `alignment.py`
- Código LEGACY de alineación: `LEGACY/AUDIOBOOKSV2/alignment.py`

## Checklist de implementación

### A. Corrección inmediata de alineación y JSON vacío

- [x] A1. Ajustar `_bounds_from_anchors` para que nunca produzca `hs >= len(asr_tokens)`; convertir esos spans en filas “solo referencia” con texto visible, WER=100 y `tc` heredado.
- [x] A2. Garantizar que la última fila(s) de referencia sin ASR se materialicen como “solo referencia” (no como filas vacías) en el JSON y en la tabla `paragraphs` de SQLite.
- [x] A3. Añadir logs de depuración controlados (`set_debug_logger`) que muestren para cada sesión: número de párrafos, anclas válidas, bounds inicial/final y conteo de filas vacías.
- [x] A4. Regenerar la alineación para `ejemplos/capitulo 4 parte 1` y verificar manualmente: no hay filas vacías, los índices ASR están en rango y el JSON contiene texto en todas las filas.

### B. Reintroducir motor LEGACY como base sentwise

- [ ] B1. Extraer desde `LEGACY/AUDIOBOOKSV2/alignment.py` las funciones clave: `_sentence_spans`, `_align_sentwise`, `_split_runs_by_pause`, `_choose_best_run` y `_rebalance_rows` (adaptadas al nuevo módulo).
- [ ] B2. Implementar en el módulo actual una función interna `_build_rows_sentwise(ref: str, asr_words: list[str], tcs: list[float]) -> list[list]` que:
  - [ ] B2.1. Use `_sentence_spans` sobre `ref_tokens` para segmentar por oraciones.
  - [ ] B2.2. Use `_align_sentwise` para obtener pares `(i_ref, j_asr)` con ventanas locales guiadas por `tcs`.
  - [ ] B2.3. Construya `rows_meta` con filas normales, “solo ASR” y “solo referencia” como en LEGACY.
  - [ ] B2.4. Pase `rows_meta` por `_rebalance_rows` y calcule WER/flags/tc en el formato esperado.
- [ ] B3. Hacer que `build_rows_from_words` llame primero a `_build_rows_sentwise` como motor principal, dejando el pipeline de anclas n‑gram y los snapshots SQLite como capa adicional de diagnóstico (no como única fuente de `(hs,he)`).
- [ ] B4. Asegurar que el nuevo `build_rows_from_words` mantenga la interfaz y formato de salida esperado por `qc_app.py` y los tests existentes.

### C. Preproceso interactivo de recorte de guion (pedidos extra 9.1)

- [ ] C1. Diseñar la ventana popup de “Selección de rango de guion” en `qc_app`:
  - [ ] C1.1. Permitir cargar/mostrar el texto original completo (capítulo/libro) y el ASR parcial asociado.
  - [ ] C1.2. Mostrar el texto original recortado en párrafos (via `prepare_paragraphs` o similar).
  - [ ] C1.3. Incluir controles para elegir el rango (inicio/fin) sobre la lista de párrafos (sliders, lista con selección, etc.).
- [ ] C2. Implementar una heurística inicial de detección automática de inicio/fin:
  - [ ] C2.1. Buscar la región de mayor coincidencia entre el ASR y el original (n‑gramas, palabras clave, títulos de capítulo).
  - [ ] C2.2. Proponer un rango de párrafos candidato y preseleccionarlo en la UI.
- [ ] C3. Al confirmar, guardar un TXT recortado:
  - [ ] C3.1. Con el mismo nombre base que el original pero con sufijo elegido por el usuario (e.g. `_parte1`).
  - [ ] C3.2. Actualizar la referencia en `qc_app` para que la alineación se haga sobre ese TXT recortado.

### D. Mapeo palabra‑a‑palabra y modelo de datos (pedidos extra 9.2)

- [ ] D1. Diseñar un esquema extendido para SQLite que incluya:
  - [ ] D1.1. Tabla `ref_tokens` con índice de palabra, texto, id de párrafo/oración y enlaces prev/next.
  - [ ] D1.2. Tabla `asr_tokens` con índice, texto crudo/normalizado y `tc_start`/`tc_end`.
  - [ ] D1.3. Tabla `alignments_word(ref_idx, asr_idx, tipo, distancia_local, ...)`.
- [ ] D2. Implementar un primer llenado de `alignments_word` a partir del alineador sentwise (LEGACY):
  - [ ] D2.1. Inferir para cada palabra si es match/inserción/omisión/sustitución según el camino de alineación.
  - [ ] D2.2. Guardar este mapeo al mismo tiempo que se generan las filas de párrafo.
- [ ] D3. Ajustar `alignment_debugger` para poder inspeccionar también el nivel palabra‑a‑palabra (consultas rápidas sobre `alignments_word`).

### E. Vista de maquetación basada en mapeo fino

- [ ] E1. Diseñar una vista en la UI de QC que construya las filas (párrafos/oraciones) **a partir del mapeo palabra‑a‑palabra**, no solo de la segmentación original.
  - [ ] E1.1. Usar la maquetación (p.ej. párrafos originales) como agrupador visual, pero mantener, detrás, la relación palabra↔tiempo.
  - [ ] E1.2. Permitir que, al seleccionar una palabra en el Original, se pueda resaltar su contraparte en ASR y saltar al tiempo `tc_start` asociado.
- [ ] E2. Verificar que las operaciones de edición en la UI (fusionar filas, corregir ASR, etc.) actualicen coherentemente el mapeo y los tiempos en la DB.

### F. Validación y regresiones

- [ ] F1. Añadir tests específicos que comparen la alineación nueva vs la LEGACY para casos controlados (incluyendo el ejemplo de `capitulo 4 parte 1`).
- [ ] F2. Medir WER global y distribución de flags antes/después para asegurar que la nueva implementación no degrada casos donde el alineador actual funcionaba bien.
- [ ] F3. Documentar en `QC_DEVLOG.md` los cambios de comportamiento, cómo usar la ventana de recorte y el impacto esperado en flujos existentes.
