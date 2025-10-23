# RFC 0001 — Transcripción sesgada multietapa (propuesto)

Estado: propuesto
Autor: Codex
Fecha: 2025‑09‑04

## Objetivo
Reducir errores de ASR debidos a palabras clave (nombres propios, lugares, tecnicismos, extranjerismos) y homofonías, sesgando la transcripción con información del guion y/o listas de palabras, y refinando por etapas sin romper el flujo actual.

## Contexto
- Fallos típicos: interpretaciones fonéticas erróneas de términos clave.
- Tenemos:
  - `faster-whisper` con `hotwords` e `initial_prompt`.
  - Alineadores (`alignment.py`, `audio_video_editor.py`) y re‑sync por CSV (`utils/resync_python_v2.py`).
  - GUI `qc_app.py` y módulo `transcriber.py` con soporte inicial de hotwords (lista plana).

## Alcance
- Añadir un pipeline “multietapa guiada” opcional que conserve los artefactos actuales (.txt, .words.csv/.json, .qc.json) y sume resultados refinados (.refined.*).
- No elimina ni cambia el flujo simple vigente; es una opción adicional.

## Diseño propuesto (resumen)
1) Bootstrap rápido
- Transcribir con modelo ligero (tiny/base) para obtener texto y/o palabras+tiempos.
- Extraer lista de palabras clave desde guion (`extract_word_list`) y usarlas como `hotwords`.

2) Alineación aproximada
- Alinear bootstrap vs guion para obtener segmentos aproximados del audio asociados a fragmentos de texto (con margen, p.ej., ±3–5 s y/o palabras). Opciones:
  - Si hay JSON/CSV word‑level: usar tiempos reales para segmentar.
  - Si solo hay texto: usar `build_rows` (tc acumulado) como aproximación.

3) Transcripción pesada guiada por fragmentos
- Para cada segmento: transcribir con modelo pesado (large‑v3/large) pasando:
  - `initial_prompt` con el fragmento de guion relevante (recortado a límite de tokens).
  - `hotwords` generados del fragmento (palabras clave, nombres propios, glosario).
  - `prefix` con cola del segmento anterior (coherencia/contexto).
- Guardar `.refined.words.json` y `.refined.txt` (y opcional `.refined.qc.json`).

4) Post‑corrección opcional (IA)
- Con guion disponible: pasar (por partes) ASR refinado a `ai_review`/LLM para corrección leve, manteniendo orden y límites temporales.
- Aplicar umbrales WER por fila para decidir cuándo corregir o reintentar.

5) Ensamblado y verificación
- Unir segmentos, recalcular WER por fila y marcar sospechosos.
- Exportar EDL y QC JSON refinado.

## API/CLI propuestas
- Python (transcriber.py):
  - `guided_transcribe(audio_path, script_path, *, fast_model="base", heavy_model="large-v3", chunk_margin=3.0, use_ai_post=False) -> dict`
  - Devuelve rutas de artefactos generados (`txt`, `words.json`, `qc.json` refinados).
- CLI:
  - `python -m transcriber guided --input a.wav --script gui.pdf --fast base --heavy large-v3 --margin 3.0 --ai-post`

Parámetros relevantes para faster‑whisper por segmento:
- `initial_prompt`: fragmento de texto esperado (capado a N tokens).
- `hotwords`: lista deduplicada y filtrada (solo caracteres válidos del idioma); ponderación básica por repetición.
- `prefix`: últimas ~10–20 tokens del segmento previo.
- Otros: `beam_size`, `temperature`, `compression_ratio_threshold`, `log_prob_threshold`.

## Fases de implementación
1. Infraestructura de segmentación
- Bootstrap + alineación aproximada → generar intervalos por fragmento (usar word‑level cuando esté; si no, acumulado).

2. Guided heavy pass (núcleo)
- Transcripción por fragmento con `initial_prompt`+`hotwords`+`prefix`.
- Ensamblado y artefactos `.refined.*`.

3. Corrección IA opcional
- Integrar `ai_review` por fragmento con prompt restrictivo (“no alteres el orden, conserva puntuación base”).

4. Integración GUI/CLI
- Nuevo subcomando `guided` y botón “Transcripción refinada” en `qc_app` (opcional en esta fase).

5. Calidad y regresión
- Métricas offline sobre WER, tasa de palabras clave acertadas y tiempo de ejecución.
- Reintentos automáticos de segmentos con WER>umbral.

## Métricas de éxito
- WER global y por fila (baja >= x%).
- Recall de palabras clave (sube >= y%).
- Tiempo total aceptable (z× el flujo simple), con posibilidad de interrumpir y retomar.

## Riesgos y mitigaciones
- Límite de tokens del modelo: recortar fragmentos y priorizar palabras clave.
- Segmentación imprecisa: usar margen configurable y VAD opcional.
- Sesgo excesivo: fallback sin `hotwords` si la confianza cae.

## Aceptación
- Pipeline corre end‑to‑end con artefactos `.refined.*` y mejora medible en palabras clave vs baseline simple.
- No rompe comandos/funciones actuales.

## Tareas (orden sugerido)
1. (Infra) Helpers para extraer fragmentos de guion y generar hotwords por segmento.
2. (Infra) Bootstrap + intervalos (con JSON/CSV si existe; si no, acumulado).
3. (Core) Transcripción pesada guiada por segmento y ensamblado.
4. (Opt) Post‑corrección con IA por segmento.
5. (CLI) Añadir `transcriber guided` con flags.
6. (GUI) Botón “Transcripción refinada” y barra de progreso.
7. (QA) Métricas y tests de no‑regresión.

