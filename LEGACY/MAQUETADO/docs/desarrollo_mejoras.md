# Plan Integral de Mejora – Motor txt2md_v2

## Objetivo
Elevar el flujo de trabajo de traduccion y maquetado hasta un nivel listo para publicacion, combinando contexto editorial, control de jerga, fortalecimiento del glosario y normalizacion de puntuacion.

---

## Fase 0 · Preparativos

- [ ] Auditar la base actual y registrar dependencias (versiones, modelos instalados, modulos opcionales).
- [ ] Definir convenciones de sesiones (`status`, `style_profile`, metadatos) y limpiar vestigios tecnicos en el output final.
- [ ] Revisar `tests/` y establecer cobertura minima para las nuevas funciones (UI, glosario, puntuacion).

---

## Fase 1 · Contexto Editorial y Extensiones UI

### 1.1 Panel de Autor y Libro
- [ ] Extraer titulo y autor durante el maquetado (`pipeline.process_file` → sesion).
- [x] Persistir en `session_data["meta"]` y exponerlo en `Txt2MdApp` (panel lateral o encabezado).
- [x] Añadir boton "Editar" que abra un pop-up con campos editables (titulo, subtitulo opcional, autor, coautores).

### 1.2 Ventana de Contexto Editorial
- [x] Diseñar `ContextInfoDialog` con secciones (datos basicos + acciones opcionales).
- [x] Buscar autor/libro en Wikipedia reutilizando idiomas (`es`, `en`) y rellenar campos.
- [x] Generar mini bio con IA (selector de modelo `gpt-5`, `gpt-5-mini`).
- [x] Generar resumen/argumento con IA y reutilizar fuentes de Wikipedia cuando existan.
- [x] Generar contraportada automatica con IA y exportarla como archivo aparte.
- [x] Registrar en la sesion que acciones se ejecutaron y donde se guardo cada artefacto.
- [x] Permitir edicion manual antes de confirmar.
- [x] Checkbox para anexar la mini bio al final del Markdown y aplicar automaticamente.

### 1.3 Reutilizacion de Materiales
- [x] Crear repositorio de perfiles guardados por autor/libro (JSON).
- [x] Integrar selector en el pop-up para cargar texto existente o sobrescribirlo.

---

## Fase 2 · Jerga y Preprocesamiento Linguistico

### 2.1 Deteccion Temprana (gpt-5-nano)
- [ ] Insertar paso opcional previo a la traduccion que envie bloques a `gpt-5-nano`.
- [x] Heuristica local para detectar anglicismos y exclamaciones pulp recurrentes (JSON `{frase, categoria, comentario}`).
- [ ] Integrar llamada real al modelo ligero y consolidar resultados en `jerga_detec.json` por sesion.

### 2.2 Validacion Experta
- [ ] Consumir el JSON anterior con un modelo mayor (configurable).
- [ ] Cada entrada produce `{"frase": "...", "traduccion": "..."}` o `{"frase": "...", "status": "error"}`.
- [ ] Filtrar entradas marcadas como error y nutrir el glosario con las validas.
- [ ] Registrar estadisticas (cuantas confirmadas / descartadas).

### 2.3 Inyeccion Automatica en Glosario
- [x] Inyectar las traducciones validadas de la jerga en `GlossaryBuilder` con `action: translate`.
- [ ] Guardar los cambios en la sesion y reflejarlos en `report.md`.

---

## Fase 3 · Revision y Edicion del Glosario

### 3.1 Nueva Interfaz
- [ ] Crear `GlossaryReviewDialog` (Tkinter) que:
  - Muestre entradas en tabla (termino fuente, traduccion, notas, estado).
  - Permita filtrar por estado (pendiente, revisado, jerga).
  - Ofrezca re-traduccion individual con campo de comentarios.

### 3.2 Re-traduccion con Variantes
- [ ] Al solicitar re-traduccion, permitir elegir "1 opcion" o "3 opciones".
- [ ] Presentar alternativas separadas en celdas; doble clic para fijar la activa (sin borrar las restantes).
- [ ] Mantener historial por entrada (ultima IA, notas del operador).

### 3.3 Exportacion y Consistencia
- [ ] Al cerrar el dialogo, validar que no queden entradas sin traduccion.
- [ ] Generar informe (`glossary_review.md`) con cambios aprobados, descartes y comentarios.
- [ ] Asegurar que las entradas marcadas "no traducir" justifiquen la decision.

---

## Fase 4 · Normalizacion de Puntuacion y Dialogos

### 4.1 Laboratorio de Puntuacion
- [x] Crear carpeta `modules/punctuation_lab/` con:
  - `README.md` (objetivos, reglas aceptadas, TODOs).
  - `experiments.md` (registro de iteraciones y resultados).
  - `regex_scanner.py` (script inicial para localizar dialogos problemáticos).
- [ ] Implementar escaner regex que identifique patrones sospechosos en toda la obra.

### 4.2 Correccion Automatica MVP
- [ ] Prototipo: transformar dialogos `"Texto"` → `— Texto`.
- [ ] Añadir salvaguardas (no tocar comillas en citas o titulos).
- [ ] Reporte con diffs sugeridos antes de aplicar.

### 4.3 Integracion y Tests
- [ ] Integrar el laboratorio como etapa opcional post-traduccion.
- [ ] Crear pruebas unitarias con fragmentos representativos.
- [ ] Monitorear metricas (cuantos reemplazos, cuantos descartados).

---

## Fase 5 · QA, Metricas y Publicacion

- [ ] Ajustar QA para cubrir capitulos completos (configurable).
- [ ] Grabar en la sesion que pasos (contexto, jerga, glosario, puntuacion) fueron aplicados.
- [ ] Extender `report.md` con:
  - Resumen de acciones ejecutadas.
  - Artefactos generados (bio, contraportada, glosario final).
  - Token usage por etapa.
- [ ] Añadir "modo publicacion" que verifique:
  - Metadatos limpios.
  - Glosario consistente.
  - QA aprobada.
  - Sesion firmada como `publish_ready`.

---

## Iteraciones Futuras

- Integrar previsualizacion EPUB/MOBI con banner "listo para publicar".
- Añadir aprendizaje de preferencias por editorial/coleccion.
- Evaluar modulos complementarios (analisis de tono, consistency checker multilingue).

---

## Proximos Pasos Inmediatos

1. Implementar flujo real de deteccion de jerga (modelo ligero → JSON → UI de glosario).
2. Conectar la nueva generacion IA con revision humana (resumen/bio/contraportada) y agregar trazabilidad en `report.md`.
3. Completar el primer experimento del laboratorio de puntuacion (usar `regex_scanner.py` con fixtures y documentar resultados).
