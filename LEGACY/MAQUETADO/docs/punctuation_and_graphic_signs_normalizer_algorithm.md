**Algoritmo De Normalización De Puntuación Y Signos Gráficos**
- Versión: 0.1 (borrador de implementación)
- Especificación base: `prompts/punctuation_and_graphic_signs_normalizer.json`
- Ámbito del repositorio: `txt2md_mvp` (integración en `pipeline.py` y módulos auxiliares)

**Objetivo**
- Lograr coherencia editorial de signos de puntuación y signos gráficos a nivel de libro, capítulo y bloque, respetando normas por idioma (ES, EN_US, EN_UK) y género (narrativa, ensayo, técnico/académico).
- Producir un texto normalizado y un registro auditable de cambios atómicos (`changes`) para revisión editorial.
- Operar sobre bloques (no línea a línea) para evitar falsos positivos en diálogos, listas, citas y encabezados.

**Alcance**
- Sí: comillas, diálogos (raya ES / comillas EN), rayas y semirrayas, guion vs dash, elipsis, signos invertidos ES (¿ ¡), rangos numéricos/fechas, espaciado tipográfico (incl. NBSP), separadores de escena, normalización Unicode.
- No: reescritura de contenido, orden de oraciones, estilo autoral, semántica no tipográfica (salvo lo mínimo necesario para etiquetar bloques).
- Mantener intactos: código, tablas y bloques “raw” (preformateados) salvo normalización segura de espacios/Unicode donde no afecte render.

**Resultados Esperados**
- `normalized_text`: string final coherente con la política elegida (por idioma + género + preferencias).
- `changes`: lista de modificaciones atómicas con `rule_id`, `lang`, `genre`, `description`, `before/after`, `start_idx/end_idx`, `notes`, `severity`.
- `stats`: métricas de recuento por tipo de cambio.

**Principios Editoriales Por Idioma**
- ES narrativa:
  - Diálogo con raya (—) al inicio de parlamento; incisos con raya parentética espaciada.
  - Jerarquía de comillas: « » > “ ” > ‘ ’.
  - Rangos con en dash (–) sin espacios: 1999–2005.
  - Elipsis como carácter único (…); no duplicar punto.
  - Signos invertidos (¿ ¡) cuando corresponda.
- EN_US narrativa/ensayo:
  - Diálogo con comillas dobles primarias “ ”; period/comma dentro de comillas cuando aplica.
  - Incisos con em dash (—) sin espacios; rangos con en dash (–) sin espacios.
  - Elipsis (…); estilo US en colocación de puntuación.
- EN_UK narrativa/ensayo:
  - Diálogo con comillas simples primarias ‘ ’; dobles “ ” para anidación.
  - Incisos con en dash (–) con espacios permitidos; rangos con en dash (–) sin espacios.
  - Puntuación preferentemente fuera si no pertenece al citado.

**Arquitectura Del Módulo**
- Entrada:
  - `text` (UTF‑8), `settings` con: `language (ES|EN_US|EN_UK)`, `genre (narrativa|ensayo|tecnico_academico)`, `quote_preference`, `dialogue_policy`, `dash_policy`, `range_dash_policy`, `ellipsis_policy`, `nbspace_policy`, `decimal_grouping_policy`.
- Salida:
  - `NormalizationResult` con `normalized_text`, `changes`, `stats` (ver `output_schema` en `prompts/punctuation_and_graphic_signs_normalizer.json`).
- Componentes:
  - `UnicodeNormalizer`: corrige mojibake CP1252/UTF‑8, homogeneiza comillas/guiones/elipsis a su punto de partida tipográfico (sin aplicar aún políticas por idioma).
  - `BlockPreprocessor`: segmenta en bloques estructurales: `Paragraph`, `Heading`, `Dialogue`, `QuoteBlock`, `List`, `Table`, `CodeFence`, `SceneBreak`, `HR`, `FrontMatter`, `Epigraph`.
  - `LanguageAndGenreDetector`: confirma/ajusta `language/genre` por bloque/capítulo si es necesario; permite override manual.
  - `InlineNormalizer`: aplica políticas por idioma/género al contenido inline de cada bloque (comillas, dashes, elipsis, signos invertidos, rangos, espaciado, NBSP).
  - `DialogueTransformer`: tratamiento especializado de parlamentos (ES ↔ EN cuando se solicite, incisos, puntuación de cierre).
  - `SpacingAndNBSPFixer`: reglas de espaciado seguro (antes/después de signos, NBSP contextuales).
  - `ChangeLogger`: calcula offsets (UTF‑16 code units) y guarda cambios atómicos por regla.
  - `CoherenceValidator`: chequeos globales por capítulo/libro (un solo sistema de comillas, densidad de encabezados, consistencia de dash, etc.).

**Data Model**
- `Block` (dataclass): `type`, `text`, `lines_from/to`, `meta` (e.g., speaker continuity, list type, quote depth, language override), `children` para anidación limitada (listas, citas).
- `InlineSpan` (opcional): para cambios localizados con contexto antes/después.
- `NormalizationResult`: como en `output_schema`.

**Preprocesador De Bloques (crítico)**
- Párrafo: agrega líneas envueltas por salto blando; conserva doble salto como separador.
- Lista: detecta bullets (`-`, `*`, `•`, dígitos con `.`/`)`); no confundir con guiones de diálogo.
- Citas en bloque: `>` de Markdown, sangrías persistentes, comillas de apertura multiline.
- Código/tabla: cercado triple ``` o bloque monoespaciado/tabulado; marcar como `raw` para evitar cambios destructivos.
- Diálogo ES: líneas que comienzan con `—` o `–` seguidas de espacio (NBSP preferente) + letra; continuidad de parlamento cuando el mismo personaje sigue en párrafo siguiente (heurística por incisos o ausencia de cierre).
- Diálogo EN: parlamentos entre comillas al inicio o tras guion largo; distinguir de citas no dialogadas por verbos de habla cercanos.
- Separadores de escena: `***`, `* * *`, `---`, `###` (mapear a `---`).
- Front matter / portada: agrupar título, autor, colaboradores y notas preliminares antes de primer índice real; evitar que se interprete como cascada de headings.

**Normalización Unicode (paso 0)**
- Convertir `"`/`'` rectas en comillas tipográficas provisionales según balanceo local (no aplicar todavía política definitiva).
- Unificar guiones: `--`/`—`→ em dash (—), `–`→ en dash (–), `-`→ guion (palabras compuestas) salvo patrón de diálogo o rango.
- Elipsis `...`→ `…`; evitar `….`.
- Corregir mojibake CP1252 (comillas/dashes) detectables por bytes típicos; usar mapeo robusto.
- Normalizar espacios: colapsar múltiples espacios salvo en bloques `raw`; preservar tabs en tablas/código.

**Comillas**
- Detección y balanceo por bloque con autómata de estados (anidación hasta 3 niveles). Tokens: `open_primary`, `close_primary`, `open_secondary`, `close_secondary`.
- Asignación por idioma:
  - ES: « » primario, “ ” secundario, ‘ ’ terciario.
  - EN_US: “ ” primario, ‘ ’ secundario.
  - EN_UK: ‘ ’ primario, “ ” secundario.
- Reglas:
  - No alternar estilos en el mismo nivel. Si se detectan estilos mezclados, normalizar al preferido y registrar `severity: fix`.
  - Puntuación final: ES fuera si no pertenece; EN_US period/comma dentro; EN_UK fuera salvo pertenencia.
  - Apostrofes: usar ’ (U+2019) en contracciones/posesivos; no confundir con comillas.

**Diálogo**
- ES (raya):
  - Cada parlamento en línea nueva: `—[NBSP]Texto…`.
  - Incisos: `— Texto —dijo— y siguió.` con espacios normales alrededor de rayas parentéticas, no NBSP.
  - Fin de parlamento tras inciso: `— Vámonos —dijo.` (sin raya de cierre).
  - Reemplazo seguro de `-` por `—` cuando patrón de diálogo: línea comienza con `-\s+\p{L}` y no es bullet/lista ni HR.
- EN (comillas):
  - Identificar parlamentos entre comillas; mover period/comma dentro (EN_US) o fuera (EN_UK) según política.
  - Transformaciones cruzadas ES↔EN disponibles bajo `cross_language_transform` y solo si son solicitadas explícitamente.

**Dashes, Rangos Y Guion**
- Incisos:
  - ES: em dash (—) con espacios: `palabra — inciso — palabra`.
  - EN_US: em dash (—) sin espacios: `word—aside—word`.
  - EN_UK: en dash (–) con espacios permitido: `word – aside – word`.
- Rangos: en dash (–) sin espacios para números/fechas: `1999–2005`.
- Oposición/relación (ES opcional): `Madrid–Barcelona` sin espacios si el proyecto lo define.
- Nunca usar `-` ASCII para incisos o rangos; `-` reservado a palabras compuestas.

**Elipsis**
- Reemplazar `...` por `…`.
- Si `…` cierra oración, no añadir punto extra.
- Espaciado según idioma: ES normalmente sin espacio antes; EN admite espacio fino en algunas guías (configurable).

**Signos Invertidos (ES)**
- Añadir `¿`/`¡` de apertura cuando la oración concluya con `?`/`!` y no exista apertura correspondiente dentro del mismo bloque.
- Heurística de seguridad: evitar añadir si la pregunta/exclamación es una “cola” corta dentro de una oración mayor. Marcar como `suggestion` en ambigüedad.

**Espaciado Y NBSP**
- Sin espacio antes de `, . ; : ? !`.
- Un espacio después de signos finales.
- NBSP:
  - ES diálogo tras `—`: `—[NBSP]Texto…`.
  - Entre número y `%`, `°`, `§`, `º`, `ª`.
  - Entre abreviaturas de una letra y número: `Tº[NBSP]3` (configurable).
  - Opcional: evitar huérfanas típicas (p. ej., preposiciones monosilábicas), sólo si el proyecto lo exige.

**Separadores De Escena**
- Detectar `***`, `* * *`, `---`, `###` aislados y normalizar a `---` (regla única para Markdown HR).

**Tablas Y Listados Tabulares**
- Detectar bloques con alineación por tabuladores o múltiples espacios en columnas; preservar como bloque preformateado o tabla Markdown, sin promover falsos encabezados.
- No alterar puntuación interna salvo normalización Unicode y espacios seguros.

**Validación De Coherencia Global**
- Un solo sistema de comillas por nivel en todo el libro.
- Un solo sistema de diálogos (raya ES o comillas EN) salvo bloques etiquetados como cita/transcripción.
- Dashes consistentes: incisos vs rangos diferenciados; ausencia de `-` como sustituto de `–`/`—`.
- Elipsis coherentes; signos invertidos presentes/ausentes según idioma.
- Reporte de incoherencias residuales: porcentaje y ejemplos por capítulo.

**Interfaz Propuesta (Python)**
- `txt2md_mvp/punctuation_normalizer.py`
  - `class NormalizerSettings(TypedDict)`: claves de `inputs_required` del JSON.
  - `@dataclass class Change`: `rule_id, lang, genre, description, before, after, start_idx, end_idx, severity, notes`.
  - `@dataclass class NormalizationResult`: `normalized_text: str, changes: List[Change], stats: Dict[str, int]`.
  - `def normalize_punctuation(text: str, settings: NormalizerSettings) -> NormalizationResult`.
  - `def detect_language_and_genre(text: str) -> Tuple[str, str]` (heurístico, con override por settings).
  - `def preprocess_blocks(text: str, settings: NormalizerSettings) -> List[Block]`.
  - `def apply_policies(blocks: List[Block], settings: NormalizerSettings) -> NormalizationResult`.

**Algoritmo (pasos concretos)**
- Paso 0: Normalización Unicode segura (comillas rectas→tipográficas provisionales, `...`→`…`, guiones múltiples→—/– según contexto potencial, limpieza CP1252).
- Paso 1: Preprocesado de bloques (párrafos, listas, citas, código, tablas, separadores, portada/preliminares, diálogos ES/EN).
- Paso 2: Detección/confirmación de `language` y `genre` (por bloque si es necesario) usando señales: signos invertidos, densidad de comillas rectas vs tipográficas, verbos de habla frecuentes (ES: “dijo”, “preguntó”; EN: “said”, “asked”).
- Paso 3: Comillas: balanceo, selección por idioma, reubicación de period/comma en EN_US, normalización de anidación.
- Paso 4: Diálogos: ES — … (incisos, NBSP); EN “…” (puntuación por variante). Opcional: transformaciones ES↔EN bajo bandera explícita.
- Paso 5: Dashes y rangos: incisos (—/– con/ sin espacio según política), rangos numéricos con en dash sin espacios; evitar `-` salvo compuestos.
- Paso 6: Elipsis: reemplazo `…`, colisiones con punto/comillas/paréntesis.
- Paso 7: Signos invertidos (ES): añadir/retirar según idioma; resolver ambigüedades con `suggestion`.
- Paso 8: Espaciado y NBSP: reglas finas por idioma; unidades y símbolos.
- Paso 9: Ensamblado: recomponer texto desde bloques, recalcular offsets, registrar `changes` y `stats`.
- Paso 10: Validación de coherencia y `report.md`: densidad de cambios, incoherencias abiertas, recomendaciones.

**Heurísticas Críticas Y Desambiguación**
- Diálogo vs lista: si `^-\s+\p{L}` al inicio y la línea anterior es vacía y hay múltiples ítems consecutivos con la misma pauta, es lista; si hay verbos de habla cercanos o mayúscula tras guion y ausencia de otros bullets, es diálogo.
- Comillas en títulos/epígrafes: si el bloque es `Heading` o `Epigraph`, aplicar política menos agresiva (no mover puntuación fuera si es parte del título original).
- Rangos con espacios (`1999 - 2005`): si ambos lados son números/fechas, normalizar a `1999–2005`; si palabras, evaluar si es oposición (`Madrid - Barcelona`) y aplicar política del proyecto o dejar como `-` y registrar `suggestion`.
- Elipsis consecutivas: colapsar `……`→`…`; si `…` seguido de `.` o `,`, eliminar duplicado según política.

**Registro De Cambios (Offsets UTF‑16)**
- Mantener un `Rope` o reconstrucción incremental por bloques para calcular `start_idx/end_idx` en términos de code units UTF‑16 (compatibilidad con editores/Windows). Ofrecer helper `to_utf16_offsets(text, start_char_idx, end_char_idx)`.

**Modos De Ejecución**
- `scan-only`: no modifica, solo devuelve `changes` sugeridos con `severity: suggestion`.
- `fix-safe`: aplica cambios de baja ambigüedad (Unicode, espaciado, rangos claros, elipsis, dash obvio, comillas balanceadas).
- `fix-all`: aplica todo lo posible y marca ambigüedades resueltas con `notes`.

**Integración En El Pipeline**
- Punto de inserción: tras normalización de estructura y antes de clasificación de encabezados (para no promover falsos `h3`). Archivo: `txt2md_mvp/pipeline.py`.
- Respetar perfiles de formato (`txt2md_mvp/format_rules/*.json`) y plantillas YAML si hay conflictos; la normalización de signos es previa y más básica.
- Usar `ai_supervisor.py` solo para QA opcional; nunca para aplicar cambios (reglas deben ser determinísticas). Si se usa IA, cumplir: modelo por defecto `gpt-5-mini`; `gpt-5` opcional; nunca `gpt-4`.

**Pruebas**
- Unitarias: `tests/test_punctuation_normalizer.py` con casos positivos/negativos por idioma y género.
- End-to-end: ejecutar sobre `demo/ejemplo.txt` y muestras reales; verificar `report.md` incluye métricas de coherencia y advertencias.
- Casos límite (extraídos de `edge_cases` del JSON):
  - Diálogo interrumpido por inciso y continuación.
  - Parlamentos que continúan en párrafo siguiente.
  - Citas dentro de diálogos y viceversa.
  - Elipsis pegada a comillas o paréntesis de cierre.
  - Guion ASCII confundido con dash.
  - Mezcla de listas y líneas con guion inicial.
  - Titulares editoriales en ALL CAPS que no son capítulos.

**Plan De Implementación (iterable)**
- 1. Crear `txt2md_mvp/punctuation_normalizer.py` con `normalize_punctuation(...)` y tipos `Change/NormalizationResult`.
- 2. Implementar `UnicodeNormalizer` y helpers de offsets UTF‑16.
- 3. Implementar `BlockPreprocessor` (párrafos, listas, citas, código, tablas, separadores, portada, diálogos). Añadir detectores: escena y portada.
- 4. Implementar `LanguageAndGenreDetector` mínimo viable (heurístico), con override por settings.
- 5. Implementar `InlineNormalizer` por idioma/género: comillas, dashes, elipsis, signos invertidos, rangos, espaciado/NBSP.
- 6. Implementar `DialogueTransformer` (ES y EN) y transformaciones cruzadas opcionales en función del JSON.
- 7. Instrumentar `ChangeLogger` y `stats`.
- 8. Integrar en `pipeline.py` con modo `scan-only` y `fix-safe` al inicio.
- 9. Añadir pruebas unitarias y fixtures; ampliar `demo/ejemplo.txt` con casos mixtos.
- 10. Documentar en `report.md` las métricas de coherencia y advertencias.

**Riesgos Y Fallbacks**
- Ambigüedad en comillas no balanceadas o apóstrofos: preferir `scan-only` o `suggestion`.
- Diálogo vs lista con guion inicial: requerir dos señales concurrentes (verbo de habla, puntuación/estructura) antes de convertir.
- Tablas/código: aplicar lista de exclusiones estricta para no “tipografiar” contenido literal.

**Extensiones Futuras (Etapas 2 y 3)**
- Máquina de estados global (portada → capítulos → cuerpo → anexos) para validar transiciones tipográficas y reforzar coherencia.
- Scoring heurístico por bloque (pesos positivos/negativos) antes de asignar tipo o aplicar cambios invasivos.
- Dataset etiquetado y clasificador tradicional (Naive Bayes / SVM) para dirimir ambigüedades; modelos compactos (DistilBERT) si fuera necesario.
- Módulo específico de portada y paratextos.

**Referencias Internas**
- Especificación completa: `prompts/punctuation_and_graphic_signs_normalizer.json`.
- Laboratorio: `modules/punctuation_lab/` (ampliar con `regex_playground.py` y fixtures de casos reales).
- Integración: `txt2md_mvp/pipeline.py`, `txt2md_mvp/formatting.py`, `txt2md_mvp/rules.py`.

**Checklist Para “Hecho” (Definición De Listo)**
- [ ] Ejecuta `scan-only` sin falsos positivos graves en docs de prueba.
- [ ] `fix-safe` no rompe bloques ni promueve falsos encabezados/listas.
- [ ] Coherencia global estable por libro en comillas, diálogos y dashes.
- [ ] Reporta métricas claras y enlaces a ejemplos.
- [ ] Tests unitarios cubren 80% de rutas críticas; casos límite documentados.

**Plan ES Detallado (Prioridad Diálogo)**
- Objetivo: corregir, uniformar y auditar signos en español según `prompts/punctuation_and_graphic_signs_normalizer.json`, priorizando diálogos.
- Entradas: texto + `settings` (idioma ES, género, políticas). Modos: scan-only, fix-safe, fix-all.

**Detección Y Conversión De Diálogo (ES)**
- Señales de diálogo (score):
  - Positivas: comienzo de bloque con comillas (« “ '), primera comilla de cierre cercana, verbos de habla (dijo, preguntó, respondió, exclamó, murmuró, susurró, gritó, añadió, replicó, contestó, llamó, ordenó, señaló, explicó, advirtió, observó, indicó, comentó, insistió, rió, sonrió…), signos enfáticos (¿ ¡ … ! ?), vocativos.
  - Negativas: títulos/encabezados, epígrafes/citas largas sin verbos de habla, referencias bibliográficas.
- Umbrales:
  - Auto (ES, diálogo con raya): convertir cuando score ≥ t_conv (p.ej., 0.5 en fix-safe; 0.35 en fix-all) y no existan señales negativas fuertes.
  - Conservador (opción): exigir verbo de habla explícito antes de convertir.
- Transformaciones (bloque por bloque):
  - “Texto.” → — NBSP Texto.
  - “Texto,” dijo X. → — NBSP Texto —dijo X.
  - “Texto,” dijo X, “continuación.” → — NBSP Texto —dijo X— continuación.
  - Preservar signos (¿ ¡ … ! ?), NBSP tras raya inicial, rayas de inciso sin NBSP (espacio normal alrededor si corresponde).
  - No separar morfemas: no espaciar rayas dentro de palabras; no convertir “Ran--dee” salvo política explícita.
  - Continuidad de parlamento: si hay comillas múltiples en el mismo bloque, unir como un parlamento salvo verbos de habla que separen incisos.

**Comillas (ES)**
- Política por proyecto: primario « » (angular) o “ ” (dobles) si así se selecciona en `settings.quote_preference`.
- Anidación: « » > “ ” > ‘ ’. Balanceo y reemplazo coherente de estilos mezclados.
- Puntuación final fuera si no pertenece a la cita (respetar excepciones de estilo si se configuran).
- Apóstrofo tipográfico: ’ en contracciones/posesivos; no confundir con comillas.

**Rayas, Incisos Y Rangos**
- Incisos: — con espacios en ES (“palabra — inciso — palabra”).
- Diálogo: — NBSP al inicio de parlamento; incisos “—dijo—”.
- Rangos: en dash – sin espacios (“1999–2005”).
- Relación/oposición: “Madrid–Barcelona” sin espacios si lo define el proyecto.
- Reglas de seguridad: no convertir “--” a raya cuando esté entre letras; en otros contextos, sí.

**Elipsis**
- Usar … (U+2026). No duplicar punto final.
- Colisiones: …” ) ] → resolver sin añadir puntos dobles.

**Signos Invertidos (¿ ¡)**
- Añadir ¿/¡ de apertura cuando exista ?/! de cierre en la misma oración y no haya apertura correspondiente.
- Evitar en colas cortas (ambigüedad) → marcar como suggestion en scan-only.

**Espaciado Y NBSP**
- Sin espacio antes de , . ; : ? !
- Un espacio después de signos finales (no antes de comillas de cierre configuradas).
- NBSP:
  - Tras raya de diálogo: — NBSP Texto…
  - Entre número y %/°/º/ª/§ cuando sea necesario.
  - Opcional: evitar huérfanas (si el proyecto lo exige) con NBSP selectivo.

**Separadores De Escena / HR**
- Detectar *** / * * * / --- / ### aislados → normalizar a HR Markdown: ---.

**Listas Y Tablas**
- Listas: no confundir con diálogos (si “- ” al inicio + múltiples ítems consecutivos → lista; presencia de verbos de habla + mayúscula inicial favorece diálogo).
- Tablas/listados tabulares: conservar como preformateado o tabla Markdown; no promover falsos encabezados.

**Bloques Especiales**
- Portada y preliminares: no promover encabezados; agrupar autor y colaboradores; no aplicar conversión de comillas a raya.
- Citas en bloque (blockquote): respetar puntuación como original salvo normalización Unicode y espaciado seguro.

**Estrategia De Validación (ES)**
- Métricas por capítulo/libro:
  - Densidad de parlamentos válidos (raya + NBSP / por 1000 palabras).
  - Estabilidad del sistema de comillas por nivel.
  - Incisos válidos (— … —) vs. rayas pegadas a morfemas.
  - Rangos normalizados vs. guiones ASCII.
  - Elipsis coherentes (… sin duplicaciones).
  - Invertidos añadidos/eliminados (ratio por 100 oraciones).
  - Alertas: comillas mezcladas, dos h2 consecutivos, diálogos con comillas restantes.
- Reporte `.report.md`: incluir resumen, ejemplos (3–5) y enlaces a líneas.

**Pruebas (ES)**
- Unitarias (tests/test_punctuation_normalizer.py):
  - Conversión “comillas→raya” (A/B/C) con signos, incisos y NBSP.
  - Falsos positivos: títulos con comillas, epígrafes, citas académicas.
  - “Ran--dee” no convertido/espaciado; “1999-2005”→“1999–2005”.
  - Invertidos ¿/¡ en oraciones interrogativas/exclamativas.
  - Elipsis y colisiones con comillas/paréntesis.
- Integración (tests/test_pipeline.py):
  - Activar `use_punctuation_module=True`, validar documento de demo con parlamentos en raya y métricas en `meta.punctuation_stats`.
  - Modo scan-only: retorna cambios sugeridos sin modificar texto.

**Configuración Y Overrides**
- YAML/JSON de proyecto (opcional):
  - Lista de verbos de habla (ampliable por género/domino).
  - Política de comillas preferida (angular/dobles) y anidación.
  - Tolerancias de score/umbrales para conversión.
  - Reglas para símbolos/medidas (NBSP, %).

**Riesgos Y Mitigaciones**
- Diálogo sin verbo de habla: usar score + patrón, empezar en fix-safe sólo si hay alta confianza.
- Mezcla de estilos (algunas líneas con comillas, otras con raya): normalizar hacia política del libro y registrar exceptions en notes.
- Tablas/código: exclusiones estrictas.
- Portada/preliminares: “no tocar” salvo Unicode/espaciado seguro.

**Roadmap De Implementación (ES, etapa 1)**
- [1] Detector de diálogo (score + reglas) y transformador quoted→raya.
- [2] Normalizador de comillas (balanceo y preferencia), sin tocar citas detectadas como no diálogo.
- [3] Dashes & incisos: separación segura (no en morfemas) y rangos en –.
- [4] Elipsis + colisiones; invertidos ¿/¡.
- [5] Espaciado + NBSP; separadores de escena.
- [6] Validación global + métricas en `.report.md`.
- [7] Modos (scan-only, fix-safe, fix-all) con umbrales configurables.
- [8] Suite de pruebas + fixtures con casos reales.
