# AGENTS

## Objetivo general
Construir un motor de conversión `.txt → .md` con calidad editorial profesional, capaz de generar metadatos analíticos y reportes accionables mientras preserva la fidelidad narrativa de los textos de origen.

## Diagnóstico actual del MVP
- `titlecase_short` ya filtra diálogos y finales de párrafo, pero sigue elevando primeras frases enfáticas y bullets narrativos; necesitamos operar sobre bloques completos antes de disparar heurísticas.
- `all_caps_short` confunde titulares o rótulos editoriales con capítulos; la jerarquía sigue rompiéndose con “MYSTERIOUS EXPLOSION!”, “THE END”, etc.
- La portada y los elementos preliminares se procesan como una cascada de encabezados en vez de reconocerse como un bloque especial.
- Las tablas y listados tabulares (ej. notas del transcriptor) se degradan en texto plano o encabezados dispersos.
- El análisis sigue siendo mayormente línea a línea; urge un preprocesador de bloques que agrupe párrafos, listas, citas y headings multilínea antes de clasificar.

## Ruta por etapas hacia un nivel profesional
Las etapas son secuenciales. Avanzamos a la siguiente solo cuando la anterior demuestra estabilidad en pruebas manuales y automáticas.

### Etapa 1 · Endurecer el motor basado en reglas (corto plazo)
- [x] Reforzar `titlecase_short` con restricciones negativas (2025-10-03): se añaden exclusiones por comillas, verbos de habla, puntuación interna, proporción de palabras capitalizadas y contexto previo/siguiente.
- [ ] Ajustar `all_caps_short` con pistas positivas: priorizar palabras clave (`CHAPTER`, `CAPÍTULO`, `PART`, `LIBRO`, etc.) y números romanos para distinguir capítulos de titulares narrativos.
- [ ] Detectar separadores de escena (`***`, `* * *`, `---`, `###`) y mapearlos a `---` en Markdown.
- [ ] Añadir reglas para portada y preliminares: agrupar título, autores y colaboradores como bloque especial antes del índice real.
- [ ] Identificar bloques con sangrías o estilo epistolar/poético y convertirlos en `blockquote` u otros formatos apropiados.
- [ ] Ampliar la suite de pruebas (incluido `demo/ejemplo.txt`) con casos negativos y positivos para cada heurística reforzada.
- [x] Fusionar encabezados partidos por guion/retorno en mayúsculas antes de clasificarlos (2025-10-03).
- [ ] Introducir un preprocesador de bloques (párrafos, listas, citas, tablas) antes del motor de reglas para evitar decisiones línea a línea.
- [ ] Revisar listas retóricas y puntos enfáticos: impedir que primeras líneas cortas asciendan a `h3` cuando el bloque arrastra texto continuo.
- [ ] Detectar bloques tabulares básicos y preservarlos como tabla o bloque preformateado para evitar falsos encabezados.

### Etapa 2 · Incorporar contexto y memoria (mediano plazo)
- [ ] Diseñar una máquina de estados que modele portada, capítulos, subtítulos, cuerpo narrativo y anexos.
- [ ] Utilizar el estado anterior y siguiente para validar transiciones: evitar dos `h2` consecutivos sin contenido intermedio, forzar párrafos tras un encabezado, etc.
- [ ] Analizar patrones globales por documento (frecuencia de `CHAPTER`, `CAPÍTULO`, numeración romana) para recalibrar heurísticas durante la ejecución.
- [x] Introducir señales de diálogos (comillas emparejadas, verbos de habla) como factor de descarte para encabezados (2025-10-02).
- [ ] Registrar métricas de consistencia (ej. densidad de encabezados válidos) y reflejarlas en `.report.md` para validar la etapa.
- [ ] Definir scoring heurístico por bloque (pesos positivos/negativos) para combinar múltiples señales antes de asignar un tipo.
- [ ] Modelar dependencias entre bloques (p.ej., headings consecutivos, listas, citas) una vez que el preprocesador esté disponible.

### Etapa 3 · Evolucionar hacia modelos de aprendizaje (largo plazo)
- [ ] Preparar un dataset etiquetado con categorías (`titulo_principal`, `titulo_capitulo`, `subtitulo`, `parrafo_dialogo`, `separador`, etc.).
- [ ] Entrenar un clasificador tradicional (Naive Bayes / SVM) con características textuales y posicionales para validar el enfoque.
- [ ] Evaluar modelos de lenguaje compactos (p.ej., DistilBERT) si la precisión del clasificador clásico no alcanza los objetivos.
- [ ] Crear un módulo específico para la portada y los paratextos, ya sea con reglas dedicadas o con un submodelo entrenado sobre esa sección.
- [ ] Integrar los modelos en la canalización existente, con métricas de confianza y fallback a reglas cuando la predicción sea incierta.

## Uso de este documento
- Registrar avances marcando cada casilla al completar una tarea y documentar decisiones clave.
- Reubicar o añadir tareas según emerjan nuevos patrones o datasets de prueba.
- Revisar el diagnóstico si aparecen nuevas categorías de error.
- Mantener un historial breve de devoluciones externas con fecha para orientar próximos sprints.

## Observaciones recientes
- 2025-10-03 · Ensayo “Waking World”: jerarquía macro detectada correctamente; persisten falsos `h3` en primeras frases enfáticas y bullets narrativos. Urge preprocesador de bloques para que las heurísticas operen sobre párrafos completos.
- 2025-10-02 · Novela “Skylark”: eliminados falsos encabezados en diálogos y finales de párrafo; titulares editoriales y tablas siguen degradándose.

## Logros del MVP inicial
- [x] Normalizar el front matter para eliminar comillas duplicadas y exponer valores limpios.
- [x] Generar anclas del índice con slugs ASCII estables para motores sensibles a acentos.
- [x] Cargar patrones desde plantillas YAML con priorización dinámica según el tipo de documento.
- [x] Añadir pruebas automatizadas básicas sobre `demo/ejemplo.txt` para validar encabezados, párrafos y metadatos.
- [x] Extender la CLI para aceptar carpetas y comodines, procesando lotes en una sola ejecución.
- [x] Afinar señales por plantilla, permitiendo desactivar heurísticas según el tipo detectado.
- [x] Ampliar `report.md` con densidad de encabezados y advertencias de jerarquía.

## Directivas sobre IA
- **NUNCA utilizar el modelo `gpt-4`**. Es un modelo obsoleto, lento y con un coste prohibitivo. Su uso está terminantemente prohibido.
- El modelo por defecto para todas las operaciones de IA debe ser **`gpt-5-mini`**.
- Para tareas que requieran mayor potencia, se puede utilizar `gpt-5`, y para tareas más sencillas o económicas, `gpt-5-nano`. La selección del modelo debe ser configurable por el usuario siempre que sea posible.

## Dependencias de Entorno
- La funcionalidad de extracción de glosario depende del modelo de lenguaje `en_core_web_sm` de spaCy. Si no está instalado, ejecute el siguiente comando:
  ```bash
  python -m spacy download en_core_web_sm
  ```
