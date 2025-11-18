# Plan de Optimización txt2md

## Etapa 1 · Motor de reglas endurecido
- [x] Reforzar `titlecase_short` con restricciones negativas (2025-10-03)
- [x] Fusionar encabezados partidos por guion/retorno en mayúsculas antes de clasificarlos (2025-10-03)
- [x] Instrumentar métricas de tokens por modelo y etapa (2025-10-21)
- [x] Ajustar `all_caps_short` con pistas positivas (2025-10-21)
- [x] Detectar separadores de escena y mapearlos a `---` (2025-10-21)
- [x] Añadir reglas para portada y preliminares (2025-10-21)
- [x] Identificar bloques con sangrías o estilo epistolar/poético (2025-10-21)
- [ ] Ampliar la suite de pruebas con casos negativos y positivos
- [ ] Introducir un preprocesador de bloques antes del motor de reglas
- [x] Revisar listas retóricas y puntos enfáticos (en progreso: requiere bloqueador de headings cortos)
- [ ] Detectar bloques tabulares básicos y preservarlos
- [x] Optimizar prompts de traducción manteniendo calidad (2025-10-21)

## Etapa 2 · Contexto y memoria
- [ ] Diseñar máquina de estados para portada/capítulos/anexos
- [ ] Validar transiciones usando estado previo/siguiente
- [ ] Analizar patrones globales por documento
- [x] Integrar señales de diálogos como descarte de encabezados (2025-10-02)
- [ ] Registrar métricas de consistencia en `.report.md`
- [ ] Definir scoring heurístico por bloque
- [ ] Modelar dependencias entre bloques

## Etapa 3 · Modelos de aprendizaje
- [ ] Preparar dataset etiquetado
- [ ] Entrenar clasificador tradicional
- [ ] Evaluar modelos de lenguaje compactos
- [ ] Crear módulo para portada y paratextos
- [ ] Integrar modelos con métricas de confianza y fallback

### Notas 2025-10-21
- Métricas de tokens accesibles desde la GUI y se resumen al cerrar cada traducción.
- Próximo objetivo: compactar prompts sin perder instrucciones críticas (prioridad actual).
