# Laboratorio de Puntuación

Este módulo experimental reúne herramientas y documentación para normalizar signos de puntuación y formatos de diálogo en traducciones literarias.

## Objetivos

- Identificar patrones problemáticos (comillas anglosajonas, diálogos con comillas dobles, combinaciones de signos atípicas).
- Probar heurísticas y expresiones regulares antes de incorporarlas al pipeline principal.
- Registrar resultados de cada experimento junto con ejemplos de texto y diffs sugeridos.

## Estructura

- `experiments.md`: bitácora de pruebas, hallazgos y próximos pasos.
- `regex_playground.py` *(por crear)*: espacio para prototipos rápidos de búsqueda/reemplazo.
- `fixtures/` *(opcional)*: fragmentos de texto utilizados en los experimentos.

## Próximos Pasos

1. Definir dataset de casos problemáticos (diálogos, citas, títulos).
2. Implementar escáner inicial que marque coincidencias sin modificar el texto.
3. Documentar cada ejecución en `experiments.md`, incluyendo métricas y decisiones.

> Nota: el laboratorio se ejecutará de forma manual hasta que las reglas alcancen un nivel de precisión razonable para integrarse al proceso automatizado.
