# Lista de Mejoras para QC-Audiolibro

Este documento describe las mejoras planificadas para la aplicación QC-Audiolibro, priorizadas según las necesidades del flujo de trabajo de producción.

## Prioridad Alta

### 1. Detección de Repeticiones con IA
- **Descripción**: Implementar una funcionalidad de revisión por IA enfocada específicamente en detectar repeticiones de frases o palabras innecesarias, que son un tipo de error común en la grabación de audiolibros.
- **Detalles Técnicos**:
    - Añadir un *checkbox* en la interfaz gráfica con la etiqueta "Detectar Repeticiones".
    - Cuando esta opción esté activa, la "Revisión AI Avanzada" utilizará un *prompt* especializado y un algoritmo más robusto para identificar frases repetidas que no formen parte del estilo literario del texto.
    - El veredicto de la IA debería ser específico, como `REPETICION`, para que el revisor pueda identificar rápidamente este tipo de error.

## Prioridad Media

### 2. Visualización de Forma de Onda
- **Descripción**: Integrar un pequeño panel que muestre la forma de onda del audio para el segmento de la fila seleccionada.
- **Beneficio**: Permitiría a los revisores identificar visualmente problemas de edición de audio, como cortes abruptos, clics, o silencios inadecuados, sin depender únicamente de la escucha.

### 3. Exportación a CSV/Excel
- **Descripción**: Añadir una opción en el menú para exportar la tabla de QC a un archivo en formato CSV o Excel.
- **Beneficio**: Facilitaría la creación de informes de calidad, el seguimiento de métricas de error y la compartición de datos con otros miembros del equipo que no necesiten usar la aplicación directamente.

### 4. Atajos de Teclado (Hotkeys)
- **Descripción**: Implementar atajos de teclado para las acciones más comunes, como reproducir/pausar, marcar una fila como "OK" o "mal", y navegar a la siguiente o anterior fila problemática.
- **Beneficio**: Aumentaría significativamente la velocidad y eficiencia de la revisión manual.

## Prioridad Baja

### 5. Soporte Multilingüe
- **Descripción**: Añadir soporte para cambiar el idioma de la interfaz de usuario (inicialmente, a inglés).
- **Beneficio**: Haría la herramienta accesible para equipos de producción internacionales.

### 6. Tema Oscuro
- **Descripción**: Implementar un tema oscuro para la interfaz de la aplicación.
- **Beneficio**: Reduciría la fatiga visual de los revisores durante sesiones de trabajo prolongadas.

### 7. Mejoras en la Integración con Audacity
- **Descripción**: Ampliar la funcionalidad de marcadores para permitir la importación y exportación de etiquetas de Audacity.
- **Beneficio**: Crearía un flujo de trabajo más fluido para los editores que utilizan Audacity para realizar las correcciones de audio finales.

## Checklist de Implementación
- [x] 1. Detección de Repeticiones con IA
- [ ] 2. Visualización de Forma de Onda
- [ ] 3. Exportación a CSV/Excel
- [ ] 4. Atajos de Teclado (Hotkeys)
- [ ] 5. Soporte Multilingüe
- [ ] 6. Tema Oscuro
- [ ] 7. Mejoras en la Integración con Audacity
