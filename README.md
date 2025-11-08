# QC-Audiolibro: Herramienta de Control de Calidad

QC-Audiolibro es una aplicación de escritorio diseñada para facilitar el proceso de control de calidad (QC) en la producción de audiolibros. Compara el guion original con la transcripción automática (ASR) del audio grabado, permitiendo a los editores identificar y corregir errores de manera eficiente.

## Funcionalidades Principales

- **Alineación de Texto y Audio**: Sincroniza el guion original con la transcripción ASR, mostrando las discrepancias línea por línea.
- **Reproducción Sincronizada**: Permite reproducir segmentos de audio correspondientes a cada línea de texto directamente desde la aplicación.
- **Revisión Asistida por IA**: Utiliza modelos de OpenAI (GPT-4o) para realizar una revisión automática, marcando posibles errores (`mal`) o confirmando lecturas correctas (`ok`).
- **Edición y Corrección Intuitiva**: Facilita la corrección de errores de transcripción, la fusión de líneas y el ajuste de la sincronización.
- **Re-transcripción de Segmentos**: Permite re-transcribir fragmentos específicos del audio que presenten errores graves utilizando un modelo de ASR más potente.
- **Revisión Avanzada con IA**: Ofrece un análisis contextual de los errores, clasificándolos en categorías como repeticiones, omisiones, errores de lectura o desalineación.
- **Exportación de Datos**: Guarda el trabajo en un archivo JSON que preserva todas las ediciones y metadatos para futuras revisiones.
- **Creación de EDL**: Genera un archivo de lista de decisiones de edición (EDL) para automatizar cortes y correcciones en software de edición de audio.

## Flujo de Trabajo Recomendado

1.  **Cargar Archivos**:
    *   **Guion**: Carga el guion del audiolibro en formato PDF o TXT.
    *   **TXT ASR**: Proporciona la transcripción generada por el sistema de reconocimiento de voz.
    *   **Audio**: Selecciona el archivo de audio correspondiente (MP3, WAV, M4A, etc.).

2.  **Procesar**:
    *   Haz clic en el botón **"Procesar"**. La aplicación alineará el guion y el ASR, generando una tabla comparativa.
    *   El resultado se guarda automáticamente en un archivo `.qc.json`. Este archivo es tu sesión de trabajo.

3.  **Revisión con IA**:
    *   **Configuración**: Antes del primer uso, asegúrate de tener una clave de API de OpenAI configurada como variable de entorno (`OPENAI_API_KEY`). Puedes crear un archivo `.env` en la carpeta del proyecto para guardarla de forma segura:
        ```
        OPENAI_API_KEY=sk-tuclaveaqui
        ```
    *   **AI Review**: Haz clic en **"AI Review"** para que la IA analice todas las líneas que aún no han sido revisadas. Marcará las líneas como `ok` o `mal` en la columna "AI".
    *   **Revisión AI Avanzada**: Para las líneas marcadas como `mal` o `⚠️`, puedes usar la **"Revisión AI Avanzada"**. Esta función proporciona un análisis más detallado del error, considerando el contexto de las líneas adyacentes.

4.  **Revisión Manual**:
    *   Navega por las filas, especialmente las marcadas como `mal`.
    *   Haz doble clic en una fila para abrir una ventana de reproducción y escuchar el audio correspondiente.
    *   Edita directamente las celdas de "Original" o "ASR" para corregir errores.
    *   Utiliza el menú contextual (clic derecho) para fusionar filas o mover palabras entre ellas.
    *   Marca las filas como "OK" en su columna correspondiente a medida que las revisas.

5.  **Guardar y Continuar**:
    *   Todos los cambios se guardan automáticamente en el archivo JSON. Puedes cerrar la aplicación y volver a cargar el archivo JSON para continuar tu trabajo en cualquier momento.

## Instalación

Para ejecutar la aplicación, necesitas Python 3.10 o superior.

1.  Clona este repositorio o descarga los archivos.
2.  Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ejecuta la aplicación:
    ```bash
    python qc_app.py
    ```

## Mejoras y Futuro Desarrollo

Esta aplicación está en desarrollo activo. Consulta el archivo `IMPROVEMENTS.md` para ver una lista de las próximas funcionalidades y mejoras planificadas.
