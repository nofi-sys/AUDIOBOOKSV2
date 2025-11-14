# Plan de Mejoras para la Aplicación de Revisión de Audiolibros

Este documento resume las especificaciones, problemas y prioridades para mejorar la aplicación de control de calidad de audiolibros, basado en la reflexión detallada del proyecto.

---

## 1. Objetivo general de la aplicación

La aplicación busca **reducir al mínimo el trabajo manual de edición** en grabaciones largas, detectando y corrigiendo eficientemente los errores groseros de lectura (repeticiones, falsos comienzos, etc.) para dejar el material en un estado publicable.

- [ ] **Meta:** Reducir la revisión manual a menos del 20-30% del material.

---

## 2. Flujo actual y problema central

El flujo actual (cargar audio/texto -> transcribir -> comparar) genera demasiados falsos errores, obligando a una revisión manual excesiva.

- [ ] **Meta:** Reducir drásticamente los falsos negativos (errores marcados como "mal" que están bien).

---

## 3. Prioridad funcional: Detección de repeticiones y micro-tomas

El error más crítico a detectar es cuando el locutor se equivoca y repite un fragmento.

- [ ] **Tarea:** Implementar un sistema robusto para detectar repeticiones de segmentos de texto.
- [ ] **Tarea:** Marcar los bloques repetidos, asumiendo por defecto que la **última toma es la válida**.

---

## 4. Problemas en la comparación texto–transcripción

### 4.1. Errores de Transcripción (ASR)
- [ ] **Tarea:** Implementar el uso de un modelo de transcripción más pesado y preciso.
- [ ] **Tarea:** Alimentar el modelo ASR con un **glosario de palabras clave** extraído del texto original para mejorar la precisión.

### 4.2. Glosario de palabras clave
- [ ] **Tarea:** Reutilizar o implementar un método para procesar el texto original (ej. con NLP) y extraer entidades nombradas y términos clave.
- [ ] **Tarea:** Integrar la generación del glosario en el flujo de la aplicación.

---

## 5. Problemas técnicos adicionales

- [ ] **Bug:** Corregir el **truncado de texto** en las celdas de la tabla de comparación.
- [ ] **Crítico:** Asegurar que la tasa de **falsos positivos** (marcar algo mal como "OK") sea **cero**.

---

## 6. Mejora del proceso de comparación con IA

- [ ] **Tarea:** Implementar un sistema de **contexto persistente (cache)** para las llamadas a la IA, incluyendo el texto y la transcripción completos del capítulo.
- [ ] **Tarea:** Realizar pruebas sistemáticas para endurecer el sistema contra falsos positivos.
- [ ] **Tarea:** Afinar el sistema para reducir los falsos negativos.

---

## 7. Sincronía audio–texto y precisión temporal

### 7.1. Refinamiento de puntos de anclaje
- [ ] **Tarea:** Implementar una pasada de análisis de audio para encontrar el **inicio exacto de cada palabra ("pie de colina")**.
- [ ] **Tarea:** Implementar la detección del **final de la palabra** y el valle de silencio antes de la siguiente.
- [ ] **Tarea:** Generar metadata de tiempo enriquecida (inicio exacto, final funcional, final de reverberación).

---

## 8. Segmentación del texto

- [ ] **Tarea:** Modificar la lógica de segmentación para que se base en la estructura del **texto original**.
- [ ] **Opción A:** Segmentar por **párrafos**, con subdivisión en oraciones si son demasiado largos.
- [ ] **Opción B:** Segmentar directamente por **oraciones** usando signos de puntuación.

---

## 9. Segunda pasada: Modelo que evalúa audio directamente

- [ ] **(Futuro):** Implementar una segunda capa de validación para los bloques dudosos, usando un modelo que compare directamente el **audio** con el texto original.

---

## 10. Especificaciones para el uso de modelos de IA

- [ ] **Tarea:** Crear un documento de especificaciones técnicas que defina **qué modelos de IA usar, con qué parámetros y límites de tokens**.
- [ ] **Meta:** Asegurarse de que los agentes de desarrollo respeten estas especificaciones para controlar costos y evitar errores.

---

## 11. Plazos y prioridades inmediatas

- [x] **Prioridad 1:** Robustecer la **transcripción** (modelo + glosario).
- [x] **Prioridad 2:** Corregir el problema de **truncado de celdas**.
- [ ] **Prioridad 3:** Reestructurar la **comparación con IA** (contexto fijo, evaluación fila a fila, pruebas de falsos positivos).
- [ ] **Prioridad 4:** Mejorar la **sincronía audio–texto** (metadata refinada por palabra).
- [ ] **Prioridad 5:** Cambiar la **segmentación** a párrafos u oraciones.

---

## 12. Integración futura en la web y modelo de negocio

- [ ] **(Futuro):** Integrar la app en la web de **Mississippi Editions**.
- [ ] **(Futuro):** Definir un modelo de negocio (gratis para herramientas sin IA, créditos/suscripción para herramientas con IA).

---

## 13. Dimensión educativa y ecosistema

- [ ] **(Futuro):** Crear contenido educativo sobre grabación de audiolibros.
- [ ] **(Futuro):** Establecer un ecosistema de servicios (conexiones con estudios, imprentas, etc.).

---

## 14. Etapa posterior: Maquetado estructurado

- [ ] **(Futuro):** Implementar un sistema de maquetado estructurado para que el resultado final esté listo para subir a plataformas como **Audible**.
