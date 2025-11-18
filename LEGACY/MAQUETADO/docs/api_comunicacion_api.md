# Guia de comunicacion con la API

Este documento registra la forma exacta en que `txt2md_mvp` conversa con la
API de OpenAI. Mantener estas reglas evita caer nuevamente en problemas como
el uso de temperaturas no soportadas o limites de tokens demasiado bajos.
Consulta y actualiza este archivo cada vez que se agregue una nueva llamada.

## Reglas generales

- **Cliente unico**. Todas las peticiones pasan por la instancia
  `openai.OpenAI` creada en `translation_engine.TranslationEngine`. Si la
  variable de entorno `OPENAI_API_KEY` no existe, el motor desactiva las
  funciones de IA sin lanzar excepciones.
- **Sin `temperature` personalizado**. Desde octubre 2025 dejamos de fijar
  valores manuales; usamos siempre la temperatura por defecto del modelo (`1`)
  para traduccion, glosario, resumen y control de calidad.
- **Limites de tokens**:
  - Revision: `max_completion_tokens = 10000` (modo completo y dictamen).
  - Glosario (curacion y traduccion): 400 tokens por lote.
  - Resumen: 300 tokens.
  - Traduccion de bloques: se deja al modelo (no se fija un maximo explicito
    porque depende del tamano del fragmento).
- **Registro obligatorio**. Cada interaccion invoca
  `api_logger.log_interaction`, que escribe un JSON individual y agrega la
  linea correspondiente al `.jsonl` de la sesion
  (`logs/<proyecto>/<modo>/<timestamp>_<modelo>.json[.jsonl]`). Esta traza es
  la base para depurar.
- **Salidas en JSON**. Cuando esperamos estructuras JSON se anade la orden en
  el prompt y, cuando la API lo permite (revision), se usa
  `response_format={"type": "json_object"}` para forzar el formato.

## Preparacion de traduccion

1. **Extraccion de terminos** (`GlossaryBuilder.extract_terms`)
   - Utiliza spaCy si esta disponible; de lo contrario cae a un extractor con
     expresiones regulares sin detener el flujo.
2. **Curacion** (`GlossaryBuilder.curate_terms`)
   - Prompt en espanol con contexto y metadatos por termino.
   - `max_completion_tokens = 400`.
   - Si no hay cliente de OpenAI se devuelve una politica `"keep"` por termino
     y se continua.
3. **Traduccion del glosario** (`GlossaryBuilder.translate_glossary`)
   - Mismos parametros que la curacion; devuelve un diccionario
     `termino -> traduccion`.
4. **Persistencia** (`TranslationEngine.run_preparation`)
   - Siempre escribe `<salida>/<archivo>_glossary.json`. No aborta si el
     glosario queda vacio, solo lo informa en la consola.

## Traduccion por bloques

- `TranslationCaller.translate_chunk` arma un prompt con:
  - Resumen global.
  - Glosario completo (obligatorio).
  - Texto previo y posterior como referencia.
  - Guia de estilo (perfil + notas).
- El mensaje se envia como `role=system`. No se fija `temperature` ni limites
  adicionales; la longitud queda a cargo del modelo.
- Cada respuesta se registra en `*_translation_log.txt` mediante
  `log_simplified_translation`, lo que permite reanudar una traduccion cortada.
- `_announce_api_call` notifica cada invocacion para que la GUI muestre la
  traza en tiempo real.

### Cancelacion y reanudacion

- La GUI dispone de un `threading.Event` para cancelar. Una vez activado, el
  traductor termina el bloque en curso, marca la corrida como cancelada a traves
  de `TranslationEngine.was_last_translation_cancelled()` y devuelve el
  documento parcial.
- El progreso parcial se guarda como “Traduccion actual (parcial)” sin
  sobreescribir la traduccion original. El usuario puede reanudar (opcion
  “Reanudar traduccion previa”) o volver a iniciar desde cero.

## Revision (QA)

- `TranslationQA.review_block` trabaja bloque a bloque con:
  - `max_completion_tokens = 10000`.
  - `response_format={"type": "json_object"}` para asegurar una respuesta
    valida.
  - Sin `temperature` personalizado.
- Las opciones del dialogo (`Solicitar observaciones`, `Solo estado`) viajan
  directamente en el prompt.
- Se imprime `finish_reason` y el uso de tokens en consola y en el `.jsonl`
  de la sesion para facilitar auditorias.

## Convenciones de registro

- Cada sesion (traduccion o revision) define su carpeta de logs con
  `api_logger.set_log_directory`. Los archivos individuales siguen el patron
  `YYYYMMDD_HHMMSS_micro_model.json`.
- La consola de la GUI informa la ruta de logs al iniciar cada flujo para
  simplificar el soporte.
- Cualquier llamada nueva a la API debe:
  1. Escribir un `log_interaction` (o `log_simplified_translation`).
  2. Mantener la temperatura por defecto, salvo que exista una justificacion
     documentada en esta guia.
  3. Respetar los limites de tokens anteriores o explicitar la excepcion.

Con estas pautas mantenemos prompts reutilizables, detectamos errores en los
logs y evitamos volver a configuraciones no soportadas. Actualiza este archivo
antes de incorporar nuevas integraciones.
