# Informe de Optimización de Costos de Traducción

## 1. Alto Consumo de Tokens de Entrada

**Estado actual:** `aggregate_mode` está activado de forma predeterminada tanto en la GUI como en `TranslationEngine.run_translation`. El traductor agrupa los bloques hasta el umbral configurado y, al finalizar cada corrida, registra un resumen de tokens (entrada, salida y cacheados) consumidos durante la traducción.

**Próximos pasos:** Ajustar el objetivo de palabras al tamaño medio de los capítulos y comparar los registros actuales con ejecuciones previas para estimar el ahorro porcentual.

## 2. Alto Consumo de Tokens de Salida

**Estado actual:** Las sesiones de revisión arrancan con `verdict_only` activo, incluso al recuperar opciones guardadas. Las observaciones se deshabilitan automáticamente cuando solo se requiere el dictamen, evitando respuestas extensas de la IA.

**Próximos pasos:** Monitorear los casos que requieran texto completo para confirmar que la opción siga disponible y documentar el impacto en el volumen de salida.

## 3. Duplicación de Subtítulos

**Estado actual:** El pipeline marca los bloques consumidos por subtítulo (`_consumed_as_subtitle`) antes de seguir escaneando, evitando repeticiones. Falta confirmar con lotes de prueba que mezclen subtítulos en la misma línea y en la siguiente.

**Próximos pasos:** Añadir casos específicos a la suite de tests y registrar evidencias en `demo/ejemplo.txt`.

## 4. QA Opcional

**Estado actual:** La GUI incorpora un conmutador para omitir la auditoría QA automática. Cuando está activo, no se realizan llamadas de muestreo ni se genera el informe JSON asociado.

**Riesgos:** Saltarse el QA reduce costes, pero puede dejar pasar calcos o inconsistencias léxicas. Se recomienda usarlo solo cuando exista supervisión humana posterior o en rehaces iterativos donde los bloques ya fueron auditados.

**Próximos pasos:** Evaluar si conviene guardar en la sesión que la QA se omitió para exigir una verificación manual antes de la entrega.
