# Vista continua basada en anclas - checklist

- [x] Escribir checklist y hitos inmediatos.
- [x] Cargar alineacion desde .align.csv/.align.db a un modelo de palabras con indices ref/asr.
- [x] Identificar bloques ancla e interancla, detectar repeticiones sospechosas y vincular cada palabra con su fila.
- [x] Implementar `block_to_time` para mapear bloques a intervalos aproximados de audio.
- [x] Crear vista Kivy de doble panel continuo (original/ASR) con coloreado y navegacion por bloques.
- [x] Exponer una entrada desde la app existente para lanzar la nueva vista continua.
