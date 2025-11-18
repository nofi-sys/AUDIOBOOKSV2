
# txt2md_mvp (Producto Mínimo Viable)

Convierte `.txt` en `.md` con detección de H1/H2/H3 basada en reglas y heurísticas.
Usa plantillas declarativas (`templates/*.yml`) para definir patrones según la familia del documento.
Incluye `analysis.json` por línea y `report.md` con un resumen enriquecido con métricas.

## Cómo usar (GUI)

```
python -m txt2md_mvp
```

La interfaz permite:
- Agregar múltiples archivos `.txt` o carpetas completas (con patrón glob y modo recursivo opcional).
- Elegir el directorio de salida.
- Ejecutar el procesamiento y revisar un log con rutas generadas.

Archivos generados por cada texto:
- `out/archivo.md`
- `out/archivo.analysis.json`
- `out/archivo.report.md`

## Automatización y pruebas

- Utiliza `txt2md_mvp.inputs.gather_inputs` para reutilizar la lógica de descubrimiento de archivos en scripts.
- Ejecuta las pruebas básicas del motor con:

```
python3 -m unittest discover -s tests
```

## Estado
MVP determinista. Próximas fases: descubrimiento de plantillas (skeleton+clustering) e integración LLM opcional.
