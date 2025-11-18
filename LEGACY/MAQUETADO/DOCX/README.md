# MD → DOCX (KDP) — MVP

Convierte un archivo Markdown (`.md`) a `.docx` listo para maquetar en Amazon KDP, con control de tamaño de página, márgenes y estilos de párrafo detectados.

## Qué hace
- **Detecta estilos** presentes en el `.md`: títulos (`#`, `##`, `###`), párrafos, citas (`>`), listas con viñetas (`-`, `*`, `+`), listas numeradas (`1.`) y **bloques de código** (```` ``` ````).
- **Configura por estilo**: tipografía, tamaño de letra, alineación, interlineado, espacio antes/después e indentación de primera línea (todo color negro).
- **Página**: presets KDP (`5x8 in` por defecto), A5/A4 o personalizado; márgenes editables.
- **Listo para integrar** como **módulo** en otra app: usa `convert_markdown_to_docx(md_path, output_path=None, page_cfg=None, style_cfgs=None)`.

## Requisitos
```bash
pip install -r requirements.txt
```

- `python-docx` para generar `.docx`.
- `tkinter` viene con Python (en Linux puede requerir `sudo apt install python3-tk`).

## Uso rápido (GUI)
```bash
python md2docx_gui.py
```
1. Clic en **Buscar…** para elegir tu `.md`.
2. Ajustá la **Página** (preset/márgenes).
3. Abrí **Configurar estilos tipográficos…** para tocar fuentes, tamaños e interlineado de los estilos **detectados** en tu documento.
4. Clic en **Convertir a DOCX**. El archivo se guarda junto al `.md` con sufijo `_KDP.docx`.

## Uso por línea de comando
```bash
python md2docx_gui.py entrada.md salida.docx
```
Si no indicás `salida.docx`, genera `entrada_KDP.docx` en la misma carpeta.

## Notas y límites del MVP
- Soporta **negrita** (`**…**` o `__…__`), *itálica* (`*…*` o `_…_`), ~~tachado~~ (`~~…~~`), __subrayado__ con `++…++`, y `código` en línea con acentos invertidos.
- Las **listas** están limitadas a un **nivel** en este MVP.
- **Imágenes** y **tablas** no están incluidas aún (se pueden añadir en siguiente iteración).
- Títulos de nivel 4–6 se tratan como párrafos, por simplicidad.

## Arquitectura (pensada para reuso)
- **Parser**: `parse_markdown_blocks(md_text)` produce una lista de bloques y el set de estilos usados.
- **Constructor DOCX**: `build_document(blocks, style_cfgs, page_cfg)` arma el documento aplicando maquetado.
- **API**: `convert_markdown_to_docx(...)` para usar como módulo desde otra app.
- **GUI**: `App` (tkinter) — abre un diálogo de estilos que se genera **a partir de los estilos detectados**.

## Defaults recomendados (KDP)
- **5x8 in**, márgenes 0.75 in.
- **Garamond 12** para cuerpo, interlineado **1.0**, **espacio después** de 6 pt (sin sangría de primera línea por defecto).
- Títulos con tamaños 16/14/12 pt.

> Puedes ajustar todo esto en la GUI y exportar.

---

Hecho con cariño tipográfico. Siguiente parada: imágenes, tablas, notas al pie y listas multinivel.
