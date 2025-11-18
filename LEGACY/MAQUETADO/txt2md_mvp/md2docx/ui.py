#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui.py
-----
Contiene los componentes de la interfaz de usuario (Tkinter) para el módulo md2docx.
Principalmente, el diálogo de configuración de estilos.
"""
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Tuple, Any

from .core import DEFAULT_STYLESET

COMMON_FONTS = [
    "Garamond",
    "Times New Roman",
    "Georgia",
    "Palatino Linotype",
    "Cambria",
    "Book Antiqua",
    "Minion Pro",
    "Libre Baskerville",
    "Merriweather",
    "Courier New",
]

class StylesDialog(tk.Toplevel):
    def __init__(self, master, used_styles: List[str], style_cfgs: Dict[str, Dict[str, Any]]):
        super().__init__(master)
        self.title("Configurar estilos tipográficos")
        self.style_cfgs = style_cfgs
        self.used_styles = used_styles
        self.vars: Dict[Tuple[str, str], tk.Variable] = {}
        self._build()

    def _build(self):
        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        info = ttk.Label(container, text="Configura tipografía y espaciamientos por estilo. (Color siempre negro en la exportación)")
        info.pack(anchor="w", pady=(0,10))

        canvas_container = ttk.Frame(container)
        canvas_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(canvas_container, highlightthickness=0)
        scroll_y = ttk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        scroll_x = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(fill="x", pady=(6, 0))

        def _sync_inner_width(event, canvas=canvas, frame=frame, item_id=window_id):
            if frame.winfo_reqwidth() <= event.width:
                canvas.itemconfigure(item_id, width=event.width)
        canvas.bind("<Configure>", _sync_inner_width)

        def _on_mousewheel(event):
            # Para Linux, se usan botones 4 y 5
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            # Para Windows y macOS, se usa delta
            else:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        frame.bind("<MouseWheel>", _on_mousewheel)
        frame.bind("<Button-4>", _on_mousewheel)
        frame.bind("<Button-5>", _on_mousewheel)
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)

        for sk in self.used_styles:
            cfg = self.style_cfgs.get(sk, DEFAULT_STYLESET[sk])
            lf = ttk.LabelFrame(frame, text=f"Estilo: {sk}", padding=10)
            lf.pack(fill="x", expand=True, pady=6)

            # Font family
            ttk.Label(lf, text="Fuente:").grid(row=0, column=0, sticky="w")
            v_font = tk.StringVar(value=str(cfg.get("font_name","Garamond")))
            cb_font = ttk.Combobox(lf, textvariable=v_font, values=COMMON_FONTS, width=24)
            cb_font.grid(row=0, column=1, sticky="w", padx=6)

            # Font size
            ttk.Label(lf, text="Tamaño (pt):").grid(row=0, column=2, sticky="w")
            v_size = tk.DoubleVar(value=float(cfg.get("font_size_pt",12)))
            sp_size = ttk.Spinbox(lf, from_=8, to=48, increment=0.5, textvariable=v_size, width=6)
            sp_size.grid(row=0, column=3, sticky="w", padx=6)

            # Alignment
            ttk.Label(lf, text="Alineación:").grid(row=0, column=4, sticky="w")
            v_align = tk.StringVar(value=str(cfg.get("align","left")))
            cb_align = ttk.Combobox(lf, textvariable=v_align, values=["left","center","right","justify"], width=10)
            cb_align.grid(row=0, column=5, sticky="w", padx=6)

            # Bold checkbox
            v_bold = tk.BooleanVar(value=bool(cfg.get("bold", False)))
            chk_bold = ttk.Checkbutton(lf, text="Negrita", variable=v_bold)
            chk_bold.grid(row=0, column=6, sticky="w", padx=6)

            # Italic checkbox
            v_italic = tk.BooleanVar(value=bool(cfg.get("italic", False)))
            chk_italic = ttk.Checkbutton(lf, text="Cursiva", variable=v_italic)
            chk_italic.grid(row=0, column=7, sticky="w", padx=6)

            # Line spacing
            ttk.Label(lf, text="Interlineado:").grid(row=1, column=0, sticky="w", pady=4)
            v_ls = tk.DoubleVar(value=float(cfg.get("line_spacing",1.0)))
            cb_ls = ttk.Combobox(lf, textvariable=v_ls, values=[1.0,1.15,1.5,2.0], width=6)
            cb_ls.grid(row=1, column=1, sticky="w", padx=6)

            # Space before
            ttk.Label(lf, text="Espacio antes (pt):").grid(row=1, column=2, sticky="w")
            v_before = tk.DoubleVar(value=float(cfg.get("space_before_pt",0)))
            sp_before = ttk.Spinbox(lf, from_=0, to=48, increment=1, textvariable=v_before, width=6)
            sp_before.grid(row=1, column=3, sticky="w", padx=6)

            # Space after
            ttk.Label(lf, text="Espacio después (pt):").grid(row=1, column=4, sticky="w")
            v_after = tk.DoubleVar(value=float(cfg.get("space_after_pt",6)))
            sp_after = ttk.Spinbox(lf, from_=0, to=48, increment=1, textvariable=v_after, width=6)
            sp_after.grid(row=1, column=5, sticky="w", padx=6)

            # First line indent
            ttk.Label(lf, text="Sangría primera línea (in):").grid(row=2, column=0, sticky="w", pady=4)
            v_indent = tk.DoubleVar(value=float(cfg.get("first_line_indent_in",0.0)))
            sp_indent = ttk.Spinbox(lf, from_=0.0, to=1.0, increment=0.05, textvariable=v_indent, width=6)
            sp_indent.grid(row=2, column=1, sticky="w", padx=6)

            # Page break before (solo para headings)
            if sk.startswith("heading"):
                v_page_break = tk.BooleanVar(value=bool(cfg.get("page_break_before", False)))
                chk_page_break = ttk.Checkbutton(lf, text="Salto de página antes", variable=v_page_break)
                chk_page_break.grid(row=2, column=2, sticky="w", padx=6, pady=4)
                self.vars[(sk, "page_break_before")] = v_page_break

            # Guardar referencias
            self.vars[(sk,"font_name")] = v_font
            self.vars[(sk,"font_size_pt")] = v_size
            self.vars[(sk,"align")] = v_align
            self.vars[(sk,"bold")] = v_bold
            self.vars[(sk,"italic")] = v_italic
            self.vars[(sk,"line_spacing")] = v_ls
            self.vars[(sk,"space_before_pt")] = v_before
            self.vars[(sk,"space_after_pt")] = v_after
            self.vars[(sk,"first_line_indent_in")] = v_indent

        # --- Global options ---
        global_opts_frame = ttk.LabelFrame(frame, text="Opciones Globales", padding=10)
        global_opts_frame.pack(fill="x", expand=True, pady=10)
        global_cfg = self.style_cfgs.setdefault("_global", {})

        # TOC generation
        v_toc = tk.BooleanVar(value=bool(global_cfg.get("generate_toc", False)))
        chk_toc = ttk.Checkbutton(global_opts_frame, text="Generar Tabla de Contenidos automática", variable=v_toc)
        chk_toc.pack(anchor="w")
        self.vars[("_global", "generate_toc")] = v_toc

        pn_frame = ttk.LabelFrame(global_opts_frame, text="Numeracion de pagina", padding=10)
        pn_frame.pack(fill="x", expand=True, pady=(10, 0))

        ttk.Label(pn_frame, text="Alineacion:").grid(row=0, column=0, sticky="w")
        v_pn_align = tk.StringVar(value=str(global_cfg.get("page_number_alignment", "center")))
        cb_pn_align = ttk.Combobox(pn_frame, textvariable=v_pn_align, values=["left", "center", "right"], width=10, state="readonly")
        cb_pn_align.grid(row=0, column=1, sticky="w", padx=6)
        self.vars[("_global", "page_number_alignment")] = v_pn_align

        ttk.Label(pn_frame, text="Fuente:").grid(row=0, column=2, sticky="w")
        v_pn_font = tk.StringVar(value=str(global_cfg.get("page_number_font_name") or ""))
        cb_pn_font = ttk.Combobox(pn_frame, textvariable=v_pn_font, values=COMMON_FONTS, width=24)
        cb_pn_font.grid(row=0, column=3, sticky="w", padx=6)
        self.vars[("_global", "page_number_font_name")] = v_pn_font

        ttk.Label(pn_frame, text="Tamano (pt):").grid(row=1, column=0, sticky="w", pady=(6,0))
        pn_size_val = global_cfg.get("page_number_font_size_pt")
        v_pn_size = tk.StringVar(value="" if pn_size_val is None else str(pn_size_val))
        ent_pn_size = ttk.Entry(pn_frame, textvariable=v_pn_size, width=6)
        ent_pn_size.grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "page_number_font_size_pt")] = v_pn_size

        ttk.Label(pn_frame, text="(dejar vacio para usar el estilo de parrafo)").grid(row=1, column=2, columnspan=2, sticky="w", pady=(6,0))

        header_frame = ttk.LabelFrame(global_opts_frame, text="Encabezados (folios)", padding=10)
        header_frame.pack(fill="x", expand=True, pady=(10, 0))

        ttk.Label(header_frame, text="Texto pagina par (izq):").grid(row=0, column=0, sticky="w")
        v_header_left = tk.StringVar(value=str(global_cfg.get("header_left_text", "")))
        entry_header_left = ttk.Entry(header_frame, textvariable=v_header_left, width=48)
        entry_header_left.grid(row=0, column=1, sticky="w", padx=6)
        self.vars[("_global", "header_left_text")] = v_header_left

        ttk.Label(header_frame, text="Texto pagina impar (der):").grid(row=1, column=0, sticky="w", pady=(6,0))
        v_header_right = tk.StringVar(value=str(global_cfg.get("header_right_text", "")))
        entry_header_right = ttk.Entry(header_frame, textvariable=v_header_right, width=48)
        entry_header_right.grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "header_right_text")] = v_header_right

        ttk.Label(header_frame, text="Fuente encabezado:").grid(row=2, column=0, sticky="w", pady=(6,0))
        v_header_font = tk.StringVar(value=str(global_cfg.get("header_font_name") or ""))
        cb_header_font = ttk.Combobox(header_frame, textvariable=v_header_font, values=COMMON_FONTS, width=24)
        cb_header_font.grid(row=2, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "header_font_name")] = v_header_font

        ttk.Label(header_frame, text="Tamano (pt):").grid(row=3, column=0, sticky="w", pady=(6,0))
        header_size_val = global_cfg.get("header_font_size_pt")
        v_header_size = tk.StringVar(value="" if header_size_val is None else str(header_size_val))
        ent_header_size = ttk.Entry(header_frame, textvariable=v_header_size, width=6)
        ent_header_size.grid(row=3, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "header_font_size_pt")] = v_header_size

        v_header_italic = tk.BooleanVar(value=bool(global_cfg.get("header_italic", True)))
        chk_header_italic = ttk.Checkbutton(header_frame, text="Cursiva", variable=v_header_italic)
        chk_header_italic.grid(row=3, column=2, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "header_italic")] = v_header_italic

        ttk.Label(header_frame, text="Alineacion encabezado:").grid(row=4, column=0, sticky="w", pady=(6,0))
        v_header_align = tk.StringVar(value=str(global_cfg.get("header_alignment", "mirror")))
        cb_header_align = ttk.Combobox(header_frame, textvariable=v_header_align, values=["mirror", "left", "center", "right"], width=12, state="readonly")
        cb_header_align.grid(row=4, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "header_alignment")] = v_header_align

        ttk.Label(header_frame, text="(dejar vacio para usar la fuente de parrafo)").grid(row=5, column=0, columnspan=3, sticky="w", pady=(6,0))

        margins_frame = ttk.LabelFrame(global_opts_frame, text="Margenes de encabezado y pie", padding=10)
        margins_frame.pack(fill="x", expand=True, pady=(10, 0))

        ttk.Label(margins_frame, text="Distancia encabezado (in):").grid(row=0, column=0, sticky="w")
        header_dist_val = global_cfg.get("header_distance_in")
        v_header_dist = tk.StringVar(value="" if header_dist_val is None else str(header_dist_val))
        ent_header_dist = ttk.Entry(margins_frame, textvariable=v_header_dist, width=8)
        ent_header_dist.grid(row=0, column=1, sticky="w", padx=6)
        self.vars[("_global", "header_distance_in")] = v_header_dist

        ttk.Label(margins_frame, text="Espacio despues encabezado (pt):").grid(row=1, column=0, sticky="w", pady=(6,0))
        header_space_val = global_cfg.get("header_space_after_pt")
        v_header_space = tk.StringVar(value="" if header_space_val is None else str(header_space_val))
        ent_header_space = ttk.Entry(margins_frame, textvariable=v_header_space, width=8)
        ent_header_space.grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "header_space_after_pt")] = v_header_space

        ttk.Label(margins_frame, text="Distancia pie (in):").grid(row=2, column=0, sticky="w", pady=(6,0))
        footer_dist_val = global_cfg.get("footer_distance_in")
        v_footer_dist = tk.StringVar(value="" if footer_dist_val is None else str(footer_dist_val))
        ent_footer_dist = ttk.Entry(margins_frame, textvariable=v_footer_dist, width=8)
        ent_footer_dist.grid(row=2, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "footer_distance_in")] = v_footer_dist

        ttk.Label(margins_frame, text="Espacio antes pie (pt):").grid(row=3, column=0, sticky="w", pady=(6,0))
        footer_space_val = global_cfg.get("footer_space_before_pt")
        v_footer_space = tk.StringVar(value="" if footer_space_val is None else str(footer_space_val))
        ent_footer_space = ttk.Entry(margins_frame, textvariable=v_footer_space, width=8)
        ent_footer_space.grid(row=3, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "footer_space_before_pt")] = v_footer_space

        chapter_frame = ttk.LabelFrame(global_opts_frame, text="Espaciado de capitulos", padding=10)
        chapter_frame.pack(fill="x", expand=True, pady=(10, 0))

        ttk.Label(chapter_frame, text="Lineas en blanco antes:").grid(row=0, column=0, sticky="w")
        chapter_top_val = global_cfg.get("chapter_padding_top_lines")
        v_chapter_top = tk.StringVar(value="" if chapter_top_val is None else str(chapter_top_val))
        ent_chapter_top = ttk.Entry(chapter_frame, textvariable=v_chapter_top, width=6)
        ent_chapter_top.grid(row=0, column=1, sticky="w", padx=6)
        self.vars[("_global", "chapter_padding_top_lines")] = v_chapter_top

        ttk.Label(chapter_frame, text="Lineas en blanco despues:").grid(row=1, column=0, sticky="w", pady=(6,0))
        chapter_bottom_val = global_cfg.get("chapter_padding_bottom_lines")
        v_chapter_bottom = tk.StringVar(value="" if chapter_bottom_val is None else str(chapter_bottom_val))
        ent_chapter_bottom = ttk.Entry(chapter_frame, textvariable=v_chapter_bottom, width=6)
        ent_chapter_bottom.grid(row=1, column=1, sticky="w", padx=6, pady=(6,0))
        self.vars[("_global", "chapter_padding_bottom_lines")] = v_chapter_bottom

        # User guidance note
        note_text = "Nota: La numeracion de pagina y la tabla de contenidos son campos dinamicos. Tras exportar, abre el DOCX y actualiza los campos (Ctrl+A y despues F9, o clic derecho -> 'Actualizar campos')."
        note_label = ttk.Label(global_opts_frame, text=note_text, wraplength=700, justify="left")
        note_label.pack(anchor="w", pady=(10, 0))


        btns = ttk.Frame(container)
        btns.pack(fill="x", pady=(10,0))
        ttk.Button(btns, text="Aceptar", command=self.on_accept).pack(side="right")
        ttk.Button(btns, text="Cargar plantilla...", command=self.load_template).pack(side="left")
        ttk.Button(btns, text="Guardar plantilla...", command=self.save_template).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancelar", command=self.destroy).pack(side="right", padx=6)

        self.geometry("880x560")

    def save_template(self):
        from tkinter import filedialog, messagebox
        import json
        filepath = filedialog.asksaveasfilename(
            title="Guardar plantilla de estilos",
            filetypes=[("JSON files", "*.json")],
            defaultextension=".json"
        )
        if not filepath:
            return

        # Primero, actualizamos el diccionario interno con los valores de la UI
        self._update_internal_cfgs()

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.style_cfgs, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Plantilla guardada", f"La plantilla de estilos se ha guardado en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error al guardar", f"No se pudo guardar la plantilla:\n{e}")

    def load_template(self):
        from tkinter import filedialog, messagebox
        import json
        filepath = filedialog.askopenfilename(
            title="Cargar plantilla de estilos",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_cfgs = json.load(f)

            # Actualizar las variables de la UI con los valores cargados
            for (sk, key), var in self.vars.items():
                if sk in loaded_cfgs and key in loaded_cfgs[sk]:
                    val = loaded_cfgs[sk][key]
                    if val is None:
                        var.set("")
                    else:
                        var.set(val)

            # Guardar la configuración cargada en el objeto
            self.style_cfgs = loaded_cfgs
            messagebox.showinfo("Plantilla cargada", "La plantilla de estilos se ha cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error al cargar", f"No se pudo cargar o aplicar la plantilla:\n{e}")

    def _update_internal_cfgs(self):
        """Actualiza el diccionario self.style_cfgs con los valores actuales de la UI."""
        numeric_keys = {
            "font_size_pt", "line_spacing", "space_before_pt", "space_after_pt",
            "first_line_indent_in", "page_number_font_size_pt", "header_font_size_pt",
            "header_distance_in", "footer_distance_in", "header_space_after_pt",
            "footer_space_before_pt", "chapter_padding_top_lines", "chapter_padding_bottom_lines",
        }
        bool_keys = {"bold", "italic", "page_break_before", "generate_toc", "header_italic"}

        for (sk, key), var in self.vars.items():
            raw_val = var.get()

            if isinstance(raw_val, str):
                raw_val = raw_val.strip()

            if key in numeric_keys:
                if raw_val in ("", None):
                    val = None
                else:
                    try:
                        val = float(raw_val)
                    except Exception:
                        val = None
            elif key in bool_keys:
                val = bool(raw_val)
            elif key in ("page_number_font_name", "header_font_name"):
                val = raw_val or None
            elif key == "page_number_alignment":
                val = (raw_val or "center").lower()
            elif key == "header_alignment":
                val = (raw_val or "mirror").lower()
            elif key in ("header_left_text", "header_right_text"):
                val = raw_val or ""
            else:
                val = raw_val

            self.style_cfgs.setdefault(sk, {}).update({key: val})

    def on_accept(self):
        # Volcar valores al style_cfgs
        self._update_internal_cfgs()
        self.destroy()
