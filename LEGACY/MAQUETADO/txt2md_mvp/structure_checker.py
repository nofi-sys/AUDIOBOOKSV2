from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional

class StructureCheckerDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, document: List[Dict[str, Any]]) -> None:
        super().__init__(parent)
        self.transient(parent)
        self.title("Revisión de Estructura")
        self.geometry("800x600")

        self.document = [item.copy() for item in document]
        self.result: Optional[List[Dict[str, Any]]] = None

        self._build_controls()
        self._populate_tree()

        self.grab_set()

    def _build_controls(self) -> None:
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.tree = ttk.Treeview(tree_frame, columns=("type", "text"), show="headings")
        self.tree.heading("type", text="Tipo")
        self.tree.heading("text", text="Texto")
        self.tree.column("type", width=100, stretch=False)
        self.tree.column("text", width=600)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_selection)

        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill="x")

        self.type_var = tk.StringVar()
        self.type_options = ["p", "h1", "h2", "h3", "h4", "blockquote", "hr"]
        self.type_menu = ttk.Combobox(
            actions_frame,
            textvariable=self.type_var,
            values=self.type_options,
            state="disabled",
        )
        self.type_menu.pack(side="left", padx=(0, 10))
        self.type_menu.bind("<<ComboboxSelected>>", self._change_type)

        footer_frame = ttk.Frame(self)
        footer_frame.pack(fill="x", side="bottom", padx=10, pady=10)
        ttk.Button(footer_frame, text="Cancelar", command=self._on_cancel).pack(side="right", padx=5)
        ttk.Button(footer_frame, text="Aceptar", command=self._on_accept, default=tk.ACTIVE).pack(side="right", padx=5)
        self.bind("<Escape>", lambda e: self._on_cancel())

    def _populate_tree(self) -> None:
        for i, block in enumerate(self.document):
            text_preview = (block.get("text", "") or "").strip().replace("\n", " ")
            if len(text_preview) > 100:
                text_preview = text_preview[:97] + "..."
            self.tree.insert("", "end", iid=str(i), values=(block.get("type", ""), text_preview))

    def _on_selection(self, event: Any) -> None:
        selected_items = self.tree.selection()
        if not selected_items:
            self.type_menu.config(state="disabled")
            return

        self.type_menu.config(state="readonly")
        first_item = selected_items[0]
        item_type = self.tree.item(first_item, "values")[0]
        if item_type in self.type_options:
            self.type_var.set(item_type)

    def _change_type(self, event: Any) -> None:
        new_type = self.type_var.get()
        if not new_type:
            return

        for item_id in self.tree.selection():
            doc_index = int(item_id)
            self.document[doc_index]["type"] = new_type
            self.tree.item(item_id, values=(new_type, self.tree.item(item_id, "values")[1]))

    def _on_accept(self) -> None:
        self.result = self.document
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()

    def show(self) -> Optional[List[Dict[str, Any]]]:
        self.wait_window(self)
        return self.result
