from __future__ import annotations

"""Simple GUI to clean JSON text columns using :mod:`clean_json_text`."""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from utils.clean_json_text import clean_file
from utils.gui_errors import show_error


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    json_path = filedialog.askopenfilename(
        title="Seleccionar JSON", filetypes=[("JSON", "*.json;*.qc.json")]
    )
    if not json_path:
        root.destroy()
        return

    out_path = filedialog.asksaveasfilename(
        title="Guardar como…",
        initialfile=Path(json_path).with_suffix(".clean.json").name,
        defaultextension=".json",
        filetypes=[("JSON", "*.json")],
    )
    if not out_path:
        root.destroy()
        return

    try:
        out = clean_file(json_path, out_path)
    except Exception as exc:  # pragma: no cover - GUI
        show_error("Error", exc)
    else:
        messagebox.showinfo("Listo", f"Guardado {out}")
    finally:
        root.destroy()


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
