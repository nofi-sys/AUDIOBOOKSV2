import sys
import traceback
from tkinter import messagebox


def show_error(title: str, exc: BaseException) -> None:
    """Show a Tk error dialog and log full traceback to stderr."""
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    messagebox.showerror(title, str(exc))
