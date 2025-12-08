from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from anchor_blocks import AlignmentBlocks, Block, load_alignment_paths


PlaybackFunc = Callable[[float, Optional[float]], None]


@dataclass
class ContinuousConfig:
    asr_path: Path
    rows: Optional[Sequence[Sequence]] = None
    audio_path: Optional[str] = None
    align_csv: Optional[Path] = None
    align_db: Optional[Path] = None
    play_callback: Optional[PlaybackFunc] = None


def launch_kivy_view(config: ContinuousConfig) -> None:
    """Entry point to start the Kivy continuous view (lazy-imported)."""
    config.asr_path = Path(config.asr_path)
    if config.align_csv:
        config.align_csv = Path(config.align_csv)
    if config.align_db:
        config.align_db = Path(config.align_db)
    if not config.play_callback and config.audio_path:
        def _pygame_play(start: float, end: Optional[float]) -> None:
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(str(config.audio_path))
                pygame.mixer.music.play(start=start)
                if end is not None:
                    dur = max(0.0, end - start)
                    timer = threading.Timer(dur, pygame.mixer.music.stop)
                    timer.daemon = True
                    timer.start()
            except Exception as exc:  # pragma: no cover
                print(f"[kivy view] playback error: {exc}")
        config.play_callback = _pygame_play

    try:
        from kivy.app import App
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.scrollview import ScrollView
        from kivy.uix.gridlayout import GridLayout
        from kivy.uix.behaviors import ButtonBehavior
        from kivy.graphics import Color, Rectangle
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Kivy is required to launch the continuous view. Install it with `pip install kivy`."
        ) from exc

    alignment = load_alignment_paths(
        config.asr_path,
        align_csv_path=config.align_csv,
        align_db_path=config.align_db,
    )

    palette = {
        "anchor": (0.83, 0.94, 0.83, 1.0),
        "inter": (0.98, 0.95, 0.78, 1.0),
        "repeat": (0.99, 0.88, 0.70, 1.0),
        "selected": (0.70, 0.82, 1.0, 1.0),
    }

    class BlockLabel(ButtonBehavior, Label):
        def __init__(self, block: Block, role: str, on_select, **kwargs):
            super().__init__(**kwargs)
            self.block = block
            self.role = role
            self._on_select = on_select
            self.markup = False
            self.size_hint_y = None
            self.halign = "left"
            self.valign = "top"
            self.padding = (12, 8)
            self.text_size = (self.width - 24, None)
            self.bind(width=self._update_text_width)
            self.bind(texture_size=self._update_height)
            with self.canvas.before:
                self._bg_color = Color(*self._base_color())
                self._bg_rect = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self._refresh_rect, size=self._refresh_rect)

        def _update_text_width(self, *_):
            self.text_size = (self.width - 24, None)

        def _update_height(self, *_):
            self.height = self.texture_size[1] + 16

        def _refresh_rect(self, *_):
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size

        def _base_color(self):
            if self.block.possible_repeat:
                return palette["repeat"]
            return palette["anchor"] if self.block.kind == "anchor" else palette["inter"]

        def on_press(self):
            if self._on_select:
                self._on_select(self.block.block_id)

        def set_selected(self, active: bool):
            if active:
                self._bg_color.rgba = palette["selected"]
            else:
                self._bg_color.rgba = self._base_color()

    class ContinuousPane(BoxLayout):
        def __init__(self, alignment_data: AlignmentBlocks, **kwargs):
            super().__init__(orientation="vertical", **kwargs)
            self.alignment = alignment_data
            self.rows = config.rows
            self.play_callback = config.play_callback
            self._syncing = False
            self.block_labels: dict[int, tuple[BlockLabel, BlockLabel]] = {}
            self.current_block_id: int = 0

            self.info_label = Label(
                text="Selecciona un bloque para ver detalles",
                size_hint_y=None,
                height=32,
                halign="left",
                valign="middle",
                padding=(10, 4),
            )
            self.info_label.bind(size=lambda lbl, _: lbl.setter("text_size")(lbl, lbl.size))

            self.nav_panel = self._build_nav_panel()
            self.left_scroll, self.right_scroll = self._build_text_panels()
            body = BoxLayout()
            body.add_widget(self.nav_panel)
            body.add_widget(self.left_scroll)
            body.add_widget(self.right_scroll)

            self.add_widget(self._build_topbar())
            self.add_widget(body)
            self.add_widget(self.info_label)
            self.select_block(0)

        def _build_topbar(self) -> BoxLayout:
            bar = BoxLayout(size_hint_y=None, height=44, spacing=6, padding=6)
            bar.add_widget(Button(text="Siguiente interancla", size_hint_x=None, width=200,
                                  on_press=lambda *_: self.select_next_inter()))
            bar.add_widget(Button(text="Reproducir bloque actual", size_hint_x=None, width=240,
                                  on_press=lambda *_: self.play_current()))
            return bar

        def _build_nav_panel(self) -> BoxLayout:
            panel = BoxLayout(orientation="vertical", size_hint_x=0.22, padding=6, spacing=4)
            panel.add_widget(Label(text="Interanclas sospechosas", size_hint_y=None, height=26, bold=True))

            repeats = [b for b in self.alignment.blocks if b.kind == "inter" and b.possible_repeat]
            repeat_scroll = ScrollView()
            rep_list = GridLayout(cols=1, size_hint_y=None, spacing=4, padding=2)
            rep_list.bind(minimum_height=rep_list.setter("height"))
            for blk in repeats:
                btn = Button(
                    text=f"Bloque {blk.block_id} ({blk.repeat_source or 'repeat'})",
                    size_hint_y=None,
                    height=36,
                    on_press=lambda _btn, bid=blk.block_id: self.select_block(bid),
                )
                rep_list.add_widget(btn)
            repeat_scroll.add_widget(rep_list)
            panel.add_widget(repeat_scroll)
            return panel

        def _build_text_panels(self):
            left_layout = GridLayout(cols=1, size_hint_y=None, padding=4, spacing=4)
            right_layout = GridLayout(cols=1, size_hint_y=None, padding=4, spacing=4)
            left_layout.bind(minimum_height=left_layout.setter("height"))
            right_layout.bind(minimum_height=right_layout.setter("height"))

            for blk in self.alignment.blocks:
                ref_lbl = BlockLabel(
                    blk,
                    "ref",
                    self.select_block,
                    text=blk.ref_text() or "(sin ref)",
                    size_hint_y=None,
                )
                asr_lbl = BlockLabel(
                    blk,
                    "asr",
                    self.select_block,
                    text=blk.asr_text() or "(sin asr)",
                    size_hint_y=None,
                )
                self.block_labels[blk.block_id] = (ref_lbl, asr_lbl)
                left_layout.add_widget(ref_lbl)
                right_layout.add_widget(asr_lbl)

            left_scroll = ScrollView(size_hint_x=0.39, bar_width=12)
            right_scroll = ScrollView(size_hint_x=0.39, bar_width=12)
            left_scroll.add_widget(left_layout)
            right_scroll.add_widget(right_layout)

            left_scroll.bind(scroll_y=lambda inst, val: self._sync_scroll(inst, val, right_scroll))
            right_scroll.bind(scroll_y=lambda inst, val: self._sync_scroll(inst, val, left_scroll))
            return left_scroll, right_scroll

        def _sync_scroll(self, source: ScrollView, value: float, target: ScrollView):
            if self._syncing:
                return
            self._syncing = True
            target.scroll_y = value
            self._syncing = False

        def select_block(self, block_id: int):
            if block_id not in self.block_labels:
                return
            self.current_block_id = block_id
            for bid, labels in self.block_labels.items():
                for lbl in labels:
                    lbl.set_selected(bid == block_id)
            blk = self.alignment.blocks[block_id]
            row_desc = ""
            if blk.row_ids:
                row_desc = f" filas {min(blk.row_ids)}..{max(blk.row_ids)}"
            ref_desc = blk.ref_range or ("-", "-")
            asr_desc = blk.asr_range or ("-", "-")
            repeat = f" ({blk.repeat_source})" if blk.possible_repeat else ""
            self.info_label.text = (
                f"Bloque {blk.block_id} [{blk.kind}] "
                f"ref {ref_desc[0]}..{ref_desc[1]} asr {asr_desc[0]}..{asr_desc[1]}"
                f"{row_desc}{repeat}"
            )
            self._scroll_to(block_id)

        def _scroll_to(self, block_id: int) -> None:
            labels = self.block_labels.get(block_id)
            if not labels:
                return
            self.left_scroll.scroll_to(labels[0], padding=10, animate=True)
            self.right_scroll.scroll_to(labels[1], padding=10, animate=True)

        def select_next_inter(self):
            start = self.current_block_id + 1
            candidates = [b.block_id for b in self.alignment.blocks if b.kind == "inter" and b.block_id >= start]
            if candidates:
                self.select_block(candidates[0])

        def play_current(self):
            blk = self.alignment.blocks[self.current_block_id]
            interval = self.alignment.block_to_time(blk, rows=self.rows)
            if not interval:
                self.info_label.text = "No hay timecodes para este bloque"
                return
            start, end = interval
            if self.play_callback:
                self.play_callback(start, end)
            else:
                self.info_label.text = f"Intervalo: {start:.2f}s -> {end:.2f}s (sin reproductor)"

    class ContinuousApp(App):
        def build(self):
            self.title = "QC Vista Continua"
            return ContinuousPane(alignment)

    ContinuousApp().run()


def _load_rows_from_json(path: Path) -> Sequence[Sequence]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:  # pragma: no cover - best-effort loader
        print(f"[kivy view] no se pudieron leer filas de {path}: {exc}")
    return []


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Lanza la vista continua Kivy usando archivos de alineacion.")
    parser.add_argument("--asr", required=True, help="Ruta al TXT/CSV de ASR usado en QC (usa .align.csv/.align.db).")
    parser.add_argument("--audio", help="Ruta al audio para reproducir bloques.")
    parser.add_argument("--align-csv", help="Ruta alternativa al .align.csv.")
    parser.add_argument("--align-db", help="Ruta alternativa al .align.db.")
    parser.add_argument("--rows-json", help="Archivo .qc.json para recuperar filas y timecodes.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows: Optional[Sequence[Sequence]] = None
    if args.rows_json:
        rows_path = Path(args.rows_json)
        if rows_path.exists():
            rows = _load_rows_from_json(rows_path)
    cfg = ContinuousConfig(
        asr_path=Path(args.asr),
        audio_path=args.audio,
        align_csv=Path(args.align_csv) if args.align_csv else None,
        align_db=Path(args.align_db) if args.align_db else None,
        rows=rows,
    )
    launch_kivy_view(cfg)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
