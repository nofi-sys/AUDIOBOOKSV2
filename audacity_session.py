from __future__ import annotations

from pathlib import Path
import subprocess
import shutil


class AudacityLabelSession:
    """Manage an Audacity label track for a given audio file."""

    def __init__(self, audio_path: str) -> None:
        self.audio_path = Path(audio_path).resolve()
        self.label_path = self.audio_path.with_suffix(".audacity.txt")
        self.markers: list[tuple[float, float, str]] = []
        if self.label_path.exists():
            for line in self.label_path.read_text(encoding="utf8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                    except ValueError:
                        continue
                    label = parts[2] if len(parts) > 2 else ""
                    self.markers.append((start, end, label))

    def add_marker(self, time_sec: float, label: str = "") -> None:
        """Add a label marker at ``time_sec`` (in seconds)."""
        self.markers.append((time_sec, time_sec, label))
        self.save()

    def save(self) -> None:
        """Write current markers to the label file."""
        with self.label_path.open("w", encoding="utf8") as fh:
            for start, end, label in self.markers:
                fh.write(f"{start:.3f}\t{end:.3f}\t{label}\n")

    def open(self) -> None:
        """Launch Audacity with the audio and label track if available."""
        exe = shutil.which("audacity")
        if not exe:
            return
        subprocess.Popen([exe, str(self.audio_path), str(self.label_path)])
