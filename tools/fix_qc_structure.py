import json
from pathlib import Path
import sys


EXPECTED_COLS = ["ID", "flag", "OK", "AI", "WER", "tc", "Original", "ASR"]


def parse_hms_to_seconds(hms: str) -> float | None:
    try:
        parts = str(hms).strip().split(":")
        if len(parts) != 3:
            return None
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def fix_row(r: list) -> list:
    # Ensure list of 8 elements
    vals = list(r) + [""] * (8 - len(r))

    # If last two look like (WER numeric, HH:MM:SS), move them to (WER, tc)
    # and clear Original/ASR
    wer_candidate = vals[6]
    tc_candidate = vals[7]

    # Detect numeric wer in col 6
    def _is_number(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    moved = False
    if _is_number(wer_candidate):
        sec = parse_hms_to_seconds(tc_candidate)
        if sec is not None:
            # Place WER as string with one decimal
            vals[4] = f"{float(wer_candidate):.1f}"
            vals[5] = f"{sec:.2f}"
            # Clear text cols
            vals[6] = ""
            vals[7] = ""
            moved = True

    # Coerce basic types and lengths
    # ID int
    try:
        vals[0] = int(vals[0])
    except Exception:
        vals[0] = 0
    # flag/OK/AI strings
    for idx in (1, 2, 3):
        vals[idx] = "" if vals[idx] is None else str(vals[idx])
    # WER string with one decimal
    if vals[4] is None or vals[4] == "":
        vals[4] = ""
    else:
        try:
            vals[4] = f"{float(vals[4]):.1f}"
        except Exception:
            vals[4] = str(vals[4])
    # tc numeric string seconds
    if vals[5] is None or vals[5] == "":
        vals[5] = "0.0"
    else:
        try:
            # if already like HH:MM:SS, convert
            sec2 = parse_hms_to_seconds(vals[5])
            if sec2 is not None:
                vals[5] = f"{sec2:.2f}"
            else:
                vals[5] = f"{float(vals[5]):.2f}"
        except Exception:
            vals[5] = "0.0"
    # Text columns
    vals[6] = "" if vals[6] is None else str(vals[6])
    vals[7] = "" if vals[7] is None else str(vals[7])

    return vals[:8]


def fix_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(data, list):
        raise SystemExit("QC JSON should be a list of rows")
    fixed = [fix_row(r) for r in data]
    backup = path.with_suffix(path.suffix + ".bakfix")
    if not backup.exists():
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"Fixed: {path}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/fix_qc_structure.py <path_to_qc.json>")
        raise SystemExit(2)
    fix_file(Path(sys.argv[1]))

