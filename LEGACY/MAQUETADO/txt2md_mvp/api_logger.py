import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

LOGS_DIR = Path("logs")
SESSION_LOG_FILE: Optional[Path] = None


def _sanitize_segment(value: str) -> str:
    """Return a filesystem-safe representation of a project/run identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "session"


def setup_logging() -> None:
    """Ensure the active log directory exists on disk."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def set_log_directory(path: Union[str, Path], session_name: Optional[str] = None) -> Path:
    """
    Override the base directory where API interactions are stored.

    If ``session_name`` is provided, a JSONL file with that name (including the date
    and hour) will aggregate every interaction for easier inspection.
    """
    global LOGS_DIR, SESSION_LOG_FILE
    LOGS_DIR = Path(path)
    setup_logging()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_segment(session_name) if session_name else timestamp
    SESSION_LOG_FILE = LOGS_DIR / f"{safe_name}.jsonl"
    try:
        SESSION_LOG_FILE.touch(exist_ok=True)
    except OSError:
        # Fall back to disabling the aggregate log if the file cannot be created
        SESSION_LOG_FILE = None

    return LOGS_DIR


def get_log_directory() -> Path:
    """Return the current directory used for API interaction logs."""
    return LOGS_DIR


def get_session_log_file() -> Optional[Path]:
    """Return the aggregate JSONL file currently in use, if configured."""
    return SESSION_LOG_FILE


def log_simplified_translation(log_path: Path, original: str, translated: str) -> None:
    """Append a simplified record of a translation to a text file."""
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write("--- ORIGINAL ---\n")
            f.write(original + "\n\n")
            f.write("--- TRADUCIDO ---\n")
            f.write(translated + "\n\n")
            f.write("=" * 20 + "\n\n")
    except Exception as exc:
        print(f"Failed to write simplified translation log: {exc}")


def _serialize_response_payload(response: Any) -> Any:
    """Convert the response object into something JSON serialisable."""
    if response is None:
        return None
    try:
        if hasattr(response, "model_dump_json"):
            return json.loads(response.model_dump_json())
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "json"):
            return response.json()
    except Exception:
        pass
    return str(response)


def log_interaction(
    model: str,
    prompt: Any,
    params: Dict[str, Any],
    response: Optional[Any] = None,
    error: Optional[Exception] = None,
) -> None:
    """
    Log a single API interaction both as an individual JSON file and into the
    running aggregate JSONL file for the active session.
    """
    setup_logging()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = LOGS_DIR / f"{timestamp}_{_sanitize_segment(model)}.json"

    log_entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "params": params,
        "prompt": prompt,
    }

    if response is not None:
        log_entry["response"] = _serialize_response_payload(response)
    if error is not None:
        log_entry["error"] = str(error)
        log_entry["error_type"] = type(error).__name__

    try:
        with log_file.open("w", encoding="utf-8") as fh:
            json.dump(log_entry, fh, indent=4, ensure_ascii=False)
    except Exception as exc:
        print(f"Failed to write log file: {exc}")

    if SESSION_LOG_FILE:
        try:
            with SESSION_LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False))
                fh.write(os.linesep)
        except Exception as exc:
            print(f"Failed to append aggregate log: {exc}")


# Ensure a default log directory is present when the module is imported
setup_logging()
