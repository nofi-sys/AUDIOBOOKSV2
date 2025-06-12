from __future__ import annotations

"""Automatic AI review of QC rows using OpenAI's o3 models."""

from pathlib import Path
from typing import List
import json
import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
if os.environ.get("AI_REVIEW_DEBUG"):
    logging.basicConfig(level=logging.INFO)

from openai import OpenAI

MODEL = "o3-2025-04-16"

_client_instance: OpenAI | None = None


def _client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        key_present = bool(os.getenv("OPENAI_API_KEY"))
        if not key_present:
            logger.info("OPENAI_API_KEY not found in environment")
        else:
            logger.info("OPENAI_API_KEY loaded")
        _client_instance = OpenAI()  # API key from env vars
    return _client_instance


def load_prompt(path: str = "prompt.txt") -> str:
    try:
        return Path(path).read_text(encoding="utf8")
    except Exception:
        return DEFAULT_PROMPT


DEFAULT_PROMPT = """You are an audiobook QA assistant. Compare the ORIGINAL line with the ASR line.
Accept differences in punctuation, accents, or abbreviations
(e.g. \"dr.\" vs \"doctor\", \"1º\" vs \"primero\"). Ignore case.

If ASR faithfully renders the meaning: respond \"ok\".
If clearly wrong: respond \"mal\".
If uncertain or garbled: respond \"dudoso\".

Respond with **only** one of those words, nothing else.
"""


def ai_verdict(original: str, asr: str, base_prompt: str | None = None) -> str:
    prompt = base_prompt or DEFAULT_PROMPT
    logger.info("Sending to OpenAI: %s | %s", original, asr)
    resp = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"ORIGINAL:\n{original}\n\nASR:\n{asr}"},
        ],
        max_tokens=1,
        temperature=0,
    )
    word = resp.choices[0].message.content.strip().lower()
    if word not in {"ok", "mal", "dudoso"}:
        logger.info("Unexpected response: %s", word)
        return "dudoso"
    return word


def review_row(row: List, base_prompt: str | None = None) -> str:
    """Annotate a single QC row using OpenAI."""
    logger.info("Reviewing single row")
    verdict = ai_verdict(str(row[5]), str(row[6]), base_prompt)
    if verdict not in {"ok", "mal", "dudoso"}:
        verdict = "dudoso"
    if len(row) == 6:
        row.insert(2, "")
    if len(row) == 7:
        row.insert(3, verdict)
    else:
        row[3] = verdict
    if verdict == "ok":
        row[2] = "OK"
    return verdict


def review_file(qc_json: str, prompt_path: str = "prompt.txt") -> None:
    logger.info("Loading QC file: %s", qc_json)
    rows: List[List] = json.loads(Path(qc_json).read_text(encoding="utf8"))
    prompt = load_prompt(prompt_path)

    sent = 0
    approved = 0

    for row in rows:
        tick = row[1]
        ok = row[2] if len(row) > 2 else ""
        if tick or ok:
            continue
        sent += 1
        verdict = ai_verdict(str(row[5]), str(row[6]), prompt)
        if verdict not in {"ok", "mal", "dudoso"}:
            verdict = "dudoso"
        if len(row) == 6:
            row.insert(2, "")
        row.insert(3, verdict)
        if verdict == "ok":
            row[2] = "OK"
            approved += 1

    Path(qc_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf8")
    logger.info("Approved %s / Remaining %s", approved, sent - approved)
    return approved, sent - approved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch review QC JSON using o3")
    parser.add_argument("file", help="QC JSON file")
    parser.add_argument("--prompt", default="prompt.txt", help="Prompt file path")
    args = parser.parse_args()
    a, b = review_file(args.file, args.prompt)
    logger.info("Auto-approved %s / Remaining %s", a, b)
    print(f"Auto-approved {a} / Remaining {b}")
