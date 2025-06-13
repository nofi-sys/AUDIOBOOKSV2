from __future__ import annotations

"""Automatic AI review of QC rows using OpenAI's o3 models."""
from pathlib import Path
from typing import List
import json
import os
import logging
import time

from dotenv import load_dotenv
load_dotenv()

# Enable debug logging if environment variable set
logger = logging.getLogger(__name__)
if os.getenv("AI_REVIEW_DEBUG", "").lower() in ("1","true","yes"):
    logging.basicConfig(level=logging.INFO)

from openai import (
    OpenAI,
    APIStatusError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    OpenAIError,
)

# Use o3 model family
MODEL = "o3"
_client_instance: OpenAI | None = None
# Global flag to allow cancelling a long batch review
_stop_review = False


def stop_review() -> None:
    """Signal any running :func:`review_file` loop to exit early."""
    global _stop_review
    _stop_review = True

def _client() -> OpenAI:
    """Singleton OpenAI client using env var for key."""
    global _client_instance
    if _client_instance is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            logger.error("OPENAI_API_KEY not found in environment")
        else:
            logger.info("OPENAI_API_KEY loaded")
        _client_instance = OpenAI()
    return _client_instance


def _chat_with_backoff(**kwargs):
    """Call chat.completions.create with basic exponential backoff."""
    delay = 1.0
    for attempt in range(5):
        try:
            return _client().chat.completions.create(**kwargs)
        except (
            RateLimitError,
            APIStatusError,
            APIConnectionError,
            APITimeoutError,
            OpenAIError,
        ) as exc:
            status = getattr(exc, "status_code", 0)
            if status == 429 or status >= 500 or status == 0:
                logger.warning(
                    "OpenAI error %s on attempt %s, retrying in %.1fs",
                    exc,
                    attempt+1,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            raise
    # Final attempt
    return _client().chat.completions.create(**kwargs)


def load_prompt(path: str = "prompt.txt") -> str:
    """Load user prompt from file, fallback to default."""
    try:
        return Path(path).read_text(encoding="utf8")
    except Exception:
        logger.info("Using built-in prompt; failed to read %s", path)
        return DEFAULT_PROMPT

# Default instruction prompt
DEFAULT_PROMPT = """
You are an audiobook QA assistant. Your job is to compare an ORIGINAL sentence (the correct text from the book) with an ASR sentence (automatic speech-to-text transcription, known to be phonetically imperfect).

Your ONLY goal is to detect clear AUDIO READING or EDITING ERRORS, such as:

Entire words or phrases clearly omitted.

Entire words or phrases clearly repeated by mistake.

Completely different words clearly added or read incorrectly, significantly changing the meaning.

DO NOT consider punctuation, accents, capitalization, spelling errors, phonetic variations of proper names or phonetic rendering that dowesn't make sense, or transcription inaccuracies as mistakes.

Evaluation criteria:

If the ASR line does NOT show clear evidence of reading or editing errors (as described above), respond: ok

If the ASR line shows clear evidence of reading or editing errors, respond: mal

Respond EXACTLY with one word, without explanations or punctuation:

ok

mal
"""
#This is a testing phase: if you respond "mal" or "dudoso", provide a brief explanation of the specific reason for your assessment.
#Respond clearly with one of these words: ok, mal, or dudoso, followed by a brief explanation when necessary.

def ai_verdict(
    original: str,
    asr: str,
    base_prompt: str | None = None,
    return_feedback: bool = False,
) -> str | tuple[str, str]:
    """Send a single comparison request and return the verdict.

    If ``return_feedback`` is ``True`` the full response text from the model is
    also returned as a second element of the tuple.
    """
    prompt = base_prompt or DEFAULT_PROMPT
    logger.info("AI review request ORIGINAL=%s | ASR=%s", original, asr)
    # Debug prints
    print(f"DEBUG ai_verdict: ORIGINAL={original}")
    print(f"DEBUG ai_verdict: ASR={asr}")
    resp = _chat_with_backoff(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"ORIGINAL:\n{original}\n\nASR:\n{asr}"},
        ],
        max_completion_tokens=2000,
        stop=None,
    )
    # Debug full response
    print("DEBUG: full response:", resp)
    content = resp.choices[0].message.content
    # Debug raw content
    print("DEBUG: raw verdict repr:", repr(content))
    trimmed = content.strip()
    word = trimmed.split()[0].lower() if trimmed else ""
    if word not in {"ok", "mal", "dudoso"}:
        logger.warning(
            "Unexpected AI response '%s', defaulting to dudoso", trimmed
        )
        word = "dudoso"
    if return_feedback:
        return word, trimmed
    return word


def review_row(row: List, base_prompt: str | None = None) -> str:
    """Annotate a single QC row with AI verdict."""
    orig, asr = row[-2], row[-1]
    verdict = ai_verdict(str(orig), str(asr), base_prompt)
    if verdict not in {"ok", "mal", "dudoso"}:
        verdict = "dudoso"
    # Insert into row preserving structure
    # Row formats: [ID, tick, OK?, WER, dur, original, asr]
    if len(row) == 6:
        row.insert(2, "")         # OK column
        row.insert(3, verdict)      # AI verdict column
    elif len(row) == 7:
        row.insert(3, verdict)
    else:
        row[3] = verdict
    if verdict == "ok":
        row[2] = "OK"
    return verdict


def review_row_feedback(row: List, base_prompt: str | None = None) -> tuple[str, str]:
    """Like :func:`review_row` but also return the model feedback text."""

    orig, asr = row[-2], row[-1]
    verdict, feedback = ai_verdict(
        str(orig),
        str(asr),
        base_prompt,
        return_feedback=True,
    )
    if verdict not in {"ok", "mal", "dudoso"}:
        verdict = "dudoso"
    if len(row) == 6:
        row.insert(2, "")
        row.insert(3, verdict)
    elif len(row) == 7:
        row.insert(3, verdict)
    else:
        row[3] = verdict
    if verdict == "ok":
        row[2] = "OK"
    return verdict, feedback


def review_file(qc_json: str, prompt_path: str = "prompt.txt") -> tuple[int,int]:
    """Batch review QC JSON file, auto-approve lines marked ok."""
    global _stop_review
    _stop_review = False
    rows = json.loads(Path(qc_json).read_text(encoding="utf8"))
    bak = Path(qc_json + ".bak")
    if not bak.exists(): bak.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    prompt = load_prompt(prompt_path)
    sent = approved = 0
    for i, row in enumerate(rows):
        if _stop_review:
            break
        tick = row[1] if len(row) > 1 else ""
        ok = row[2] if len(row) >= 7 else ""
        ai = row[3] if len(row) >= 8 else ""
        if tick == "✅" or ok.lower() == "ok" or ai.lower() == "ok":
            continue
        sent += 1
        verdict = ai_verdict(str(row[-2]), str(row[-1]), prompt)
        if verdict not in {"ok", "mal", "dudoso"}:
            verdict = "dudoso"
        # Insert verdict column
        if len(row) == 6:
            row.insert(2,"")
            row.insert(3, verdict)
        elif len(row)==7:
            row.insert(3, verdict)
        else:
            row[3]=verdict
        if verdict=="ok":
            row[2]="OK"
            approved+=1
        # Save update
        Path(qc_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    logger.info("Approved %d / Remaining %d", approved, sent-approved)
    return approved, sent-approved


def review_file_feedback(
    qc_json: str, prompt_path: str = "prompt.txt"
) -> tuple[int, int, List[str]]:
    """Batch review returning feedback strings for each processed row."""

    global _stop_review
    _stop_review = False
    rows = json.loads(Path(qc_json).read_text(encoding="utf8"))
    bak = Path(qc_json + ".bak")
    if not bak.exists():
        bak.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    prompt = load_prompt(prompt_path)
    sent = approved = 0
    feedback: List[str] = []
    for i, row in enumerate(rows):
        if _stop_review:
            break
        tick = row[1] if len(row) > 1 else ""
        ok = row[2] if len(row) >= 7 else ""
        ai = row[3] if len(row) >= 8 else ""
        if tick == "✅" or ok.lower() == "ok" or ai.lower() == "ok":
            continue
        sent += 1
        verdict, fb = ai_verdict(
            str(row[-2]),
            str(row[-1]),
            prompt,
            return_feedback=True,
        )
        feedback.append(fb)
        if verdict not in {"ok", "mal", "dudoso"}:
            verdict = "dudoso"
        if len(row) == 6:
            row.insert(2, "")
            row.insert(3, verdict)
        elif len(row) == 7:
            row.insert(3, verdict)
        else:
            row[3] = verdict
        if verdict == "ok":
            row[2] = "OK"
            approved += 1
        Path(qc_json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
    logger.info("Approved %d / Remaining %d", approved, sent - approved)
    return approved, sent - approved, feedback


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch review QC JSON using o3 model")
    parser.add_argument("file", help="QC JSON file path")
    parser.add_argument("--prompt", default="prompt.txt")
    args = parser.parse_args()
    a,b = review_file(args.file, args.prompt)
    print(f"Auto-approved {a} / Remaining {b}")
