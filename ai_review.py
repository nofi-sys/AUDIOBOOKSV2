from __future__ import annotations

"""Automatic AI review of QC rows using OpenAI's o3 models."""
from pathlib import Path
from typing import List, Callable
import json
import os
import logging
import time
import threading

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIStatusError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    OpenAIError,
    BadRequestError,
)

# Enable debug logging if environment variable set
logger = logging.getLogger(__name__)
if os.getenv("AI_REVIEW_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.INFO)

load_dotenv()

# Use o3 model family
MODEL_DEFAULT = "gpt-5"
_client_instance: OpenAI | None = None
# Global flag to allow cancelling a long batch review
_stop_review = False

# Maximum number of OpenAI requests per batch review
MAX_MESSAGES = int(os.getenv("AI_REVIEW_MAX_MESSAGES", "100"))
ROW_TIMEOUT_SEC = int(os.getenv("AI_REVIEW_ROW_TIMEOUT_SEC", "300"))  # 5 min default


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
                    attempt + 1,
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


def _ai_verdict_with_timeout(
    original: str, asr: str, prompt: str | None, timeout_sec: int, model: str | None = None
) -> str:
    """Run ai_verdict in a helper thread and enforce a timeout.

    Returns the verdict string or raises TimeoutError/propagates exceptions.
    """
    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def _runner():
        try:
            result["v"] = ai_verdict(original, asr, prompt, model=model)
        except BaseException as exc:  # noqa: BLE001
            error["e"] = exc

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    if th.is_alive():
        # Leave the worker thread to finish in the background; we will ignore it.
        raise TimeoutError("ai_verdict timed out")
    if error:
        exc = error["e"]
        raise exc
    return result.get("v", "dudoso")


def _mark_error(row: List) -> None:
    """Insert or update the AI verdict column with 'error'."""
    if len(row) == 6:
        row.insert(2, "")
        row.insert(3, "error")
    elif len(row) == 7:
        row.insert(3, "error")
    else:
        row[3] = "error"


# Default instruction prompt
DEFAULT_PROMPT = """
You are an audiobook QA assistant. Your job is to compare an ORIGINAL sentence
(the correct text from the book) with an ASR sentence (automatic speech-to-text
transcription, known to be phonetically imperfect).

Your ONLY goal is to detect clear AUDIO READING or EDITING ERRORS that
significantly affect the meaning, such as:

- Entire words or phrases clearly omitted.
- Entire words or phrases clearly repeated by mistake, causing confusion.
- Completely different words clearly added or read incorrectly, substantially
  changing the meaning of the sentence.

DO NOT consider the following as mistakes:

- Punctuation, accents, capitalization, spelling errors. Non standard
  characters (like ||)
- Minor phonetic variations, especially in proper names or foreign words.
- Slight repetitions or brief pauses if they do not significantly alter the
  sentence's meaning.
- Transcription inaccuracies that don't significantly impact understanding.

Evaluation criteria:

- If the ASR line does NOT show clear evidence of unacceptable reading or
  editing errors (as described above), respond exactly with: ok
- If the ASR line shows clear evidence of unacceptable reading or editing
  errors, respond exactly with: mal

Respond EXACTLY with one word, without explanations or punctuation:

ok

mal

"""

# Extra prompt for re-review scoring from 1 (mal) to 5 (ok)
REREVIEW_PROMPT = (
    DEFAULT_PROMPT
    + "\n\nAfter your assessment respond ONLY with a single number from 1 to 5 "
    + "where 1 means mal and 5 means ok."
)

# Prompt used when re-transcribing problematic lines. The ASR text is provided
# line by line and numbered as -1, 0 and 1. Line 0 corresponds to the new
# transcription of the target segment. Lines -1 and 1 contain surrounding
# context. The model must judge ONLY line 0 against the ORIGINAL text.
RETRANS_PROMPT = (
    DEFAULT_PROMPT
    + "\n\nThe ASR text is numbered by lines as -1, 0 and 1. "
    + "Line 0 must be compared with the ORIGINAL text. The other lines are"
    + " context."
)
# # This is a testing phase: if you respond "mal" or "dudoso", provide a brief
# # explanation of the specific reason for your assessment.
# # Respond clearly with one of these words: ok, mal, or dudoso, followed by a
# # brief explanation when necessary.

CORRECTION_PROMPT = """
You are an expert audio editor. Your task is to correct an ASR (Automatic
Speech Recognition) transcript based on a provided ORIGINAL script. The ASR
may contain phonetic mistakes, misinterpretations, or minor omissions.

Your goal is to produce a corrected version of the ASR text that is as
faithful as possible to the ORIGINAL script, while only using words that are
phonetically plausible from the audio. DO NOT invent information or add words
from the original script that were clearly not spoken in the ASR.

Analyze the ASR and the ORIGINAL text. Then, return ONLY the corrected ASR
text. Do not include any explanations, greetings, or extra formatting.

Example:
ORIGINAL: "The quick brown fox jumps over the lazy dog."
ASR: "the Glick brown fox dumps over the hazy dog"

Your response should be:
"the quick brown fox jumps over the hazy dog"

(Note: "hazy" is kept because it's phonetically similar and could be a valid
interpretation of the audio, whereas "Glick" and "dumps" are clear errors
corrected to match the original script.)

Return only the corrected text.
"""

SUPERVISION_PROMPT = """
You are a QA supervisor for an audiobook transcription correction system.
Your task is to compare an original ASR transcription (ASR_V1) with a corrected version (ASR_V2) generated by another AI.

ASR_V1 is a raw transcription and may contain phonetic errors.
ASR_V2 is the proposed correction.

Your ONLY job is to determine if ASR_V2 is a *phonetically plausible* correction of ASR_V1. Ask yourself: "Could a human have said what's in ASR_V2, and an imperfect transcription system have written ASR_V1?"

- If the correction is plausible (e.g., "Glick" corrected to "quick", "dumps" to "jumps"), respond EXACTLY with: **plausible**
- If the correction seems unlikely or invented, and does not seem to be a phonetically similar equivalent to what was originally transcribed, respond EXACTLY with: **implausible**

Do not consider the original book script. Only compare ASR_V1 and ASR_V2. Respond with a single word.

Example 1:
ASR_V1: "the Glick brown fox dumps over the hazy dog"
ASR_V2: "the quick brown fox jumps over the hazy dog"
Your response: plausible

Example 2:
ASR_V1: "he went to the store"
ASR_V2: "he navigated to the retail establishment"
Your response: implausible

Example 3:
ASR_V1: "she said hello"
ASR_V2: "she said hello world"
Your response: implausible (a word was added without phonetic basis in V1)
"""


def _get_ai_correction(original: str, asr: str, model: str | None = None) -> str:
    """Send a correction request and return the corrected ASR text."""
    logger.info("AI correction request ORIGINAL=%s | ASR=%s", original, asr)
    current_model = model or MODEL_DEFAULT
    resp = _chat_with_backoff(
        model=current_model,
        messages=[
            {"role": "system", "content": CORRECTION_PROMPT},
            {"role": "user", "content": f"ORIGINAL:\n{original}\n\nASR:\n{asr}"},
        ],
        max_completion_tokens=2000,
        stop=None,
    )
    corrected_text = resp.choices[0].message.content.strip()
    logger.info("AI corrected ASR: %s", corrected_text)
    return corrected_text


def _get_supervisor_verdict(asr_v1: str, asr_v2: str, model: str | None = None) -> str:
    """Call supervisor AI to check if a correction is plausible."""
    logger.info("AI supervision request ASR_V1=%s | ASR_V2=%s", asr_v1, asr_v2)
    current_model = model or MODEL_DEFAULT
    resp = _chat_with_backoff(
        model=current_model,
        messages=[
            {"role": "system", "content": SUPERVISION_PROMPT},
            {
                "role": "user",
                "content": f"ASR_V1:\n{asr_v1}\n\nASR_V2:\n{asr_v2}",
            },
        ],
        max_completion_tokens=10,
        stop=None,
    )
    verdict = resp.choices[0].message.content.strip().lower()
    logger.info("AI supervision verdict: %s", verdict)
    if "plausible" in verdict:
        return "plausible"
    return "implausible"


def correct_and_supervise_text(original: str, asr: str, model: str | None = None) -> tuple[str, str, str]:
    """
    Performs a 2-step correction:
    1. Generates a corrected version of the ASR.
    2. Asks a supervisor AI if the correction is plausible.
    Returns (final_asr, supervisor_verdict, proposed_correction).
    """
    proposed_correction = _get_ai_correction(original, asr, model=model)

    if proposed_correction.strip() == asr.strip():
        return asr, "no_change", asr

    supervisor_verdict = _get_supervisor_verdict(asr, proposed_correction, model=model)

    final_asr = proposed_correction if supervisor_verdict == "plausible" else asr
    return final_asr, supervisor_verdict, proposed_correction


ADVANCED_REVIEW_PROMPT = """
You are a senior QA analyst for audiobooks. Your task is to re-evaluate a transcription row that was previously flagged as 'mal' (bad). You will be given context from the previous and next rows. Your goal is to provide a more precise error classification.

**CONTEXT:**
You will receive three blocks of text: [ANTERIOR], [ACTUAL] (the one you must analyze), and [POSTERIOR]. Each block contains the original script text (ORIGINAL) and the automatic transcription (ASR).

**YOUR TASK:**
1.  Focus on the [ACTUAL] block.
2.  Use the [ANTERIOR] and [POSTERIOR] blocks to understand the context. Pay close attention to words that might be misplaced between rows.
3.  Based on your analysis, classify the error in the [ACTUAL] row using ONE of the following categories:
    *   `REPETICION`: The speaker repeated one or more words by mistake.
    *   `OMISION`: The speaker omitted one or more words from the original script.
    *   `ADICION`: The speaker added words not present in the original script.
    *   `ERROR_LECTURA`: The speaker read a word incorrectly, changing the meaning (e.g., read "casa" instead of "caza").
    *   `INSIGNIFICANTE`: A minor transcription error that doesn't affect meaning (e.g., "treinta" vs "30", punctuation, a minor phonetic mistake that is understandable).
    *   `DESALINEADO`: The error is clearly a misalignment. For example, a word from the end of the previous ASR belongs at the start of the current ASR, or vice versa. If you choose this, explain where the word should go.
    *   `OK`: After reviewing the context, you realize there is no significant error and the initial 'mal' flag was a false positive.

4.  Provide a brief, one-sentence explanation for your verdict.

**RESPONSE FORMAT:**
Respond with a single line in the format: `VERDICT: <category> | COMMENT: <your explanation>`

**Example Response:**
`VERDICT: DESALINEADO | COMMENT: La última palabra del ASR anterior ('casa') pertenece al inicio del ASR actual.`

---
Now, analyze the following context:
"""

def get_advanced_review_verdict(context: dict) -> tuple[str, str]:
    """
    Performs an advanced, contextual AI review.
    Returns a tuple of (verdict, comment).
    """
    logger.info("Performing advanced AI review with context: %s", context)

    # Build the user prompt string from the context dictionary
    user_prompt = ""
    if "previous" in context:
        user_prompt += "[ANTERIOR]\n"
        user_prompt += f"- ORIGINAL: {context['previous']['original']}\n"
        user_prompt += f"- ASR: {context['previous']['asr']}\n\n"

    user_prompt += "[ACTUAL]\n"
    user_prompt += f"- ORIGINAL: {context['current']['original']}\n"
    user_prompt += f"- ASR: {context['current']['asr']}\n\n"

    if "next" in context:
        user_prompt += "[POSTERIOR]\n"
        user_prompt += f"- ORIGINAL: {context['next']['original']}\n"
        user_prompt += f"- ASR: {context['next']['asr']}\n"

    full_prompt = ADVANCED_REVIEW_PROMPT + user_prompt

    # Always use the best model for this advanced review
    resp = _chat_with_backoff(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a senior QA analyst for audiobooks."},
            {"role": "user", "content": full_prompt},
        ],
        max_completion_tokens=100,
        stop=None,
    )

    response_text = resp.choices[0].message.content.strip()

    # Parse the response
    try:
        verdict_part, comment_part = response_text.split("|", 1)
        verdict = verdict_part.replace("VERDICT:", "").strip().upper()
        comment = comment_part.replace("COMMENT:", "").strip()

        valid_verdicts = ["REPETICION", "OMISION", "ADICION", "ERROR_LECTURA", "INSIGNIFICANTE", "DESALINEADO", "OK"]
        if verdict not in valid_verdicts:
            # If the model didn't follow instructions, use a default and return its full response as comment.
            return "DUDOSO", response_text

        return verdict, comment
    except ValueError:
        # The response was not in the expected format
        logger.warning(f"Unexpected advanced AI response format: '{response_text}'")
        return "DUDOSO", response_text


def ai_verdict(
    original: str,
    asr: str,
    base_prompt: str | None = None,
    return_feedback: bool = False,
    model: str | None = None,
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
    current_model = model or MODEL_DEFAULT
    resp = _chat_with_backoff(
        model=current_model,
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
    if word not in {"ok", "mal", "dudoso", "error"}:
        logger.warning("Unexpected AI response '%s', defaulting to dudoso", trimmed)
        word = "dudoso"
    if return_feedback:
        return word, trimmed
    return word


def review_row(row: List, base_prompt: str | None = None, model: str | None = None) -> str:
    """Annotate a single QC row with AI verdict."""
    orig, asr = row[-2], row[-1]
    try:
        verdict = ai_verdict(str(orig), str(asr), base_prompt, model=model)
    except BadRequestError as exc:
        if "max_tokens" in str(exc) or "model output limit" in str(exc):
            _mark_error(row)
            return "error"
        raise
    if verdict not in {"ok", "mal", "dudoso", "error"}:
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


def review_row_feedback(row: List, base_prompt: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Like :func:`review_row` but also return the model feedback text."""

    orig, asr = row[-2], row[-1]
    try:
        verdict, feedback = ai_verdict(
            str(orig),
            str(asr),
            base_prompt,
            return_feedback=True,
            model=model,
        )
    except BadRequestError as exc:
        if "max_tokens" in str(exc) or "model output limit" in str(exc):
            _mark_error(row)
            return "error", ""
        raise
    if verdict not in {"ok", "mal", "dudoso", "error"}:
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


def ai_score(
    original: str,
    asr: str,
    base_prompt: str | None = None,
    return_feedback: bool = False,
    model: str | None = None,
) -> str | tuple[str, str]:
    """Send a single re-review request and return a 1-5 score."""

    prompt = base_prompt or REREVIEW_PROMPT
    logger.info("AI score request ORIGINAL=%s | ASR=%s", original, asr)
    current_model = model or MODEL_DEFAULT
    resp = _chat_with_backoff(
        model=current_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"ORIGINAL:\n{original}\n\nASR:\n{asr}"},
        ],
        max_completion_tokens=2000,
        stop=None,
    )
    content = resp.choices[0].message.content
    trimmed = content.strip()
    rating = trimmed.split()[0]
    if rating not in {"1", "2", "3", "4", "5"}:
        logger.warning("Unexpected AI score '%s', defaulting to 3", trimmed)
        rating = "3"
    if return_feedback:
        return rating, trimmed
    return rating


def score_row(row: List, base_prompt: str | None = None, model: str | None = None) -> str:
    """Return 1-5 score for a single QC row."""

    orig, asr = row[-2], row[-1]
    try:
        rating = ai_score(str(orig), str(asr), base_prompt, model=model)
    except BadRequestError as exc:
        if "max_tokens" in str(exc) or "model output limit" in str(exc):
            _mark_error(row)
            return "0"
        raise
    return rating


def review_file(
    qc_json: str,
    prompt_path: str = "prompt.txt",
    limit: int | None = None,
    progress_callback: Callable[[str, int, list], None] | None = None,
    model: str | None = None,
) -> tuple[int, int]:
    """Batch review QC JSON file, auto-approve lines marked ok.

    Parameters
    ----------
    qc_json:
        Path to the QC JSON file.
    prompt_path:
        Optional prompt file path.
    limit:
        Maximum number of requests to send. ``None`` uses ``MAX_MESSAGES``.
    progress_callback:
        Optional callable invoked as ``callback(stage, index, row)`` before
        (``stage='start'``) and after (``stage='done'``) each reviewed row.
    """
    global _stop_review
    _stop_review = False
    rows = json.loads(Path(qc_json).read_text(encoding="utf8"))
    bak = Path(qc_json + ".bak")
    if not bak.exists():
        bak.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
    prompt = load_prompt(prompt_path)
    sent = approved = 0
    current_model = model or MODEL_DEFAULT
    max_requests = limit if limit is not None else MAX_MESSAGES
    for i, row in enumerate(rows):
        if _stop_review or (max_requests and sent >= max_requests):
            break
        tick = row[1] if len(row) > 1 else ""
        ok = row[2] if len(row) >= 7 else ""
        ai = row[3] if len(row) >= 8 else ""
        # Skip rows already reviewed by AI regardless of verdict
        if tick == "✅" or ok.lower() == "ok" or ai:
            continue

        # Skip rows with empty Original or ASR text
        original_text = str(row[-2]).strip()
        asr_text = str(row[-1]).strip()
        if not original_text or not asr_text:
            logger.info(f"Skipping row {i+1} due to missing Original or ASR text.")
            continue

        if progress_callback:
            try:
                progress_callback("start", i, row)
            except Exception:
                logger.exception("progress_callback start failed")
        attempts = 0
        verdict = None  # type: ignore
        while True:
            # Check max requests budget before each attempt
            if max_requests and sent >= max_requests:
                break
            sent += 1
            try:
                verdict = _ai_verdict_with_timeout(
                    str(row[-2]), str(row[-1]), prompt, ROW_TIMEOUT_SEC, model=current_model
                )
            except TimeoutError:
                attempts += 1
                logger.warning(
                    "Row %d timed out after %ds (attempt %d)", i, ROW_TIMEOUT_SEC, attempts
                )
                if attempts >= 2:
                    # Stop processing; show popup and exit loop
                    try:
                        from tkinter import messagebox  # type: ignore

                        messagebox.showerror(
                            "AI Review detenido",
                            f"Se detuvo el AI Review por timeout repetido en la fila {i+1}.",
                        )
                    except Exception:
                        logger.exception("Failed to show error popup")
                    _stop_review = True
                    # Clean processing tag in GUI by sending 'done' without verdict
                    if progress_callback:
                        try:
                            progress_callback("done", i, row)
                        except Exception:
                            logger.exception("progress_callback done failed (timeout)")
                    break
                # Retry same row once
                continue
            except BadRequestError as exc:
                if "max_tokens" in str(exc) or "model output limit" in str(exc):
                    _mark_error(row)
                    Path(qc_json).write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf8",
                    )
                    # Inform GUI that this row ended processing
                    if progress_callback:
                        try:
                            progress_callback("done", i, row)
                        except Exception:
                            logger.exception("progress_callback done failed (bad request)")
                    verdict = "error"
                    break
                raise
            # Success
            break

        # If we broke due to stop signal or budget, exit outer loop
        if max_requests and sent >= max_requests and verdict is None:
            break
        if verdict is None:
            # No verdict produced (e.g., budget exhausted). Stop.
            break
        if verdict not in {"ok", "mal", "dudoso", "error"}:
            verdict = "dudoso"
        # Insert verdict column
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
        # Save update
        Path(qc_json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
        if progress_callback:
            try:
                progress_callback("done", i, row)
            except Exception:
                logger.exception("progress_callback done failed")
        if _stop_review or (max_requests and sent >= max_requests):
            break
    logger.info("Approved %d / Remaining %d", approved, sent - approved)
    return approved, sent - approved


def review_file_feedback(
    qc_json: str,
    prompt_path: str = "prompt.txt",
    limit: int | None = None,
    model: str | None = None,
) -> tuple[int, int, List[str]]:
    """Batch review returning feedback strings for each processed row.

    ``limit`` restricts how many requests are sent. ``None`` means use
    ``MAX_MESSAGES``.
    """

    global _stop_review
    _stop_review = False
    rows = json.loads(Path(qc_json).read_text(encoding="utf8"))
    bak = Path(qc_json + ".bak")
    if not bak.exists():
        bak.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    prompt = load_prompt(prompt_path)
    sent = approved = 0
    max_requests = limit if limit is not None else MAX_MESSAGES
    feedback: List[str] = []
    current_model = model or MODEL_DEFAULT
    for i, row in enumerate(rows):
        if _stop_review or (max_requests and sent >= max_requests):
            break
        tick = row[1] if len(row) > 1 else ""
        ok = row[2] if len(row) >= 7 else ""
        ai = row[3] if len(row) >= 8 else ""
        # Skip rows already reviewed by AI regardless of verdict
        if tick == "✅" or ok.lower() == "ok" or ai:
            continue
        sent += 1
        try:
            verdict, fb = ai_verdict(
                str(row[-2]),
                str(row[-1]),
                prompt,
                return_feedback=True,
                model=current_model,
            )
        except BadRequestError as exc:
            if "max_tokens" in str(exc) or "model output limit" in str(exc):
                _mark_error(row)
                Path(qc_json).write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2),
                    encoding="utf8",
                )
                feedback.append("")
                continue
            raise
        feedback.append(fb)
        if verdict not in {"ok", "mal", "dudoso", "error"}:
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of lines to review",
    )
    args = parser.parse_args()
    a, b = review_file(args.file, args.prompt, args.limit)
    print(f"Auto-approved {a} / Remaining {b}")