from __future__ import annotations

"""Automatic AI review of QC rows using OpenAI's gpt-5.1 models."""
from difflib import SequenceMatcher
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
from qc_utils import canonical_row
from text_utils import find_repeated_sequences, normalize

# Enable debug logging if environment variable set
logger = logging.getLogger(__name__)
if os.getenv("AI_REVIEW_DEBUG", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.INFO)

load_dotenv()

# Default GPT model family
MODEL_DEFAULT = "gpt-5.1"
_client_instance: OpenAI | None = None
# Global flag to allow cancelling a long batch review
_stop_review = False

# Maximum number of OpenAI requests per batch review
MAX_MESSAGES = int(os.getenv("AI_REVIEW_MAX_MESSAGES", "100"))
ROW_TIMEOUT_SEC = int(os.getenv("AI_REVIEW_ROW_TIMEOUT_SEC", "300"))  # 5 min default


def _to_float(val: str, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return default


WER_VETO_THRESHOLD = _to_float(os.getenv("AI_REVIEW_WER_VETO", "7.0"), 7.0)
MAX_INSERT_RUN_VETO = int(os.getenv("AI_REVIEW_MAX_INS_RUN_VETO", "2") or 2)
REPEAT_MIN_WORDS = int(os.getenv("AI_REVIEW_REPEAT_MIN_WORDS", "3") or 3)

ALLOWED_VERDICTS = {"ok", "mal", "error"}
SKIP_TICKS = {"バ.", "ƒo."}
TRACE_IO = os.getenv("AI_REVIEW_TRACE_IO", "").strip().lower() in {"1", "true", "yes"}


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


def _call_with_timeout(fn: Callable, timeout_sec: int, *args, **kwargs):
    """Run ``fn`` in a helper thread and enforce a timeout."""
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def _runner():
        try:
            result["v"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            error["e"] = exc

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    if th.is_alive():
        raise TimeoutError(f"{fn.__name__} timed out")
    if error:
        raise error["e"]
    return result.get("v")


def _mark_error(row: List) -> None:
    """Insert or update the AI verdict column with 'error'."""
    normalized = canonical_row(row)
    row[:] = normalized
    row[3] = "error"


PASS1_SYSTEM_PROMPT = """
Eres un corrector extremadamente estricto de audiolibros.
Tu tarea es detectar errores de lectura en voz alta comparando el ORIGINAL con la TRANSCRIPCION (ASR).
Ignora ortografia, tildes, mayusculas y puntuacion. Ignora variaciones minimas de estilo si el contenido es el mismo.
Debes proteger al editor: ante cualquier duda razonable, responde "mal".
""".strip()

PASS1_USER_TEMPLATE = """
ORIGINAL (guion):

```
{original}
```

TRANSCRIPCION (ASR):

```
{asr}
```

Instrucciones:
1. Imagina que escuchas al locutor leyendo el ORIGINAL.
2. Ignora diferencias de ortografia, tildes, mayusculas y puntuacion. Variantes minimas como "y"/"e" o "del"/"de el" son aceptables si el sentido es el mismo.
3. Marca "mal" si falta una parte del ORIGINAL, si el locutor repite algo que en el ORIGINAL aparece solo una vez, si añade informacion nueva o si cambia palabras por otras que alteran el sentido u orden.
4. Si tienes cualquier duda razonable sobre si la lectura respeta fielmente el ORIGINAL, responde "mal".
5. Marca "ok" solo si la lectura dice lo mismo que el ORIGINAL, sin omisiones ni repeticiones.

Responde solo con una de estas palabras en minusculas:
ok
mal
""".strip()

PASS2_SYSTEM_PROMPT = """
Eres un corrector de audiolibros muy estricto. Revisas con contexto para detectar repeticiones u omisiones entre frases consecutivas.
Ignora ortografia, tildes, mayusculas y puntuacion. Ante cualquier duda razonable, responde "mal".
""".strip()

PASS2_USER_TEMPLATE = """
[ANTERIOR]
ORIGINAL:
{prev_original}
ASR:
{prev_asr}

[ACTUAL]
ORIGINAL:
{original}
ASR:
{asr}

[POSTERIOR]
ORIGINAL:
{next_original}
ASR:
{next_asr}

Instrucciones:
1. Verifica si la lectura del bloque ACTUAL respeta el contenido y el orden del ORIGINAL sin omisiones ni repeticiones.
2. Usa el contexto (anterior y posterior) para detectar frases repetidas u omitidas entre lineas.
3. Ignora ortografia, tildes, mayusculas y puntuacion; importa solo la fidelidad auditiva.
4. Ante cualquier duda razonable, marca "mal".

Responde solo con una de estas palabras en minusculas:
ok
mal
""".strip()

# New standard prompts that enfatize errores de lectura reales,
# diferencias fonéticas y nombres propios mal transcriptos.
STANDARD_PASS1_SYSTEM_PROMPT = """
Eres un corrector de lecturas en voz alta para audiolibros.
Tu tarea es decidir si el LOCUTOR leyó correctamente el texto ORIGINAL comparándolo con la TRANSCRIPCION automática (ASR).
La ASR es ruidosa: puede tener faltas de ortografia, palabras inventadas o nombres propios mal escritos.
Antes de marcar un error debes imaginar cómo suenan en voz alta el ORIGINAL y la ASR y juzgar si podrían corresponder a la misma grabación.
Da prioridad a detectar REPETICIONES y OMISIONES GRAVES. Las diferencias menores de redacción no importan.
Debes proteger al editor: si después de pensar en la relación fonética sigues con una duda razonable sobre si el LOCUTOR respetó el ORIGINAL, responde "mal".
""".strip()

STANDARD_PASS1_USER_TEMPLATE = """
ORIGINAL (guion):

```
{original}
```

TRANSCRIPCION (ASR):

```
{asr}
```

Instrucciones:
1. Imagina que escuchas al locutor leyendo el ORIGINAL y luego imagina cómo sonaría alguien leyendo en voz alta la TRANSCRIPCION (ASR).
2. Ignora SIEMPRE: faltas de ortografia, tildes, mayusculas, uso de b/v/c/s/z, y pequeñas variaciones de estilo o sinónimos que no cambian el sentido.
3. Ten en cuenta que la ASR puede escribir muy mal los nombres propios y apellidos. Antes de concluir que el locutor se equivocó, compara mentalmente el sonido del nombre en el ORIGINAL y en la ASR. Si podrían sonar como la misma persona o lugar, asume que es error de transcripción, no de lectura.
4. Marca "mal" SOLO si, pensando en lo que se escucha en el audio, ves con alta probabilidad un error de lectura del LOCUTOR, por ejemplo:
   - REPETICION: una palabra o grupo de palabras repetidos por el locutor que en el ORIGINAL aparecen solo una vez (por ejemplo "la la casa"). Una sola palabra repetida ya es un error.
   - OMISION GRAVE: palabras o frases completas que faltan y cuya ausencia cambia claramente el significado o deja el texto incompleto.
   - ADICION RELEVANTE: palabras o frases nuevas que añaden información o cambian el mensaje del ORIGINAL.
   - CAMBIO FUERTE DE SENTIDO: números, fechas, nombres propios, negaciones ("no"), términos técnicos o palabras clave cambiados por otros que no son fonéticamente compatibles con el ORIGINAL.
5. Si la diferencia puede explicarse razonablemente como un error de la ASR (otra forma de escribir casi lo mismo que se oye), responde "ok".
6. Marca "ok" solo si, después de este análisis fonético, crees que el locutor dijo esencialmente lo mismo que el ORIGINAL, sin omisiones graves ni repeticiones.

Responde solo con una de estas palabras en minusculas:
ok
mal
""".strip()

STANDARD_PASS2_SYSTEM_PROMPT = """
Eres un corrector de audiolibros muy estricto que revisa con CONTEXTO para detectar principalmente repeticiones y omisiones graves entre frases consecutivas.
La TRANSCRIPCION (ASR) es automática y puede contener muchos errores de ortografia y nombres propios mal transcriptos.
Antes de concluir que el LOCUTOR se equivocó debes comparar fonéticamente el ORIGINAL y la ASR en los tres bloques (ANTERIOR, ACTUAL, POSTERIOR).
""".strip()

STANDARD_PASS2_USER_TEMPLATE = """
[ANTERIOR]
ORIGINAL:
{prev_original}
ASR:
{prev_asr}

[ACTUAL]
ORIGINAL:
{original}
ASR:
{asr}

[POSTERIOR]
ORIGINAL:
{next_original}
ASR:
{next_asr}

Instrucciones:
1. Concéntrate en el bloque ACTUAL y usa los bloques ANTERIOR y POSTERIOR solo como contexto.
2. Marca "mal" si, teniendo en cuenta el contexto, observas que el LOCUTOR:
   - repite una palabra o frase (una o más palabras) que en el ORIGINAL solo aparece una vez, o
   - omite una parte importante del ORIGINAL, o
   - mueve palabras o frases clave a otra línea de forma que el texto quede truncado o desordenado.
3. Ignora diferencias de ortografia, tildes, mayusculas y pequeñas variaciones de estilo que puedan explicarse como errores de ASR. Presta especial atención a los nombres propios: la ASR puede deformarlos, pero si suenan fonéticamente compatibles con el ORIGINAL, no los consideres error de lectura.
4. Si después de comparar fonéticamente sigues con una duda razonable sobre si la lectura respeta el ORIGINAL, marca "mal".
5. Marca "ok" solo si el contenido efectivo leído en ACTUAL coincide con el ORIGINAL, sin repeticiones ni omisiones graves.

Responde solo con una de estas palabras en minusculas:
ok
mal
""".strip()

# Default instruction prompt (used as system fallback)
DEFAULT_PROMPT = STANDARD_PASS1_SYSTEM_PROMPT

# Extra prompt for re-review scoring from 1 (mal) to 5 (ok)
REREVIEW_PROMPT = (
    STANDARD_PASS1_SYSTEM_PROMPT
    + "\n\nCalifica la fidelidad de la lectura con un numero del 1 al 5 (1=mal y 5=ok). "
    + "Responde solo con ese numero."
)

# Prompt used when re-transcribing problematic lines. The ASR text is provided
# line by line and numbered as -1, 0 and 1. Line 0 corresponds to the new
# transcription of the target segment. Lines -1 and 1 contain surrounding
# context. The model must judge ONLY line 0 against the ORIGINAL text.
RETRANS_PROMPT = (
    STANDARD_PASS1_SYSTEM_PROMPT
    + "\n\nThe ASR text is numbered by lines as -1, 0 and 1. "
    + "Line 0 must be compared with the ORIGINAL text. The other lines are context."
)

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
""".strip()

SUPERVISION_PROMPT = """
You are a QA supervisor for an audiobook transcription correction system.
Your task is to compare an original ASR transcription (ASR_V1) with a corrected version (ASR_V2) generated by another AI.

ASR_V1 is a raw transcription and may contain phonetic errors.
ASR_V2 is the proposed correction.

Your ONLY job is to determine if ASR_V2 is a *phonetically plausible* correction of ASR_V1. Ask yourself: "Could a human have said what's in ASR_V2, and an imperfect transcription system have written ASR_V1?"

- If the correction is plausible (e.g., "Glick" corrected to "quick", "dumps" to "jumps"), respond EXACTLY with: plausible
- If the correction seems unlikely or invented, and does not seem to be a phonetically similar equivalent to what was originally transcribed, respond EXACTLY with: implausible

Do not consider the original book script. Only compare ASR_V1 and ASR_V2. Respond with a single word.
""".strip()

ADVANCED_REVIEW_PROMPT = """
Eres un analista senior de QA para audiolibros. Tu tarea es volver a evaluar una fila de transcripción que fue marcada como 'mal'. Tendrás contexto de las filas anterior y posterior. Tu objetivo es clasificar con mayor precisión el tipo de error.

CONTEXTO:
Recibirás tres bloques de texto: [ANTERIOR], [ACTUAL] (el que debes analizar) y [POSTERIOR]. Cada bloque contiene el texto del guion (ORIGINAL) y la transcripción automática (ASR).

TEN EN CUENTA:
- La ASR es automática y puede tener muchas faltas de ortografía, palabras inventadas y nombres propios mal transcriptos.
- Antes de atribuir un error al LOCUTOR, imagina cómo sonarían en voz alta el ORIGINAL y la ASR y decide si podrían corresponder a la misma grabación.

TU TAREA:
1. Concéntrate en el bloque [ACTUAL].
2. Usa [ANTERIOR] y [POSTERIOR] para entender el contexto. Presta atención a palabras que se repiten o se desplazan entre filas.
3. Según tu análisis, clasifica el error en [ACTUAL] usando UNA de estas categorías:
   - REPETICION: El locutor repitió por error una palabra o grupo de palabras (una o más palabras) que en el ORIGINAL aparecen solo una vez.
   - OMISION: El locutor omitió una o más palabras del guion de forma que el sentido queda claramente incompleto o alterado.
   - ADICION: El locutor añadió palabras que no están en el guion y cambian el mensaje.
   - ERROR_LECTURA: El locutor leyó una palabra de forma claramente distinta, cambiando el significado (por ejemplo, otro número, otro nombre propio, otra negación).
   - INSIGNIFICANTE: Solo hay errores de ASR o diferencias mínimas que no afectan el significado (incluyendo errores en nombres propios que siguen siendo fonéticamente compatibles).
   - DESALINEADO: El problema principal es de alineación entre filas, no de lectura del locutor.
   - OK: Tras revisar el contexto, no hay un error significativo del locutor.

4. Proporciona una explicación breve, en una sola frase, para tu veredicto.

FORMATO DE RESPUESTA:
Responde con una sola línea en el formato: VERDICT: <categoria> | COMMENT: <tu explicación>
""".strip()

REPETITION_PROMPT = """
Eres un analista de QA especializado en audiolibros, enfocado únicamente en detectar repeticiones no intencionales del LOCUTOR.

CONTEXTO:
Recibirás tres bloques de texto: [ANTERIOR], [ACTUAL] y [POSTERIOR]. Cada uno contiene el guion (ORIGINAL) y la transcripción automática (ASR).

TEN EN CUENTA:
- La ASR es automática y puede contener errores graves de ortografía y nombres propios deformados.
- Antes de marcar una repetición, imagina cómo sonarían en voz alta el ORIGINAL y la ASR y decide si podrían ser dos transcripciones de la misma frase. Si la diferencia puede explicarse solo por error de ASR, no lo consideres un error del locutor.

TU TAREA:
1. Tu único objetivo es determinar si el LOCUTOR repitió de forma no intencional palabras o frases en el ASR del bloque [ACTUAL] que en el ORIGINAL aparecen una sola vez.
2. Considera repetición significativa tanto una sola palabra repetida ("la la") como grupos de varias palabras ("en la casa en la casa").
3. Usa el contexto de los bloques ANTERIOR y POSTERIOR para confirmar que se trata de una repetición real dentro de la misma zona del texto, y no de una estructura repetida del guion o un problema de alineación.
4. Según tu análisis, devuelve uno de estos veredictos para el bloque [ACTUAL]:
   - REPETICION: Detectas una repetición clara de una o más palabras que no está así en el guion ORIGINAL.
   - OK: No hay repeticiones significativas del locutor.

FORMATO DE RESPUESTA:
Responde con una sola línea en el formato: VERDICT: <categoria> | COMMENT: <explicación breve de la frase repetida>
""".strip()


def _normalize_verdict(text: str | None) -> str:
    word = ""
    if text:
        word = str(text).strip().split()[0].lower()
    if word == "dudoso":
        return "mal"
    if word in ALLOWED_VERDICTS:
        return word
    if text:
        logger.warning("Unexpected AI verdict '%s', defaulting to 'mal'", str(text).strip())
    return "mal"


def _trace_io(label: str, messages: list[dict], content: str | None, *, max_len: int = 4000) -> None:
    """Print prompt/response for debugging when trace flag is on."""

    def _fmt(val: object) -> str:
        try:
            s = str(val)
        except Exception:
            s = repr(val)
        if len(s) > max_len:
            s = s[:max_len] + "...[truncated]"
        return s.encode("utf-8", "backslashreplace").decode("utf-8", "replace")

    try:
        print(f"\n[TRACE {label}] >>> prompt")
        for msg in messages:
            role = msg.get("role", "?")
            body = _fmt(msg.get("content", ""))
            print(f"[{role}]\n{body}\n")
        print(f"[TRACE {label}] <<< response\n{_fmt(content)}\n")
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to trace AI IO: %s", exc)


def _normalized_tokens(text: str) -> list[str]:
    return normalize(text, strip_punct=True).split()


def _extract_text(val: object) -> str:
    """Return a safe text from possible dicts used in the GUI (corrected/asr_original)."""
    if isinstance(val, dict):
        for key in ("corrected", "asr_original", "original", "text"):
            if key in val and val[key]:
                return str(val[key])
    return "" if val is None else str(val)


def _max_insert_run(ref_tokens: list[str], asr_tokens: list[str]) -> int:
    """Approximate the longest run of inserted tokens in ASR vs reference."""
    if not asr_tokens:
        return 0
    matcher = SequenceMatcher(a=ref_tokens, b=asr_tokens, autojunk=False)
    max_run = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            max_run = max(max_run, j2 - j1)
        elif tag == "replace":
            extra = (j2 - j1) - (i2 - i1)
            if extra > 0:
                max_run = max(max_run, extra)
    return max_run


def _structure_signals(original: str, asr: str, wer_value) -> dict:
    ref_tokens = _normalized_tokens(original)
    asr_tokens = _normalized_tokens(asr)
    wer = None
    try:
        wer = float(wer_value)
    except Exception:
        wer = None
    repeats = find_repeated_sequences(
        asr,
        min_length=max(3, REPEAT_MIN_WORDS),
        max_length=40,
        similarity_threshold=85.0,
    )
    return {
        "wer": wer,
        "max_insert_run": _max_insert_run(ref_tokens, asr_tokens),
        "repeats": repeats,
        "has_tokens": bool(ref_tokens and asr_tokens),
    }


def _structure_is_clean(signals: dict) -> bool:
    wer_ok = signals.get("wer") is not None and signals["wer"] <= WER_VETO_THRESHOLD
    if not wer_ok:
        return False
    if not signals.get("has_tokens"):
        return False
    if signals.get("max_insert_run", 0) > MAX_INSERT_RUN_VETO:
        return False
    if signals.get("repeats"):
        return False
    return True


def _context_text(rows: list[list], idx: int) -> tuple[str, str]:
    if idx < 0 or idx >= len(rows):
        return "", ""
    row = canonical_row(rows[idx])
    rows[idx] = row
    original = row[6] if len(row) > 6 else ""
    asr = row[7] if len(row) > 7 else ""
    return str(original).strip(), str(asr).strip()


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


def get_advanced_review_verdict(context: dict, model: str | None = None, repetition_check: bool = False) -> tuple[str | None, str | None]:
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

    prompt_template = REPETITION_PROMPT if repetition_check else ADVANCED_REVIEW_PROMPT
    full_prompt = prompt_template + "\n\n" + user_prompt

    system_content = (
        "You are a specialized QA analyst for audiobooks, focused only on detecting unintentional repetitions."
        if repetition_check
        else "You are a senior QA analyst for audiobooks."
    )

    current_model = model or MODEL_DEFAULT
    resp = _chat_with_backoff(
        model=current_model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_prompt},
        ],
        max_completion_tokens=100,
        stop=None,
    )

    response_text = resp.choices[0].message.content.strip()

    if not response_text:
        logger.warning("Advanced AI review returned an empty response.")
        return None, None

    # Parse the response
    try:
        verdict_part, comment_part = response_text.split("|", 1)
        verdict = verdict_part.replace("VERDICT:", "").strip().upper()
        comment = comment_part.replace("COMMENT:", "").strip()

        valid_verdicts = [
            "REPETICION",
            "OMISION",
            "ADICION",
            "ERROR_LECTURA",
            "INSIGNIFICANTE",
            "DESALINEADO",
            "OK",
        ]
        if verdict not in valid_verdicts:
            # If the model didn't follow instructions, use a default and return its full response as comment.
            return "DUDOSO", response_text

        return verdict, comment
    except ValueError:
        # The response was not in the expected format
        logger.warning("Unexpected advanced AI response format: '%s'", response_text)
        return None, None


def ai_verdict_pass1(
    original: str,
    asr: str,
    base_prompt: str | None = None,
    return_feedback: bool = False,
    model: str | None = None,
    trace_io: bool | None = None,
) -> str | tuple[str, str]:
    """Single-row, no-context verdict (strict, orthography-tolerant)."""
    trace = TRACE_IO if trace_io is None else trace_io
    system_prompt = base_prompt or STANDARD_PASS1_SYSTEM_PROMPT
    original = _extract_text(original)
    asr = _extract_text(asr)
    logger.info("AI pass1 ORIGINAL=%s | ASR=%s", original, asr)
    current_model = model or MODEL_DEFAULT
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": STANDARD_PASS1_USER_TEMPLATE.format(original=original, asr=asr),
        },
    ]
    resp = _chat_with_backoff(
        model=current_model,
        messages=messages,
        max_completion_tokens=800,
        stop=None,
    )
    content = resp.choices[0].message.content or ""
    if trace:
        _trace_io("pass1", messages, content)
    verdict = _normalize_verdict(content)
    if return_feedback:
        return verdict, content.strip()
    # Por defecto devolvemos el texto literal de la IA;
    # los llamadores aplican _normalize_verdict() solo para lógica interna.
    return content.strip()


def ai_verdict(
    original: str,
    asr: str,
    base_prompt: str | None = None,
    return_feedback: bool = False,
    model: str | None = None,
    trace_io: bool | None = None,
) -> str | tuple[str, str]:
    """Backward-compatible wrapper over :func:`ai_verdict_pass1`."""
    return ai_verdict_pass1(
        original,
        asr,
        base_prompt=base_prompt,
        return_feedback=return_feedback,
        model=model,
        trace_io=trace_io,
    )


def ai_verdict_pass2(
    prev_original: str,
    prev_asr: str,
    original: str,
    asr: str,
    next_original: str,
    next_asr: str,
    model: str | None = None,
    trace_io: bool | None = None,
) -> str:
    """Contextual verdict used only after pass1 OK + structure healthy."""
    trace = TRACE_IO if trace_io is None else trace_io
    prev_original = _extract_text(prev_original)
    prev_asr = _extract_text(prev_asr)
    original = _extract_text(original)
    asr = _extract_text(asr)
    next_original = _extract_text(next_original)
    next_asr = _extract_text(next_asr)
    logger.info("AI pass2 ORIGINAL=%s | ASR=%s", original, asr)
    current_model = model or MODEL_DEFAULT
    messages = [
        {"role": "system", "content": STANDARD_PASS2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": STANDARD_PASS2_USER_TEMPLATE.format(
                prev_original=prev_original or "(sin contexto)",
                prev_asr=prev_asr or "(sin contexto)",
                original=original,
                asr=asr,
                next_original=next_original or "(sin contexto)",
                next_asr=next_asr or "(sin contexto)",
            ),
        },
    ]
    resp = _chat_with_backoff(
        model=current_model,
        messages=messages,
        max_completion_tokens=800,
        stop=None,
    )
    content = resp.choices[0].message.content or ""
    if trace:
        _trace_io("pass2", messages, content)
    return content.strip()


def _apply_verdict_to_row(row: List, ai_text: str, norm_verdict: str, auto_ok: bool) -> None:
    """
    Escribe siempre en la columna AI el texto literal devuelto por la IA.
    ``norm_verdict`` se usa solo para marcar/limpiar la columna OK.
    """
    row[:] = canonical_row(row)
    ai_text = "" if ai_text is None else str(ai_text)
    verdict = _normalize_verdict(norm_verdict)
    if len(row) == 6:
        row.insert(2, "")
        row.insert(3, ai_text)
    elif len(row) == 7:
        row.insert(3, ai_text)
    else:
        row[3] = ai_text
    if auto_ok and verdict == "ok":
        row[2] = "OK"
    elif verdict != "ok" and row[2] == "OK":
        row[2] = ""


def review_row(
    row: List,
    base_prompt: str | None = None,
    model: str | None = None,
    previous: List | None = None,
    next_row: List | None = None,
    trace_io: bool | None = None,
    dual_pass: bool = True,
) -> str:
    """Annotate a single QC row with AI verdict using the two-pass flow."""
    row[:] = canonical_row(row)
    orig = _extract_text(row[6] if len(row) > 6 else "").strip()
    asr = _extract_text(row[7] if len(row) > 7 else "").strip()
    pass1_prompt = base_prompt or STANDARD_PASS1_SYSTEM_PROMPT
    try:
        ai_text1 = _call_with_timeout(
            ai_verdict_pass1,
            ROW_TIMEOUT_SEC,
            orig,
            asr,
            pass1_prompt,
            False,
            model,
            trace_io,
        )
    except BadRequestError as exc:
        if "max_tokens" in str(exc) or "model output limit" in str(exc):
            _mark_error(row)
            return "error"
        raise
    # Normalizar solo para lógica interna; la columna AI guarda el texto literal.
    verdict1 = _normalize_verdict(ai_text1)
    final_ai_text = ai_text1
    final_verdict = verdict1
    if verdict1 == "ok" and dual_pass:
        prev_orig, prev_asr = ("", "")
        next_orig, next_asr = ("", "")
        if previous:
            prev_orig, prev_asr = str(previous[-2]).strip(), str(previous[-1]).strip()
        if next_row:
            next_orig, next_asr = str(next_row[-2]).strip(), str(next_row[-1]).strip()
        try:
            ai_text2 = _call_with_timeout(
                ai_verdict_pass2,
                ROW_TIMEOUT_SEC,
                _extract_text(prev_orig),
                _extract_text(prev_asr),
                orig,
                asr,
                _extract_text(next_orig),
                _extract_text(next_asr),
                model,
                trace_io,
            )
        except Exception:
            ai_text2 = "mal"
        final_ai_text = ai_text2
        final_verdict = _normalize_verdict(ai_text2)
    auto_ok = final_verdict == "ok"
    _apply_verdict_to_row(row, final_ai_text, final_verdict, auto_ok)
    return final_verdict


def review_row_feedback(
    row: List,
    base_prompt: str | None = None,
    model: str | None = None,
    trace_io: bool | None = None,
) -> tuple[str, str]:
    """Like :func:`review_row` but also return the model feedback text (pass1)."""

    row[:] = canonical_row(row)
    orig, asr = _extract_text(row[6] if len(row) > 6 else ""), _extract_text(
        row[7] if len(row) > 7 else ""
    )
    try:
        verdict, feedback = ai_verdict_pass1(
            str(orig),
            str(asr),
            base_prompt,
            return_feedback=True,
            model=model,
            trace_io=trace_io,
        )
    except BadRequestError as exc:
        if "max_tokens" in str(exc) or "model output limit" in str(exc):
            _mark_error(row)
            return "error", ""
        raise
    verdict = _normalize_verdict(verdict)
    # Guardar en AI el texto completo devuelto por la IA (feedback),
    # pero seguir devolviendo/verificando el veredicto normalizado.
    full_text = feedback.strip() if feedback else verdict
    _apply_verdict_to_row(row, full_text, verdict, auto_ok=False)
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
    trace_io: bool | None = None,
    dual_pass: bool = True,
) -> tuple[int, int]:
    """Batch review QC JSON file using the AI flow.

    If ``dual_pass`` is True (por defecto), usa la comprobación contextual
    de la segunda pasada para auto-aprobar filas "ok". Si es False, solo
    se utiliza la primera pasada más las heurísticas estructurales.

    Returns (auto_approved_rows, processed_rows_minus_auto_approved).
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
    pass1_prompt = load_prompt(prompt_path)
    approved = 0
    processed = 0
    current_model = model or MODEL_DEFAULT
    max_requests = limit if limit is not None else MAX_MESSAGES
    requests_sent = 0
    trace = TRACE_IO if trace_io is None else trace_io
    for i, row in enumerate(rows):
        row = canonical_row(row)
        rows[i] = row
        if _stop_review or (max_requests and requests_sent >= max_requests):
            break
        tick = row[1] if len(row) > 1 else ""
        ok = str(row[2]).strip().lower() if len(row) >= 7 else ""
        ai = str(row[3]).strip().lower() if len(row) >= 8 else ""
        if tick.strip() in SKIP_TICKS or ok == "ok" or ai:
            logger.info("Skipping row %s due to prior status tick=%s ok=%s ai=%s", i, tick, ok, ai)
            continue

        original_text = _extract_text(row[6] if len(row) > 6 else "").strip()
        asr_text = _extract_text(row[7] if len(row) > 7 else "").strip()
        if not original_text or not asr_text:
            logger.info(f"Skipping row {i+1} due to missing Original or ASR text.")
            continue

        if progress_callback:
            try:
                progress_callback("start", i, row)
            except Exception:
                logger.exception("progress_callback start failed")
        attempts = 0
        ai_text1 = None  # type: ignore
        while True:
            if max_requests and requests_sent >= max_requests:
                break
            try:
                requests_sent += 1
                ai_text1 = _call_with_timeout(
                    ai_verdict_pass1,
                    ROW_TIMEOUT_SEC,
                    original_text,
                    asr_text,
                    pass1_prompt,
                    False,
                    current_model,
                    trace,
                )
            except TimeoutError:
                attempts += 1
                logger.warning(
                    "Row %d timed out after %ds in pass1 (attempt %d)", i, ROW_TIMEOUT_SEC, attempts
                )
                if attempts >= 2:
                    try:
                        from tkinter import messagebox  # type: ignore

                        messagebox.showerror(
                            "AI Review detenido",
                            f"Se detuvo el AI Review por timeout repetido en la fila {i+1}.",
                        )
                    except Exception:
                        logger.exception("Failed to show error popup")
                    _stop_review = True
                    if progress_callback:
                        try:
                            progress_callback("done", i, row)
                        except Exception:
                            logger.exception("progress_callback done failed (timeout)")
                    break
                continue
            except BadRequestError as exc:
                if "max_tokens" in str(exc) or "model output limit" in str(exc):
                    _mark_error(row)
                    Path(qc_json).write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf8",
                    )
                    if progress_callback:
                        try:
                            progress_callback("done", i, row)
                        except Exception:
                            logger.exception("progress_callback done failed (bad request)")
                    ai_text1 = "error"
                    break
                raise
            break

        if max_requests and requests_sent >= max_requests and ai_text1 is None:
            break
        if ai_text1 is None:
            break

        verdict1 = _normalize_verdict(ai_text1)
        final_ai_text = ai_text1
        final_verdict = verdict1
        if verdict1 == "ok" and dual_pass and not (max_requests and requests_sent >= max_requests):
            prev_orig, prev_asr = _context_text(rows, i - 1)
            next_orig, next_asr = _context_text(rows, i + 1)
            try:
                requests_sent += 1
                ai_text2 = _call_with_timeout(
                    ai_verdict_pass2,
                    ROW_TIMEOUT_SEC,
                    prev_orig,
                    prev_asr,
                    original_text,
                    asr_text,
                    next_orig,
                    next_asr,
                    current_model,
                    trace,
                )
            except TimeoutError:
                logger.warning("Row %d timed out in pass2", i)
                ai_text2 = "mal"
            except BadRequestError as exc:
                if "max_tokens" in str(exc) or "model output limit" in str(exc):
                    ai_text2 = "mal"
                else:
                    raise
            final_ai_text = ai_text2
            final_verdict = _normalize_verdict(ai_text2)
        auto_ok = final_verdict == "ok"

        _apply_verdict_to_row(row, final_ai_text, final_verdict, auto_ok)
        rows[i] = row
        if auto_ok:
            approved += 1
        processed += 1
        Path(qc_json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
        if progress_callback:
            try:
                progress_callback("done", i, row)
            except Exception:
                logger.exception("progress_callback done failed")
        if _stop_review or (max_requests and requests_sent >= max_requests):
            break
    logger.info("Approved %d / Remaining %d", approved, processed - approved)
    return approved, processed - approved


def review_file_feedback(
    qc_json: str,
    prompt_path: str = "prompt.txt",
    limit: int | None = None,
    model: str | None = None,
    trace_io: bool | None = None,
) -> tuple[int, int, List[str]]:
    """Batch review returning feedback strings for each processed row (pass1 only).

    ``limit`` restricts how many requests are sent. ``None`` means use
    ``MAX_MESSAGES``.
    """

    global _stop_review
    _stop_review = False
    trace = TRACE_IO if trace_io is None else trace_io
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
        row = canonical_row(row)
        rows[i] = row
        if _stop_review or (max_requests and sent >= max_requests):
            break
        tick = row[1] if len(row) > 1 else ""
        ok = str(row[2]).strip().lower() if len(row) >= 7 else ""
        ai = str(row[3]).strip().lower() if len(row) >= 8 else ""
        # Skip rows already reviewed by AI regardless of verdict
        if tick.strip() in SKIP_TICKS or ok == "ok" or ai:
            logger.info("Skipping row %s due to prior status tick=%s ok=%s ai=%s", i, tick, ok, ai)
            continue
        sent += 1
        try:
            verdict, fb = ai_verdict_pass1(
                _extract_text(row[6] if len(row) > 6 else ""),
                _extract_text(row[7] if len(row) > 7 else ""),
                prompt,
                return_feedback=True,
                model=current_model,
                trace_io=trace,
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
        verdict = _normalize_verdict(verdict)
        ai_text = fb.strip() if fb else verdict
        _apply_verdict_to_row(row, ai_text, verdict, auto_ok=False)
        if verdict == "ok":
            approved += 1
        Path(qc_json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
    logger.info("Approved %d / Remaining %d", approved, sent - approved)
    return approved, sent - approved, feedback


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch review QC JSON using gpt-5.1")
    parser.add_argument("file", help="QC JSON file path")
    parser.add_argument("--prompt", default="prompt.txt")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of lines to review",
    )
    parser.add_argument(
        "--trace-io",
        action="store_true",
        help="Print prompts and responses for debugging",
    )
    args = parser.parse_args()
    a, b = review_file(
        qc_json=args.file,
        prompt_path=args.prompt,
        limit=args.limit,
        trace_io=args.trace_io,
    )
    print(f"Auto-approved {a} / Remaining {b}")
