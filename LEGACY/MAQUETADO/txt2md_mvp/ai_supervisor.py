import os
import json
from typing import Optional, Dict, Any, Callable


# Check for OpenAI library and API key
try:
    import openai
    from openai.types.chat import ChatCompletion
    from .api_logger import log_interaction
except ImportError:
    openai = None
    ChatCompletion = None
    # This fallback allows running the script directly for testing
    from api_logger import log_interaction # type: ignore


def get_openai_client() -> Optional["openai.OpenAI"]:
    """Returns an OpenAI client if the library is installed and the API key is set."""
    if openai and os.getenv("OPENAI_API_KEY"):
        return openai.OpenAI()
    return None


def supervise_heading(
    candidate_heading: str,
    previous_block: Optional[str],
    next_block: Optional[str],
    current_decision: str,
    confidence: float,
    model: str = "gpt-5-mini",
    token_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """
    Uses an AI model to verify if a candidate heading is correctly classified.
    """
    client = get_openai_client()
    if not client:
        # If the OpenAI client is not available, trust the original decision
        return {"is_heading": True, "correction": current_decision, "reason": "OpenAI client not configured."}

    print(f"--- AI Supervisor: Using model {model} ---")

    system_prompt = (
        "You are an expert book structuring assistant. Your task is to analyze a candidate heading "
        "in the context of the preceding and succeeding text blocks. Determine if the candidate is a "
        "structural heading (like a chapter or section title) or if it is part of the narrative prose. "
        "Respond only with a JSON object in the format: "
        '{\\"is_heading\\": boolean, \\\"correction\\\": \\\"p\\\" or \\\"h1\\\" or \\\"h2\\\" or \\\"h3\\\"}. '
        'If \\\"is_heading\\\" is true, \\\"correction\\\" should be the same as the original decision.'
    )

    user_prompt = {
        "previous_block": previous_block or "",
        "candidate_heading": candidate_heading,
        "next_block": next_block or "",
        "current_decision": current_decision,
        "confidence": confidence
    }

    try:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
            ],
            "response_format": {"type": "json_object"},
        }
        response = client.chat.completions.create(**params)
        log_interaction(model, user_prompt, params, response=response)

        if token_callback and response.usage:
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            cached_tokens = (
                getattr(usage, "cached_prompt_tokens", None)
                or getattr(usage, "cached_input_tokens", None)
                or getattr(usage, "cached_tokens", None)
            )
            if prompt_tokens:
                token_callback(prompt_tokens, model=model, type="input", purpose="ai_supervision")
            if completion_tokens:
                token_callback(completion_tokens, model=model, type="output", purpose="ai_supervision")
            if cached_tokens:
                token_callback(cached_tokens, model=model, type="cached_input", purpose="ai_supervision")
            if not (prompt_tokens or completion_tokens):
                total_tokens = getattr(usage, "total_tokens", None)
                if total_tokens:
                    token_callback(total_tokens, model=model, type="output", purpose="ai_supervision")

        content = response.choices[0].message.content
        if content:
            # The API should return a JSON object directly, so we parse it.
            return json.loads(content)
        return {"is_heading": True, "correction": current_decision, "reason": "AI returned empty content."}

    except Exception as e:
        # In case of any API error, fall back to the original decision
        log_interaction(model, user_prompt, params, error=e)
        return {"is_heading": True, "correction": current_decision, "reason": f"API call failed: {e}"}
