import json
from unittest import mock

import pytest
from ai_review import get_advanced_review_verdict

def _build_context_from_rows(rows: list, target_id: int) -> dict:
    """Builds the context dictionary for a specific row."""
    context = {}
    try:
        current_row_data = next(r for r in rows if r[0] == target_id)
        current_row_index = rows.index(current_row_data)

        context["current"] = {
            "id": current_row_data[0],
            "original": current_row_data[6],
            "asr": current_row_data[7],
        }
        if current_row_index > 0:
            prev_row_data = rows[current_row_index - 1]
            context["previous"] = {
                "id": prev_row_data[0],
                "original": prev_row_data[6],
                "asr": prev_row_data[7],
            }
        if current_row_index < len(rows) - 1:
            next_row_data = rows[current_row_index + 1]
            context["next"] = {
                "id": next_row_data[0],
                "original": next_row_data[6],
                "asr": next_row_data[7],
            }
    except (StopIteration, IndexError):
        return {}
    return context


def test_repetition_detection():
    # Load the test data
    with open("ejemplos/repetition_test.qc.json", "r", encoding="utf-8") as f:
        rows = json.load(f)

    # Build the context for the row with the repetition
    context = _build_context_from_rows(rows, target_id=1)

    # Mock the AI call to return a REPETICION verdict
    with mock.patch("ai_review._chat_with_backoff") as mock_chat:
        # Configure the mock to return the expected response structure
        mock_choice = mock.Mock()
        mock_choice.message.content = "VERDICT: REPETICION | COMMENT: Se repite 'la'."
        mock_response = mock.Mock()
        mock_response.choices = [mock_choice]
        mock_chat.return_value = mock_response

        # Call the function with repetition_check=True
        verdict, comment = get_advanced_review_verdict(context, repetition_check=True)

        # Assert that the correct verdict and comment are returned
        assert verdict == "REPETICION"
        assert comment == "Se repite 'la'."

        # Verify that the prompt sent to the AI was the repetition prompt
        call_args, call_kwargs = mock_chat.call_args
        messages = call_kwargs.get("messages", [])
        user_message = next((m for m in messages if m["role"] == "user"), None)
        assert user_message is not None
        assert "[ACTUAL]" in user_message["content"]
        assert "Esta es la la segunda linea." in user_message["content"]
