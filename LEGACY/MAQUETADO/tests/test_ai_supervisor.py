import os
import json
from unittest.mock import patch, MagicMock

import pytest

# Mock the OpenAI library before it's imported by the module under test
mock_openai = MagicMock()

# Define a mock for the chat completions response
class MockChoice:
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content

class MockCompletion:
    def __init__(self, choices):
        self.choices = choices

# We need to patch 'openai' in the context of the module where it is imported
# This is why we patch 'txt2md_mvp.ai_supervisor.openai'
@patch('txt2md_mvp.ai_supervisor.openai', new=mock_openai)
def test_supervise_heading_ai_corrects_to_paragraph():
    """
    Test that the AI supervisor corrects a candidate heading to a paragraph
    when the AI response indicates it is not a heading.
    """
    from txt2md_mvp.ai_supervisor import supervise_heading, get_openai_client

    # Configure the mock response from the AI
    ai_response_payload = {"is_heading": False, "correction": "p"}
    mock_completion = MockCompletion([MockChoice(json.dumps(ai_response_payload))])

    # Set up the mock client and its create method
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    # Ensure get_openai_client returns our mock client
    with patch('txt2md_mvp.ai_supervisor.get_openai_client', return_value=mock_client):
        result = supervise_heading(
            candidate_heading="TRELAWNEY",
            previous_block="Some text before.",
            next_block="Some text after.",
            current_decision="h3",
            confidence=0.4
        )

        # Verify that the AI was called with the correct parameters
        mock_client.chat.completions.create.assert_called_once()
        # Assert that the result is the correction from the AI
        assert result == ai_response_payload

@patch('txt2md_mvp.ai_supervisor.openai', new=mock_openai)
def test_supervise_heading_ai_confirms_heading():
    """
    Test that the AI supervisor confirms a candidate heading when the
    AI response indicates it is a heading.
    """
    from txt2md_mvp.ai_supervisor import supervise_heading

    ai_response_payload = {"is_heading": True, "correction": "h2"}
    mock_completion = MockCompletion([MockChoice(json.dumps(ai_response_payload))])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch('txt2md_mvp.ai_supervisor.get_openai_client', return_value=mock_client):
        result = supervise_heading(
            candidate_heading="A REAL HEADING",
            previous_block="Some text before.",
            next_block="Some text after.",
            current_decision="h2",
            confidence=0.6
        )
        assert result == ai_response_payload

@patch('txt2md_mvp.ai_supervisor.openai', new=None)
def test_supervise_heading_openai_not_installed():
    """
    Test that the supervisor falls back to the original decision
    if the openai library is not installed.
    """
    from txt2md_mvp.ai_supervisor import supervise_heading

    result = supervise_heading(
        candidate_heading="A HEADING",
        previous_block="Prev",
        next_block="Next",
        current_decision="h2",
        confidence=0.5
    )

    assert result["is_heading"] is True
    assert result["correction"] == "h2"
    assert "OpenAI client not configured" in result["reason"]

@patch.dict(os.environ, {}, clear=True)
@patch('txt2md_mvp.ai_supervisor.openai', new=mock_openai)
def test_supervise_heading_api_key_not_set():
    """
    Test that the supervisor falls back when the OPENAI_API_KEY is not set.
    """
    from txt2md_mvp.ai_supervisor import supervise_heading

    result = supervise_heading(
        candidate_heading="A HEADING",
        previous_block="Prev",
        next_block="Next",
        current_decision="h2",
        confidence=0.5
    )

    assert result["is_heading"] is True
    assert result["correction"] == "h2"
    assert "OpenAI client not configured" in result["reason"]

@patch('txt2md_mvp.ai_supervisor.openai', new=mock_openai)
def test_supervise_heading_api_call_fails():
    """
    Test that the supervisor falls back if the API call raises an exception.
    """
    from txt2md_mvp.ai_supervisor import supervise_heading

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    with patch('txt2md_mvp.ai_supervisor.get_openai_client', return_value=mock_client):
        result = supervise_heading(
            candidate_heading="A HEADING",
            previous_block="Prev",
            next_block="Next",
            current_decision="h2",
            confidence=0.5
        )

        assert result["is_heading"] is True
        assert result["correction"] == "h2"
        assert "API call failed" in result["reason"]