# AGENTS instructions

This file coordinates sequential work on the Check_Audiobook project. Each agent should read the tasks below before starting and update the status after completing their assignment. Work sequentially to avoid merge conflicts.

## Guidelines
- Work on the `main` branch only. Pull the latest commits before starting a task.
- After finishing a task, update this file marking the task status as `done`.
- Run `flake8` and `pytest` before committing.
- Keep commits focused on one task at a time.

## Task list

| ID | Agent | Description | Status |
|----|-------|-------------|--------|
| 1 | Agent-Normalize | Expand normalization in `text_utils.py` (`normalize`, `token_equal`) with more abbreviations and punctuation equivalences. Update tests accordingly. | done |
| 2 | Agent-Transcriber | Convert the provided Whisper script into module `transcriber.py` with a CLI function to transcribe audio. | done |
| 3 | Agent-WordList | Implement word list extraction from PDFs and feed it to Whisper in `transcriber.py`. | done |
| 4 | Agent-Alignment | Adjust DTW parameters in `alignment.py` to leverage the word list and improve alignment accuracy. | done |
| 5 | Agent-AIReview | Create module `ai_review.py` to send flagged lines to GPT for validation. | done |
| 6 | Agent-GUI | Begin migrating the current Tkinter interface to a Kivy GUI, keeping existing features. | done |
| 7 | Agent-Tests | Add and update tests to cover new modules and features. Ensure `pytest` passes. | done |

