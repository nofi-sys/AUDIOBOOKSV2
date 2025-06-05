# Audiobook QC

This project provides a Tkinter application to align audiobook scripts with ASR transcripts.

## Requirements

- Python 3.12
- `unidecode`
- `pdfplumber`
- `rapidfuzz`
- `flake8` and `black` for linting (optional)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the GUI with:

```bash
python qc_app.py
```

The application lets you select a script (PDF or TXT) and an ASR transcript, performs alignment and saves a `.qc.json` file.

## Implementation notes

Alignment relies on dynamic time warping with anchor trigrams to keep the
matching monotonic.  Each output row shows the WER of a chunk computed using
word-level Levenshtein distance.
