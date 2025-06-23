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

The application lets you select a script (PDF or TXT) and an ASR transcript,
performs alignment and saves a `.qc.json` file.


### Command line usage

You can also transcribe from the command line:

```bash
python -m transcriber myaudio.mp3 --script book.txt
```

To generate a word-level QC file in one step use `--word-align`:

```bash
python -m transcriber myaudio.mp3 --script book.txt --word-align
```

This creates `myaudio.words.qc.json` without overwriting the regular QC file.

If you already have a QC JSON and a CSV with word times you can resync the
timecodes directly:

```bash
python -m transcriber myaudio.qc.json --resync-csv myaudio.words.csv
```

The updated rows are written to `myaudio.resync.json` next to the original file.


## Manual review

Double-click the **OK** column in the results table to mark or unmark a row as
reviewed. Use the **Guardar JSON** button to save the current table so these
marks persist. Loading a previously saved file that includes an **OK** column
will restore the review status for each row.

Right-click a cell under **Original** or **ASR** to open a menu. Besides moving
the entire cell up or down, the menu includes options to send only the first or
last word to the adjacent row.

## Implementation notes

Alignment relies on dynamic time warping with anchor trigrams to keep the
matching monotonic.  Each output row shows the WER of a chunk computed using
word-level Levenshtein distance.

### AI Review

With a JSON file loaded you can click **AI Review (o3)** to send unchecked lines
to OpenAI and automatically fill the *AI* column with `ok`, `mal` or `dudoso`.
Lines marked `ok` are also auto-approved. Use **Detener análisis** to interrupt
the batch review if needed.

### OpenAI setup

This project uses the OpenAI Python client (v1+) for AI review.
Set your API key in the OPENAI_API_KEY environment variable.
You can create a .env file with the key and optional debugging flag:

```
OPENAI_API_KEY=sk-yourkey
AI_REVIEW_DEBUG=1
```

The .env file is ignored by git so your credentials remain private.

