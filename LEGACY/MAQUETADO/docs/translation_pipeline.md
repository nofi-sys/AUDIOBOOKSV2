# Translation Pipeline Overview

This document describes how the txt2md MVP converts a plain text book into a translated Markdown document and, optionally, a DOCX export. The explanation follows the real execution order and highlights every context signal that reaches the AI models.

## 1. Input Preparation

1. **Source ingestion** – The user picks one or more `.txt` files. Each file is normalised (BOM removal, soft line‐break collapse, Gutenberg header/footer cleaning).
2. **Structural detection** – The pipeline scans the cleaned text to identify headings, subtitles, paragraphs, block quotes, page breaks, etc. Heuristics are rule based and consult helper functions such as `detect_subtitle` and `is_title_like_for_merge`.
3. **Document model** – The detected blocks are stored as an ordered list of dictionaries. Every block keeps at least `type`, `text`, and possibly classification hints (e.g. indentation, level, detected metadata). This list is what later feeds the Markdown renderer and the translation engine.

## 2. Glossary Generation

1. **Term extraction (`GlossaryBuilder.extract_terms`)**
   - Runs spaCy over the full text to grab named entities (PERSON, ORG, GPE, LOC).
   - Counts frequent nouns/proper nouns while filtering stopwords, too short tokens, and noisy fragments (e.g. trailing hyphens).
   - Builds context snippets around each candidate (maximum 3 per term, 120 characters on either side).
2. **AI curation (`GlossaryBuilder.curate_terms`)**
   - Candidates are split into batches (≤20 per call) and sent to `gpt-5`.
   - Prompt instructs the model to label each term with `category`, decide an `action` (`translate`, `keep`, `ignore`), and add a short rationale.
   - If a batch response is empty or truncated, the batch falls back to default policies instead of wiping the list.
3. **AI translation (`GlossaryBuilder.translate_glossary`)**
   - Only terms marked `translate` are forwarded (again in batches of ≤20) to the translator model (`gpt-5` by default, overridable via GUI).
   - Context snippets (up to 3) plus category/rationale accompany each term.
   - Empty translations are discarded; “keep” terms retain English, and “ignore” terms are removed.
4. **Glossary file**
   - The cleaned result is written as `<input>_glossary.json` with structured metadata:
     ```json
     "TERM": {
       "translation": "TRADUCCION",
       "category": "organization",
       "action": "translate",
       "rationale": "term frequently repeated",
       "frequency": 5,
       "examples": ["short snippet 1", "short snippet 2", …]
     }
     ```
   - The GUI only prompts the user to review the JSON if “Modo Interactivo” is enabled; otherwise the AI output is used directly (empty entries are filtered out).

## 3. Summary Generation

`SummaryGenerator.generate` sends the first 5 000 characters of the book to `gpt-5-mini` with a prompt that requests a 200‑word synopsis describing genre, tone, style, and main plot. If the API key is missing, the summary stage is skipped gracefully.

## 4. Translation Loop

The translation engine walks each block in the document list and uses the glossary, summary, and style options to drive `TranslationCaller`.

### Context passed to every translation call

For each block of type paragraph or heading:

1. **Global summary** – the 200‑word synopsis from step 3.
2. **Style guide** – selected profile (`Literario`, `Tecnico`, `Ensayo`) plus any optional notes typed in the GUI. This is injected as:
   ```
   ### GUIA DE ESTILO Y TONO ###
   1. TONO GENERAL: … 
   2. EXPRESIONES IDIOMATICAS Y JERGA: …
   3. COMPLEJIDAD SINTACTICA: …
   ### NOTAS ADICIONALES DEL USUARIO ###
   (solo si se ingresaron)
   ```
3. **Glossary** – the full JSON (after cleaning) is embedded verbatim, with the instruction “Debes usar EXACTAMENTE las siguientes traducciones…”.
4. **Immediate context** – the previous block text and the next block text (when available) labelled as “Texto Anterior” and “Texto Posterior”.
5. **Chunk to translate** – the block’s own text inside quotes.

The complete prompt is sent to `gpt-5-mini` (or whichever model was selected in the GUI). The response must be a plain translation; errors are caught and returned as `[[TRADUCTION_ERROR: …]]` so they are visible in the output.

Blocks that aren’t text (e.g. horizontal rules) are copied without modification.

## 5. Quality Assurance (optional but automatic)

After the translation finishes:

1. **Glossary consistency** – `ConsistencyChecker` scans the translated text to ensure no English terms remain when the glossary provided a translation.
2. **Trap words** – `TranslationQA.detect_calques` looks for common English leftovers (“shot”, “deal”, etc.) and reports matches with surrounding snippets.
3. **Sampling QA (if a QA model is chosen)** – A configurable percentage (default 5%, max 8 samples) of block pairs is re‑examined by a QA prompt that rates tone fidelity and flags literal calques. Requires a valid API key.
4. **QA report** – Results are saved to `<input>_qa_report.json` and summarised in the GUI log.

## 6. Markdown & DOCX Output

1. Markdown is assembled from the translated document blocks, respecting headings, paragraphs, blockquotes, etc.
2. DOCX export (if requested) reuses the block structure, applies the chosen style presets, and uses the same translated content.

## Model usage summary

| Stage                     | Default model | Context provided                                                                               |
|---------------------------|---------------|-------------------------------------------------------------------------------------------------|
| Glossary curation         | `gpt-5`       | Candidate term batches with up to 3 snippets, metadata (POS, entity labels, casing)            |
| Glossary translation      | `gpt-5`       | Actionable terms with category, rationale, up to 3 context snippets                             |
| Summary                   | `gpt-5-mini`  | First 5 000 characters of the book                                                              |
| Block translation         | `gpt-5-mini`  | Global summary, selected style profile + notes, full glossary JSON, previous/next block context |
| QA sampling (optional)    | `gpt-5-mini`  | Original vs translated block pair                                                               |

The glossary stages can be assigned different models (e.g. `gpt-5` for curation, `gpt-5-mini` for translation) via the GUI. If no API key is present, the system falls back to deterministic behaviour (skip AI calls, keep original text).
