# Phase 1 — Domain-Specific Text Analysis & Information Retrieval System

**Domain:** Financial / Economic Documents
**Institution:** Vidyashilp University — Bangalore  **Course:** NLP
**Phase:** 1 of 4 (Phase 1 = corpus engineering, Phase 2 = NLP experiments,
Phase 3 = retrieval, Phase 4 = GUI)

---

## 1. What Phase 1 does

Phase 1 converts the raw PDF reports in the input folder into a **traceable,
machine-readable corpus**. It is the only place in the project that ever reads a
PDF. Every later phase consumes the artefacts Phase 1 writes:

```
PDF files  (READ-ONLY)
      |
      v
[Phase 1] discovery -> ids -> validation -> extraction -> structure -> cleaning
      |
      v
data/corpus/structured/corpus.jsonl      <-- MASTER CORPUS (one line = one unit)
      |
      v
[Phase 2] preprocessing / POS / NER / n-grams / BPE   (not implemented yet)
```

**Phase 2 must never re-read a PDF.** Every record in `corpus.jsonl` already
carries `source_id`, `document_id`, `filename`, `page_number`, `page_id`,
`section_id`, `section_number`, `section_title` and a stable `unit_id`.

---

## 2. Input directory

Configured in `config/phase1_config.yaml`:

```yaml
input:
  directory: "C:\\Users\\user\\Downloads\\NLP Assigment 1\\NLP document"
  supported_extensions: [".pdf"]
  recursive: true
```

The folder is treated as **READ-ONLY**. Documents are opened in binary read mode
only; validation rule **R15** re-hashes every PDF after the run and fails if a
single byte changed. `data/source_pdfs/` is intentionally left empty — the
originals are never copied, moved or overwritten.

No path is hard-coded in the source code. The input directory can also be
overridden without touching the YAML:

```bash
python -m src.phase1.run --input "D:\some\other\folder"
# or
set NLP_PHASE1_INPUT_DIR=D:\some\other\folder
```

---

## 3. Document identification strategy

Discovery is **deterministic**. Candidate files are collected recursively and
sorted by

1. normalised relative path (`\` → `/`, lower-cased), then
2. filename

so `echap01.pdf … echap16-2.pdf` always receive `D01 … D17`, independent of the
order in which the filesystem enumerates them.

Document ids are persisted in `data/corpus/metadata/document_registry.csv`.
On a subsequent run the registry is re-loaded first, so an id that was issued
once is **never** re-issued differently. Only genuinely new files receive new
ids.

---

## 4. Source vs file vs retrieval unit

Three distinct identity levels are maintained, because *a filename is not
necessarily a report*.

| Level | Id | Meaning | Example |
|---|---|---|---|
| **Source / parent document** | `SRC01` | the real report | Economic Survey 2025-26 |
| **Document / ingested file** | `D01` | one PDF file | `echap01.pdf` |
| **Page** | `D01_P002` | one page | page 2 of `echap01.pdf` |
| **Section** | `D01_SEC_1_1` | `1.1` | *GLOBAL ECONOMIC GROWTH – FRAGILE AND DIVERGING* |
| **Content unit (retrieval unit)** | `D01_P002_SEC_1_1_PAR001` | the atom of the corpus | one paragraph / table / chart / box / footnote |

The source registry lives in `data/corpus/metadata/source_registry.csv`:

| source_id | source_title | publisher | source_type | files_count |
|---|---|---|---|---|
| SRC01 | Economic Survey 2025-26 | Ministry of Finance, Government of India | economic_survey | 17 |
| SRC02 | Financial Stability Report | Reserve Bank of India | financial_stability_report | 2 |
| SRC03 | SEBI Annual Report | Securities and Exchange Board of India | regulator_annual_report | 12 |

`echap01.pdf … echap16-2.pdf` are **chapter parts of one parent report**, not 17
independent reports — they all map to `SRC01`.

### Metadata provenance

Nothing is invented. Every document-registry field carries a provenance column
(`title_provenance`, `publisher_provenance`, `year_provenance`):

* `pdf_metadata` — read from the PDF `/Info` dictionary
* `inferred_from_folder` — from a matching source rule
* `inferred_from_filename` — from the file name
* `registry_previous_run` — restored from a previous run (reproducibility)
* `missing` — unknown

A file that cannot be confidently attached to a parent report gets
`source_id = UNKNOWN` rather than a guess.

---

## 5. PDF extraction library and why

| Engine | Role | Reason |
|---|---|---|
| **PyMuPDF (`fitz`)** | primary | exposes the `block → line → span` tree with **font size**, **font name** and **writing direction** per line. The structural detectors need exactly those signals. |
| **pdfplumber** | secondary | understands drawn ruling lines/rectangles (ruled-table detection) and provides a word-level fallback when PyMuPDF returns no text for a page. |

**Rejected approach:** `extract_text()` on the whole document. It flattens the
page into one string and destroys everything Phase 2 needs — headings stop being
headings, tables stop being tables, rotated chart axis labels get mixed into
prose, and nothing can be traced back to a page.

### What is extracted per page

* `blocks`, `lines`, `spans` with geometry, font size, bold/italic and direction
* **character-accurate tokens** (`rawdict`), used to rebuild borderless tables
* image and vector-drawing counts (scanned-page detection)
* dominant body font size (heading/footnote classification)

### Borderless table reconstruction

The Economic Survey tables have **no ruling lines**, so a line-based table
finder sees nothing. Tables are rebuilt geometrically instead:

1. **Row grouping** — tokens whose vertical centres fall within
   `row_tolerance_pt` form one visual row.
2. **Cell segmentation** — inside a row, a horizontal gap wider than
   `column_gap_min_pt` separates two cells. Ordinary word spacing (≈2.8 pt in
   this corpus) is far below the threshold, so a long row label such as
   `Private Final Consumption Expenditure (PFCE)` stays in one cell.
3. **Column alignment** — the midpoints of every detected gap across all rows
   are clustered; consecutive clusters define the column bands.
4. **Quality grading** — `HIGH` (≥3 rows, ≥2 columns, ≥50 % multi-cell rows),
   `MEDIUM`, `LOW`. The raw extracted text is **always** stored as well, so a
   failed reconstruction still preserves the numbers.

Recovered example (`data/corpus/tables/D01_T002.json`):

```
['Agriculture, Livestock, Forestry & Fishing', '', '', '', '2.7', '3.6', '4.6', '3.1']
['Industry',                                     '', '', '', '6.1', '7',   '5.9', '6.2']
['Mining & Quarrying',                           '', '', '', '3.6', '-1.8','2.7', '-0.7']
```

---

## 6. Structure preservation

| Unit type | Detection rule |
|---|---|
| `heading` | standalone short line, uppercase / title-case / numbered, set in body-or-larger type |
| `paragraph` | wrapped lines inside one PDF block, re-joined; vertical gap > 1.45 × line height starts a new paragraph |
| `table` | `Table …:` caption anchor → body → `Source:` note |
| `figure` | `Chart / Figure / Exhibit …:` caption anchor + surrounding text + rotated axis labels |
| `box` | `Box …:` caption anchor + body |
| `footnote` | small type (< 0.93 × body font) in the lower 28 % of the page, or a citation/URL pattern |
| `reference` | isolated numeric footnote marker in the lower page band |
| `other` | anything kept verbatim rather than promoted or dropped |

Additional safeguards:

* **Side-by-side items.** Two charts or two tables printed next to each other get
  separate anchors split per caption line, each restricted to its own
  horizontal band.
* **Prose stop.** A figure/box region ends when real full-width prose begins, so
  a chart never swallows the paragraph below it.
* **Rotated text.** Chart axis labels are written bottom-up and would read as
  gibberish in a paragraph. They are stored in `axis_labels` and excluded from
  prose.
* **Page furniture.** Folios (`2`, `4 4`, roman numerals) are excluded from the
  retrieval units but remain in `raw/` and `cleaned/`.
* **Running headers.** Short lines in the top/bottom band that repeat on ≥3 pages
  (or ≥50 % of the document) are detected per document and excluded from units.
* **Uncertainty is never hidden.** If a detector is not confident it writes
  `quality`, `quality_reason` or `extraction_note` rather than guessing.

### Honest figure handling

`data_extracted` is `false` for a figure unless a genuine tabular layout of
numbers was recovered. Most chart *values* are drawn as vector graphics and are
**not** claimed as accurately extracted; only the caption, source note, legend
text and axis labels are preserved:

```json
"data_extracted": false,
"data_note": "only caption/source/axis labels recovered; chart data values are NOT claimed as accurately extracted"
```

---

## 7. Light structural cleaning

Cleaning happens in `src/cleaning/cleaner.py` and is deliberately conservative.

**Applied:** line-ending normalisation, whitespace collapsing, repeated
blank-line removal, safe intra-paragraph line re-wrapping, hyphenated line-break
repair, private-use glyph replacement (Wingdings spacing characters), running
header/footer removal.

**Explicitly NOT applied (Phase 2 work):** lowercasing, stopword removal,
stemming, lemmatization, tokenization, punctuation stripping, number removal,
URL removal, currency-symbol removal. All of these are `false` in
`config/phase1_config.yaml` so they cannot be switched on by accident.

### Numbers must survive Phase 1

These all pass through untouched and are asserted by
`tests/test_extraction.py::test_numbers_percentages_and_currency_survive_the_pipeline`:

```
7.4 per cent   2 per cent   ₹346 lakh crore   US$   USD   EUR   2%   124.2%
2025   2026   FY26   FY27   Q1 Q2 Q3 Q4   H1 H2
GDP   GVA   GFCF   PFCE   FDI   CPI   WPI   IIP   RBI   SEBI   IMF   UNCTAD   MoSPI
fiscal deficit   primary deficit   capital formation   monetary policy   bond yields
```

URLs inside footnotes (`https://tinyurl.com/5n5ku8rm`,
`https://rbidocs.rbi.org.in/...`) are preserved verbatim for Phase 2 to decide on.

---

## 8. Outputs

```
data/corpus/
├── raw/                D01.txt …             faithful extractor output, no cleaning
├── cleaned/            D01.txt …, all_documents.txt
├── structured/         corpus.jsonl          *** MASTER CORPUS ***
├── pages/              D01_P001.json …       one file per page
├── tables/             D01_T001.json …        one file per table (grid + raw text)
├── figures/            D01_F001.json …        one file per chart/figure/box
├── metadata/
│   ├── document_registry.csv
│   ├── source_registry.csv
│   ├── unit_registry.csv
│   ├── sections_index.csv
│   ├── corpus_index.csv
│   ├── vocabulary_baseline.csv
│   └── baseline_term_dictionary.csv
└── manifests/
    ├── run_manifest.json
    └── source_manifest.json

results/phase1/
├── file_inventory.csv
├── pdf_validation.csv
├── document_statistics.csv
├── extraction_quality_report.csv
├── duplicate_report.csv
├── errors.csv
├── validation_report.json
└── validation_rules.csv

logs/phase1.log
```

### `corpus.jsonl` record types

Paragraph:

```json
{"unit_id":"D01_P002_SEC_1_1_PAR001","document_id":"D01","source_id":"SRC01",
 "filename":"echap01.pdf","page_id":"D01_P002","page_number":2,
 "section_id":"D01_SEC_1_1","section_number":"1.1",
 "section_title":"GLOBAL ECONOMIC GROWTH – FRAGILE AND DIVERGING",
 "unit_type":"paragraph","unit_index":3,"text":"…","char_count":1234,"word_count":210}
```

Table:

```json
{"unit_id":"D01_T002","document_id":"D01","source_id":"SRC01","page_id":"D01_P009",
 "page_number":9,"section_id":"D01_SEC_1_8","unit_type":"table",
 "label":"Table I.2a","caption":"Table I.2a: Demand and Supply side drivers of growth",
 "source_note":"Source: MoSPI","text":"…","table_rows":26,"table_columns":8,
 "table_extraction_quality":"HIGH","data_extracted":true,
 "extraction_note":"26x8 aligned grid, 69% multi-cell rows","table_data":[["…"]]}
```

Figure:

```json
{"unit_id":"D01_F001","document_id":"D01","page_id":"D01_P003","page_number":3,
 "section_id":"D01_SEC_1_3","unit_type":"figure","label":"Chart I.2",
 "caption":"Chart I.2: Policy rates in AEs","text":"…",
 "axis_labels":["Jun-23","Nov-23","Apr-24"],"data_extracted":false,
 "extraction_note":"only caption/source/axis labels recovered; …"}
```

### Flattened TXT corpus

`data/corpus/cleaned/all_documents.txt` keeps full markers so a token or a
search hit can always be walked back to its origin:

```
[DOCUMENT_ID=D01]
[SOURCE_ID=SRC01]
[FILENAME=echap01.pdf]
------------------------------------------------------------------------------
[PAGE=2]
[SECTION=1.1 | GLOBAL ECONOMIC GROWTH – FRAGILE AND DIVERGING]
[UNIT=D01_P002_SEC_1_1_PAR001]
[TYPE=paragraph]
1.1. Since the last version of the Economic Survey was published, …

[UNIT=D01_T002]
[TYPE=table]
[CAPTION=Table I.2a: Demand and Supply side drivers of growth]
[LABEL=Table I.2a]
[SOURCE_NOTE=Source: MoSPI]
Agriculture, Livestock, Forestry & Fishing |  |  |  | 2.7 | 3.6 | 4.6 | 3.1
…
[TABLE_QUALITY=HIGH]
```

---

## 9. Metadata schema (summary)

`document_registry.csv` — `document_id, source_id, filename, relative_path,
title, publisher, year, document_type, parent_folder, page_count,
file_size_bytes, sha256, ingestion_status, extraction_status,
extraction_quality, title_provenance, publisher_provenance, year_provenance`

`source_registry.csv` — `source_id, source_title, publisher,
publisher_provenance, year, year_provenance, source_type, description,
files_count, total_pages, notes`

`unit_registry.csv` — `unit_id, document_id, source_id, filename, page_id,
page_number, section_id, section_number, section_title, unit_type, unit_index,
char_count, word_count`

`page json` — `document_id, source_id, page_id, page_number, text,
cleaned_text, char_count, word_count, line_count, block_count, image_count,
is_text_extractable, is_likely_scanned, extraction_warning, body_font_size,
page_width, page_height, unit_count, section_id, extraction_engine`

---

## 10. Statistics definitions

These are **Phase 1 baseline** statistics. They are **not** the tokenizer
experiment of Phase 2 and are never labelled "final token count".

| Column | Definition |
|---|---|
| `sentences` | NLTK **punkt_tab** (trained, abbreviation-aware) segmentation, plus a documented abbreviation post-guard. Method name recorded as `nltk:punkt_tab+abbrev_guard`. Falls back to `regex_abbreviation_aware` when the NLTK data package is unavailable. Original text is never modified — only boundaries are counted. |
| `words` | whitespace-delimited word count |
| `characters` | characters of unit text |
| `baseline_token_count` | tokens from the single regex in `statistics.baseline_token_pattern`. Written to **keep** numbers (`7.4`, `2025`), percentages (`7.4 per cent`), currency (`₹346`, `US$`) and fiscal periods (`FY26`, `Q3`, `H1`). |
| `raw_vocabulary_size` | unique **case-sensitive** surface forms |
| `vocabulary_size` | unique **NFKC + casefolded** forms — no stemming, no lemmatization |
| `average_document_length` | `baseline_token_count / units` |

`data/corpus/metadata/baseline_term_dictionary.csv` holds
`term, frequency, document_frequency, documents`. **This is a preliminary term
dictionary, not the retrieval index** — the inverted index belongs to Phase 3.

### Sentence-counting hazards handled

`7.4 per cent`, `U.S.`, `IMF.`, `RBI.`, `FY26.`, `Rs.`, `etc.` and single-letter
initials are protected. Punkt is trained on general English and splits
`RBI.`/`IMF.`/`GDP.` in economic prose, so a documented post-guard merges those
fragments again and the method name records that it was applied.

---

## 11. Extraction quality score

Fully documented, not invented:

```
score = 100 * ( 0.30*text_coverage
               + 0.20*page_completeness
               + 0.15*unit_density
               + 0.15*structure
               + 0.20*structural_fidelity )
```

| Component | Definition |
|---|---|
| `text_coverage` | share of pages carrying at least `low_text_char_threshold` characters |
| `page_completeness` | share of pages that produced ≥ `min_units_per_page` typed units |
| `unit_density` | units per page, saturating at `quality.unit_density_target` |
| `structure` | (tables + figures + boxes + sections/2) per `quality.structure_per_element_pages` pages, capped at 1 |
| `structural_fidelity` | (HIGH + 0.5 × MEDIUM) / all detected tables; neutral (1.0) when a document has no tables |

Grades: A ≥ 85, B ≥ 70, C ≥ 50, D otherwise. Each component is written to
`results/phase1/extraction_quality_report.csv` so the score can be recomputed by
hand. The current corpus is uniformly digital text, so every document grades A
(90.0 – 100.0); the spread comes from table-reconstruction fidelity.

---

## 12. Validation rules

`results/phase1/validation_report.json` is computed, never hard-coded. All 15
rules pass on the current corpus:

| Rule | Check | Result |
|---|---|---|
| R01 | every discovered PDF has a document id | PASS 31/31 |
| R02 | every PDF appears in `document_registry.csv` | PASS |
| R03 | every page has a page record | PASS 885/885 |
| R04 | every structured unit has a `unit_id` | PASS 8104 |
| R05 | no `unit_id` is duplicated | PASS |
| R06 | every unit references an existing `document_id` | PASS |
| R07 | every unit references a valid page | PASS |
| R08 | raw text exists for every processed document | PASS 31/31 |
| R09 | cleaned text exists for every processed document | PASS 31/31 |
| R10 | `corpus.jsonl` loads without errors | PASS |
| R11 | character counts non-negative | PASS |
| R12 | word counts non-negative | PASS |
| R13 | duplicate detection executed | PASS, 0 duplicates |
| R14 | extraction errors logged | PASS, `errors.csv` empty |
| R15 | original PDFs not modified | PASS 31/31 |

---

## 13. Error handling

Every document is processed inside `try/except`. One corrupt PDF logs a row in
`results/phase1/errors.csv`
(`document_id, file, stage, error_type, error_message, recoverable`) and the run
continues with the next file. Page-level failures degrade to an empty page and
are recorded as warnings. This behaviour is covered by
`tests/test_pipeline.py::test_a_single_bad_pdf_does_not_stop_the_pipeline`.

`results/phase1/pdf_validation.csv` records per-page
`text_char_count, word_count, is_text_extractable, is_likely_scanned,
image_count, drawing_count, unit_count, extraction_warning`. A page with almost
no text but embedded images/vector drawings is flagged as *likely scanned*
rather than silently accepted.

---

## 14. Duplicate detection

* **Exact:** SHA-256 of every input file. `is_duplicate` / `duplicate_of` in
  `file_inventory.csv`.
* **Extracted text:** identical raw-text fingerprints and a 200-character
  shingle similarity probe over document openings, written to
  `results/phase1/duplicate_report.csv`.
* Duplicates are **flagged, never deleted** (`duplicates.delete_duplicates: false`).

---

## 15. How to run Phase 1

```bash
cd NLP_Domain_Text_Analysis
pip install -r requirements.txt
python -m src.phase1.run
```

Optional flags:

```bash
python -m src.phase1.run --config path\to\phase1_config.yaml
python -m src.phase1.run --input "C:\path\to\other\pdfs"
python -m src.phase1.run --verbose
```

Tests:

```bash
python -m pytest tests -q
```

The exit code is `0` when validation is `PASS`, `1` otherwise.

---

## 16. How Phase 2 will consume `corpus.jsonl`

```python
import json

with open("data/corpus/structured/corpus.jsonl", encoding="utf-8") as handle:
    units = [json.loads(line) for line in handle]

prose = [u for u in units if u["unit_type"] in ("paragraph", "box", "footnote")]

# every unit can be traced back to its origin
u = units[0]
u["unit_id"] -> u["page_id"] -> u["document_id"] -> u["filename"] -> u["source_id"]
```

Notes for Phase 2:

* filter on `unit_type` to isolate prose / tables / charts
* `section_number` gives ground-truth-ish section numbers for POS & NER analysis
* `table_data` is ready for numeric experiments without re-parsing the PDF
* the baseline token count in `document_statistics.csv` is the *comparison
  baseline*; NLTK / spaCy / custom / hybrid / BPE results should be reported
  against it
* `data/corpus/metadata/baseline_term_dictionary.csv` is the starting point for
  the Phase 3 inverted index — it is **not** an index yet

---

## 17. Known extraction limitations

1. **Chart data values are not recovered.** Charts are vector graphics; the
   caption, legend text and axis labels are stored, but the plotted values are
   not claimed. `data_extracted` is `false` unless a genuine tabular layout was
   recovered.
2. **Tables split across a page boundary** are preserved as a caption-only unit
   with `table_extraction_quality = LOW` and
   `body_continues_on_next_page = true` (8 of the 9 LOW tables in the current
   corpus; the 9th is a sparse progress-card layout whose raw text is kept).
3. **Widely-spaced chart data callouts** that fall outside their chart's
   horizontal band are emitted as separate short `other` units rather than being
   attached to the figure.
4. **Inherited section titles.** A dotted-number section inherits the nearest
   preceding standalone heading. In the Economic Survey this is the chapter
   title, which is correct for chapters but means a subsection such as `1.5`
   carries the chapter title rather than a subsection-specific one.
5. **`sentence` counts are a baseline.** Punkt is trained on general English;
   the abbreviation post-guard reduces but does not eliminate domain
   segmentation error. This is exactly the kind of measurement Phase 2 will
   compare against.
6. **Rotated text** that is not a chart axis (rare, e.g. a rotated side table) is
   not reconstructed into a grid.
7. **No OCR.** One page in the corpus (`D15`, page 7 — a full-page graphic) is
   flagged `is_likely_scanned` and has almost no extractable text. Phase 1
   deliberately reports this instead of hiding it.

---

## 18. Phase boundaries — what is *not* implemented here

* Phase 2: stopwords, stemming, lemmatization, POS tagging, custom POS, ML POS,
  NER, n-gram experiments, BPE, tokenizer comparison — **not implemented**
* Phase 3: inverted index, Boolean AND/OR/NOT, phrase search, retrieval
  evaluation — **not implemented**
* Phase 4: GUI — **not implemented**

The corpus schema was designed so all of them can consume it without touching a
PDF again.