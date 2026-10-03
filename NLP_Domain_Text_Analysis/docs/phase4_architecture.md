# Phase 4 Architecture — Domain-Specific Text Analysis and Retrieval System

**Domain:** Financial / Economic documents
**Scope of this document:** what Phases 1–3 already produce, which of those artefacts
Phase 4 exposes, and how the new Next.js frontend reaches them through FastAPI.

Every path, column name and number in this document was read from the actual project
tree, not assumed. Where a value is quoted it is the value currently in the artefact.

---

## 1. Guiding rule: one source of truth per concern

| Concern | Source of truth | Why |
|---|---|---|
| Corpus, documents, pages, sections, content units | `data/corpus/**` (Phase 1) | The frozen extraction |
| Tokenization, stopwords, stemming, lemmatization, POS, NER, n-grams, BPE | `results/phase2/**` (Phase 2) | The measured experiments |
| Pipeline choice, inverted index, keyword/phrase/Boolean retrieval | `src/phase3/**` + `results/phase3/**` (Phase 3) | The retrieval subsystem |
| Evaluation metrics, relevance judgments, API surface, GUI payloads | `src/phase4/**`, `backend/**`, `results/phase4/**` (Phase 4) | New work |

**Next.js contains no NLP.** There is no tokenizer, stemmer, lemmatizer, POS tagger,
NER model, n-gram generator, BPE, inverted index, Boolean parser or metric calculator in
TypeScript. The browser receives numbers that Python computed and renders them.

Phase 4 does not rerun Phase 1–3. It reads their artefacts and, for search, calls the
existing `src.phase3.retrieval.RetrievalEngine` in-process.

---

## 2. What Phase 1 produces (corpus engineering)

**Code:** `src/core`, `src/ingestion`, `src/extraction`, `src/cleaning`, `src/structure`,
`src/statistics`, `src/validation`, orchestrated by `src/phase1/run.py`.
**Config:** `config/phase1_config.yaml`.

### 2.1 Data artefacts

| Path | Content | Row / record count |
|---|---|---|
| `data/corpus/structured/corpus.jsonl` | one JSON object per content unit | 8,104 |
| `data/corpus/raw/D*.txt` | raw per-page text | 31 |
| `data/corpus/cleaned/D*.json` | cleaned per-page text | 31 |
| `data/corpus/pages/D##_P###.json` | page record incl. full page text | 885 |
| `data/corpus/tables/D##_T###.json` | extracted table | 137 |
| `data/corpus/figures/D##_F###.json` | figure caption record | 121 |
| `data/corpus/metadata/document_registry.csv` | one row per document | 31 |
| `data/corpus/metadata/source_registry.csv` | one row per source report | 3 |
| `data/corpus/metadata/unit_registry.csv` | unit provenance without text | 8,104 |
| `data/corpus/metadata/sections_index.csv` | section spans with confidence | — |
| `data/corpus/metadata/corpus_index.csv` | unit → `jsonl_line` pointer | 8,104 |
| `data/corpus/metadata/baseline_term_dictionary.csv` | `term, frequency, document_frequency, documents` | — |
| `data/corpus/manifests/run_manifest.json`, `source_manifest.json` | run provenance | — |

### 2.2 `corpus.jsonl` unit schema (one object per line)

```json
{
  "unit_id": "D01_P001_UNKNOWN_PAR001",
  "document_id": "D01",
  "source_id": "SRC01",
  "filename": "echap01.pdf",
  "page_id": "D01_P001",
  "page_number": 1,
  "section_id": "UNKNOWN",
  "section_number": "UNKNOWN",
  "section_title": "UNKNOWN",
  "unit_type": "paragraph",
  "unit_index": 2,
  "text": "STATE OF THE ECONOMY: PUSHING THE GROWTH FRONTIER",
  "char_count": 49,
  "word_count": 8,
  "bbox": [67.54, 90.54, 367.82, 168.08],
  "font_size": 21.0
}
```

`unit_type` ∈ `paragraph, table, figure, box, heading, footnote, reference, other`.

### 2.3 `document_registry.csv` columns

`document_id, source_id, filename, relative_path, title, publisher, year, document_type,
parent_folder, page_count, file_size_bytes, sha256, ingestion_status, extraction_status,
extraction_quality, title_provenance, publisher_provenance, year_provenance`

Example: `D01 | SRC01 | echap01.pdf | … | Ministry of Finance, Government of India | 2026 |
economic_survey_chapter | 35 | 1053419 | … | OK | SUCCESS | A (100.0)`

### 2.4 `source_registry.csv` columns

`source_id, source_title, publisher, publisher_provenance, year, year_provenance,
source_type, description, files_count, total_pages, notes`

Three sources: `SRC01` Economic Survey 2025-26 (17 files, 676 pages),
`SRC02` financial stability report, `SRC03` regulator annual report.

### 2.5 Result artefacts

| Path | Key columns |
|---|---|
| `results/phase1/document_statistics.csv` | `document_id, source_id, filename, page_count, sentences, words, characters, baseline_token_count, unique_words, vocabulary_size, raw_vocabulary_size, average_document_length, sentences_per_page, sections, units, paragraphs, tables, figures, footnotes, raw_characters, cleaned_characters` |
| `results/phase1/file_inventory.csv` | `file_id, document_id, source_id, source_path, relative_path, filename, extension, file_size_bytes, page_count, parent_folder, detected_source, sha256, is_duplicate, duplicate_of, status, error_message` |
| `results/phase1/pdf_validation.csv` | per page: `document_id, page_number, text_char_count, word_count, is_text_extractable, is_likely_scanned, image_count, drawing_count, unit_count, extraction_warning` |
| `results/phase1/extraction_quality_report.csv` | `quality_score, quality_grade, component_text_coverage, …` |
| `results/phase1/validation_rules.csv` | `rule_id, rule, status, checked_items, failures, detail` (15 rules) |
| `results/phase1/validation_report.json` | `status, documents_found/processed/failed, pages_validated: 885, units_validated: 8104, rule_counts, duplicate_summary, quality_score_formula` |
| `results/phase1/duplicate_report.csv`, `results/phase1/errors.csv` | headers present, no rows |

### 2.6 Text selection policy

Phase 2 and Phase 3 both select unit types `paragraph, reference, box, table`
(policy `prose_tables`) → **6,005 of 8,104 units, 2,084,810 characters**. Footnotes,
figures and headings are excluded from the NLP experiments. This policy is read from
config, never hard-coded in Phase 4.

---

## 3. What Phase 2 produces (NLP experiments)

**Code:** `src/phase2/*.py` (tokenizers, stopwords, stemming, lemmatization,
pos_tagging, custom_pos, ml_pos, ner, ngrams, bpe, date_number_tokenizer, comparisons,
statistics, figures, reporting, validation, load_corpus, config, run).
**Config:** `config/phase2_config.yaml`.
**Gold annotation input:** `data/phase2/annotations/pos_gold_annotations.csv`.

### 3.1 Artefact map (all under `results/phase2/`)

| Experiment | Files | Key columns |
|---|---|---|
| Tokenization | `tokenization/tokenization_comparison.csv` | `tokenizer, total_tokens, unique_tokens, vocabulary_size, avg_tokens_per_sentence, numeric_tokens, date_tokens, fiscal_year_tokens, percentage_tokens, currency_tokens, abbreviation_tokens, hyphenated_tokens, special_financial_tokens, execution_time_seconds` |
| | `tokenization/tokenization_examples_comparison.csv` | `original_text, nltk_tokens, spacy_tokens, custom_tokens, hybrid_tokens` |
| | `tokenization/{nltk,spacy,custom,hybrid}_examples.csv` | `original_text, tokens, token_count` |
| | `tokenization/custom_tokenizer_rules.csv` | `rule_id, priority, pattern, description, example, expected_behavior, actual_behavior` |
| | `tokenization/date_number_{comparison,patterns}.csv` | `category, class, expressions_detected, expressions_intact_after_standard_tokenization, intact_rate_percent` |
| Stopwords | `preprocessing/stopword_comparison.csv` | `strategy, stopword_list_size, total_tokens, tokens_removed, vocabulary_size, token_reduction_percent, top_remaining_terms` |
| | `preprocessing/domain_stopword_analysis.csv`, `stopwords_protected_terms.csv` | `term, in_standard_stopword_list, in_domain_aware_stopword_list, corpus_occurrences, outcome, rationale` |
| | `preprocessing/final_stopwords.txt` | the effective list |
| Preprocessing chain | `preprocessing/preprocessing_comparison.csv` | `stage, method, total_tokens, unique_tokens, vocabulary_size, type_token_ratio, token_change_vs_baseline_percent` |
| | `preprocessing/before_after_examples.csv` | `original_text, original, tokenized_hybrid, stopword_removed, stemmed_after_stopwords, lemmatized_after_stopwords` + per-stage counts |
| Stemming | `stemming/stemming_comparison.csv` | `algorithm, vocabulary_before, vocabulary_after, unique_stems, vocabulary_reduction_percent, colliding_stems, execution_time_seconds` |
| | `stemming/stemming_examples.csv`, `stemming_families.csv` | `original_word, porter, snowball, lancaster, collision_group, observation` |
| | `stemming/stopword_stemming_order_comparison.csv` | the order experiment that Phase 3's pipeline A/B restates |
| Lemmatization | `lemmatization/lemmatization_comparison.csv` | `method, pos_required, lemma_resource, total_tokens, unique_lemmas, vocabulary_reduction_percent, limitations` |
| | `lemmatization/lemmatization_examples.csv` | per-word outcome of four methods |
| POS | `pos/default_pos_distribution.csv` | `tag, coarse_category, count, percent_of_tokens, documents_containing_tag` |
| | `pos/default_pos_coarse.csv` | `coarse_category, count, percent_of_tokens` |
| | `pos/default_pos_examples.csv` | `text, tagged_tokens, pos_sequence` |
| | `pos/custom_pos_dictionary.csv` | `term, default_tag, default_tag_share, custom_tag, rule, reason, occurrences_in_corpus, documents_affected, example` |
| | `pos/custom_pos_changes.csv` | `token, token_index, default_tag, custom_tag, sentence` (observed corrections) |
| | `pos/custom_pos_candidates.csv` | `term, occurrences, dominant_tag, rule, correction_applies` |
| | `pos/ml_pos_results.csv` | `classifier, features, training_tokens, accuracy, macro_f1, default_spacy_accuracy_same_subset, …` |
| | `pos/ml_pos_classification_report.csv` | per-label `precision, recall, f1, support` |
| | `pos/gold_annotation_alignment.csv` | `sentence_id, token, gold_upos, spacy_tag, status` |
| | `pos/pos_gold_errors.csv` | `method, token, gold_pos, predicted_pos, error_type` |
| NER | `ner/ner_entities.csv` | `entity_id, document_id, page_number, unit_id, section_id, section_number, section_title, entity_text, entity_label, token_start, token_end, context` (26,793) |
| | `ner/custom_ner_results.csv` | `entity_id, unit_id, entity_text, domain_category, also_tagged_by_general_ner, context` (4,951) |
| | `ner/ner_label_distribution.csv` | `entity_label, count, percent_of_entities` (18 labels) |
| | `ner/domain_entity_dictionary.csv` | `term, domain_category, corpus_mentions, documents_mentioned, justification` |
| | `ner/ner_error_analysis.csv` | `text, default_label, expected_or_interpreted_label, error_type, context` (146) |
| n-grams | `ngrams/{unigram,bigram,trigram,fourgram,fivegram}_results.csv` | `ngram, n, frequency, document_frequency, example_document, example_page, example_unit_id, contains_domain_term, is_function_word_phrase, has_content_token, is_repeated_token` |
| | `ngrams/domain_phrases.csv` | `ngram, frequency, document_frequency, domain_terms_in_phrase, domain_relevance_reason, example_context` |
| BPE | `bpe/bpe_vocabulary.csv` | `token_id, token, is_special` (8,000) |
| | `bpe/bpe_statistics.csv` | `metric, value, detail` (12 metrics) |
| | `bpe/bpe_examples.csv` | `example, category, bpe_tokens, bpe_token_count, is_single_token, pieces_rejoined` |
| | `bpe/tokenizer_model/{tokenizer.json, vocab.json, merges.txt, training_manifest.json}` | trained model |
| Cross | `comparisons/phase2_master_comparison.csv` | one row per experiment |
| | `comparisons/representative_sample.csv` | 10 required categories, full provenance |
| | `comparisons/pos_comparison.csv`, `comparisons/tokenization_method_comparison.csv`, `comparisons/ngram_comparison.csv` | method-level summaries |
| | `phase2_summary.json` | corpus block + every experiment as a list |
| | `phase2_summary.csv` | `experiment, method, total_tokens, unique_tokens, vocabulary_size, main_metric, main_result, notes` |
| | `phase2_validation_report.csv` | 14 rules, all PASS |
| | `figures/01..11_*.png` | 11 charts |
| | `README.md`, `summaries/phase2_methodology_justification.md` | narrative |

### 3.2 Headline measured values (as they stand)

- Token counts: NLTK 349,891 · spaCy 366,035 · custom 346,276 · **hybrid 344,722**
- Typed date/number expressions: 49,926 detected, 6,413 (12.85%) survive standard
  tokenization intact
- POS gold accuracy: spaCy default 0.9223 · custom ML 0.8769 · custom rules 0.7913
  (412 gold tokens, 30 sentences, single annotator)
- NER: 26,793 general entities + 4,951 domain mentions across 8 domain categories
- BPE: vocabulary 8,000, 7,276 merges, 35.66 bytes/token, 469,401 training tokens

---

## 4. What Phase 3 produces (pipelines, index, retrieval)

**Code:** `src/phase3/*.py`. **Config:** `config/phase3_config.yaml`.
**Public entry points used by Phase 4:**

```python
from src.phase3.config import load_config                  # Phase3Config
from src.phase3.inverted_index import InvertedIndex       # .load(path), .posting_list(term)
from src.phase3.pipelines import load_pipeline_specs       # {'pipeline_a': ..., 'pipeline_b': ...}
from src.phase3.pipeline_runner import PipelineRunner      # .normalize_query(spec, q)
from src.phase3.retrieval import build_engine              # -> RetrievalEngine
from src.phase2.load_corpus import load_corpus             # -> Corpus / Unit
```

### 4.1 The two pipelines

| | Pipeline A | Pipeline B **(selected)** |
|---|---|---|
| Order | `tokenize → date_number → stopwords → morphology → pos → ner → ngrams` | `tokenize → date_number → morphology → stopwords → pos → ner → ngrams` |
| Stopwords | `domain_aware` | `standard_english` |
| Morphology | `lemmatize` (WordNet, POS-agnostic) | `stem` (Porter) |
| Index terms | 22,552 | 21,305 |
| Total postings | 186,189 | 191,817 |
| Total tokens | 241,977 | 246,965 |
| Processing time | 12.616 s | 21.636 s |
| Weighted score | 0.785496 | **0.827162** |

Selection criteria and weights (`config/phase3_config.yaml` → `selection.criteria`):
financial-expression preservation 0.35 · domain-term recall 0.25 · variant-collapse rate
0.25 · query answerability 0.15. `processing_time_seconds` is measured, reported and
**excluded from the score**. The margin (+0.0417) comes entirely from variant collapse
(7/12 vs 5/12); the other three criteria tie.

### 4.2 Inverted index

`results/phase3/inverted_index.json` (9.9 MB) is loaded with `InvertedIndex.load(path)`.
Measured load time on this machine: **~1.2 s**. The API loads it lazily and caches it.

Statistics (`results/phase3/index_statistics.csv` = `metric, value`;
`index_manifest.json` → `statistics`):

| metric | value |
|---|---|
| index terms | 21,305 |
| unigram terms / phrase terms | 18,735 / 2,570 |
| total postings | 191,817 |
| postings per term | 9.0034 |
| indexed units / documents | 6,005 / 31 |
| total positions | 226,368 |
| singleton terms | 10,857 |
| longest posting list | 1,074 (`thi`) |
| mean document frequency | 3.2179 |

### 4.3 Retrieval

`RetrievalEngine` methods: `search_keyword`, `search_phrase`, `search_boolean_and`,
`search_boolean_or`, `search_boolean_not`, and `search(query, query_type=None)` which
auto-detects via `query_type_of`. `SearchOutcome` exposes `matched_terms`, `missing_terms`,
`unit_hits`, `execution_time_ms`, `unit_count`, `document_count`, `document_hits`.
`UnitHit.to_row()` emits: `unit_id, document_id, source_id, page_number, section_id,
section_number, section_title, unit_type, score, matched_terms, matched_term_count,
matched_term_frequency, phrase_match, snippet`.

Ranking (config `retrieval.ranking`):
`score = 1.0 × (distinct matched terms) + 2.0 (if the phrase occurs adjacently)
        + 0.25 × log10(1 + total matched term frequency)`, tie-break `unit_id` ascending.
Snippet = 220 chars around the first matched surface form, 90 chars of leading context.
`max_results_per_query = 50`.

Boolean grammar: recursive-descent parser in `src/phase3/query_parser.py`
(`AND`, `OR`, `A AND NOT B`, parentheses, `"quoted phrase"`). No `eval`/`exec`.
Unrestricted `NOT` is rejected with `QuerySyntaxError`.

### 4.4 Query set and results

15 queries in `config/phase3_config.yaml` → `queries`, executed into
`results/phase3/query_registry.csv`, `retrieval_results.csv` (506 rows),
`retrieval_summary.csv`, `document_results.csv`, `query_examples/Q01..Q15.txt`.
All 15 answered; mean 239.54 ms/query; validation 18/18 PASS.

---

## 5. Outputs the frontend needs, and the endpoint that serves each

No endpoint returns a value that Python did not compute. Files never leave the server
as raw CSV; they are parsed, filtered and paginated server-side.

| Frontend need | Source artefact(s) | Endpoint |
|---|---|---|
| Landing KPIs, charts | phase1 `validation_report.json`, `document_statistics.csv`, phase2 summary, phase3 `phase3_summary.json` | `GET /api/overview` |
| Document list + filters | `document_registry.csv`, phase1 `document_statistics.csv` | `GET /api/documents` |
| Document detail, pages, sections, units | `corpus.jsonl`, `sections_index.csv`, `pages/*.json` | `GET /api/documents/{document_id}` |
| Assignment Table A statistics | phase1 `document_statistics.csv`, phase2 `phase2_summary.json` | `GET /api/statistics` |
| Tokenizer comparison (NLTK/spaCy/custom/hybrid/BPE) | `tokenization/tokenization_comparison.csv`, `comparisons/bpe_tokenization_comparison.csv` | `GET /api/tokenization` |
| Cleaning → stopword → stem → lemma chain | `preprocessing/preprocessing_comparison.csv`, `before_after_examples.csv` | `GET /api/preprocessing` |
| Stemming | `stemming/stemming_comparison.csv`, `stemming_examples.csv`, `stopword_stemming_order_comparison.csv` | `GET /api/stemming` |
| Lemmatization | `lemmatization/lemmatization_comparison.csv`, `lemmatization_examples.csv` | `GET /api/lemmatization` |
| POS (default / rules / ML) | `pos/*.csv` | `GET /api/pos` |
| NER (general + domain) | `ner/ner_entities.csv`, `custom_ner_results.csv`, `ner_label_distribution.csv`, `ner_error_analysis.csv` | `GET /api/ner` |
| n-grams 1–5 | `ngrams/*gram_results.csv`, `domain_phrases.csv` | `GET /api/ngrams?n=` |
| BPE | `bpe/bpe_statistics.csv`, `bpe_vocabulary.csv`, `bpe_examples.csv`, tokenizer model | `GET /api/bpe` |
| Pipeline A vs B | `pipeline_comparison.csv`, `final_pipeline.json` | `GET /api/pipelines` |
| Index statistics | `index_statistics.csv`, `index_manifest.json` | `GET /api/index/statistics` |
| Term lookup | `inverted_index.json` (via `InvertedIndex.posting_list`) | `GET /api/index/term/{term}` |
| Keyword / phrase / AND / OR / NOT / grouped | `RetrievalEngine` | `POST /api/search`, `POST /api/boolean-search` |
| Query set | `query_registry.csv` | `GET /api/queries`, `GET /api/queries/{query_id}` |
| Unit context viewer | `corpus.jsonl` | `GET /api/units/{unit_id}` |
| Evaluation | `results/phase4/evaluation_results.csv` | `GET /api/evaluation`, `/api/evaluation/summary` |
| Relevance judgments | `results/phase4/relevance_judgments.csv` | `GET /api/evaluation/relevance`, `POST /api/evaluation/relevance` |
| Per-phase inventory | every `results/*/README.md` + summary JSON | `GET /api/results/phase{1,2,3,4}` |
| Liveness | — | `GET /api/health` |

---

## 6. How Next.js talks to Python

```
Browser (Next.js client components, client-side hooks)
   │  fetch, JSON only
   ▼
NEXT_PUBLIC_API_BASE_URL  (default http://localhost:8000)
   │  CORS preflight + request
   ▼
FastAPI  backend/  (uvicorn, python -m backend.main)
   │  in-process calls, no reimplementation
   ├── backend/services/retrieval.py      → src.phase3.retrieval.RetrievalEngine
   ├── backend/services/artifacts.py      → csv/json readers over results/**
   ├── backend/services/corpus.py         → src.phase2.load_corpus.load_corpus
   └── backend/services/evaluation.py     → src.phase4.evaluator
   ▼
results/phase{1,2,3,4}/** and data/corpus/**
```

Rules enforced in code:

- One API client, `frontend/lib/api.ts`. No component calls `fetch` directly.
- The base URL comes from `NEXT_PUBLIC_API_BASE_URL` only.
- Backend errors are mapped to 400 (invalid query / invalid id), 404 (unknown
  document/unit/term), 422 (Pydantic validation) and 500 (unexpected). 500 responses
  carry a generic message; the traceback goes to the backend log, never to the browser.
- Large collections (`ner_entities.csv` 6.5 MB, `inverted_index.json` 9.9 MB,
  n-gram tables 0.3–0.4 MB each) are **never** shipped whole. They are read once,
  cached in the service, and served with `limit` / `offset` / filter parameters.
- Identifiers are validated against `^[A-Za-z0-9_.:-]{1,120}$` and matched against
  registries before any file access. The API accepts **no** filesystem path from the
  client; it resolves paths itself from `results/` and `data/corpus/`.
- The index is loaded lazily on the first search/term request and reused, so a cold
  start does not pay the ~1.2 s index parse.

---

## 7. New Phase 4 code

### 7.1 `src/phase4/` — evaluation layer (computational, Python only)

| Module | Responsibility |
|---|---|
| `config.py` | `config/phase4_config.yaml` loader, managed-output list, logging |
| `pooling.py` | builds the judgment pool from Phase 3 retrieval (depth declared in config) |
| `relevance.py` | `RelevanceStore` — read/validate/upsert `results/phase4/relevance_judgments.csv` |
| `metrics.py` | precision, recall, F1, P@K, R@K, macro/micro averages, explicit zero handling |
| `evaluator.py` | runs the 15 queries through the real engine, joins judgments, computes metrics |
| `reporting.py` | writes the Phase 4 artefact set |
| `validation.py` | the Phase 4 validation rules |
| `run.py` | `python -m src.phase4.run` |

**Relevance judgments are human, not derived.** `relevance_judgments.csv` columns:
`query_id, query, unit_id, rank, document_id, page_number, section_number, relevance,
notes, annotator, judgment_method, judged_at`. `relevance ∈ {0, 1}`. Judgments are
pooled (top-`k` retrieved per query) and the pool depth is recorded in the artefact.
No label is ever computed from a similarity score, because a metric derived from the
ranker that is being measured would not measure anything.

### 7.2 `backend/` — FastAPI bridge (no algorithms)

```
backend/
  main.py            app factory, CORS, exception handlers, /api router mount
  config/settings.py paths, CORS origins, pagination caps
  api/               system.py corpus.py experiments.py index.py retrieval.py evaluation.py
  schemas/           pydantic request/response models
  services/          artifacts.py corpus.py retrieval.py experiments.py evaluation.py
```

### 7.3 `frontend/` — Next.js presentation only

App Router, TypeScript, Tailwind CSS. `lib/api.ts` is the only network boundary;
`types/api.ts` mirrors the Pydantic models; `hooks/useApi.ts` provides
loading / error / empty states. Recharts renders charts built from backend arrays.

### 7.4 Phase 4 artefacts

```
results/phase4/
  relevance_judgments.csv        manual labels (query_id, unit_id, relevance, notes, …)
  evaluation_results.csv         per (query, pipeline) precision/recall/F1/P@K/R@K
  evaluation_summary.csv         macro / micro / per-query-type aggregates
  pipeline_final_comparison.csv  Table J: both pipelines × all available metrics
  final_summary.csv              one flat row per project-level metric
  final_summary.json             the whole Phase 4 report, machine readable
  gui_summary.json               exactly the payload the dashboard renders
  validation_report.csv          Phase 4 validation rules
```

---

## 8. Integration flow

```
USER
  │ Next.js dashboard
  ├── /documents      → /api/documents        → document_registry.csv + document_statistics.csv
  ├── /statistics     → /api/statistics       → phase1 + phase2 statistics
  ├── /tokenization   → /api/tokenization     → tokenization_comparison.csv
  ├── /preprocessing  → /api/preprocessing    → preprocessing_comparison.csv
  ├── /pos            → /api/pos              → pos/*.csv
  ├── /ner            → /api/ner              → ner/*.csv
  ├── /ngrams         → /api/ngrams           → ngrams/*gram_results.csv
  ├── /bpe            → /api/bpe              → bpe/*.csv
  ├── /pipelines      → /api/pipelines        → pipeline_comparison.csv
  ├── /index          → /api/index/*          → inverted_index.json
  ├── /search         → /api/search           → RetrievalEngine (Phase 3)
  └── /evaluation     → /api/evaluation/*     → src.phase4 metrics
        │
        ▼
FastAPI ──► Phase 1 corpus · Phase 2 experiments · Phase 3 index+retrieval · Phase 4 evaluation
```

---

## 9. Preserved behaviour — what Phase 4 must never do

- Modify the 31 source PDFs. (They are not in this tree; `data/source_pdfs/` holds only
  a `.gitkeep`.)
- Delete or rewrite any `results/phase1`, `results/phase2` or `results/phase3` artefact.
  Phase 4 writes only inside `results/phase4`; its cleaner is restricted to an explicit
  managed-file list and refuses to touch anything outside that directory.
- Rerun the expensive NLP experiments. Phase 4 reads CSVs; it never calls
  `src.phase2.run` or `src.phase3.run`.
- Reimplement any NLP step in TypeScript.
- Report a number that Python did not compute. Where a metric does not exist the API
  returns `null` and the UI renders `N/A — not calculated`.
