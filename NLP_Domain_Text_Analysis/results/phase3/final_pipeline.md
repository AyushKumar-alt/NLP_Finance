# Phase 3 - Final pipeline: Pipeline B - Stemming (Snowball English)

Pipeline key: `pipeline_b`

## Component order

| # | step | component |
|---|---|---|
| 1 | tokenize | tokenization (hybrid) |
| 2 | date_number | date/number handling (typed date/number detection with expression protection) |
| 3 | stopwords | stopword handling (domain-aware (protected financial terms kept)) |
| 4 | morphology | morphology (stemming (snowball_english)) |
| 5 | pos | POS tagging metadata (phase2_evidence) |
| 6 | ner | NER metadata (phase2_consumed) |
| 7 | ngrams | n-gram generation (1..5) |

Index representation: `stem_terms_with_positions` (case-folded, positional, provenance preserved per posting).

## Why this pipeline

Scores are equal (0.8265) and no tie-breaker separated the pipelines; the lexicographically first pipeline name is kept.

## Selected index

| metric | value |
|---|---|
| index terms | 21403 |
| unigram terms | 18763 |
| phrase terms (2-3 grams, min frequency 5) | 2640 |
| postings | 195567 |
| postings per term | 9.1374 |
| positions stored | 230004 |
| content units indexed | 6134 |
| documents indexed | 31 |
| units per document | 197.87 |
| singleton terms | 10869 |
| hapax ratio | 0.5078 |
| terms holding 90% of postings | 6931 |
| longest posting list | 's (1043) |

## Retrieval behaviour

| metric | value |
|---|---|
| queries executed | 15 |
| queries answered | 15 |
| answerability | 1.0 |
| units returned in total | 507 |
| mean units per query | 33.8 |
| median units per query | 50 |
| mean documents per query | 8.0667 |
| mean execution time (ms) | 166.437 |
| max execution time (ms) | 489.168 |

## Query coverage

| query | type | units | documents | missing terms | ms |
|---|---|---|---|---|---|
| Q01 inflation | keyword | 50 | 7 | - | 230.182 |
| Q02 GDP | keyword | 50 | 14 | - | 319.701 |
| Q03 FDI | keyword | 50 | 4 | - | 116.175 |
| Q04 monetary policy | phrase | 23 | 6 | - | 64.235 |
| Q05 financial stability | phrase | 25 | 9 | - | 109.590 |
| Q06 current account deficit | phrase | 6 | 5 | - | 33.320 |
| Q07 GDP AND inflation | boolean_and | 11 | 7 | - | 38.632 |
| Q08 RBI AND "repo rate" | boolean_and | 3 | 2 | - | 10.492 |
| Q09 banking AND credit | boolean_and | 50 | 11 | - | 162.182 |
| Q10 GDP OR GVA | boolean_or | 50 | 10 | - | 386.621 |
| Q11 inflation OR disinflation | boolean_or | 50 | 7 | - | 223.746 |
| Q12 inflation AND NOT food | boolean_not | 50 | 7 | - | 190.090 |
| Q13 banking AND NOT insurance | boolean_not | 50 | 13 | - | 489.168 |
| Q14 GDP AND investment | boolean_and | 24 | 10 | - | 72.835 |
| Q15 (GDP OR GVA) AND policy | boolean_group | 15 | 9 | - | 49.592 |

Answerability: 15/15 (100.0%). Expected domains covered: banking, economic, financial_stability, regulator.

Retrieval quality (precision/recall of these results) is **not** measured here; that is the job of Phase 4.

## Traceability

Every result row keeps the full Phase 1 chain: `source_id -> document_id -> page_number -> section_id -> unit_id`, plus a snippet of the matched text.

