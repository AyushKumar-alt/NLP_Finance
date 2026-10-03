# Phase 3 - Final pipeline: Pipeline B - stemming then stopword removal

Pipeline key: `pipeline_b`

## Component order

| # | step | component |
|---|---|---|
| 1 | tokenize | tokenization (hybrid) |
| 2 | date_number | date/number handling (typed date/number detection with expression protection) |
| 3 | morphology | morphology (stemming (porter)) |
| 4 | stopwords | stopword handling (standard English list) |
| 5 | pos | POS tagging metadata (phase2_evidence) |
| 6 | ner | NER metadata (phase2_consumed) |
| 7 | ngrams | n-gram generation (1..5) |

Index representation: `stem_terms_with_positions` (case-folded, positional, provenance preserved per posting).

## Why this pipeline

Pipeline B - stemming then stopword removal wins on the weighted score (0.8272 vs 0.7855, margin +0.0417). Decomposition: financial_expression_preservation 3048/3792 = 0.8038 x0.35 = 0.2813; domain_term_recall 101/101 = 1.0000 x0.25 = 0.2500; variant_collapse_rate 7/12 = 0.5833 x0.25 = 0.1458; query_answerability 15/15 = 1.0000 x0.15 = 0.1500. The two pipelines are equal on 3 of 4 criteria; the margin comes from variant_collapse_rate (7/12 = 0.5833 vs 5/12 = 0.4167, +0.1667 x0.25 = +0.0417).

## Selected index

| metric | value |
|---|---|
| index terms | 21305 |
| unigram terms | 18735 |
| phrase terms (2-3 grams, min frequency 5) | 2570 |
| postings | 191817 |
| postings per term | 9.0034 |
| positions stored | 226368 |
| content units indexed | 6005 |
| documents indexed | 31 |
| units per document | 193.71 |
| singleton terms | 10857 |
| hapax ratio | 0.5096 |
| terms holding 90% of postings | 6934 |
| longest posting list | thi (1074) |

## Retrieval behaviour

| metric | value |
|---|---|
| queries executed | 15 |
| queries answered | 15 |
| answerability | 1.0 |
| units returned in total | 506 |
| mean units per query | 33.7333 |
| median units per query | 50 |
| mean documents per query | 8.1333 |
| mean execution time (ms) | 239.54 |
| max execution time (ms) | 536.471 |

## Query coverage

| query | type | units | documents | missing terms | ms |
|---|---|---|---|---|---|
| Q01 inflation | keyword | 50 | 7 | - | 476.482 |
| Q02 GDP | keyword | 50 | 14 | - | 357.755 |
| Q03 FDI | keyword | 50 | 4 | - | 157.501 |
| Q04 monetary policy | phrase | 23 | 6 | - | 149.710 |
| Q05 financial stability | phrase | 25 | 9 | - | 372.893 |
| Q06 current account deficit | phrase | 6 | 5 | - | 140.480 |
| Q07 GDP AND inflation | boolean_and | 11 | 7 | - | 123.265 |
| Q08 RBI AND "repo rate" | boolean_and | 3 | 2 | - | 21.094 |
| Q09 banking AND credit | boolean_and | 50 | 11 | - | 200.285 |
| Q10 GDP OR GVA | boolean_or | 50 | 10 | - | 406.587 |
| Q11 inflation OR disinflation | boolean_or | 50 | 7 | - | 272.722 |
| Q12 inflation AND NOT food | boolean_not | 50 | 7 | - | 233.770 |
| Q13 banking AND NOT insurance | boolean_not | 50 | 14 | - | 536.471 |
| Q14 GDP AND investment | boolean_and | 23 | 10 | - | 83.681 |
| Q15 (GDP OR GVA) AND policy | boolean_group | 15 | 9 | - | 60.405 |

Answerability: 15/15 (100.0%). Expected domains covered: banking, economic, financial_stability, regulator.

Retrieval quality (precision/recall of these results) is **not** measured here; that is the job of Phase 4.

## Traceability

Every result row keeps the full Phase 1 chain: `source_id -> document_id -> page_number -> section_id -> unit_id`, plus a snippet of the matched text.

