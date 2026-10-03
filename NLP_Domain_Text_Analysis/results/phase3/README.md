# Phase 3 results - pipeline comparison and retrieval

Phase 3 consumes Phase 1 (structured corpus) and Phase 2 (NLP experiments) as inputs and never rewrites either phase's artefacts.

## Outcome

- Selected pipeline: **Pipeline B - stemming then stopword removal** (`pipeline_b`), score 0.8272
- Runner-up margin: +0.0417
- Index: 21305 terms (18735 unigrams, 2570 phrases), 191817 postings, 6005 content units, 31 documents
- Queries: 15/15 answered (100.0%), 506 unit results in total
- Validation: 18/18 rules PASS

## Files

| file | what it contains |
|---|---|
| phase3/document_results.csv | The same queries aggregated to document level |
| phase3/final_pipeline.json | The selection decision as data (scores, weights, reason, tie-breakers) |
| phase3/final_pipeline.md | The selected pipeline, its index statistics and its query coverage |
| phase3/index_manifest.json | What was indexed, from which inputs, with what settings and hashes |
| phase3/index_statistics.csv | Index size, distribution and concentration measurements |
| phase3/inverted_index.json | The serialized inverted index: terms, postings, positions and provenance |
| phase3/phase3_summary.json | The machine-readable summary of the whole Phase 3 run |
| phase3/pipeline_comparison.csv | One row per pipeline with every criterion, raw rate and weighted score |
| phase3/pipeline_comparison.md | The A vs B write-up, with the Phase 2 evidence it relies on |
| phase3/query_examples/Q01.txt | Readable result example for one query |
| phase3/query_examples/Q02.txt | Readable result example for one query |
| phase3/query_examples/Q03.txt | Readable result example for one query |
| phase3/query_examples/Q04.txt | Readable result example for one query |
| phase3/query_examples/Q05.txt | Readable result example for one query |
| phase3/query_examples/Q06.txt | Readable result example for one query |
| phase3/query_examples/Q07.txt | Readable result example for one query |
| phase3/query_examples/Q08.txt | Readable result example for one query |
| phase3/query_examples/Q09.txt | Readable result example for one query |
| phase3/query_examples/Q10.txt | Readable result example for one query |
| phase3/query_examples/Q11.txt | Readable result example for one query |
| phase3/query_examples/Q12.txt | Readable result example for one query |
| phase3/query_examples/Q13.txt | Readable result example for one query |
| phase3/query_examples/Q14.txt | Readable result example for one query |
| phase3/query_examples/Q15.txt | Readable result example for one query |
| phase3/query_registry.csv | The 15 domain queries with their normalization, results and timing |
| phase3/README.md | How to read this directory |
| phase3/retrieval_results.csv | One row per retrieved content unit, ranked, with provenance and snippet |
| phase3/retrieval_summary.csv | One row per query: answerability, coverage and execution time |
| phase3/validation_report.csv | Every Phase 3 validation rule with its status and evidence |

## Query set

| id | query | type | units | documents | answered |
|---|---|---|---|---|---|
| Q01 | inflation | keyword | 50 | 7 | yes |
| Q02 | GDP | keyword | 50 | 14 | yes |
| Q03 | FDI | keyword | 50 | 4 | yes |
| Q04 | monetary policy | phrase | 23 | 6 | yes |
| Q05 | financial stability | phrase | 25 | 9 | yes |
| Q06 | current account deficit | phrase | 6 | 5 | yes |
| Q07 | GDP AND inflation | boolean_and | 11 | 7 | yes |
| Q08 | RBI AND "repo rate" | boolean_and | 3 | 2 | yes |
| Q09 | banking AND credit | boolean_and | 50 | 11 | yes |
| Q10 | GDP OR GVA | boolean_or | 50 | 10 | yes |
| Q11 | inflation OR disinflation | boolean_or | 50 | 7 | yes |
| Q12 | inflation AND NOT food | boolean_not | 50 | 7 | yes |
| Q13 | banking AND NOT insurance | boolean_not | 50 | 14 | yes |
| Q14 | GDP AND investment | boolean_and | 23 | 10 | yes |
| Q15 | (GDP OR GVA) AND policy | boolean_group | 15 | 9 | yes |

## Inputs used

Phase 2 evidence cited by the comparison:

- Phase 2 tokenization: hybrid produced 344722 tokens (fewest of the four methods) and kept 2258 percentage tokens against NLTK's 361 (tokenization/tokenization_comparison.csv)
- Phase 2 tokenization: hybrid kept 1805 fiscal-year tokens against spaCy's 659 (tokenization/tokenization_comparison.csv)
- Phase 2 tokenization: the custom rule tokenizer detected 2170 percentage tokens against NLTK's 361 (tokenization/tokenization_comparison.csv)
- Phase 2 date/number experiment: 49926 typed financial expressions were detected but only 6413 (12.85%) survived standard tokenization intact (tokenization/date_number_comparison.csv)
- Phase 2 stopwords: standard list removed 103444 tokens (30.01%), domain-aware 102745 (29.81%), because 43 protected financial terms survive (preprocessing/stopword_comparison.csv)
- Phase 2 order experiment: stopword-removal-then-stemming left 241977 tokens, stemming-then-stopword-removal left 247815; the wrong order left 1 unmatched stems such as 'abov', 'everi', 'furthermor' (stemming/stopword_stemming_order_comparison.csv)
- Phase 2 lemmatization: WordNet POS-agnostic lookup reduced the vocabulary to 12195 (23.44% reduction) (lemmatization/lemmatization_comparison.csv)
- Phase 2 lemmatization: spaCy's rule lemmatizer kept 13330 lemmas, i.e. it normalizes less aggressively than the WordNet lookup (lemmatization/lemmatization_comparison.csv)
- Phase 2 POS: default_pos scored 0.9223 accuracy on 412 manually annotated tokens (comparisons/pos_comparison.csv)
- Phase 2 POS: custom_rule_pos scored 0.7913 accuracy on 412 manually annotated tokens (comparisons/pos_comparison.csv)
- Phase 2 POS: custom_ml_pos scored 0.8769 accuracy on 130 manually annotated tokens (comparisons/pos_comparison.csv)

## How to reproduce

```
python -m src.phase3.run
```

Random seed: 42. All ordering is deterministic (posting lists sorted by `unit_id`, ties in ranking broken by `unit_id`), so a second run produces byte-identical result tables.

