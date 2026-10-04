# Phase 2 - NLP experimentation results

Domain: **Financial / Economic documents**. Phase 2 consumes the Phase 1 structured
corpus and runs the twenty required experiments. It never modifies Phase 1 outputs
or the source PDFs.

## 1. What Phase 2 does

Phase 1 produced a traceable structured corpus. Phase 2 compares how different
NLP techniques behave on that financial/economic text: tokenization, date and number
handling, stopword strategies, stemming, lemmatization, POS tagging, NER, n-grams and
BPE. Every experiment is measured on the same text, every result is written to CSV or
JSON, and every example keeps its `document_id / page_number / section / unit_id`.

## 2. Input from Phase 1

* `data/corpus/structured/corpus.jsonl` - 8311 units from
  31 documents (885 pages)
* `data/corpus/metadata/document_registry.csv`, `source_registry.csv` - provenance
* text selection policy: **`prose_tables`** =
  paragraph, reference, box, table -> 6134 units,
  2081530 characters

The loader builds `Corpus -> Document -> Page -> Section -> Unit`, so a result can
always be cited as `document -> page -> section -> unit`.

## 3. Tokenization methods

| method | implementation |
|---|---|
| nltk | `nltk.tokenize.word_tokenize` (Treebank + Punkt) |
| spacy | `en_core_web_sm` tokenizer (install with `python -m spacy download en_core_web_sm`) |
| custom | 17 ordered financial rules in `src/phase2/custom_tokenizer.py` |
| hybrid | NLTK + domain protection/repair layer |
| bpe | byte-level BPE trained on this corpus (`tokenizers`) |
| date/number | typed date and number tokenizer (`src/phase2/date_number_tokenizer.py`) |

## 4. Custom financial tokenization rules

Rules preserve fiscal years (`FY2025-26`), percentages (`7.4%`, `7.4 per cent`),
currency with scale words (`Rs 1.25 lakh crore`, `US$ 2.5 billion`), Indian digit
grouping (`1,25,000`), quarters (`Q1 FY26`), dates (`31 March 2026`), dotted
abbreviations (`B.Tech`), acronyms and hyphenated terms (`year-on-year`, `repo-rate`).
Every rule, its pattern, its example and its observed behaviour is in
`tokenization/custom_tokenizer_rules.csv`.

## 5. Stopword strategy

Three strategies: none, standard English list, and domain-aware
(standard minus protected financial terms plus domain function words). The
`standard English list` source actually used is recorded in every row of
`preprocessing/stopword_comparison.csv` (NLTK was not downloadable in this
environment, so spaCy's published list is used and recorded).

## 6. Stemming

Porter, Snowball (english) and Lancaster are compared with vocabulary reduction,
collision counts, observed word families and the over/under-stemming observations for
financial terms. The stopword-then-stem vs stem-then-stopword order is measured too.

## 7. Lemmatization

WordNet-derived lookup (POS-agnostic and POS-aware using the WordNet surface index),
spaCy's rule-based lemmatizer, and lemminflect's rule-based morphology. The POS
dependency of each method is stated explicitly.

## 8. POS methods

1. default: spaCy `en_core_web_sm`
2. custom rule/dictionary: built from observed tag distributions
3. custom ML: `SGDClassifier` over shape/affix/context features, trained on silver
   (spaCy) + gold (manual) data, evaluated only on held-out manually annotated sentences

## 9. NER methods

1. general NER (spaCy) with all labels reported
2. financial domain entity layer with 8 categories, complementary to (not replacing)
   the general model
3. error analysis split into systematic and manually reviewed rows

## 10. N-gram analysis

Unigram through 5-gram with frequency, document frequency and an example trace.
Domain phrases are mined with explicit thresholds. The effect of stopword removal on
phrase quality is reported.

## 11. BPE

Trained on the experiment sample with `min_frequency` and `vocab_size` from
`config/phase2_config.yaml`, persisted to `bpe/tokenizer_model`, and analysed for
common words, rare words, acronyms, numbers and dates.

## 12. Measured results

### Tokenization (whole experiment sample)

| method | tokens | vocabulary | date tokens | percentage tokens | currency tokens |
|---|---|---|---|---|---|
| nltk | 349344 | 18035 | 6939 | 359 | 1016 |
| custom | 345677 | 18628 | 6169 | 2167 | 1046 |
| spacy | 365068 | 15899 | 7090 | 354 | 1036 |
| hybrid | 344141 | 19463 | 6613 | 2255 | 1155 |

### Date and number aware tokenization

| category | expressions detected | surviving standard tokenization | intact rate |
|---|---|---|---|
| TOTAL | 49851 | 6431 | 12.9% |

### Stopwords

| strategy | list size | remaining tokens | token reduction |
|---|---|---|---|
| none | 0 | 344141 | 0.0% |
| standard_english | 198 | 250215 | 27.29% |
| domain_aware | 213 | 249744 | 27.43% |

### Stemming

| stemmer | unique stems | vocabulary reduction | colliding stems |
|---|---|---|---|
| porter | 14243 | 26.77% | 7169 |
| snowball_english | 14250 | 26.73% | 7142 |
| lancaster | 13023 | 33.04% | 6274 |

### Lemmatization

| method | unique lemmas | vocabulary reduction | needs POS |
|---|---|---|---|
| wordnet_lookup_pos_agnostic | 15886 | 0.0% | False |
| wordnet_lookup_pos_aware | 15886 | 0.0% | True |
| spacy_rule_based | 13295 | 16.31% | False |
| lemminflect_rulebased | 13135 | 17.32% | True |

### POS tagging (scored only on the manually annotated gold sample)

| method | tokens scored | gold sentences | accuracy | macro F1 |
|---|---|---|---|---|
| default_pos | 412 | 30 | 0.9223 | 0.7571 |
| custom_rule_pos | 412 | 30 | 0.7913 | 0.6139 |
| custom_ml_pos |  | 0 |  |  |

### Named entities

| measure | value |
|---|---|
| general spaCy entities | 26717 |
| general spaCy labels | 18 |
| domain dictionary mentions | 4968 |
| domain categories | 8 |
| error-analysis rows | 145 |

### N-grams

| n | total | unique | top n-gram with words |
|---|---|---|---|
| 1 | 344141 | 24389 | achieving (x40) |
| 2 | 338007 | 156887 | achieving strategic (x1) |
| 3 | 331878 | 268305 | achieving strategic resilience (x1) |
| 4 | 325751 | 303211 | achieving strategic resilience and (x1) |
| 5 | 319627 | 309454 | achieving strategic resilience and indispensability (x1) |

### BPE

| measure | value |
|---|---|
| vocabulary | 8000 |
| tokens for the whole sample | 468203 |
| bytes per token | 4.446 |

## 13. Output files

Key artefacts (the full list of 107 written files is in
`phase2_summary.json` -> `outputs`):

* `bpe/bpe_examples.csv`
* `bpe/bpe_statistics.csv`
* `bpe/bpe_vocabulary.csv`
* `bpe/tokenizer_model/merges.txt`
* `bpe/tokenizer_model/tokenizer.json`
* `bpe/tokenizer_model/training_manifest.json`
* `bpe/tokenizer_model/vocab.json`
* `bpe/bpe_training_corpus/` - 31 per-document training files (`bpe/bpe_training_corpus/D01.txt` ... `bpe/bpe_training_corpus/D31.txt`)
* `comparisons/bpe_tokenization_comparison.csv`
* `comparisons/domain_tokenization_testset.csv`
* `comparisons/experiment_manifest.csv`
* `comparisons/ngram_comparison.csv`
* `comparisons/phase2_master_comparison.csv`
* `comparisons/pipeline_comparison.csv`
* `comparisons/pipeline_comparison.md`
* `comparisons/pos_comparison.csv`
* `comparisons/representative_sample.csv`
* `comparisons/tokenization_method_comparison.csv`
* `figures/01_token_count_by_tokenizer.png`
* `figures/02_vocabulary_by_tokenizer.png`
* `figures/03_stopword_reduction.png`
* `figures/04_stemming_vocabulary_reduction.png`
* `figures/05_lemmatization_vocabulary_reduction.png`
* `figures/06_pos_distribution.png`
* `figures/07_top_unigrams.png`
* `figures/08_top_bigrams.png`
* `figures/09_top_trigrams.png`
* `figures/10_ngram_vocabulary_growth.png`
* `figures/11_bpe_token_vocabulary_comparison.png`
* `lemmatization/lemmatization_comparison.csv`
* `lemmatization/lemmatization_examples.csv`
* `ner/custom_ner_results.csv`
* `ner/domain_entity_dictionary.csv`
* `ner/ner_entities.csv`
* `ner/ner_error_analysis.csv`
* `ner/ner_label_distribution.csv`
* `ngrams/bigram_results.csv`
* `ngrams/domain_phrases.csv`
* `ngrams/fivegram_results.csv`
* `ngrams/fourgram_results.csv`
* `ngrams/trigram_results.csv`
* `ngrams/unigram_results.csv`
* `pos/custom_pos_candidates.csv`
* `pos/custom_pos_changes.csv`
* `pos/custom_pos_dictionary.csv`
* `pos/default_pos_coarse.csv`
* `pos/default_pos_distribution.csv`
* `pos/default_pos_examples.csv`
* `pos/gold_annotation_alignment.csv`
* `pos/ml_pos_classification_report.csv`
* `pos/ml_pos_misclassifications.csv`
* `pos/ml_pos_results.csv`
* `pos/pos_gold_errors.csv`
* `preprocessing/before_after_examples.csv`
* `preprocessing/domain_stopword_analysis.csv`
* `preprocessing/final_stopwords.txt`
* `preprocessing/preprocessing_comparison.csv`
* `preprocessing/stopword_comparison.csv`
* `preprocessing/stopwords_protected_terms.csv`
* `README.md`
* `phase2_summary.csv`
* `phase2_summary.json`
* `phase2_validation_report.csv`
* `stemming/stemming_comparison.csv`
* `stemming/stemming_examples.csv`
* `stemming/stemming_families.csv`
* `stemming/stopword_stemming_order_comparison.csv`
* `summaries/phase2_methodology_justification.md`
* `tokenization/custom_examples.csv`
* `tokenization/custom_tokenizer_rules.csv`
* `tokenization/date_number_comparison.csv`
* `tokenization/date_number_patterns.csv`
* `tokenization/hybrid_examples.csv`
* `tokenization/nltk_examples.csv`
* `tokenization/spacy_examples.csv`
* `tokenization/tokenization_comparison.csv`
* `tokenization/tokenization_examples_comparison.csv`

Figures: 01_token_count_by_tokenizer, 02_vocabulary_by_tokenizer, 03_stopword_reduction, 04_stemming_vocabulary_reduction, 05_lemmatization_vocabulary_reduction, 06_pos_distribution, 07_top_unigrams, 08_top_bigrams, 09_top_trigrams, 10_ngram_vocabulary_growth, 11_bpe_token_vocabulary_comparison

## 13. How to reproduce

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.phase2.run
```

Configuration: `config/phase2_config.yaml` (all paths, policies, seeds and
thresholds). Random seed is fixed for the ML experiment.

## 14. Limitations

* The environment could not download the NLTK `wordnet`, `stopwords` or
  `averaged_perceptron_tagger` corpora (network policy). Real substitutes are used and
  the substitution is recorded in the artefacts: spaCy's stopword list, the
  WordNet-derived table shipped with `spacy-lookups-data`, and spaCy for default POS.
* The POS gold set is a single-annotator sample of
  412 tokens, so POS accuracy figures carry
  real uncertainty and are reported with the sample size. The annotated sentences were
  re-verified against the corpus after annotation; provenance columns were corrected
  where a sentence was traced to the wrong unit, and the UPOS labels themselves were
  never edited.
* The domain entity layer is dictionary based and context free. Its *counts* cover the
  whole sample, while `custom_ner_results.csv` keeps at most three mentions per
  category per unit so that table-heavy units cannot flood the table; the cap is
  recorded in `ner_error_analysis.csv` metadata.
* spaCy's parser is run on units of at most
  20000 characters
  (`tokenization.spacy.max_chars_per_chunk`). On this corpus no unit exceeded the limit,
  but if one had, an entity spanning a chunk boundary would not be detected.
* Phase 1 emits `U+FFFD` where a publisher glyph could not be mapped; Phase 2 treats
  it as a separator and never lets it enter a token.
* The Phase 1 text of some map/figure pages contains extraction noise made of short
  letter runs (for example the `AI AI AI ...` legend of a regional map in D17). The
  n-gram tables keep it because it is genuinely in the corpus and flag it with
  `is_repeated_token`; the top-n-gram charts skip such runs, because an n-gram made of
  one repeated token carries no collocational information.

## 15. What Phase 3 will consume from Phase 2

* the chosen tokenization representation (domain-aware, with percentages and fiscal
  years intact) and the token vocabulary per document,
* the unit-level provenance table so a retrieved hit can be cited as
  document / page / section / unit,
* entity rows (`ner_entities.csv`, `custom_ner_results.csv`) for entity filters,
* n-gram counts with document frequency for phrase postings,
* the measured trade-offs (stopword strategy, stemming vs lemmatization) that justify
  the Phase 3 pipeline order.

Phase 3 (indexing and retrieval) is **not** part of this deliverable.
