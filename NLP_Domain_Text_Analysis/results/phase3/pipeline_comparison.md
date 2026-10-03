# Phase 3 - Pipeline A vs Pipeline B

Phase 3 consumes Phase 1 (structured corpus) and Phase 2 (NLP experiments) as inputs and never rewrites either phase's artefacts.

## 1. The two pipelines

| pipeline | tokenizer | stopwords | morphology | index terms are | order |
|---|---|---|---|---|---|
| pipeline_a | hybrid | domain_aware | lemmatization (wordnet_lookup_pos_agnostic) | lemma | tokenize -> date_number -> stopwords -> morphology -> pos -> ner -> ngrams |
| pipeline_b | hybrid | standard_english | stemming (porter) | stem | tokenize -> date_number -> morphology -> stopwords -> pos -> ner -> ngrams |

**pipeline_a**: Baseline order recommended by the assignment; the Phase 2 order experiment (stopword_removal_then_stemming vs stemming_then_stopword_removal) is the evidence for putting stopword removal before morphology.

**pipeline_b**: Tests the effect of changing the processing order: morphology is applied first, so stopword matching happens on stems.

## 2. Phase 2 evidence used as input

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

- Phase 2 sample: 6005 units, 2084810 characters, policy 'prose_tables'

## 3. Measured criteria

Every value below is computed from the artifacts each pipeline produced; nothing is copied from a previous phase except the inputs listed above.

| criterion | pipeline_a | pipeline_b |
|---|---|---|
| financial_expression_preservation<br>(weight 0.35 x = 0.2813) | 0.8038<br>3048/3792 = 0.8038 | 0.8038<br>3048/3792 = 0.8038 |
| domain_term_recall<br>(weight 0.25 x = 0.2500) | 1.0000<br>101/101 = 1.0000 | 1.0000<br>101/101 = 1.0000 |
| variant_collapse_rate<br>(weight 0.25 x = 0.1042) | 0.4167<br>5/12 = 0.4167 | 0.5833<br>7/12 = 0.5833 |
| query_answerability<br>(weight 0.15 x = 0.1500) | 1.0000<br>15/15 = 1.0000 | 1.0000<br>15/15 = 1.0000 |
| **total score** | **0.7855** | **0.8272** |

## 4. What each criterion measures

> Both pipelines preserve exactly 3048 of 3792 financial expressions. That is expected, not a copy-paste error: expression protection runs *before* morphology and the Phase 2 tokenizer already emits `7.4 per cent` as one token, and stemming leaves digits and currency expressions untouched. This criterion therefore confirms the protection step works, but it does not separate the two pipelines on this corpus.

- **financial_expression_preservation** - share of the typed financial expressions found in the raw text (`7.4 per cent`, `FY2025-26`, `Rs 1.25 lakh crore`) that are still a single index term. Phase 2 measured that standard tokenization keeps only 12.85% of them intact, which is why this criterion carries the largest weight.
- **domain_term_recall** - share of the domain vocabulary (the configured reference terms plus the `domain_terms_in_phrase` values Phase 2 exported) that resolves to an index term.
- **variant_collapse_rate** - share of the configured variant groups whose surface forms all map onto one indexed term, so a query for `investment` also finds `investments` and `investors`.
- **query_answerability** - share of the 15 configured domain queries that return at least one result on that pipeline's own index.

Timing is measured and reported but excluded from the score (`exclude_from_score: ['processing_time_seconds']`): a slower pipeline that answers more queries is still the better index.

## 5. Decision

**Selected: Pipeline B - stemming then stopword removal** (pipeline_b)

Pipeline B - stemming then stopword removal wins on the weighted score (0.8272 vs 0.7855, margin +0.0417). Decomposition: financial_expression_preservation 3048/3792 = 0.8038 x0.35 = 0.2813; domain_term_recall 101/101 = 1.0000 x0.25 = 0.2500; variant_collapse_rate 7/12 = 0.5833 x0.25 = 0.1458; query_answerability 15/15 = 1.0000 x0.15 = 0.1500. The two pipelines are equal on 3 of 4 criteria; the margin comes from variant_collapse_rate (7/12 = 0.5833 vs 5/12 = 0.4167, +0.1667 x0.25 = +0.0417).

Runner-up: Pipeline A - domain-aware stopwords then lemmatization (pipeline_a) at 0.7855.

Tie-breakers, in order: smaller index vocabulary, fewer index terms per posting, lexicographically first pipeline name

## 6. Index size consequence

| pipeline | index terms | unigrams | phrases | postings | units | tokens | processing s |
|---|---|---|---|---|---|---|---|
| pipeline_a | 22552 | 20249 | 2303 | 186189 | 6005 | 241977 | 12.616 |
| pipeline_b | 21305 | 18735 | 2570 | 191817 | 6005 | 246965 | 21.636 |

## 7. What survived, and what did not

**pipeline_a**

- preserved: 7.4 per cent -> '7.4 per cent'; FY26 -> 'fy26'; FY26 -> 'fy26'; 7 per cent -> '7 per cent'; FY27 -> 'fy27'; 2 per cent -> '2 per cent'; 2 per cent -> '2 per cent'; 5.15 per cent -> '5.15 per cent'
- not preserved: ₹ 2.5; ₹6.95; $25; $ 2.3; ₹29,000; ₹1,000; ₹2,050; ₹680
- collapsed variant groups: expenditure~expenditures -> 'expenditure'; deficit~deficits -> 'deficit'; payment~payments -> 'payment'; export~exports~exported -> 'export'; import~imports~imported -> 'import'
- domain terms missing from the index: none
- unanswered queries: none

**pipeline_b**

- preserved: 7.4 per cent -> '7.4 per cent'; FY26 -> 'fy26'; FY26 -> 'fy26'; 7 per cent -> '7 per cent'; FY27 -> 'fy27'; 2 per cent -> '2 per cent'; 2 per cent -> '2 per cent'; 5.15 per cent -> '5.15 per cent'
- not preserved: ₹ 2.5; ₹6.95; $25; $ 2.3; ₹29,000; ₹1,000; ₹2,050; ₹680
- collapsed variant groups: expenditure~expenditures -> 'expenditur'; deficit~deficits -> 'deficit'; payment~payments -> 'payment'; savings~saving -> 'save'; investment~invest -> 'invest'; export~exports~exported -> 'export'; import~imports~imported -> 'import'
- domain terms missing from the index: none
- unanswered queries: none

The expressions that do *not* survive are the bare currency amounts (`₹ 2.5`, `$25`): the Phase 2 tokenizer splits the symbol from the number, so the expression-protection step has nothing to re-join. This is a tokenizer-level limitation inherited from Phase 2, reported here rather than patched, because Phase 3 must not change the tokenizer the earlier phase measured.

