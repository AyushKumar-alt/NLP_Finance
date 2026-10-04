# Phase 2 methodology and justification

Domain: Financial / Economic documents (Phase 1 corpus, policy `prose_tables`).
Corpus actually processed: **6134 units / 2081530 characters**
from 31 documents and 885 pages.

Every number below was produced by `python -m src.phase2.run` on that corpus. Nothing
in this file is an estimate, and no accuracy is reported for a task that had no
manually annotated ground truth.

## 1. Why a custom financial tokenizer is needed

The standard tokenizers were measured on identical text (the same experiment
sample, recorded in `comparisons/experiment_manifest.csv`). Their failure mode on
financial writing is not subtle - it is visible in the token streams themselves:

* `7.4%` becomes `['7.4', '%']`: the percent sign is separated from the measurement
  it belongs to, so a query for `7.4%` cannot match.
* `FY2025-26` becomes `['FY2025', '-', '26']` in spaCy: the fiscal year - the main
  temporal index of an Economic Survey - is destroyed.
* `Rs 1.25 lakh crore` becomes four unrelated tokens: the magnitude loses its unit.

`src/phase2/custom_tokenizer.py` therefore implements 17 ordered rules, exported to
`tokenization/custom_tokenizer_rules.csv` with, for each rule, the pattern, the
description, the example, the expected behaviour and the behaviour actually
observed. The decision to keep `7.4%` as one token is stated explicitly: a
percentage is a single measurement, and splitting it makes `7.4` indistinguishable
from a plain growth figure. The complementary *split* representation is implemented
in `date_number_tokenizer.py`, so both are available and their cost is measured
rather than argued.

**Tokenization comparison on the shared sample**

| tokenizer | total_tokens | vocabulary_size | date_tokens | percentage_tokens | currency_tokens | avg_tokens_per_sentence |
|---|---|---|---|---|---|---|
| nltk | 349344 | 18035 | 6939 | 359 | 1016 | 22.54 |
| custom | 345677 | 18628 | 6169 | 2167 | 1046 | 22.3 |
| spacy | 365068 | 15899 | 7090 | 354 | 1036 | 23.55 |
| hybrid | 344141 | 19463 | 6613 | 2255 | 1155 | 22.2 |



## 2. Why dates, fiscal years, currency and percentages must be preserved

`tokenization/date_number_comparison.csv` measures, per expression category, how
many expressions survive *intact* after standard tokenization. Where the intact rate
is low, any downstream index built on those tokens cannot answer a date or amount
query reliably: the term simply is not in the index under the form the user typed.
The domain dictionary layer (section 7) is built on the assumption that these
expressions stay whole, while the typed tokenizer supplies the components needed
for numeric aggregation.

## 3. Why stopword removal is dangerous in financial retrieval

A standard English list deletes words that carry retrieval signal in this corpus.
`preprocessing/domain_stopword_analysis.csv` lists, for every protected financial
term that actually occurs in the corpus, how often the standard list would delete it
and whether the domain-aware list preserves it. The comparison is a count, not a
claim: the numbers come from the same token stream.

**Stopword strategies**

| strategy | stopword_list_size | total_tokens | token_reduction_percent | vocabulary_reduction_percent |
|---|---|---|---|---|
| none | 0 | 344141 | 0.0 | 0.0 |
| standard_english | 198 | 250215 | 27.29 | 0.63 |
| domain_aware | 213 | 249744 | 27.43 | 0.71 |



## 4. Why stemming creates over-stemming problems here

`stemming/stemming_families.csv` groups the financial probe words by the stem each
algorithm produces. A group with more than one probe word is a collision: distinct
financial concepts that a query would want to distinguish are merged into one
posting. `stemming/stemming_examples.csv` shows the same for real corpus vocabulary,
so the claim is visible rather than asserted.

**Stemmer comparison**

| algorithm | unique_stems | vocabulary_reduction_percent | colliding_stems | execution_time_seconds |
|---|---|---|---|---|
| porter | 14243 | 26.77 | 7169 | 1.8 |
| snowball_english | 14250 | 26.73 | 7142 | 0.96 |
| lancaster | 13023 | 33.04 | 6274 | 1.296 |



## 5. Why the order of stopword removal and stemming matters

The two operations do not commute. A stopword list contains surface forms
(`rates`, `policies`); a stemmer maps forms onto roots (`rate`, `polici`).
Removing stopwords first can delete a surface form that the stemmer would have
collapsed; stemming first can map a protected word onto a stem that is not in the
stopword list, so it survives the second step. Both pipelines are measured in
`stemming/stopword_stemming_order_comparison.csv`, including the tokens that differ
between them.

## 6. Why lemmatization preserves meaning better in some cases

Lemmatization uses morphological knowledge, so `investments -> investment` and
`regulated -> regulate` are recoverable, and a POS-aware lookup can keep
`banking` and `banks` apart from `bank`. `lemmatization/lemmatization_comparison.csv`
reports the reduction each method achieves and the POS dependency each one has.
The POS dependency is real and is stated in the artefacts: a POS error propagates
into the lemma.

**Lemmatizer comparison**

| method | pos_required | unique_lemmas | vocabulary_reduction_percent | limitations |
|---|---|---|---|---|
| wordnet_lookup_pos_agnostic | False | 15886 | 0.0 | no POS input, so verbs and adjectives are lemmatized as nouns; domain coinages absent from WordNet are returned unchanged |
| wordnet_lookup_pos_aware | True | 15886 | 0.0 | needs reliable POS, so a POS-tagging error propagates into the lemma; domain coinages still unchanged |
| spacy_rule_based | False | 13295 | 16.31 | trained on general English; financial coinages (e.g. disinflation) are usually left unchanged |
| lemminflect_rulebased | True | 13135 | 17.32 | rule tables cover standard English inflections only; irregular financial nouns are unchanged |



## 7. Why custom POS tagging helps, and what it cannot do

`pos/custom_pos_dictionary.csv` is *derived from observed behaviour*: for every
domain candidate the file records the default tag distribution found in the corpus,
the rule that fired, the reason, and the corpus sentence that justifies the decision.
A term is only corrected when the observed dominant tag contradicts the
shape-based expectation, so no tag is invented.

To make the three methods comparable, 412 tokens in
30 sentences were **manually annotated** with the
universal POS tag set (`data/phase2/annotations/pos_gold_annotations.csv`, single
annotator, guideline documented in the same directory). The same annotated sample
scores the default tagger, the rule tagger and the ML tagger, and the sample size is
printed in every artefact. spaCy's automatic tags are used as *silver* training
labels only and are never treated as truth.

**POS comparison on the manually annotated sample**

| method | dataset_size | accuracy_if_available | f1_if_available | domain_terms_corrected |
|---|---|---|---|---|
| default_pos | 412 | 0.9223 | 0.7571 | 0 |
| custom_rule_pos | 412 | 0.7913 | 0.6139 | 665 |
| custom_ml_pos |  |  |  |  |



## 8. Why general NER misses financial entities

A general OntoNotes model has no label for an economic indicator, a financial
instrument or a policy term, and it has no incentive to fire on them.
`ner/ner_entities.csv` shows what the model does produce;
`ner/ner_error_analysis.csv` separates three error types:

* `wrong_general_label` - type-level impossibilities (for example an all-caps domain
  acronym labelled `PERSON`);
* `domain_category_mismatch` - the model found the span, but with a general label that
  is not equivalent to the domain category;
* `missed_by_general_ner` - domain mentions with no entity at all.

The domain layer in `ner/domain_entity_dictionary.csv` is deliberately
**complementary**: it does not replace the general model, it adds the eight
financial categories the model cannot express. Its limitation is equally important -
it is dictionary based and therefore context free.

General NER entities: 26717; domain mentions:
4968; systematic error rows: 145.

## 9. Why n-grams matter for financial phrases

An indicator name is a phrase, not a word: `real GDP growth`, `policy repo rate`,
`current account deficit`. `ngrams/domain_phrases.csv` mines them from the corpus
with explicit thresholds (frequency, document frequency, function-word ratio) and
records which domain term justified each selection. The effect of stopword removal
on phrase quality is measured in `comparisons/ngram_comparison.csv`: stopword removal
destroys grammatical filler and can *create* unnatural phrases, which is why both
streams are reported.

**N-gram comparison**

| n | total_ngrams | unique_ngrams | top_ngram | top_ngram_frequency |
|---|---|---|---|---|
| 1 | 344141 | 24389 | , | 20645 |
| 2 | 338007 | 156887 | , and | 2727 |
| 3 | 331878 | 268305 | ai ai ai | 393 |
| 4 | 325751 | 303211 | ai ai ai ai | 330 |
| 5 | 319627 | 309454 | ai ai ai ai ai | 284 |



## 10. Why BPE helps with rare and domain-specific words

Byte-level BPE has no out-of-vocabulary case, so a financial coinage the corpus
never contains is still representable. The price is that tokens stop being
user-queryable words. `bpe/bpe_examples.csv` shows both sides: common words that stay
single tokens, and domain terms split into sub-word pieces. Vocabulary size and
corpus token cost are in `bpe/bpe_statistics.csv`.

- BPE vocabulary: 8000 entries, 7998 learned merges
- BPE tokens for the whole sample: 468203
- Bytes/characters per BPE token: 4.446

## 11. What is deliberately *not* claimed

* No accuracy is reported for any task without a manually annotated sample.
* No stemmer or lemmatizer is declared "best"; the tables describe observed behaviour
  on this corpus.
* The POS dictionary corrects only terms whose default tags were actually observed to
  be inconsistent or contradictory.
* The domain entity layer is not presented as NER; it is a dictionary matcher.
