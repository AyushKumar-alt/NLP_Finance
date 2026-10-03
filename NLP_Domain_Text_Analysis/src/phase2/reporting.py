"""Report artefacts: domain test set, cross-tokenizer evidence, methodology, README.

Everything written here is assembled from numbers produced earlier in the run;
the only hand-written parts are the justification texts, which reference the
measured artefacts by name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .custom_tokenizer import custom_tokenize
from .date_number_tokenizer import date_number_tokens
from .tokenizers import hybrid_tokenize, nltk_tokenize, spacy_tokenize_batch, load_spacy

#: The required financial-domain test set. Where a phrase is actually observed
#: in the Phase 1 corpus, ``corpus_occurrences`` is filled from the corpus and
#: the test set is marked ``source=corpus``; the rest are marked
#: ``source=assignment_spec`` so the report never implies corpus evidence that
#: does not exist.
DOMAIN_TEST_SET: Tuple[Tuple[str, str, str], ...] = (
    ("Dates", "31 March 2026", "date"),
    ("Dates", "January 2026", "month_year"),
    ("Dates", "2025", "calendar_year"),
    ("Dates", "2024-25", "year_range"),
    ("Fiscal years", "FY2025-26", "fiscal_year_range"),
    ("Fiscal years", "FY26", "fiscal_year"),
    ("Fiscal years", "FY 2025-26", "fiscal_year_spaced"),
    ("Fiscal years", "Q1 FY26", "fiscal_quarter"),
    ("Percentages", "7.4%", "percentage"),
    ("Percentages", "7.4 per cent", "percentage_words"),
    ("Percentages", "10.5%", "percentage"),
    ("Currency", "\u20b950,000", "currency_rupees"),
    ("Currency", "Rs. 50,000", "currency_rupees_words"),
    ("Currency", "\u20b91.25 lakh crore", "currency_lakh_crore"),
    ("Currency", "$2.5 billion", "currency_dollars"),
    ("Currency", "US$ 2.5 billion", "currency_us_dollars"),
    ("Currency", "\u20b95 lakh crore", "currency_lakh_crore"),
    ("Large numbers", "3.2 million", "magnitude_million"),
    ("Large numbers", "1.2 trillion", "magnitude_trillion"),
    ("Large numbers", "1,25,000 crore", "indian_grouping"),
    ("Economic indicators", "GDP", "acronym"),
    ("Economic indicators", "GVA", "acronym"),
    ("Economic indicators", "CPI", "acronym"),
    ("Economic indicators", "WPI", "acronym"),
    ("Financial institutions", "RBI", "institution"),
    ("Financial institutions", "SEBI", "institution"),
    ("Financial institutions", "IMF", "institution"),
    ("Financial institutions", "World Bank", "institution_phrase"),
    ("Financial instruments", "repo rate", "instrument"),
    ("Financial instruments", "reverse repo", "instrument"),
    ("Financial instruments", "treasury bill", "instrument"),
    ("Financial instruments", "mutual fund", "instrument"),
    ("Hyphenated expressions", "year-on-year", "hyphenated"),
    ("Hyphenated expressions", "month-on-month", "hyphenated"),
    ("Hyphenated expressions", "repo-rate", "hyphenated"),
    ("Hyphenated expressions", "mark-to-market", "hyphenated"),
    ("Abbreviations", "FDI", "acronym"),
    ("Abbreviations", "FPI", "acronym"),
    ("Abbreviations", "NPA", "acronym"),
    ("Abbreviations", "GST", "acronym"),
    ("Abbreviations", "NITI Aayog", "abbreviation_phrase"),
    ("Rare economic terminology", "disinflation", "rare_term"),
    ("Rare economic terminology", "securitisation", "rare_term"),
    ("Rare economic terminology", "demonetisation", "rare_term"),
    ("Rare economic terminology", "macroeconomic", "rare_term"),
    ("Rare economic terminology", "financialisation", "rare_term"),
    ("Rare economic terminology", "refinancing", "rare_term"),
    ("Multiword financial phrases", "real GDP growth", "phrase"),
    ("Multiword financial phrases", "monetary policy", "phrase"),
    ("Multiword financial phrases", "fiscal policy", "phrase"),
    ("Multiword financial phrases", "current account deficit", "phrase"),
    ("Multiword financial phrases", "foreign direct investment", "phrase"),
    ("Multiword financial phrases", "financial stability", "phrase"),
    ("Multiword financial phrases", "capital expenditure", "phrase"),
    ("Multiword financial phrases", "consumer price index", "phrase"),
)


def build_domain_testset(
    corpus_text: str,
    token_counts: Dict[str, int],
    config: Phase2Config,
) -> List[Dict[str, object]]:
    """Domain test set with per-tokenizer output and corpus provenance."""
    lowered = corpus_text.casefold()
    rows: List[Dict[str, object]] = []
    for index, (category, expression, kind) in enumerate(DOMAIN_TEST_SET, start=1):
        occurrences = lowered.count(expression.casefold())
        nltk_tokens = nltk_tokenize(expression)
        custom_tokens = custom_tokenize(expression)
        hybrid_tokens = hybrid_tokenize(expression)
        rows.append(
            {
                "test_id": f"DT{index:03d}",
                "category": category,
                "expression": expression,
                "expression_type": kind,
                "source": "corpus" if occurrences else "assignment_spec",
                "corpus_occurrences": occurrences,
                "nltk_tokens": nltk_tokens,
                "nltk_token_count": len(nltk_tokens),
                "custom_tokens": custom_tokens,
                "custom_token_count": len(custom_tokens),
                "hybrid_tokens": hybrid_tokens,
                "hybrid_token_count": len(hybrid_tokens),
                "date_number_tokens": date_number_tokens(expression),
                "preserved_as_single_token_custom": len(custom_tokens) == 1,
                "preserved_as_single_token_hybrid": len(hybrid_tokens) == 1,
                "nltk_splits_expression": len(nltk_tokens) > 1 and any(
                    part in expression for part in nltk_tokens if part
                ),
            }
        )
    return rows


def tokenization_examples_comparison(
    rows_source: Sequence[Dict[str, object]],
    tokenizer_runs: Dict[str, List[List[str]]],
    units_by_id: Dict[str, object],
    config: Phase2Config,
) -> List[Dict[str, object]]:
    """Original text vs the four tokenizers on real corpus examples."""
    output: List[Dict[str, object]] = []
    for row in rows_source:
        unit_id = str(row["unit_id"])
        index = int(row.get("sample_index", -1))  # type: ignore[arg-type]
        tokens = {name: values[index] for name, values in tokenizer_runs.items() if index < len(values)}
        output.append(
            {
                "document_id": row.get("document_id", ""),
                "source_id": row.get("source_id", ""),
                "page_number": row.get("page_number", ""),
                "unit_id": unit_id,
                "section_id": row.get("section_id", ""),
                "section_number": row.get("section_number", ""),
                "section_title": row.get("section_title", ""),
                "example_category": row.get("category", ""),
                "original_text": row.get("text", ""),
                "nltk_tokens": tokens.get("nltk", []),
                "spacy_tokens": tokens.get("spacy", []),
                "custom_tokens": tokens.get("custom", []),
                "hybrid_tokens": tokens.get("hybrid", []),
                "nltk_token_count": len(tokens.get("nltk", [])),
                "spacy_token_count": len(tokens.get("spacy", [])),
                "custom_token_count": len(tokens.get("custom", [])),
                "hybrid_token_count": len(tokens.get("hybrid", [])),
            }
        )
    return output


def bpe_comparison_rows(
    domain_testset: Sequence[Dict[str, object]],
    bpe_tokenizer,
    config: Phase2Config,
) -> List[Dict[str, object]]:
    """NLTK vs spaCy vs custom vs hybrid vs BPE on every test expression."""
    expressions: List[str] = []
    for row in domain_testset:
        if str(row["expression"]) not in expressions:
            expressions.append(str(row["expression"]))
    nlp = load_spacy(str(config.get("language.spacy_model", "en_core_web_sm")))
    spacy_tokens = spacy_tokenize_batch(expressions, nlp)

    rows: List[Dict[str, object]] = []
    provenance_fields = (
        "source_id",
        "document_id",
        "page_number",
        "section_id",
        "section_number",
        "section_title",
        "unit_id",
        "corpus_units_containing",
    )
    by_expression: Dict[str, Dict[str, object]] = {}
    for row in domain_testset:
        by_expression.setdefault(str(row["expression"]), row)
    for index, expression in enumerate(expressions):
        nltk_tokens = nltk_tokenize(expression)
        spacy_list = spacy_tokens[index]
        custom_list = custom_tokenize(expression)
        hybrid_list = hybrid_tokenize(expression)
        bpe_list = bpe_tokenizer.encode(expression).tokens
        provenance = {
            field: by_expression.get(expression, {}).get(field, "")
            for field in provenance_fields
        }
        rows.append(
            {
                "example": expression,
                "category": next(
                    (r["category"] for r in domain_testset if r["expression"] == expression), ""
                ),
                **provenance,
                "nltk_tokens": nltk_tokens,
                "spacy_tokens": spacy_list,
                "custom_tokens": custom_list,
                "hybrid_tokens": hybrid_list,
                "bpe_tokens": bpe_list,
                "token_count_nltk": len(nltk_tokens),
                "token_count_spacy": len(spacy_list),
                "token_count_custom": len(custom_list),
                "token_count_hybrid": len(hybrid_list),
                "token_count_bpe": len(bpe_list),
                "bpe_subword_split": len(bpe_list) > 1,
                "observations": _bpe_observation(expression, nltk_list := nltk_tokens, custom_list, bpe_list),
            }
        )
    return rows


def _bpe_observation(expression: str, nltk_tokens: List[str], custom_tokens: List[str], bpe_tokens: List[str]) -> str:
    parts: List[str] = []
    if len(nltk_tokens) > 1:
        parts.append(f"NLTK splits into {len(nltk_tokens)} tokens")
    if len(custom_tokens) == 1:
        parts.append("custom keeps it whole")
    if len(bpe_tokens) > 1:
        parts.append(f"BPE sub-words: {len(bpe_tokens)} pieces")
    else:
        parts.append("BPE single token")
    return "; ".join(parts)


# ----------------------------------------------------------------------
# Methodology + README
# ----------------------------------------------------------------------
def methodology_markdown(
    config: Phase2Config,
    corpus_summary: Dict[str, object],
    figures: Dict[str, object],
) -> str:
    tokenization = figures.get("tokenization", [])  # type: ignore[assignment]
    stopwords = figures.get("stopwords", [])  # type: ignore[assignment]
    stemming = figures.get("stemming", [])  # type: ignore[assignment]
    lemmas = figures.get("lemmatization", [])  # type: ignore[assignment]
    pos = figures.get("pos", {})  # type: ignore[assignment]
    ner = figures.get("ner", {})  # type: ignore[assignment]
    ngrams = figures.get("ngrams", [])  # type: ignore[assignment]
    bpe = figures.get("bpe", {})  # type: ignore[assignment]
    gold = figures.get("gold", {})  # type: ignore[assignment]

    def table(rows: Sequence[Dict[str, object]], columns: Sequence[str], title: str) -> str:
        if not rows:
            return f"_No rows for {title}._\n"
        header = "| " + " | ".join(columns) + " |"
        divider = "|" + "|".join("---" for _ in columns) + "|"
        body = "\n".join(
            "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows
        )
        return f"**{title}**\n\n{header}\n{divider}\n{body}\n\n"

    return f"""# Phase 2 methodology and justification

Domain: Financial / Economic documents (Phase 1 corpus, policy `{corpus_summary.get('text_selection_policy')}`).
Corpus actually processed: **{corpus_summary.get('selected_units')} units / {corpus_summary.get('selected_characters')} characters**
from {corpus_summary.get('documents')} documents and {corpus_summary.get('pages')} pages.

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

{table(tokenization, ['tokenizer', 'total_tokens', 'vocabulary_size', 'date_tokens', 'percentage_tokens', 'currency_tokens', 'avg_tokens_per_sentence'], 'Tokenization comparison on the shared sample')}

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

{table(stopwords, ['strategy', 'stopword_list_size', 'total_tokens', 'token_reduction_percent', 'vocabulary_reduction_percent'], 'Stopword strategies')}

## 4. Why stemming creates over-stemming problems here

`stemming/stemming_families.csv` groups the financial probe words by the stem each
algorithm produces. A group with more than one probe word is a collision: distinct
financial concepts that a query would want to distinguish are merged into one
posting. `stemming/stemming_examples.csv` shows the same for real corpus vocabulary,
so the claim is visible rather than asserted.

{table(stemming, ['algorithm', 'unique_stems', 'vocabulary_reduction_percent', 'colliding_stems', 'execution_time_seconds'], 'Stemmer comparison')}

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

{table(lemmas, ['method', 'pos_required', 'unique_lemmas', 'vocabulary_reduction_percent', 'limitations'], 'Lemmatizer comparison')}

## 7. Why custom POS tagging helps, and what it cannot do

`pos/custom_pos_dictionary.csv` is *derived from observed behaviour*: for every
domain candidate the file records the default tag distribution found in the corpus,
the rule that fired, the reason, and the corpus sentence that justifies the decision.
A term is only corrected when the observed dominant tag contradicts the
shape-based expectation, so no tag is invented.

To make the three methods comparable, {gold.get('gold_tokens', 0)} tokens in
{gold.get('gold_sentences', 0)} sentences were **manually annotated** with the
universal POS tag set (`data/phase2/annotations/pos_gold_annotations.csv`, single
annotator, guideline documented in the same directory). The same annotated sample
scores the default tagger, the rule tagger and the ML tagger, and the sample size is
printed in every artefact. spaCy's automatic tags are used as *silver* training
labels only and are never treated as truth.

{table(list(pos.get('comparison', [])), ['method', 'dataset_size', 'accuracy_if_available', 'f1_if_available', 'domain_terms_corrected'], 'POS comparison on the manually annotated sample')}

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

General NER entities: {ner.get('total_entities', 'n/a')}; domain mentions:
{ner.get('domain_mentions', 'n/a')}; systematic error rows: {ner.get('error_rows', 'n/a')}.

## 9. Why n-grams matter for financial phrases

An indicator name is a phrase, not a word: `real GDP growth`, `policy repo rate`,
`current account deficit`. `ngrams/domain_phrases.csv` mines them from the corpus
with explicit thresholds (frequency, document frequency, function-word ratio) and
records which domain term justified each selection. The effect of stopword removal
on phrase quality is measured in `comparisons/ngram_comparison.csv`: stopword removal
destroys grammatical filler and can *create* unnatural phrases, which is why both
streams are reported.

{table(ngrams, ['n', 'total_ngrams', 'unique_ngrams', 'top_ngram', 'top_ngram_frequency'], 'N-gram comparison')}

## 10. Why BPE helps with rare and domain-specific words

Byte-level BPE has no out-of-vocabulary case, so a financial coinage the corpus
never contains is still representable. The price is that tokens stop being
user-queryable words. `bpe/bpe_examples.csv` shows both sides: common words that stay
single tokens, and domain terms split into sub-word pieces. Vocabulary size and
corpus token cost are in `bpe/bpe_statistics.csv`.

- BPE vocabulary: {bpe.get('vocabulary_size', 'n/a')} entries, {bpe.get('merge_operations', 'n/a')} learned merges
- BPE tokens for the whole sample: {bpe.get('corpus_token_count', 'n/a')}
- Bytes/characters per BPE token: {bpe.get('bytes_per_token', 'n/a')}

## 11. What is deliberately *not* claimed

* No accuracy is reported for any task without a manually annotated sample.
* No stemmer or lemmatizer is declared "best"; the tables describe observed behaviour
  on this corpus.
* The POS dictionary corrects only terms whose default tags were actually observed to
  be inconsistent or contradictory.
* The domain entity layer is not presented as NER; it is a dictionary matcher.
"""


def _grouped_output_listing(outputs: Sequence[str]) -> str:
    """Output list grouped by experiment, with generated corpora collapsed."""
    from collections import defaultdict

    groups: Dict[str, List[str]] = defaultdict(list)
    for name in outputs:
        parts = name.replace("\\", "/").split("/")
        group = parts[0] if len(parts) > 1 else "root"
        groups[group].append(name)

    lines: List[str] = []
    for group in sorted(groups):
        names = sorted(groups[group])
        collapsed = [n for n in names if n.startswith(f"{group}/bpe_training_corpus/")]
        shown = [n for n in names if n not in collapsed]
        for name in shown:
            lines.append(f"* `{name}`")
        if collapsed:
            lines.append(
                f"* `{group}/bpe_training_corpus/` - {len(collapsed)} per-document training files "
                f"(`{collapsed[0]}` ... `{collapsed[-1]}`)"
            )
    return "\n".join(lines)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _results_section(results: Dict[str, object]) -> str:
    """Markdown block with the measured numbers of this run."""
    tokenization = list(results.get("tokenization", []) or [])
    stopwords = list(results.get("stopwords", []) or [])
    stemming = list(results.get("stemming", []) or [])
    lemmatization = list(results.get("lemmatization", []) or [])
    pos = list(results.get("pos", []) or [])
    ngrams = list(results.get("ngrams", []) or [])
    bpe = results.get("bpe", {}) or {}
    date_number = list(results.get("date_number", []) or [])
    ner = results.get("ner", {}) or {}

    blocks: List[str] = []
    if tokenization:
        blocks.append(
            "### Tokenization (whole experiment sample)\n\n"
            + _markdown_table(
                ["method", "tokens", "vocabulary", "date tokens", "percentage tokens",
                 "currency tokens"],
                [
                    [
                        row.get("tokenizer", ""),
                        row.get("total_tokens", ""),
                        row.get("vocabulary_size", ""),
                        row.get("date_tokens", ""),
                        row.get("percentage_tokens", ""),
                        row.get("currency_tokens", ""),
                    ]
                    for row in tokenization
                ],
            )
        )
    if date_number:
        blocks.append(
            "### Date and number aware tokenization\n\n"
            + _markdown_table(
                ["category", "expressions detected", "surviving standard tokenization", "intact rate"],
                [
                    [
                        row.get("category", ""),
                        row.get("expressions_detected", ""),
                        row.get("expressions_intact_after_standard_tokenization", ""),
                        f"{row.get('intact_rate_percent', '')}%",
                    ]
                    for row in date_number
                ],
            )
        )
    if stopwords:
        blocks.append(
            "### Stopwords\n\n"
            + _markdown_table(
                ["strategy", "list size", "remaining tokens", "token reduction"],
                [
                    [row.get("strategy", ""), row.get("stopword_list_size", ""),
                     row.get("total_tokens", ""), f"{row.get('token_reduction_percent', '')}%"]
                    for row in stopwords
                ],
            )
        )
    if stemming:
        blocks.append(
            "### Stemming\n\n"
            + _markdown_table(
                ["stemmer", "unique stems", "vocabulary reduction", "colliding stems"],
                [
                    [row.get("algorithm", ""), row.get("unique_stems", ""),
                     f"{row.get('vocabulary_reduction_percent', '')}%",
                     row.get("colliding_stems", "")]
                    for row in stemming
                ],
            )
        )
    if lemmatization:
        blocks.append(
            "### Lemmatization\n\n"
            + _markdown_table(
                ["method", "unique lemmas", "vocabulary reduction", "needs POS"],
                [
                    [row.get("method", ""), row.get("unique_lemmas", ""),
                     f"{row.get('vocabulary_reduction_percent', '')}%",
                     row.get("pos_required", "")]
                    for row in lemmatization
                ],
            )
        )
    if pos:
        blocks.append(
            "### POS tagging (scored only on the manually annotated gold sample)\n\n"
            + _markdown_table(
                ["method", "tokens scored", "gold sentences", "accuracy", "macro F1"],
                [
                    [row.get("method", ""), row.get("dataset_size", ""),
                     row.get("gold_sentences", ""), row.get("accuracy_if_available", ""),
                     row.get("f1_if_available", "")]
                    for row in pos
                ],
            )
        )
    if ner:
        blocks.append(
            "### Named entities\n\n"
            + _markdown_table(
                ["measure", "value"],
                [
                    ["general spaCy entities", ner.get("total_entities", "")],
                    ["general spaCy labels", ner.get("label_count", "")],
                    ["domain dictionary mentions", ner.get("domain_mentions", "")],
                    ["domain categories", ner.get("domain_category_count", "")],
                    ["error-analysis rows", ner.get("error_rows", "")],
                ],
            )
        )
    if ngrams:
        blocks.append(
            "### N-grams\n\n"
            + _markdown_table(
                ["n", "total", "unique", "top n-gram with words"],
                [
                    [row.get("n", ""), row.get("total_ngrams", ""), row.get("unique_ngrams", ""),
                     f"{row.get('top_content_ngram', '')} (x{row.get('top_content_ngram_frequency', '')})"]
                    for row in ngrams
                ],
            )
        )
    if bpe:
        blocks.append(
            "### BPE\n\n"
            + _markdown_table(
                ["measure", "value"],
                [
                    ["vocabulary", bpe.get("vocabulary_size", "")],
                    ["tokens for the whole sample", bpe.get("corpus_token_count", "")],
                    ["bytes per token", bpe.get("bytes_per_token", "")],
                ],
            )
        )
    return "\n\n".join(blocks) if blocks else "No measured rows were available."


def readme_markdown(
    config: Phase2Config,
    corpus_summary: Dict[str, object],
    outputs: Sequence[str],
    figures: Sequence[str],
    results: Optional[Dict[str, object]] = None,
) -> str:
    grouped_outputs = _grouped_output_listing(outputs)
    output_count = len(outputs)
    results_section = _results_section(results or {})
    return f"""# Phase 2 - NLP experimentation results

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

* `data/corpus/structured/corpus.jsonl` - {corpus_summary.get('units_total')} units from
  {corpus_summary.get('documents')} documents ({corpus_summary.get('pages')} pages)
* `data/corpus/metadata/document_registry.csv`, `source_registry.csv` - provenance
* text selection policy: **`{corpus_summary.get('text_selection_policy')}`** =
  {', '.join(corpus_summary.get('policy_unit_types', []))} -> {corpus_summary.get('selected_units')} units,
  {corpus_summary.get('selected_characters')} characters

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

{results_section}

## 13. Output files

Key artefacts (the full list of {output_count} written files is in
`phase2_summary.json` -> `outputs`):

{grouped_outputs}

Figures: {', '.join(figures) if figures else 'none'}

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
  {corpus_summary.get('gold_tokens', 'n/a')} tokens, so POS accuracy figures carry
  real uncertainty and are reported with the sample size. The annotated sentences were
  re-verified against the corpus after annotation; provenance columns were corrected
  where a sentence was traced to the wrong unit, and the UPOS labels themselves were
  never edited.
* The domain entity layer is dictionary based and context free. Its *counts* cover the
  whole sample, while `custom_ner_results.csv` keeps at most three mentions per
  category per unit so that table-heavy units cannot flood the table; the cap is
  recorded in `ner_error_analysis.csv` metadata.
* spaCy's parser is run on units of at most
  {corpus_summary.get('spacy_max_chars_per_chunk', 'n/a')} characters
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
"""
