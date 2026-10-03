"""Cross-experiment comparison tables and the master Phase 2 summary.

Nothing in this module computes a new metric: it reorganises the numbers that
the experiment modules already produced, so a comparison row can always be
traced back to the experiment that measured it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from .config import Phase2Config, log_event
from .statistics import TokenizedCorpus, VocabularyMetrics, percent_reduction

TOKENIZATION_STRENGTHS = {
    "nltk": (
        "Treebank rules handle contractions and punctuation faithfully; mature and fast",
        "no domain awareness: splits 7.4%, FY2025-26 components and currency symbols",
        "numeric/date expressions are fragmented; orphaned '%' and currency symbols",
    ),
    "spacy": (
        "trained orthographic rules; fastest; keeps 6.50 and repo-rate whole",
        "still splits percentages, fiscal-year ranges and hyphenated compounds",
        "'FY2025-26' becomes FY2025 / - / 26, so fiscal-year indexing fails",
    ),
    "custom": (
        "17 ordered, inspectable rules; preserves fiscal years, percentages, currency, scale words",
        "no linguistic model: cannot disambiguate, and rules are tuned to this domain",
        "exactly the target behaviour: 7.4%, FY2025-26, Rs 1.25 lakh crore stay single tokens",
    ),
    "hybrid": (
        "keeps NLTK's linguistic behaviour while protecting domain expressions",
        "three-pass cost (protect/tokenize/repair) and private-use sentinels in between",
        "same domain outcomes as the custom tokenizer, with NLTK's word handling",
    ),
    "bpe": (
        "sub-word units: no OOV, morphologically aware, data-driven from this corpus",
        "produces non-word pieces that no keyword query can match directly",
        "financial coinages split into pieces; dates/numbers handled as byte sequences",
    ),
}

STEMMING_NOTES = {
    "porter": "moderate reduction; conservative with financial suffixes",
    "snowball_english": "slightly stronger reduction than Porter on -ation/-ise forms",
    "lancaster": "most aggressive; largest reduction and the most collisions",
    "snowball_porter2": "Porter via the Snowball implementation",
}

LEMMATIZATION_NOTES = {
    "wordnet_lookup_pos_agnostic": "noun-first WordNet lookup; no POS needed",
    "wordnet_lookup_pos_aware": "WordNet surface index consulted for the observed POS",
    "spacy_rule_based": "context-sensitive rule lemmatizer from the shared spaCy pass",
    "lemminflect_rulebased": "pure inflectional rule tables, POS driven",
}

DOMAIN_HANDLING = {
    "nltk": "poor (splits fiscal years, percentages, currency)",
    "spacy": "partial (keeps numbers, splits hyphens and fiscal ranges)",
    "custom": "strong (purpose-built rules, inspectable)",
    "hybrid": "strong (domain layer over a standard tokenizer)",
    "stopword_removal": "neutral, but can delete modality/domain nouns (see stopword analysis)",
    "porter": "reduces inflections; can over-stem financial terms",
    "snowball_english": "reduces inflections; slightly more aggressive than Porter",
    "lancaster": "very aggressive; highest collision rate",
    "wordnet_lookup_pos_agnostic": "leaves domain coinages unchanged",
    "wordnet_lookup_pos_aware": "POS-dependent, so also leaves coinages unchanged",
    "spacy_rule_based": "general English rules only",
    "lemminflect_rulebased": "standard inflection only",
    "default_pos": "no domain awareness; acronyms tagged as common nouns",
    "custom_rule_pos": "data-driven corrections for observed domain mis-tags",
    "custom_ml_pos": "learned from this corpus + manual gold sample",
    "general_ner": "OntoNotes classes; misses indicators, instruments and policy terms",
    "custom_domain_ner": "financial/economic categories, complementary to general NER",
    "unigram": "term-level matching only; no phrase semantics",
    "bigram": "captures short collocations such as 'policy rate'",
    "trigram": "captures indicator phrases such as 'real GDP growth'",
    "4-gram": "captures longer policy expressions; high document-frequency threshold needed",
    "5-gram": "few high-confidence phrases; very sparse",
    "bpe": "byte-level sub-words; robust to unseen financial coinages",
}

DOWNSTREAM_USE = {
    "nltk": "baseline experiments, teaching comparison, generic indexing",
    "spacy": "fast indexing when domain expressions are handled as extra fields",
    "custom": "exact indexing of fiscal years, percentages and currency amounts",
    "hybrid": "Phase 3 inverted index over domain-aware terms",
    "stopword_removal": "query-side filtering only, never on protected financial terms",
    "porter": "recall-oriented index with morphological conflation",
    "snowball_english": "precision/recall balance for English financial prose",
    "lancaster": "aggressive conflation, useful for fuzzy matching demos",
    "wordnet_lookup_pos_agnostic": "light normalisation without POS cost",
    "wordnet_lookup_pos_aware": "normalisation when POS is already available",
    "spacy_rule_based": "normalisation inside an existing spaCy pipeline",
    "lemminflect_rulebased": "offline normalisation with explicit POS",
    "default_pos": "linguistic features for ranking and filtering",
    "custom_rule_pos": "domain-aware syntax features",
    "custom_ml_pos": "comparison baseline for POS research in the report",
    "general_ner": "PERSON/ORG/GPE/DATE/MONEY reporting and query filtering",
    "custom_domain_ner": "domain entity filters (institution, indicator, policy term)",
    "unigram": "term postings and term-frequency ranking",
    "bigram": "phrase postings, phrase queries in Phase 3",
    "trigram": "multi-word indicator queries",
    "4-gram": "long phrase evidence, precision-oriented",
    "5-gram": "document-level topic evidence, sparse",
    "bpe": "fallback representation for rare/seen-again financial terms",
}


def tokenization_method_comparison(
    tokenization_rows: Sequence[Dict[str, object]],
    token_counts_by_method: Dict[str, int],
    vocab_by_method: Dict[str, int],
    example_cases: Dict[str, str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in tokenization_rows:
        method = str(row["tokenizer"])
        strengths, limitations, domain_behaviour = TOKENIZATION_STRENGTHS.get(
            method, ("", "", "")
        )
        rows.append(
            {
                "method": method,
                "description": row["description"],
                "token_count": row["total_tokens"],
                "unique_tokens": row["unique_tokens"],
                "vocabulary_size": row["vocabulary_size"],
                "avg_tokens_per_sentence": row["avg_tokens_per_sentence"],
                "numeric_tokens": row["numeric_tokens"],
                "date_tokens": row["date_tokens"],
                "percentage_tokens": row["percentage_tokens"],
                "currency_tokens": row["currency_tokens"],
                "strengths": strengths,
                "limitations": limitations,
                "financial_domain_behavior": domain_behaviour,
                "example_cases": example_cases.get(method, ""),
            }
        )
    return rows


def method_comparison_rows(
    tokenization_rows: Sequence[Dict[str, object]],
    stopword_rows: Sequence[Dict[str, object]],
    stemming_rows: Sequence[Dict[str, object]],
    lemmatization_rows: Sequence[Dict[str, object]],
    pos_comparison: Sequence[Dict[str, object]],
    ner_rows: Sequence[Dict[str, object]],
    ngram_summary: Sequence[Dict[str, object]],
    bpe_rows: Sequence[Dict[str, object]],
    examples: Dict[str, str],
) -> List[Dict[str, object]]:
    """The 21-method master comparison required by the assignment."""
    rows: List[Dict[str, object]] = []
    by_tokenizer = {str(r["tokenizer"]): r for r in tokenization_rows}
    by_stemmer = {str(r["algorithm"]): r for r in stemming_rows}
    by_lemma = {str(r["method"]): r for r in lemmatization_rows}
    by_strategy = {str(r["strategy"]): r for r in stopword_rows}
    by_ngram = {int(r["n"]): r for r in ngram_summary}
    by_pos = {str(r["method"]): r for r in pos_comparison}
    bpe_row = bpe_rows[0] if bpe_rows else {}

    def add(method: str, purpose: str, tokens: object, vocab: object, advantages: str, limitations: str) -> None:
        rows.append(
            {
                "method": method,
                "purpose": purpose,
                "token_count": tokens,
                "vocabulary_size": vocab,
                "domain_handling": DOMAIN_HANDLING.get(method, ""),
                "advantages_observed": advantages,
                "limitations_observed": limitations,
                "example": examples.get(method, ""),
                "suitable_downstream_use": DOWNSTREAM_USE.get(method, ""),
            }
        )

    for method in ("nltk", "spacy", "custom", "hybrid"):
        row = by_tokenizer.get(method, {})
        add(
            method,
            f"{method} tokenization",
            row.get("total_tokens", ""),
            row.get("vocabulary_size", ""),
            TOKENIZATION_STRENGTHS.get(method, ("", ""))[0],
            TOKENIZATION_STRENGTHS.get(method, ("", ""))[1],
        )
    if "bpe" in bpe_row:
        add(
            "bpe",
            "byte-level sub-word tokenization",
            bpe_row.get("total_tokens", ""),
            bpe_row.get("vocabulary_size", ""),
            "no out-of-vocabulary; learns merges from this corpus",
            "pieces are not user-queryable words",
        )
    stop_row = by_strategy.get("standard_english", {})
    add(
        "stopword_removal",
        "remove standard English stopwords",
        stop_row.get("total_tokens", ""),
        stop_row.get("vocabulary_size", ""),
        f"{stop_row.get('token_reduction_percent', '')}% of tokens removed",
        "deletes protected financial terms (see domain_stopword_analysis.csv)",
    )
    for method in ("porter", "snowball_english", "lancaster"):
        row = by_stemmer.get(method, {})
        add(
            method,
            f"{method} stemming",
            row.get("total_tokens_after", ""),
            row.get("unique_stems", ""),
            f"{row.get('vocabulary_reduction_percent', '')}% vocabulary reduction",
            STEMMING_NOTES.get(method, ""),
        )
    for method, note in LEMMATIZATION_NOTES.items():
        row = by_lemma.get(method, {})
        add(
            method,
            f"{method} lemmatization",
            row.get("total_tokens", ""),
            row.get("unique_lemmas", ""),
            f"{row.get('vocabulary_reduction_percent', '')}% vocabulary reduction",
            note,
        )
    for method in ("default_pos", "custom_rule_pos", "custom_ml_pos"):
        row = by_pos.get(method, {})
        add(
            method,
            f"{method} part-of-speech tagging",
            row.get("evaluation_sample_size_tokens", ""),
            "",
            (
                f"accuracy {row.get('accuracy')} on {row.get('evaluation_sample_size_tokens', 'n/a')} "
                f"manually annotated tokens"
                if row.get("accuracy") not in (None, "")
                else str(row.get("observed_changes", ""))
            ),
            str(row.get("limitations", "")),
        )
    add(
        "general_ner",
        "general named entity recognition",
        ner_rows[0].get("total_entities", "") if ner_rows else "",
        ner_rows[0].get("distinct_entity_types", "") if ner_rows else "",
        "PERSON/ORG/GPE/DATE/MONEY/PRODUCT/EVENT entities with provenance",
        "misses economic indicators, instruments and policy terms",
    )
    add(
        "custom_domain_ner",
        "financial domain entity dictionary",
        ner_rows[1].get("total_mentions", "") if len(ner_rows) > 1 else "",
        ner_rows[1].get("distinct_terms", "") if len(ner_rows) > 1 else "",
        "adds 8 financial categories the general model has no label for",
        "dictionary based: no context, so a term in another sense still matches",
    )
    for n in (1, 2, 3, 4, 5):
        row = by_ngram.get(n, {})
        add(
            {1: "unigram", 2: "bigram", 3: "trigram", 4: "4-gram", 5: "5-gram"}[n],
            f"{n}-gram analysis",
            row.get("total_ngrams", ""),
            row.get("unique_ngrams", ""),
            f"top: {row.get('top_ngram', '')} ({row.get('top_ngram_frequency', '')})",
            DOMAIN_HANDLING.get({1: "unigram", 2: "bigram", 3: "trigram", 4: "4-gram", 5: "5-gram"}[n], ""),
        )
    return rows


def preprocessing_comparison_rows(
    stages: Sequence[Tuple[str, str, TokenizedCorpus]],
) -> List[Dict[str, object]]:
    """Original -> tokenized -> stopword-removed -> stemmed -> lemmatized."""
    if not stages:
        return []
    baseline_tokens = stages[0][2].total_tokens
    baseline_vocab = stages[0][2].vocabulary_size
    baseline_units = stages[0][2].avg_tokens_per_unit()
    rows: List[Dict[str, object]] = []
    for stage, method, tokenized in stages:
        metrics = VocabularyMetrics(
            stage=stage,
            method=method,
            total_tokens=tokenized.total_tokens,
            unique_tokens=tokenized.unique_tokens,
            vocabulary_size=tokenized.vocabulary_size,
            documents=tokenized.document_count,
            units=len(tokenized.units),
            avg_tokens_per_document=tokenized.avg_tokens_per_document(),
            extra={
                "avg_tokens_per_unit": tokenized.avg_tokens_per_unit(),
                "type_token_ratio": tokenized.type_token_ratio(),
            },
        )
        rows.append(
            metrics.as_row(
                baseline_total=baseline_tokens,
                baseline_vocab=baseline_vocab,
            )
        )
        rows[-1]["avg_tokens_per_unit_baseline"] = baseline_units
        rows[-1]["avg_tokens_per_unit_change_percent"] = percent_reduction(
            baseline_units, tokenized.avg_tokens_per_unit()
        )
    return rows


def before_after_rows(sample_units, views: Dict[str, TokenizedCorpus], limit: int = 12) -> List[Dict[str, object]]:
    """Human readable side-by-side of the pipeline stages."""
    candidates = [u for u in sample_units if 120 <= len(u.text) <= 400]
    if not candidates:
        candidates = list(sample_units)[:limit]
    candidates = candidates[:: max(1, len(candidates) // max(1, limit))][:limit]
    rows: List[Dict[str, object]] = []
    for unit in candidates:
        row: Dict[str, object] = {
            "document_id": unit.document_id,
            "source_id": unit.source_id,
            "page_number": unit.page_number,
            "unit_id": unit.unit_id,
            "section_id": unit.section_id,
            "section_number": unit.section_number,
            "section_title": unit.section_title,
            "original_text": unit.text.replace("\n", " | ")[:600],
        }
        for name, view in views.items():
            try:
                index = view.units.index(unit)
            except ValueError:
                row[name] = ""
                continue
            tokens = view.tokens[index]
            row[name] = " ".join(tokens)[:600]
            row[f"{name}_token_count"] = len(tokens)
        rows.append(row)
    return rows


def pos_comparison_rows(
    default_summary: Dict[str, object],
    rule_summary: Dict[str, object],
    ml_summary: Dict[str, object],
    rule_changes: Dict[str, object],
) -> List[Dict[str, object]]:
    return [
        {
            "method": "default_pos",
            "dataset_size": default_summary.get("evaluation_sample_size_tokens", ""),
            "gold_sentences": default_summary.get("gold_sentences", ""),
            "accuracy_if_available": default_summary.get("accuracy", ""),
            "precision_if_available": default_summary.get("macro_precision", ""),
            "recall_if_available": default_summary.get("macro_recall", ""),
            "f1_if_available": default_summary.get("macro_f1", ""),
            "domain_terms_corrected": 0,
            "observed_changes": "baseline distribution over the whole corpus",
            "limitations": "no domain awareness; acronyms and -isation nouns mis-tagged",
        },
        {
            "method": "custom_rule_pos",
            "dataset_size": rule_summary.get("evaluation_sample_size_tokens", ""),
            "gold_sentences": rule_summary.get("gold_sentences", ""),
            "accuracy_if_available": rule_summary.get("accuracy", ""),
            "precision_if_available": rule_summary.get("macro_precision", ""),
            "recall_if_available": rule_summary.get("macro_recall", ""),
            "f1_if_available": rule_summary.get("macro_f1", ""),
            "domain_terms_corrected": rule_changes.get("changed_tokens", ""),
            "observed_changes": (
                f"{rule_changes.get('changed_tokens', 0)} of {rule_changes.get('total_tokens', 0)} corpus "
                f"tokens re-tagged ({rule_changes.get('changed_percent', 0)}%)"
            ),
            "limitations": "dictionary based; no context beyond the token itself",
        },
        {
            "method": "custom_ml_pos",
            "dataset_size": ml_summary.get("evaluation_sample_size_tokens", ""),
            "gold_sentences": ml_summary.get("gold_sentences", ""),
            "accuracy_if_available": ml_summary.get("accuracy", ""),
            "precision_if_available": ml_summary.get("macro_precision", ""),
            "recall_if_available": ml_summary.get("macro_recall", ""),
            "f1_if_available": ml_summary.get("macro_f1", ""),
            "domain_terms_corrected": "",
            "observed_changes": (
                f"trained on {ml_summary.get('training_tokens', '')} tokens "
                f"({ml_summary.get('training_silver_tokens', '')} silver, "
                f"{ml_summary.get('training_gold_tokens', '')} gold)"
            ),
            "limitations": (
                "single annotator gold set of "
                f"{ml_summary.get('evaluation_sample_size_tokens', '')} tokens; "
                "silver training labels come from spaCy"
            ),
        },
    ]


def summary_rows(
    tokenization_rows: Sequence[Dict[str, object]],
    stopword_rows: Sequence[Dict[str, object]],
    stemming_rows: Sequence[Dict[str, object]],
    lemmatization_rows: Sequence[Dict[str, object]],
    preprocessing_rows: Sequence[Dict[str, object]],
    ngram_summary: Sequence[Dict[str, object]],
    bpe_stats: Sequence[Dict[str, object]],
    pos_comparison: Sequence[Dict[str, object]],
    ner_totals: Dict[str, object],
    date_number_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """``phase2_summary.csv`` - one row per experiment."""
    rows: List[Dict[str, object]] = []

    def add(experiment: str, method: str, tokens: object, unique: object, vocab: object,
            metric: str, result: str, notes: str) -> None:
        rows.append(
            {
                "experiment": experiment,
                "method": method,
                "total_tokens": tokens,
                "unique_tokens": unique,
                "vocabulary_size": vocab,
                "main_metric": metric,
                "main_result": result,
                "notes": notes,
            }
        )

    for row in tokenization_rows:
        add(
            "tokenization",
            row["tokenizer"],
            row["total_tokens"],
            row["unique_tokens"],
            row["vocabulary_size"],
            "avg_tokens_per_sentence",
            row["avg_tokens_per_sentence"],
            f"{row['numeric_tokens']} numeric / {row['date_tokens']} date / {row['percentage_tokens']} percentage / "
            f"{row['currency_tokens']} currency tokens; {row['execution_time_seconds']}s",
        )
    for row in stopword_rows:
        add(
            "stopword_removal",
            row["strategy"],
            row["total_tokens"],
            row["unique_tokens"],
            row["vocabulary_size"],
            "token_reduction_percent",
            row["token_reduction_percent"],
            f"list size {row['stopword_list_size']} ({row['stopword_list_source']})",
        )
    for row in stemming_rows:
        add(
            "stemming",
            row["algorithm"],
            row["total_tokens_after"],
            row["unique_tokens_after"],
            row["unique_stems"],
            "vocabulary_reduction_percent",
            row["vocabulary_reduction_percent"],
            f"{row['colliding_stems']} stems shared by several surface forms",
        )
    for row in lemmatization_rows:
        add(
            "lemmatization",
            row["method"],
            row["total_tokens"],
            row["unique_tokens"],
            row["unique_lemmas"],
            "vocabulary_reduction_percent",
            row["vocabulary_reduction_percent"],
            f"POS required: {row['pos_required']} ({row['pos_source']})",
        )
    for row in preprocessing_rows:
        add(
            "preprocessing_pipeline",
            f"{row['stage']} ({row['method']})",
            row["total_tokens"],
            row["unique_tokens"],
            row["vocabulary_size"],
            "vocabulary_change_vs_baseline_percent",
            row.get("vocabulary_change_vs_baseline_percent", ""),
            f"avg tokens/unit {row['avg_tokens_per_unit']}",
        )
    for row in ngram_summary:
        add(
            "ngrams",
            f"{row['n']}-gram",
            row["total_ngrams"],
            row["unique_ngrams"],
            row["unique_ngrams"],
            "top_ngram_frequency",
            row["top_ngram_frequency"],
            f"top n-gram: {row['top_ngram']}",
        )
    for row in bpe_stats:
        if row["metric"] in ("vocabulary_size", "corpus_token_count", "bytes_per_token", "merge_operations"):
            add(
                "bpe",
                "byte_level_bpe",
                row["value"] if row["metric"] == "corpus_token_count" else "",
                "",
                row["value"] if row["metric"] == "vocabulary_size" else "",
                row["metric"],
                row["value"],
                row["detail"],
            )
    for row in pos_comparison:
        add(
            "pos_tagging",
            row["method"],
            row.get("dataset_size", ""),
            "",
            "",
            "accuracy_on_manually_annotated_sample",
            row.get("accuracy_if_available", "not available without annotation"),
            str(row.get("limitations", ""))[:200],
        )
    add(
        "ner",
        "general_ner",
        ner_totals.get("total_entities", ""),
        ner_totals.get("distinct_entity_types", ""),
        "",
        "entities_found",
        ner_totals.get("total_entities", ""),
        f"top labels: {ner_totals.get('top_labels', '')}",
    )
    add(
        "ner",
        "custom_domain_ner",
        ner_totals.get("domain_mentions", ""),
        ner_totals.get("domain_terms", ""),
        "",
        "domain_mentions",
        ner_totals.get("domain_mentions", ""),
        "complementary dictionary layer, not a replacement",
    )
    for row in date_number_rows:
        if str(row["category"]) == "TOTAL":
            add(
                "date_number_tokenization",
                "standard_vs_typed",
                "",
                "",
                "",
                "expressions_intact_after_standard_tokenization_percent",
                row["intact_rate_percent"],
                f"{row['expressions_detected']} typed expressions detected; "
                f"{row['typed_component_tokens']} typed component tokens produced",
            )
    return rows
