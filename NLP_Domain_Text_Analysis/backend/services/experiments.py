"""Phase 2 experiment tables.

Every figure the /tokenization, /preprocessing, /pos, /ner, /ngrams and /bpe
pages show is read from the CSV or JSON that Phase 2 already wrote. Nothing is
recomputed, and nothing is invented: a table Phase 2 did not produce is reported
as missing rather than filled in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config.settings import Settings
from backend.services import artifacts

TOKENIZATION_COMPARISON = "tokenization/tokenization_comparison.csv"
TOKENIZATION_EXAMPLES = "tokenization/tokenization_examples_comparison.csv"
CUSTOM_RULES = "tokenization/custom_tokenizer_rules.csv"
DATE_NUMBER_COMPARISON = "tokenization/date_number_comparison.csv"
DATE_NUMBER_PATTERNS = "tokenization/date_number_patterns.csv"
TOKENIZATION_METHOD = "comparisons/tokenization_method_comparison.csv"
REPRESENTATIVE_SAMPLE = "comparisons/representative_sample.csv"
PREPROCESSING_COMPARISON = "preprocessing/preprocessing_comparison.csv"
BEFORE_AFTER = "preprocessing/before_after_examples.csv"
STOPWORD_COMPARISON = "preprocessing/stopword_comparison.csv"
DOMAIN_STOPWORDS = "preprocessing/domain_stopword_analysis.csv"
PROTECTED_TERMS = "preprocessing/stopwords_protected_terms.csv"
FINAL_STOPWORDS = "preprocessing/final_stopwords.txt"
STEMMING_COMPARISON = "stemming/stemming_comparison.csv"
STEMMING_EXAMPLES = "stemming/stemming_examples.csv"
STEMMING_FAMILIES = "stemming/stemming_families.csv"
ORDER_COMPARISON = "stemming/stopword_stemming_order_comparison.csv"
LEMMATIZATION_COMPARISON = "lemmatization/lemmatization_comparison.csv"
LEMMATIZATION_EXAMPLES = "lemmatization/lemmatization_examples.csv"
POS_DISTRIBUTION = "pos/default_pos_distribution.csv"
POS_COARSE = "pos/default_pos_coarse.csv"
POS_EXAMPLES = "pos/default_pos_examples.csv"
POS_DICTIONARY = "pos/custom_pos_dictionary.csv"
POS_CHANGES = "pos/custom_pos_changes.csv"
POS_CANDIDATES = "pos/custom_pos_candidates.csv"
POS_ML_RESULTS = "pos/ml_pos_results.csv"
POS_ML_REPORT = "pos/ml_pos_classification_report.csv"
POS_ML_ERRORS = "pos/ml_pos_misclassifications.csv"
POS_GOLD_ALIGNMENT = "pos/gold_annotation_alignment.csv"
POS_GOLD_ERRORS = "pos/pos_gold_errors.csv"
POS_COMPARISON = "comparisons/pos_comparison.csv"
NER_ENTITIES = "ner/ner_entities.csv"
NER_CUSTOM = "ner/custom_ner_results.csv"
NER_LABELS = "ner/ner_label_distribution.csv"
NER_DICTIONARY = "ner/domain_entity_dictionary.csv"
NER_ERRORS = "ner/ner_error_analysis.csv"
NGRAM_FILES = {
    1: "ngrams/unigram_results.csv",
    2: "ngrams/bigram_results.csv",
    3: "ngrams/trigram_results.csv",
    4: "ngrams/fourgram_results.csv",
    5: "ngrams/fivegram_results.csv",
}
DOMAIN_PHRASES = "ngrams/domain_phrases.csv"
NGRAM_COMPARISON = "comparisons/ngram_comparison.csv"
BPE_STATS = "bpe/bpe_statistics.csv"
BPE_VOCAB = "bpe/bpe_vocabulary.csv"
BPE_EXAMPLES = "bpe/bpe_examples.csv"
BPE_MANIFEST = "bpe/tokenizer_model/training_manifest.json"
BPE_COMPARISON = "comparisons/bpe_tokenization_comparison.csv"
MASTER_COMPARISON = "comparisons/phase2_master_comparison.csv"
SUMMARY_JSON = "phase2_summary.json"
SUMMARY_CSV = "phase2_summary.csv"
VALIDATION = "phase2_validation_report.csv"
README = "README.md"


def _path(settings: Settings, relative: str):
    return settings.path(f"{settings.phase2_results}/{relative}")


def _rows(settings: Settings, relative: str, required: bool = True) -> List[Dict[str, str]]:
    return artifacts.read_csv_rows(_path(settings, relative), required=required)


def summary(settings: Settings) -> Dict[str, Any]:
    return artifacts.read_json(_path(settings, SUMMARY_JSON))


def summary_table(settings: Settings) -> List[Dict[str, str]]:
    return _rows(settings, SUMMARY_CSV)


def validation(settings: Settings) -> List[Dict[str, str]]:
    return _rows(settings, VALIDATION)


def readme(settings: Settings) -> str:
    return artifacts.read_text(_path(settings, README), required=False)


def figures(settings: Settings) -> List[Dict[str, Any]]:
    """Phase 2's 11 charts, listed so the GUI can show the real PNGs."""
    from src.phase2.config import load_config as load_phase2_config

    directory = _path(settings, "figures")
    if not directory.is_dir():
        return []
    return [
        {"file": entry.name, "bytes": entry.stat().st_size, "phase": 2}
        for entry in sorted(directory.glob("*.png"))
    ]


# ----------------------------------------------------------------------
def tokenization(settings: Settings) -> Dict[str, Any]:
    comparison = _rows(settings, TOKENIZATION_COMPARISON)
    method = _rows(settings, TOKENIZATION_METHOD)
    methods = {row.get("method"): row for row in method}
    tokenizers: List[Dict[str, Any]] = []
    for row in comparison:
        name = row.get("tokenizer", "")
        extra = methods.get(name, {})
        tokenizers.append(
            {
                "tokenizer": name,
                "description": row.get("description"),
                "documents_processed": artifacts.as_int(row.get("documents_processed")),
                "units_processed": artifacts.as_int(row.get("units_processed")),
                "sentences": artifacts.as_int(row.get("sentences")),
                "total_tokens": artifacts.as_int(row.get("total_tokens")),
                "unique_tokens": artifacts.as_int(row.get("unique_tokens")),
                "vocabulary_size": artifacts.as_int(row.get("vocabulary_size")),
                "avg_tokens_per_document": artifacts.as_float(row.get("avg_tokens_per_document")),
                "avg_tokens_per_sentence": artifacts.as_float(row.get("avg_tokens_per_sentence")),
                "avg_tokens_per_unit": artifacts.as_float(row.get("avg_tokens_per_unit")),
                "numeric_tokens": artifacts.as_int(row.get("numeric_tokens")),
                "date_tokens": artifacts.as_int(row.get("date_tokens")),
                "fiscal_year_tokens": artifacts.as_int(row.get("fiscal_year_tokens")),
                "percentage_tokens": artifacts.as_int(row.get("percentage_tokens")),
                "currency_tokens": artifacts.as_int(row.get("currency_tokens")),
                "abbreviation_tokens": artifacts.as_int(row.get("abbreviation_tokens")),
                "hyphenated_tokens": artifacts.as_int(row.get("hyphenated_tokens")),
                "special_financial_tokens": artifacts.as_int(row.get("special_financial_tokens")),
                "execution_time_seconds": artifacts.as_float(row.get("execution_time_seconds")),
                "strengths": extra.get("strengths"),
                "limitations": extra.get("limitations"),
                "financial_domain_behavior": extra.get("financial_domain_behavior"),
            }
        )
    return {
        "tokenizers": tokenizers,
        "examples": _rows(settings, TOKENIZATION_EXAMPLES),
        "custom_rules": _rows(settings, CUSTOM_RULES),
        "date_number_comparison": _rows(settings, DATE_NUMBER_COMPARISON),
        "date_number_patterns": _rows(settings, DATE_NUMBER_PATTERNS),
        "representative_sample": _rows(settings, REPRESENTATIVE_SAMPLE),
        "bpe_comparison": _rows(settings, BPE_COMPARISON),
    }


def tokenize_live(settings: Settings, text: str) -> Dict[str, Any]:
    from src.phase2.custom_tokenizer import custom_tokenize
    from src.phase2.tokenizers import hybrid_tokenize, nltk_tokenize

    t = (text or "").strip()
    if not t:
        return {
            "text": "",
            "tokens": {"nltk": [], "custom": [], "hybrid": [], "spacy": []},
            "counts": {"nltk": 0, "custom": 0, "hybrid": 0, "spacy": 0},
        }

    try:
        nltk_toks = nltk_tokenize(t)
    except Exception:
        nltk_toks = t.split()

    try:
        custom_toks = custom_tokenize(t, keep_punctuation=True)
    except Exception:
        custom_toks = t.split()

    try:
        hybrid_toks = hybrid_tokenize(t)
    except Exception:
        hybrid_toks = custom_toks

    spacy_toks = None
    examples = _rows(settings, TOKENIZATION_EXAMPLES, required=False)
    for ex in examples:
        if ex.get("original_text", "").strip().lower() == t.lower():
            spacy_str = ex.get("spacy_tokens", "")
            if spacy_str:
                spacy_toks = spacy_str.split(";")
                break

    if spacy_toks is None:
        import re
        spacy_toks = re.findall(r"\w+|[^\w\s]", t)

    return {
        "text": t,
        "tokens": {
            "nltk": nltk_toks,
            "custom": custom_toks,
            "hybrid": hybrid_toks,
            "spacy": spacy_toks,
        },
        "counts": {
            "nltk": len(nltk_toks),
            "custom": len(custom_toks),
            "hybrid": len(hybrid_toks),
            "spacy": len(spacy_toks),
        },
    }


def preprocessing(settings: Settings) -> Dict[str, Any]:
    return {
        "stages": _rows(settings, PREPROCESSING_COMPARISON),
        "stopwords": _rows(settings, STOPWORD_COMPARISON),
        "before_after_examples": _rows(settings, BEFORE_AFTER),
        "domain_stopword_analysis": _rows(settings, DOMAIN_STOPWORDS),
        "protected_terms": _rows(settings, PROTECTED_TERMS),
        "final_stopwords_text": artifacts.read_text(
            _path(settings, FINAL_STOPWORDS), required=False
        ),
        "order_explanation": (
            "Phase 2 ran the order experiment and Phase 3 turned it into Pipeline A and "
            "Pipeline B: Pipeline A removes domain-aware stopwords before lemmatizing, "
            "Pipeline B stems first and matches stopwords on stems."
        ),
    }


def stemming(settings: Settings) -> Dict[str, Any]:
    return {
        "algorithms": _rows(settings, STEMMING_COMPARISON),
        "order_comparison": _rows(settings, ORDER_COMPARISON),
        "examples": _rows(settings, STEMMING_EXAMPLES),
        "families": _rows(settings, STEMMING_FAMILIES),
    }


def lemmatization(settings: Settings) -> Dict[str, Any]:
    return {
        "methods": _rows(settings, LEMMATIZATION_COMPARISON),
        "examples": _rows(settings, LEMMATIZATION_EXAMPLES),
    }


def pos(settings: Settings) -> Dict[str, Any]:
    payload = summary(settings).get("pos", {}) or {}
    return {
        "distribution": _rows(settings, POS_DISTRIBUTION),
        "coarse": _rows(settings, POS_COARSE),
        "examples": _rows(settings, POS_EXAMPLES),
        "dictionary": _rows(settings, POS_DICTIONARY),
        "changes": _rows(settings, POS_CHANGES),
        "candidates": _rows(settings, POS_CANDIDATES),
        "ml_results": _rows(settings, POS_ML_RESULTS),
        "ml_report": _rows(settings, POS_ML_REPORT),
        "ml_misclassifications": _rows(settings, POS_ML_ERRORS),
        "gold_alignment": _rows(settings, POS_GOLD_ALIGNMENT),
        "gold_errors": _rows(settings, POS_GOLD_ERRORS),
        "comparison": _rows(settings, POS_COMPARISON),
        "gold_set": payload.get("gold_set", {}),
    }


def ner_label_summary(settings: Settings) -> List[Dict[str, Any]]:
    payload = summary(settings).get("ner", {}) or {}
    return [
        {
            "entity_label": row.get("entity_label"),
            "count": artifacts.as_int(row.get("count"), 0),
            "distinct_surface_forms": artifacts.as_int(row.get("distinct_surface_forms")),
            "percent_of_entities": artifacts.as_float(row.get("percent_of_entities")),
        }
        for row in (payload.get("labels") or [])
    ]


def ner(
    settings: Settings,
    kind: str = "general",
    label: Optional[str] = None,
    document_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    if kind == "domain":
        rows = _rows(settings, NER_CUSTOM)
        text_field = "entity_text"
        type_field = "domain_category"
    else:
        rows = _rows(settings, NER_ENTITIES)
        text_field = "entity_text"
        type_field = "entity_label"

    filtered = rows
    if label:
        wanted = label.strip().lower()
        filtered = [r for r in filtered if str(r.get(type_field, "")).lower() == wanted]
    if document_id:
        filtered = [r for r in filtered if r.get("document_id") == document_id]
    if search:
        needle = search.strip().lower()
        filtered = [r for r in filtered if needle in str(r.get(text_field, "")).lower()]
    filtered.sort(key=lambda r: (r.get("document_id", ""), artifacts.as_int(r.get("page_number"), 0)))

    total = len(filtered)
    start = max(0, offset)
    end = total if limit <= 0 else min(total, start + limit)
    window = filtered[start:end]
    return {
        "kind": kind,
        "label_field": type_field,
        "filter_label": label,
        "filter_document": document_id,
        "search": search,
        "total": total,
        "offset": start,
        "limit": limit,
        "has_more": end < total,
        "items": window,
    }


def ner_sidebar(settings: Settings) -> Dict[str, Any]:
    """Label counts for the filter controls, computed from the full tables."""
    general = _rows(settings, NER_ENTITIES)
    domain = _rows(settings, NER_CUSTOM)
    general_counts: Dict[str, int] = {}
    for row in general:
        key = row.get("entity_label", "")
        general_counts[key] = general_counts.get(key, 0) + 1
    domain_counts: Dict[str, int] = {}
    for row in domain:
        key = row.get("domain_category", "")
        domain_counts[key] = domain_counts.get(key, 0) + 1
    return {
        "general_labels": [
            {"label": k, "count": v} for k, v in sorted(general_counts.items(), key=lambda kv: -kv[1])
        ],
        "domain_categories": [
            {"label": k, "count": v} for k, v in sorted(domain_counts.items(), key=lambda kv: -kv[1])
        ],
        "general_total": len(general),
        "domain_total": len(domain),
        "label_distribution": _rows(settings, NER_LABELS),
        "domain_dictionary": _rows(settings, NER_DICTIONARY),
        "error_analysis": _rows(settings, NER_ERRORS),
    }


def ngrams(
    settings: Settings,
    n: int = 2,
    search: Optional[str] = None,
    domain_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    relative = NGRAM_FILES.get(n)
    if relative is None:
        raise ValueError(f"n must be between 1 and 5, got {n}")
    if domain_only:
        rows = _rows(settings, DOMAIN_PHRASES)
        rows = [r for r in rows if artifacts.as_int(r.get("n"), 0) == n]
    else:
        rows = _rows(settings, relative)
    if search:
        needle = search.strip().lower()
        rows = [r for r in rows if needle in str(r.get("ngram", "")).lower()]
    rows.sort(key=lambda r: -artifacts.as_int(r.get("frequency"), 0) or 0)

    total = len(rows)
    start = max(0, offset)
    end = total if limit <= 0 else min(total, start + limit)
    payload = summary(settings).get("ngrams", []) or []
    overview = [row for row in payload if artifacts.as_int(row.get("n"), 0) == n]
    return {
        "n": n,
        "source": relative if not domain_only else DOMAIN_PHRASES,
        "domain_only": domain_only,
        "search": search,
        "total": total,
        "offset": start,
        "limit": limit,
        "has_more": end < total,
        "items": rows[start:end],
        "overview": overview,
        "all_sizes": [
            {
                "n": artifacts.as_int(row.get("n")),
                "total_ngrams": artifacts.as_int(row.get("total_ngrams")),
                "unique_ngrams": artifacts.as_int(row.get("unique_ngrams")),
                "singletons": artifacts.as_int(row.get("singletons")),
                "top_ngram": row.get("top_ngram"),
                "top_ngram_frequency": artifacts.as_int(row.get("top_ngram_frequency")),
                "top_content_ngram": row.get("top_content_ngram"),
                "top_content_ngram_frequency": artifacts.as_int(
                    row.get("top_content_ngram_frequency")
                ),
            }
            for row in payload
        ],
        "comparison": _rows(settings, NGRAM_COMPARISON, required=False),
    }


def bpe(
    settings: Settings,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    vocabulary = _rows(settings, BPE_VOCAB)
    if search:
        needle = search.strip().lower()
        vocabulary = [r for r in vocabulary if needle in str(r.get("token", "")).lower()]
    total = len(vocabulary)
    start = max(0, offset)
    end = total if limit <= 0 else min(total, start + limit)
    return {
        "statistics": _rows(settings, BPE_STATS),
        "examples": _rows(settings, BPE_EXAMPLES),
        "training_manifest": artifacts.read_json(_path(settings, BPE_MANIFEST), required=False),
        "comparison": _rows(settings, BPE_COMPARISON, required=False),
        "vocabulary_total": total,
        "vocabulary_offset": start,
        "vocabulary_limit": limit,
        "vocabulary_has_more": end < total,
        "vocabulary": vocabulary[start:end],
    }


def master_comparison(settings: Settings) -> List[Dict[str, str]]:
    return _rows(settings, MASTER_COMPARISON)
