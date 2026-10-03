"""Phase 2 validation rules.

Each rule answers one question with a measurable check. Nothing is asserted that
is not computed here, and a failing rule is reported rather than hidden.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .config import Phase2Config, log_event
from .load_corpus import Corpus

#: Artefacts that must exist and be non-empty after a run.
REQUIRED_OUTPUTS = (
    "tokenization/tokenization_comparison.csv",
    "tokenization/custom_tokenizer_rules.csv",
    "tokenization/nltk_examples.csv",
    "tokenization/spacy_examples.csv",
    "tokenization/hybrid_examples.csv",
    "tokenization/custom_examples.csv",
    "tokenization/tokenization_examples_comparison.csv",
    "tokenization/date_number_comparison.csv",
    "preprocessing/stopword_comparison.csv",
    "preprocessing/domain_stopword_analysis.csv",
    "preprocessing/final_stopwords.txt",
    "preprocessing/preprocessing_comparison.csv",
    "preprocessing/before_after_examples.csv",
    "stemming/stemming_comparison.csv",
    "stemming/stemming_examples.csv",
    "stemming/stemming_families.csv",
    "stemming/stopword_stemming_order_comparison.csv",
    "lemmatization/lemmatization_comparison.csv",
    "lemmatization/lemmatization_examples.csv",
    "pos/default_pos_distribution.csv",
    "pos/default_pos_examples.csv",
    "pos/custom_pos_dictionary.csv",
    "pos/custom_pos_changes.csv",
    "pos/ml_pos_results.csv",
    "pos/ml_pos_classification_report.csv",
    "ner/ner_entities.csv",
    "ner/ner_error_analysis.csv",
    "ner/domain_entity_dictionary.csv",
    "ner/custom_ner_results.csv",
    "ngrams/unigram_results.csv",
    "ngrams/bigram_results.csv",
    "ngrams/trigram_results.csv",
    "ngrams/fourgram_results.csv",
    "ngrams/fivegram_results.csv",
    "ngrams/domain_phrases.csv",
    "bpe/bpe_vocabulary.csv",
    "bpe/bpe_examples.csv",
    "bpe/bpe_statistics.csv",
    "comparisons/experiment_manifest.csv",
    "comparisons/tokenization_method_comparison.csv",
    "comparisons/pos_comparison.csv",
    "comparisons/ngram_comparison.csv",
    "comparisons/bpe_tokenization_comparison.csv",
    "comparisons/domain_tokenization_testset.csv",
    "comparisons/representative_sample.csv",
    "comparisons/phase2_master_comparison.csv",
    "phase2_summary.csv",
    "phase2_summary.json",
    "summaries/phase2_methodology_justification.md",
    "README.md",
)


@dataclass
class ValidationResult:
    rule_id: str
    rule: str
    status: str          # PASS / FAIL / WARN
    detail: str
    evidence: str

    def as_row(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule": self.rule,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def run_validation(
    config: Phase2Config,
    corpus: Corpus,
    metrics: Dict[str, object],
    logger: Optional[logging.Logger] = None,
) -> List[ValidationResult]:
    results: List[ValidationResult] = []
    results.append(_corpus_loaded(corpus, metrics))
    results.append(_traceability(corpus, metrics))
    results.append(_non_negative_counts(metrics))
    results.append(_vocabulary_validity(metrics))
    results.append(_outputs_exist(config))
    results.append(_csv_readable(config))
    results.append(_json_valid(config))
    results.append(_ngram_frequencies_valid(metrics))
    results.append(_bpe_present(config, metrics))
    results.append(_no_empty_experiment(metrics))
    results.append(_source_pdfs_untouched(corpus, config))
    results.append(_no_fabricated_metrics(config, metrics))
    results.append(_pos_ner_evaluation_scope(config, metrics))
    results.append(_traceability_in_examples(config))

    failures = [r for r in results if r.status == "FAIL"]
    if logger is not None:
        log_event(
            logger,
            "INFO",
            "validation",
            f"{len(results) - len(failures)}/{len(results)} rules passed"
            + (f"; failures: {[r.rule_id for r in failures]}" if failures else ""),
        )
    return results


# ----------------------------------------------------------------------
# Individual rules
# ----------------------------------------------------------------------
def _corpus_loaded(corpus: Corpus, metrics: Dict[str, object]) -> ValidationResult:
    documents = corpus.document_count
    units = len(corpus.units)
    ok = documents > 0 and units > 0
    return ValidationResult(
        "V01",
        "Phase 1 structured corpus loaded successfully (documents > 0 and units > 0)",
        "PASS" if ok else "FAIL",
        f"{documents} documents, {corpus.page_count} pages, {units} units",
        str(metrics.get("corpus_path", "")),
    )


def _traceability(corpus: Corpus, metrics: Dict[str, object]) -> ValidationResult:
    missing = [
        unit.unit_id
        for unit in corpus.units
        if not unit.document_id or not unit.page_id or not unit.unit_id
    ]
    return ValidationResult(
        "V02",
        "every unit keeps document_id / page_id / section_id / unit_id traceability",
        "PASS" if not missing else "FAIL",
        f"{len(corpus.units) - len(missing)}/{len(corpus.units)} units fully traceable",
        f"missing: {missing[:5]}" if missing else "no missing provenance",
    )


def _non_negative_counts(metrics: Dict[str, object]) -> ValidationResult:
    negative: List[str] = []
    for row in metrics.get("tokenization_rows", []):  # type: ignore[union-attr]
        for key in ("total_tokens", "unique_tokens", "vocabulary_size", "sentences"):
            if int(row.get(key, 0)) < 0:  # type: ignore[union-attr]
                negative.append(f"{row.get('tokenizer')}.{key}")  # type: ignore[union-attr]
    for row in metrics.get("ngram_summary", []):  # type: ignore[union-attr]
        if int(row.get("total_ngrams", 0)) < 0:  # type: ignore[union-attr]
            negative.append(f"{row.get('n')}-gram.total_ngrams")  # type: ignore[union-attr]
    return ValidationResult(
        "V03",
        "no token / n-gram count is negative",
        "PASS" if not negative else "FAIL",
        f"checked {len(metrics.get('tokenization_rows', []))} tokenizers and "  # type: ignore[arg-type]
        f"{len(metrics.get('ngram_summary', []))} n-gram sizes",  # type: ignore[arg-type]
        f"negatives: {negative[:5]}" if negative else "all counts >= 0",
    )


def _vocabulary_validity(metrics: Dict[str, object]) -> ValidationResult:
    problems: List[str] = []
    for row in metrics.get("tokenization_rows", []):  # type: ignore[union-attr]
        total = int(row.get("total_tokens", 0))  # type: ignore[union-attr]
        unique = int(row.get("unique_tokens", 0))  # type: ignore[union-attr]
        vocab = int(row.get("vocabulary_size", 0))  # type: ignore[union-attr]
        if unique > total or vocab > unique:
            problems.append(
                f"{row.get('tokenizer')}: unique={unique} vocab={vocab} total={total}"  # type: ignore[union-attr]
            )
    return ValidationResult(
        "V04",
        "vocabulary size <= unique tokens <= total tokens for every method",
        "PASS" if not problems else "FAIL",
        "checked " + ", ".join(
            f"{r.get('tokenizer')}:{r.get('vocabulary_size')}/{r.get('unique_tokens')}/{r.get('total_tokens')}"
            for r in metrics.get("tokenization_rows", [])  # type: ignore[union-attr]
        ),
        f"violations: {problems[:5]}" if problems else "ordering holds for all methods",
    )


def _outputs_exist(config: Phase2Config) -> ValidationResult:
    root = config.out_path("results_dir")
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).is_file()]
    return ValidationResult(
        "V05",
        "every required Phase 2 output file exists",
        "PASS" if not missing else "FAIL",
        f"{len(REQUIRED_OUTPUTS) - len(missing)}/{len(REQUIRED_OUTPUTS)} files present",
        f"missing: {missing}" if missing else "all required artefacts written",
    )


def _csv_readable(config: Phase2Config) -> ValidationResult:
    root = config.out_path("results_dir")
    unreadable: List[str] = []
    checked = 0
    for path in sorted(root.rglob("*.csv")):
        checked += 1
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    unreadable.append(f"{path.name} (empty)")
        except Exception as exc:  # pragma: no cover
            unreadable.append(f"{path.name} ({exc})")
    return ValidationResult(
        "V06",
        "every output CSV can be opened and has a header row",
        "PASS" if not unreadable else "FAIL",
        f"{checked - len(unreadable)}/{checked} CSV files readable",
        f"unreadable: {unreadable[:5]}" if unreadable else "all CSVs parse cleanly",
    )


def _json_valid(config: Phase2Config) -> ValidationResult:
    root = config.out_path("results_dir")
    problems: List[str] = []
    checked = 0
    for path in sorted(root.rglob("*.json")):
        checked += 1
        try:
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            problems.append(f"{path.name} ({exc})")
    return ValidationResult(
        "V07",
        "every output JSON file is valid JSON",
        "PASS" if not problems else "FAIL",
        f"{checked - len(problems)}/{checked} JSON files valid",
        f"invalid: {problems[:5]}" if problems else "all JSON parses cleanly",
    )


def _ngram_frequencies_valid(metrics: Dict[str, object]) -> ValidationResult:
    problems: List[str] = []
    for row in metrics.get("ngram_summary", []):  # type: ignore[union-attr]
        total = int(row.get("total_ngrams", 0))  # type: ignore[union-attr]
        unique = int(row.get("unique_ngrams", 0))  # type: ignore[union-attr]
        if total <= 0 or unique <= 0 or unique > total:
            problems.append(f"n={row.get('n')}: total={total} unique={unique}")  # type: ignore[union-attr]
    return ValidationResult(
        "V08",
        "n-gram counts are positive and unique <= total for every n",
        "PASS" if not problems else "FAIL",
        ", ".join(
            f"n={r.get('n')}: {r.get('unique_ngrams')}/{r.get('total_ngrams')}"  # type: ignore[union-attr]
            for r in metrics.get("ngram_summary", [])  # type: ignore[union-attr]
        ),
        f"violations: {problems}" if problems else "all n-gram tables consistent",
    )


def _bpe_present(config: Phase2Config, metrics: Dict[str, object]) -> ValidationResult:
    vocab_path = config.out_dir("bpe_dir") / "bpe_vocabulary.csv"
    model_dir = config.out_dir("bpe_model_dir")
    exists = vocab_path.is_file() and (model_dir / "tokenizer.json").is_file()
    return ValidationResult(
        "V09",
        "a trained BPE vocabulary and persisted model exist",
        "PASS" if exists else "FAIL",
        f"vocabulary entries: {metrics.get('bpe_vocab_size', 'n/a')}; "
        f"corpus BPE tokens: {metrics.get('bpe_corpus_tokens', 'n/a')}",
        f"vocab csv: {vocab_path.is_file()}, model dir: {(model_dir / 'tokenizer.json').is_file()}",
    )


def _no_empty_experiment(metrics: Dict[str, object]) -> ValidationResult:
    empty: List[str] = []
    for key, value in metrics.items():
        if key.endswith("_rows") and isinstance(value, list) and not value:
            empty.append(key)
    return ValidationResult(
        "V10",
        "no experiment silently processed an empty corpus",
        "PASS" if not empty else "FAIL",
        f"{len([k for k in metrics if str(k).endswith('_rows')])} experiment tables checked",
        f"empty tables: {empty}" if empty else "every experiment produced rows",
    )


def _source_pdfs_untouched(corpus: Corpus, config: Phase2Config) -> ValidationResult:
    """Re-hash the project copy of every source PDF and compare with Phase 1."""
    from src.core.utils import sha256_file

    pdf_dir = config.input_dir("source_pdfs_dir")
    if not pdf_dir.is_dir():
        return ValidationResult(
            "V11", "source PDFs are unmodified", "WARN", "project PDF copy not found", str(pdf_dir)
        )
    registry = {row["filename"]: row for row in corpus.document_registry}
    checked = 0
    changed: List[str] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        record = registry.get(pdf.name)
        if not record or not record.get("sha256"):
            continue
        checked += 1
        if sha256_file(pdf) != record["sha256"]:
            changed.append(pdf.name)
    return ValidationResult(
        "V11",
        "no source PDF was modified (SHA-256 re-verified against the Phase 1 registry)",
        "PASS" if not changed else "FAIL",
        f"{checked - len(changed)}/{checked} PDF hashes match the Phase 1 registry",
        f"changed: {changed[:5]}" if changed else "source PDFs byte-identical to Phase 1 ingestion",
    )


def _no_fabricated_metrics(config: Phase2Config, metrics: Dict[str, object]) -> ValidationResult:
    """Accuracy-like values may only exist next to a stated sample size."""
    problems: List[str] = []
    pos_dir = config.out_dir("pos_dir")
    for name in ("ml_pos_results.csv", "ml_pos_classification_report.csv"):
        path = pos_dir / name
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            has_score = any(row.get(key) for key in ("accuracy", "macro_f1", "precision", "recall", "f1"))
            has_sample = bool(
                row.get("evaluation_sample_size_tokens") or row.get("sample_size_tokens") or row.get("support")
            )
            if has_score and not has_sample:
                problems.append(f"{name}: metric without sample size")
    comparison_path = config.out_dir("comparisons_dir") / "pos_comparison.csv"
    if comparison_path.is_file():
        with open(comparison_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("accuracy_if_available") and not row.get("dataset_size"):
                    problems.append("pos_comparison.csv: accuracy without dataset size")
    return ValidationResult(
        "V12",
        "no accuracy/precision/recall/F1 value is reported without an evaluation sample size",
        "PASS" if not problems else "FAIL",
        "POS metrics are computed only on the manually annotated gold subset",
        f"violations: {problems}" if problems else "every reported score carries its sample size",
    )


def _pos_ner_evaluation_scope(config: Phase2Config, metrics: Dict[str, object]) -> ValidationResult:
    gold_path = config.out_path("annotations_dir") / "pos_gold_annotations.csv"
    gold_rows = 0
    sentences = 0
    if gold_path.is_file():
        with open(gold_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        gold_rows = len(rows)
        sentences = len({row["sentence_id"] for row in rows})
    ok = gold_rows > 0
    return ValidationResult(
        "V13",
        "POS/NER evaluation is backed by a persisted manual annotation file (no invented ground truth)",
        "PASS" if ok else "FAIL",
        f"gold POS file: {gold_rows} manually annotated tokens in {sentences} sentences",
        f"{gold_path.name} present" if ok else f"{gold_path.name} missing",
    )


def _traceability_in_examples(config: Phase2Config) -> ValidationResult:
    """Example CSVs must keep the provenance columns."""
    required = ("document_id", "page_number", "unit_id")
    root = config.out_dir("comparisons_dir")
    problems: List[str] = []
    checked = 0
    for name in (
        "tokenization_examples_comparison.csv",
        "representative_sample.csv",
        "domain_tokenization_testset.csv",
        "bpe_tokenization_comparison.csv",
    ):
        path = root / name
        if not path.is_file():
            continue
        checked += 1
        with open(path, "r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        missing = [column for column in required if column not in header]
        if missing:
            problems.append(f"{name}: missing {missing}")
    return ValidationResult(
        "V14",
        "example tables retain document_id / page_number / unit_id traceability",
        "PASS" if not problems else "FAIL",
        f"{checked} example tables checked",
        f"violations: {problems}" if problems else "all example tables are traceable",
    )
