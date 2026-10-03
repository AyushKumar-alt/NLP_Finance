"""Read the Phase 2 results Phase 3 depends on.

Nothing here recomputes a Phase 2 experiment. Every value is read from the
artefacts Phase 2 wrote, and every read is checked: a missing or malformed file
raises a clear error instead of producing an invented value.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.utils import read_csv_rows, read_jsonl
from src.phase2.config import log_event as phase2_log_event  # noqa: F401  (re-export friendly)

from .config import Phase3Config, log_event


@dataclass
class Phase2Evidence:
    """Phase 2 artefacts, loaded once and shared by every Phase 3 step."""

    config: Phase3Config
    tokenizer_comparison: List[Dict[str, str]] = field(default_factory=list)
    stopword_comparison: List[Dict[str, str]] = field(default_factory=list)
    stemming_comparison: List[Dict[str, str]] = field(default_factory=list)
    order_comparison: List[Dict[str, str]] = field(default_factory=list)
    lemmatization_comparison: List[Dict[str, str]] = field(default_factory=list)
    date_number_comparison: List[Dict[str, str]] = field(default_factory=list)
    pos_comparison: List[Dict[str, str]] = field(default_factory=list)
    domain_phrases: List[Dict[str, str]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    validation_report: List[Dict[str, str]] = field(default_factory=list)
    ner_entities_by_unit: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    domain_entities_by_unit: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    ngram_counts: Dict[int, Dict[str, int]] = field(default_factory=dict)
    files_read: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def tokeniser_row(self, name: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.tokenizer_comparison if r.get("tokenizer") == name), None)

    def stopword_row(self, name: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.stopword_comparison if r.get("strategy") == name), None)

    def lemmatization_row(self, name: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.lemmatization_comparison if r.get("method") == name), None)

    def stemming_row(self, name: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.stemming_comparison if r.get("algorithm") == name), None)

    def order_row(self, fragment: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.order_comparison if fragment in str(r.get("pipeline", ""))), None)

    def date_number_row(self, category: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.date_number_comparison if r.get("category") == category), None)

    def pos_row(self, method: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.pos_comparison if r.get("method") == method), None)

    def phase2_passed(self) -> Optional[int]:
        for row in self.validation_report:
            if row.get("status") == "PASS":
                continue
            return None
        return len(self.validation_report) if self.validation_report else None

    def citations(self) -> List[str]:
        """Human readable statements of Phase 2 facts used by Phase 3."""
        out: List[str] = []
        hybrid = self.tokeniser_row("hybrid")
        nltk = self.tokeniser_row("nltk")
        spacy = self.tokeniser_row("spacy")
        custom = self.tokeniser_row("custom")
        if hybrid and nltk:
            out.append(
                f"Phase 2 tokenization: hybrid produced {hybrid['total_tokens']} tokens "
                f"(fewest of the four methods) and kept {hybrid['percentage_tokens']} "
                f"percentage tokens against NLTK's {nltk['percentage_tokens']} "
                f"(tokenization/tokenization_comparison.csv)"
            )
        if hybrid and spacy:
            out.append(
                f"Phase 2 tokenization: hybrid kept {hybrid['fiscal_year_tokens']} fiscal-year "
                f"tokens against spaCy's {spacy['fiscal_year_tokens']} "
                f"(tokenization/tokenization_comparison.csv)"
            )
        if custom and nltk:
            out.append(
                f"Phase 2 tokenization: the custom rule tokenizer detected "
                f"{custom['percentage_tokens']} percentage tokens against NLTK's "
                f"{nltk['percentage_tokens']} (tokenization/tokenization_comparison.csv)"
            )
        total = self.date_number_row("TOTAL")
        if total:
            out.append(
                f"Phase 2 date/number experiment: {total['expressions_detected']} typed financial "
                f"expressions were detected but only {total['expressions_intact_after_standard_tokenization']} "
                f"({total['intact_rate_percent']}%) survived standard tokenization intact "
                f"(tokenization/date_number_comparison.csv)"
            )
        standard = self.stopword_row("standard_english")
        domain = self.stopword_row("domain_aware")
        if standard and domain:
            out.append(
                f"Phase 2 stopwords: standard list removed {standard['tokens_removed']} tokens "
                f"({standard['token_reduction_percent']}%), domain-aware "
                f"{domain['tokens_removed']} ({domain['token_reduction_percent']}%), because 43 "
                f"protected financial terms survive (preprocessing/stopword_comparison.csv)"
            )
        first = self.order_row("stopword_removal_then_stemming")
        second = self.order_row("stemming_then_stopword_removal")
        if first and second:
            out.append(
                "Phase 2 order experiment: stopword-removal-then-stemming left "
                f"{first['total_tokens']} tokens, stemming-then-stopword-removal left "
                f"{second['total_tokens']}; the wrong order left "
                f"{len([t for t in second.get('count_only_in_this_pipeline', '').split(';') if t])} "
                "unmatched stems such as 'abov', 'everi', 'furthermor' "
                "(stemming/stopword_stemming_order_comparison.csv)"
            )
        lemma = self.lemmatization_row("wordnet_lookup_pos_agnostic")
        if lemma:
            out.append(
                f"Phase 2 lemmatization: WordNet POS-agnostic lookup reduced the vocabulary to "
                f"{lemma['unique_lemmas']} ({lemma['vocabulary_reduction_percent']}% reduction) "
                f"(lemmatization/lemmatization_comparison.csv)"
            )
        spacy_lemma = self.lemmatization_row("spacy_rule_based")
        if spacy_lemma:
            out.append(
                f"Phase 2 lemmatization: spaCy's rule lemmatizer kept "
                f"{spacy_lemma['unique_lemmas']} lemmas, i.e. it normalizes less aggressively "
                f"than the WordNet lookup (lemmatization/lemmatization_comparison.csv)"
            )
        for method in ("default_pos", "custom_rule_pos", "custom_ml_pos"):
            row = self.pos_row(method)
            if row:
                out.append(
                    f"Phase 2 POS: {method} scored {row['accuracy_if_available']} accuracy on "
                    f"{row['dataset_size']} manually annotated tokens "
                    f"(comparisons/pos_comparison.csv)"
                )
        return out

    def summary_line(self) -> str:
        corpus = self.summary.get("corpus", {}) if isinstance(self.summary, dict) else {}
        return (
            f"Phase 2 sample: {corpus.get('selected_units', 'n/a')} units, "
            f"{corpus.get('selected_characters', 'n/a')} characters, policy "
            f"'{corpus.get('text_selection_policy', 'n/a')}'"
        )


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _read_csv(path: Path, label: str, required: bool, problems: List[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        message = f"Phase 2 {label} not found: {path}"
        if required:
            raise FileNotFoundError(message + "\nRun Phase 2 first:  python -m src.phase2.run")
        problems.append(message)
        return []
    try:
        return read_csv_rows(path)
    except Exception as error:  # pragma: no cover - defensive
        message = f"Phase 2 {label} could not be read ({error}): {path}"
        if required:
            raise ValueError(message) from error
        problems.append(message)
        return []


def _read_json(path: Path, label: str, problems: List[str]) -> Dict[str, Any]:
    if not path.is_file():
        problems.append(f"Phase 2 {label} not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        problems.append(f"Phase 2 {label} is not valid JSON ({error}): {path}")
        return {}


def _read_ngram_counts(ngrams_dir: Path, sizes: List[int], problems: List[str]) -> Dict[int, Dict[str, int]]:
    """Total/unique counts per n from the Phase 2 n-gram summary, not recomputed."""
    counts: Dict[int, Dict[str, int]] = {}
    names = {1: "unigram_results.csv", 2: "bigram_results.csv", 3: "trigram_results.csv",
             4: "fourgram_results.csv", 5: "fivegram_results.csv"}
    for n in sizes:
        path = ngrams_dir / names.get(n, f"{n}gram_results.csv")
        if not path.is_file():
            problems.append(f"Phase 2 n-gram table missing: {path}")
            continue
        rows = _read_csv(path, f"{n}-gram table", False, problems)
        if not rows:
            continue
        total = sum(int(float(r.get("frequency", 0) or 0)) for r in rows)
        unique = len(rows)
        counts[n] = {"rows_written": unique, "frequency_sum_in_written_rows": total}
    return counts


def load_phase2_evidence(
    config: Phase3Config,
    logger: Optional[logging.Logger] = None,
) -> Phase2Evidence:
    """Load every Phase 2 artefact Phase 3 cites, validating presence."""
    strict = bool(config.get("run.strict_input_check", True))
    problems: List[str] = []
    evidence = Phase2Evidence(config=config)

    evidence.tokenizer_comparison = _read_csv(
        config.input_path("phase2_tokenization_comparison"), "tokenization comparison", strict, problems
    )
    evidence.stopword_comparison = _read_csv(
        config.input_path("phase2_stopword_comparison"), "stopword comparison", strict, problems
    )
    evidence.stemming_comparison = _read_csv(
        config.input_path("phase2_stemming_comparison"), "stemming comparison", strict, problems
    )
    evidence.order_comparison = _read_csv(
        config.input_path("phase2_order_comparison"), "stemming/stopword order comparison", strict, problems
    )
    evidence.lemmatization_comparison = _read_csv(
        config.input_path("phase2_lemmatization_comparison"), "lemmatization comparison", strict, problems
    )
    evidence.date_number_comparison = _read_csv(
        config.input_path("phase2_date_number_comparison"), "date/number comparison", strict, problems
    )
    evidence.pos_comparison = _read_csv(
        config.input_path("phase2_pos_comparison"), "POS comparison", False, problems
    )
    evidence.domain_phrases = _read_csv(
        config.input_path("phase2_domain_phrases"), "domain phrase table", False, problems
    )
    evidence.validation_report = _read_csv(
        config.input_path("phase2_validation_report"), "validation report", False, problems
    )
    evidence.summary = _read_json(config.input_path("phase2_ngram_summary"), "summary JSON", problems)

    # NER is consumed as per-unit metadata instead of being recomputed.
    for key, target in (
        ("phase2_ner_entities", evidence.ner_entities_by_unit),
        ("phase2_domain_entities", evidence.domain_entities_by_unit),
    ):
        rows = _read_csv(config.input_path(key), key, False, problems)
        if not rows:
            continue
        for row in rows:
            target.setdefault(str(row.get("unit_id", "")), []).append(row)

    sizes = [int(n) for n in (config.get("ngrams.sizes", [2, 3]) or [2, 3])] + [1, 5]
    evidence.ngram_counts = _read_ngram_counts(config.input_path("phase2_ngrams_dir"), sorted(set(sizes)), problems)

    evidence.files_read = [
        key for key in config.section("input")
        if str(key).startswith("phase2_") and config.project_path(str(config.get(f"input.{key}"))).exists()
    ]

    if logger is not None:
        for problem in problems:
            log_event(logger, "WARNING", "phase2_inputs", problem)
        log_event(
            logger,
            "INFO",
            "phase2_inputs",
            f"loaded Phase 2 evidence: {evidence.summary_line()}; "
            f"{len(evidence.ner_entities_by_unit)} units carry Phase 2 entities, "
            f"{len(evidence.domain_entities_by_unit)} units carry domain mentions",
        )
    evidence.missing_inputs = problems  # type: ignore[attr-defined]
    return evidence
