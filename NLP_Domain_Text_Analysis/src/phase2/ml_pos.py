"""Custom ML-based POS tagging experiment (scikit-learn).

Honesty rules implemented here
------------------------------
* spaCy's automatic tags are treated as **silver** labels: usable for training,
  never as ground truth.
* Accuracy / precision / recall / F1 are computed **only against the manually
  annotated gold set** (``data/phase2/annotations/pos_gold_annotations.csv``),
  which contains 30 hand-annotated sentences / 412 tokens labelled with the
  universal POS tag set by a single annotator (the implementing author).
* The same gold set is used to score the default tagger and the custom rule
  tagger, so the three methods in ``pos_comparison.csv`` are directly
  comparable, and the sample size is written into every artefact.
* The gold sentences are split by *sentence*, never by token, so a sentence
  cannot appear in both train and test.

Feature set (configurable in ``config/phase2_config.yaml``): token shape, affixes,
case, digit/percent/currency flags and neighbouring tokens.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .spacy_pipeline import SpacyAnnotations

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]


# ----------------------------------------------------------------------
# Gold / silver label loading
# ----------------------------------------------------------------------
@dataclass
class GoldToken:
    sentence_id: str
    document_id: str
    page_number: int
    unit_id: str
    token_index: int
    token: str
    gold_pos: str
    gold_tag: str  # Penn tag predicted by the tagger being evaluated
    corpus_token_index: int = -1  # position of this token inside its unit

    @property
    def prediction_key(self) -> Tuple[str, int]:
        """Key used to look up a per-token prediction over the whole unit."""
        return (self.unit_id, self.corpus_token_index)


def load_gold(path: Path, logger: Optional[logging.Logger] = None) -> List[GoldToken]:
    rows: List[GoldToken] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("gold_pos"):
                continue
            rows.append(
                GoldToken(
                    sentence_id=row["sentence_id"],
                    document_id=row["document_id"],
                    page_number=int(row["page_number"]),
                    unit_id=row["unit_id"],
                    token_index=int(row["token_index"]),
                    token=row.get("token", ""),
                    gold_pos=row["gold_pos"].strip(),
                    gold_tag=row.get("gold_tag", ""),
                )
            )
    if logger is not None:
        sentences = {r.sentence_id for r in rows}
        log_event(
            logger,
            "INFO",
            "ml_pos",
            f"gold set: {len(rows)} manually annotated tokens in {len(sentences)} sentences",
        )
    return rows


def align_gold_with_spacy(
    gold: Sequence[GoldToken],
    annotations: SpacyAnnotations,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[GoldToken], List[Dict[str, str]]]:
    """Attach the real corpus token text and spaCy's tags to the gold rows.

    The gold file stores the annotated *sentence* and the token index inside
    that sentence. The spaCy pass over the corpus covers whole units, so the
    sentence is first located inside its unit by character offset; only then is
    the token index resolved. Every sentence is verified token by token against
    the gold surface form, and anything that does not match is reported rather
    than guessed.
    """
    by_unit: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    for annotation, unit in zip(annotations.annotations, annotations.units):
        by_unit[annotation.unit_id] = (annotation.tokens, annotation.tags, annotation.pos)

    sentence_rows: Dict[str, List[GoldToken]] = {}
    for row in gold:
        sentence_rows.setdefault(row.sentence_id, []).append(row)

    aligned: List[GoldToken] = []
    problems: List[Dict[str, str]] = []
    modes: Dict[str, Tuple[List[int], str]] = {}
    for sentence_id, rows_in_sentence in sorted(sentence_rows.items()):
        rows_in_sentence.sort(key=lambda r: r.token_index)
        first = rows_in_sentence[0]
        entry = by_unit.get(first.unit_id)
        if entry is None:
            problems.append(
                {
                    "sentence_id": sentence_id,
                    "unit_id": first.unit_id,
                    "issue": "unit not found in the spaCy pass",
                }
            )
            continue
        tokens, tags, _pos = entry
        if sentence_id not in modes:
            annotated = [r.token for r in rows_in_sentence if r.token.strip()]
            modes[sentence_id] = _sentence_token_indices(tokens, annotated)
        positions, mode = modes[sentence_id]
        if not positions:
            problems.append(
                {
                    "sentence_id": sentence_id,
                    "unit_id": first.unit_id,
                    "issue": "annotated sentence not found as a token run in the unit text",
                }
            )
            continue
        if len(positions) != len(rows_in_sentence):
            problems.append(
                {
                    "sentence_id": sentence_id,
                    "unit_id": first.unit_id,
                    "issue": f"annotated {len(rows_in_sentence)} tokens but the corpus run has "
                    f"{len(positions)} ({mode}); only the shared prefix is scored",
                }
            )
        for row in rows_in_sentence:
            if row.token_index >= len(positions):
                problems.append(
                    {
                        "sentence_id": sentence_id,
                        "unit_id": row.unit_id,
                        "issue": f"token index {row.token_index} out of range for the sentence "
                        f"({len(positions)} tokens)",
                    }
                )
                continue
            token_position = positions[row.token_index]
            aligned.append(
                GoldToken(
                    sentence_id=row.sentence_id,
                    document_id=row.document_id,
                    page_number=row.page_number,
                    unit_id=row.unit_id,
                    token_index=row.token_index,
                    token=tokens[token_position],
                    gold_pos=row.gold_pos,
                    gold_tag=tags[token_position],
                    corpus_token_index=token_position,
                )
            )
    if logger is not None:
        mode_counts = Counter(mode for _positions, mode in modes.values())
        log_event(
            logger,
            "INFO" if not problems else "WARNING",
            "ml_pos",
            f"aligned {len(aligned)}/{len(gold)} gold tokens to the spaCy pass "
            f"({len(modes)} annotated sentences located in their units; "
            f"alignment modes: {dict(mode_counts)})"
            + (f"; {len(problems)} alignment problems reported" if problems else ""),
        )
    return aligned, problems


_GLYPH_FOLD = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
}


def _fold_glyphs(token: str) -> str:
    """Curly quotes and dashes to their ASCII form.

    Phase 1 text keeps the typographic characters of the PDF (``insurers'``),
    while the annotation file is written in ASCII. Folding only these glyphs
    keeps the comparison honest - no character class, no word or punctuation is
    otherwise relaxed.
    """
    for old, new in _GLYPH_FOLD.items():
        if old in token:
            token = token.replace(old, new)
    return token


def _sentence_token_indices(
    unit_tokens: Sequence[str],
    sentence_tokens: Sequence[str],
) -> Tuple[List[int], str]:
    """Token positions (into the unit) that make up the annotated sentence.

    Modes, tried in order and reported back so no alignment is silent:

    ``contiguous``
        the annotated surfaces appear as one uninterrupted run in the unit, which
        is the normal case for prose units;
    ``case_insensitive_contiguous`` / ``glyph_folded_contiguous``
        the same run with case or typographic glyphs folded;
    ``gap_aligned*``
        the surfaces appear in the same order but with extra unit tokens between
        them. Phase 1 table cells are newline separated, so a sentence that spans
        a cell boundary keeps its words but not its token adjacency.
    """
    target = [token for token in sentence_tokens if token]
    if not target:
        return [], "unlocatable"
    length = len(target)
    total = len(unit_tokens)

    attempts: List[Tuple[str, List[str], List[str]]] = [
        ("contiguous", list(unit_tokens), list(target)),
        (
            "case_insensitive_contiguous",
            [token.lower() for token in unit_tokens],
            [token.lower() for token in target],
        ),
        (
            "glyph_folded_contiguous",
            [_fold_glyphs(token) for token in unit_tokens],
            [_fold_glyphs(token) for token in target],
        ),
        (
            "glyph_folded_case_insensitive_contiguous",
            [_fold_glyphs(token).lower() for token in unit_tokens],
            [_fold_glyphs(token).lower() for token in target],
        ),
    ]
    for mode, haystack, needle in attempts:
        for start in range(0, total - length + 1):
            if haystack[start] != needle[0]:
                continue
            if haystack[start : start + length] == needle:
                return list(range(start, start + length)), mode

    for mode, haystack, needle in attempts:
        positions: List[int] = []
        cursor = 0
        for index, token in enumerate(haystack):
            if cursor < length and token == needle[cursor]:
                positions.append(index)
                cursor += 1
        if cursor == length:
            return positions, mode.replace("_contiguous", "") + "_gap_aligned"

    return [], "unlocatable"


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def word_shape(token: str) -> str:
    shape: List[str] = []
    previous = ""
    for char in token:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        else:
            shape.append(char)
        previous = char
    # collapse runs
    collapsed: List[str] = []
    for char in shape:
        if collapsed and collapsed[-1] == char:
            continue
        collapsed.append(char)
    return "".join(collapsed)


def token_features(
    token: str,
    prev_token: str,
    next_token: str,
    enabled: Sequence[str],
) -> Dict[str, str]:
    lower = token.casefold()
    features: Dict[str, str] = {}
    wanted = set(enabled or ())
    if "token" in wanted:
        features["token"] = token
    if "lower" in wanted:
        features["lower"] = lower
    for size in (1, 2, 3):
        if f"prefix{size}" in wanted:
            features[f"prefix{size}"] = lower[:size]
        if f"suffix{size}" in wanted:
            features[f"suffix{size}"] = lower[-size:]
    if "shape" in wanted:
        features["shape"] = word_shape(token)
    if "prev_lower" in wanted:
        features["prev_lower"] = prev_token.casefold()
    if "next_lower" in wanted:
        features["next_lower"] = next_token.casefold()
    if "is_numeric" in wanted:
        features["is_numeric"] = str(token.replace(",", "").replace(".", "").isdigit())
    if "contains_digit" in wanted:
        features["contains_digit"] = str(any(c.isdigit() for c in token))
    if "contains_percentage" in wanted:
        features["contains_percentage"] = str(("%" in token) or ("per cent" in token.casefold()))
    if "contains_currency" in wanted:
        features["contains_currency"] = str(any(c in token for c in ("\u20b9", "$", "\u20ac", "\u00a3")) or token.casefold().startswith(("rs", "inr", "usd")))
    if "is_capitalised" in wanted:
        features["is_capitalised"] = str(token[:1].isupper())
    if "is_all_caps" in wanted:
        features["is_all_caps"] = str(token.isupper() and len(token) > 1)
    return features


# ----------------------------------------------------------------------
# Metrics (computed locally so nothing is invented)
# ----------------------------------------------------------------------
def classification_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> Dict[str, object]:
    correct = sum(1 for g, p in zip(gold, predicted) if g == p)
    total = len(gold)
    labels = sorted(set(gold) | set(predicted))
    per_label: List[Dict[str, object]] = []
    macro_f1: List[float] = []
    macro_p: List[float] = []
    macro_r: List[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        macro_p.append(precision)
        macro_r.append(recall)
        macro_f1.append(f1)
        per_label.append(
            {
                "label": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
    n = len(per_label) or 1
    return {
        "sample_size_tokens": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct_tokens": correct,
        "macro_precision": round(sum(macro_p) / n, 4),
        "macro_recall": round(sum(macro_r) / n, 4),
        "macro_f1": round(sum(macro_f1) / n, 4),
        "labels": len(labels),
        "per_label": per_label,
    }


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
@dataclass
class MlPosResult:
    comparison_row: Dict[str, object]
    classification_rows: List[Dict[str, object]]
    misclassifications: List[Dict[str, object]] = field(default_factory=list)
    model_info: Dict[str, object] = field(default_factory=dict)
    gold_predictions: List[Tuple[str, str, str]] = field(default_factory=list)


def run_ml_pos(
    gold: Sequence[GoldToken],
    annotations: SpacyAnnotations,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> MlPosResult:
    """Train a small POS classifier on silver + gold-train and score it on gold-test."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split

    seed = int(config.get("reproducibility.random_seed", 42))
    test_size = float(config.get("pos.ml.test_size", 0.3))
    enabled_features = list(config.get("pos.ml.features", []) or [])

    # ---------------- training data: silver (spaCy) + gold-train ----------------
    feature_rows: List[Dict[str, str]] = []
    labels: List[str] = []
    provenance: List[str] = []

    for annotation in annotations.annotations:
        tokens = annotation.tokens
        for index, token in enumerate(tokens):
            prev_token = tokens[index - 1] if index > 0 else "<BOS>"
            next_token = tokens[index + 1] if index + 1 < len(tokens) else "<EOS>"
            feature_rows.append(token_features(token, prev_token, next_token, enabled_features))
            labels.append(annotation.pos[index])
            provenance.append("silver")

    gold_by_sentence: Dict[str, List[GoldToken]] = {}
    for row in gold:
        gold_by_sentence.setdefault(row.sentence_id, []).append(row)
    sentence_ids = sorted(gold_by_sentence)
    if len(sentence_ids) < 5:
        raise ValueError(
            f"the manually annotated gold set has only {len(sentence_ids)} aligned sentences; "
            "a train/test split by sentence is not possible. Run Phase 2 on the full corpus "
            "so the annotated units are inside the experiment sample."
        )
    train_ids, test_ids = train_test_split(sentence_ids, test_size=test_size, random_state=seed)

    gold_train_rows: List[Dict[str, str]] = []
    gold_train_labels: List[str] = []
    for sentence_id in train_ids:
        tokens = [row.token for row in sorted(gold_by_sentence[sentence_id], key=lambda r: r.token_index)]
        for row in sorted(gold_by_sentence[sentence_id], key=lambda r: r.token_index):
            position = tokens.index(row.token) if row.token in tokens else 0
            prev_token = tokens[position - 1] if position > 0 else "<BOS>"
            next_token = tokens[position + 1] if position + 1 < len(tokens) else "<EOS>"
            gold_train_rows.append(token_features(row.token, prev_token, next_token, enabled_features))
            gold_train_labels.append(row.gold_pos)
            provenance.append("gold_train")

    train_X = feature_rows + gold_train_rows
    train_y = labels + gold_train_labels

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(train_X)
    classifier = SGDClassifier(loss="log_loss", max_iter=50, tol=1e-4, random_state=seed)
    classifier.fit(X, train_y)

    # ---------------- evaluation on the held-out gold sentences ----------------
    # Context features are rebuilt per sentence, then the model is asked once
    # per sentence so the neighbour tokens are the real ones.
    test_tokens: List[str] = []
    test_gold: List[str] = []
    predicted: List[str] = []
    for sentence_id in test_ids:
        rows = sorted(gold_by_sentence[sentence_id], key=lambda r: r.token_index)
        tokens = [row.token for row in rows]
        context_rows: List[Dict[str, str]] = []
        for position, row in enumerate(rows):
            prev_token = tokens[position - 1] if position > 0 else "<BOS>"
            next_token = tokens[position + 1] if position + 1 < len(tokens) else "<EOS>"
            context_rows.append(token_features(row.token, prev_token, next_token, enabled_features))
        context_predictions = list(classifier.predict(vectorizer.transform(context_rows)))
        for offset, row in enumerate(rows):
            test_tokens.append(row.token)
            test_gold.append(row.gold_pos)
            predicted.append(context_predictions[offset])

    metrics = classification_metrics(test_gold, predicted)
    predictions_ml = metrics

    # score the default spaCy tagger on exactly the same test tokens
    default_upos = _default_upos_for(annotations, test_ids, gold_by_sentence)
    metrics_default = classification_metrics(test_gold, default_upos)

    classification_rows: List[Dict[str, object]] = []
    for row in predictions_ml["per_label"]:  # type: ignore[index]
        default_row = next(
            (d for d in predictions_default_per_label(metrics_default) if d["label"] == row["label"]),
            {},
        )
        classification_rows.append(
            {
                "method": "custom_ml_pos",
                "label": row["label"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "support": row["support"],
                "default_spacy_precision": default_row.get("precision", ""),
                "default_spacy_recall": default_row.get("recall", ""),
                "default_spacy_f1": default_row.get("f1", ""),
            }
        )

    misclassifications = [
        {
            "method": "custom_ml_pos",
            "token": test_tokens[index],
            "gold_pos": test_gold[index],
            "predicted_pos": predicted[index],
            "default_spacy_pos": default_upos[index],
            "error_type": "wrong_tag",
        }
        for index in range(len(test_gold))
        if test_gold[index] != predicted[index]
    ][:400]

    comparison_row = {
        "method": "custom_ml_pos",
        "classifier": "sklearn SGDClassifier(loss=log_loss)",
        "features": ";".join(enabled_features),
        "random_seed": seed,
        "training_tokens": len(train_X),
        "training_silver_tokens": len(labels),
        "training_gold_tokens": len(gold_train_labels),
        "gold_sentences_total": len(sentence_ids),
        "gold_sentences_train": len(train_ids),
        "gold_sentences_test": len(test_ids),
        "evaluation_sample_size_tokens": metrics["sample_size_tokens"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "default_spacy_accuracy_same_subset": metrics_default["accuracy"],
        "default_spacy_macro_f1_same_subset": metrics_default["macro_f1"],
        "feature_count": len(vectorizer.feature_names_),
        "note": (
            "accuracy is measured only on the manually annotated held-out gold "
            "sentences; spaCy tags are silver labels used for training, never as truth"
        ),
    }

    model_info = {
        "vectorizer_features": len(vectorizer.feature_names_),
        "classes": list(classifier.classes_),
        "train_sentences": list(train_ids),
        "test_sentences": list(test_ids),
    }

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "ml_pos",
            f"ML POS trained on {len(train_X)} tokens; gold-test accuracy "
            f"{metrics['accuracy']} on {metrics['sample_size_tokens']} held-out annotated tokens "
            f"(spaCy default on the same tokens: {metrics_default['accuracy']})",
        )

    return MlPosResult(
        comparison_row=comparison_row,
        classification_rows=classification_rows,
        misclassifications=misclassifications,
        model_info=model_info,
        gold_predictions=list(zip(test_tokens, test_gold, predicted)),
    )


def predictions_default_per_label(metrics: Dict[str, object]) -> List[Dict[str, object]]:
    return metrics.get("per_label", [])  # type: ignore[return-value]


def _default_upos_for(
    annotations: SpacyAnnotations,
    test_ids: Sequence[str],
    gold_by_sentence: Dict[str, List[GoldToken]],
) -> List[str]:
    """spaCy's universal POS for exactly the tokens used in the gold test split."""
    by_unit: Dict[str, List[str]] = {}
    for annotation in annotations.annotations:
        by_unit[annotation.unit_id] = list(annotation.pos)
    out: List[str] = []
    for sentence_id in test_ids:
        for row in sorted(gold_by_sentence[sentence_id], key=lambda r: r.token_index):
            unit_pos = by_unit.get(row.unit_id, [])
            if 0 <= row.corpus_token_index < len(unit_pos):
                out.append(unit_pos[row.corpus_token_index])
            else:
                out.append("X")
    return out


# ----------------------------------------------------------------------
# Scoring the other two methods on the same gold subset
# ----------------------------------------------------------------------
def score_method_on_gold(
    method: str,
    gold: Sequence[GoldToken],
    predicted_by_token: Dict[Tuple[str, int], str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Accuracy/precision/recall/F1 for a method that predicts a UPOS per token."""
    gold_labels: List[str] = []
    predictions: List[str] = []
    details: List[Dict[str, object]] = []
    for row in gold:
        key = row.prediction_key
        predicted = predicted_by_token.get(key, "X")
        gold_labels.append(row.gold_pos)
        predictions.append(predicted)
        if predicted != row.gold_pos:
            details.append(
                {
                    "method": method,
                    "sentence_id": row.sentence_id,
                    "document_id": row.document_id,
                    "page_number": row.page_number,
                    "unit_id": row.unit_id,
                    "token": row.token,
                    "token_index": row.token_index,
                    "gold_pos": row.gold_pos,
                    "predicted_pos": predicted,
                    "error_type": "wrong_tag",
                }
            )
    metrics = classification_metrics(gold_labels, predictions)
    summary = {
        "method": method,
        "evaluation_sample_size_tokens": metrics["sample_size_tokens"],
        "gold_sentences": len({row.sentence_id for row in gold}),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }
    return summary, details
