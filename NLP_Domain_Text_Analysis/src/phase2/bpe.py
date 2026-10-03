"""Byte-level BPE training and analysis on the Phase 1 corpus.

A ``ByteLevelBPETokenizer`` from HuggingFace ``tokenizers`` is trained on the
corpus itself, so the merge table is derived from financial text rather than a
generic English vocabulary. The trained tokenizer is persisted to
``results/phase2/bpe/tokenizer_model`` so the run is reproducible without
retraining.

The interesting question for this domain is how sub-word units treat rare
financial vocabulary (``securitisation``, ``disinflation``) versus high-frequency
general words, and how many tokens a whole corpus costs under each scheme. Both
are measured.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .load_corpus import ExperimentSample
from .statistics import TokenizedCorpus

RE_SPECIAL = re.compile(r"\s+|(?!\S)\s+")


@dataclass
class BpeArtifacts:
    tokenizer: object
    vocabulary: List[Tuple[str, int]]
    merges: int
    vocab_size: int
    corpus_token_count: int
    stats_rows: List[Dict[str, object]]
    example_rows: List[Dict[str, object]]


def train_bpe(
    sample: ExperimentSample,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
) -> BpeArtifacts:
    """Train the BPE model and gather its statistics."""
    from tokenizers import ByteLevelBPETokenizer

    model_dir = config.out_dir("bpe_model_dir")
    vocab_size = int(config.get("bpe.vocab_size", 8000))
    min_frequency = int(config.get("bpe.min_frequency", 2))
    special_tokens = list(config.get("bpe.special_tokens", []) or [])
    seed = int(config.get("reproducibility.random_seed", 42))
    training_mode = str(config.get("bpe.training_corpus", "all_selected_units"))

    # ---- training corpus: one text file per training document ----
    # (a file list is used rather than a directory because the Rust trainer
    #  cannot walk directories on this platform)
    train_dir = model_dir.parent / "bpe_training_corpus"
    train_dir.mkdir(parents=True, exist_ok=True)
    for stale in train_dir.glob("*.txt"):
        stale.unlink()
    documents = sorted({unit.document_id for unit in sample.units})
    held_out: set = set()
    if training_mode == "train_split":
        ratio = float(config.get("bpe.train_split_document_ratio", 0.2))
        count = max(1, int(round(len(documents) * ratio)))
        held_out = set(documents[-count:])

    train_units = [u for u in sample.units if u.document_id not in held_out]
    per_document: Dict[str, List[str]] = {}
    for unit in train_units:
        per_document.setdefault(unit.document_id, []).append(unit.text)
    training_files: List[str] = []
    for document_id, texts in sorted(per_document.items()):
        path = train_dir / f"{document_id}.txt"
        path.write_text("\n".join(texts), encoding="utf-8")
        training_files.append(str(path))

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        training_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.save_model(str(model_dir))
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "training_mode": training_mode,
                "training_documents": len(training_files),
                "training_units": len(train_units),
                "training_files": [Path(p).name for p in training_files],
                "held_out_documents": sorted(held_out),
                "vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "special_tokens": special_tokens,
                "random_seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_vocab: Dict[str, int] = tokenizer.get_vocab()
    vocabulary = sorted(raw_vocab.items(), key=lambda row: (-row[1], row[0]))

    # ---- corpus-wide cost ----
    corpus_token_count = 0
    unit_token_counts: List[int] = []
    for unit in sample.units:
        encoded = tokenizer.encode(unit.text).tokens
        corpus_token_count += len(encoded)
        unit_token_counts.append(len(encoded))

    stats_rows = build_statistics(
        tokenizer=tokenizer,
        sample=sample,
        config=config,
        corpus_token_count=corpus_token_count,
        unit_token_counts=unit_token_counts,
        vocab_size=len(raw_vocab),
        training_units=len(train_units),
    )
    example_rows = build_examples(tokenizer, config)

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "bpe",
            f"BPE trained on {len(train_units)} units: vocab {len(raw_vocab)}, "
            f"corpus token count {corpus_token_count} (hybrid tokenizer: see comparison tables)",
        )

    return BpeArtifacts(
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        merges=len(raw_vocab) - len(special_tokens),
        vocab_size=len(raw_vocab),
        corpus_token_count=corpus_token_count,
        stats_rows=stats_rows,
        example_rows=example_rows,
    )


def build_statistics(
    tokenizer,
    sample: ExperimentSample,
    config: Phase2Config,
    corpus_token_count: int,
    unit_token_counts: Sequence[int],
    vocab_size: int,
    training_units: int,
) -> List[Dict[str, object]]:
    """Vocabulary size, token count, compression and sub-word behaviour."""
    observed_vocab: Counter = Counter()
    for unit in sample.units:
        for token in unit.text.casefold().split():
            cleaned = token.strip(".,;:()[]{}\"'")
            if cleaned:
                observed_vocab[cleaned] += 1

    rows: List[Dict[str, object]] = [
        {
            "metric": "vocabulary_size",
            "value": vocab_size,
            "detail": "size of the trained BPE vocabulary (includes special tokens)",
        },
        {
            "metric": "merge_operations",
            "value": max(0, vocab_size - len(list(config.get("bpe.special_tokens", []) or []))),
            "detail": "vocabulary entries beyond the special tokens; each corresponds to one learned merge",
        },
        {
            "metric": "training_units",
            "value": training_units,
            "detail": f"corpus units used for training (policy '{sample.policy}')",
        },
        {
            "metric": "corpus_token_count",
            "value": corpus_token_count,
            "detail": "tokens produced by BPE over the whole experiment sample",
        },
        {
            "metric": "corpus_characters",
            "value": sum(len(unit.text) for unit in sample.units),
            "detail": "characters in the experiment sample (compression reference)",
        },
        {
            "metric": "bytes_per_token",
            "value": round(
                sum(len(unit.text) for unit in sample.units) / corpus_token_count, 3
            )
            if corpus_token_count
            else 0.0,
            "detail": "characters per BPE token",
        },
        {
            "metric": "observed_word_types",
            "value": len(observed_vocab),
            "detail": "distinct whitespace word types in the sample (pre-tokenization reference)",
        },
        {
            "metric": "out_of_vocabulary_words",
            "value": 0,
            "detail": "byte-level BPE has no OOV: every byte sequence is representable",
        },
    ]

    # ---- sub-word behaviour on probe words ----
    probe_groups: Dict[str, Sequence[str]] = {
        "common": list(config.get("bpe.probe_words.common", []) or []),
        "financial": list(config.get("bpe.probe_words.financial", []) or []),
        "abbreviation": list(config.get("bpe.probe_words.abbreviations", []) or []),
        "number_date": list(config.get("bpe.probe_words.numbers", []) or []),
    }
    for group, words in probe_groups.items():
        for word in words:
            tokens = tokenizer.encode(word).tokens
            rows.append(
                {
                    "metric": f"subword_tokens[{group}]",
                    "value": len(tokens),
                    "detail": f"{word} -> {' + '.join(tokens)}",
                }
            )
    return rows


def build_examples(tokenizer, config: Phase2Config) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_rows = int(config.get("bpe.max_examples_written", 200))
    for word in list(config.get("bpe.probe_words.common", []) or []) + list(
        config.get("bpe.probe_words.financial", [])
    ) + list(config.get("bpe.probe_words.abbreviations", []) or []) + list(
        config.get("bpe.probe_words.numbers", []) or []
    ):
        encoded = tokenizer.encode(word)
        tokens = encoded.tokens
        rows.append(
            {
                "example": word,
                "category": _category_of(word, config),
                "bpe_tokens": tokens,
                "bpe_token_count": len(tokens),
                "is_single_token": len(tokens) == 1,
                "pieces_rejoined": "".join(tokens).replace("\u0120", " "),
                "byte_level_marked": "true",
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def _category_of(word: str, config: Phase2Config) -> str:
    lower = word.casefold()
    for group, words in (
        ("common", config.get("bpe.probe_words.common", []) or []),
        ("financial", config.get("bpe.probe_words.financial", []) or []),
        ("abbreviation", config.get("bpe.probe_words.abbreviations", []) or []),
        ("number_date", config.get("bpe.probe_words.numbers", []) or []),
    ):
        if word in words or lower in {str(w).casefold() for w in words}:
            return group
    return "other"


def vocabulary_rows(artifacts: BpeArtifacts, limit: int = 8000) -> List[Dict[str, object]]:
    return [
        {"token_id": index, "token": token, "is_special": token in {"<unk>", "<pad>"}}
        for index, (token, _) in enumerate(artifacts.vocabulary[:limit])
    ]


def encode_corpus_tokens(artifacts: BpeArtifacts, unit_texts: Sequence[str]) -> List[List[str]]:
    return [artifacts.tokenizer.encode(text).tokens for text in unit_texts]
