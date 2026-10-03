"""One shared spaCy pass for every experiment that needs spaCy.

The assignment requires four separate experiments (tokenization, POS tagging,
NER, lemmatization) but running the pipeline four times over the corpus would
be wasteful and could produce inconsistent results if the model were ever
reloaded differently. This module therefore performs **one** batched pass and
hands the compact per-unit results to the individual experiments.

Stored per unit (compact tuples, not ``Doc`` objects, to keep memory flat):

    tokens, tags (Penn Treebank), pos (universal), lemmas, entities
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .load_corpus import Unit, normalise_for_tokenisation
from .tokenizers import load_spacy


@dataclass
class UnitAnnotation:
    unit_id: str
    document_id: str
    page_number: int
    unit_type: str
    tokens: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    pos: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    sentences: List[int] = field(default_factory=list)
    entities: List[Tuple[str, str, int, int]] = field(default_factory=list)  # text,label,start,end

    @property
    def sentence_count(self) -> int:
        return max(1, len(self.sentences))


@dataclass
class SpacyAnnotations:
    """Annotations for a list of units, aligned index-for-index."""

    model_name: str
    model_version: str
    units: List[Unit]
    annotations: List[UnitAnnotation]
    seconds: float
    nlp_language: str = "en"
    chunk_count: int = 0
    chunked_units: int = 0
    max_chars_per_chunk: int = 0

    def __len__(self) -> int:
        return len(self.annotations)

    @property
    def total_tokens(self) -> int:
        return sum(len(a.tokens) for a in self.annotations)

    def token_views(self) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
        return (
            [a.tokens for a in self.annotations],
            [a.tags for a in self.annotations],
            [a.pos for a in self.annotations],
            [a.lemmas for a in self.annotations],
        )


@dataclass
class ChunkPlan:
    """One chunk of one unit, as handed to spaCy."""

    unit_index: int
    chunk_index: int
    start_char: int
    end_char: int


def split_text(text: str, max_chars: int) -> List[Tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` spans of at most ``max_chars``.

    Splits fall on whitespace so no word is broken. Very long words (long table
    rows without spaces) are hard-split, which costs token-boundary fidelity but
    keeps the model inside its memory limit.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [(0, len(text))]

    spans: List[Tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = text.rfind(" ", start + 1, end)
            if window > start:
                end = window
        spans.append((start, end))
        start = end
    return spans


def _plan_chunks(units: Sequence[Unit], max_chars: int) -> Tuple[List[ChunkPlan], List[str]]:
    plans: List[ChunkPlan] = []
    notes: List[str] = []
    for unit_index, unit in enumerate(units):
        spans = split_text(unit.text, max_chars)
        if len(spans) > 1:
            notes.append(
                f"unit {unit.unit_id} ({unit.unit_type}, {len(unit.text)} chars) was split into "
                f"{len(spans)} chunks of at most {max_chars} chars for the spaCy pass; an entity "
                f"that spans a chunk boundary is not detected"
            )
        for chunk_index, (start, end) in enumerate(spans):
            plans.append(
                ChunkPlan(
                    unit_index=unit_index,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
    return plans, notes


def annotate(
    units: Sequence[Unit],
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
    disable: Optional[Sequence[str]] = None,
) -> SpacyAnnotations:
    """Run one batched spaCy pass over ``units``."""
    model_name = str(config.get("language.spacy_model", "en_core_web_sm"))
    nlp = load_spacy(model_name, logger)
    batch_size = int(config.get("tokenization.spacy.batch_size", 128))
    max_chars = int(config.get("tokenization.spacy.max_chars_per_chunk", 20000))
    units = list(units)
    plans, chunk_notes = _plan_chunks(units, max_chars)
    texts = [
        normalise_for_tokenisation(units[plan.unit_index].text[plan.start_char : plan.end_char])
        for plan in plans
    ]

    start = time.perf_counter()
    pipe_kwargs = {"batch_size": batch_size}
    if disable:
        pipe_kwargs["disable"] = list(disable)
    docs = nlp.pipe(texts, **pipe_kwargs)
    annotations: Dict[int, UnitAnnotation] = {
        index: UnitAnnotation(
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            page_number=unit.page_number,
            unit_type=unit.unit_type,
        )
        for index, unit in enumerate(units)
    }
    for plan, doc in zip(plans, docs):
        annotation = annotations[plan.unit_index]
        offset = len(annotation.tokens)
        annotation.tokens.extend(t.text for t in doc)
        annotation.tags.extend(t.tag_ for t in doc)
        annotation.pos.extend(t.pos_ for t in doc)
        annotation.lemmas.extend(t.lemma_ for t in doc)
        if not annotation.sentences:
            annotation.sentences.append(0)
        annotation.sentences.extend(
            offset + token.i for token in doc if token.is_sent_start and token.i != 0
        )
        annotation.entities.extend(
            (e.text, e.label_, offset + e.start, offset + e.end) for e in doc.ents
        )
    ordered = [annotations[index] for index in range(len(units))]
    elapsed = time.perf_counter() - start

    version = str(getattr(nlp, "meta", {}).get("version", "unknown"))
    if logger is not None:
        for note in chunk_notes:
            log_event(logger, "WARNING", "spacy_pass", note)
        log_event(
            logger,
            "INFO",
            "spacy_pass",
            f"spaCy '{model_name}' ({version}) annotated {len(units)} units / "
            f"{sum(len(a.tokens) for a in ordered)} tokens in {len(plans)} chunks "
            f"(max {max_chars} chars) in {elapsed:.2f}s",
        )
    return SpacyAnnotations(
        model_name=model_name,
        model_version=version,
        units=units,
        annotations=ordered,
        seconds=elapsed,
        nlp_language=getattr(nlp, "lang", "en"),
        chunk_count=len(plans),
        chunked_units=len(chunk_notes),
        max_chars_per_chunk=max_chars,
    )
