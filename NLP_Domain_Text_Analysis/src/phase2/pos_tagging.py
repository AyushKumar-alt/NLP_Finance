"""Default POS tagging experiment (spaCy ``en_core_web_sm``).

Produces the tag distribution, coarse category counts and fully traceable
tagged examples. The same tagger output is reused by the custom rule and ML
experiments so the three methods are compared on identical tokens.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Phase2Config, log_event
from .spacy_pipeline import SpacyAnnotations
from .tokenizers import TOKENIZER_DESCRIPTIONS

#: Coarse category per universal POS, used for the report-level distribution.
COARSE_OF_UPOS = {
    "NOUN": "noun",
    "PROPN": "proper_noun",
    "VERB": "verb",
    "ADJ": "adjective",
    "ADV": "adverb",
    "NUM": "number",
    "SYM": "symbol",
    "PUNCT": "punctuation",
    "ADP": "adposition",
    "AUX": "auxiliary",
    "CCONJ": "coordinator",
    "SCONJ": "subordinating_conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "PART": "particle",
    "PRON": "pronoun",
    "X": "other",
    "SPACE": "whitespace",
}


def run_default_pos(
    annotations: SpacyAnnotations,
    config: Phase2Config,
    logger: Optional[logging.Logger] = None,
    example_sentences: int = 40,
) -> Dict[str, object]:
    """Tag distribution + coarse counts + traceable examples."""
    tag_counter: Counter = Counter()
    coarse_counter: Counter = Counter()
    tag_documents: Dict[str, set] = defaultdict(set)
    total_tokens = 0
    sentences = 0

    for annotation in annotations.annotations:
        sentences += annotation.sentence_count
        for tag, pos in zip(annotation.tags, annotation.pos):
            total_tokens += 1
            tag_counter[tag] += 1
            tag_documents[tag].add(annotation.document_id)
            coarse_counter[COARSE_OF_UPOS.get(pos, "other")] += 1

    distribution_rows: List[Dict[str, object]] = [
        {
            "tag": tag,
            "coarse_category": COARSE_OF_UPOS.get(_upos_of(tag), "other"),
            "tag_description": _TAG_DESCRIPTIONS.get(tag, ""),
            "count": count,
            "percent_of_tokens": round(100.0 * count / total_tokens, 4) if total_tokens else 0.0,
            "documents_containing_tag": len(tag_documents.get(tag, ())),
        }
        for tag, count in tag_counter.most_common()
    ]

    coarse_rows: List[Dict[str, object]] = [
        {
            "coarse_category": category,
            "count": count,
            "percent_of_tokens": round(100.0 * count / total_tokens, 4) if total_tokens else 0.0,
        }
        for category, count in coarse_counter.most_common()
    ]

    example_rows = tagged_examples(annotations, example_sentences)

    if logger is not None:
        log_event(
            logger,
            "INFO",
            "pos",
            f"default POS: {total_tokens} tokens tagged, {len(tag_counter)} distinct tags, "
            f"top tag '{distribution_rows[0]['tag']}' ({distribution_rows[0]['percent_of_tokens']}%)"
            if distribution_rows
            else "default POS: no tokens",
        )

    return {
        "distribution": distribution_rows,
        "coarse": coarse_rows,
        "examples": example_rows,
        "tag_counter": tag_counter,
        "total_tokens": total_tokens,
        "sentences": sentences,
    }


def tagged_examples(annotations: SpacyAnnotations, limit: int) -> List[Dict[str, object]]:
    """Sentences with their default tags, spread deterministically across documents."""
    by_document: Dict[str, List[int]] = defaultdict(list)
    for index, annotation in enumerate(annotations.annotations):
        if 6 <= len(annotation.tokens) <= 45:
            by_document[annotation.document_id].append(index)

    rows: List[Dict[str, object]] = []
    for document_id in sorted(by_document):
        indexes = by_document[document_id]
        if not indexes:
            continue
        # median-length sentence of the document keeps examples typical
        ordered = sorted(indexes, key=lambda i: (len(annotations.annotations[i].tokens), i))
        chosen = ordered[len(ordered) // 2]
        annotation = annotations.annotations[chosen]
        unit = annotations.units[chosen]
        rows.append(
            {
                "document_id": unit.document_id,
                "source_id": unit.source_id,
                "page_number": unit.page_number,
                "unit_id": unit.unit_id,
                "section_id": unit.section_id,
                "section_number": unit.section_number,
                "section_title": unit.section_title,
                "unit_type": unit.unit_type,
                "text": " ".join(annotation.tokens),
                "tagged_tokens": [f"{t}/{tag}" for t, tag in zip(annotation.tokens, annotation.tags)],
                "pos_sequence": " ".join(annotation.pos),
                "token_count": len(annotation.tokens),
            }
        )
        if len(rows) >= limit:
            break
    return rows


# ----------------------------------------------------------------------
# Penn Treebank tag reference (Penn tag -> universal POS, spaCy style)
# ----------------------------------------------------------------------
def _upos_of(tag: str) -> str:
    if tag in _TAG_TO_UPOS:
        return _TAG_TO_UPOS[tag]
    if tag.startswith("NNP"):
        return "PROPN"
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"
    if tag == "CD":
        return "NUM"
    return "X"


_TAG_TO_UPOS = {
    "IN": "ADP",
    "DT": "DET",
    "CC": "CCONJ",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "PRON",
    "PRP": "PRON",
    "PRP$": "PRON",
    "TO": "PART",
    "MD": "AUX",
    "POS": "PART",
    "UH": "INTJ",
    "SYM": "SYM",
    "EX": "PRON",
    "FW": "X",
    "RP": "PART",
    "PDT": "DET",
    ". ": "PUNCT",
}

_TAG_DESCRIPTIONS = {
    "ADJ": "adjective",
    "ADP": "adposition (preposition/subordinating conjunction)",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "singular or mass noun",
    "NNOUN": "plural noun (spaCy plural)",
    "NUM": "cardinal number",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other / unusable token",
}


def sentence_level_tag_text(unit_tokens: Sequence[str], tags: Sequence[str]) -> str:
    return " ".join(f"{token}/{tag}" for token, tag in zip(unit_tokens, tags))
