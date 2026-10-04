"""Pipeline definitions for Phase 3.

A pipeline is an ordered list of components applied to every selected content
unit. Two pipelines are defined:

* **Pipeline A** - domain-aware stopword removal *before* lemmatization.
* **Pipeline B** - stemming *before* stopword removal.

Both use the same tokenizer and the same date/number handling so the comparison
isolates the effect of processing order and morphology, which is exactly what
Phase 2's order experiment flagged as risky.

The ordering is data, not code: it is read from ``config/phase3_config.yaml``
and executed by :mod:`src.phase3.pipeline_runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

#: Component names understood by the runner, in the order they can appear.
SUPPORTED_STEPS: Tuple[str, ...] = (
    "tokenize",
    "date_number",
    "stopwords",
    "morphology",
    "pos",
    "ner",
    "ngrams",
)


@dataclass(frozen=True)
class PipelineSpec:
    """Declarative description of one NLP pipeline."""

    key: str
    name: str
    tokenizer: str
    date_number_strategy: str
    stopword_strategy: str
    morphology: str
    morphology_algorithm: str
    order: Tuple[str, ...]
    pos_strategy: str
    ner_strategy: str
    ngram_max: int
    index_representation: str
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def morphology_label(self) -> str:
        if self.morphology == "stem":
            return f"stemming ({self.morphology_algorithm})"
        if self.morphology == "lemmatize":
            return f"lemmatization ({self.morphology_algorithm})"
        if self.morphology == "none":
            return "none"
        return str(self.morphology)

    @property
    def stopword_label(self) -> str:
        return {
            "domain_aware": "domain-aware (protected financial terms kept)",
            "standard_english": "standard English list",
            "none": "none",
        }.get(self.stopword_strategy, str(self.stopword_strategy))

    @property
    def date_number_label(self) -> str:
        return {
            "typed_protect": "typed date/number detection with expression protection",
            "none": "none",
        }.get(self.date_number_strategy, str(self.date_number_strategy))

    @property
    def term_kind(self) -> str:
        """``lemma`` or ``stem`` - used by the comparison table."""
        return "stem" if self.morphology == "stem" else "lemma"

    def step_labels(self) -> List[Dict[str, Any]]:
        label = {
            "tokenize": f"tokenization ({self.tokenizer})",
            "date_number": f"date/number handling ({self.date_number_label})",
            "stopwords": f"stopword handling ({self.stopword_label})",
            "morphology": f"morphology ({self.morphology_label})",
            "pos": f"POS tagging metadata ({self.pos_strategy})",
            "ner": f"NER metadata ({self.ner_strategy})",
            "ngrams": f"n-gram generation (1..{self.ngram_max})",
        }
        return [{"order": i + 1, "step": step, "component": label.get(step, step)}
                for i, step in enumerate(self.order)]

    def description(self) -> str:
        return " -> ".join(self.step_labels() and [s["component"] for s in self.step_labels()])


def _spec_from_config(key: str, section: Dict[str, Any]) -> PipelineSpec:
    order = tuple(str(step) for step in section.get("order", SUPPORTED_STEPS))
    unknown = [step for step in order if step not in SUPPORTED_STEPS]
    if unknown:
        raise ValueError(f"pipeline '{key}' has unsupported steps {unknown}; allowed: {SUPPORTED_STEPS}")
    return PipelineSpec(
        key=key,
        name=str(section.get("name", key)),
        tokenizer=str(section.get("tokenizer", "hybrid")),
        date_number_strategy=str(section.get("date_number_strategy", "typed_protect")),
        stopword_strategy=str(section.get("stopword_strategy", "domain_aware")),
        morphology=str(section.get("morphology", "lemmatize")),
        morphology_algorithm=str(section.get("morphology_algorithm", "")),
        order=order,
        pos_strategy=str(section.get("pos_strategy", "phase2_evidence")),
        ner_strategy=str(section.get("ner_strategy", "phase2_consumed")),
        ngram_max=int(section.get("ngram_max", 5)),
        index_representation=str(section.get("index_representation", "")),
        notes=str(section.get("notes", "")).strip(),
        config=dict(section),
    )


def load_pipeline_specs(config: Any) -> Dict[str, PipelineSpec]:
    """Read pipeline_a and pipeline_b (and pipeline_a_lemma / pipeline_b_stem) from configuration."""
    specs: Dict[str, PipelineSpec] = {}
    for key in ("pipeline_a", "pipeline_b", "pipeline_a_lemma", "pipeline_b_stem"):
        section = config.section(key)
        if section:
            specs[key] = _spec_from_config(key, section)
    if "pipeline_a" not in specs or "pipeline_b" not in specs:
        raise KeyError("pipeline_a and pipeline_b must be defined in config/phase3_config.yaml")
    if specs["pipeline_a"].morphology == specs["pipeline_b"].morphology and specs["pipeline_a"].morphology_algorithm == specs["pipeline_b"].morphology_algorithm and specs["pipeline_a"].order == specs["pipeline_b"].order:
        raise ValueError(
            "pipeline_a and pipeline_b are identical; the comparison would not test anything."
        )
    return specs
