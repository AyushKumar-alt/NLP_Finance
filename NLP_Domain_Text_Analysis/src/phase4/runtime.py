"""Engine factory shared by the Phase 4 CLI and the FastAPI backend.

Loading the Phase 3 index costs ~1.2 s, so it is done once and reused. The
factory deliberately builds on ``src.phase3`` rather than re-reading the index
JSON by hand, so the API and the Phase 3 CLI cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.phase2.load_corpus import Corpus, Unit, load_corpus
from src.phase3.config import Phase3Config, load_config as load_phase3_config
from src.phase3.index_builder import IndexBuilder
from src.phase3.inverted_index import InvertedIndex
from src.phase3.pipeline_runner import PipelineRunner
from src.phase3.pipelines import PipelineSpec, load_pipeline_specs
from src.phase3.retrieval import RetrievalEngine, build_engine


@dataclass
class Phase3Runtime:
    """Everything needed to answer a query, loaded from existing artefacts."""

    config: Phase3Config
    corpus: Corpus
    index: InvertedIndex
    runner: PipelineRunner
    specs: Dict[str, PipelineSpec]
    final_pipeline: str
    load_seconds: float = 0.0
    logger: Optional[logging.Logger] = None
    _engine_cache: Dict[str, RetrievalEngine] = field(default_factory=dict)

    @property
    def spec(self) -> PipelineSpec:
        return self.specs[self.final_pipeline]

    def engine(self, pipeline_key: Optional[str] = None) -> RetrievalEngine:
        key = pipeline_key or self.final_pipeline
        if key not in self.specs:
            raise KeyError(f"unknown pipeline '{key}'; known: {sorted(self.specs)}")

        if key in self._engine_cache:
            return self._engine_cache[key]

        spec = self.specs[key]
        # The persisted inverted_index.json belongs exclusively to Pipeline B / pipeline_b_stem.
        if key in ("pipeline_b", "pipeline_b_stem") and (
            self.final_pipeline in ("pipeline_b", "pipeline_b_stem") or self.index.term_count == 21403
        ):
            idx = self.index
        else:
            builder = IndexBuilder(self.config, self.runner)
            pipe_res = self.runner.run(spec, self.corpus.selected_units())
            build_res = builder.build(pipe_res, self.corpus)
            idx = build_res.index

        eng = build_engine(idx, spec, self.runner, self.config, self.corpus)
        self._engine_cache[key] = eng
        return eng

    def unit(self, unit_id: str) -> Optional[Unit]:
        for unit in self.corpus.units:
            if unit.unit_id == unit_id:
                return unit
        return None


def load_phase3_runtime(
    config_path: Optional[str] = None,
    pipeline_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Phase3Runtime:
    """Load the Phase 1 corpus, the Phase 3 index and the selected pipeline.

    Raises ``FileNotFoundError`` with the missing path when Phase 1 or Phase 3
    has not been run, so the API can answer with a clear 503 instead of a
    fabricated empty result.
    """
    import time

    started = time.perf_counter()
    config = load_phase3_config(config_path)

    index_path = config.out_path("inverted_index_json")
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Phase 3 inverted index not found: {index_path}\n"
            "Run Phase 3 first:  python -m src.phase3.run"
        )

    corpus = load_corpus(config.phase2_config(), logger=logger)
    index = InvertedIndex.load(index_path)
    runner = PipelineRunner(config, logger=logger)
    specs = load_pipeline_specs(config)

    chosen = pipeline_key or str(config.get("selection.final_pipeline") or "")
    if not chosen:
        final_path = config.out_path("final_pipeline_json")
        if final_path.is_file():
            import json

            with final_path.open("r", encoding="utf-8") as handle:
                chosen = str(json.load(handle).get("final_pipeline", ""))
    if chosen not in specs:
        chosen = "pipeline_a_lemma" if "pipeline_a_lemma" in specs else ("pipeline_a" if "pipeline_a" in specs else sorted(specs)[0])

    runtime = Phase3Runtime(
        config=config,
        corpus=corpus,
        index=index,
        runner=runner,
        specs=specs,
        final_pipeline=chosen,
        load_seconds=round(time.perf_counter() - started, 3),
        logger=logger,
    )
    if logger is not None:
        from src.phase3.config import log_event

        log_event(
            logger,
            "INFO",
            "phase3_runtime",
            f"loaded in {runtime.load_seconds}s: {len(corpus.units)} units, "
            f"{index.term_count} index terms, {index.posting_count} postings, "
            f"selected pipeline '{chosen}'",
        )
    return runtime
