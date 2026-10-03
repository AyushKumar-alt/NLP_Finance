"""Phase 3 - pipeline comparison, inverted index and retrieval.

Phase 3 consumes the Phase 1 structured corpus and the Phase 2 NLP results. It
runs two preprocessing pipelines over the same text, scores them on four declared
criteria, builds an inverted index from the winner and answers keyword, phrase
and Boolean queries with full traceability back to the Phase 1 content units.

Entry point: ``python -m src.phase3.run``
"""

__all__ = [
    "boolean_search",
    "config",
    "index_builder",
    "inverted_index",
    "keyword_search",
    "load_phase2",
    "pipeline_comparison",
    "pipeline_runner",
    "pipelines",
    "posting_list",
    "phrase_search",
    "query_parser",
    "query_registry",
    "result_formatter",
    "retrieval",
    "reporting",
    "statistics",
    "validation",
]

__version__ = "3.0.0"
