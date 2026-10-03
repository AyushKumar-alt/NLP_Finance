"""Phase 4: evaluation, relevance judgments, API integration and GUI data feed.

Phase 4 owns no NLP. It evaluates the Phase 3 retrieval subsystem against human
relevance judgments and exposes Phases 1-3 to the frontend through a read-only
Python API.
"""

__all__ = [
    "config",
    "evaluator",
    "metrics",
    "pooling",
    "relevance",
    "reporting",
    "runtime",
    "validation",
]
