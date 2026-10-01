"""Phase 1 test suite.

Run with::

    python -m pytest tests -q

The tests do not need the real PDF corpus: they build a tiny synthetic PDF in a
temporary folder and assert that every layer of the pipeline behaves as
specified (ids, structure, traceability, cleaning policy, statistics labels).
"""
