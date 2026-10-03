"""Cached, read-only access to the project's CSV/JSON artefacts.

The API never recomputes an experiment and never rewrites a Phase 1-3 file. It
reads the artefacts those phases wrote, caches the parse, and hands plain Python
structures to the route handlers.

Large tables are never returned whole: the helpers here always take a limit.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}


class ArtifactMissing(FileNotFoundError):
    """A declared artefact is absent, so the API must say so rather than guess."""


def _cache_key(path: Path, kind: str) -> str:
    return f"{kind}:{path}"


def read_csv_rows(path: Path, required: bool = True) -> List[Dict[str, str]]:
    """All rows of a CSV, cached. Returns ``[]`` when absent unless required."""
    if not path.is_file():
        if required:
            raise ArtifactMissing(str(path))
        return []
    key = _cache_key(path, "csv")
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with _LOCK:
        _CACHE[key] = rows
    return rows


def read_csv_header(path: Path) -> List[str]:
    if not path.is_file():
        raise ArtifactMissing(str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            return row
    return []


def read_json(path: Path, required: bool = True, default: Any = None) -> Any:
    if not path.is_file():
        if required:
            raise ArtifactMissing(str(path))
        return default
    key = _cache_key(path, "json")
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    with _LOCK:
        _CACHE[key] = payload
    return payload


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        raise ArtifactMissing(str(path))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_text(path: Path, required: bool = True, default: str = "") -> str:
    if not path.is_file():
        if required:
            raise ArtifactMissing(str(path))
        return default
    key = _cache_key(path, "text")
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    text = path.read_text(encoding="utf-8", errors="replace")
    with _LOCK:
        _CACHE[key] = text
    return text


def clear_cache() -> None:
    """Drop every cached artefact. Used after a judgment is written."""
    with _LOCK:
        _CACHE.clear()


# ----------------------------------------------------------------------
# Coercion helpers: Phase artefacts store numbers as strings in CSV and as
# real numbers in JSON, so responses must not leak that difference.
# ----------------------------------------------------------------------
def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def split_list(value: Any, separator: str = ";") -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [item for item in str(value).split(separator) if item]


def page(rows: Sequence[Dict[str, Any]], limit: int, offset: int) -> Dict[str, Any]:
    """Slice ``rows`` and report the totals the UI needs for pagination."""
    total = len(rows)
    start = max(0, offset)
    end = total if limit <= 0 else min(total, start + limit)
    return {
        "items": list(rows[start:end]),
        "total": total,
        "offset": start,
        "limit": limit,
        "returned": max(0, end - start),
        "has_more": end < total,
    }
