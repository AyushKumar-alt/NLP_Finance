"""Shared helpers: logging, deterministic ids, hashing, CSV/JSONL writers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(document_id)s | %(operation)s | %(message)s"


class _ContextFilter(logging.Filter):
    """Adds ``document_id`` / ``operation`` so every record carries context."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "document_id"):
            record.document_id = "-"
        if not hasattr(record, "operation"):
            record.operation = "-"
        return True


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure the single Phase 1 log file + console stream."""
    logger = logging.getLogger("phase1")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler.addFilter(_ContextFilter())

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    console.addFilter(_ContextFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def log_event(
    logger: logging.Logger,
    level: str,
    document_id: str,
    operation: str,
    message: str,
) -> None:
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        message,
        extra={"document_id": document_id, "operation": operation},
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# Deterministic identifiers
# ----------------------------------------------------------------------
def normalise_relpath(rel_path: str) -> str:
    """Lower-cased, forward-slash relative path used for deterministic ordering."""
    return str(rel_path).replace("\\", "/").strip().lstrip("./").casefold()


def sort_key_for(rel_path: str, filename: str) -> tuple:
    """Deterministic ordering key: (normalised relative path, filename)."""
    return (normalise_relpath(rel_path), str(filename).casefold())


def format_id(prefix: str, number: int, padding: int) -> str:
    return f"{prefix}{number:0{padding}d}"


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------
def sha256_file(path: Path, chunk_bytes: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ----------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------
PRIVATE_USE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")
_WS = re.compile(r"[ \t\u00a0\u2007\u202f]+")


def strip_private_use(text: str) -> str:
    """Replace publisher symbol-font glyphs (Wingdings spacing) with a space."""
    return PRIVATE_USE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text)


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalise_key(text: str) -> str:
    """Documented baseline normalisation: NFKC + casefold. No stemming."""
    return nfkc(strip_private_use(text)).casefold().strip()


def is_rotation_private(text: str) -> bool:
    """Detect the reversed glyph soup produced by rotated chart text."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 6:
        return False
    lower = sum(1 for c in letters if c.islower())
    return lower / len(letters) > 0.9


def count_words(text: str) -> int:
    """Whitespace word count - deliberately dumb, structure preserving."""
    return len([tok for tok in _WS.split(text.strip()) if tok])


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------
def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
