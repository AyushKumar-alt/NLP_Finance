"""Step 1-2: recursive discovery + PDF inventory.

Design rules enforced here
--------------------------
* The input directory is READ-ONLY. Files are opened with mode "rb" only.
* Discovery is deterministic: candidates are sorted by
  ``(normalised relative path, filename)`` so document ids never change
  between runs, even if the filesystem enumeration order does.
* No file is silently ignored. Anything that cannot be opened is still
  emitted into the inventory with ``status = ERROR`` plus an error column.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import Phase1Config
from ..core.utils import sha256_file, sort_key_for

INVENTORY_COLUMNS = [
    "file_id",
    "document_id",
    "source_id",
    "source_path",
    "relative_path",
    "filename",
    "extension",
    "file_size_bytes",
    "page_count",
    "parent_folder",
    "detected_source",
    "sha256",
    "is_duplicate",
    "duplicate_of",
    "status",
    "error_message",
]


@dataclass
class DiscoveredFile:
    """One PDF found on disk, before any content is read."""

    file_id: str = ""
    document_id: str = ""
    source_id: str = ""
    source_path: str = ""
    relative_path: str = ""
    filename: str = ""
    extension: str = ""
    file_size_bytes: int = 0
    page_count: int = 0
    parent_folder: str = ""
    detected_source: str = ""
    sha256: str = ""
    is_duplicate: bool = False
    duplicate_of: str = ""
    status: str = "DISCOVERED"
    error_message: str = ""
    pdf_metadata: Dict[str, str] = field(default_factory=dict)
    restored_keys: set = field(default_factory=set)

    def row(self) -> Dict[str, object]:
        return {
            "file_id": self.file_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "filename": self.filename,
            "extension": self.extension,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "parent_folder": self.parent_folder,
            "detected_source": self.detected_source,
            "sha256": self.sha256,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "status": self.status,
            "error_message": self.error_message,
        }


def _is_excluded_dir(name: str, excluded: tuple) -> bool:
    lowered = name.casefold()
    return any(lowered == str(item).casefold() for item in excluded)


def discover_files(config: Phase1Config) -> List[Path]:
    """Recursively find every supported file under the configured input dir."""
    root = config.input_directory
    if not root.exists():
        raise FileNotFoundError(f"Configured input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Configured input path is not a directory: {root}")

    extensions = {ext.casefold() for ext in config.extensions}
    excluded = config.excluded_dirs
    recursive = bool(config.get("input.recursive", True))

    found: List[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not _is_excluded_dir(d, excluded)
            )
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.casefold() in extensions:
                    found.append(path)
    else:
        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix.casefold() in extensions:
                found.append(entry)

    # Deterministic global ordering.
    found.sort(
        key=lambda p: sort_key_for(_relative(root, p), p.name)
    )
    return found


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_inventory(config: Phase1Config, paths: List[Path], hash_bytes: int) -> List[DiscoveredFile]:
    """Create inventory rows with size, sha256 and a first-pass open attempt."""
    root = config.input_directory
    records: List[DiscoveredFile] = []

    for path in paths:
        rel = _relative(root, path)
        record = DiscoveredFile(
            source_path=str(path),
            relative_path=rel,
            filename=path.name,
            extension=path.suffix.casefold(),
            parent_folder=str(Path(rel).parent).replace("\\", "/") or ".",
        )
        try:
            record.file_size_bytes = path.stat().st_size
        except OSError as exc:  # pragma: no cover - rare
            record.status = "ERROR"
            record.error_message = f"stat failed: {exc}"
            records.append(record)
            continue

        if record.file_size_bytes < int(
            config.get("validation.min_valid_file_size_bytes", 512)
        ):
            record.status = "INVALID"
            record.error_message = (
                f"file smaller than minimum valid size "
                f"({record.file_size_bytes} bytes)"
            )
            records.append(record)
            continue

        try:
            record.sha256 = sha256_file(path, hash_bytes)
            record.status = "OK"
        except OSError as exc:
            record.status = "ERROR"
            record.error_message = f"hashing failed: {exc}"

        records.append(record)

    # file_id mirrors the document id so the two artefacts cross-reference
    # without a lookup table.
    for index, record in enumerate(records, start=1):
        record.file_id = f"F{index:03d}"
    return records


def mark_duplicates(records: List[DiscoveredFile]) -> Dict[str, object]:
    """Flag exact (sha256) duplicates. Duplicates are never deleted."""
    seen: Dict[str, str] = {}
    exact_groups: Dict[str, List[str]] = {}
    for record in records:
        if not record.sha256 or record.status in {"ERROR", "INVALID"}:
            continue
        if record.sha256 in seen:
            record.is_duplicate = True
            record.duplicate_of = seen[record.sha256]
            exact_groups.setdefault(record.sha256, [seen[record.sha256]])
            exact_groups[record.sha256].append(record.document_id)
        else:
            seen[record.sha256] = record.document_id

    duplicate_file_ids = [r.document_id for r in records if r.is_duplicate]
    return {
        "exact_duplicate_count": len(duplicate_file_ids),
        "exact_duplicate_groups": exact_groups,
        "duplicate_document_ids": duplicate_file_ids,
    }
