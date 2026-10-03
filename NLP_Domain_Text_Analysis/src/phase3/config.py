"""Phase 3 configuration access, logging and output management.

Phase 3 consumes Phase 1 (structured corpus) and Phase 2 (NLP results) and
writes exclusively to ``results/phase3``. The Phase 2 configuration object is
reused rather than duplicated so that stopword lists, tokenizer settings and
the text selection policy are guaranteed to be the ones the experiments
actually used.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.phase2.config import Phase2Config, load_config as load_phase2_config

DEFAULT_CONFIG_RELPATH = Path("config") / "phase3_config.yaml"

#: Files Phase 3 owns. Cleaning is restricted to these so no Phase 1 or Phase 2
#: artefact can be deleted by a Phase 3 run.
MANAGED_OUTPUT_KEYS: tuple = (
    "pipeline_comparison_csv",
    "pipeline_comparison_md",
    "final_pipeline_json",
    "final_pipeline_md",
    "inverted_index_json",
    "index_statistics_csv",
    "index_manifest_json",
    "query_registry_csv",
    "retrieval_results_csv",
    "retrieval_summary_csv",
    "document_results_csv",
    "validation_report_csv",
    "summary_json",
    "readme",
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(step)s | %(message)s"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "step"):
            record.step = "-"
        return True


class Phase3Config:
    """Thin typed wrapper around ``config/phase3_config.yaml``."""

    def __init__(self, data: Dict[str, Any], config_path: Path, project_root: Path) -> None:
        self._data = data
        self.config_path = config_path
        self.project_root = project_root

    # ------------------------------------------------------------------
    # generic access
    # ------------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    @property
    def encoding(self) -> str:
        return str(self.get("run.encoding", "utf-8"))

    @property
    def random_seed(self) -> int:
        return int(self.get("reproducibility.random_seed", 42))

    @property
    def active_policy(self) -> str:
        return str(self.get("text_selection.policy", "prose_tables"))

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    def project_path(self, value: str) -> Path:
        return (self.project_root / value).resolve()

    def input_path(self, key: str) -> Path:
        rel = self.get(f"input.{key}")
        if rel is None:
            raise KeyError(f"input.{key} missing from configuration")
        return self.project_path(str(rel))

    def out_path(self, key: str) -> Path:
        rel = self.get(f"output.{key}")
        if rel is None:
            raise KeyError(f"output.{key} missing from configuration")
        return self.project_path(str(rel))

    def out_dir(self, key: str) -> Path:
        path = self.out_path(key)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_dir(self) -> Path:
        path = self.out_path("results_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def corpus_jsonl(self) -> Path:
        return self.input_path("corpus_jsonl")

    @property
    def phase2_results_dir(self) -> Path:
        return self.input_path("phase2_results_dir")

    def managed_paths(self) -> List[Path]:
        paths = [self.out_path(key) for key in MANAGED_OUTPUT_KEYS]
        paths.append(self.out_path("query_examples_dir"))
        return paths

    # ------------------------------------------------------------------
    # Phase 2 bridge
    # ------------------------------------------------------------------
    def phase2_config(self) -> Phase2Config:
        """The Phase 2 configuration object, loaded from its own file."""
        rel = self.get("input.phase2_config", "config/phase2_config.yaml")
        path = self.project_path(str(rel))
        if not path.is_file():
            raise FileNotFoundError(f"Phase 2 configuration not found: {path}")
        return load_phase2_config(path)


def project_root_from_config(config_path: Path) -> Path:
    return config_path.parent.parent.resolve()


def load_config(config_path: str | os.PathLike | None = None) -> Phase3Config:
    """Load ``config/phase3_config.yaml``.

    Resolution order: explicit argument, ``NLP_PHASE3_CONFIG`` environment
    variable, ``<cwd>/config/phase3_config.yaml``, then the packaged config.
    """
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    env_value = os.environ.get("NLP_PHASE3_CONFIG")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_RELPATH)
    candidates.append(Path(__file__).resolve().parent.parent.parent / DEFAULT_CONFIG_RELPATH)

    chosen: Optional[Path] = None
    for candidate in candidates:
        if candidate.is_file():
            chosen = candidate.resolve()
            break
    if chosen is None:
        raise FileNotFoundError(
            "phase3_config.yaml not found. Looked in: " + ", ".join(str(c) for c in candidates)
        )

    with open(chosen, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Phase3Config(data, chosen, project_root_from_config(chosen))


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(config: Phase3Config, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("phase3")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_file = config.out_path("log_file")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler.addFilter(_ContextFilter())
    logger.addHandler(file_handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(_ContextFilter())
    logger.addHandler(console)
    return logger


def log_event(logger: logging.Logger, level: str, step: str, message: str) -> None:
    logger.log(getattr(logging, level.upper(), logging.INFO), message, extra={"step": step})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
def _unlink_with_retry(path: Path, attempts: int = 5, delay: float = 0.4) -> None:
    """Delete a file, retrying the transient Windows sharing violations.

    An editor, a search indexer or an antivirus scanner can hold an output file
    open for a moment. Retrying keeps a rerun from failing on a lock the run
    itself did not create; a genuinely locked file still raises.
    """
    import time

    last: OSError | None = None
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except PermissionError as error:      # pragma: no cover - platform dependent
            last = error
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
    raise last if last is not None else OSError(f"could not delete {path}")


def clean_managed_output(config: Phase3Config, logger: Optional[logging.Logger] = None) -> List[str]:
    """Remove only the artefacts Phase 3 owns."""
    import shutil

    results_root = config.results_dir
    removed: List[str] = []
    for path in config.managed_paths():
        try:
            path.relative_to(results_root)
        except ValueError:
            raise RuntimeError(
                f"refusing to clean {path}: outside the Phase 3 results directory"
            ) from None
        if path.is_file():
            _unlink_with_retry(path)
            removed.append(str(path))
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
            removed.append(str(path))
    if logger is not None:
        log_event(logger, "INFO", "setup", f"cleaned {len(removed)} Phase 3 output paths")
    return removed
