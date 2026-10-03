"""Phase 4 configuration access, logging and output management.

Phase 4 reads Phases 1-3 and writes only to ``results/phase4``. Cleaning is
restricted to an explicit managed-file list, so no Phase 1, Phase 2 or Phase 3
artefact can be deleted by a Phase 4 run.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_RELPATH = Path("config") / "phase4_config.yaml"

#: Files Phase 4 owns. Nothing outside this tuple is ever deleted. The log file
#: is not listed: ``setup_logging`` already truncates it on every run, and it
#: lives in ``logs/`` rather than in the results directory.
MANAGED_OUTPUT_KEYS: tuple = (
    "evaluation_results_csv",
    "evaluation_summary_csv",
    "pipeline_final_comparison_csv",
    "final_summary_csv",
    "final_summary_json",
    "gui_summary_json",
    "validation_report_csv",
    "readme",
)

#: ``relevance_judgments_csv`` is deliberately *not* managed: a Phase 4 rerun
#: must never discard human judgments.

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(step)s | %(message)s"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "step"):
            record.step = "-"
        return True


class Phase4Config:
    """Thin typed wrapper around ``config/phase4_config.yaml``."""

    def __init__(self, data: Dict[str, Any], config_path: Path, project_root: Path) -> None:
        self._data = data
        self.config_path = config_path
        self.project_root = project_root

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

    # ------------------------------------------------------------------
    def project_path(self, value: str) -> Path:
        return (self.project_root / value).resolve()

    def input_path(self, key: str) -> Path:
        rel = self.get(f"input.{key}")
        if rel is None:
            raise KeyError(f"input.{key} missing from configuration")
        return self.project_path(str(rel))

    @property
    def relevance_judgments_path(self) -> Path:
        return self.project_path(str(self.get("relevance.judgments_file")))

    def out_path(self, key: str) -> Path:
        rel = self.get(f"output.{key}")
        if rel is None:
            raise KeyError(f"output.{key} missing from configuration")
        return self.project_path(str(rel))

    @property
    def results_dir(self) -> Path:
        path = self.out_path("results_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def managed_paths(self) -> List[Path]:
        return [self.out_path(key) for key in MANAGED_OUTPUT_KEYS]

    # ------------------------------------------------------------------
    @property
    def pool_depth(self) -> int:
        return int(self.get("relevance.pool_depth", 10))

    @property
    def k_values(self) -> List[int]:
        values = self.get("evaluation.k_values", [5, 10]) or [5, 10]
        return sorted({int(v) for v in values})

    @property
    def unjudged_policy(self) -> str:
        return str(self.get("evaluation.unjudged_policy", "exclude_from_precision_denominator"))


def project_root_from_config(config_path: Path) -> Path:
    return config_path.parent.parent.resolve()


def load_config(config_path: str | os.PathLike | None = None) -> Phase4Config:
    """Load ``config/phase4_config.yaml``.

    Resolution order: explicit argument, ``NLP_PHASE4_CONFIG``, the current
    working directory, then the config packaged beside this file.
    """
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    env_value = os.environ.get("NLP_PHASE4_CONFIG")
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
            "phase4_config.yaml not found. Looked in: " + ", ".join(str(c) for c in candidates)
        )

    with open(chosen, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Phase4Config(data, chosen, project_root_from_config(chosen))


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(config: Phase4Config, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("phase4")
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
def clean_managed_output(config: Phase4Config, logger: Optional[logging.Logger] = None) -> List[str]:
    """Remove only the artefacts Phase 4 owns. Judgments are never removed."""
    results_root = config.results_dir
    removed: List[str] = []
    for path in config.managed_paths():
        try:
            path.relative_to(results_root)
        except ValueError:
            raise RuntimeError(f"refusing to clean {path}: outside the Phase 4 results directory")
        if path.is_file():
            _unlink_with_retry(path)
            removed.append(str(path))
    if logger is not None:
        log_event(logger, "INFO", "setup", f"cleaned {len(removed)} Phase 4 output paths")
    return removed


def _unlink_with_retry(path: Path, attempts: int = 5, delay: float = 0.4) -> None:
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
