"""Phase 2 configuration access and logging.

Phase 2 is a *consumer* of Phase 1. It reads ``config/phase2_config.yaml`` for
its own policies and output locations and never rewrites Phase 1 artefacts.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_RELPATH = Path("config") / "phase2_config.yaml"

#: Directories managed exclusively by Phase 2. Cleaning is restricted to them so
#: that no Phase 1 artefact can ever be deleted by a Phase 2 run.
MANAGED_RESULT_DIRS = (
    "tokenization_dir",
    "preprocessing_dir",
    "stemming_dir",
    "lemmatization_dir",
    "pos_dir",
    "ner_dir",
    "ngrams_dir",
    "bpe_dir",
    "comparisons_dir",
    "summaries_dir",
    "figures_dir",
)


class Phase2Config:
    """Thin typed wrapper around ``config/phase2_config.yaml``."""

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

    # ------------------------------------------------------------------
    # frequently used values
    # ------------------------------------------------------------------
    @property
    def encoding(self) -> str:
        return str(self.get("run.encoding", "utf-8"))

    @property
    def random_seed(self) -> int:
        return int(self.get("reproducibility.random_seed", 42))

    @property
    def active_policy(self) -> str:
        return str(self.get("text_selection.active_policy", "prose_tables"))

    def policy_unit_types(self, policy: Optional[str] = None) -> List[str]:
        """Unit types selected by ``policy`` (default: the active policy)."""
        name = policy or self.active_policy
        policies = self.section("text_selection").get("policies", {})
        if name not in policies:
            raise KeyError(f"text_selection.policies.{name} missing from configuration")
        return list(policies[name])

    # ------------------------------------------------------------------
    # path resolution
    # ------------------------------------------------------------------
    def out_path(self, key: str) -> Path:
        """Resolve ``output.<key>`` to an absolute path (parents created)."""
        rel = self.get(f"output.{key}")
        if rel is None:
            raise KeyError(f"output.{key} missing from configuration")
        path = (self.project_root / str(rel)).resolve()
        return path

    def out_dir(self, key: str) -> Path:
        path = self.out_path(key)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def corpus_jsonl(self) -> Path:
        return self.input_path("corpus_jsonl")

    def input_path(self, key: str) -> Path:
        rel = self.get(f"input.{key}")
        if rel is None:
            raise KeyError(f"input.{key} missing from configuration")
        return (self.project_root / str(rel)).resolve()

    def input_dir(self, key: str) -> Path:
        return self.input_path(key)

    def managed_dirs(self) -> List[Path]:
        return [self.out_path(key) for key in MANAGED_RESULT_DIRS]


def project_root_from_config(config_path: Path) -> Path:
    return config_path.parent.parent.resolve()


def load_config(config_path: str | os.PathLike | None = None) -> Phase2Config:
    """Load the Phase 2 configuration.

    Resolution order: explicit argument, ``NLP_PHASE2_CONFIG`` env var,
    ``<cwd>/config/phase2_config.yaml``, then the packaged config directory.
    """
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    env_value = os.environ.get("NLP_PHASE2_CONFIG")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_RELPATH)
    candidates.append(Path(__file__).resolve().parent.parent.parent / DEFAULT_CONFIG_RELPATH)

    chosen: Optional[Path] = None
    for cand in candidates:
        if cand.is_file():
            chosen = cand.resolve()
            break
    if chosen is None:
        raise FileNotFoundError(
            "phase2_config.yaml not found. Looked in: " + ", ".join(str(c) for c in candidates)
        )

    with open(chosen, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return Phase2Config(data, chosen, project_root_from_config(chosen))


# ----------------------------------------------------------------------
# Logging (Phase 2 gets its own logger name so Phase 1 logging is untouched)
# ----------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(experiment)s | %(message)s"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "experiment"):
            record.experiment = "-"
        return True


def setup_logging(config: Phase2Config, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("phase2")
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


def log_event(logger: logging.Logger, level: str, experiment: str, message: str) -> None:
    logger.log(getattr(logging, level.upper(), logging.INFO), message, extra={"experiment": experiment})


def seed_everything(seed: int) -> None:
    """Fix every RNG that Phase 2 touches so ML experiments are reproducible."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # numpy is a scikit-learn dependency, so this always succeeds in practice
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - defensive
        pass


def clean_managed_output(config: Phase2Config, logger: Optional[logging.Logger] = None) -> List[str]:
    """Delete the directories Phase 2 owns so no stale artefact survives a run.

    Safety: only paths declared in ``output.<MANAGED_RESULT_DIRS>`` are removed and
    each path must live under ``results/phase2`` (or ``data/phase2/annotations``).
    """
    import shutil

    results_root = config.out_path("results_dir")
    annotations_root = config.out_path("annotations_dir")
    allowed_roots = [results_root, annotations_root]
    removed: List[str] = []
    for path in config.managed_dirs():
        if not any(_is_within(path, root) for root in allowed_roots):
            raise RuntimeError(f"refusing to clean {path}: outside the Phase 2 managed roots")
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    for key in ("summary_csv", "summary_json", "validation_report", "readme", "methodology"):
        target = config.out_path(key)
        if target.is_file():
            target.unlink()
            removed.append(str(target))
    if logger is not None:
        log_event(logger, "INFO", "setup", f"cleaned {len(removed)} Phase 2 output paths")
    return removed


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
