"""Configuration loading for Phase 1.

Nothing in this project hard-codes a filesystem path. Every path, threshold and
regex policy is declared in ``config/phase1_config.yaml`` and resolved here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_RELPATH = Path("config") / "phase1_config.yaml"


class Phase1Config:
    """Thin typed wrapper around the YAML configuration."""

    def __init__(self, data: Dict[str, Any], config_path: Path, project_root: Path):
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
    def unknown_label(self) -> str:
        return str(self.get("run.unknown_label", "UNKNOWN"))

    @property
    def id_prefix(self) -> str:
        return str(self.get("run.id_prefix", "D"))

    @property
    def id_padding(self) -> int:
        return int(self.get("run.id_padding", 2))

    @property
    def source_id_prefix(self) -> str:
        return str(self.get("run.source_id_prefix", "SRC"))

    @property
    def source_id_padding(self) -> int:
        return int(self.get("run.source_id_padding", 2))

    # ------------------------------------------------------------------
    # path resolution
    # ------------------------------------------------------------------
    @property
    def input_directory(self) -> Path:
        return Path(self.get("input.directory")).expanduser().resolve()

    @property
    def extensions(self) -> tuple:
        return tuple(self.get("input.supported_extensions", [".pdf"]))

    @property
    def excluded_dirs(self) -> tuple:
        return tuple(self.get("input.exclude_directories", []) or ())

    def out_path(self, key: str) -> Path:
        """Resolve ``output.<key>`` to an absolute path (creating parents)."""
        rel = self.get(f"output.{key}")
        if rel is None:
            raise KeyError(f"output.{key} missing from configuration")
        path = (self.project_root / str(rel)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def results_path(self, artifact_key: str) -> Path:
        """Resolve ``artifacts.<artifact_key>`` inside ``results_dir``."""
        name = self.get(f"artifacts.{artifact_key}")
        if name is None:
            raise KeyError(f"artifacts.{artifact_key} missing from configuration")
        return self.out_path("results_dir") / str(name)

    def metadata_path(self, artifact_key: str) -> Path:
        name = self.get(f"artifacts.{artifact_key}")
        if name is None:
            raise KeyError(f"artifacts.{artifact_key} missing from configuration")
        return self.out_path("metadata_dir") / str(name)


def project_root_from_config(config_path: Path) -> Path:
    """Project root = parent of the ``config`` directory holding the YAML."""
    return config_path.parent.parent.resolve()


def load_config(config_path: str | os.PathLike | None = None) -> Phase1Config:
    """Load Phase 1 configuration.

    Resolution order for ``config_path``:
      1. explicit argument
      2. ``NLP_PHASE1_CONFIG`` environment variable
      3. ``<cwd>/config/phase1_config.yaml``
      4. ``<package>/../../config/phase1_config.yaml``
    """
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    env_value = os.environ.get("NLP_PHASE1_CONFIG")
    if env_value:
        candidates.append(Path(env_value))
    cwd_candidate = Path.cwd() / DEFAULT_CONFIG_RELPATH
    candidates.append(cwd_candidate)
    pkg_candidate = Path(__file__).resolve().parent.parent.parent / DEFAULT_CONFIG_RELPATH
    candidates.append(pkg_candidate)

    chosen: Path | None = None
    for cand in candidates:
        if cand.is_file():
            chosen = cand.resolve()
            break
    if chosen is None:
        raise FileNotFoundError(
            "phase1_config.yaml not found. Looked in: "
            + ", ".join(str(c) for c in candidates)
        )

    with open(chosen, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    root = project_root_from_config(chosen)
    # Environment override keeps CI / notebook usage possible without editing YAML.
    env_input = os.environ.get("NLP_PHASE1_INPUT_DIR")
    if env_input:
        data.setdefault("input", {})["directory"] = env_input

    return Phase1Config(data, chosen, root)
