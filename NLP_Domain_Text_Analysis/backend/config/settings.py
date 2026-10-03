"""Backend settings.

Every path is derived from the project root; the client never supplies one.
CORS origins are explicit and read from the environment so a deployment can
narrow them without a code change.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Identifiers the API accepts: document ids, unit ids, index terms, source ids.
#: Deliberately excludes '/', '\\' and '.' so no request can name a path.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_:.+-]{0,119}$"


class Settings(BaseSettings):
    """Environment-driven configuration for the API process."""

    model_config = SettingsConfigDict(env_prefix="NLP_API_", extra="ignore")

    project_root: Path = PROJECT_ROOT

    # --- server -------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

    # --- CORS ---------------------------------------------------------
    # Default is the Next.js dev server. A wildcard is refused unless it is set
    # explicitly, because a wildcard lets any page read the API from a browser.
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
    allow_credentialed_cors: bool = False

    # --- data locations (relative to project_root) --------------------
    phase1_results: str = "results/phase1"
    phase2_results: str = "results/phase2"
    phase3_results: str = "results/phase3"
    phase4_results: str = "results/phase4"
    corpus_dir: str = "data/corpus"
    phase3_config: str = "config/phase3_config.yaml"
    phase4_config: str = "config/phase4_config.yaml"

    # --- limits -------------------------------------------------------
    # Server-side pagination is mandatory: the corpus, the inverted index and
    # the NER table are far too large to send to a browser.
    max_page_size: int = 500
    default_page_size: int = 50
    max_ner_rows: int = 200
    max_ngram_rows: int = 300
    max_index_postings: int = 200
    max_top_k: int = 200
    default_top_k: int = 10

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def path(self, relative: str) -> Path:
        """Resolve a project-relative path. Never accepts client input."""
        return (self.project_root / relative).resolve()

    @property
    def is_wildcard_cors(self) -> bool:
        return "*" in self.cors_origins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings so every request shares one resolved configuration."""
    override = os.environ.get("NLP_PROJECT_ROOT")
    if override:
        return Settings(project_root=Path(override).resolve())
    return Settings()
