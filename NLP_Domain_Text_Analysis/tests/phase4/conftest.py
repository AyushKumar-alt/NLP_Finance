"""Fixtures for the Phase 4 tests.

The API tests read the real Phase 1-4 artefacts, because the point of the
backend is to serve exactly what those phases produced. Only the judgment write
test redirects that file, so the suite never mutates the real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JUDGMENT_FILE = PROJECT_ROOT / "results" / "phase4" / "relevance_judgments.csv"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the 9.9 MB inverted index")


@pytest.fixture(scope="session")
def client():
    """A ``TestClient`` for the real application.

    Session-scoped because the first call loads the Phase 3 runtime, and that
    load is the expensive part; the app caches it for the process lifetime.
    """
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture()
def temp_judgment_file(tmp_path, monkeypatch):
    """Point the judgment CSV at a temporary copy for one test.

    The real ``relevance_judgments.csv`` is copied first, so the test starts from
    the real 139-row dataset. The guard is then *verified*: if the patch did not
    take effect the fixture fails here, before any request is made, so a test can
    never write to the repository copy. The original bound method is captured
    before patching, otherwise the replacement would call itself and recurse.
    """
    from backend.config.settings import get_settings
    from backend.services import artifacts

    target = tmp_path / "relevance_judgments.csv"
    target.write_bytes(JUDGMENT_FILE.read_bytes())

    settings = get_settings()
    original = type(settings).path

    def resolve(self, relative: str) -> Path:
        if relative.endswith("phase4/relevance_judgments.csv"):
            return target
        return original(self, relative)

    monkeypatch.setattr(type(settings), "path", resolve, raising=False)
    artifacts.clear_cache()

    resolved = settings.path("results/phase4/relevance_judgments.csv").resolve()
    assert resolved == target.resolve(), (
        f"the judgment redirect did not take effect: the API would write to "
        f"{resolved}, not the temporary copy"
    )
    assert JUDGMENT_FILE.resolve() not in resolved.parents

    yield target
    artifacts.clear_cache()
