"""The judgment store: validation, upsert, persistence and the pooling boundary."""

from __future__ import annotations

import pytest

from src.phase4.relevance import (
    FIELDNAMES,
    RELEVANT,
    RelevanceStore,
    RelevanceValidationError,
)


def _row(**overrides):
    row = {
        "query_id": "Q01",
        "query": "inflation",
        "unit_id": "D01_P002_SEC_1_1_PAR001",
        "rank": "1",
        "document_id": "D01",
        "page_number": "2",
        "section_number": "1",
        "relevance": "1",
        "notes": "discusses CPI inflation directly",
        "annotator": "author",
        "judgment_method": "pooled manual judgment",
        "judged_at": "2026-01-01T00:00:00",
    }
    row.update(overrides)
    return row


class TestFieldnames:
    def test_schema_is_the_one_the_phase_wrote(self):
        assert FIELDNAMES == (
            "query_id", "query", "unit_id", "rank", "document_id", "page_number",
            "section_number", "relevance", "notes", "annotator",
            "judgment_method", "judged_at",
        )


class TestUpsert:
    def test_records_a_new_judgment(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        judgment = store.upsert("Q01", "D01_A", 1, notes="direct", annotator="author")
        assert judgment.relevance == RELEVANT
        assert store.label("Q01", "D01_A") == 1
        assert len(store) == 1

    def test_replacing_a_judgment_does_not_duplicate_it(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        store.upsert("Q01", "D01_A", 1, annotator="author")
        store.upsert("Q01", "D01_A", 0, notes="re-read: only a table cell")
        assert len(store) == 1
        assert store.label("Q01", "D01_A") == 0
        assert store.get("Q01", "D01_A").notes == "re-read: only a table cell"

    def test_replacing_keeps_provenance_the_caller_omitted(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        store.upsert("Q01", "D01_A", 1, annotator="author", judgment_method="manual")
        store.upsert("Q01", "D01_A", 0)
        judgment = store.get("Q01", "D01_A")
        assert judgment.annotator == "author"
        assert judgment.judgment_method == "manual"

    def test_upsert_carries_provenance_columns(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        store.upsert(
            "Q04", "D05_B", 1, annotator="author",
            query="monetary policy", rank="3", document_id="D05", page_number="7",
        )
        row = store.get("Q04", "D05_B").as_row()
        assert row["query"] == "monetary policy"
        assert row["rank"] == "3"
        assert row["document_id"] == "D05"
        assert row["page_number"] == "7"

    @pytest.mark.parametrize("label", [2, -1, "relevant", None, 1.5])
    def test_rejects_a_label_that_is_not_zero_or_one(self, tmp_path, label):
        store = RelevanceStore(path=tmp_path / "j.csv")
        with pytest.raises(RelevanceValidationError):
            store.upsert("Q01", "D01_A", label)

    @pytest.mark.parametrize("query_id,unit_id", [("", "D01_A"), ("Q01", ""), ("", "")])
    def test_requires_both_identifiers(self, tmp_path, query_id, unit_id):
        store = RelevanceStore(path=tmp_path / "j.csv")
        with pytest.raises(RelevanceValidationError):
            store.upsert(query_id, unit_id, 1)


class TestSaveAndLoad:
    def test_round_trips_through_csv(self, tmp_path):
        path = tmp_path / "j.csv"
        store = RelevanceStore(path=path)
        store.upsert("Q01", "D01_A", 1, notes="yes", annotator="author")
        store.upsert("Q01", "D01_B", 0, notes="no", annotator="author")
        store.upsert("Q02", "D01_C", 1, notes="also", annotator="author")
        store.save(annotator="author")

        reloaded = RelevanceStore.load(path, strict=True)
        assert len(reloaded) == 3
        assert reloaded.label("Q01", "D01_A") == 1
        assert reloaded.label("Q01", "D01_B") == 0
        assert reloaded.get("Q01", "D01_B").notes == "no"

    def test_output_is_ordered_so_the_file_is_reproducible(self, tmp_path):
        path = tmp_path / "j.csv"
        store = RelevanceStore(path=path)
        for query_id, unit_id in [("Q02", "D01_C"), ("Q01", "D01_B"), ("Q01", "D01_A")]:
            store.upsert(query_id, unit_id, 1)
        store.save()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == ",".join(FIELDNAMES)
        assert [line.split(",")[0] for line in lines[1:]] == ["Q01", "Q01", "Q02"]
        assert [line.split(",")[2] for line in lines[1:]] == ["D01_A", "D01_B", "D01_C"]

    def test_loading_a_missing_file_is_reported_not_fatal(self, tmp_path):
        # The store treats an absent judgment file as "nothing judged yet", so
        # the API can start before Phase 4 has produced one.
        store = RelevanceStore.load(tmp_path / "absent.csv")
        assert len(store) == 0
        assert any("does not exist" in problem for problem in store.problems)

    def test_loading_a_missing_file_is_still_not_fatal_when_strict(self, tmp_path):
        # An absent file is a valid empty state, not a corrupt one, so even a
        # strict load reports it through ``problems`` instead of raising.
        store = RelevanceStore.load(tmp_path / "absent.csv", strict=True)
        assert len(store) == 0
        assert store.problems


class TestLoadValidation:
    def _write(self, tmp_path, rows):
        import csv

        path = tmp_path / "j.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def test_a_bad_label_is_reported_and_skipped(self, tmp_path):
        path = self._write(tmp_path, [_row(), _row(unit_id="D01_X", relevance="maybe")])
        store = RelevanceStore.load(path)
        assert len(store) == 1
        assert any("relevance" in problem for problem in store.problems)

    def test_a_duplicate_pair_is_reported_and_the_last_wins(self, tmp_path):
        path = self._write(tmp_path, [_row(), _row(relevance="0")])
        store = RelevanceStore.load(path)
        assert len(store) == 1
        assert store.label("Q01", "D01_P002_SEC_1_1_PAR001") == 0
        assert any("duplicate" in problem for problem in store.problems)


class TestQueries:
    def test_relevant_units_per_query(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        store.upsert("Q01", "A", 1)
        store.upsert("Q01", "B", 0)
        store.upsert("Q02", "C", 1)
        assert store.relevant_units("Q01") == {"A"}
        assert store.relevant_units("Q02") == {"C"}

    def test_coverage_counts_judged_per_query(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        store.upsert("Q01", "A", 1)
        store.upsert("Q01", "B", 0)
        store.upsert("Q02", "C", 1)
        coverage = store.coverage(["Q01", "Q02", "Q03"])
        assert coverage["Q01"]["judged"] == 2
        assert coverage["Q01"]["relevant"] == 1
        assert coverage["Q01"]["not_relevant"] == 1
        assert coverage["Q03"]["judged"] == 0

    def test_query_ids_are_sorted_and_unique(self, tmp_path):
        store = RelevanceStore(path=tmp_path / "j.csv")
        for unit_id in ("a", "b"):
            store.upsert("Q02", unit_id, 1)
        store.upsert("Q01", "a", 1)
        assert store.query_ids() == ["Q01", "Q02"]


class TestRealJudgmentFile:
    """The committed dataset must satisfy the rules the validator enforces."""

    @pytest.fixture(scope="class")
    @classmethod
    def store(cls):
        from tests.phase4.conftest import JUDGMENT_FILE

        return RelevanceStore.load(JUDGMENT_FILE, strict=True)

    def test_loads_without_problems(self, store):
        assert store.problems == []

    def test_covers_every_registered_query(self, store):
        assert len(store.query_ids()) == 15

    def test_every_judgment_records_why(self, store):
        for judgment in store.judgments.values():
            assert judgment.notes.strip(), f"{judgment.query_id}/{judgment.unit_id} has no note"

    def test_every_judgment_names_its_annotator(self, store):
        for judgment in store.judgments.values():
            assert judgment.annotator.strip()

    def test_pool_depth_is_ten_except_for_short_rankings(self, store):
        coverage = store.coverage()
        for query_id, counts in coverage.items():
            assert counts["judged"] <= 20, query_id
        assert sum(1 for c in coverage.values() if c["judged"] >= 10) >= 13
