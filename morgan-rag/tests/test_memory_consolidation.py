"""
Tests for the memory consolidation module.

Covers:
- daily_log: DailyLogManager append, read, list operations
- hybrid_search: BM25 keyword search, cosine similarity, hybrid merge
- consolidator: MemoryConsolidator with mocked LLM
"""

import math
import pytest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

from morgan.memory_consolidation.daily_log import DailyLogManager
from morgan.memory_consolidation.hybrid_search import HybridMemorySearch
from morgan.memory_consolidation.consolidator import MemoryConsolidator


# =========================================================================
# DailyLogManager tests
# =========================================================================


class TestDailyLogManagerAppend:
    """Tests for DailyLogManager.append."""

    def test_append_creates_file(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("Test entry", date=date(2026, 3, 15))

        log_file = tmp_path / "memory" / "2026-03-15.md"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test entry" in content

    def test_append_adds_timestamp_prefix(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("Hello world", date=date(2026, 1, 1))

        content = (tmp_path / "memory" / "2026-01-01.md").read_text()
        # Should have a [HH:MM] prefix
        import re
        assert re.search(r"\[\d{2}:\d{2}\]", content)

    def test_append_multiple_entries_same_day(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("First entry", date=date(2026, 5, 10))
        mgr.append("Second entry", date=date(2026, 5, 10))

        content = (tmp_path / "memory" / "2026-05-10.md").read_text()
        assert "First entry" in content
        assert "Second entry" in content

    def test_append_default_date_is_today(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("Today's entry")

        today_str = date.today().isoformat()
        log_file = tmp_path / "memory" / f"{today_str}.md"
        assert log_file.exists()

    def test_append_creates_directory(self, tmp_path):
        memory_dir = tmp_path / "nested" / "memory"
        mgr = DailyLogManager(memory_dir)
        mgr.append("Entry", date=date(2026, 1, 1))

        assert (memory_dir / "2026-01-01.md").exists()


class TestDailyLogManagerRead:
    """Tests for DailyLogManager read operations."""

    def test_read_today_returns_todays_content(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("Morning check-in")

        content = mgr.read_today()
        assert "Morning check-in" in content

    def test_read_today_returns_empty_when_no_log(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        content = mgr.read_today()
        assert content == ""

    def test_read_date_specific(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("March entry", date=date(2026, 3, 20))

        content = mgr.read_date(date(2026, 3, 20))
        assert "March entry" in content

    def test_read_date_nonexistent(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        content = mgr.read_date(date(2020, 1, 1))
        assert content == ""


class TestDailyLogManagerListRecent:
    """Tests for DailyLogManager.list_recent."""

    def test_list_recent_returns_sorted_newest_first(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        from datetime import timedelta

        today = date.today()
        day1 = today - timedelta(days=3)
        day2 = today - timedelta(days=2)
        day3 = today - timedelta(days=1)

        mgr.append("Day 1 entry", date=day1)
        mgr.append("Day 2 entry", date=day2)
        mgr.append("Day 3 entry", date=day3)

        results = mgr.list_recent(days=7)
        assert len(results) == 3
        assert results[0]["date"] == day3.isoformat()
        assert results[1]["date"] == day2.isoformat()
        assert results[2]["date"] == day1.isoformat()

    def test_list_recent_includes_content_and_path(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("Some content", date=date(2026, 3, 15))

        results = mgr.list_recent(days=30)
        assert len(results) >= 1
        entry = results[0]
        assert "content" in entry
        assert "path" in entry
        assert "date" in entry
        assert "Some content" in entry["content"]

    def test_list_recent_respects_days_limit(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        # Create logs across many days
        mgr.append("Old entry", date=date(2026, 1, 1))
        mgr.append("Recent entry", date=date.today())

        results = mgr.list_recent(days=7)
        # Only the recent one should be within 7 days
        dates = [r["date"] for r in results]
        assert "2026-01-01" not in dates

    def test_list_recent_default_7_days(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        mgr.append("Today's log")

        results = mgr.list_recent()
        assert len(results) >= 1

    def test_list_recent_empty_directory(self, tmp_path):
        mgr = DailyLogManager(tmp_path / "memory")
        results = mgr.list_recent()
        assert results == []


# =========================================================================
# HybridMemorySearch tests
# =========================================================================


class TestKeywordSearch:
    """Tests for BM25-based keyword search."""

    def _setup_logs(self, tmp_path):
        """Helper to create a memory dir with some log files."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        (memory_dir / "2026-03-01.md").write_text(
            "[09:00] Worked on Python refactoring\n"
            "[10:30] Discussed API design patterns\n"
        )
        (memory_dir / "2026-03-02.md").write_text(
            "[08:00] Morning standup meeting\n"
            "[14:00] Python code review with team\n"
            "[16:00] Fixed Python import bug\n"
        )
        (memory_dir / "2026-03-03.md").write_text(
            "[09:00] Started new JavaScript project\n"
            "[11:00] Database migration planning\n"
        )
        return memory_dir

    def test_keyword_search_returns_results(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results = search.keyword_search("Python")
        assert len(results) > 0

    def test_keyword_search_ranks_relevant_higher(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results = search.keyword_search("Python")
        # The file with more "Python" mentions (2026-03-02) should rank higher
        assert len(results) >= 2
        # First result should be the one with more Python mentions
        assert "2026-03-02" in results[0]["path"]

    def test_keyword_search_respects_limit(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results = search.keyword_search("Python", limit=1)
        assert len(results) == 1

    def test_keyword_search_no_matches(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results = search.keyword_search("xyznonexistent")
        assert results == []

    def test_keyword_search_case_insensitive(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results_lower = search.keyword_search("python")
        results_upper = search.keyword_search("Python")
        assert len(results_lower) == len(results_upper)

    def test_keyword_search_empty_directory(self, tmp_path):
        memory_dir = tmp_path / "empty_memory"
        memory_dir.mkdir()
        search = HybridMemorySearch(memory_dir)

        results = search.keyword_search("anything")
        assert results == []

    def test_keyword_search_multi_word_query(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results = search.keyword_search("Python code review")
        assert len(results) > 0


class TestBM25Scoring:
    """Tests for BM25 scoring internals."""

    def test_bm25_idf_positive_for_rare_terms(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        # JavaScript only appears in one doc
        results = search.keyword_search("JavaScript")
        assert len(results) >= 1
        assert results[0]["score"] > 0

    def _setup_logs(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        (memory_dir / "2026-03-01.md").write_text(
            "[09:00] Worked on Python refactoring\n"
        )
        (memory_dir / "2026-03-02.md").write_text(
            "[08:00] Python code review\n[14:00] Python testing\n"
        )
        (memory_dir / "2026-03-03.md").write_text(
            "[09:00] Started new JavaScript project\n"
        )
        return memory_dir


class TestCosineSimilarity:
    """Tests for cosine similarity helper."""

    def test_identical_vectors_score_one(self):
        from morgan.memory_consolidation.hybrid_search import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        from morgan.memory_consolidation.hybrid_search import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors_score_negative(self):
        from morgan.memory_consolidation.hybrid_search import _cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) < 0

    def test_zero_vector_returns_zero(self):
        from morgan.memory_consolidation.hybrid_search import _cosine_similarity
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


class TestHybridSearch:
    """Tests for hybrid (keyword + vector) search."""

    def _setup_logs(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        (memory_dir / "2026-03-01.md").write_text(
            "[09:00] Worked on Python refactoring\n"
        )
        (memory_dir / "2026-03-02.md").write_text(
            "[08:00] Python code review\n"
        )
        (memory_dir / "2026-03-03.md").write_text(
            "[09:00] JavaScript project setup\n"
        )
        return memory_dir

    def test_hybrid_search_falls_back_to_keyword_only(self, tmp_path):
        """When embeddings are unavailable, hybrid_search should still work."""
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        # Without any embedding service configured, should fall back
        results = search.hybrid_search("Python")
        assert len(results) > 0

    def test_hybrid_search_with_mock_embeddings(self, tmp_path):
        """Test hybrid search with mocked embedding service."""
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        mock_embedding_svc = MagicMock()
        # Return simple embeddings: query=[1,0,0], doc embeddings vary
        mock_embedding_svc.encode.side_effect = lambda text, **kwargs: (
            [1.0, 0.0, 0.0] if "Python" in text
            else [0.5, 0.5, 0.0] if "JavaScript" in text
            else [0.8, 0.2, 0.0]
        )

        with patch(
            "morgan.services.get_embedding_service",
            return_value=mock_embedding_svc,
        ):
            results = search.hybrid_search(
                "Python",
                limit=10,
                vector_weight=0.6,
                keyword_weight=0.4,
            )

        assert len(results) > 0
        # Results should have combined scores
        for r in results:
            assert "score" in r

    def test_hybrid_search_respects_limit(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        results = search.hybrid_search("Python", limit=1)
        assert len(results) <= 1

    def test_hybrid_search_custom_weights(self, tmp_path):
        memory_dir = self._setup_logs(tmp_path)
        search = HybridMemorySearch(memory_dir)

        # All keyword weight
        results_kw = search.hybrid_search(
            "Python", keyword_weight=1.0, vector_weight=0.0
        )
        # These should match keyword-only results
        results_pure_kw = search.keyword_search("Python")

        assert len(results_kw) == len(results_pure_kw)


# =========================================================================
# MemoryConsolidator tests
# =========================================================================


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator."""

    def test_consolidate_returns_none_when_no_logs(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "memory").mkdir()

        consolidator = MemoryConsolidator(workspace)
        result = consolidator.consolidate()
        assert result is None

    def test_consolidate_reads_recent_logs(self, tmp_path):
        workspace = tmp_path / "workspace"
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True)

        # Create a log for today
        today = date.today().isoformat()
        (memory_dir / f"{today}.md").write_text("[09:00] Important discovery\n")

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content="# MEMORY.md\n\n## Key Insights\n- Important discovery\n"
        )

        with patch(
            "morgan.services.get_llm_service",
            return_value=mock_llm,
        ):
            consolidator = MemoryConsolidator(workspace)
            result = consolidator.consolidate(days_to_review=7)

        assert result is not None
        assert "Important discovery" in result

    def test_consolidate_writes_memory_md(self, tmp_path):
        workspace = tmp_path / "workspace"
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True)

        today = date.today().isoformat()
        (memory_dir / f"{today}.md").write_text("[10:00] Learned about RAG\n")

        new_content = "# MEMORY.md\n\n## Learnings\n- RAG architecture\n"
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(content=new_content)

        with patch(
            "morgan.services.get_llm_service",
            return_value=mock_llm,
        ):
            consolidator = MemoryConsolidator(workspace)
            consolidator.consolidate()

        memory_md = workspace / "MEMORY.md"
        assert memory_md.exists()
        assert "RAG architecture" in memory_md.read_text()

    def test_consolidate_includes_existing_memory_md_in_prompt(self, tmp_path):
        workspace = tmp_path / "workspace"
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True)

        # Create existing MEMORY.md
        (workspace / "MEMORY.md").write_text("# Existing Memory\n- Old fact\n")

        today = date.today().isoformat()
        (memory_dir / f"{today}.md").write_text("[09:00] New insight\n")

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content="# Updated Memory\n- Old fact\n- New insight\n"
        )

        with patch(
            "morgan.services.get_llm_service",
            return_value=mock_llm,
        ):
            consolidator = MemoryConsolidator(workspace)
            consolidator.consolidate()

        # Verify the LLM was called with a prompt containing existing memory
        call_args = mock_llm.generate.call_args
        prompt = call_args[1].get("prompt") or call_args[0][0]
        assert "Old fact" in prompt

    def test_consolidate_custom_days_to_review(self, tmp_path):
        workspace = tmp_path / "workspace"
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True)

        # Only create a log far in the past
        (memory_dir / "2020-01-01.md").write_text("[09:00] Ancient log\n")

        consolidator = MemoryConsolidator(workspace)
        # With only 1 day review, the old log should not be found
        result = consolidator.consolidate(days_to_review=1)
        assert result is None

    def test_consolidator_handles_llm_error_gracefully(self, tmp_path):
        workspace = tmp_path / "workspace"
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True)

        today = date.today().isoformat()
        (memory_dir / f"{today}.md").write_text("[09:00] Some log\n")

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM unavailable")

        with patch(
            "morgan.services.get_llm_service",
            return_value=mock_llm,
        ):
            consolidator = MemoryConsolidator(workspace)
            result = consolidator.consolidate()

        # Should return None on error, not crash
        assert result is None


# =========================================================================
# Integration-style tests
# =========================================================================


class TestIntegration:
    """Integration tests combining daily log and search."""

    def test_append_then_search(self, tmp_path):
        memory_dir = tmp_path / "memory"
        mgr = DailyLogManager(memory_dir)
        mgr.append("Implemented vector search with FAISS", date=date(2026, 3, 15))
        mgr.append("Fixed bug in embedding pipeline", date=date(2026, 3, 16))
        mgr.append("Team meeting about project roadmap", date=date(2026, 3, 17))

        search = HybridMemorySearch(memory_dir)
        results = search.keyword_search("vector search")
        assert len(results) >= 1
        assert "vector search" in results[0]["content"].lower()

    def test_daily_log_and_consolidator_workflow(self, tmp_path):
        workspace = tmp_path / "workspace"
        memory_dir = workspace / "memory"

        mgr = DailyLogManager(memory_dir)
        mgr.append("Started the Morgan project")
        mgr.append("Implemented daily logging feature")

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content="# MEMORY.md\n\n- Started Morgan project\n- Implemented daily logging\n"
        )

        with patch(
            "morgan.services.get_llm_service",
            return_value=mock_llm,
        ):
            consolidator = MemoryConsolidator(workspace)
            result = consolidator.consolidate()

        assert result is not None
        assert (workspace / "MEMORY.md").exists()
