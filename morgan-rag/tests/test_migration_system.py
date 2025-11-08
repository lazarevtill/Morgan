"""
Tests for the migration system.

Tests the migration, validation, and rollback functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from morgan.migration import KnowledgeBaseMigrator, MigrationValidator, RollbackManager


class TestKnowledgeBaseMigrator:
    """Test the KnowledgeBaseMigrator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.migrator = KnowledgeBaseMigrator()

    @patch("morgan.migration.migrator.VectorDBClient")
    def test_analyze_collection_exists(self, mock_vector_db):
        """Test analyzing an existing collection."""
        # Mock vector DB responses
        mock_client = Mock()
        mock_vector_db.return_value = mock_client
        mock_client.collection_exists.return_value = True
        mock_client.get_collection_info.return_value = {
            "points_count": 100,
            "disk_usage": 1024 * 1024,  # 1MB
        }
        mock_client.scroll_points.return_value = [
            {
                "id": "1",
                "vector": [0.1, 0.2, 0.3],  # Legacy format
                "payload": {"content": "test"},
            }
        ]

        # Create new migrator instance to use mocked client
        migrator = KnowledgeBaseMigrator()

        result = migrator.analyze_collection("test_collection")

        assert result["exists"] is True
        assert result["collection_name"] == "test_collection"
        assert result["total_points"] == 100
        assert result["has_legacy_format"] is True
        assert result["has_hierarchical_format"] is False
        assert result["migration_needed"] == "legacy_to_hierarchical"

    @patch("morgan.migration.migrator.VectorDBClient")
    def test_analyze_collection_not_exists(self, mock_vector_db):
        """Test analyzing a non-existent collection."""
        mock_client = Mock()
        mock_vector_db.return_value = mock_client
        mock_client.collection_exists.return_value = False

        migrator = KnowledgeBaseMigrator()

        result = migrator.analyze_collection("nonexistent")

        assert result["exists"] is False
        assert "error" in result

    def test_create_migration_plan(self):
        """Test creating a migration plan."""
        with patch.object(self.migrator, "analyze_collection") as mock_analyze:
            mock_analyze.return_value = {
                "exists": True,
                "total_points": 50,
                "estimated_migration_time_minutes": 1.0,
            }

            plan = self.migrator.create_migration_plan(
                source_collection="test_source", batch_size=25
            )

            assert plan.source_collection == "test_source"
            assert plan.target_collection == "test_source_hierarchical"
            assert plan.total_points == 50
            assert plan.batch_size == 25
            assert plan.estimated_time_minutes == 1.0
            assert plan.backup_path.endswith(".json")

    def test_create_migration_plan_nonexistent_collection(self):
        """Test creating plan for non-existent collection."""
        with patch.object(self.migrator, "analyze_collection") as mock_analyze:
            mock_analyze.return_value = {"exists": False}

            with pytest.raises(ValueError, match="does not exist"):
                self.migrator.create_migration_plan("nonexistent")


class TestMigrationValidator:
    """Test the MigrationValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MigrationValidator()

    @patch("morgan.migration.validator.VectorDBClient")
    def test_validate_migration_success(self, mock_vector_db):
        """Test successful migration validation."""
        mock_client = Mock()
        mock_vector_db.return_value = mock_client
        mock_client.collection_exists.return_value = True
        mock_client.get_collection_info.side_effect = [
            {"points_count": 10},  # source
            {"points_count": 10},  # target
        ]
        mock_client.scroll_points.return_value = [
            {"id": "1", "vector": [0.1, 0.2], "payload": {"content": "test"}}
        ]

        # Mock the Qdrant client retrieve method
        with patch.object(mock_client, "client") as mock_qdrant_client:
            mock_point = Mock()
            mock_point.id = "1"
            mock_point.vector = {
                "coarse": [0.1, 0.2],
                "medium": [0.1, 0.2, 0.3, 0.4],
                "fine": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
            mock_point.payload = {
                "content": "test",
                "migrated_at": "2024-01-01T00:00:00",
                "embedding_type": "hierarchical",
            }
            mock_qdrant_client.retrieve.return_value = [mock_point]

            validator = MigrationValidator()
            result = validator.validate_migration("source", "target", sample_size=1)

            assert result.is_valid is True
            assert result.points_valid == 1
            assert result.points_invalid == 0

    def test_validate_backup_valid(self):
        """Test validating a valid backup file."""
        backup_data = {
            "collection_name": "test_collection",
            "backup_timestamp": "2024-01-01T00:00:00",
            "total_points": 2,
            "points": [
                {"id": "1", "vector": [0.1, 0.2], "payload": {"content": "test1"}},
                {"id": "2", "vector": [0.3, 0.4], "payload": {"content": "test2"}},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(backup_data, f)
            backup_path = f.name

        try:
            result = self.validator.validate_backup(backup_path)

            assert result["is_valid"] is True
            assert result["collection_name"] == "test_collection"
            assert result["total_points"] == 2
            assert result["sample_validation"]["sample_size"] == 2
            assert result["sample_validation"]["valid_points"] == 2
        finally:
            Path(backup_path).unlink()

    def test_validate_backup_invalid(self):
        """Test validating an invalid backup file."""
        result = self.validator.validate_backup("/nonexistent/path.json")

        assert result["is_valid"] is False
        assert "does not exist" in result["error"]


class TestRollbackManager:
    """Test the RollbackManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rollback_manager = RollbackManager()

    def test_list_available_backups_empty(self):
        """Test listing backups when none exist."""
        with patch.object(self.rollback_manager, "backup_dir") as mock_backup_dir:
            mock_backup_dir.exists.return_value = False

            backups = self.rollback_manager.list_available_backups()

            assert backups == []

    def test_validate_backup_for_rollback_valid(self):
        """Test validating a backup for rollback."""
        backup_data = {
            "collection_name": "test_collection",
            "backup_timestamp": "2024-01-01T00:00:00",
            "total_points": 1,
            "points": [
                {"id": "1", "vector": [0.1, 0.2], "payload": {"content": "test"}}
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(backup_data, f)
            backup_path = f.name

        try:
            with patch.object(
                self.rollback_manager.vector_db, "collection_exists"
            ) as mock_exists:
                mock_exists.return_value = False

                result = self.rollback_manager.validate_backup_for_rollback(backup_path)

                assert result["is_valid"] is True
                assert result["collection_name"] == "test_collection"
                assert result["total_points"] == 1
                assert result["target_collection_exists"] is False
        finally:
            Path(backup_path).unlink()

    def test_validate_backup_for_rollback_missing_fields(self):
        """Test validating backup with missing required fields."""
        backup_data = {
            "collection_name": "test_collection"
            # Missing required fields
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(backup_data, f)
            backup_path = f.name

        try:
            result = self.rollback_manager.validate_backup_for_rollback(backup_path)

            assert result["is_valid"] is False
            assert "missing required fields" in result["error"].lower()
        finally:
            Path(backup_path).unlink()

    def test_cleanup_old_backups(self):
        """Test cleaning up old backup files."""
        with patch.object(self.rollback_manager, "backup_dir") as mock_backup_dir:
            # Mock backup directory with files
            mock_backup_dir.exists.return_value = True

            # Create mock backup files
            old_file = Mock()
            old_file.stat.return_value.st_mtime = 0  # Very old file
            old_file.stat.return_value.st_size = 1024
            old_file.name = "old_backup.json"

            new_file = Mock()
            new_file.stat.return_value.st_mtime = 9999999999  # Very new file
            new_file.name = "new_backup.json"

            mock_backup_dir.glob.return_value = [old_file, new_file]

            result = self.rollback_manager.cleanup_old_backups(keep_days=1)

            assert result["files_deleted"] == 1
            assert result["space_freed_mb"] > 0
            old_file.unlink.assert_called_once()
            new_file.unlink.assert_not_called()


class TestMigrationIntegration:
    """Integration tests for the migration system."""

    @patch("morgan.migration.migrator.VectorDBClient")
    @patch("morgan.migration.migrator.get_hierarchical_embedding_service")
    def test_full_migration_workflow(self, mock_hierarchical_service, mock_vector_db):
        """Test the complete migration workflow."""
        # Mock services
        mock_client = Mock()
        mock_vector_db.return_value = mock_client
        mock_client.collection_exists.return_value = True
        mock_client.get_collection_info.return_value = {"points_count": 2}
        mock_client.scroll_points.return_value = [
            {"id": "1", "vector": [0.1, 0.2], "payload": {"content": "test1"}},
            {"id": "2", "vector": [0.3, 0.4], "payload": {"content": "test2"}},
        ]
        mock_client.ensure_hierarchical_collection.return_value = True
        mock_client.upsert_hierarchical_points.return_value = True

        mock_hierarchical_emb_service = Mock()
        mock_hierarchical_service.return_value = mock_hierarchical_emb_service

        # Mock hierarchical embedding
        mock_embedding = Mock()
        mock_embedding.get_embedding.side_effect = lambda scale: (
            [0.1, 0.2] if scale == "coarse" else [0.1, 0.2, 0.3, 0.4]
        )
        mock_embedding.texts = {"coarse": "test", "medium": "test", "fine": "test"}
        mock_embedding.metadata = {}
        mock_hierarchical_emb_service.create_hierarchical_embeddings.return_value = (
            mock_embedding
        )

        # Create migrator and execute workflow
        migrator = KnowledgeBaseMigrator()

        # 1. Analyze
        analysis = migrator.analyze_collection("test_collection")
        assert analysis["exists"] is True

        # 2. Create plan
        plan = migrator.create_migration_plan("test_collection", dry_run=True)
        assert plan.source_collection == "test_collection"

        # 3. Execute (dry run)
        with patch.object(migrator, "_create_backup") as mock_backup:
            mock_backup.return_value = True
            result = migrator.execute_migration(plan)

            assert result.success is True
            assert result.points_migrated == 2
