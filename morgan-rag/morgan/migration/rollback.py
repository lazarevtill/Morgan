"""
Migration Rollback System

Provides rollback capabilities for failed or problematic migrations.

Implements requirements R10.4 and R10.5 for migration safety.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient

logger = get_logger(__name__)


@dataclass
class RollbackResult:
    """Rollback execution result."""
    success: bool
    points_restored: int
    execution_time_seconds: float
    collection_restored: str
    backup_file_used: str
    error_message: Optional[str] = None


class RollbackManager:
    """
    Manages rollback operations for migrations.
    
    Provides safe rollback from backups with validation.
    """
    
    def __init__(self):
        """Initialize the rollback manager."""
        self.vector_db = VectorDBClient()
        self.backup_dir = Path.home() / ".morgan" / "migration_backups"
        
        logger.info("Rollback manager initialized")
    
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backup files.
        
        Returns:
            List of backup file information
        """
        try:
            if not self.backup_dir.exists():
                return []
            
            backups = []
            
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    # Get file info
                    stat = backup_file.stat()
                    file_size_mb = stat.st_size / (1024 * 1024)
                    
                    # Try to read backup metadata
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        # Read just the first part to get metadata
                        content = f.read(1024)  # Read first 1KB
                        f.seek(0)
                        
                        # Try to parse as JSON to get metadata
                        try:
                            backup_data = json.load(f)
                            collection_name = backup_data.get("collection_name", "unknown")
                            backup_timestamp = backup_data.get("backup_timestamp", "unknown")
                            total_points = backup_data.get("total_points", 0)
                        except json.JSONDecodeError:
                            # If JSON is malformed, use file info only
                            collection_name = backup_file.stem.split("_backup_")[0]
                            backup_timestamp = datetime.fromtimestamp(stat.st_mtime).isoformat()
                            total_points = 0
                    
                    backups.append({
                        "file_path": str(backup_file),
                        "file_name": backup_file.name,
                        "collection_name": collection_name,
                        "backup_timestamp": backup_timestamp,
                        "total_points": total_points,
                        "file_size_mb": file_size_mb,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read backup file {backup_file}: {e}")
                    continue
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def validate_backup_for_rollback(self, backup_path: str) -> Dict[str, Any]:
        """
        Validate that a backup can be used for rollback.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Validation result
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return {
                    "is_valid": False,
                    "error": f"Backup file does not exist: {backup_path}"
                }
            
            # Load and validate backup
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Check required fields
            required_fields = ["collection_name", "backup_timestamp", "total_points", "points"]
            missing_fields = [field for field in required_fields if field not in backup_data]
            
            if missing_fields:
                return {
                    "is_valid": False,
                    "error": f"Backup missing required fields: {missing_fields}"
                }
            
            collection_name = backup_data["collection_name"]
            total_points = backup_data["total_points"]
            points = backup_data.get("points", [])
            
            # Validate point count
            if len(points) != total_points:
                return {
                    "is_valid": False,
                    "error": f"Point count mismatch: expected {total_points}, found {len(points)}"
                }
            
            # Check if target collection exists (warn if it does)
            collection_exists = self.vector_db.collection_exists(collection_name)
            
            # Sample validate some points
            sample_size = min(5, len(points))
            valid_points = 0
            
            for i in range(sample_size):
                point = points[i]
                if self._validate_point_structure(point):
                    valid_points += 1
            
            validation_rate = valid_points / sample_size if sample_size > 0 else 0
            
            return {
                "is_valid": validation_rate >= 0.8,  # At least 80% of sample should be valid
                "collection_name": collection_name,
                "total_points": total_points,
                "backup_timestamp": backup_data["backup_timestamp"],
                "target_collection_exists": collection_exists,
                "sample_validation_rate": validation_rate,
                "warnings": [
                    f"Target collection '{collection_name}' already exists and will be overwritten"
                ] if collection_exists else []
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Backup validation failed: {str(e)}"
            }
    
    def execute_rollback(
        self,
        backup_path: str,
        target_collection: Optional[str] = None,
        confirm_overwrite: bool = False
    ) -> RollbackResult:
        """
        Execute rollback from a backup file.
        
        Args:
            backup_path: Path to backup file
            target_collection: Target collection name (uses backup collection name if None)
            confirm_overwrite: Confirm overwriting existing collection
            
        Returns:
            Rollback result
        """
        start_time = time.time()
        points_restored = 0
        error_message = None
        
        try:
            # Validate backup first
            validation = self.validate_backup_for_rollback(backup_path)
            if not validation["is_valid"]:
                raise Exception(f"Backup validation failed: {validation.get('error', 'Unknown error')}")
            
            # Load backup data
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            collection_name = target_collection or backup_data["collection_name"]
            points = backup_data["points"]
            
            logger.info(f"Starting rollback to collection '{collection_name}' from {backup_path}")
            
            # Check if collection exists and handle overwrite
            if self.vector_db.collection_exists(collection_name):
                if not confirm_overwrite:
                    raise Exception(
                        f"Collection '{collection_name}' already exists. "
                        f"Use confirm_overwrite=True to overwrite."
                    )
                
                logger.warning(f"Overwriting existing collection '{collection_name}'")
                # Delete existing collection
                self.vector_db.delete_collection(collection_name)
            
            # Recreate collection
            # Determine if this is a legacy or hierarchical collection
            sample_point = points[0] if points else {}
            sample_vector = sample_point.get("vector", [])
            
            if isinstance(sample_vector, dict):
                # Hierarchical collection
                logger.info("Restoring hierarchical collection")
                success = self.vector_db.ensure_hierarchical_collection(
                    name=collection_name,
                    coarse_size=384,
                    medium_size=768,
                    fine_size=1536,
                    distance="cosine"
                )
                if not success:
                    raise Exception(f"Failed to create hierarchical collection '{collection_name}'")
                
                # Restore hierarchical points
                self._restore_hierarchical_points(collection_name, points)
                
            else:
                # Legacy collection
                logger.info("Restoring legacy collection")
                vector_size = len(sample_vector) if isinstance(sample_vector, list) else 1536
                
                self.vector_db.create_collection(
                    name=collection_name,
                    vector_size=vector_size,
                    distance="cosine"
                )
                
                # Restore legacy points
                self._restore_legacy_points(collection_name, points)
            
            points_restored = len(points)
            execution_time = time.time() - start_time
            
            logger.info(f"Rollback completed: {points_restored} points restored in {execution_time:.1f}s")
            
            return RollbackResult(
                success=True,
                points_restored=points_restored,
                execution_time_seconds=execution_time,
                collection_restored=collection_name,
                backup_file_used=backup_path
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Rollback failed after {execution_time:.1f}s: {error_message}")
            
            return RollbackResult(
                success=False,
                points_restored=points_restored,
                execution_time_seconds=execution_time,
                collection_restored=target_collection or "unknown",
                backup_file_used=backup_path,
                error_message=error_message
            )
    
    def _validate_point_structure(self, point: Dict[str, Any]) -> bool:
        """
        Validate basic point structure.
        
        Args:
            point: Point data to validate
            
        Returns:
            True if point structure is valid
        """
        try:
            # Check required fields
            if "id" not in point:
                return False
            
            if "vector" not in point:
                return False
            
            if "payload" not in point:
                return False
            
            # Check vector is list or dict
            vector = point["vector"]
            if not isinstance(vector, (list, dict)):
                return False
            
            # Check payload has content
            payload = point["payload"]
            if not isinstance(payload, dict):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _restore_hierarchical_points(self, collection_name: str, points: List[Dict[str, Any]]) -> None:
        """
        Restore hierarchical points to collection.
        
        Args:
            collection_name: Target collection name
            points: Points to restore
        """
        batch_size = 50
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            # Prepare hierarchical points
            hierarchical_points = []
            for point in batch:
                hierarchical_point = {
                    "id": point["id"],
                    "vector": point["vector"],  # Should already be dict with named vectors
                    "payload": {
                        **point["payload"],
                        "restored_at": datetime.now().isoformat(),
                        "restored_from_backup": True
                    }
                }
                hierarchical_points.append(hierarchical_point)
            
            # Upsert batch
            success = self.vector_db.upsert_hierarchical_points(collection_name, hierarchical_points)
            if not success:
                raise Exception(f"Failed to restore batch {i//batch_size + 1}")
            
            logger.debug(f"Restored batch {i//batch_size + 1}: {len(batch)} points")
    
    def _restore_legacy_points(self, collection_name: str, points: List[Dict[str, Any]]) -> None:
        """
        Restore legacy points to collection.
        
        Args:
            collection_name: Target collection name
            points: Points to restore
        """
        batch_size = 100
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            # Prepare legacy points
            legacy_points = []
            for point in batch:
                legacy_point = {
                    "id": point["id"],
                    "vector": point["vector"],  # Should be list
                    "payload": {
                        **point["payload"],
                        "restored_at": datetime.now().isoformat(),
                        "restored_from_backup": True
                    }
                }
                legacy_points.append(legacy_point)
            
            # Upsert batch
            success = self.vector_db.upsert_points(collection_name, legacy_points)
            if not success:
                raise Exception(f"Failed to restore batch {i//batch_size + 1}")
            
            logger.debug(f"Restored batch {i//batch_size + 1}: {len(batch)} points")
    
    def cleanup_old_backups(self, keep_days: int = 30) -> Dict[str, Any]:
        """
        Clean up old backup files.
        
        Args:
            keep_days: Number of days to keep backups
            
        Returns:
            Cleanup result
        """
        try:
            if not self.backup_dir.exists():
                return {
                    "files_deleted": 0,
                    "space_freed_mb": 0,
                    "message": "No backup directory found"
                }
            
            cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
            
            files_deleted = 0
            space_freed = 0
            
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    if backup_file.stat().st_mtime < cutoff_time:
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()
                        files_deleted += 1
                        space_freed += file_size
                        logger.info(f"Deleted old backup: {backup_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to delete backup {backup_file}: {e}")
                    continue
            
            space_freed_mb = space_freed / (1024 * 1024)
            
            return {
                "files_deleted": files_deleted,
                "space_freed_mb": space_freed_mb,
                "message": f"Deleted {files_deleted} old backups, freed {space_freed_mb:.1f} MB"
            }
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return {
                "files_deleted": 0,
                "space_freed_mb": 0,
                "error": str(e)
            }