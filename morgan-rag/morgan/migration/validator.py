"""
Migration Validation System

Validates migration integrity and provides verification tools.

Implements requirements R10.4 and R10.5 for migration safety.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient
from morgan.services.embedding_service import get_embedding_service

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Validation result for a migration."""
    is_valid: bool
    total_points_checked: int
    points_valid: int
    points_invalid: int
    validation_errors: List[str]
    performance_metrics: Dict[str, float]


class MigrationValidator:
    """
    Validates migration integrity and correctness.
    
    Provides tools to verify that migrations completed successfully
    and that data integrity is maintained.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.vector_db = VectorDBClient()
        self.embedding_service = get_embedding_service()
        
        logger.info("Migration validator initialized")
    
    def validate_migration(
        self,
        source_collection: str,
        target_collection: str,
        sample_size: int = 100
    ) -> ValidationResult:
        """
        Validate a completed migration.
        
        Args:
            source_collection: Original collection name
            target_collection: Migrated collection name
            sample_size: Number of points to sample for validation
            
        Returns:
            Validation result
        """
        validation_errors = []
        points_valid = 0
        points_invalid = 0
        performance_metrics = {}
        
        try:
            # Check collections exist
            if not self.vector_db.collection_exists(source_collection):
                validation_errors.append(f"Source collection '{source_collection}' does not exist")
                
            if not self.vector_db.collection_exists(target_collection):
                validation_errors.append(f"Target collection '{target_collection}' does not exist")
            
            if validation_errors:
                return ValidationResult(
                    is_valid=False,
                    total_points_checked=0,
                    points_valid=0,
                    points_invalid=0,
                    validation_errors=validation_errors,
                    performance_metrics={}
                )
            
            # Get collection info
            source_info = self.vector_db.get_collection_info(source_collection)
            target_info = self.vector_db.get_collection_info(target_collection)
            
            source_count = source_info.get("points_count", 0)
            target_count = target_info.get("points_count", 0)
            
            # Check point counts
            if source_count != target_count:
                validation_errors.append(
                    f"Point count mismatch: source={source_count}, target={target_count}"
                )
            
            # Sample points for detailed validation
            sample_size = min(sample_size, source_count)
            
            if sample_size > 0:
                # Get sample from source
                source_sample = self.vector_db.scroll_points(
                    collection_name=source_collection,
                    limit=sample_size
                )
                
                # Validate each sampled point
                for source_point in source_sample:
                    point_id = source_point.get("id")
                    
                    # Get corresponding point from target
                    target_point = self._get_point_by_id(target_collection, point_id)
                    
                    if target_point is None:
                        validation_errors.append(f"Point {point_id} missing in target collection")
                        points_invalid += 1
                        continue
                    
                    # Validate point structure and content
                    point_validation = self._validate_point_pair(source_point, target_point)
                    
                    if point_validation["is_valid"]:
                        points_valid += 1
                    else:
                        points_invalid += 1
                        validation_errors.extend(point_validation["errors"])
            
            # Calculate performance metrics
            performance_metrics = {
                "source_points": source_count,
                "target_points": target_count,
                "migration_efficiency": target_count / source_count if source_count > 0 else 0,
                "validation_coverage": sample_size / source_count if source_count > 0 else 0,
                "error_rate": points_invalid / sample_size if sample_size > 0 else 0
            }
            
            # Determine overall validity
            is_valid = (
                len(validation_errors) == 0 and
                points_invalid == 0 and
                source_count == target_count
            )
            
            return ValidationResult(
                is_valid=is_valid,
                total_points_checked=sample_size,
                points_valid=points_valid,
                points_invalid=points_invalid,
                validation_errors=validation_errors,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                total_points_checked=0,
                points_valid=0,
                points_invalid=0,
                validation_errors=[f"Validation error: {str(e)}"],
                performance_metrics={}
            )
    
    def _get_point_by_id(self, collection_name: str, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific point by ID from a collection.
        
        Args:
            collection_name: Collection name
            point_id: Point ID to retrieve
            
        Returns:
            Point data or None if not found
        """
        try:
            # Use Qdrant's retrieve method to get specific point
            points = self.vector_db.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if points and len(points) > 0:
                point = points[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve point {point_id}: {e}")
            return None
    
    def _validate_point_pair(
        self, 
        source_point: Dict[str, Any], 
        target_point: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a source-target point pair.
        
        Args:
            source_point: Point from source collection
            target_point: Point from target collection
            
        Returns:
            Validation result for the point pair
        """
        errors = []
        
        try:
            # Check IDs match
            if source_point.get("id") != target_point.get("id"):
                errors.append(f"ID mismatch: {source_point.get('id')} != {target_point.get('id')}")
            
            # Check payload content
            source_payload = source_point.get("payload", {})
            target_payload = target_point.get("payload", {})
            
            source_content = source_payload.get("content", "")
            target_content = target_payload.get("content", "")
            
            if source_content != target_content:
                errors.append(f"Content mismatch for point {source_point.get('id')}")
            
            # Check vector structure
            source_vector = source_point.get("vector")
            target_vector = target_point.get("vector")
            
            # Source should be single vector (list), target should be named vectors (dict)
            if not isinstance(source_vector, list):
                errors.append(f"Source vector should be list, got {type(source_vector)}")
            
            if not isinstance(target_vector, dict):
                errors.append(f"Target vector should be dict, got {type(target_vector)}")
            else:
                # Check hierarchical vector structure
                required_vectors = ["coarse", "medium", "fine"]
                for vector_name in required_vectors:
                    if vector_name not in target_vector:
                        errors.append(f"Missing {vector_name} vector in target point")
                    elif not isinstance(target_vector[vector_name], list):
                        errors.append(f"{vector_name} vector should be list")
            
            # Check migration metadata
            if "migrated_at" not in target_payload:
                errors.append("Missing migration timestamp in target point")
            
            if target_payload.get("embedding_type") != "hierarchical":
                errors.append("Target point should have embedding_type='hierarchical'")
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Point validation error: {str(e)}"]
            }
    
    def validate_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Validate a migration backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Backup validation result
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return {
                    "is_valid": False,
                    "error": f"Backup file does not exist: {backup_path}"
                }
            
            # Load backup data
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Validate backup structure
            required_fields = ["collection_name", "backup_timestamp", "total_points", "points"]
            missing_fields = [field for field in required_fields if field not in backup_data]
            
            if missing_fields:
                return {
                    "is_valid": False,
                    "error": f"Missing required fields: {missing_fields}"
                }
            
            # Validate points data
            points = backup_data.get("points", [])
            total_points = backup_data.get("total_points", 0)
            
            if len(points) != total_points:
                return {
                    "is_valid": False,
                    "error": f"Point count mismatch: expected {total_points}, got {len(points)}"
                }
            
            # Sample validate some points
            sample_size = min(10, len(points))
            valid_points = 0
            
            for i in range(sample_size):
                point = points[i]
                if self._validate_backup_point(point):
                    valid_points += 1
            
            return {
                "is_valid": True,
                "collection_name": backup_data["collection_name"],
                "backup_timestamp": backup_data["backup_timestamp"],
                "total_points": total_points,
                "file_size_mb": backup_file.stat().st_size / (1024 * 1024),
                "sample_validation": {
                    "sample_size": sample_size,
                    "valid_points": valid_points,
                    "validation_rate": valid_points / sample_size if sample_size > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Backup validation failed: {str(e)}"
            }
    
    def _validate_backup_point(self, point: Dict[str, Any]) -> bool:
        """
        Validate a single point from backup.
        
        Args:
            point: Point data from backup
            
        Returns:
            True if point is valid
        """
        try:
            # Check required fields
            if "id" not in point:
                return False
            
            if "vector" not in point:
                return False
            
            if "payload" not in point:
                return False
            
            # Check payload has content
            payload = point.get("payload", {})
            if "content" not in payload:
                return False
            
            return True
            
        except Exception:
            return False
    
    def compare_collections(
        self,
        collection1: str,
        collection2: str,
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """
        Compare two collections for differences.
        
        Args:
            collection1: First collection name
            collection2: Second collection name
            sample_size: Number of points to sample for comparison
            
        Returns:
            Comparison result
        """
        try:
            # Get collection info
            info1 = self.vector_db.get_collection_info(collection1)
            info2 = self.vector_db.get_collection_info(collection2)
            
            count1 = info1.get("points_count", 0)
            count2 = info2.get("points_count", 0)
            
            # Sample points from both collections
            sample1 = self.vector_db.scroll_points(collection1, limit=sample_size)
            sample2 = self.vector_db.scroll_points(collection2, limit=sample_size)
            
            # Create ID sets for comparison
            ids1 = {point.get("id") for point in sample1}
            ids2 = {point.get("id") for point in sample2}
            
            common_ids = ids1.intersection(ids2)
            unique_to_1 = ids1 - ids2
            unique_to_2 = ids2 - ids1
            
            return {
                "collection1": {
                    "name": collection1,
                    "total_points": count1,
                    "sample_size": len(sample1)
                },
                "collection2": {
                    "name": collection2,
                    "total_points": count2,
                    "sample_size": len(sample2)
                },
                "comparison": {
                    "common_points": len(common_ids),
                    "unique_to_collection1": len(unique_to_1),
                    "unique_to_collection2": len(unique_to_2),
                    "similarity_ratio": len(common_ids) / max(len(ids1), len(ids2)) if max(len(ids1), len(ids2)) > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Collection comparison failed: {e}")
            return {
                "error": str(e)
            }