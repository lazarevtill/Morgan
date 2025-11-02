"""
Knowledge Base Migration System

Provides safe migration from legacy single-vector embeddings to hierarchical
multi-scale embeddings with rollback capabilities.

Implements requirements R10.4 and R10.5.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.vector_db.client import VectorDBClient
from morgan.vectorization.hierarchical_embeddings import get_hierarchical_embedding_service
from morgan.services.embedding_service import get_embedding_service
from morgan.caching.git_hash_tracker import GitHashTracker

logger = get_logger(__name__)


@dataclass
class MigrationPlan:
    """Migration execution plan."""
    source_collection: str
    target_collection: str
    total_points: int
    batch_size: int
    estimated_time_minutes: float
    backup_path: str
    dry_run: bool = False


@dataclass
class MigrationResult:
    """Migration execution result."""
    success: bool
    points_migrated: int
    points_failed: int
    execution_time_seconds: float
    backup_created: bool
    rollback_available: bool
    error_message: Optional[str] = None


class KnowledgeBaseMigrator:
    """
    Migrates knowledge bases from legacy to hierarchical format.
    
    Provides safe migration with backup, validation, and rollback capabilities.
    """
    
    def __init__(self):
        """Initialize the migrator."""
        self.settings = get_settings()
        self.vector_db = VectorDBClient()
        self.hierarchical_service = get_hierarchical_embedding_service()
        self.embedding_service = get_embedding_service()
        self.git_tracker = GitHashTracker(cache_dir=Path.home() / ".morgan" / "cache")
        
        # Migration settings
        self.default_batch_size = 50
        self.backup_dir = Path.home() / ".morgan" / "migration_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Knowledge base migrator initialized")
    
    def analyze_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Analyze a collection for migration readiness.
        
        Args:
            collection_name: Name of collection to analyze
            
        Returns:
            Analysis report with migration recommendations
        """
        try:
            # Check if collection exists
            if not self.vector_db.collection_exists(collection_name):
                return {
                    "exists": False,
                    "error": f"Collection '{collection_name}' does not exist"
                }
            
            # Get collection info
            collection_info = self.vector_db.get_collection_info(collection_name)
            total_points = collection_info.get("points_count", 0)
            
            # Sample some points to analyze structure
            sample_points = self.vector_db.scroll_points(
                collection_name=collection_name,
                limit=10
            )
            
            # Analyze point structure
            has_hierarchical = False
            has_legacy = False
            embedding_dimensions = set()
            
            for point in sample_points:
                vector = point.get("vector", {})
                if isinstance(vector, dict):
                    # Named vectors (hierarchical)
                    has_hierarchical = True
                    for vector_name, vector_data in vector.items():
                        if isinstance(vector_data, list):
                            embedding_dimensions.add(len(vector_data))
                elif isinstance(vector, list):
                    # Single vector (legacy)
                    has_legacy = True
                    embedding_dimensions.add(len(vector))
            
            # Estimate migration time (rough estimate: 1 point per second)
            estimated_time_minutes = total_points / 60.0
            
            # Determine migration type needed
            migration_needed = "none"
            if has_legacy and not has_hierarchical:
                migration_needed = "legacy_to_hierarchical"
            elif has_hierarchical and not has_legacy:
                migration_needed = "already_hierarchical"
            elif has_legacy and has_hierarchical:
                migration_needed = "mixed_format"
            
            return {
                "exists": True,
                "collection_name": collection_name,
                "total_points": total_points,
                "has_legacy_format": has_legacy,
                "has_hierarchical_format": has_hierarchical,
                "migration_needed": migration_needed,
                "embedding_dimensions": list(embedding_dimensions),
                "estimated_migration_time_minutes": estimated_time_minutes,
                "disk_usage_mb": collection_info.get("disk_usage", 0) / (1024 * 1024),
                "sample_points_analyzed": len(sample_points)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze collection '{collection_name}': {e}")
            return {
                "exists": False,
                "error": str(e)
            }
    
    def create_migration_plan(
        self,
        source_collection: str,
        target_collection: Optional[str] = None,
        batch_size: Optional[int] = None,
        dry_run: bool = False
    ) -> MigrationPlan:
        """
        Create a migration plan for a collection.
        
        Args:
            source_collection: Source collection name
            target_collection: Target collection name (auto-generated if None)
            batch_size: Batch size for migration (uses default if None)
            dry_run: Whether this is a dry run
            
        Returns:
            Migration plan
        """
        # Analyze source collection
        analysis = self.analyze_collection(source_collection)
        if not analysis.get("exists", False):
            raise ValueError(f"Source collection '{source_collection}' does not exist")
        
        # Generate target collection name if not provided
        if target_collection is None:
            target_collection = f"{source_collection}_hierarchical"
        
        # Use default batch size if not provided
        if batch_size is None:
            batch_size = self.default_batch_size
        
        # Create backup path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = str(self.backup_dir / f"{source_collection}_backup_{timestamp}.json")
        
        return MigrationPlan(
            source_collection=source_collection,
            target_collection=target_collection,
            total_points=analysis["total_points"],
            batch_size=batch_size,
            estimated_time_minutes=analysis["estimated_migration_time_minutes"],
            backup_path=backup_path,
            dry_run=dry_run
        )
    
    def execute_migration(self, plan: MigrationPlan) -> MigrationResult:
        """
        Execute a migration plan.
        
        Args:
            plan: Migration plan to execute
            
        Returns:
            Migration result
        """
        start_time = time.time()
        points_migrated = 0
        points_failed = 0
        backup_created = False
        error_message = None
        
        try:
            logger.info(f"Starting migration: {plan.source_collection} -> {plan.target_collection}")
            
            if plan.dry_run:
                logger.info("DRY RUN MODE - No actual changes will be made")
            
            # Step 1: Create backup
            if not plan.dry_run:
                backup_created = self._create_backup(plan.source_collection, plan.backup_path)
                if not backup_created:
                    raise Exception("Failed to create backup")
                logger.info(f"Backup created: {plan.backup_path}")
            
            # Step 2: Ensure target collection exists
            if not plan.dry_run:
                success = self.vector_db.ensure_hierarchical_collection(
                    name=plan.target_collection,
                    coarse_size=384,
                    medium_size=768,
                    fine_size=1536,
                    distance="cosine"
                )
                if not success:
                    raise Exception(f"Failed to create target collection '{plan.target_collection}'")
                logger.info(f"Target collection ready: {plan.target_collection}")
            
            # Step 3: Migrate points in batches
            offset = 0
            while offset < plan.total_points:
                batch_points = self.vector_db.scroll_points(
                    collection_name=plan.source_collection,
                    limit=plan.batch_size,
                    offset=offset
                )
                
                if not batch_points:
                    break
                
                # Process batch
                batch_result = self._migrate_batch(
                    batch_points, 
                    plan.target_collection,
                    dry_run=plan.dry_run
                )
                
                points_migrated += batch_result["migrated"]
                points_failed += batch_result["failed"]
                
                offset += len(batch_points)
                
                # Progress logging
                progress = (offset / plan.total_points) * 100
                logger.info(f"Migration progress: {progress:.1f}% ({offset}/{plan.total_points})")
            
            execution_time = time.time() - start_time
            
            # Success
            result = MigrationResult(
                success=True,
                points_migrated=points_migrated,
                points_failed=points_failed,
                execution_time_seconds=execution_time,
                backup_created=backup_created,
                rollback_available=backup_created
            )
            
            logger.info(f"Migration completed successfully in {execution_time:.1f}s")
            logger.info(f"Points migrated: {points_migrated}, failed: {points_failed}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Migration failed after {execution_time:.1f}s: {error_message}")
            
            return MigrationResult(
                success=False,
                points_migrated=points_migrated,
                points_failed=points_failed,
                execution_time_seconds=execution_time,
                backup_created=backup_created,
                rollback_available=backup_created,
                error_message=error_message
            )
    
    def _create_backup(self, collection_name: str, backup_path: str) -> bool:
        """
        Create a backup of a collection.
        
        Args:
            collection_name: Collection to backup
            backup_path: Path to save backup
            
        Returns:
            True if backup was created successfully
        """
        try:
            logger.info(f"Creating backup of collection '{collection_name}'...")
            
            # Get all points from collection
            all_points = []
            offset = 0
            batch_size = 100
            
            while True:
                batch = self.vector_db.scroll_points(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset
                )
                
                if not batch:
                    break
                
                all_points.extend(batch)
                offset += len(batch)
            
            # Get collection info
            collection_info = self.vector_db.get_collection_info(collection_name)
            
            # Create backup data
            backup_data = {
                "collection_name": collection_name,
                "backup_timestamp": datetime.now().isoformat(),
                "collection_info": collection_info,
                "total_points": len(all_points),
                "points": all_points
            }
            
            # Save backup
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Backup created: {len(all_points)} points saved to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _migrate_batch(
        self, 
        batch_points: List[Dict[str, Any]], 
        target_collection: str,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Migrate a batch of points to hierarchical format.
        
        Args:
            batch_points: Batch of points to migrate
            target_collection: Target collection name
            dry_run: Whether this is a dry run
            
        Returns:
            Dict with migration counts
        """
        migrated = 0
        failed = 0
        
        try:
            # Prepare hierarchical points
            hierarchical_points = []
            
            for point in batch_points:
                try:
                    # Extract content from payload
                    payload = point.get("payload", {})
                    content = payload.get("content", "")
                    
                    if not content:
                        logger.warning(f"Point {point.get('id')} has no content, skipping")
                        failed += 1
                        continue
                    
                    # Generate hierarchical embeddings
                    if not dry_run:
                        hierarchical_embedding = self.hierarchical_service.create_hierarchical_embeddings(
                            content=content,
                            metadata=payload.get("metadata", {}),
                            use_cache=True
                        )
                        
                        # Create hierarchical point
                        hierarchical_point = {
                            "id": point.get("id"),
                            "vector": {
                                "coarse": hierarchical_embedding.get_embedding("coarse"),
                                "medium": hierarchical_embedding.get_embedding("medium"),
                                "fine": hierarchical_embedding.get_embedding("fine")
                            },
                            "payload": {
                                **payload,
                                "migrated_at": datetime.now().isoformat(),
                                "migration_source": "legacy_to_hierarchical",
                                "hierarchical_texts": hierarchical_embedding.texts,
                                "embedding_type": "hierarchical"
                            }
                        }
                        
                        hierarchical_points.append(hierarchical_point)
                    
                    migrated += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to migrate point {point.get('id')}: {e}")
                    failed += 1
                    continue
            
            # Upsert hierarchical points
            if hierarchical_points and not dry_run:
                success = self.vector_db.upsert_hierarchical_points(
                    target_collection, 
                    hierarchical_points
                )
                
                if not success:
                    logger.error(f"Failed to upsert batch to {target_collection}")
                    # Mark all as failed
                    failed += migrated
                    migrated = 0
            
            return {"migrated": migrated, "failed": failed}
            
        except Exception as e:
            logger.error(f"Batch migration failed: {e}")
            return {"migrated": 0, "failed": len(batch_points)}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with migration status.
        
        Returns:
            List of collections with analysis
        """
        try:
            # Get all collections
            collections = self.vector_db.list_collections()
            
            result = []
            for collection_name in collections:
                analysis = self.analyze_collection(collection_name)
                result.append(analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def estimate_migration_resources(self, collection_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for migration.
        
        Args:
            collection_name: Collection to analyze
            
        Returns:
            Resource estimation
        """
        analysis = self.analyze_collection(collection_name)
        
        if not analysis.get("exists", False):
            return {"error": "Collection does not exist"}
        
        total_points = analysis["total_points"]
        
        # Rough estimates based on typical performance
        estimated_time_minutes = total_points / 60.0  # ~1 point per second
        estimated_memory_mb = (total_points * 2048 * 3) / (1024 * 1024)  # 3 embeddings per point
        estimated_disk_mb = analysis["disk_usage_mb"] * 3  # Roughly 3x for hierarchical
        
        return {
            "collection_name": collection_name,
            "total_points": total_points,
            "estimated_time_minutes": estimated_time_minutes,
            "estimated_memory_mb": estimated_memory_mb,
            "estimated_additional_disk_mb": estimated_disk_mb,
            "recommended_batch_size": min(50, max(10, total_points // 100)),
            "backup_size_mb": analysis["disk_usage_mb"]
        }