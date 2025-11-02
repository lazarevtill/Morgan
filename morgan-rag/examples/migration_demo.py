#!/usr/bin/env python3
"""
Migration System Demo

Demonstrates the migration system functionality for converting
legacy knowledge bases to hierarchical format.

This example shows:
1. Collection analysis
2. Migration planning
3. Migration execution (dry run)
4. Validation
5. Backup management
"""

import sys
from pathlib import Path

# Add morgan-rag to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.migration import KnowledgeBaseMigrator, MigrationValidator, RollbackManager
from morgan.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def demo_collection_analysis():
    """Demonstrate collection analysis functionality."""
    print("=" * 60)
    print("MIGRATION DEMO: Collection Analysis")
    print("=" * 60)
    
    migrator = KnowledgeBaseMigrator()
    
    # List and analyze all collections
    print("\n1. Analyzing all collections...")
    collections = migrator.list_collections()
    
    if not collections:
        print("No collections found. Please create some knowledge bases first.")
        return False
    
    print(f"Found {len(collections)} collections:")
    print(f"{'Collection':<30} {'Points':<10} {'Format':<20} {'Migration Needed':<20}")
    print("-" * 80)
    
    migratable_collections = []
    
    for analysis in collections:
        if analysis.get("exists", False):
            format_str = ""
            if analysis["has_legacy_format"]:
                format_str += "Legacy"
            if analysis["has_hierarchical_format"]:
                format_str += " + Hierarchical" if format_str else "Hierarchical"
            
            print(f"{analysis['collection_name']:<30} {analysis['total_points']:<10} {format_str:<20} {analysis['migration_needed']:<20}")
            
            if analysis["migration_needed"] == "legacy_to_hierarchical":
                migratable_collections.append(analysis["collection_name"])
    
    if migratable_collections:
        print(f"\nCollections that can be migrated: {', '.join(migratable_collections)}")
        return migratable_collections[0]  # Return first migratable collection
    else:
        print("\nNo collections need migration to hierarchical format.")
        return None


def demo_migration_planning(collection_name):
    """Demonstrate migration planning."""
    print("\n" + "=" * 60)
    print("MIGRATION DEMO: Migration Planning")
    print("=" * 60)
    
    migrator = KnowledgeBaseMigrator()
    
    print(f"\n2. Creating migration plan for '{collection_name}'...")
    
    try:
        plan = migrator.create_migration_plan(
            source_collection=collection_name,
            batch_size=25
        )
        
        print(f"Migration Plan:")
        print(f"  Source Collection: {plan.source_collection}")
        print(f"  Target Collection: {plan.target_collection}")
        print(f"  Total Points: {plan.total_points}")
        print(f"  Batch Size: {plan.batch_size}")
        print(f"  Estimated Time: {plan.estimated_time_minutes:.1f} minutes")
        print(f"  Backup Path: {plan.backup_path}")
        
        return plan
        
    except Exception as e:
        print(f"Failed to create migration plan: {e}")
        return None


def demo_dry_run_migration(plan):
    """Demonstrate dry run migration."""
    print("\n" + "=" * 60)
    print("MIGRATION DEMO: Dry Run Migration")
    print("=" * 60)
    
    migrator = KnowledgeBaseMigrator()
    
    print(f"\n3. Executing dry run migration...")
    print("This will simulate the migration without making any changes.")
    
    # Set dry run flag
    plan.dry_run = True
    
    try:
        result = migrator.execute_migration(plan)
        
        if result.success:
            print(f"✅ Dry run migration completed successfully!")
            print(f"  Points that would be migrated: {result.points_migrated}")
            print(f"  Points that would fail: {result.points_failed}")
            print(f"  Execution time: {result.execution_time_seconds:.1f}s")
            print(f"  Backup would be created: {result.backup_created}")
        else:
            print(f"❌ Dry run migration failed: {result.error_message}")
            
        return result.success
        
    except Exception as e:
        print(f"Dry run migration failed: {e}")
        return False


def demo_validation():
    """Demonstrate validation functionality."""
    print("\n" + "=" * 60)
    print("MIGRATION DEMO: Validation")
    print("=" * 60)
    
    validator = MigrationValidator()
    
    print(f"\n4. Demonstrating validation capabilities...")
    
    # Compare collections (if any hierarchical collections exist)
    migrator = KnowledgeBaseMigrator()
    collections = migrator.list_collections()
    
    hierarchical_collections = [
        c["collection_name"] for c in collections 
        if c.get("exists") and c.get("has_hierarchical_format")
    ]
    
    if len(hierarchical_collections) >= 2:
        col1, col2 = hierarchical_collections[:2]
        print(f"Comparing collections '{col1}' and '{col2}'...")
        
        try:
            comparison = validator.compare_collections(col1, col2, sample_size=10)
            
            print(f"Comparison Results:")
            print(f"  Collection 1: {comparison['collection1']['total_points']} points")
            print(f"  Collection 2: {comparison['collection2']['total_points']} points")
            print(f"  Common points: {comparison['comparison']['common_points']}")
            print(f"  Similarity ratio: {comparison['comparison']['similarity_ratio']:.2f}")
            
        except Exception as e:
            print(f"Collection comparison failed: {e}")
    else:
        print("Not enough hierarchical collections for comparison demo.")


def demo_backup_management():
    """Demonstrate backup management."""
    print("\n" + "=" * 60)
    print("MIGRATION DEMO: Backup Management")
    print("=" * 60)
    
    rollback_manager = RollbackManager()
    
    print(f"\n5. Demonstrating backup management...")
    
    # List available backups
    backups = rollback_manager.list_available_backups()
    
    if backups:
        print(f"Found {len(backups)} backup files:")
        print(f"{'File Name':<40} {'Collection':<20} {'Points':<10} {'Size (MB)':<12}")
        print("-" * 82)
        
        for backup in backups[:5]:  # Show first 5 backups
            print(f"{backup['file_name']:<40} {backup['collection_name']:<20} {backup['total_points']:<10} {backup['file_size_mb']:<12.1f}")
        
        if len(backups) > 5:
            print(f"... and {len(backups) - 5} more backups")
        
        # Validate a backup
        print(f"\nValidating backup: {backups[0]['file_name']}")
        validation = rollback_manager.validate_backup_for_rollback(backups[0]['file_path'])
        
        if validation["is_valid"]:
            print("✅ Backup is valid and can be used for rollback")
            print(f"  Collection: {validation['collection_name']}")
            print(f"  Points: {validation['total_points']}")
            print(f"  Validation rate: {validation['sample_validation_rate']:.1%}")
        else:
            print(f"❌ Backup validation failed: {validation.get('error', 'Unknown error')}")
    
    else:
        print("No backup files found.")
        print("Backups are created automatically during migration.")


def demo_resource_estimation():
    """Demonstrate resource estimation."""
    print("\n" + "=" * 60)
    print("MIGRATION DEMO: Resource Estimation")
    print("=" * 60)
    
    migrator = KnowledgeBaseMigrator()
    collections = migrator.list_collections()
    
    print(f"\n6. Resource estimation for migration...")
    
    for analysis in collections[:3]:  # Show first 3 collections
        if analysis.get("exists", False):
            collection_name = analysis["collection_name"]
            
            try:
                estimation = migrator.estimate_migration_resources(collection_name)
                
                if "error" not in estimation:
                    print(f"\nCollection: {collection_name}")
                    print(f"  Total points: {estimation['total_points']}")
                    print(f"  Estimated time: {estimation['estimated_time_minutes']:.1f} minutes")
                    print(f"  Estimated memory: {estimation['estimated_memory_mb']:.1f} MB")
                    print(f"  Additional disk space: {estimation['estimated_additional_disk_mb']:.1f} MB")
                    print(f"  Recommended batch size: {estimation['recommended_batch_size']}")
                    print(f"  Backup size: {estimation['backup_size_mb']:.1f} MB")
                
            except Exception as e:
                print(f"Resource estimation failed for {collection_name}: {e}")


def main():
    """Run the migration demo."""
    print("🔄 Morgan Knowledge Base Migration System Demo")
    print("This demo showcases the migration system capabilities.")
    print("Note: This demo uses read-only operations and dry runs for safety.")
    
    try:
        # 1. Collection Analysis
        migratable_collection = demo_collection_analysis()
        
        if migratable_collection:
            # 2. Migration Planning
            plan = demo_migration_planning(migratable_collection)
            
            if plan:
                # 3. Dry Run Migration
                demo_dry_run_migration(plan)
        
        # 4. Validation Demo
        demo_validation()
        
        # 5. Backup Management
        demo_backup_management()
        
        # 6. Resource Estimation
        demo_resource_estimation()
        
        print("\n" + "=" * 60)
        print("MIGRATION DEMO COMPLETED")
        print("=" * 60)
        print("\nTo perform actual migrations:")
        print("1. Use 'morgan migrate' commands from the CLI")
        print("2. Use the standalone script: python scripts/migrate_knowledge_base.py")
        print("3. Always test with --dry-run first")
        print("4. Use --confirm for actual execution")
        print("\nSee docs/MIGRATION_GUIDE.md for detailed instructions.")
        
    except KeyboardInterrupt:
        print("\nDemo cancelled by user.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()