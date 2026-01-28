#!/usr/bin/env python3
"""
Standalone Knowledge Base Migration Script

Provides command-line migration tools for Morgan knowledge bases.
Can be run independently of the main Morgan CLI.

Implements requirements R10.4 and R10.5.

Usage:
    python migrate_knowledge_base.py analyze [collection_name]
    python migrate_knowledge_base.py migrate <source> [--target <target>] [--dry-run]
    python migrate_knowledge_base.py validate <source> <target>
    python migrate_knowledge_base.py rollback <backup_path> [--target <target>]
    python migrate_knowledge_base.py list-backups
    python migrate_knowledge_base.py cleanup [--keep-days <days>]
"""

import sys
import argparse
from pathlib import Path

# Add morgan-rag to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.migration import KnowledgeBaseMigrator, MigrationValidator, RollbackManager
from morgan.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Morgan Knowledge Base Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_knowledge_base.py analyze                    # Analyze all collections
  python migrate_knowledge_base.py analyze my_collection      # Analyze specific collection
  python migrate_knowledge_base.py migrate my_collection --dry-run  # Test migration
  python migrate_knowledge_base.py migrate my_collection --confirm   # Execute migration
  python migrate_knowledge_base.py validate source target     # Validate migration
  python migrate_knowledge_base.py rollback backup.json --confirm    # Rollback migration
  python migrate_knowledge_base.py list-backups               # List available backups
  python migrate_knowledge_base.py cleanup --keep-days 7      # Clean old backups
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="action", help="Migration actions")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze collections for migration readiness"
    )
    analyze_parser.add_argument(
        "collection",
        nargs="?",
        help="Collection name to analyze (analyzes all if not specified)",
    )

    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate collection to hierarchical format"
    )
    migrate_parser.add_argument("source_collection", help="Source collection name")
    migrate_parser.add_argument(
        "--target", help="Target collection name (auto-generated if not specified)"
    )
    migrate_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for migration (default: 50)",
    )
    migrate_parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without making changes"
    )
    migrate_parser.add_argument(
        "--confirm", action="store_true", help="Confirm migration execution"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate completed migration"
    )
    validate_parser.add_argument("source_collection", help="Source collection name")
    validate_parser.add_argument("target_collection", help="Target collection name")
    validate_parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of points to sample for validation (default: 100)",
    )

    # Rollback command
    rollback_parser = subparsers.add_parser(
        "rollback", help="Rollback migration using backup"
    )
    rollback_parser.add_argument("backup_path", help="Path to backup file")
    rollback_parser.add_argument(
        "--target",
        help="Target collection name (uses backup collection name if not specified)",
    )
    rollback_parser.add_argument(
        "--confirm", action="store_true", help="Confirm rollback execution"
    )

    # List backups command
    list_parser = subparsers.add_parser(
        "list-backups", help="List available migration backups"
    )

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Clean up old migration backups"
    )
    cleanup_parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Number of days to keep backups (default: 30)",
    )

    return parser


def cmd_analyze(args):
    """Handle analyze command."""
    migrator = KnowledgeBaseMigrator()

    if args.collection:
        print(f"Analyzing collection '{args.collection}'...")
        analysis = migrator.analyze_collection(args.collection)

        if not analysis.get("exists", False):
            print(f"Error: {analysis.get('error', 'Collection not found')}")
            return 1

        print(f"\nCollection: {args.collection}")
        print(f"Total Points: {analysis['total_points']}")
        print(f"Has Legacy Format: {analysis['has_legacy_format']}")
        print(f"Has Hierarchical Format: {analysis['has_hierarchical_format']}")
        print(f"Migration Needed: {analysis['migration_needed']}")
        print(
            f"Estimated Time: {analysis['estimated_migration_time_minutes']:.1f} minutes"
        )
        print(f"Disk Usage: {analysis['disk_usage_mb']:.1f} MB")

        if analysis["migration_needed"] == "legacy_to_hierarchical":
            print(
                "\nRecommendation: This collection can be migrated to hierarchical format"
            )
        elif analysis["migration_needed"] == "already_hierarchical":
            print("\nThis collection is already in hierarchical format")
        elif analysis["migration_needed"] == "mixed_format":
            print("\nWarning: This collection has mixed formats and may need cleanup")

    else:
        print("Analyzing all collections...")
        collections = migrator.list_collections()

        if not collections:
            print("No collections found")
            return 0

        print(
            f"\n{'Collection':<30} {'Points':<10} {'Format':<20} {'Migration Needed':<20} {'Est. Time (min)':<15}"
        )
        print("-" * 95)

        for analysis in collections:
            if analysis.get("exists", False):
                format_str = ""
                if analysis["has_legacy_format"]:
                    format_str += "Legacy"
                if analysis["has_hierarchical_format"]:
                    format_str += " + Hierarchical" if format_str else "Hierarchical"

                print(
                    f"{analysis['collection_name']:<30} {analysis['total_points']:<10} {format_str:<20} {analysis['migration_needed']:<20} {analysis['estimated_migration_time_minutes']:<15.1f}"
                )

    return 0


def cmd_migrate(args):
    """Handle migrate command."""
    if not args.confirm and not args.dry_run:
        print("Error: Migration requires --confirm flag or --dry-run for safety")
        print("Use --dry-run to test without making changes")
        return 1

    migrator = KnowledgeBaseMigrator()

    print(
        f"{'Dry run' if args.dry_run else 'Executing'} migration for '{args.source_collection}'..."
    )

    try:
        # Create plan
        plan = migrator.create_migration_plan(
            source_collection=args.source_collection,
            target_collection=args.target,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

        print(f"Source: {plan.source_collection} -> Target: {plan.target_collection}")
        print(f"Points to migrate: {plan.total_points}")
        print(f"Batch size: {plan.batch_size}")
        print(f"Estimated time: {plan.estimated_time_minutes:.1f} minutes")

        if not args.dry_run:
            print(f"Backup will be created at: {plan.backup_path}")

        # Execute migration
        result = migrator.execute_migration(plan)

        if result.success:
            print(
                f"\nMigration {'simulation' if args.dry_run else 'completed'} successfully!"
            )
            print(f"Points migrated: {result.points_migrated}")
            if result.points_failed > 0:
                print(f"Points failed: {result.points_failed}")
            print(f"Execution time: {result.execution_time_seconds:.1f}s")

            if result.backup_created:
                print("Backup created successfully")

            if not args.dry_run:
                print(
                    f"\nUse 'python {sys.argv[0]} validate {plan.source_collection} {plan.target_collection}' to verify"
                )

            return 0
        else:
            print(f"Migration failed: {result.error_message}")
            if result.backup_created:
                print("Backup was created and can be used for rollback")
            return 1

    except Exception as e:
        print(f"Migration execution failed: {e}")
        return 1


def cmd_validate(args):
    """Handle validate command."""
    validator = MigrationValidator()

    print(
        f"Validating migration: {args.source_collection} -> {args.target_collection}..."
    )

    result = validator.validate_migration(
        args.source_collection, args.target_collection, sample_size=args.sample_size
    )

    if result.is_valid:
        print("Migration validation passed!")
    else:
        print("Migration validation failed!")

    print(f"\nValidation Results:")
    print(f"Points Checked: {result.total_points_checked}")
    print(f"Valid Points: {result.points_valid}")
    print(f"Invalid Points: {result.points_invalid}")
    if result.total_points_checked > 0:
        print(
            f"Validation Rate: {(result.points_valid / result.total_points_checked * 100):.1f}%"
        )

    if result.validation_errors:
        print(f"\nValidation Errors ({len(result.validation_errors)}):")
        for error in result.validation_errors[:10]:  # Show first 10 errors
            print(f"  • {error}")
        if len(result.validation_errors) > 10:
            print(f"  ... and {len(result.validation_errors) - 10} more errors")

    return 0 if result.is_valid else 1


def cmd_rollback(args):
    """Handle rollback command."""
    if not args.confirm:
        print("Error: Rollback requires --confirm flag for safety")
        return 1

    rollback_manager = RollbackManager()

    print(f"Executing rollback from {args.backup_path}...")

    # Validate backup first
    validation = rollback_manager.validate_backup_for_rollback(args.backup_path)
    if not validation["is_valid"]:
        print(f"Backup validation failed: {validation.get('error', 'Unknown error')}")
        return 1

    # Show warnings
    if validation.get("warnings"):
        for warning in validation["warnings"]:
            print(f"Warning: {warning}")

    # Execute rollback
    result = rollback_manager.execute_rollback(
        args.backup_path, target_collection=args.target, confirm_overwrite=True
    )

    if result.success:
        print("Rollback completed successfully!")
        print(f"Points restored: {result.points_restored}")
        print(f"Collection: {result.collection_restored}")
        print(f"Execution time: {result.execution_time_seconds:.1f}s")
        return 0
    else:
        print(f"Rollback failed: {result.error_message}")
        return 1


def cmd_list_backups(args):
    """Handle list-backups command."""
    rollback_manager = RollbackManager()

    print("Available migration backups:")
    backups = rollback_manager.list_available_backups()

    if not backups:
        print("No migration backups found")
        return 0

    print(
        f"\n{'File Name':<40} {'Collection':<20} {'Points':<10} {'Size (MB)':<12} {'Created':<20}"
    )
    print("-" * 102)

    for backup in backups:
        print(
            f"{backup['file_name']:<40} {backup['collection_name']:<20} {backup['total_points']:<10} {backup['file_size_mb']:<12.1f} {backup['backup_timestamp'][:19]}"
        )

    return 0


def cmd_cleanup(args):
    """Handle cleanup command."""
    rollback_manager = RollbackManager()

    print(f"Cleaning up backups older than {args.keep_days} days...")
    result = rollback_manager.cleanup_old_backups(keep_days=args.keep_days)

    if "error" in result:
        print(f"Cleanup failed: {result['error']}")
        return 1
    else:
        print(result["message"])
        return 0


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        return 1

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    try:
        if args.action == "analyze":
            return cmd_analyze(args)
        elif args.action == "migrate":
            return cmd_migrate(args)
        elif args.action == "validate":
            return cmd_validate(args)
        elif args.action == "rollback":
            return cmd_rollback(args)
        elif args.action == "list-backups":
            return cmd_list_backups(args)
        elif args.action == "cleanup":
            return cmd_cleanup(args)
        else:
            print(f"Unknown action: {args.action}")
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Migration script error: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
