# Morgan Knowledge Base Migration Guide

This guide covers migrating existing Morgan knowledge bases from legacy single-vector embeddings to hierarchical multi-scale embeddings.

## Overview

The migration system provides safe, validated migration with backup and rollback capabilities. It implements requirements R10.4 and R10.5 for migration safety.

### Key Features

- **Safe Migration**: Automatic backup creation before migration
- **Validation**: Comprehensive validation of migrated data
- **Rollback**: Full rollback capability using backups
- **Dry Run**: Test migrations without making changes
- **Batch Processing**: Configurable batch sizes for large collections
- **Progress Tracking**: Real-time progress monitoring

## Migration Process

### 1. Analysis Phase

Before migrating, analyze your collections to understand their current state:

```bash
# Analyze all collections
morgan migrate analyze

# Analyze specific collection
morgan migrate analyze my_collection
```

The analysis will show:
- Total number of points
- Current format (legacy/hierarchical/mixed)
- Migration requirements
- Estimated migration time
- Disk usage

### 2. Planning Phase

Create a migration plan to review before execution:

```bash
# Create migration plan
morgan migrate plan my_collection

# Specify custom target collection
morgan migrate plan my_collection --target my_collection_v2

# Adjust batch size for performance
morgan migrate plan my_collection --batch-size 100
```

### 3. Execution Phase

Execute the migration with safety checks:

```bash
# Dry run (recommended first)
morgan migrate execute my_collection --dry-run

# Execute migration (requires confirmation)
morgan migrate execute my_collection --confirm

# Custom target and batch size
morgan migrate execute my_collection --target my_collection_v2 --batch-size 100 --confirm
```

### 4. Validation Phase

Validate the migration completed successfully:

```bash
# Validate migration
morgan migrate validate my_collection my_collection_hierarchical

# Custom sample size for validation
morgan migrate validate my_collection my_collection_hierarchical --sample-size 200
```

### 5. Rollback (if needed)

If issues are found, rollback using the automatic backup:

```bash
# List available backups
morgan migrate list-backups

# Rollback using backup
morgan migrate rollback /path/to/backup.json --confirm

# Rollback to different collection name
morgan migrate rollback /path/to/backup.json --target restored_collection --confirm
```

## Standalone Script Usage

For environments where the full Morgan CLI is not available, use the standalone script:

```bash
# Navigate to morgan-rag directory
cd morgan-rag

# Analyze collections
python scripts/migrate_knowledge_base.py analyze

# Execute migration
python scripts/migrate_knowledge_base.py migrate my_collection --confirm

# Validate migration
python scripts/migrate_knowledge_base.py validate my_collection my_collection_hierarchical

# List backups
python scripts/migrate_knowledge_base.py list-backups

# Rollback
python scripts/migrate_knowledge_base.py rollback backup.json --confirm
```

## Migration Safety Features

### Automatic Backups

Every migration automatically creates a backup before making changes:

- Backup location: `~/.morgan/migration_backups/`
- Backup format: JSON with complete collection data
- Backup naming: `{collection}_backup_{timestamp}.json`

### Validation Checks

The migration system performs multiple validation checks:

1. **Pre-migration**: Collection existence, format analysis
2. **During migration**: Point structure validation, embedding generation
3. **Post-migration**: Data integrity, point count verification
4. **Backup validation**: Backup file integrity and completeness

### Rollback Capabilities

Complete rollback support:

- Restore from any backup file
- Validate backup before rollback
- Overwrite protection with confirmation
- Point-by-point restoration with progress tracking

## Performance Considerations

### Batch Size Tuning

Adjust batch size based on your system:

- **Small systems**: 10-25 points per batch
- **Medium systems**: 50-100 points per batch  
- **Large systems**: 100-200 points per batch

### Memory Usage

Hierarchical embeddings use approximately 3x more memory:

- **Coarse**: 384 dimensions
- **Medium**: 768 dimensions
- **Fine**: 1536 dimensions

### Disk Space

Plan for additional disk space:

- **Backup**: Same size as original collection
- **Target collection**: ~3x size of original
- **Temporary space**: ~1x size during migration

## Troubleshooting

### Common Issues

#### Migration Fails with Memory Error

```bash
# Reduce batch size
morgan migrate execute my_collection --batch-size 10 --confirm
```

#### Validation Shows Errors

```bash
# Check specific validation errors
morgan migrate validate my_collection my_collection_hierarchical --sample-size 50

# If errors are minor, migration may still be usable
# If errors are major, consider rollback and retry
```

#### Backup File Corrupted

```bash
# List all available backups
morgan migrate list-backups

# Use an earlier backup if available
morgan migrate rollback /path/to/earlier_backup.json --confirm
```

### Recovery Procedures

#### Partial Migration Failure

1. Check if backup was created
2. Validate what was migrated
3. Decide whether to rollback or continue
4. If continuing, re-run migration (it will skip existing points)

#### Complete Migration Failure

1. Use rollback to restore original collection
2. Analyze the failure cause
3. Adjust parameters (batch size, memory limits)
4. Retry migration

#### Backup Restoration

```bash
# Validate backup before restoration
python scripts/migrate_knowledge_base.py rollback backup.json --dry-run

# Restore to original collection
python scripts/migrate_knowledge_base.py rollback backup.json --confirm

# Restore to new collection
python scripts/migrate_knowledge_base.py rollback backup.json --target restored_collection --confirm
```

## Best Practices

### Before Migration

1. **Analyze first**: Always run analysis to understand your data
2. **Test with dry run**: Use `--dry-run` to test the process
3. **Check disk space**: Ensure sufficient space for backups and new collections
4. **Plan downtime**: Migration can take time for large collections

### During Migration

1. **Monitor progress**: Watch for errors or performance issues
2. **Don't interrupt**: Let migration complete to avoid partial state
3. **Check system resources**: Monitor memory and disk usage

### After Migration

1. **Validate immediately**: Run validation right after migration
2. **Test functionality**: Verify search and retrieval work correctly
3. **Keep backups**: Don't delete backups until confident in migration
4. **Update applications**: Point applications to new hierarchical collections

### Maintenance

1. **Regular cleanup**: Clean old backups periodically
2. **Monitor performance**: Compare search performance before/after
3. **Document changes**: Keep records of what was migrated when

## Migration Checklist

### Pre-Migration

- [ ] Analyze all collections to be migrated
- [ ] Verify sufficient disk space (4x original size)
- [ ] Test migration with dry run
- [ ] Plan for downtime if needed
- [ ] Backup any critical configurations

### Migration

- [ ] Execute migration with confirmation
- [ ] Monitor progress and system resources
- [ ] Verify backup creation
- [ ] Check for any error messages

### Post-Migration

- [ ] Validate migration results
- [ ] Test search functionality
- [ ] Update application configurations
- [ ] Document migration completion
- [ ] Schedule backup cleanup

### Rollback (if needed)

- [ ] Identify appropriate backup file
- [ ] Validate backup integrity
- [ ] Execute rollback with confirmation
- [ ] Verify restoration success
- [ ] Update application configurations

## API Reference

### Migration Classes

#### KnowledgeBaseMigrator

Main migration orchestrator:

```python
from morgan.migration import KnowledgeBaseMigrator

migrator = KnowledgeBaseMigrator()

# Analyze collection
analysis = migrator.analyze_collection("my_collection")

# Create migration plan
plan = migrator.create_migration_plan("my_collection")

# Execute migration
result = migrator.execute_migration(plan)
```

#### MigrationValidator

Validation and verification:

```python
from morgan.migration import MigrationValidator

validator = MigrationValidator()

# Validate migration
result = validator.validate_migration("source", "target")

# Compare collections
comparison = validator.compare_collections("col1", "col2")
```

#### RollbackManager

Backup and rollback operations:

```python
from morgan.migration import RollbackManager

rollback = RollbackManager()

# List backups
backups = rollback.list_available_backups()

# Execute rollback
result = rollback.execute_rollback("backup.json")
```

## Support

For migration issues:

1. Check this guide for common solutions
2. Review migration logs for specific errors
3. Use validation tools to diagnose problems
4. Consider rollback if migration is problematic

Remember: Migration is a one-way process without rollback, so always test thoroughly and maintain backups.