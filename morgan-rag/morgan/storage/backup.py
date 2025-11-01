"""
Backup Storage - Data backup and recovery operations

Provides backup and recovery for all Morgan data components.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.4, 23.5
"""

from typing import Dict, Any, List, Optional
import logging
import json
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupStorage:
    """
    Backup storage following KISS principles.
    
    Single responsibility: Manage data backup and recovery operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.backup_dir = Path(self.config.get('backup_dir', './data/backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup retention settings
        self.max_backups = self.config.get('max_backups', 10)
        
    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive backup.
        
        Args:
            backup_name: Optional backup name (defaults to timestamp)
            
        Returns:
            Backup result information
        """
        try:
            # Generate backup name if not provided
            if not backup_name:
                backup_name = f"morgan_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            backup_info = {
                'name': backup_name,
                'created_at': datetime.now().isoformat(),
                'components': [],
                'success': True,
                'errors': []
            }
            
            # Backup memory storage
            try:
                self._backup_memory_storage(backup_path)
                backup_info['components'].append('memory')
            except Exception as e:
                backup_info['errors'].append(f"Memory backup failed: {e}")
                
            # Backup user profiles
            try:
                self._backup_profile_storage(backup_path)
                backup_info['components'].append('profiles')
            except Exception as e:
                backup_info['errors'].append(f"Profile backup failed: {e}")
                
            # Save backup metadata
            backup_info_file = backup_path / 'backup_info.json'
            with open(backup_info_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Backup created: {backup_name}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return {
                'name': backup_name or 'unknown',
                'success': False,
                'error': str(e)
            }
            
    def _backup_memory_storage(self, backup_path: Path) -> None:
        """Backup memory storage files."""
        memory_backup_path = backup_path / 'memory'
        memory_backup_path.mkdir(exist_ok=True)
        
        # Define memory storage paths
        memory_paths = [
            './data/memory/memories.jsonl',
            './data/memory/emotions.jsonl',
            './data/memory/profiles.json'
        ]
        
        for memory_file in memory_paths:
            source_path = Path(memory_file)
            if source_path.exists():
                dest_path = memory_backup_path / source_path.name
                shutil.copy2(source_path, dest_path)
                
    def _backup_profile_storage(self, backup_path: Path) -> None:
        """Backup profile storage files."""
        profile_backup_path = backup_path / 'profiles'
        profile_backup_path.mkdir(exist_ok=True)
        
        # Define profile storage paths
        profile_paths = [
            './data/profiles/companion_profiles.json',
            './data/profiles/user_preferences.json',
            './data/profiles/milestones.jsonl'
        ]
        
        for profile_file in profile_paths:
            source_path = Path(profile_file)
            if source_path.exists():
                dest_path = profile_backup_path / source_path.name
                shutil.copy2(source_path, dest_path)
                
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        try:
            backups = []
            
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    info_file = backup_dir / 'backup_info.json'
                    if info_file.exists():
                        with open(info_file, 'r', encoding='utf-8') as f:
                            backup_info = json.load(f)
                            backup_info['path'] = str(backup_dir)
                            backups.append(backup_info)
                            
            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return backups
            
        except Exception as e:
            logger.error("Failed to list backups: %s", e)
            return []
            
    def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Restore from a backup.
        
        Args:
            backup_name: Name of backup to restore
            
        Returns:
            Restore result information
        """
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                return {
                    'success': False,
                    'error': f'Backup {backup_name} not found'
                }
                
            restore_info = {
                'name': backup_name,
                'restored_at': datetime.now().isoformat(),
                'components': [],
                'success': True,
                'errors': []
            }
            
            # Restore memory storage
            try:
                self._restore_memory_storage(backup_path)
                restore_info['components'].append('memory')
            except Exception as e:
                restore_info['errors'].append(f"Memory restore failed: {e}")
                
            # Restore user profiles
            try:
                self._restore_profile_storage(backup_path)
                restore_info['components'].append('profiles')
            except Exception as e:
                restore_info['errors'].append(f"Profile restore failed: {e}")
                
            logger.info("Backup restored: %s", backup_name)
            return restore_info
            
        except Exception as e:
            logger.error("Backup restore failed: %s", e)
            return {
                'name': backup_name,
                'success': False,
                'error': str(e)
            }
            
    def _restore_memory_storage(self, backup_path: Path) -> None:
        """Restore memory storage files."""
        memory_backup_path = backup_path / 'memory'
        if not memory_backup_path.exists():
            return
            
        # Ensure target directory exists
        target_dir = Path('./data/memory')
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Restore memory files
        memory_files = [
            'memories.jsonl',
            'emotions.jsonl',
            'profiles.json'
        ]
        
        for memory_file in memory_files:
            source_path = memory_backup_path / memory_file
            if source_path.exists():
                target_path = target_dir / memory_file
                shutil.copy2(source_path, target_path)
                
    def _restore_profile_storage(self, backup_path: Path) -> None:
        """Restore profile storage files."""
        profile_backup_path = backup_path / 'profiles'
        if not profile_backup_path.exists():
            return
            
        # Ensure target directory exists
        target_dir = Path('./data/profiles')
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Restore profile files
        profile_files = [
            'companion_profiles.json',
            'user_preferences.json',
            'milestones.jsonl'
        ]
        
        for profile_file in profile_files:
            source_path = profile_backup_path / profile_file
            if source_path.exists():
                target_path = target_dir / profile_file
                shutil.copy2(source_path, target_path)
                
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_name: Name of backup to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            backup_path = self.backup_dir / backup_name
            if backup_path.exists():
                shutil.rmtree(backup_path)
                logger.info("Deleted backup: %s", backup_name)
                return True
            else:
                logger.warning("Backup not found: %s", backup_name)
                return False
                
        except Exception as e:
            logger.error("Failed to delete backup %s: %s", backup_name, e)
            return False
            
    def cleanup_old_backups(self) -> int:
        """
        Clean up old backups based on retention policy.
        
        Returns:
            Number of backups deleted
        """
        try:
            backups = self.list_backups()
            if len(backups) <= self.max_backups:
                return 0
                
            # Sort by creation date and keep only the newest
            backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            backups_to_delete = backups[self.max_backups:]
            
            deleted_count = 0
            for backup in backups_to_delete:
                backup_name = backup['name']
                if self.delete_backup(backup_name):
                    deleted_count += 1
                    
            logger.info("Cleaned up %d old backups", deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old backups: %s", e)
            return 0
            
    def get_backup_stats(self) -> Dict[str, Any]:
        """
        Get backup statistics.
        
        Returns:
            Backup statistics
        """
        try:
            backups = self.list_backups()
            
            stats = {
                'total_backups': len(backups),
                'backup_dir': str(self.backup_dir),
                'max_backups': self.max_backups
            }
            
            if backups:
                stats['latest_backup'] = backups[0]['name']
                stats['oldest_backup'] = backups[-1]['name']
                
                # Calculate total backup size
                total_size = 0
                for backup in backups:
                    backup_path = Path(backup['path'])
                    if backup_path.exists():
                        for file_path in backup_path.rglob('*'):
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
                                
                stats['total_size_bytes'] = total_size
                stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
                
            return stats
            
        except Exception as e:
            logger.error("Error getting backup stats: %s", e)
            return {'error': str(e)}