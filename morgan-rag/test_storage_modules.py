#!/usr/bin/env python3
"""
Simple test script to verify storage modules functionality.
"""

import sys
import os
from pathlib import Path

# Add the morgan-rag directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "morgan-rag"))

def test_storage_imports():
    """Test that all storage modules can be imported."""
    try:
        from morgan.storage import (
            VectorStorage,
            MemoryStorage,
            ProfileStorage,
            CacheStorage,
            BackupStorage
        )
        print("✓ All storage modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_storage_initialization():
    """Test that storage modules can be initialized."""
    try:
        from morgan.storage import (
            MemoryStorage,
            ProfileStorage,
            CacheStorage,
            BackupStorage
        )
        
        # Test memory storage
        memory_config = {'storage_dir': './test_data/memory'}
        memory_storage = MemoryStorage(memory_config)
        print("✓ MemoryStorage initialized")
        
        # Test profile storage
        profile_config = {'storage_dir': './test_data/profiles'}
        profile_storage = ProfileStorage(profile_config)
        print("✓ ProfileStorage initialized")
        
        # Test cache storage
        cache_config = {}
        cache_storage = CacheStorage(cache_config)
        print("✓ CacheStorage initialized")
        
        # Test backup storage
        backup_config = {'backup_dir': './test_data/backups'}
        backup_storage = BackupStorage(backup_config)
        print("✓ BackupStorage initialized")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Morgan Storage Modules")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_storage_imports():
        success = False
    
    # Test initialization
    if not test_storage_initialization():
        success = False
    
    print("=" * 40)
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())