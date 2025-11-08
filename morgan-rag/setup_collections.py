#!/usr/bin/env python3
"""
Setup Collections Script

Initialize the vector database collections needed for the enhanced search integration.
"""

import sys
import os

# Add the morgan package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from morgan.vector_db.client import VectorDBClient
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


def setup_collections():
    """Set up the necessary vector database collections."""
    print("🔧 Setting up vector database collections...")
    
    try:
        # Initialize vector database client
        vector_db = VectorDBClient()
        
        # Test connection
        if not vector_db.health_check():
            print("❌ Vector database health check failed!")
            return False
        
        print("✅ Vector database connection successful")
        
        # Initialize companion collections
        print("📦 Initializing companion collections...")
        success = vector_db.initialize_companion_collections()
        
        if success:
            print("✅ Companion collections initialized successfully")
        else:
            print("⚠️ Some companion collections may not have been created")
        
        # Create hierarchical knowledge collection
        print("🏗️ Creating hierarchical knowledge collection...")
        hierarchical_success = vector_db.create_hierarchical_collection(
            name="morgan_knowledge",
            coarse_size=384,
            medium_size=768,
            fine_size=1536
        )
        
        if hierarchical_success:
            print("✅ Hierarchical knowledge collection created")
        else:
            print("⚠️ Hierarchical knowledge collection may already exist")
        
        # Get collection statistics
        print("\n📊 Collection Statistics:")
        stats = vector_db.get_collection_stats()
        
        for collection_name, info in stats.items():
            print(f"  {collection_name}:")
            print(f"    Status: {info.status}")
            print(f"    Points: {info.points_count}")
            print(f"    Vectors: {info.vectors_count}")
        
        print("\n🎉 Vector database setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        logger.error(f"Collection setup failed: {e}")
        return False


if __name__ == "__main__":
    success = setup_collections()
    sys.exit(0 if success else 1)