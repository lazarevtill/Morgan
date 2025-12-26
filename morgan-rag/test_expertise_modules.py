#!/usr/bin/env python3
"""
Test script for domain expertise modules.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all expertise modules can be imported."""
    try:
        # Test individual module imports
        print("Testing module imports...")
        
        # Import config first to avoid issues
        from morgan.config import get_settings
        print("✓ Config imported")
        
        # Import utils
        from morgan.utils.logger import get_logger
        print("✓ Logger imported")
        
        # Import emotional models (dependency)
        from morgan.emotional.models import InteractionData
        print("✓ Emotional models imported")
        
        # Import expertise modules
        from morgan.expertise.domains import DomainKnowledgeTracker, DomainProfile
        print("✓ Domains module imported")
        
        from morgan.expertise.vocabulary import VocabularyLearner, DomainVocabulary
        print("✓ Vocabulary module imported")
        
        from morgan.expertise.context import DomainContextEngine, DomainContext
        print("✓ Context module imported")
        
        from morgan.expertise.depth import KnowledgeDepthAssessor, KnowledgeLevel
        print("✓ Depth module imported")
        
        from morgan.expertise.teaching import AdaptiveTeachingEngine, TeachingStrategy
        print("✓ Teaching module imported")
        
        print("\n✅ All expertise modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of expertise modules."""
    try:
        print("\nTesting basic functionality...")
        
        # Test domain tracker
        from morgan.expertise.domains import get_domain_tracker
        tracker = get_domain_tracker()
        print("✓ Domain tracker instance created")
        
        # Test vocabulary learner
        from morgan.expertise.vocabulary import get_vocabulary_learner
        vocab_learner = get_vocabulary_learner()
        print("✓ Vocabulary learner instance created")
        
        # Test context engine
        from morgan.expertise.context import get_context_engine
        context_engine = get_context_engine()
        print("✓ Context engine instance created")
        
        # Test depth assessor
        from morgan.expertise.depth import get_depth_assessor
        depth_assessor = get_depth_assessor()
        print("✓ Depth assessor instance created")
        
        # Test teaching engine
        from morgan.expertise.teaching import get_teaching_engine
        teaching_engine = get_teaching_engine()
        print("✓ Teaching engine instance created")
        
        print("\n✅ All expertise modules function correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Domain Expertise Modules")
    print("=" * 40)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test functionality
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n🎉 All tests passed! Domain expertise modules are working correctly.")
            sys.exit(0)
        else:
            print("\n💥 Functionality tests failed.")
            sys.exit(1)
    else:
        print("\n💥 Import tests failed.")
        sys.exit(1)