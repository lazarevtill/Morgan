
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    print("Verifying core module imports...")
    
    # 1. Intelligence
    print("Checking morgan.intelligence...")
    from morgan.intelligence.core.intelligence_engine import get_emotional_intelligence_engine
    print("  - Intelligence engine imported.")
    
    # 2. Embeddings
    print("Checking morgan.embeddings...")
    from morgan.embeddings.service import get_embedding_service
    service = get_embedding_service()
    print(f"  - Embedding Service initialized. Available: {service.is_available()}")
    
    # 3. Background Service
    print("Checking morgan.background...")
    from morgan.background.application.orchestrators import BackgroundOrchestrator
    orchestrator = BackgroundOrchestrator()
    print("  - Background Orchestrator initialized.")
    
    print("\nSUCCESS: All core modules verified.")
    
except Exception as e:
    print(f"\nFAILURE: Verification failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
