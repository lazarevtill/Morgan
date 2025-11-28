#!/usr/bin/env python3
"""Quick test of Morgan's working components"""
import sys
from pathlib import Path

local_libs = Path(__file__).parent.parent / "local_libs"
if local_libs.exists():
    sys.path.insert(0, str(local_libs))
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing Morgan Components...")
print("=" * 60)

# 1. Config
print("\n1. Testing Configuration...")
from morgan.config import get_settings
settings = get_settings()
print(f"   ✅ LLM: {settings.llm_model} @ {settings.llm_base_url}")
print(f"   ✅ Embedding: {settings.embedding_model}")
print(f"   ✅ Qdrant: {settings.qdrant_url}")

# 2. Embedding Service
print("\n2. Testing Embedding Service...")
from morgan.services.embedding_service import EmbeddingService
emb_service = EmbeddingService()
test_emb = emb_service.encode("Hello world")
print(f"   ✅ Generated embedding: {len(test_emb)} dimensions")

# 3. LLM Service
print("\n3. Testing LLM Service...")
from morgan.services.llm_service import get_llm_service
llm = get_llm_service()
response = llm.generate("Say 'test' only", max_tokens=10)
print(f"   ✅ LLM response: {response.content[:50]}")

# 4. Qdrant
print("\n4. Testing Qdrant Connection...")
from morgan.vector_db.client import VectorDBClient
vdb = VectorDBClient()
collections = vdb.list_collections()
print(f"   ✅ Connected. Collections: {len(collections)}")
for c in collections[:5]:
    print(f"      - {c}")

# 5. Emotion Detection
print("\n5. Testing Emotion Detection...")
from morgan.emotional.intelligence_engine import EmotionalIntelligenceEngine
emotion_engine = EmotionalIntelligenceEngine()
emotion_state = emotion_engine.analyze_text("I'm so excited about this!")
print(f"   ✅ Detected: {emotion_state.primary_emotion} ({emotion_state.intensity:.2f})")

# 6. Memory
print("\n6. Testing Memory System...")
from morgan.core.memory import ConversationMemory
memory = ConversationMemory()
print(f"   ✅ Memory system initialized")

print("\n" + "=" * 60)
print("✅ All core components working!")
print("\nRun Morgan:")
print("  export PYTHONPATH=./local_libs:$PYTHONPATH")
print("  python3 scripts/run_morgan.py")
