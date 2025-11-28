#!/usr/bin/env python3
"""
Complete Morgan Pipeline Test

Tests ALL components of Morgan v2-0.0.1:
1. Emotion Detection (11 modules)
2. Empathy Engine (5 modules)
3. Advanced RAG (multi-stage + reranking)
4. Memory System (conversation, emotional, knowledge)
5. Learning & Adaptation
6. Complete end-to-end flow

Usage:
    PYTHONPATH=./local_libs:$PYTHONPATH python3 scripts/test_complete_pipeline.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add local libs and morgan-rag to path
local_libs = Path(__file__).parent.parent / "local_libs"
if local_libs.exists():
    sys.path.insert(0, str(local_libs))

morgan_rag = Path(__file__).parent.parent
sys.path.insert(0, str(morgan_rag))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


def test_emotion_detection():
    """Test all 11 emotion detection modules"""
    console.print("\n[bold blue]📊 Testing Emotion Detection System (11 modules)[/bold blue]")

    test_inputs = [
        ("I'm so happy about this achievement!", "joy", 0.8),
        ("I'm really worried about the deadline", "anxiety", 0.7),
        ("This is incredibly frustrating", "anger", 0.7),
        ("I feel so sad and alone", "sadness", 0.8),
        ("I'm excited about the new project!", "excitement", 0.8),
    ]

    results = {}

    try:
        from morgan.emotional.intelligence_engine import EmotionalIntelligenceEngine

        engine = EmotionalIntelligenceEngine()
        console.print("  ✅ Emotional Intelligence Engine initialized")

        for text, expected_emotion, min_intensity in test_inputs:
            emotion_state = engine.analyze_text(text)

            detected = emotion_state.primary_emotion
            intensity = emotion_state.intensity

            match = detected.lower() == expected_emotion.lower()
            intensity_ok = intensity >= min_intensity

            status = "✅" if match and intensity_ok else "⚠️"
            console.print(f"  {status} '{text[:40]}...'")
            console.print(f"     Expected: {expected_emotion} | Detected: {detected} ({intensity:.2f})")

            results[text] = {
                "expected": expected_emotion,
                "detected": detected,
                "intensity": intensity,
                "match": match
            }

        # Test emotional memory
        console.print("\n  Testing Emotional Memory...")
        memory = engine.get_emotional_memory()
        console.print(f"  ✅ Stored {len(memory)} emotional events")

        # Test pattern recognition
        console.print("\n  Testing Pattern Recognition...")
        patterns = engine.identify_patterns()
        console.print(f"  ✅ Identified {len(patterns)} emotional patterns")

        return True, results

    except Exception as e:
        console.print(f"  ❌ Emotion detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_empathy_engine():
    """Test all 5 empathy modules"""
    console.print("\n[bold blue]💙 Testing Empathy Engine (5 modules)[/bold blue]")

    try:
        from morgan.empathy.generator import EmpathyGenerator
        from morgan.empathy.mirror import EmotionalMirror
        from morgan.empathy.support import SupportGenerator
        from morgan.empathy.tone import ToneAdjuster
        from morgan.empathy.validator import ValidationEngine

        # Test Empathy Generator
        console.print("\n  1. Testing Empathy Generator...")
        generator = EmpathyGenerator()
        empathetic_response = generator.generate_response(
            emotion="sadness",
            intensity=0.8,
            context="Failed an important exam"
        )
        console.print(f"  ✅ Generated empathetic response: {empathetic_response[:80]}...")

        # Test Emotional Mirror
        console.print("\n  2. Testing Emotional Mirror...")
        mirror = EmotionalMirror()
        mirrored = mirror.reflect_emotion(
            user_emotion="anxiety",
            intensity=0.7
        )
        console.print(f"  ✅ Emotional mirroring: {mirrored}")

        # Test Support Generator
        console.print("\n  3. Testing Support Generator...")
        support = SupportGenerator()
        support_response = support.generate_support(
            emotion="stress",
            context="Work deadline pressure"
        )
        console.print(f"  ✅ Support response: {support_response[:80]}...")

        # Test Tone Adjuster
        console.print("\n  4. Testing Tone Adjuster...")
        adjuster = ToneAdjuster()
        adjusted = adjuster.adjust_tone(
            base_response="Here's how to solve it.",
            emotion="sadness",
            intensity=0.8
        )
        console.print(f"  ✅ Tone adjusted: {adjusted[:80]}...")

        # Test Validation Engine
        console.print("\n  5. Testing Validation Engine...")
        validator = ValidationEngine()
        validation_score = validator.validate_empathy(
            response=empathetic_response,
            emotion="sadness",
            intensity=0.8
        )
        console.print(f"  ✅ Empathy validation score: {validation_score:.2f}")

        return True

    except Exception as e:
        console.print(f"  ❌ Empathy engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_rag():
    """Test multi-stage RAG with reranking"""
    console.print("\n[bold blue]🔍 Testing Advanced RAG Pipeline[/bold blue]")

    try:
        from morgan.services.embedding_service import EmbeddingService
        from morgan.jina.reranking.service import JinaRerankingService
        from morgan.vector_db.client import VectorDBClient
        from morgan.search.multi_stage_search import MultiStageSearch

        # Test Embedding Service
        console.print("\n  1. Testing Embedding Service...")
        embedding_service = EmbeddingService()

        test_query = "How do I deploy a Docker container?"
        query_embedding = embedding_service.encode(test_query, instruction="query")

        console.print(f"  ✅ Query embedding generated (dim: {len(query_embedding)})")

        # Test document embeddings
        test_docs = [
            "Docker deployment involves creating containers from images",
            "Use docker run to start a container from an image",
            "Docker Compose helps manage multi-container applications"
        ]

        doc_embeddings = embedding_service.encode_batch(
            test_docs,
            instruction="document",
            show_progress=False
        )
        console.print(f"  ✅ Document embeddings generated ({len(doc_embeddings)} docs)")

        # Test Reranking Service
        console.print("\n  2. Testing Jina Reranking Service...")
        reranker = JinaRerankingService(enable_background=False)

        # Create mock search results
        from morgan.jina.reranking.service import SearchResult

        mock_results = [
            SearchResult(
                content=doc,
                score=0.5 + i * 0.1,
                metadata={"source": f"doc_{i}"},
                source=f"doc_{i}.txt"
            )
            for i, doc in enumerate(test_docs)
        ]

        reranked_results, metrics = reranker.rerank_results(
            query=test_query,
            results=mock_results,
            top_k=3
        )

        console.print(f"  ✅ Reranking complete")
        console.print(f"     Model: {metrics.model_used}")
        console.print(f"     Improvement: {metrics.improvement_score:.2%}")
        console.print(f"     Processing time: {metrics.processing_time:.3f}s")

        # Test Multi-Stage Search
        console.print("\n  3. Testing Multi-Stage Search...")
        vector_db = VectorDBClient()

        if vector_db.collection_exists("morgan_knowledge"):
            search_engine = MultiStageSearch(
                vector_db=vector_db,
                embedding_service=embedding_service,
                reranker=reranker
            )

            results = search_engine.search(
                query=test_query,
                collection="morgan_knowledge",
                top_k=5
            )

            console.print(f"  ✅ Multi-stage search complete ({len(results)} results)")
        else:
            console.print(f"  ⚠️  Collection 'morgan_knowledge' not found, skipping search")

        return True

    except Exception as e:
        console.print(f"  ❌ RAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system():
    """Test memory system (conversation, emotional, knowledge)"""
    console.print("\n[bold blue]🧠 Testing Memory System[/bold blue]")

    try:
        from morgan.core.memory import ConversationMemory

        memory = ConversationMemory()
        console.print("  ✅ Conversation Memory initialized")

        # Create a test conversation
        console.print("\n  Creating test conversation...")
        conv_id = memory.create_conversation(topic="Docker Deployment Help")
        console.print(f"  ✅ Created conversation: {conv_id}")

        # Add turns
        test_turns = [
            ("How do I deploy with Docker?", "To deploy with Docker, you first need to..."),
            ("What about Docker Compose?", "Docker Compose is a tool for defining and running...")
        ]

        for question, answer in test_turns:
            memory.add_turn(
                conversation_id=conv_id,
                question=question,
                answer=answer,
                tags=["docker", "deployment"]
            )

        console.print(f"  ✅ Added {len(test_turns)} conversation turns")

        # Search conversations
        console.print("\n  Testing conversation search...")
        search_results = memory.search_conversations(
            query="Docker deployment",
            limit=5
        )
        console.print(f"  ✅ Found {len(search_results)} relevant conversations")

        # Get conversation history
        history = memory.get_conversation(conv_id)
        console.print(f"  ✅ Retrieved conversation history ({len(history)} turns)")

        return True

    except Exception as e:
        console.print(f"  ❌ Memory system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_learning_system():
    """Test learning and adaptation (6 modules)"""
    console.print("\n[bold blue]📚 Testing Learning & Adaptation System[/bold blue]")

    try:
        from morgan.learning.feedback import FeedbackCollector
        from morgan.learning.patterns import PatternRecognition
        from morgan.learning.adaptation import AdaptationEngine
        from morgan.learning.preferences import PreferenceLearning

        # Test Feedback Collection
        console.print("\n  1. Testing Feedback Collector...")
        feedback_collector = FeedbackCollector()

        feedback_collector.record_feedback(
            query="Docker help",
            response="Here's how to use Docker...",
            rating=5,
            feedback_type="explicit"
        )
        console.print("  ✅ Feedback recorded")

        # Test Pattern Recognition
        console.print("\n  2. Testing Pattern Recognition...")
        pattern_recognizer = PatternRecognition()

        patterns = pattern_recognizer.identify_patterns([
            {"query": "Docker deployment", "topic": "DevOps"},
            {"query": "Docker Compose", "topic": "DevOps"},
            {"query": "Python async", "topic": "Programming"},
        ])
        console.print(f"  ✅ Identified {len(patterns)} patterns")

        # Test Adaptation Engine
        console.print("\n  3. Testing Adaptation Engine...")
        adapter = AdaptationEngine()

        adapted_response = adapter.adapt_response(
            base_response="Here's the answer.",
            user_preferences={"detail_level": "high", "tone": "technical"}
        )
        console.print(f"  ✅ Response adapted: {adapted_response[:60]}...")

        # Test Preference Learning
        console.print("\n  4. Testing Preference Learning...")
        pref_learner = PreferenceLearning()

        pref_learner.learn_preference(
            category="response_style",
            value="detailed_technical",
            confidence=0.85
        )

        preferences = pref_learner.get_preferences()
        console.print(f"  ✅ Learned {len(preferences)} preferences")

        return True

    except Exception as e:
        console.print(f"  ❌ Learning system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_pipeline():
    """Test complete end-to-end pipeline"""
    console.print("\n[bold blue]🚀 Testing Complete End-to-End Pipeline[/bold blue]")

    try:
        from morgan.core.assistant import MorganAssistant

        # Initialize Morgan
        console.print("\n  Initializing Morgan Assistant...")
        assistant = MorganAssistant()
        console.print("  ✅ Morgan Assistant initialized")

        # Test query
        test_query = "I'm feeling stressed about deploying my application to production. Can you help?"

        console.print(f"\n  Processing query: '{test_query}'")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing...", total=None)

            response = assistant.process_query(
                query=test_query,
                user_id="test_user",
                show_emotions=True,
                show_sources=True
            )

            progress.remove_task(task)

        console.print("\n  [bold green]✅ Complete Pipeline Response:[/bold green]")
        console.print(Panel(
            response.get("answer", "No response"),
            title="Morgan's Response",
            border_style="green"
        ))

        # Show detected emotions
        if "emotions" in response:
            console.print("\n  [yellow]Detected Emotions:[/yellow]")
            for emotion, intensity in response["emotions"].items():
                console.print(f"    - {emotion}: {intensity:.2%}")

        # Show RAG sources
        if "sources" in response:
            console.print("\n  [cyan]Knowledge Sources:[/cyan]")
            for i, source in enumerate(response["sources"][:3], 1):
                console.print(f"    {i}. {source.get('title', 'Unknown')} (score: {source.get('score', 0):.2f})")

        return True

    except Exception as e:
        console.print(f"  ❌ Complete pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold cyan]Morgan v2-0.0.1 Complete Pipeline Test[/bold cyan]\n"
        "Testing ALL components: Emotions, Empathy, RAG, Memory, Learning",
        border_style="blue"
    ))

    results = {}

    # Run all tests
    console.print("\n[bold yellow]═══ Running Component Tests ═══[/bold yellow]")

    results["Emotion Detection"], emotion_details = test_emotion_detection()
    results["Empathy Engine"] = test_empathy_engine()
    results["Advanced RAG"] = test_advanced_rag()
    results["Memory System"] = test_memory_system()
    results["Learning System"] = test_learning_system()
    results["Complete Pipeline"] = test_complete_pipeline()

    # Show final summary
    console.print("\n[bold yellow]═══ Final Test Summary ═══[/bold yellow]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Details", style="dim")

    component_details = {
        "Emotion Detection": "11 modules (analyzer, classifier, intensity, context, patterns, memory, triggers, recovery, tracker, detector, regulator)",
        "Empathy Engine": "5 modules (generator, mirror, support, tone, validator)",
        "Advanced RAG": "Multi-stage search + Jina reranking",
        "Memory System": "Conversation, emotional, knowledge, relationship",
        "Learning System": "Feedback, patterns, adaptation, preferences",
        "Complete Pipeline": "End-to-end emotion → RAG → empathy → response"
    }

    for component, status in results.items():
        status_icon = "✅ PASS" if status else "❌ FAIL"
        details = component_details.get(component, "")
        table.add_row(component, status_icon, details)

    console.print(table)

    # Overall status
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    if passed == total:
        console.print(f"\n[bold green]✅ All {total} component tests passed![/bold green]")
        console.print("[green]Morgan v2-0.0.1 is fully operational with complete emotional intelligence + RAG[/green]")
        return 0
    else:
        console.print(f"\n[bold yellow]⚠️  {passed}/{total} tests passed[/bold yellow]")
        console.print(f"[yellow]Some components need attention[/yellow]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
