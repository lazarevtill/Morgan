"""
Comprehensive system integration tests for Morgan RAG Advanced Vectorization System.

Tests the complete end-to-end workflows from document ingestion to companion responses,
validating all performance targets and companion-aware features.

This implements the testing requirements for task 9.2: Complete system integration and testing.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

from morgan.core.system_integration import (
    AdvancedVectorizationSystem,
    SystemConfiguration,
    WorkflowResult,
    SystemHealthStatus,
    get_advanced_vectorization_system,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class TestSystemIntegration:
    """Test suite for complete system integration and end-to-end workflows."""

    @pytest.fixture
    def system_config(self):
        """Create test system configuration."""
        return SystemConfiguration(
            enable_companion_features=True,
            enable_emotional_intelligence=True,
            enable_hierarchical_search=True,
            enable_batch_optimization=True,
            enable_intelligent_caching=True,
            enable_performance_monitoring=True,
            target_processing_rate=50.0,  # Lower for testing
            target_search_latency=1.0,  # Higher for testing
            target_cache_speedup=2.0,  # Lower for testing
            target_candidate_reduction=0.5,  # Lower for testing
        )

    @pytest.fixture
    def integration_system(self, system_config):
        """Create and initialize integration system for testing."""
        system = AdvancedVectorizationSystem(system_config)

        # Mock external dependencies for testing
        with patch.multiple(
            system,
            vector_db=AsyncMock(),
            document_processor=AsyncMock(),
            embedding_service=AsyncMock(),
            multi_stage_search=AsyncMock(),
            relationship_manager=AsyncMock(),
            emotional_engine=AsyncMock(),
            memory_processor=AsyncMock(),
            batch_processor=Mock(),
            performance_monitor=Mock(),
        ):
            # Mock successful initialization
            system.is_initialized = True
            yield system

    @pytest.mark.asyncio
    async def test_system_initialization(self, system_config):
        """Test complete system initialization with all components."""
        system = AdvancedVectorizationSystem(system_config)

        # Mock all component initializations
        with patch.multiple(
            system,
            _initialize_core_components=Mock(),
            _initialize_vectorization_components=Mock(),
            _initialize_companion_components=Mock(),
            _initialize_monitoring_components=Mock(),
            _initialize_vector_collections=AsyncMock(),
            _validate_system_components=AsyncMock(
                return_value={"success": True, "errors": []}
            ),
            _run_performance_baseline=AsyncMock(),
            _initialize_companion_system=AsyncMock(),
        ):
            # Test initialization
            success = await system.initialize_system()

            assert success is True
            assert system.is_initialized is True

            # Verify all initialization methods were called
            system._initialize_core_components.assert_called_once()
            system._initialize_vectorization_components.assert_called_once()
            system._initialize_companion_components.assert_called_once()
            system._initialize_monitoring_components.assert_called_once()

    @pytest.mark.asyncio
    async def test_document_processing_workflow(self, integration_system):
        """Test complete document processing workflow with companion features."""
        system = integration_system

        # Mock document processing results
        mock_chunks = [
            Mock(
                content="Test document content 1",
                source="test_doc_1.txt",
                chunk_id="chunk_1",
                metadata={"category": "test", "type": "text"},
            ),
            Mock(
                content="Test document content 2",
                source="test_doc_2.txt",
                chunk_id="chunk_2",
                metadata={"category": "test", "type": "text"},
            ),
        ]

        system.document_processor.process_documents.return_value = Mock(
            chunks=mock_chunks
        )

        # Mock embedding results
        mock_embedding = Mock(coarse=[0.1] * 384, medium=[0.2] * 768, fine=[0.3] * 1536)
        system.embedding_service.create_hierarchical_embeddings.return_value = (
            mock_embedding
        )
        system.clustering_engine.apply_contrastive_bias.return_value = mock_embedding

        # Mock batch processing
        system.batch_processor.process_embeddings_batch.return_value = Mock(
            results=[mock_embedding, mock_embedding],
            processing_time=0.5,
            success_rate=100.0,
        )
        system.batch_processor.process_vector_operations_batch.return_value = Mock(
            processing_time=0.2, success_rate=100.0
        )

        # Mock companion context
        system.relationship_manager.get_user_context.return_value = {
            "user_id": "test_user",
            "interests": ["technology", "ai"],
            "communication_style": "technical",
        }

        # Test workflow
        result = await system.process_documents_workflow(
            documents=["Test document 1", "Test document 2"],
            source_type="text",
            user_id="test_user",
            emotional_context={"primary_emotion": "curiosity", "intensity": 0.8},
            show_progress=False,
        )

        # Validate results
        assert result.success is True
        assert result.workflow_type == "document_processing"
        assert result.items_processed == 2
        assert result.processing_time > 0

        # Validate performance metrics
        assert "processing_rate" in result.performance_metrics
        assert result.performance_metrics["processing_rate"] > 0

        # Validate companion metrics
        assert result.companion_metrics["companion_enhanced"] is True
        assert result.companion_metrics["user_id"] == "test_user"
        assert result.companion_metrics["emotional_context_applied"] is True

        # Verify component interactions
        system.document_processor.process_documents.assert_called()
        system.relationship_manager.get_user_context.assert_called_with("test_user")
        system.relationship_manager.update_knowledge_interaction.assert_called()

    @pytest.mark.asyncio
    async def test_search_workflow_with_companion_features(self, integration_system):
        """Test complete search workflow with companion awareness and emotional intelligence."""
        system = integration_system

        # Mock emotional analysis
        mock_emotional_context = {
            "primary_emotion": "frustration",
            "intensity": 0.7,
            "confidence": 0.9,
            "secondary_emotions": ["anxiety"],
            "emotional_indicators": ["stuck", "problem"],
        }
        system.emotional_engine.analyze_emotion.return_value = mock_emotional_context

        # Mock companion context
        mock_companion_context = {
            "user_id": "test_user",
            "interests": ["docker", "deployment"],
            "communication_style": "detailed",
            "relationship_duration": 30,
            "interaction_count": 150,
        }
        system.relationship_manager.get_user_context.return_value = (
            mock_companion_context
        )

        # Mock search results
        mock_search_results = Mock(
            results=[
                Mock(
                    content="Docker deployment guide content",
                    source="docker_guide.md",
                    score=0.95,
                    result_type="semantic",
                    metadata={"category": "deployment"},
                ),
                Mock(
                    content="Container orchestration content",
                    source="orchestration.md",
                    score=0.87,
                    result_type="hierarchical",
                    metadata={"category": "containers"},
                ),
            ],
            strategies_used=["semantic", "contextual", "emotional"],
            fusion_applied=True,
            search_time=0.3,
        )
        mock_search_results.get_reduction_ratio.return_value = 0.85

        system.multi_stage_search.search.return_value = mock_search_results

        # Mock enhanced results
        system.relationship_manager.enhance_search_result.return_value = Mock(
            content="Enhanced result with personalization",
            score=0.98,
            enhancement_factors=["user_interest_match", "communication_style_adapted"],
        )

        # Test workflow
        result = await system.search_workflow(
            query="How to deploy Docker containers?",
            user_id="test_user",
            conversation_id="conv_123",
            max_results=5,
            use_hierarchical=True,
            include_memories=True,
        )

        # Validate results
        assert result.success is True
        assert result.workflow_type == "search"
        assert result.items_processed == 2
        assert result.processing_time > 0

        # Validate performance metrics
        assert result.performance_metrics["search_latency"] > 0
        assert result.performance_metrics["candidate_reduction"] == 0.85
        assert result.performance_metrics["strategies_used"] == [
            "semantic",
            "contextual",
            "emotional",
        ]
        assert result.performance_metrics["hierarchical_used"] is True

        # Validate companion metrics
        assert result.companion_metrics["companion_enhanced"] is True
        assert result.companion_metrics["emotional_enhanced"] is True
        assert result.companion_metrics["personalization_applied"] is True

        # Verify component interactions
        system.emotional_engine.analyze_emotion.assert_called()
        system.relationship_manager.get_user_context.assert_called_with("test_user")
        system.multi_stage_search.search.assert_called()
        system.memory_processor.process_search_interaction.assert_called()

    @pytest.mark.asyncio
    async def test_conversation_workflow_end_to_end(self, integration_system):
        """Test complete conversation workflow with all companion features."""
        system = integration_system

        # Mock emotional analysis
        mock_emotional_context = {
            "primary_emotion": "excitement",
            "intensity": 0.8,
            "confidence": 0.95,
        }
        system.emotional_engine.analyze_emotion.return_value = mock_emotional_context

        # Mock conversation creation
        system.conversation_memory.create_conversation.return_value = "conv_456"

        # Mock search workflow result
        mock_search_result = Mock(
            success=True,
            processing_time=0.4,
            items_processed=3,
            performance_metrics={"search_latency": 0.4, "results_count": 3},
        )

        # Mock assistant response
        mock_assistant_response = Mock(
            answer="Here's how to deploy Docker containers effectively...",
            sources=["docker_guide.md", "deployment_best_practices.md"],
            confidence=0.92,
            suggestions=["Learn about Docker Compose", "Explore Kubernetes"],
        )
        system.assistant.ask.return_value = mock_assistant_response

        # Mock memory processing
        mock_memory_result = Mock(
            memories=[
                Mock(
                    content="User interested in Docker deployment", importance_score=0.8
                ),
                Mock(
                    content="User prefers detailed explanations", importance_score=0.7
                ),
            ]
        )
        system.memory_processor.extract_memories.return_value = mock_memory_result

        # Mock companion context
        system.relationship_manager.get_user_context.return_value = {
            "user_id": "test_user",
            "communication_style": "detailed",
            "interests": ["docker", "devops"],
        }

        # Patch search workflow to return mock result
        with patch.object(system, "search_workflow", return_value=mock_search_result):
            # Test workflow
            result = await system.conversation_workflow(
                user_message="I want to learn about Docker deployment",
                user_id="test_user",
                include_emotional_analysis=True,
                update_relationship=True,
            )

        # Validate results
        assert result.success is True
        assert result.workflow_type == "conversation"
        assert result.processing_time > 0

        # Validate performance metrics
        assert result.performance_metrics["conversation_processing_time"] > 0
        assert result.performance_metrics["search_time"] == 0.4
        assert result.performance_metrics["memories_extracted"] == 2
        assert result.performance_metrics["response_confidence"] == 0.92

        # Validate companion metrics
        assert result.companion_metrics["emotional_analysis_performed"] is True
        assert result.companion_metrics["companion_personalization"] is True
        assert result.companion_metrics["relationship_updated"] is True
        assert result.companion_metrics["detected_emotion"] == "excitement"
        assert result.companion_metrics["emotion_intensity"] == 0.8

        # Verify component interactions
        system.emotional_engine.analyze_emotion.assert_called()
        system.conversation_memory.create_conversation.assert_called()
        system.assistant.ask.assert_called()
        system.memory_processor.extract_memories.assert_called()
        system.relationship_manager.update_conversation_interaction.assert_called()

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, integration_system):
        """Test comprehensive system health monitoring."""
        system = integration_system

        # Mock component health checks
        system.assistant.health_check = AsyncMock(return_value={"status": "healthy"})
        system.knowledge_base.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        system.smart_search.health_check = AsyncMock(return_value={"status": "healthy"})
        system.vector_db.health_check = AsyncMock(return_value={"status": "healthy"})

        # Mock performance summary
        system.performance_monitor.get_performance_summary.return_value = {
            "application_performance": {
                "search": {"p95_duration": 0.3},
                "processing": {"p95_duration": 2.0},
            },
            "system_resources": {
                "current_cpu_percent": 45.0,
                "current_memory_percent": 60.0,
            },
        }

        # Test health check
        health_status = await system.get_system_health()

        # Validate results
        assert isinstance(health_status, SystemHealthStatus)
        assert health_status.overall_status in ["healthy", "warning", "critical"]
        assert "assistant" in health_status.component_statuses
        assert "knowledge_base" in health_status.component_statuses
        assert "smart_search" in health_status.component_statuses
        assert "vector_db" in health_status.component_statuses

        # Validate companion health
        if system.config.enable_companion_features:
            assert "relationship_manager" in health_status.companion_health
            assert "companion_storage" in health_status.companion_health

        if system.config.enable_emotional_intelligence:
            assert "emotional_intelligence" in health_status.companion_health

        # Validate timestamp
        assert isinstance(health_status.timestamp, datetime)
        assert health_status.timestamp <= datetime.now()

    @pytest.mark.asyncio
    async def test_performance_target_validation(self, integration_system):
        """Test validation of all performance targets."""
        system = integration_system

        # Add mock workflow history with performance data
        system.workflow_history = [
            Mock(
                workflow_type="document_processing",
                timestamp=datetime.now(),
                performance_metrics={"processing_rate": 60.0},  # Above target of 50
            ),
            Mock(
                workflow_type="search",
                timestamp=datetime.now(),
                performance_metrics={
                    "search_latency": 0.8,  # Below target of 1.0
                    "candidate_reduction": 0.7,  # Above target of 0.5
                    "hierarchical_used": True,
                },
            ),
        ]

        # Mock cache statistics
        system.cache_manager.get_cache_statistics = AsyncMock(
            return_value={"average_speedup": 3.0}  # Above target of 2.0
        )

        # Test validation
        validation_results = await system.validate_performance_targets()

        # Validate results structure
        assert "overall_success" in validation_results
        assert "target_results" in validation_results
        assert "recommendations" in validation_results

        # Validate specific targets
        if "processing_rate" in validation_results["target_results"]:
            processing_result = validation_results["target_results"]["processing_rate"]
            assert processing_result["target"] == 50.0
            assert processing_result["actual"] == 60.0
            assert processing_result["success"] is True

        if "search_latency" in validation_results["target_results"]:
            search_result = validation_results["target_results"]["search_latency"]
            assert search_result["target"] == 1.0
            assert search_result["actual"] == 0.8
            assert search_result["success"] is True

        if "candidate_reduction" in validation_results["target_results"]:
            reduction_result = validation_results["target_results"][
                "candidate_reduction"
            ]
            assert reduction_result["target"] == 0.5
            assert reduction_result["actual"] == 0.7
            assert reduction_result["success"] is True

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integration_system):
        """Test error handling and graceful degradation."""
        system = integration_system

        # Test document processing with component failure
        system.embedding_service.create_hierarchical_embeddings.side_effect = Exception(
            "Embedding service failed"
        )

        result = await system.process_documents_workflow(
            documents=["Test document"], source_type="text", show_progress=False
        )

        # Should handle error gracefully
        assert result.success is False
        assert len(result.errors) > 0
        assert "Embedding service failed" in str(result.errors)

        # Test search with component failure
        system.multi_stage_search.search.side_effect = Exception("Search engine failed")

        search_result = await system.search_workflow(query="test query", max_results=5)

        # Should handle error gracefully
        assert search_result.success is False
        assert len(search_result.errors) > 0
        assert "Search engine failed" in str(search_result.errors)

    @pytest.mark.asyncio
    async def test_batch_optimization_integration(self, integration_system):
        """Test integration with batch optimization for performance."""
        system = integration_system

        # Mock batch processor results
        system.batch_processor.process_embeddings_batch.return_value = Mock(
            results=[[0.1] * 1536] * 10,  # 10 embeddings
            processing_time=0.5,
            throughput=20.0,  # 20 items/second
            success_rate=100.0,
        )

        system.batch_processor.process_vector_operations_batch.return_value = Mock(
            processing_time=0.2, throughput=50.0, success_rate=100.0
        )

        # Mock document chunks
        mock_chunks = [
            Mock(
                content=f"Document {i}",
                source=f"doc_{i}.txt",
                chunk_id=f"chunk_{i}",
                metadata={"category": "test"},
            )
            for i in range(10)
        ]

        system.document_processor.process_documents.return_value = Mock(
            chunks=mock_chunks
        )

        # Test batch processing workflow
        result = await system.process_documents_workflow(
            documents=[f"Document {i}" for i in range(10)],
            source_type="text",
            show_progress=False,
        )

        # Validate batch optimization was used
        assert result.success is True
        assert result.items_processed == 10

        # Verify batch processor was called
        system.batch_processor.process_embeddings_batch.assert_called()
        system.batch_processor.process_vector_operations_batch.assert_called()

    @pytest.mark.asyncio
    async def test_companion_personalization_integration(self, integration_system):
        """Test integration of companion personalization across workflows."""
        system = integration_system

        user_id = "test_user_123"

        # Mock comprehensive companion context
        companion_context = {
            "user_id": user_id,
            "interests": ["machine_learning", "python", "data_science"],
            "communication_style": "technical_detailed",
            "relationship_duration": 45,  # days
            "interaction_count": 200,
            "preferred_topics": ["tutorials", "best_practices"],
            "emotional_patterns": {
                "common_emotions": ["curiosity", "determination"],
                "stress_indicators": ["time_pressure", "complexity"],
            },
        }

        system.relationship_manager.get_user_context.return_value = companion_context

        # Test document processing with personalization
        doc_result = await system.process_documents_workflow(
            documents=["Machine learning tutorial content"],
            user_id=user_id,
            emotional_context={"primary_emotion": "curiosity", "intensity": 0.8},
        )

        assert doc_result.companion_metrics["companion_enhanced"] is True
        assert doc_result.companion_metrics["user_id"] == user_id

        # Test search with personalization
        search_result = await system.search_workflow(
            query="Python machine learning best practices", user_id=user_id
        )

        assert search_result.companion_metrics["companion_enhanced"] is True
        assert search_result.companion_metrics["personalization_applied"] is True

        # Test conversation with personalization
        conv_result = await system.conversation_workflow(
            user_message="I need help with ML model deployment", user_id=user_id
        )

        assert conv_result.companion_metrics["companion_personalization"] is True
        assert conv_result.companion_metrics["user_id"] == user_id

        # Verify relationship manager interactions
        assert system.relationship_manager.get_user_context.call_count >= 3
        system.relationship_manager.update_knowledge_interaction.assert_called()
        system.relationship_manager.update_search_interaction.assert_called()
        system.relationship_manager.update_conversation_interaction.assert_called()

    @pytest.mark.asyncio
    async def test_singleton_system_access(self, system_config):
        """Test singleton access to the advanced vectorization system."""
        # Mock the initialization to avoid actual system setup
        with patch(
            "morgan.core.system_integration.AdvancedVectorizationSystem"
        ) as MockSystem:
            mock_instance = Mock()
            mock_instance.initialize_system = AsyncMock(return_value=True)
            MockSystem.return_value = mock_instance

            # Get system instance
            system1 = await get_advanced_vectorization_system(system_config)
            system2 = await get_advanced_vectorization_system(system_config)

            # Should be the same instance
            assert system1 is system2

            # Should only create one instance
            assert MockSystem.call_count == 1

    @pytest.mark.asyncio
    async def test_system_shutdown(self, integration_system):
        """Test graceful system shutdown."""
        system = integration_system

        # Mock shutdown methods
        system.performance_monitor.stop_system_monitoring = Mock()
        system.batch_processor.shutdown = Mock()
        system.vector_db.close = AsyncMock()
        system.companion_storage.close = AsyncMock()

        # Test shutdown
        await system.shutdown()

        # Verify shutdown methods were called
        system.performance_monitor.stop_system_monitoring.assert_called_once()
        system.batch_processor.shutdown.assert_called_once()
        system.vector_db.close.assert_called_once()
        system.companion_storage.close.assert_called_once()


class TestPerformanceIntegration:
    """Test performance aspects of system integration."""

    @pytest.mark.asyncio
    async def test_processing_performance_targets(self):
        """Test that processing meets performance targets."""
        config = SystemConfiguration(
            target_processing_rate=100.0,  # docs per minute
            target_search_latency=0.5,  # seconds
            enable_batch_optimization=True,
        )

        system = AdvancedVectorizationSystem(config)

        # Mock high-performance components
        with patch.multiple(
            system,
            document_processor=Mock(),
            batch_processor=Mock(),
            vector_db=AsyncMock(),
        ):
            # Mock fast processing
            system.batch_processor.process_embeddings_batch.return_value = Mock(
                processing_time=0.1,  # Very fast
                throughput=200.0,  # High throughput
                success_rate=100.0,
            )

            start_time = time.time()

            # Process test documents
            result = await system.process_documents_workflow(
                documents=["Test doc"] * 10, show_progress=False
            )

            processing_time = time.time() - start_time

            # Validate performance
            assert processing_time < 2.0  # Should be fast
            assert result.success is True

    @pytest.mark.asyncio
    async def test_search_performance_targets(self):
        """Test that search meets performance targets."""
        config = SystemConfiguration(
            target_search_latency=0.5,
            target_candidate_reduction=0.9,
            enable_hierarchical_search=True,
        )

        system = AdvancedVectorizationSystem(config)

        # Mock high-performance search
        with patch.multiple(system, multi_stage_search=Mock(), emotional_engine=Mock()):
            # Mock fast search with good reduction
            mock_results = Mock(
                results=[Mock(score=0.9), Mock(score=0.8)],
                search_time=0.2,  # Fast search
                strategies_used=["semantic", "hierarchical"],
                fusion_applied=True,
            )
            mock_results.get_reduction_ratio.return_value = 0.95  # Excellent reduction

            system.multi_stage_search.search.return_value = mock_results

            start_time = time.time()

            # Execute search
            result = await system.search_workflow(
                query="test query", use_hierarchical=True
            )

            search_time = time.time() - start_time

            # Validate performance
            assert search_time < 1.0  # Should be fast
            assert result.success is True
            assert result.performance_metrics["candidate_reduction"] >= 0.9


class TestCompanionIntegration:
    """Test companion-aware features integration."""

    @pytest.mark.asyncio
    async def test_emotional_intelligence_integration(self):
        """Test emotional intelligence integration across workflows."""
        config = SystemConfiguration(
            enable_emotional_intelligence=True, enable_companion_features=True
        )

        system = AdvancedVectorizationSystem(config)

        with patch.multiple(
            system,
            emotional_engine=Mock(),
            relationship_manager=Mock(),
            multi_stage_search=Mock(),
        ):
            # Mock emotional analysis
            system.emotional_engine.analyze_emotion.return_value = {
                "primary_emotion": "frustration",
                "intensity": 0.8,
                "confidence": 0.9,
                "emotional_indicators": ["stuck", "problem", "help"],
            }

            # Mock search results
            mock_results = Mock(
                results=[Mock(score=0.9, content="Helpful solution")],
                strategies_used=["semantic", "emotional"],
                fusion_applied=True,
            )
            mock_results.get_reduction_ratio.return_value = 0.8
            system.multi_stage_search.search.return_value = mock_results

            # Test search with emotional context
            result = await system.search_workflow(
                query="I'm stuck with this Docker problem", user_id="frustrated_user"
            )

            # Validate emotional integration
            assert result.success is True
            assert result.companion_metrics["emotional_enhanced"] is True

            # Verify emotional analysis was called
            system.emotional_engine.analyze_emotion.assert_called()

            # Verify search used emotional strategy
            search_call = system.multi_stage_search.search.call_args
            assert "emotional" in search_call[1]["strategies"]

    @pytest.mark.asyncio
    async def test_relationship_building_integration(self):
        """Test relationship building across multiple interactions."""
        config = SystemConfiguration(
            enable_companion_features=True, enable_emotional_intelligence=True
        )

        system = AdvancedVectorizationSystem(config)

        with patch.multiple(
            system,
            relationship_manager=Mock(),
            memory_processor=Mock(),
            assistant=Mock(),
        ):
            user_id = "relationship_test_user"

            # Mock relationship progression
            initial_context = {
                "user_id": user_id,
                "interaction_count": 1,
                "relationship_duration": 1,
                "interests": [],
            }

            evolved_context = {
                "user_id": user_id,
                "interaction_count": 5,
                "relationship_duration": 7,
                "interests": ["docker", "deployment", "troubleshooting"],
                "communication_style": "detailed_technical",
            }

            # First interaction - minimal context
            system.relationship_manager.get_user_context.return_value = initial_context
            system.assistant.ask.return_value = Mock(
                answer="Basic Docker help", confidence=0.7, sources=["docker_basics.md"]
            )

            result1 = await system.conversation_workflow(
                user_message="How do I use Docker?", user_id=user_id
            )

            # Later interaction - rich context
            system.relationship_manager.get_user_context.return_value = evolved_context
            system.assistant.ask.return_value = Mock(
                answer="Advanced Docker deployment strategies based on your experience",
                confidence=0.9,
                sources=["advanced_docker.md", "deployment_patterns.md"],
            )

            result2 = await system.conversation_workflow(
                user_message="I need help with Docker deployment issues",
                user_id=user_id,
            )

            # Validate relationship evolution
            assert result1.success is True
            assert result2.success is True

            # Should show progression in companion metrics
            assert result1.companion_metrics["companion_personalization"] is True
            assert result2.companion_metrics["companion_personalization"] is True

            # Verify relationship updates were called
            assert (
                system.relationship_manager.update_conversation_interaction.call_count
                == 2
            )


if __name__ == "__main__":
    # Run tests with asyncio support
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
