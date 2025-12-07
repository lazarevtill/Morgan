
import asyncio
import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../morgan-rag')))

from morgan.core.system_integration import ComprehensiveSystemIntegration
from morgan.jina.reranking.service import SearchResult

@pytest.mark.asyncio
async def test_emotional_analysis_flow():
    """Test the emotional analysis flow using the integration facade."""
    integration = ComprehensiveSystemIntegration()
    
    # Check if necessary components are available
    if not integration.batch_optimizer:
        pytest.skip("Batch optimizer not available")
        
    user_message = "I am feeling a bit anxious about this deployment."
    
    emotional_states, metrics = await integration.batch_optimizer.optimize_emotional_processing_batch(
        texts=[user_message],
        user_ids=["test_user_pytest"],
        contexts=[{"source": "pytest"}]
    )
    
    assert emotional_states is not None
    assert isinstance(emotional_states, list)
    assert len(emotional_states) == 1
    # Add more specific assertions if possible regarding the content of emotional_states

@pytest.mark.asyncio
async def test_embedding_flow():
    """Test the embedding generation flow."""
    integration = ComprehensiveSystemIntegration()
    
    if not integration.batch_optimizer:
        pytest.skip("Batch optimizer not available")
        
    if not integration.embedding_service:
        pytest.skip("Embedding service not available")

    user_message = "System architecture documentation."
    
    embeddings, metrics = await integration.batch_optimizer.optimize_jina_embedding_batch(
        texts=[user_message],
        model_name="jina-embeddings-v4",
        embedding_service=integration.embedding_service
    )
    
    assert embeddings is not None
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    # Check embedding dimension if known, e.g. assert len(embeddings[0]) > 0

@pytest.mark.asyncio
async def test_code_intelligence_flow():
    """Test code intelligence analysis on this very file."""
    integration = ComprehensiveSystemIntegration()
    
    if not integration.code_service:
        pytest.skip("Code service not available")

    current_script_path = os.path.abspath(__file__)
    with open(current_script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    analysis = integration.code_service.analyze_code_file(current_script_path, content)
    
    assert analysis is not None
    assert analysis.language is not None
    assert isinstance(analysis.functions, list)
    assert isinstance(analysis.classes, list)

@pytest.mark.asyncio
async def test_reranking_flow():
    """Test the reranking flow with mock results."""
    integration = ComprehensiveSystemIntegration()
    
    if not integration.reranking_service or not integration.batch_optimizer:
        pytest.skip("Reranking service or batch optimizer not available")

    user_message = "AI architecture"
    mock_results = [
        SearchResult(content="Deep learning architecture for AI", score=0.9, metadata={}, source="mock"),
        SearchResult(content="Cooking recipes for dinner", score=0.1, metadata={}, source="mock")
    ]
    
    ranked, metrics = await integration.batch_optimizer.optimize_jina_reranking_batch(
        queries=[user_message],
        result_sets=[mock_results],
        reranking_service=integration.reranking_service
    )
    
    assert ranked is not None
    assert len(ranked) > 0
    assert len(ranked[0]) > 0
    # Expect the AI result to be ranked higher or at least present
    top_result = ranked[0][0]
    assert isinstance(top_result, SearchResult)

@pytest.mark.asyncio
async def test_full_orchestrator_integration():
    """Test the Orchestrator's comprehensive integration test, verifying Code Intelligence is real."""
    integration = ComprehensiveSystemIntegration()
    
    # We are testing the facade which delegates to the orchestrator
    results = await integration.run_comprehensive_integration_test()
    
    assert results is not None
    assert "code_intelligence" in results
    
    code_result = results["code_intelligence"]
    assert code_result.workflow_name == "Code Intelligence"
    # Verify it is not the stub message
    assert "Stubbed" not in (code_result.error_message or "")
    # Real flow should have 1 document processed (the sample code snippet)
    assert code_result.documents_processed == 1 
    assert code_result.success_rate == 1.0

