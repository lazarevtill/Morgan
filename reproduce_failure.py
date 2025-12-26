
import sys
import os
from unittest.mock import Mock

# Add project root to sys.path
sys.path.append(os.getcwd())

from morgan.vectorization.hierarchical_embeddings import HierarchicalEmbeddingService, HierarchicalEmbedding

def reproduce():
    service = HierarchicalEmbeddingService()
    mock_embedding_service = Mock()
    service.embedding_service = mock_embedding_service
    
    mock_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    mock_embedding_service.encode_batch.return_value = mock_embeddings
    
    content = "def login(username, password):\n    return authenticate(username, password)"
    metadata = {"title": "Login Function", "source": "auth.py"}
    
    try:
        result = service.create_hierarchical_embeddings(
            content, metadata, category="code"
        )
        print("Success!")
        print(f"Coarse: {result.coarse}")
        
        mock_embedding_service.encode_batch.assert_called_once()
        call_args = mock_embedding_service.encode_batch.call_args
        print(f"Call args[0][0] len: {len(call_args[0][0])}")
        print(f"Call args[1]: {call_args[1]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
