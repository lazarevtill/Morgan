"""Complete RAG pipeline example demonstrating all refactored components.

This example shows:
1. Document ingestion with streaming
2. Embedding generation with batching
3. Vector storage in Qdrant
4. Multi-stage hierarchical search
5. Reranking with cross-encoder
6. Proper resource cleanup
"""

import asyncio
import logging
from pathlib import Path

from morgan import (
    EnhancedDocumentProcessor,
    EmbeddingService,
    MultiStageSearch,
    ProcessingConfig,
    QdrantClient,
    RerankingService,
    SearchConfig,
)
from morgan.vector_db.client import QdrantConfig
from qdrant_client.http import models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def ingest_documents(
    processor: EnhancedDocumentProcessor,
    embedding_service: EmbeddingService,
    vector_db: QdrantClient,
    documents_path: Path,
    collection_name: str,
) -> int:
    """Ingest documents with streaming processing.

    Args:
        processor: Document processor.
        embedding_service: Embedding service.
        vector_db: Vector database client.
        documents_path: Path to documents directory.
        collection_name: Target collection name.

    Returns:
        Number of chunks ingested.
    """
    logger.info(f"Starting document ingestion from {documents_path}")

    total_chunks = 0

    # Process documents with streaming
    async for doc in processor.process_directory(documents_path, recursive=True):
        if doc.error:
            logger.error(f"Failed to process {doc.file_path}: {doc.error}")
            continue

        if not doc.chunks:
            logger.warning(f"No chunks generated for {doc.file_path}")
            continue

        logger.info(
            f"Processed {doc.file_path}: {len(doc.chunks)} chunks "
            f"in {doc.processing_time:.2f}s"
        )

        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in doc.chunks]
        embeddings = await embedding_service.embed_batch(
            texts=chunk_texts,
            show_progress=False
        )

        # Prepare points for Qdrant
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=embedding.tolist(),
                payload={
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "metadata": chunk.metadata,
                }
            )
            for chunk, embedding in zip(doc.chunks, embeddings)
        ]

        # Upsert to vector database
        await vector_db.upsert_batch(
            collection_name=collection_name,
            points=points,
            batch_size=100,
            wait=True,
        )

        total_chunks += len(doc.chunks)

    logger.info(f"Ingestion completed: {total_chunks} chunks ingested")
    return total_chunks


async def setup_collections(vector_db: QdrantClient, vector_size: int = 768):
    """Create collections for hierarchical search.

    Args:
        vector_db: Vector database client.
        vector_size: Embedding dimension.
    """
    collections = ["documents_coarse", "documents_medium", "documents_fine"]

    for collection in collections:
        try:
            await vector_db.create_collection(
                collection_name=collection,
                vector_size=vector_size,
                distance=models.Distance.COSINE,
            )
            logger.info(f"Created collection: {collection}")
        except Exception as e:
            logger.warning(f"Collection {collection} may already exist: {e}")


async def example_search(
    search: MultiStageSearch,
    query: str,
) -> None:
    """Execute example search with detailed output.

    Args:
        search: Multi-stage search instance.
        query: Search query.
    """
    logger.info(f"Executing search: '{query}'")

    results, metrics = await search.search(
        query=query,
        top_k=5,
        enable_reranking=True,
    )

    logger.info(
        f"Search completed in {metrics.total_duration_ms:.2f}ms "
        f"(reranked: {metrics.reranked})"
    )

    logger.info("Stage timings:")
    for stage, duration in metrics.stages_duration_ms.items():
        logger.info(f"  {stage}: {duration:.2f}ms")

    logger.info(f"\nTop {len(results)} results:")
    for result in results:
        logger.info(
            f"{result.rank}. [{result.source_stage}] "
            f"Score: {result.score:.3f}\n"
            f"   {result.content[:200]}..."
        )


async def main():
    """Main pipeline demonstration."""

    # Configuration
    qdrant_config = QdrantConfig(
        host="localhost",
        port=6333,
        timeout=30.0,
        max_retries=3,
        connection_pool_size=100,
    )

    processing_config = ProcessingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        max_concurrent_files=10,
    )

    search_config = SearchConfig(
        coarse_top_k=50,
        medium_top_k=30,
        fine_top_k=20,
        final_top_k=10,
        enable_reranking=True,
        rerank_score_weight=0.7,
    )

    # Initialize services with context managers for automatic cleanup
    async with QdrantClient(qdrant_config) as vector_db:
        async with EmbeddingService() as embedding_service:
            async with RerankingService() as reranking_service:

                # Setup collections
                await setup_collections(vector_db)

                # Initialize document processor
                processor = EnhancedDocumentProcessor(processing_config)

                # Example 1: Ingest documents (if path exists)
                docs_path = Path("./sample_documents")
                if docs_path.exists():
                    logger.info("=== Document Ingestion ===")
                    total_chunks = await ingest_documents(
                        processor=processor,
                        embedding_service=embedding_service,
                        vector_db=vector_db,
                        documents_path=docs_path,
                        collection_name="documents_fine",
                    )
                    logger.info(f"Ingested {total_chunks} chunks\n")
                else:
                    logger.warning(f"Documents path {docs_path} not found, skipping ingestion\n")

                # Example 2: Create search engine
                logger.info("=== Multi-Stage Search ===")
                search = MultiStageSearch(
                    vector_db=vector_db,
                    embedding_service=embedding_service,
                    reranking_service=reranking_service,
                    config=search_config,
                )

                # Example 3: Execute searches
                queries = [
                    "What is machine learning?",
                    "How does neural network training work?",
                    "Explain gradient descent",
                ]

                for query in queries:
                    await example_search(search, query)
                    print()  # Blank line

                # Example 4: Batch search
                logger.info("=== Batch Search ===")
                batch_results = await search.search_batch(queries, top_k=3)

                for query, (results, metrics) in zip(queries, batch_results):
                    logger.info(
                        f"Query: '{query}' - {len(results)} results "
                        f"in {metrics.total_duration_ms:.2f}ms"
                    )

                # Example 5: Get statistics
                logger.info("\n=== Statistics ===")
                logger.info(f"Vector DB: {vector_db.get_stats()}")
                logger.info(f"Embedding Service: {embedding_service.get_stats()}")
                logger.info(f"Reranking Service: {reranking_service.get_stats()}")
                logger.info(f"Search Engine: {search.get_stats()}")

    logger.info("\n=== Pipeline Complete ===")
    logger.info("All resources cleaned up automatically via context managers")


if __name__ == "__main__":
    asyncio.run(main())
