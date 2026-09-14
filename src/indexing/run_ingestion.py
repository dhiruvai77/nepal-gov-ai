"""Command-line entry point for NepalGov AI production ingestion.

This module wires together the processed chunk loader, embedding service, and
Qdrant client. The actual embedding, validation, BM25 construction, and upsert
logic remain in the dedicated embedding and indexing modules.
"""

from qdrant_client import QdrantClient

from src.embeddings.base import EmbeddingService
from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.qdrant_ingestion import (
    COLLECTION_NAME,
    QDRANT_URL,
    ingest_chunks,
    load_chunks,
)
from src.indexing.qdrant_setup import ensure_collection


DEFAULT_BATCH_SIZE = 64


def run_ingestion(
    embedding_service: EmbeddingService | None = None,
    client: QdrantClient | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Run the production dense + BM25 ingestion pipeline.

    Dependency injection is supported so tests can provide fake embedding
    services and Qdrant clients without making network requests or requiring
    a running Qdrant server.

    Hosted Hugging Face E5 inference is currently the default production
    embedding backend because the local Windows environment cannot load the
    PyTorch runtime under Smart App Control. The pipeline still depends only
    on the generic EmbeddingService interface, so this default can be changed
    later without modifying the ingestion layer.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    # Construct production dependencies only when callers do not inject them.
    # Hosted E5 avoids the current local PyTorch DLL restriction while keeping
    # the same multilingual-e5-large-instruct model and 1024-vector contract.
    active_embedding_service = (
        embedding_service
        if embedding_service is not None
        else HuggingFaceE5EmbeddingService()
    )

    active_client = (
        client
        if client is not None
        else QdrantClient(
            url=QDRANT_URL
        )
    )

    # Ensure the collection contains the required dense schema, BM25 sparse
    # schema, and payload indexes before attempting production upserts.
    ensure_collection(
        active_client
    )

    chunks = load_chunks()

    total_ingested = ingest_chunks(
        client=active_client,
        embedding_service=active_embedding_service,
        chunks=chunks,
        batch_size=batch_size,
    )

    return total_ingested


def main() -> None:
    """Execute production ingestion from the command line."""

    total_ingested = run_ingestion()

    print(
        f"Ingestion complete: {total_ingested} chunks "
        f"stored in '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()