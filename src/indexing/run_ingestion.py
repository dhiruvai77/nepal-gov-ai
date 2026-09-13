"""Command-line entry point for NepalGov AI dense-vector ingestion.

This module wires together the processed chunk loader, embedding service, and
Qdrant client. The actual embedding, validation, and upsert logic remains in
the dedicated service and ingestion modules.
"""

from qdrant_client import QdrantClient

from src.embeddings.e5_service import E5EmbeddingService
from src.indexing.qdrant_ingestion import (
    COLLECTION_NAME,
    QDRANT_URL,
    ingest_chunks,
    load_chunks,
)
from src.indexing.qdrant_setup import ensure_collection


DEFAULT_BATCH_SIZE = 64


def run_ingestion(
    embedding_service: E5EmbeddingService | None = None,
    client: QdrantClient | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Run the production dense-vector ingestion pipeline.

    Dependency injection is supported so tests can provide fake embedding
    services and Qdrant clients without loading PyTorch or requiring a running
    Qdrant server.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    # Construct production dependencies only when callers do not provide them.
    # This keeps the entry point usable in tests despite the current Windows
    # restriction that prevents the local PyTorch runtime from loading.
    active_embedding_service = (
        embedding_service
        if embedding_service is not None
        else E5EmbeddingService()
    )

    active_client = (
        client
        if client is not None
        else QdrantClient(url=QDRANT_URL)
    )

    # Ensure the dense collection exists before attempting any upserts.
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