"""Manual end-to-end smoke test for dense + BM25 retrieval.

This script intentionally indexes only five real processed chunks. It validates
the hosted E5 embedding service, Qdrant hybrid-ready ingestion, dense retrieval,
and sparse BM25 retrieval before the full 2,276-chunk corpus is ingested.
"""

from qdrant_client import QdrantClient

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.qdrant_ingestion import (
    QDRANT_URL,
    ingest_chunks,
    load_chunks,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    ensure_collection,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
)
from src.retrieval.sparse_retriever import (
    SparseRetriever,
)


SMOKE_CHUNK_COUNT = 5
TOP_K = 3

# Use a query that should be relevant to the first processed Budget Speech
# chunks selected by deterministic chunk loading.
TEST_QUERY = "government budget and expenditure"


def print_results(
    label: str,
    results,
) -> None:
    """Print a compact retrieval summary for manual validation."""

    print(
        f"\n{label}"
    )
    print(
        "-" * len(label)
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. score={result.score:.4f} "
            f"title={result.title} "
            f"pages={result.page_start}-{result.page_end}"
        )

        # A short text preview is enough to confirm that the returned payload
        # corresponds to a meaningful government-document chunk.
        preview = (
            result.chunk_text
            .replace("\n", " ")
            [:250]
        )

        print(
            f"   {preview}"
        )


def main() -> None:
    """Run the controlled five-chunk ingestion and retrieval smoke test."""

    client = QdrantClient(
        url=QDRANT_URL
    )

    ensure_collection(
        client
    )

    all_chunks = load_chunks()

    smoke_chunks = all_chunks[
        :SMOKE_CHUNK_COUNT
    ]

    print(
        f"Loaded {len(all_chunks)} total chunks."
    )
    print(
        f"Using {len(smoke_chunks)} chunks for smoke testing."
    )

    # Hosted inference is used here intentionally so the smoke test exercises
    # the same dense embedding path planned for production ingestion.
    embedding_service = (
        HuggingFaceE5EmbeddingService(
            batch_size=SMOKE_CHUNK_COUNT
        )
    )

    inserted_count = ingest_chunks(
        client=client,
        chunks=smoke_chunks,
        embedding_service=embedding_service,
        batch_size=SMOKE_CHUNK_COUNT,
    )

    print(
        f"Inserted {inserted_count} real dense + BM25 points."
    )

    dense_retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )

    sparse_retriever = SparseRetriever(
        client=client
    )

    dense_results = dense_retriever.retrieve(
        TEST_QUERY,
        top_k=TOP_K,
    )

    sparse_results = sparse_retriever.retrieve(
        TEST_QUERY,
        top_k=TOP_K,
    )

    print_results(
        "Dense retrieval",
        dense_results,
    )

    print_results(
        "BM25 retrieval",
        sparse_results,
    )

    print(
        f"\nSmoke test complete. "
        f"Temporary points remain in '{COLLECTION_NAME}' "
        "for inspection."
    )


if __name__ == "__main__":
    main()