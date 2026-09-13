"""Command-line entry point for dense retrieval in NepalGov AI.

This module wires together the production embedding service, Qdrant client,
and DenseRetriever. Dependency injection keeps the orchestration testable
without requiring the local PyTorch runtime.
"""

from qdrant_client import QdrantClient

from src.embeddings.e5_service import E5EmbeddingService
from src.indexing.qdrant_setup import QDRANT_URL
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
)


DEFAULT_TOP_K = 5


def run_dense_retrieval(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, str] | None = None,
    embedding_service: E5EmbeddingService | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievalResult]:
    """Execute one dense retrieval request.

    Callers may inject dependencies so tests avoid loading E5 or connecting to
    a real Qdrant instance.
    """

    # Construct production dependencies only when they are not supplied by the
    # caller. This preserves a clean CLI path while keeping tests lightweight.
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

    retriever = DenseRetriever(
        client=active_client,
        embedding_service=active_embedding_service,
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
    )


def print_results(
    results: list[RetrievalResult],
) -> None:
    """Print retrieved chunks in a compact human-readable format."""

    if not results:
        print("No retrieval results found.")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"[{rank}] score={result.score:.4f} "
            f"title={result.title} "
            f"pages={result.page_start}-{result.page_end}"
        )
        print(
            f"    chunk_id={result.chunk_id}"
        )
        print(
            f"    source={result.source_url}"
        )
        print(
            f"    text={result.chunk_text}"
        )


def main() -> None:
    """Run an interactive dense retrieval query from the command line."""

    # A simple prompt is sufficient for the current development-stage CLI.
    # API and Streamlit interfaces will later call the same retrieval service.
    query = input(
        "Enter a Nepal government question: "
    ).strip()

    results = run_dense_retrieval(
        query=query,
    )

    print_results(
        results
    )


if __name__ == "__main__":
    main()