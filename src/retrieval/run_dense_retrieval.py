"""Command-line entry point for dense retrieval in NepalGov AI.

This module wires together the production hosted embedding service, Qdrant
client, and DenseRetriever. Dependency injection keeps orchestration testable
without making network calls during unit tests.
"""

from qdrant_client import QdrantClient

from src.embeddings.base import EmbeddingService
from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
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
    embedding_service: EmbeddingService | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievalResult]:
    """Execute one production dense-retrieval request.

    Dependencies can be injected by tests or future application layers. When
    omitted, the Windows-compatible hosted E5 provider and local Qdrant service
    are constructed automatically.
    """

    # Hosted E5 is the production default because Windows Smart App Control can
    # block the native PyTorch/SciPy DLLs required by the local model runtime.
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
    """Print retrieved chunks with citation-relevant metadata."""

    if not results:
        print(
            "No retrieval results found."
        )
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"[{rank}] "
            f"score={result.score:.4f} "
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
    """Run one interactive dense-retrieval query."""

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