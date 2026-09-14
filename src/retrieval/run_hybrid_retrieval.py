"""Production command-line entry point for NepalGov AI hybrid retrieval.

The entry point combines hosted multilingual E5 dense retrieval with Qdrant
BM25 sparse retrieval and fuses both ranked candidate lists using RRF.

For cross-lingual retrieval, BM25 is skipped because lexical overlap is not
reliable when the query language differs from the target document language.
"""

from qdrant_client import QdrantClient

from src.embeddings.base import (
    EmbeddingService,
)
from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.qdrant_setup import (
    QDRANT_URL,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
)
from src.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    HybridRetriever,
)
from src.retrieval.run_dense_retrieval import (
    DEFAULT_TOP_K,
    print_results,
)
from src.retrieval.sparse_retriever import (
    SparseRetriever,
)


def detect_query_language(
    query: str,
) -> str:
    """Classify a query as English or Nepali for retrieval routing.

    NepalGov AI V1 supports English and Nepali. Devanagari detection is
    sufficient for deciding whether lexical BM25 retrieval is appropriate
    for the current bilingual corpus.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "query must contain non-whitespace text."
        )

    devanagari_count = sum(
        1
        for character in clean_query
        if "\u0900" <= character <= "\u097F"
    )

    return (
        "ne"
        if devanagari_count > 0
        else "en"
    )


def run_hybrid_retrieval(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, str] | None = None,
    embedding_service: EmbeddingService | None = None,
    client: QdrantClient | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
) -> list[RetrievalResult]:
    """Execute retrieval using hybrid RRF or dense-only cross-lingual routing.

    Same-language retrieval uses dense semantic retrieval plus BM25 and fuses
    both rankings with Reciprocal Rank Fusion.

    When a language filter explicitly requests documents in a different
    language from the query, the function falls back to multilingual dense
    retrieval because BM25 depends on lexical overlap and can introduce noise.

    Dependencies can be injected by tests or future FastAPI/Streamlit layers.
    """

    # Use the same hosted E5 provider as ingestion so query and document vectors
    # remain in the same multilingual embedding space.
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

    dense_retriever = DenseRetriever(
        client=active_client,
        embedding_service=active_embedding_service,
    )

    query_language = detect_query_language(
        query
    )

    target_language = (
        filters.get("language")
        if filters
        else None
    )

    # BM25 is a lexical retriever. When the query language and requested corpus
    # language differ, sparse matches can be dominated by accidental token
    # overlap. Multilingual E5 is specifically suited to this cross-lingual case.
    if (
        target_language is not None
        and target_language != query_language
    ):
        return dense_retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    sparse_retriever = SparseRetriever(
        client=active_client,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rrf_k=rrf_k,
        candidate_multiplier=candidate_multiplier,
    )

    return hybrid_retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
    )


def main() -> None:
    """Run one interactive production retrieval query."""

    query = input(
        "Enter a Nepal government question: "
    ).strip()

    results = run_hybrid_retrieval(
        query=query,
    )

    print_results(
        results
    )


if __name__ == "__main__":
    main()