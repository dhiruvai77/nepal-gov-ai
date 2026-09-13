"""Sparse/BM25 retrieval for NepalGov AI.

This module sends raw query text to Qdrant's BM25 inference path, searches the
named sparse vector, and returns the same RetrievalResult structure used by
dense retrieval so both paths can later be fused with RRF.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Document

from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    SPARSE_VECTOR_NAME,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
    build_metadata_filter,
    normalize_search_result,
)


# Qdrant's built-in BM25 sparse model identifier.
BM25_MODEL = "Qdrant/bm25"


class SparseRetriever:
    """Retrieve chunks through Qdrant's BM25 sparse-vector path."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """Store the Qdrant dependency and target collection."""

        self.client = client
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the highest-scoring lexical matches for one query."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Query must not be blank."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Reuse the exact metadata-filter behavior from dense retrieval so both
        # retrieval paths remain compatible before hybrid fusion.
        query_filter = build_metadata_filter(
            filters
        )

        # Document instructs Qdrant to convert the raw query text into its BM25
        # sparse representation server-side rather than requiring a local model.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=Document(
                text=clean_query,
                model=BM25_MODEL,
            ),
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        # Sparse and dense retrieval intentionally return the same normalized
        # result type. This will make RRF fusion independent of Qdrant internals.
        return [
            normalize_search_result(point)
            for point in response.points
        ]