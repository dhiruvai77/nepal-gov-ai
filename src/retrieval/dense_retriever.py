"""Dense-vector retrieval for NepalGov AI.

This module embeds a user query through the generic EmbeddingService interface,
optionally applies metadata filters, searches a selected Qdrant named dense
vector, and returns normalized retrieval results for downstream ranking and
RAG stages.
"""

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.embeddings.base import EmbeddingService
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)


# These fields have keyword payload indexes in Qdrant and are therefore safe
# to expose through the dense retriever's metadata-filter interface.
FILTERABLE_FIELDS = {
    "document_id",
    "language",
    "category",
    "document_type",
    "organization",
}

# Dense retrieval currently supports the original raw chunk embedding and the
# metadata-contextualized experimental representation. Keeping this explicit
# prevents accidental queries against unrelated named vectors such as BM25.
SUPPORTED_DENSE_VECTOR_NAMES = {
    DENSE_VECTOR_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
}


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized result returned by the dense retrieval layer."""

    point_id: str
    score: float
    chunk_id: str
    document_id: str
    title: str
    organization: str
    language: str
    page_start: int
    page_end: int
    source_url: str
    chunk_text: str

    # Optional metadata is retained because later retrieval stages may use it
    # for reranking, filtering, citation display, or context selection.
    category: str | None = None
    document_type: str | None = None
    publication_date: str | None = None
    section: str | None = None
    subsection: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    extraction_method: str | None = None


def build_metadata_filter(
    filters: dict[str, str] | None,
) -> Filter | None:
    """Convert supported metadata filters into a Qdrant Filter.

    Multiple filter fields are combined with Qdrant's ``must`` semantics,
    meaning every supplied condition must match.
    """

    if not filters:
        return None

    unsupported_fields = sorted(
        set(filters) - FILTERABLE_FIELDS
    )

    if unsupported_fields:
        raise ValueError(
            "Unsupported retrieval filter fields: "
            + ", ".join(unsupported_fields)
        )

    conditions: list[FieldCondition] = []

    for field_name, value in filters.items():
        clean_value = str(value).strip()

        if not clean_value:
            raise ValueError(
                f"Filter '{field_name}' must contain "
                "non-whitespace text."
            )

        # All currently supported metadata indexes use Qdrant's keyword type,
        # so exact MatchValue filtering is appropriate here.
        conditions.append(
            FieldCondition(
                key=field_name,
                match=MatchValue(
                    value=clean_value
                ),
            )
        )

    return Filter(
        must=conditions
    )


def normalize_search_result(
    point: Any,
) -> RetrievalResult:
    """Convert one Qdrant scored point into the project result contract."""

    payload = point.payload or {}

    required_payload_fields = (
        "chunk_id",
        "document_id",
        "title",
        "organization",
        "language",
        "page_start",
        "page_end",
        "source_url",
        "chunk_text",
    )

    missing_fields = [
        field_name
        for field_name in required_payload_fields
        if payload.get(field_name) is None
    ]

    if missing_fields:
        raise RuntimeError(
            "Qdrant result is missing required payload fields: "
            + ", ".join(sorted(missing_fields))
        )

    # Returning an explicit result object prevents downstream RAG code from
    # depending directly on Qdrant's client-specific response models.
    return RetrievalResult(
        point_id=str(point.id),
        score=float(point.score),
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        title=str(payload["title"]),
        organization=str(payload["organization"]),
        language=str(payload["language"]),
        page_start=int(payload["page_start"]),
        page_end=int(payload["page_end"]),
        source_url=str(payload["source_url"]),
        chunk_text=str(payload["chunk_text"]),
        category=payload.get("category"),
        document_type=payload.get("document_type"),
        publication_date=payload.get(
            "publication_date"
        ),
        section=payload.get("section"),
        subsection=payload.get("subsection"),
        article_number=payload.get(
            "article_number"
        ),
        article_title=payload.get(
            "article_title"
        ),
        extraction_method=payload.get(
            "extraction_method"
        ),
    )


class DenseRetriever:
    """Retrieve semantically similar government-document chunks from Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        embedding_service: EmbeddingService,
        collection_name: str = COLLECTION_NAME,
        vector_name: str = DENSE_VECTOR_NAME,
    ) -> None:
        """Store retrieval dependencies and select the named dense vector.

        The default remains the original raw dense representation so adding
        contextual embeddings does not silently change production retrieval.
        """

        if (
            vector_name
            not in SUPPORTED_DENSE_VECTOR_NAMES
        ):
            raise ValueError(
                "Unsupported dense vector name: "
                f"{vector_name}. Supported values: "
                + ", ".join(
                    sorted(
                        SUPPORTED_DENSE_VECTOR_NAMES
                    )
                )
            )

        self.client = client
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.vector_name = vector_name

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the highest-scoring dense matches for a user query."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "query must contain non-whitespace text."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Query formatting and normalization remain the responsibility of the
        # active embedding provider. For E5 this adds the retrieval instruction.
        query_vector = (
            self.embedding_service.embed_query(
                clean_query
            )
        )

        if (
            len(query_vector)
            != self.embedding_service.dimension
        ):
            raise RuntimeError(
                "Embedding service returned a query vector "
                f"with {len(query_vector)} dimensions; expected "
                f"{self.embedding_service.dimension}."
            )

        query_filter = build_metadata_filter(
            filters
        )

        # `using` selects either the baseline raw dense vector or the
        # contextual dense representation. Raw dense remains the default.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            normalize_search_result(point)
            for point in response.points
        ]