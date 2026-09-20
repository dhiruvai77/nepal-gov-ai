"""Unit tests for NepalGov AI dense-vector retrieval."""

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
    build_metadata_filter,
    normalize_search_result,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Deterministic lightweight embedding provider for retrieval tests."""

    @property
    def dimension(self) -> int:
        """Use three dimensions so tests remain independent from E5."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return valid passage vectors for interface completeness."""

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Return a deterministic query vector."""

        return [0.8, 0.1, 0.1]


class WrongDimensionEmbeddingService(
    FakeEmbeddingService
):
    """Fake provider that deliberately returns a malformed query vector."""

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Return fewer dimensions than declared by the provider."""

        return [1.0, 0.0]


def sample_payload() -> dict:
    """Return representative metadata from an indexed government chunk."""

    return {
        "chunk_id": "constitution_chunk_00001",
        "document_id": "constitution_nepal_current_en",
        "title": "Constitution of Nepal",
        "organization": "Nepal Law Commission",
        "category": "constitution_law",
        "document_type": "constitution",
        "language": "en",
        "publication_date": None,
        "page_start": 12,
        "page_end": 13,
        "section": "Fundamental Rights",
        "subsection": None,
        "article_number": "16",
        "article_title": "Right to live with dignity",
        "source_url": "https://example.gov.np/constitution",
        "chunk_text": (
            "Every person shall have the right to live with dignity."
        ),
        "chunk_index": 1,
        "token_count": 14,
        "extraction_method": "native",
    }


def sample_point(
    score: float = 0.91,
) -> SimpleNamespace:
    """Return a lightweight stand-in for a Qdrant scored point."""

    return SimpleNamespace(
        id="930a4fc2-1d87-58c8-b579-89313cf1dc00",
        score=score,
        payload=sample_payload(),
    )


def test_build_metadata_filter_returns_none_without_filters() -> None:
    """An unfiltered search should not create an unnecessary Qdrant filter."""

    assert build_metadata_filter(None) is None
    assert build_metadata_filter({}) is None


def test_build_metadata_filter_builds_keyword_conditions() -> None:
    """Supported metadata fields should become exact-match conditions."""

    result = build_metadata_filter(
        {
            "language": "en",
            "document_type": "constitution",
        }
    )

    assert result is not None
    assert result.must is not None
    assert len(result.must) == 2

    assert (
        result.must[0].key
        == "language"
    )

    assert (
        result.must[0].match.value
        == "en"
    )

    assert (
        result.must[1].key
        == "document_type"
    )

    assert (
        result.must[1].match.value
        == "constitution"
    )


def test_build_metadata_filter_rejects_unsupported_field() -> None:
    """Retrieval should not silently filter on non-indexed metadata."""

    with pytest.raises(
        ValueError,
        match="Unsupported retrieval filter fields",
    ):
        build_metadata_filter(
            {
                "unknown_field": "value",
            }
        )


def test_build_metadata_filter_rejects_blank_value() -> None:
    """Keyword filters require an actual value rather than whitespace."""

    with pytest.raises(
        ValueError,
        match="must contain non-whitespace text",
    ):
        build_metadata_filter(
            {
                "language": "   ",
            }
        )


def test_normalize_search_result_preserves_citation_metadata() -> None:
    """Normalized results must retain evidence needed for downstream RAG."""

    result = normalize_search_result(
        sample_point()
    )

    assert isinstance(
        result,
        RetrievalResult,
    )

    assert (
        result.score
        == pytest.approx(
            0.91
        )
    )

    assert (
        result.chunk_id
        == "constitution_chunk_00001"
    )

    assert (
        result.document_id
        == "constitution_nepal_current_en"
    )

    assert (
        result.title
        == "Constitution of Nepal"
    )

    assert (
        result.organization
        == "Nepal Law Commission"
    )

    assert (
        result.page_start
        == 12
    )

    assert (
        result.page_end
        == 13
    )

    assert (
        result.article_number
        == "16"
    )

    assert (
        result.source_url
        == "https://example.gov.np/constitution"
    )

    assert (
        result.chunk_index
        == 1
    )

    assert (
        result.token_count
        == 14
    )


def test_normalize_search_result_allows_missing_optional_context_metadata() -> None:
    """Older or alternate payloads may omit context-selection metadata."""

    point = sample_point()

    del point.payload[
        "chunk_index"
    ]

    del point.payload[
        "token_count"
    ]

    result = normalize_search_result(
        point
    )

    assert (
        result.chunk_index
        is None
    )

    assert (
        result.token_count
        is None
    )


def test_normalize_search_result_rejects_incomplete_payload() -> None:
    """Malformed indexed points should fail instead of hiding bad metadata."""

    point = sample_point()

    del point.payload[
        "source_url"
    ]

    with pytest.raises(
        RuntimeError,
        match="source_url",
    ):
        normalize_search_result(
            point
        )


def test_dense_retriever_queries_named_dense_vector() -> None:
    """Default dense retrieval should continue using the raw dense vector."""

    client = Mock()

    client.query_points.return_value = (
        SimpleNamespace(
            points=[
                sample_point(),
            ]
        )
    )

    embedding_service = FakeEmbeddingService()

    retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )

    results = retriever.retrieve(
        query=(
            "What does the Constitution "
            "say about dignity?"
        ),
        top_k=5,
    )

    assert len(
        results
    ) == 1

    assert (
        results[0].score
        == pytest.approx(
            0.91
        )
    )

    client.query_points.assert_called_once()

    call_kwargs = (
        client.query_points.call_args.kwargs
    )

    assert (
        call_kwargs[
            "collection_name"
        ]
        == COLLECTION_NAME
    )

    assert (
        call_kwargs[
            "query"
        ]
        == [
            0.8,
            0.1,
            0.1,
        ]
    )

    # Backward compatibility is important: creating a DenseRetriever without
    # selecting a vector must still query the original production baseline.
    assert (
        call_kwargs[
            "using"
        ]
        == DENSE_VECTOR_NAME
    )

    assert (
        call_kwargs[
            "query_filter"
        ]
        is None
    )

    assert (
        call_kwargs[
            "limit"
        ]
        == 5
    )

    assert (
        call_kwargs[
            "with_payload"
        ]
        is True
    )


def test_dense_retriever_can_query_contextual_vector() -> None:
    """Evaluation should be able to select contextual dense retrieval."""

    client = Mock()

    client.query_points.return_value = (
        SimpleNamespace(
            points=[
                sample_point(),
            ]
        )
    )

    retriever = DenseRetriever(
        client=client,
        embedding_service=FakeEmbeddingService(),
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
    )

    results = retriever.retrieve(
        query=(
            "What does the Constitution "
            "say about health?"
        ),
        top_k=5,
    )

    assert len(
        results
    ) == 1

    call_kwargs = (
        client.query_points.call_args.kwargs
    )

    # The query embedding itself is unchanged. Only the named Qdrant passage
    # representation against which it is compared should differ.
    assert (
        call_kwargs[
            "using"
        ]
        == CONTEXTUAL_DENSE_VECTOR_NAME
    )

    assert (
        call_kwargs[
            "query"
        ]
        == [
            0.8,
            0.1,
            0.1,
        ]
    )


def test_dense_retriever_rejects_unknown_vector_name() -> None:
    """Dense retrieval should not silently query an unintended named vector."""

    client = Mock()

    with pytest.raises(
        ValueError,
        match="Unsupported dense vector name",
    ):
        DenseRetriever(
            client=client,
            embedding_service=FakeEmbeddingService(),
            vector_name="unknown_dense_vector",
        )

    # Invalid configuration should fail during construction before any search
    # request reaches Qdrant.
    client.query_points.assert_not_called()


def test_dense_retriever_applies_metadata_filters() -> None:
    """Retriever filters should be forwarded to Qdrant."""

    client = Mock()

    client.query_points.return_value = (
        SimpleNamespace(
            points=[]
        )
    )

    retriever = DenseRetriever(
        client=client,
        embedding_service=FakeEmbeddingService(),
    )

    results = retriever.retrieve(
        query="health policy",
        filters={
            "language": "en",
            "category": "health_population",
        },
    )

    assert results == []

    query_filter = (
        client.query_points.call_args.kwargs[
            "query_filter"
        ]
    )

    assert query_filter is not None
    assert query_filter.must is not None

    assert len(
        query_filter.must
    ) == 2


def test_dense_retriever_rejects_blank_query() -> None:
    """Blank user input should fail before embedding or Qdrant access."""

    client = Mock()

    retriever = DenseRetriever(
        client=client,
        embedding_service=FakeEmbeddingService(),
    )

    with pytest.raises(
        ValueError,
        match="query must contain non-whitespace text",
    ):
        retriever.retrieve(
            query="   "
        )

    client.query_points.assert_not_called()


def test_dense_retriever_rejects_invalid_top_k() -> None:
    """Retrieval requires at least one requested result."""

    client = Mock()

    retriever = DenseRetriever(
        client=client,
        embedding_service=FakeEmbeddingService(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            query="Nepal Constitution",
            top_k=0,
        )

    client.query_points.assert_not_called()


def test_dense_retriever_rejects_wrong_query_dimension() -> None:
    """Malformed query embeddings must never be sent to Qdrant."""

    client = Mock()

    retriever = DenseRetriever(
        client=client,
        embedding_service=(
            WrongDimensionEmbeddingService()
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="2 dimensions; expected 3",
    ):
        retriever.retrieve(
            query="education policy"
        )

    client.query_points.assert_not_called()