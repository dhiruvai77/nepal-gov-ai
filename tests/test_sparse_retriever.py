"""Unit tests for NepalGov AI sparse/BM25 retrieval."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.models import Document

from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    SPARSE_VECTOR_NAME,
)
from src.retrieval.sparse_retriever import (
    BM25_MODEL,
    SparseRetriever,
)


def sample_payload() -> dict:
    """Return realistic chunk metadata used by retrieval tests."""

    return {
        "chunk_id": "constitution_chunk_00001",
        "document_id": "constitution_nepal_current_en",
        "title": "Constitution of Nepal",
        "organization": "Nepal Law Commission",
        "category": "constitution_law",
        "document_type": "constitution",
        "language": "en",
        "publication_date": "2015",
        "page_start": 12,
        "page_end": 13,
        "source_url": "https://example.gov.np/constitution",
        "chunk_text": (
            "Every person shall have the right to live with dignity."
        ),
        "extraction_method": "native",
    }


def sample_point() -> SimpleNamespace:
    """Return a minimal Qdrant-like scored point."""

    return SimpleNamespace(
        id="930a4fc2-1d87-58c8-b579-89313cf1dc00",
        score=4.25,
        payload=sample_payload(),
    )


def test_sparse_retriever_queries_bm25_vector() -> None:
    """Sparse retrieval should use Qdrant's BM25 document inference."""

    client = Mock()

    client.query_points.return_value = (
        SimpleNamespace(
            points=[
                sample_point(),
            ]
        )
    )

    retriever = SparseRetriever(
        client=client
    )

    results = retriever.retrieve(
        query="right to dignity",
        top_k=5,
    )

    assert len(results) == 1

    call_kwargs = (
        client.query_points.call_args.kwargs
    )

    assert (
        call_kwargs["collection_name"]
        == COLLECTION_NAME
    )

    assert (
        call_kwargs["using"]
        == SPARSE_VECTOR_NAME
    )

    assert (
        call_kwargs["limit"]
        == 5
    )

    assert (
        call_kwargs["with_payload"]
        is True
    )

    query_document = (
        call_kwargs["query"]
    )

    assert isinstance(
        query_document,
        Document,
    )

    assert (
        query_document.text
        == "right to dignity"
    )

    assert (
        query_document.model
        == BM25_MODEL
    )


def test_sparse_retriever_normalizes_results() -> None:
    """BM25 hits should use the shared RetrievalResult representation."""

    client = Mock()

    client.query_points.return_value = (
        SimpleNamespace(
            points=[
                sample_point(),
            ]
        )
    )

    retriever = SparseRetriever(
        client=client
    )

    result = retriever.retrieve(
        "right to dignity"
    )[0]

    assert result.chunk_id == "constitution_chunk_00001"
    assert result.document_id == "constitution_nepal_current_en"
    assert result.score == 4.25
    assert result.page_start == 12
    assert result.page_end == 13


def test_sparse_retriever_forwards_metadata_filters() -> None:
    """Sparse retrieval should support the same metadata filters as dense."""

    client = Mock()

    client.query_points.return_value = (
        SimpleNamespace(
            points=[]
        )
    )

    retriever = SparseRetriever(
        client=client
    )

    retriever.retrieve(
        query="health service",
        filters={
            "language": "en",
            "document_type": "act",
        },
    )

    query_filter = (
        client.query_points.call_args.kwargs[
            "query_filter"
        ]
    )

    assert query_filter is not None
    assert len(query_filter.must) == 2


def test_sparse_retriever_rejects_blank_query() -> None:
    """Blank queries should fail before contacting Qdrant."""

    client = Mock()

    retriever = SparseRetriever(
        client=client
    )

    with pytest.raises(
        ValueError,
        match="Query must not be blank",
    ):
        retriever.retrieve(
            "   "
        )

    client.query_points.assert_not_called()


def test_sparse_retriever_rejects_invalid_top_k() -> None:
    """A non-positive retrieval limit is invalid."""

    client = Mock()

    retriever = SparseRetriever(
        client=client
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            query="budget",
            top_k=0,
        )

    client.query_points.assert_not_called()


def test_sparse_retriever_rejects_unsupported_filter() -> None:
    """Filter validation should remain identical to dense retrieval."""

    client = Mock()

    retriever = SparseRetriever(
        client=client
    )



    # Match the exact validation wording already used by the shared
    # dense/sparse metadata-filter helper.
    with pytest.raises(
        ValueError,
        match="Unsupported retrieval filter fields",
    ):

        retriever.retrieve(
            query="education",
            filters={
                "unknown_field": "value",
            },
        )

    client.query_points.assert_not_called()