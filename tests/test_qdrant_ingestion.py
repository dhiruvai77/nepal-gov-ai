"""Unit tests for NepalGov AI Qdrant ingestion."""

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.models import Document

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.indexing.qdrant_ingestion import (
    BM25_MODEL,
    build_payload,
    build_points,
    ingest_chunks,
    stable_point_id,
    validate_chunk,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Deterministic embedding provider used to isolate ingestion tests."""

    @property
    def dimension(self) -> int:
        """Return the test embedding dimension."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return one predictable dense vector for each passage."""

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Return a deterministic query vector for interface completeness."""

        return [1.0, 0.0, 0.0]


class WrongDimensionEmbeddingService(
    FakeEmbeddingService
):
    """Embedding provider that intentionally violates the vector contract."""

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return vectors with the wrong dimension."""

        return [
            [1.0, 0.0]
            for _ in texts
        ]


class WrongCountEmbeddingService(
    FakeEmbeddingService
):
    """Embedding provider that returns fewer vectors than requested."""

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return only one vector regardless of input size."""

        return [
            [1.0, 0.0, 0.0]
        ]


def sample_chunk(
    chunk_id: str = "constitution_chunk_00001",
) -> dict:
    """Return one realistic processed child chunk."""

    return {
        "chunk_id": chunk_id,
        "parent_chunk_id": None,
        "document_id": "constitution_nepal_current_en",
        "title": "Constitution of Nepal",
        "organization": "Nepal Law Commission",
        "category": "constitution_law",
        "document_type": "constitution",
        "language": "en",
        "publication_date": "2015",
        "source_url": "https://example.gov.np/constitution",
        "download_url": "https://example.gov.np/constitution.pdf",
        "retrieved_at": "2026-09-01T12:00:00Z",
        "page_start": 12,
        "page_end": 13,
        "section": None,
        "subsection": None,
        "article_number": "16",
        "article_title": "Right to live with dignity",
        "chunk_index": 1,
        "chunk_text": (
            "Every person shall have the right to live with dignity."
        ),
        "token_count": 14,
        "extraction_method": "native",
    }


def test_stable_point_id_is_deterministic() -> None:
    """The same chunk ID should always map to the same Qdrant point ID."""

    first = stable_point_id(
        "chunk-123"
    )

    second = stable_point_id(
        "chunk-123"
    )

    assert first == second


def test_stable_point_id_changes_for_different_chunks() -> None:
    """Different chunk IDs must not collapse onto the same point ID."""

    assert (
        stable_point_id(
            "chunk-123"
        )
        != stable_point_id(
            "chunk-456"
        )
    )


def test_stable_point_id_rejects_blank_value() -> None:
    """Blank chunk IDs cannot produce stable point identifiers."""

    with pytest.raises(
        ValueError,
        match="chunk_id must not be blank",
    ):
        stable_point_id(
            "   "
        )


def test_validate_chunk_accepts_valid_chunk() -> None:
    """A complete chunk should satisfy the ingestion contract."""

    validate_chunk(
        sample_chunk()
    )


def test_validate_chunk_rejects_missing_required_field() -> None:
    """Required retrieval/citation metadata must be present."""

    chunk = sample_chunk()

    del chunk[
        "source_url"
    ]

    with pytest.raises(
        ValueError,
        match="source_url",
    ):
        validate_chunk(
            chunk
        )


def test_validate_chunk_rejects_blank_text() -> None:
    """Chunks without usable text should never be indexed."""

    chunk = sample_chunk()

    chunk[
        "chunk_text"
    ] = "   "

    with pytest.raises(
        ValueError,
        match="chunk_text must not be blank",
    ):
        validate_chunk(
            chunk
        )


def test_build_payload_preserves_chunk_metadata() -> None:
    """The Qdrant payload should retain citation and retrieval metadata."""

    chunk = sample_chunk()

    payload = build_payload(
        chunk
    )

    assert payload == chunk
    assert (
        payload["article_number"]
        == "16"
    )
    assert (
        payload["source_url"]
        == "https://example.gov.np/constitution"
    )


def test_build_points_uses_embedding_service() -> None:
    """Dense vectors should come from the injected provider."""

    service = FakeEmbeddingService()

    points = build_points(
        chunks=[
            sample_chunk(),
        ],
        embedding_service=service,
    )

    assert len(
        points
    ) == 1

    assert (
        points[0].vector[
            DENSE_VECTOR_NAME
        ]
        == [1.0, 0.0, 0.0]
    )


def test_build_points_adds_bm25_sparse_document() -> None:
    """Each point should contain a BM25 sparse representation request."""

    service = FakeEmbeddingService()

    chunk = sample_chunk()

    point = build_points(
        chunks=[
            chunk,
        ],
        embedding_service=service,
    )[0]

    sparse_value = point.vector[
        SPARSE_VECTOR_NAME
    ]

    assert isinstance(
        sparse_value,
        Document,
    )

    assert (
        sparse_value.text
        == chunk["chunk_text"]
    )

    assert (
        sparse_value.model
        == BM25_MODEL
    )


def test_build_points_preserves_payload() -> None:
    """Point payload should retain the original chunk metadata."""

    service = FakeEmbeddingService()

    chunk = sample_chunk()

    point = build_points(
        chunks=[
            chunk,
        ],
        embedding_service=service,
    )[0]

    assert (
        point.payload[
            "chunk_id"
        ]
        == chunk["chunk_id"]
    )

    assert (
        point.payload[
            "document_id"
        ]
        == chunk["document_id"]
    )


def test_build_points_rejects_wrong_embedding_dimension() -> None:
    """Dense vectors must match the provider's declared dimension."""

    service = (
        WrongDimensionEmbeddingService()
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding dimension mismatch",
    ):
        build_points(
            chunks=[
                sample_chunk(),
            ],
            embedding_service=service,
        )


def test_build_points_rejects_wrong_vector_count() -> None:
    """Embedding providers must return one vector per chunk."""

    service = (
        WrongCountEmbeddingService()
    )

    with pytest.raises(
        RuntimeError,
        match="different number of vectors",
    ):
        build_points(
            chunks=[
                sample_chunk(
                    "chunk-1"
                ),
                sample_chunk(
                    "chunk-2"
                ),
            ],
            embedding_service=service,
        )


def test_ingest_chunks_batches_upserts() -> None:
    """Ingestion should upsert bounded batches rather than all points at once."""

    client = Mock()
    service = FakeEmbeddingService()

    chunks = [
        sample_chunk(
            "chunk-1"
        ),
        sample_chunk(
            "chunk-2"
        ),
        sample_chunk(
            "chunk-3"
        ),
    ]

    inserted = ingest_chunks(
        client=client,
        chunks=chunks,
        embedding_service=service,
        batch_size=2,
    )

    assert inserted == 3

    # Three chunks with a batch size of two should produce two Qdrant writes.
    assert (
        client.upsert.call_count
        == 2
    )

    first_call = (
        client.upsert.call_args_list[
            0
        ].kwargs
    )

    assert (
        first_call["collection_name"]
        == COLLECTION_NAME
    )

    assert (
        first_call["wait"]
        is True
    )

    assert len(
        first_call["points"]
    ) == 2


def test_ingest_chunks_rejects_invalid_batch_size() -> None:
    """A non-positive batch size cannot define valid ingestion batches."""

    client = Mock()
    service = FakeEmbeddingService()

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        ingest_chunks(
            client=client,
            chunks=[
                sample_chunk(),
            ],
            embedding_service=service,
            batch_size=0,
        )

    client.upsert.assert_not_called()