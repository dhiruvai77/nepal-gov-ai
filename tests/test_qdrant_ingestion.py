"""Unit tests for production Qdrant ingestion logic."""

import uuid
from collections.abc import Sequence

import pytest

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.indexing.qdrant_ingestion import (
    build_payload,
    build_points,
    stable_point_id,
    validate_chunk,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Small deterministic embedding provider for unit tests."""

    @property
    def dimension(self) -> int:
        """Use a tiny vector size so tests remain lightweight."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return one deterministic vector per input passage."""

        return [
            [
                float(index + 1),
                0.0,
                0.0,
            ]
            for index, _ in enumerate(texts)
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
    """Fake provider that deliberately returns malformed vector dimensions."""

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return vectors that violate the declared three-dimensional schema."""

        return [
            [1.0, 0.0]
            for _ in texts
        ]


class WrongCountEmbeddingService(
    FakeEmbeddingService
):
    """Fake provider that returns fewer vectors than input passages."""

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return no vectors to simulate a broken embedding provider."""

        return []


def sample_chunk() -> dict:
    """Return a minimal valid production chunk."""

    return {
        "chunk_id": "constitution_chunk_00001",
        "parent_chunk_id": None,
        "document_id": "constitution_nepal_current_en",
        "title": "Constitution of Nepal",
        "organization": "Nepal Law Commission",
        "category": "constitution_law",
        "document_type": "constitution",
        "language": "en",
        "publication_date": None,
        "source_url": "https://example.gov.np/constitution",
        "download_url": "https://example.gov.np/constitution.pdf",
        "retrieved_at": "2026-09-13T00:00:00+00:00",
        "page_start": 1,
        "page_end": 1,
        "section": None,
        "subsection": None,
        "article_number": None,
        "article_title": None,
        "chunk_index": 0,
        "chunk_text": (
            "Every person has the right to live with dignity."
        ),
        "token_count": 12,
        "extraction_method": "native",
    }


def test_stable_point_id_is_deterministic() -> None:
    """The same chunk ID must always map to the same Qdrant UUID."""

    first = stable_point_id(
        "constitution_chunk_00001"
    )
    second = stable_point_id(
        "constitution_chunk_00001"
    )

    assert first == second

    # Parsing the value verifies that we produce a canonical UUID rather than
    # an arbitrary opaque string.
    assert str(uuid.UUID(first)) == first


def test_stable_point_id_changes_for_different_chunks() -> None:
    """Different chunks should produce different deterministic IDs."""

    first = stable_point_id(
        "constitution_chunk_00001"
    )
    second = stable_point_id(
        "constitution_chunk_00002"
    )

    assert first != second


def test_stable_point_id_rejects_blank_chunk_id() -> None:
    """A point ID must never be generated from an empty chunk identifier."""

    with pytest.raises(
        ValueError,
        match="chunk_id must contain non-whitespace text.",
    ):
        stable_point_id("   ")


def test_validate_chunk_accepts_valid_chunk() -> None:
    """A complete chunk should pass validation without raising."""

    validate_chunk(
        sample_chunk()
    )


def test_validate_chunk_rejects_missing_required_field() -> None:
    """Citation-critical metadata must be present before ingestion."""

    chunk = sample_chunk()
    del chunk["source_url"]

    with pytest.raises(
        ValueError,
        match="source_url",
    ):
        validate_chunk(chunk)


def test_validate_chunk_rejects_empty_text() -> None:
    """Chunks with no retrievable content must not enter Qdrant."""

    chunk = sample_chunk()
    chunk["chunk_text"] = "   "

    with pytest.raises(
        ValueError,
        match="empty chunk_text",
    ):
        validate_chunk(chunk)


def test_build_payload_preserves_citation_metadata() -> None:
    """Qdrant payloads must retain fields needed for answer citations."""

    payload = build_payload(
        sample_chunk()
    )

    assert payload["chunk_id"] == (
        "constitution_chunk_00001"
    )
    assert payload["title"] == (
        "Constitution of Nepal"
    )
    assert payload["page_start"] == 1
    assert payload["page_end"] == 1
    assert payload["source_url"] == (
        "https://example.gov.np/constitution"
    )


def test_build_points_uses_embedding_service() -> None:
    """Production point construction should remain provider-independent."""

    chunks = [
        sample_chunk(),
    ]

    service = FakeEmbeddingService()

    points = build_points(
        chunks,
        service,
    )

    assert len(points) == 1
    assert points[0].id == stable_point_id(
        chunks[0]["chunk_id"]
    )

    assert points[0].vector == {
        "dense": [1.0, 0.0, 0.0]
    }

    assert points[0].payload[
        "document_id"
    ] == "constitution_nepal_current_en"


def test_build_points_rejects_wrong_vector_dimension() -> None:
    """Vectors incompatible with the provider schema must fail before Qdrant."""

    with pytest.raises(
        RuntimeError,
        match="produced 2 dimensions; expected 3",
    ):
        build_points(
            [sample_chunk()],
            WrongDimensionEmbeddingService(),
        )


def test_build_points_rejects_wrong_vector_count() -> None:
    """Every input chunk must receive exactly one embedding."""

    with pytest.raises(
        RuntimeError,
        match="different number of vectors",
    ):
        build_points(
            [sample_chunk()],
            WrongCountEmbeddingService(),
        )