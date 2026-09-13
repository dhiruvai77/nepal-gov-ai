"""Unit tests for the production ingestion entry point."""

from collections.abc import Sequence
from unittest.mock import Mock, patch

import pytest

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.indexing.run_ingestion import run_ingestion


class FakeEmbeddingService(
    EmbeddingService
):
    """Minimal embedding provider for orchestration tests."""

    @property
    def dimension(self) -> int:
        """Return a tiny deterministic vector size."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return one valid vector per input passage."""

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Return a valid query vector for interface completeness."""

        return [1.0, 0.0, 0.0]


def sample_chunk() -> dict:
    """Return one valid processed chunk for orchestration tests."""

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
        "chunk_text": "Every person has the right to live with dignity.",
        "token_count": 12,
        "extraction_method": "native",
    }


def test_run_ingestion_wires_components_together() -> None:
    """The entry point should orchestrate setup, loading, and ingestion."""

    embedding_service = FakeEmbeddingService()
    client = Mock()

    with (
        patch(
            "src.indexing.run_ingestion.ensure_collection"
        ) as ensure_collection_mock,
        patch(
            "src.indexing.run_ingestion.load_chunks",
            return_value=[sample_chunk()],
        ) as load_chunks_mock,
        patch(
            "src.indexing.run_ingestion.ingest_chunks",
            return_value=1,
        ) as ingest_chunks_mock,
    ):
        result = run_ingestion(
            embedding_service=embedding_service,
            client=client,
            batch_size=32,
        )

    assert result == 1

    ensure_collection_mock.assert_called_once_with(
        client
    )

    load_chunks_mock.assert_called_once_with()

    ingest_chunks_mock.assert_called_once_with(
        client=client,
        embedding_service=embedding_service,
        chunks=[sample_chunk()],
        batch_size=32,
    )


def test_run_ingestion_rejects_invalid_batch_size() -> None:
    """Invalid batch sizes should fail before any external work begins."""

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero.",
    ):
        run_ingestion(
            embedding_service=FakeEmbeddingService(),
            client=Mock(),
            batch_size=0,
        )


def test_run_ingestion_uses_supplied_dependencies() -> None:
    """Injected dependencies should prevent production constructors from running."""

    embedding_service = FakeEmbeddingService()
    client = Mock()

    with (
        patch(
            "src.indexing.run_ingestion.E5EmbeddingService"
        ) as e5_constructor_mock,
        patch(
            "src.indexing.run_ingestion.QdrantClient"
        ) as qdrant_constructor_mock,
        patch(
            "src.indexing.run_ingestion.ensure_collection"
        ),
        patch(
            "src.indexing.run_ingestion.load_chunks",
            return_value=[sample_chunk()],
        ),
        patch(
            "src.indexing.run_ingestion.ingest_chunks",
            return_value=1,
        ),
    ):
        run_ingestion(
            embedding_service=embedding_service,
            client=client,
        )

    e5_constructor_mock.assert_not_called()
    qdrant_constructor_mock.assert_not_called()