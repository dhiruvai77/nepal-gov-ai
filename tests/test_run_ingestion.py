"""Tests for the NepalGov AI production ingestion entry point."""

from unittest.mock import Mock, patch

import pytest

from src.embeddings.base import EmbeddingService
from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.run_ingestion import (
    DEFAULT_BATCH_SIZE,
    run_ingestion,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Minimal embedding provider used to avoid external inference in tests."""

    @property
    def dimension(self) -> int:
        """Match the dense-vector dimension used by the production schema."""

        return 1024

    def embed_passages(
        self,
        texts,
    ) -> list[list[float]]:
        """Return deterministic vectors without loading a real model."""

        return [
            [0.1] * self.dimension
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """Return one deterministic query vector for interface completeness."""

        return [0.1] * self.dimension


def test_default_batch_size_is_positive() -> None:
    """Production ingestion should have a valid default batch size."""

    assert DEFAULT_BATCH_SIZE > 0


def test_run_ingestion_rejects_invalid_batch_size() -> None:
    """Invalid batching should fail before constructing production services."""

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        run_ingestion(
            batch_size=0
        )


@patch(
    "src.indexing.run_ingestion.ingest_chunks"
)
@patch(
    "src.indexing.run_ingestion.load_chunks"
)
@patch(
    "src.indexing.run_ingestion.ensure_collection"
)
def test_run_ingestion_uses_injected_dependencies(
    mock_ensure_collection: Mock,
    mock_load_chunks: Mock,
    mock_ingest_chunks: Mock,
) -> None:
    """Injected dependencies should be passed through to the ingestion layer."""

    fake_client = Mock()
    fake_embedding_service = FakeEmbeddingService()

    chunks = [
        {
            "chunk_id": "chunk-1",
            "chunk_text": "Example government passage.",
        }
    ]

    mock_load_chunks.return_value = chunks
    mock_ingest_chunks.return_value = 1

    result = run_ingestion(
        embedding_service=fake_embedding_service,
        client=fake_client,
        batch_size=8,
    )

    assert result == 1

    mock_ensure_collection.assert_called_once_with(
        fake_client
    )

    mock_load_chunks.assert_called_once_with()

    mock_ingest_chunks.assert_called_once_with(
        client=fake_client,
        embedding_service=fake_embedding_service,
        chunks=chunks,
        batch_size=8,
    )


@patch(
    "src.indexing.run_ingestion.ingest_chunks"
)
@patch(
    "src.indexing.run_ingestion.load_chunks"
)
@patch(
    "src.indexing.run_ingestion.ensure_collection"
)
@patch(
    "src.indexing.run_ingestion.QdrantClient"
)
@patch(
    "src.indexing.run_ingestion.HuggingFaceE5EmbeddingService"
)
def test_run_ingestion_uses_hosted_e5_by_default(
    mock_embedding_service_class: Mock,
    mock_qdrant_client_class: Mock,
    mock_ensure_collection: Mock,
    mock_load_chunks: Mock,
    mock_ingest_chunks: Mock,
) -> None:
    """Default production wiring should use hosted E5 and the Qdrant client."""

    fake_embedding_service = Mock(
        spec=HuggingFaceE5EmbeddingService
    )
    fake_client = Mock()

    mock_embedding_service_class.return_value = (
        fake_embedding_service
    )
    mock_qdrant_client_class.return_value = (
        fake_client
    )

    chunks = [
        {
            "chunk_id": "chunk-1",
            "chunk_text": "Example passage.",
        }
    ]

    mock_load_chunks.return_value = chunks
    mock_ingest_chunks.return_value = 1

    result = run_ingestion()

    assert result == 1

    # The hosted embedding provider should be created only when no custom
    # embedding service is injected.
    mock_embedding_service_class.assert_called_once_with()

    mock_qdrant_client_class.assert_called_once()

    mock_ensure_collection.assert_called_once_with(
        fake_client
    )

    mock_ingest_chunks.assert_called_once_with(
        client=fake_client,
        embedding_service=fake_embedding_service,
        chunks=chunks,
        batch_size=DEFAULT_BATCH_SIZE,
    )


@patch(
    "src.indexing.run_ingestion.ingest_chunks"
)
@patch(
    "src.indexing.run_ingestion.load_chunks"
)
@patch(
    "src.indexing.run_ingestion.ensure_collection"
)
@patch(
    "src.indexing.run_ingestion.HuggingFaceE5EmbeddingService"
)
def test_injected_embedding_service_prevents_default_provider_creation(
    mock_embedding_service_class: Mock,
    mock_ensure_collection: Mock,
    mock_load_chunks: Mock,
    mock_ingest_chunks: Mock,
) -> None:
    """Dependency injection must not unnecessarily create the hosted provider."""

    fake_embedding_service = FakeEmbeddingService()
    fake_client = Mock()

    mock_load_chunks.return_value = []
    mock_ingest_chunks.return_value = 0

    result = run_ingestion(
        embedding_service=fake_embedding_service,
        client=fake_client,
    )

    assert result == 0

    # This matters because constructing the hosted provider requires HF_TOKEN.
    # Tests using injected services should remain fully offline.
    mock_embedding_service_class.assert_not_called()

    mock_ensure_collection.assert_called_once_with(
        fake_client
    )


@patch(
    "src.indexing.run_ingestion.ingest_chunks"
)
@patch(
    "src.indexing.run_ingestion.load_chunks"
)
@patch(
    "src.indexing.run_ingestion.ensure_collection"
)
def test_run_ingestion_returns_ingested_count(
    mock_ensure_collection: Mock,
    mock_load_chunks: Mock,
    mock_ingest_chunks: Mock,
) -> None:
    """The entry point should return the number reported by ingestion."""

    fake_embedding_service = FakeEmbeddingService()
    fake_client = Mock()

    mock_load_chunks.return_value = [
        {
            "chunk_id": "chunk-a",
            "chunk_text": "First passage.",
        },
        {
            "chunk_id": "chunk-b",
            "chunk_text": "Second passage.",
        },
    ]

    mock_ingest_chunks.return_value = 2

    result = run_ingestion(
        embedding_service=fake_embedding_service,
        client=fake_client,
        batch_size=4,
    )

    assert result == 2