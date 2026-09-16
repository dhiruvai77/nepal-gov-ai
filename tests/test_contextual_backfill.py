"""Tests for resumable contextual-vector Qdrant backfill."""

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.indexing.backfill_contextual_vectors import (
    backfill_contextual_vectors,
)
from src.indexing.qdrant_ingestion import (
    stable_point_id,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Small deterministic provider for migration tests."""

    def __init__(self) -> None:
        """Track passages so tests can verify what was embedded."""

        self.embedded_texts: list[str] = []

    @property
    def dimension(self) -> int:
        """Use a tiny test-only vector dimension."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return one deterministic contextual vector per passage."""

        self.embedded_texts.extend(
            texts
        )

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Implement the interface even though backfill does not use queries."""

        return [0.1, 0.2, 0.3]


def sample_chunk(
    chunk_id: str = "constitution_chunk_00001",
) -> dict:
    """Return one processed chunk compatible with contextual representation."""

    return {
        "chunk_id": chunk_id,
        "document_id": "constitution_nepal_current_en",
        "title": "Constitution of Nepal",
        "organization": "Nepal Law Commission",
        "category": "constitution_law",
        "document_type": "constitution",
        "language": "en",
        "publication_date": "2015",
        "page_start": 17,
        "page_end": 17,
        "source_url": "https://example.gov.np/constitution",
        "section": "Fundamental Rights",
        "subsection": None,
        "article_number": "35",
        "article_title": "Right relating to Health",
        "chunk_text": (
            "Every citizen shall have the right "
            "to free basic health services."
        ),
        "token_count": 14,
        "extraction_method": "native",
    }


def point_record(
    point_id: str,
    *,
    contextual: bool,
) -> SimpleNamespace:
    """Return a lightweight retrieved Qdrant point."""

    vectors = {
        DENSE_VECTOR_NAME: [
            1.0,
            0.0,
            0.0,
        ],
        SPARSE_VECTOR_NAME: {
            "indices": [1, 2],
            "values": [0.5, 0.7],
        },
    }

    if contextual:
        vectors[
            CONTEXTUAL_DENSE_VECTOR_NAME
        ] = [
            0.1,
            0.2,
            0.3,
        ]

    return SimpleNamespace(
        id=point_id,
        vector=vectors,
    )


def test_backfill_updates_only_contextual_vector() -> None:
    """Migration should add contextual dense while preserving baselines."""

    chunk = sample_chunk()

    point_id = stable_point_id(
        chunk["chunk_id"]
    )

    before = point_record(
        point_id,
        contextual=False,
    )

    after = point_record(
        point_id,
        contextual=True,
    )

    client = Mock()

    # First retrieval is the baseline state; second retrieval verifies the
    # point after update_vectors has completed.
    client.retrieve.side_effect = [
        [before],
        [after],
    ]

    service = FakeEmbeddingService()

    stats = backfill_contextual_vectors(
        client=client,
        chunks=[
            chunk,
        ],
        embedding_service=service,
        batch_size=1,
    )

    assert stats.updated == 1
    assert stats.skipped == 0

    client.update_vectors.assert_called_once()

    update_call = (
        client.update_vectors.call_args.kwargs
    )

    assert (
        update_call[
            "collection_name"
        ]
        == COLLECTION_NAME
    )

    assert (
        update_call[
            "wait"
        ]
        is True
    )

    updates = update_call[
        "points"
    ]

    assert len(
        updates
    ) == 1

    # The migration request itself must contain only dense_contextual. Supplying
    # dense or BM25 here would risk rewriting the existing baseline vectors.
    assert set(
        updates[0].vector.keys()
    ) == {
        CONTEXTUAL_DENSE_VECTOR_NAME,
    }

    assert (
        updates[0].vector[
            CONTEXTUAL_DENSE_VECTOR_NAME
        ]
        == [0.1, 0.2, 0.3]
    )

    assert len(
        service.embedded_texts
    ) == 1

    # Confirm the metadata-enriched representation, not raw chunk_text alone,
    # was sent to the embedding provider.
    assert (
        "Document: Constitution of Nepal"
        in service.embedded_texts[0]
    )

    assert (
        "Article number: 35"
        in service.embedded_texts[0]
    )


def test_backfill_skips_existing_contextual_vector() -> None:
    """Already migrated points should be skipped without another API call."""

    chunk = sample_chunk()

    point_id = stable_point_id(
        chunk["chunk_id"]
    )

    client = Mock()

    client.retrieve.return_value = [
        point_record(
            point_id,
            contextual=True,
        )
    ]

    service = FakeEmbeddingService()

    stats = backfill_contextual_vectors(
        client=client,
        chunks=[
            chunk,
        ],
        embedding_service=service,
        batch_size=1,
    )

    assert stats.updated == 0
    assert stats.skipped == 1

    client.update_vectors.assert_not_called()

    assert (
        service.embedded_texts
        == []
    )


def test_backfill_rejects_missing_existing_point() -> None:
    """Backfill must never silently create a point that ingestion did not."""

    chunk = sample_chunk()

    client = Mock()

    client.retrieve.return_value = []

    service = FakeEmbeddingService()

    with pytest.raises(
        RuntimeError,
        match="existing Qdrant points are missing",
    ):
        backfill_contextual_vectors(
            client=client,
            chunks=[
                chunk,
            ],
            embedding_service=service,
            batch_size=1,
        )

    client.update_vectors.assert_not_called()


def test_backfill_rejects_invalid_batch_size() -> None:
    """Backfill requires a positive migration batch size."""

    client = Mock()
    service = FakeEmbeddingService()

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        backfill_contextual_vectors(
            client=client,
            chunks=[
                sample_chunk(),
            ],
            embedding_service=service,
            batch_size=0,
        )

    client.update_vectors.assert_not_called()