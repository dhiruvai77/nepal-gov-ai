"""Unit tests for the multilingual E5 embedding service.

These tests deliberately avoid loading Sentence Transformers or PyTorch.
They validate the provider-independent behavior that should remain stable
regardless of where the embedding model eventually runs.
"""

import pytest

from src.embeddings.e5_service import (
    E5EmbeddingService,
    EXPECTED_DIMENSION,
    QUERY_INSTRUCTION,
)


def test_dimension_matches_qdrant_schema() -> None:
    """The embedding service must match the dense vector size used by Qdrant."""

    service = E5EmbeddingService()

    assert service.dimension == EXPECTED_DIMENSION
    assert service.dimension == 1024


def test_format_query_adds_e5_instruction() -> None:
    """Queries should follow the instruction/query structure required by E5."""

    query = "What are the fundamental rights in Nepal?"

    formatted = E5EmbeddingService.format_query(query)

    assert formatted == (
        f"Instruct: {QUERY_INSTRUCTION}\n"
        f"Query: {query}"
    )


def test_format_query_preserves_nepali_text() -> None:
    """Nepali queries should pass through unchanged apart from E5 formatting."""

    query = "नेपालको संविधानमा मौलिक हक सम्बन्धी के व्यवस्था छ?"

    formatted = E5EmbeddingService.format_query(query)

    assert query in formatted
    assert formatted.startswith("Instruct:")
    assert "\nQuery: " in formatted


def test_format_query_strips_outer_whitespace() -> None:
    """User input whitespace should not become part of the embedded query."""

    formatted = E5EmbeddingService.format_query(
        "   What is Nepal's fiscal policy?   "
    )

    assert formatted.endswith(
        "Query: What is Nepal's fiscal policy?"
    )


def test_format_query_rejects_empty_query() -> None:
    """Whitespace-only queries should fail before reaching the model runtime."""

    with pytest.raises(
        ValueError,
        match="Query must contain non-whitespace text.",
    ):
        E5EmbeddingService.format_query("   ")


def test_validate_passages_strips_whitespace() -> None:
    """Passage validation should return clean text ready for embedding."""

    cleaned = E5EmbeddingService._validate_passages(
        [
            "  First government passage.  ",
            "Second government passage.",
        ]
    )

    assert cleaned == [
        "First government passage.",
        "Second government passage.",
    ]


def test_validate_passages_rejects_empty_sequence() -> None:
    """Embedding an empty passage collection should fail clearly."""

    with pytest.raises(
        ValueError,
        match="At least one passage is required.",
    ):
        E5EmbeddingService._validate_passages([])


def test_validate_passages_rejects_blank_passage() -> None:
    """Individual blank passages should be rejected with their position."""

    with pytest.raises(
        ValueError,
        match="Passage at index 1 is empty.",
    ):
        E5EmbeddingService._validate_passages(
            [
                "Valid passage.",
                "   ",
            ]
        )


def test_invalid_batch_size_is_rejected() -> None:
    """Batch size must be positive before any model execution begins."""

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero.",
    ):
        E5EmbeddingService(batch_size=0)


def test_service_is_lazy_loaded() -> None:
    """Constructing the service must not initialize the model runtime."""

    service = E5EmbeddingService()

    assert service._model is None