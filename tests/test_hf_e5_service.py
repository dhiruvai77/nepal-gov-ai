"""Unit tests for the hosted Hugging Face E5 embedding service."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.embeddings.e5_service import (
    EXPECTED_DIMENSION,
    QUERY_INSTRUCTION,
)


def build_vector(
    value: float = 0.1,
) -> list[float]:
    """Create one deterministic vector with the expected E5 dimension."""

    return [
        value
    ] * EXPECTED_DIMENSION


def test_dimension_matches_e5_schema() -> None:
    """Hosted E5 vectors must match the Qdrant dense-vector schema."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token"
    )

    assert (
        service.dimension
        == EXPECTED_DIMENSION
    )


def test_format_query_uses_e5_instruction() -> None:
    """Hosted queries must follow the same E5-instruct format as local E5."""

    formatted = (
        HuggingFaceE5EmbeddingService.format_query(
            "What is the right to education?"
        )
    )

    assert formatted == (
        f"Instruct: {QUERY_INSTRUCTION}\n"
        "Query: What is the right to education?"
    )


def test_format_query_rejects_blank_query() -> None:
    """Blank retrieval queries should fail before any remote request."""

    with pytest.raises(
        ValueError,
        match="Query must contain non-whitespace text",
    ):
        HuggingFaceE5EmbeddingService.format_query(
            "   "
        )


def test_service_requires_token() -> None:
    """Hosted inference should not silently run without authentication."""

    with pytest.raises(
        ValueError,
        match="HF_TOKEN must be set",
    ):
        HuggingFaceE5EmbeddingService(
            token=""
        )


def test_embed_passages_uses_remote_client() -> None:
    """Passage embedding should call feature extraction with normalization."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        batch_size=2,
    )

    mock_client = Mock()

    mock_client.feature_extraction.return_value = [
        build_vector(0.1),
        build_vector(0.2),
    ]

    # Inject the fake client so this unit test never makes a network request.
    service._client = mock_client

    result = service.embed_passages(
        [
            "First passage",
            "Second passage",
        ]
    )

    assert len(
        result
    ) == 2

    assert len(
        result[0]
    ) == EXPECTED_DIMENSION

    mock_client.feature_extraction.assert_called_once_with(
        text=[
            "First passage",
            "Second passage",
        ],
        model=service.model_name,
        normalize=True,
    )


def test_embed_passages_batches_requests() -> None:
    """Long passage lists should be split into bounded hosted requests."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        batch_size=2,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = [
        [
            build_vector(0.1),
            build_vector(0.2),
        ],
        [
            build_vector(0.3),
        ],
    ]

    service._client = mock_client

    result = service.embed_passages(
        [
            "one",
            "two",
            "three",
        ]
    )

    assert len(
        result
    ) == 3

    assert (
        mock_client.feature_extraction.call_count
        == 2
    )


def test_embed_query_returns_single_vector() -> None:
    """Query embedding should return one flat E5 vector."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token"
    )

    mock_client = Mock()

    mock_client.feature_extraction.return_value = [
        build_vector()
    ]

    service._client = mock_client

    vector = service.embed_query(
        "What does the constitution say about education?"
    )

    assert len(
        vector
    ) == EXPECTED_DIMENSION

    call = (
        mock_client.feature_extraction.call_args.kwargs
    )

    assert call[
        "text"
    ][0].startswith(
        "Instruct:"
    )


def test_embed_passages_rejects_blank_passage() -> None:
    """Blank passages should fail locally rather than consuming API calls."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token"
    )

    with pytest.raises(
        ValueError,
        match="Passage at index 1 is empty",
    ):
        service.embed_passages(
            [
                "valid",
                "   ",
            ]
        )


def test_embed_passages_rejects_wrong_dimension() -> None:
    """Hosted responses must match the 1024-dimensional E5 contract."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token"
    )

    mock_client = Mock()

    mock_client.feature_extraction.return_value = [
        [
            0.1,
            0.2,
        ]
    ]

    service._client = mock_client

    with pytest.raises(
        RuntimeError,
        match="Unexpected hosted embedding dimension",
    ):
        service.embed_passages(
            [
                "test passage",
            ]
        )


def test_embed_passages_wraps_provider_failure() -> None:
    """Provider-specific failures should become stable service errors."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token"
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = RuntimeError(
        "provider unavailable"
    )

    service._client = mock_client

    with pytest.raises(
        RuntimeError,
        match="Hosted E5 embedding request failed",
    ):
        service.embed_passages(
            [
                "test passage",
            ]
        )