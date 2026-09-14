"""Unit tests for the hosted Hugging Face E5 embedding service."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from src.embeddings.hf_e5_service import (
    DEFAULT_MAX_ATTEMPTS,
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


def build_http_error(
    status_code: int,
) -> httpx.HTTPStatusError:
    """Create an HTTP error with a real response status for retry tests."""

    request = httpx.Request(
        "POST",
        "https://router.huggingface.co/test",
    )

    response = httpx.Response(
        status_code=status_code,
        request=request,
    )

    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


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


def test_service_rejects_invalid_max_attempts() -> None:
    """Retry configuration must always permit at least one request attempt."""

    with pytest.raises(
        ValueError,
        match="max_attempts must be greater than zero",
    ):
        HuggingFaceE5EmbeddingService(
            token="test-token",
            max_attempts=0,
        )


def test_service_rejects_negative_retry_delay() -> None:
    """Backoff delay cannot be negative."""

    with pytest.raises(
        ValueError,
        match="retry_base_delay_seconds cannot be negative",
    ):
        HuggingFaceE5EmbeddingService(
            token="test-token",
            retry_base_delay_seconds=-1,
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


@patch(
    "src.embeddings.hf_e5_service.time.sleep"
)
def test_transient_transport_failure_is_retried(
    mock_sleep: Mock,
) -> None:
    """A dropped connection should be retried and eventually succeed."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=3,
        retry_base_delay_seconds=1.0,
    )

    mock_client = Mock()

    # This mirrors the class of network failure observed during the full
    # production ingestion: no valid HTTP response was received.
    mock_client.feature_extraction.side_effect = [
        httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        ),
        [
            build_vector()
        ],
    ]

    service._client = mock_client

    result = service.embed_passages(
        [
            "test passage",
        ]
    )

    assert len(
        result
    ) == 1

    assert (
        mock_client.feature_extraction.call_count
        == 2
    )

    mock_sleep.assert_called_once_with(
        1.0
    )


@patch(
    "src.embeddings.hf_e5_service.time.sleep"
)
def test_retry_uses_exponential_backoff(
    mock_sleep: Mock,
) -> None:
    """Successive transient failures should use increasing retry delays."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=3,
        retry_base_delay_seconds=1.0,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = [
        httpx.ConnectError(
            "temporary connection failure"
        ),
        httpx.ReadTimeout(
            "temporary read timeout"
        ),
        [
            build_vector()
        ],
    ]

    service._client = mock_client

    result = service.embed_passages(
        [
            "test passage",
        ]
    )

    assert len(
        result
    ) == 1

    assert (
        mock_client.feature_extraction.call_count
        == 3
    )

    assert mock_sleep.call_args_list == [
        ((1.0,),),
        ((2.0,),),
    ]


@patch(
    "src.embeddings.hf_e5_service.time.sleep"
)
def test_retryable_http_503_is_retried(
    mock_sleep: Mock,
) -> None:
    """Temporary server failures should be retried."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=3,
        retry_base_delay_seconds=0.5,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = [
        build_http_error(
            503
        ),
        [
            build_vector()
        ],
    ]

    service._client = mock_client

    result = service.embed_passages(
        [
            "test passage",
        ]
    )

    assert len(
        result
    ) == 1

    assert (
        mock_client.feature_extraction.call_count
        == 2
    )

    mock_sleep.assert_called_once_with(
        0.5
    )


@patch(
    "src.embeddings.hf_e5_service.time.sleep"
)
def test_rate_limit_429_is_retried(
    mock_sleep: Mock,
) -> None:
    """Provider rate limits should be treated as temporary failures."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=3,
        retry_base_delay_seconds=1.0,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = [
        build_http_error(
            429
        ),
        [
            build_vector()
        ],
    ]

    service._client = mock_client

    result = service.embed_passages(
        [
            "test passage",
        ]
    )

    assert len(
        result
    ) == 1

    assert (
        mock_client.feature_extraction.call_count
        == 2
    )

    mock_sleep.assert_called_once_with(
        1.0
    )


@patch(
    "src.embeddings.hf_e5_service.time.sleep"
)
def test_authentication_failure_is_not_retried(
    mock_sleep: Mock,
) -> None:
    """A 401 response is permanent and should fail immediately."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=3,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = (
        build_http_error(
            401
        )
    )

    service._client = mock_client

    with pytest.raises(
        RuntimeError,
        match=r"failed after 1 attempt\(s\)",
    ):
        service.embed_passages(
            [
                "test passage",
            ]
        )

    assert (
        mock_client.feature_extraction.call_count
        == 1
    )

    mock_sleep.assert_not_called()


@patch(
    "src.embeddings.hf_e5_service.time.sleep"
)
def test_transient_failure_stops_after_max_attempts(
    mock_sleep: Mock,
) -> None:
    """Persistent transport failures must stop after the configured limit."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        retry_base_delay_seconds=1.0,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = (
        httpx.RemoteProtocolError(
            "provider connection dropped"
        )
    )

    service._client = mock_client

    with pytest.raises(
        RuntimeError,
        match=r"failed after 3 attempt\(s\)",
    ):
        service.embed_passages(
            [
                "test passage",
            ]
        )

    assert (
        mock_client.feature_extraction.call_count
        == DEFAULT_MAX_ATTEMPTS
    )

    assert mock_sleep.call_args_list == [
        ((1.0,),),
        ((2.0,),),
    ]


def test_unknown_provider_failure_is_not_retried() -> None:
    """Unknown application errors should fail instead of being repeated."""

    service = HuggingFaceE5EmbeddingService(
        token="test-token",
        max_attempts=3,
    )

    mock_client = Mock()

    mock_client.feature_extraction.side_effect = RuntimeError(
        "unexpected provider failure"
    )

    service._client = mock_client

    with pytest.raises(
        RuntimeError,
        match=r"failed after 1 attempt\(s\)",
    ):
        service.embed_passages(
            [
                "test passage",
            ]
        )

    assert (
        mock_client.feature_extraction.call_count
        == 1
    )