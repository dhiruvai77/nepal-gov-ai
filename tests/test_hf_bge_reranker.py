"""Unit tests for the hosted TEI-backed BGE reranker."""

from unittest.mock import Mock, patch

import httpx
import pytest

from src.reranking.hf_bge_reranker import (
    HuggingFaceBGEReranker,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_result(
    point_id: str,
    text: str,
) -> RetrievalResult:
    """Create one deterministic retrieval candidate for reranker tests."""

    return RetrievalResult(
        point_id=point_id,
        score=0.5,
        chunk_id=f"chunk-{point_id}",
        document_id="test-document",
        title="Test Government Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/document.pdf",
        chunk_text=text,
        category="test",
        document_type="report",
    )


def make_response(
    data,
) -> Mock:
    """Create a successful mocked HTTP response."""

    response = Mock()

    response.raise_for_status.return_value = None

    response.json.return_value = data

    return response


def build_http_error(
    status_code: int,
) -> httpx.HTTPStatusError:
    """Create an HTTP status error containing a real response code."""

    request = httpx.Request(
        "POST",
        "https://reranker.example.com/rerank",
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


def test_service_requires_endpoint_url() -> None:
    """Hosted reranking must know which TEI endpoint to call."""

    with patch.dict(
        "os.environ",
        {},
        clear=True,
    ):
        with pytest.raises(
            ValueError,
            match="HF_RERANKER_ENDPOINT_URL must be set",
        ):
            HuggingFaceBGEReranker()


def test_service_builds_rerank_url() -> None:
    """A TEI base URL should automatically resolve to its rerank route."""

    service = HuggingFaceBGEReranker(
        endpoint_url=(
            "https://reranker.example.com/"
        ),
        token="test-token",
    )

    assert (
        service.rerank_url
        == "https://reranker.example.com/rerank"
    )


def test_format_candidate_text_adds_document_title() -> None:
    """Production reranking should use the evaluated title-aware input."""

    candidate = make_result(
        "point-1",
        "Every citizen has the right to education.",
    )

    assert (
        HuggingFaceBGEReranker._format_candidate_text(
            candidate
        )
        == (
            "Document: Test Government Document\n\n"
            "Every citizen has the right to education."
        )
    )

    # Formatting for reranking must not mutate the original evidence passage.
    assert (
        candidate.chunk_text
        == "Every citizen has the right to education."
    )


def test_rerank_sends_tei_payload_and_orders_results() -> None:
    """The provider should use TEI's query-plus-texts reranking contract."""

    service = HuggingFaceBGEReranker(
        endpoint_url="https://reranker.example.com",
        token="test-token",
    )

    mock_client = Mock()

    mock_client.post.return_value = (
        make_response(
            [
                {
                    "index": 1,
                    "score": 0.95,
                },
                {
                    "index": 2,
                    "score": 0.60,
                },
                {
                    "index": 0,
                    "score": 0.20,
                },
            ]
        )
    )

    # Inject the client so the unit test never performs external inference.
    service._client = mock_client

    candidates = [
        make_result(
            "point-1",
            "Unrelated passage.",
        ),
        make_result(
            "point-2",
            "The Constitution guarantees education rights.",
        ),
        make_result(
            "point-3",
            "Another partly relevant passage.",
        ),
    ]

    results = service.rerank(
        "What does the Constitution say about education?",
        candidates,
        top_k=2,
    )

    assert [
        result.result.point_id
        for result in results
    ] == [
        "point-2",
        "point-3",
    ]

    assert [
        result.original_rank
        for result in results
    ] == [
        2,
        3,
    ]

    call = (
        mock_client.post.call_args
    )

    assert (
        call.args[0]
        == "https://reranker.example.com/rerank"
    )

    assert call.kwargs[
        "json"
    ] == {
        "query": (
            "What does the Constitution "
            "say about education?"
        ),
        "texts": [
            (
                "Document: Test Government Document\n\n"
                "Unrelated passage."
            ),
            (
                "Document: Test Government Document\n\n"
                "The Constitution guarantees "
                "education rights."
            ),
            (
                "Document: Test Government Document\n\n"
                "Another partly relevant passage."
            ),
        ],
        "truncate": True,
        "raw_scores": False,
        "return_text": False,
    }


def test_rerank_preserves_original_candidate_text() -> None:
    """Title-aware inference must not alter returned retrieval evidence."""

    service = HuggingFaceBGEReranker(
        endpoint_url="https://reranker.example.com",
        token="test-token",
    )

    mock_client = Mock()

    mock_client.post.return_value = (
        make_response(
            [
                {
                    "index": 0,
                    "score": 0.9,
                },
            ]
        )
    )

    service._client = mock_client

    candidate = make_result(
        "point-1",
        "Original evidence passage.",
    )

    results = service.rerank(
        "test query",
        [
            candidate,
        ],
    )

    assert len(
        results
    ) == 1

    assert (
        results[0].result
        is candidate
    )

    assert (
        results[0].result.chunk_text
        == "Original evidence passage."
    )


def test_rerank_rejects_mismatched_response_count() -> None:
    """Every submitted candidate must receive one returned rank."""

    service = HuggingFaceBGEReranker(
        endpoint_url="https://reranker.example.com",
        token="test-token",
    )

    mock_client = Mock()

    mock_client.post.return_value = (
        make_response(
            [
                {
                    "index": 0,
                    "score": 0.8,
                },
            ]
        )
    )

    service._client = mock_client

    candidates = [
        make_result(
            "point-1",
            "First passage.",
        ),
        make_result(
            "point-2",
            "Second passage.",
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match=(
            "different number of ranks "
            "than input candidates"
        ),
    ):
        service.rerank(
            "test query",
            candidates,
        )


def test_rerank_rejects_duplicate_indices() -> None:
    """Malformed provider output must not map two scores to one candidate."""

    service = HuggingFaceBGEReranker(
        endpoint_url="https://reranker.example.com",
        token="test-token",
    )

    mock_client = Mock()

    mock_client.post.return_value = (
        make_response(
            [
                {
                    "index": 0,
                    "score": 0.9,
                },
                {
                    "index": 0,
                    "score": 0.7,
                },
            ]
        )
    )

    service._client = mock_client

    candidates = [
        make_result(
            "point-1",
            "First passage.",
        ),
        make_result(
            "point-2",
            "Second passage.",
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="duplicate candidate index",
    ):
        service.rerank(
            "test query",
            candidates,
        )


@patch(
    "src.reranking.hf_bge_reranker.time.sleep"
)
def test_transient_failure_is_retried(
    mock_sleep: Mock,
) -> None:
    """Temporary endpoint failures should retry with bounded backoff."""

    service = HuggingFaceBGEReranker(
        endpoint_url="https://reranker.example.com",
        token="test-token",
        max_attempts=3,
        retry_base_delay_seconds=1.0,
    )

    mock_client = Mock()

    mock_client.post.side_effect = [
        build_http_error(
            503
        ),
        make_response(
            [
                {
                    "index": 0,
                    "score": 0.9,
                },
            ]
        ),
    ]

    service._client = mock_client

    result = service.rerank(
        "education rights",
        [
            make_result(
                "point-1",
                "Relevant passage.",
            ),
        ],
    )

    assert len(
        result
    ) == 1

    assert (
        mock_client.post.call_count
        == 2
    )

    mock_sleep.assert_called_once_with(
        1.0
    )


@patch(
    "src.reranking.hf_bge_reranker.time.sleep"
)
def test_authentication_failure_is_not_retried(
    mock_sleep: Mock,
) -> None:
    """Permanent authentication failures should stop immediately."""

    service = HuggingFaceBGEReranker(
        endpoint_url="https://reranker.example.com",
        token="test-token",
        max_attempts=3,
    )

    mock_client = Mock()

    mock_client.post.side_effect = (
        build_http_error(
            401
        )
    )

    service._client = mock_client

    with pytest.raises(
        RuntimeError,
        match=r"failed after 1 attempt\(s\)",
    ):
        service.rerank(
            "education rights",
            [
                make_result(
                    "point-1",
                    "Relevant passage.",
                ),
            ],
        )

    assert (
        mock_client.post.call_count
        == 1
    )

    mock_sleep.assert_not_called()