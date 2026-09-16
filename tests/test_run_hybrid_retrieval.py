"""Unit tests for the production hybrid-retrieval entry point."""

from collections.abc import Sequence
from unittest.mock import Mock, patch

import pytest

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.indexing.qdrant_setup import (
    CONTEXTUAL_DENSE_VECTOR_NAME,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)
from src.retrieval.run_hybrid_retrieval import (
    detect_query_language,
    run_hybrid_retrieval,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Small deterministic embedding provider for orchestration tests."""

    @property
    def dimension(self) -> int:
        """Return a test-only embedding dimension."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return deterministic passage embeddings."""

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Return one deterministic query vector."""

        return [
            0.8,
            0.1,
            0.1,
        ]


def sample_result() -> RetrievalResult:
    """Build one normalized result returned by mocked retrieval layers."""

    return RetrievalResult(
        point_id="point-1",
        score=0.031,
        chunk_id="chunk-1",
        document_id="constitution_nepal_current_en",
        title="Constitution of Nepal",
        organization="Nepal Law Commission",
        language="en",
        page_start=16,
        page_end=16,
        source_url="https://example.gov.np/constitution",
        chunk_text=(
            "Every citizen shall have the right "
            "to education."
        ),
        category="constitution_law",
        document_type="constitution",
    )


def test_detect_query_language_recognizes_english_and_nepali() -> None:
    """Routing should distinguish Latin English from Devanagari Nepali."""

    assert (
        detect_query_language(
            "What does the economic survey say?"
        )
        == "en"
    )

    assert (
        detect_query_language(
            "आर्थिक सर्वेक्षणमा शिक्षा"
        )
        == "ne"
    )


def test_detect_query_language_rejects_blank_query() -> None:
    """Language routing should reject empty input early."""

    with pytest.raises(
        ValueError,
        match="query must contain non-whitespace text",
    ):
        detect_query_language(
            "   "
        )


def test_run_hybrid_retrieval_wires_all_retrievers() -> None:
    """Same-language retrieval should connect contextual dense, sparse, and RRF."""

    embedding_service = (
        FakeEmbeddingService()
    )

    client = Mock()

    expected_results = [
        sample_result(),
    ]

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ) as dense_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.SparseRetriever"
        ) as sparse_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        hybrid_class_mock.return_value.retrieve.return_value = (
            expected_results
        )

        results = run_hybrid_retrieval(
            query="education rights",
            top_k=7,
            filters={
                "language": "en",
            },
            embedding_service=embedding_service,
            client=client,
            rrf_k=50,
            candidate_multiplier=4,
        )

    assert (
        results
        == expected_results
    )

    # Production must explicitly use contextual dense retrieval rather than the
    # raw-vector default retained inside DenseRetriever.
    dense_class_mock.assert_called_once_with(
        client=client,
        embedding_service=embedding_service,
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
    )

    sparse_class_mock.assert_called_once_with(
        client=client,
    )

    hybrid_class_mock.assert_called_once_with(
        dense_retriever=(
            dense_class_mock.return_value
        ),
        sparse_retriever=(
            sparse_class_mock.return_value
        ),
        rrf_k=50,
        candidate_multiplier=4,
    )

    hybrid_class_mock.return_value.retrieve.assert_called_once_with(
        query="education rights",
        top_k=7,
        filters={
            "language": "en",
        },
    )


def test_cross_lingual_request_uses_contextual_dense_only() -> None:
    """English-to-Nepali requests should skip BM25 and use contextual dense."""

    embedding_service = (
        FakeEmbeddingService()
    )

    client = Mock()

    expected_results = [
        sample_result(),
    ]

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ) as dense_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.SparseRetriever"
        ) as sparse_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        dense_class_mock.return_value.retrieve.return_value = (
            expected_results
        )

        results = run_hybrid_retrieval(
            query=(
                "What does the survey say "
                "about schools?"
            ),
            top_k=5,
            filters={
                "language": "ne",
            },
            embedding_service=embedding_service,
            client=client,
        )

    assert (
        results
        == expected_results
    )

    dense_class_mock.assert_called_once_with(
        client=client,
        embedding_service=embedding_service,
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
    )

    dense_class_mock.return_value.retrieve.assert_called_once_with(
        query=(
            "What does the survey say "
            "about schools?"
        ),
        top_k=5,
        filters={
            "language": "ne",
        },
    )

    sparse_class_mock.assert_not_called()
    hybrid_class_mock.assert_not_called()


def test_nepali_to_english_request_uses_contextual_dense_only() -> None:
    """Nepali-to-English requests should also skip BM25."""

    embedding_service = (
        FakeEmbeddingService()
    )

    client = Mock()

    expected_results = [
        sample_result(),
    ]

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ) as dense_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.SparseRetriever"
        ) as sparse_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        dense_class_mock.return_value.retrieve.return_value = (
            expected_results
        )

        results = run_hybrid_retrieval(
            query="शिक्षाको अधिकार के हो?",
            top_k=5,
            filters={
                "language": "en",
            },
            embedding_service=embedding_service,
            client=client,
        )

    assert (
        results
        == expected_results
    )

    dense_class_mock.assert_called_once_with(
        client=client,
        embedding_service=embedding_service,
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
    )

    dense_class_mock.return_value.retrieve.assert_called_once_with(
        query="शिक्षाको अधिकार के हो?",
        top_k=5,
        filters={
            "language": "en",
        },
    )

    sparse_class_mock.assert_not_called()
    hybrid_class_mock.assert_not_called()


def test_run_hybrid_retrieval_uses_supplied_dependencies() -> None:
    """Injected dependencies should bypass hosted and Qdrant constructors."""

    embedding_service = (
        FakeEmbeddingService()
    )

    client = Mock()

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval."
            "HuggingFaceE5EmbeddingService"
        ) as embedding_constructor_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.QdrantClient"
        ) as qdrant_constructor_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        hybrid_class_mock.return_value.retrieve.return_value = []

        run_hybrid_retrieval(
            query="health services",
            embedding_service=embedding_service,
            client=client,
        )

    embedding_constructor_mock.assert_not_called()
    qdrant_constructor_mock.assert_not_called()


def test_run_hybrid_retrieval_uses_hosted_e5_by_default() -> None:
    """Production path should use hosted E5 and contextual dense by default."""

    hosted_embedding = Mock()
    client = Mock()

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval."
            "HuggingFaceE5EmbeddingService",
            return_value=hosted_embedding,
        ) as embedding_constructor_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ) as dense_class_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        hybrid_class_mock.return_value.retrieve.return_value = []

        run_hybrid_retrieval(
            query="economic policy",
            client=client,
        )

    embedding_constructor_mock.assert_called_once_with()

    dense_class_mock.assert_called_once_with(
        client=client,
        embedding_service=hosted_embedding,
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
    )


def test_run_hybrid_retrieval_returns_hybrid_results() -> None:
    """Same-language entry point should return the fused ranking unchanged."""

    expected_results = [
        sample_result(),
    ]

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ),
        patch(
            "src.retrieval.run_hybrid_retrieval.SparseRetriever"
        ),
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        hybrid_class_mock.return_value.retrieve.return_value = (
            expected_results
        )

        results = run_hybrid_retrieval(
            query="constitutional rights",
            embedding_service=(
                FakeEmbeddingService()
            ),
            client=Mock(),
        )

    assert (
        results
        == expected_results
    )