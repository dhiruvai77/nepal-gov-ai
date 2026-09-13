"""Unit tests for the dense retrieval orchestration layer."""

from collections.abc import Sequence
from unittest.mock import Mock, patch

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.retrieval.dense_retriever import RetrievalResult
from src.retrieval.run_dense_retrieval import (
    print_results,
    run_dense_retrieval,
)


class FakeEmbeddingService(
    EmbeddingService
):
    """Minimal deterministic embedding provider for orchestration tests."""

    @property
    def dimension(self) -> int:
        """Return a small vector size suitable for tests."""

        return 3

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Return valid passage vectors for interface completeness."""

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Return one deterministic query embedding."""

        return [0.8, 0.1, 0.1]


def sample_result() -> RetrievalResult:
    """Return one normalized retrieval result for output tests."""

    return RetrievalResult(
        point_id="930a4fc2-1d87-58c8-b579-89313cf1dc00",
        score=0.91,
        chunk_id="constitution_chunk_00001",
        document_id="constitution_nepal_current_en",
        title="Constitution of Nepal",
        organization="Nepal Law Commission",
        language="en",
        page_start=12,
        page_end=13,
        source_url="https://example.gov.np/constitution",
        chunk_text=(
            "Every person shall have the right to live with dignity."
        ),
        category="constitution_law",
        document_type="constitution",
        article_number="16",
        article_title="Right to live with dignity",
        extraction_method="native",
    )


def test_run_dense_retrieval_wires_dependencies() -> None:
    """The orchestration layer should construct and call DenseRetriever."""

    embedding_service = FakeEmbeddingService()
    client = Mock()
    expected_results = [
        sample_result(),
    ]

    with patch(
        "src.retrieval.run_dense_retrieval.DenseRetriever"
    ) as retriever_class_mock:
        retriever_mock = (
            retriever_class_mock.return_value
        )
        retriever_mock.retrieve.return_value = (
            expected_results
        )

        results = run_dense_retrieval(
            query="What is the right to dignity?",
            top_k=7,
            filters={
                "language": "en",
            },
            embedding_service=embedding_service,
            client=client,
        )

    assert results == expected_results

    retriever_class_mock.assert_called_once_with(
        client=client,
        embedding_service=embedding_service,
    )

    retriever_mock.retrieve.assert_called_once_with(
        query="What is the right to dignity?",
        top_k=7,
        filters={
            "language": "en",
        },
    )


def test_run_dense_retrieval_uses_supplied_dependencies() -> None:
    """Injected dependencies should avoid production constructors."""

    embedding_service = FakeEmbeddingService()
    client = Mock()

    with (
        patch(
            "src.retrieval.run_dense_retrieval.E5EmbeddingService"
        ) as e5_constructor_mock,
        patch(
            "src.retrieval.run_dense_retrieval.QdrantClient"
        ) as qdrant_constructor_mock,
        patch(
            "src.retrieval.run_dense_retrieval.DenseRetriever"
        ) as retriever_class_mock,
    ):
        retriever_class_mock.return_value.retrieve.return_value = []

        run_dense_retrieval(
            query="health policy",
            embedding_service=embedding_service,
            client=client,
        )

    e5_constructor_mock.assert_not_called()
    qdrant_constructor_mock.assert_not_called()


def test_print_results_handles_empty_results(
    capsys,
) -> None:
    """Empty retrieval output should remain explicit to the user."""

    print_results(
        []
    )

    captured = capsys.readouterr()

    assert (
        captured.out.strip()
        == "No retrieval results found."
    )


def test_print_results_includes_citation_information(
    capsys,
) -> None:
    """CLI output should expose score, pages, chunk ID, and source URL."""

    print_results(
        [
            sample_result(),
        ]
    )

    captured = capsys.readouterr()

    assert "score=0.9100" in captured.out
    assert "Constitution of Nepal" in captured.out
    assert "pages=12-13" in captured.out
    assert (
        "constitution_chunk_00001"
        in captured.out
    )
    assert (
        "https://example.gov.np/constitution"
        in captured.out
    )