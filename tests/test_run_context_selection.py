"""Tests for the production context-selection entry point."""

from unittest.mock import (
    Mock,
    patch,
)

from src.context_selection.base import (
    ContextSelector,
)
from src.context_selection.run_context_selection import (
    DEFAULT_CONTEXT_COUNT,
    run_context_selection,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)
from src.retrieval.run_reranked_retrieval import (
    DEFAULT_RERANK_CANDIDATE_COUNT,
)


def make_reranked_result(
    index: int,
) -> RerankedResult:
    """Create one deterministic reranked candidate."""

    result = RetrievalResult(
        point_id=f"point-{index}",
        score=0.03,
        chunk_id=f"chunk-{index}",
        document_id="constitution_nepal_current_en",
        title="Constitution of Nepal",
        organization="Nepal Law Commission",
        language="en",
        page_start=index,
        page_end=index,
        source_url="https://example.gov.np/constitution",
        chunk_text=f"Evidence passage {index}.",
        chunk_index=index,
        token_count=300,
    )

    return RerankedResult(
        result=result,
        rerank_score=1.0 / index,
        original_rank=index,
    )


class RecordingSelector(
    ContextSelector
):
    """Selector used to verify production orchestration."""

    def __init__(
        self,
    ) -> None:
        self.received: list[
            RerankedResult
        ] = []

    def select(
        self,
        candidates,
    ) -> list[RerankedResult]:
        """Record the candidate pool and return the first two."""

        self.received = list(
            candidates
        )

        return self.received[
            :2
        ]


def test_default_context_count_is_five() -> None:
    """Production should use the evaluated fixed top-5 context size."""

    assert (
        DEFAULT_CONTEXT_COUNT
        == 5
    )


def test_run_context_selection_uses_fixed_top_five_by_default() -> None:
    """Default production selection should preserve the first five ranks."""

    reranked = [
        make_reranked_result(
            index
        )
        for index in range(
            1,
            8,
        )
    ]

    with patch(
        "src.context_selection.run_context_selection."
        "run_reranked_retrieval",
        return_value=reranked,
    ):
        selected = (
            run_context_selection(
                "What rights are guaranteed?"
            )
        )

    assert len(
        selected
    ) == 5

    assert [
        item.result.point_id
        for item in selected
    ] == [
        "point-1",
        "point-2",
        "point-3",
        "point-4",
        "point-5",
    ]


def test_run_context_selection_passes_full_reranked_pool_to_selector() -> None:
    """Context selection must happen only after complete reranking."""

    reranked = [
        make_reranked_result(
            index
        )
        for index in range(
            1,
            7,
        )
    ]

    selector = RecordingSelector()

    with patch(
        "src.context_selection.run_context_selection."
        "run_reranked_retrieval",
        return_value=reranked,
    ):
        selected = (
            run_context_selection(
                "education rights",
                selector=selector,
            )
        )

    assert (
        selector.received
        == reranked
    )

    assert (
        selected
        == reranked[
            :2
        ]
    )


def test_run_context_selection_forwards_pipeline_dependencies() -> None:
    """Production wrapper should forward retrieval and reranker dependencies."""

    embedding_service = Mock()
    client = Mock()
    reranker = Mock(
        spec=Reranker
    )
    selector = RecordingSelector()

    reranked = [
        make_reranked_result(
            1
        ),
    ]

    with patch(
        "src.context_selection.run_context_selection."
        "run_reranked_retrieval",
        return_value=reranked,
    ) as reranked_mock:
        run_context_selection(
            "health rights",
            candidate_count=20,
            filters={
                "language": "en",
            },
            embedding_service=embedding_service,
            client=client,
            reranker=reranker,
            selector=selector,
        )

    reranked_mock.assert_called_once_with(
        query="health rights",
        candidate_count=20,
        top_k=None,
        filters={
            "language": "en",
        },
        embedding_service=embedding_service,
        client=client,
        reranker=reranker,
    )


def test_default_candidate_depth_matches_reranking_pipeline() -> None:
    """Context selection should inherit the validated top-20 retrieval depth."""

    with patch(
        "src.context_selection.run_context_selection."
        "run_reranked_retrieval",
        return_value=[],
    ) as reranked_mock:
        run_context_selection(
            "health rights"
        )

    reranked_mock.assert_called_once_with(
        query="health rights",
        candidate_count=(
            DEFAULT_RERANK_CANDIDATE_COUNT
        ),
        top_k=None,
        filters=None,
        embedding_service=None,
        client=None,
        reranker=None,
    )


def test_run_context_selection_returns_empty_when_reranking_is_empty() -> None:
    """No retrieved evidence should result in no selected context."""

    selector = Mock(
        spec=ContextSelector
    )

    with patch(
        "src.context_selection.run_context_selection."
        "run_reranked_retrieval",
        return_value=[],
    ):
        selected = (
            run_context_selection(
                "unknown question",
                selector=selector,
            )
        )

    assert selected == []

    selector.select.assert_not_called()


def test_default_selector_returns_all_when_fewer_than_five_exist() -> None:
    """Production should not require five candidates when fewer are available."""

    reranked = [
        make_reranked_result(
            1
        ),
        make_reranked_result(
            2
        ),
    ]

    with patch(
        "src.context_selection.run_context_selection."
        "run_reranked_retrieval",
        return_value=reranked,
    ):
        selected = (
            run_context_selection(
                "education rights"
            )
        )

    assert (
        selected
        == reranked
    )