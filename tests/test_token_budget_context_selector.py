"""Tests for token-budget context selection."""

import pytest

from src.context_selection.token_budget import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    TokenBudgetContextSelector,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_result(
    point_id: str,
    *,
    token_count: int | None,
    rerank_score: float,
    original_rank: int,
) -> RerankedResult:
    """Create one deterministic reranked result."""

    retrieval_result = RetrievalResult(
        point_id=point_id,
        score=0.5,
        chunk_id=f"chunk-{point_id}",
        document_id="test-document",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/document",
        chunk_text=f"Evidence passage {point_id}.",
        chunk_index=original_rank,
        token_count=token_count,
    )

    return RerankedResult(
        result=retrieval_result,
        rerank_score=rerank_score,
        original_rank=original_rank,
    )


def test_default_token_budget_is_1800() -> None:
    """Initial token budget should approximate the measured top-5 baseline."""

    assert (
        DEFAULT_CONTEXT_TOKEN_BUDGET
        == 1800
    )


def test_token_budget_selects_ranked_prefix_that_fits() -> None:
    """Candidates should be selected in reranker order within the budget."""

    candidates = [
        make_result(
            "a",
            token_count=400,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            token_count=350,
            rerank_score=0.8,
            original_rank=2,
        ),
        make_result(
            "c",
            token_count=300,
            rerank_score=0.7,
            original_rank=3,
        ),
    ]

    selector = TokenBudgetContextSelector(
        max_tokens=800
    )

    selected = selector.select(
        candidates
    )

    assert [
        item.result.point_id
        for item in selected
    ] == [
        "a",
        "b",
    ]


def test_token_budget_does_not_skip_ranked_candidate_to_fit_lower_one() -> None:
    """Lower-ranked short chunks must not leapfrog a higher-ranked passage."""

    candidates = [
        make_result(
            "a",
            token_count=500,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            token_count=400,
            rerank_score=0.8,
            original_rank=2,
        ),
        make_result(
            "c",
            token_count=50,
            rerank_score=0.7,
            original_rank=3,
        ),
    ]

    selector = TokenBudgetContextSelector(
        max_tokens=700
    )

    selected = selector.select(
        candidates
    )

    assert [
        item.result.point_id
        for item in selected
    ] == [
        "a",
    ]


def test_token_budget_can_select_entire_pool() -> None:
    """All candidates should survive when the complete pool fits."""

    candidates = [
        make_result(
            "a",
            token_count=100,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            token_count=100,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    selector = TokenBudgetContextSelector(
        max_tokens=500
    )

    assert (
        selector.select(
            candidates
        )
        == candidates
    )


def test_token_budget_handles_empty_candidates() -> None:
    """Empty reranker output should produce empty context."""

    selector = (
        TokenBudgetContextSelector()
    )

    assert (
        selector.select([])
        == []
    )


def test_token_budget_returns_empty_when_first_chunk_exceeds_budget() -> None:
    """Strict budget enforcement must never silently exceed the limit."""

    candidate = make_result(
        "a",
        token_count=400,
        rerank_score=0.9,
        original_rank=1,
    )

    selector = TokenBudgetContextSelector(
        max_tokens=300
    )

    assert (
        selector.select(
            [
                candidate,
            ]
        )
        == []
    )


def test_token_budget_rejects_missing_token_count() -> None:
    """Production token selection requires indexed token metadata."""

    candidate = make_result(
        "a",
        token_count=None,
        rerank_score=0.9,
        original_rank=1,
    )

    selector = (
        TokenBudgetContextSelector()
    )

    with pytest.raises(
        ValueError,
        match="token_count is required",
    ):
        selector.select(
            [
                candidate,
            ]
        )


def test_token_budget_rejects_negative_token_count() -> None:
    """Malformed negative token counts should fail explicitly."""

    candidate = make_result(
        "a",
        token_count=-1,
        rerank_score=0.9,
        original_rank=1,
    )

    selector = (
        TokenBudgetContextSelector()
    )

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        selector.select(
            [
                candidate,
            ]
        )


@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
    ],
)
def test_token_budget_rejects_invalid_budget(
    max_tokens: int,
) -> None:
    """A usable context budget must be positive."""

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        TokenBudgetContextSelector(
            max_tokens=max_tokens
        )