"""Tests for context-selection benchmark orchestration."""

import pytest

from src.context_selection.fixed_top_k import (
    FixedTopKContextSelector,
)
from src.evaluation.compare_context_selection import (
    collect_reranked_candidates,
    count_adjacent_pairs,
    evaluate_selector,
    selected_token_count,
    validate_context_counts,
    validate_token_budgets,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_result(
    point_id: str,
    *,
    token_count: int | None = 100,
) -> RetrievalResult:
    """Create one deterministic retrieval result."""

    return RetrievalResult(
        point_id=point_id,
        score=0.5,
        chunk_id=f"chunk-{point_id}",
        document_id="test-document",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/test",
        chunk_text=f"Evidence {point_id}",
        chunk_index=1,
        token_count=token_count,
    )


def make_record() -> EvaluationRecord:
    """Create one evaluation record with two relevant chunks."""

    return EvaluationRecord(
        question_id="q1",
        query="education rights",
        query_language="en",
        target_language="en",
        category="education",
        expected_document_ids=(
            "test-document",
        ),
        primary_relevant_chunk_ids=(
            "point-b",
        ),
        relevant_chunk_ids=(
            "point-b",
            "point-c",
        ),
        notes="Test record.",
    )


class ReverseReranker(
    Reranker
):
    """Simple deterministic reranker for orchestration tests."""

    def rerank(
        self,
        query,
        candidates,
        *,
        top_k=None,
    ):
        candidate_list = list(
            candidates
        )

        ranked = [
            RerankedResult(
                result=candidate,
                rerank_score=float(
                    index
                ),
                original_rank=index,
            )
            for index, candidate
            in enumerate(
                candidate_list,
                start=1,
            )
        ]

        ranked.reverse()

        if top_k is None:
            return ranked

        return ranked[
            :top_k
        ]


def test_validate_context_counts_accepts_unique_positive_values() -> None:
    """Valid fixed-count experiment values should be preserved."""

    assert (
        validate_context_counts(
            [
                3,
                5,
                8,
            ]
        )
        == (
            3,
            5,
            8,
        )
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0],
        [-1],
    ],
)
def test_validate_context_counts_rejects_invalid_values(
    values,
) -> None:
    """Context counts must contain positive values."""

    with pytest.raises(
        ValueError,
    ):
        validate_context_counts(
            values
        )


def test_validate_context_counts_rejects_duplicates() -> None:
    """Duplicate fixed-count configurations are unnecessary."""

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        validate_context_counts(
            [
                3,
                3,
            ]
        )


def test_validate_token_budgets_accepts_unique_positive_values() -> None:
    """Valid token-budget experiments should be preserved."""

    assert (
        validate_token_budgets(
            [
                1400,
                1800,
                2200,
            ]
        )
        == (
            1400,
            1800,
            2200,
        )
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0],
        [-1],
    ],
)
def test_validate_token_budgets_rejects_invalid_values(
    values,
) -> None:
    """Token-budget experiments require positive budgets."""

    with pytest.raises(
        ValueError,
    ):
        validate_token_budgets(
            values
        )


def test_validate_token_budgets_rejects_duplicates() -> None:
    """Duplicate budgets should not produce duplicate benchmark runs."""

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        validate_token_budgets(
            [
                1800,
                1800,
            ]
        )


def test_collect_reranked_candidates_retrieves_once() -> None:
    """One query should cause one retrieval and one reranking request."""

    record = make_record()

    candidates = [
        make_result(
            "point-a"
        ),
        make_result(
            "point-b"
        ),
        make_result(
            "point-c"
        ),
    ]

    calls = []

    def retrieve(
        received_record,
        depth,
    ):
        calls.append(
            (
                received_record.question_id,
                depth,
            )
        )

        return candidates

    collected = collect_reranked_candidates(
        [
            record,
        ],
        retrieval_depth=20,
        reranker=ReverseReranker(),
        retrieve=retrieve,
    )

    assert calls == [
        (
            "q1",
            20,
        )
    ]

    assert [
        item.result.point_id
        for item in collected[
            "q1"
        ]
    ] == [
        "point-c",
        "point-b",
        "point-a",
    ]


def test_collect_reranked_candidates_rejects_candidate_count_change() -> None:
    """Reranking must preserve the first-stage candidate count."""

    class DroppingReranker(
        Reranker
    ):
        def rerank(
            self,
            query,
            candidates,
            *,
            top_k=None,
        ):
            candidate_list = list(
                candidates
            )

            return [
                RerankedResult(
                    result=candidate_list[
                        0
                    ],
                    rerank_score=1.0,
                    original_rank=1,
                )
            ]

    candidates = [
        make_result(
            "point-a"
        ),
        make_result(
            "point-b"
        ),
    ]

    def retrieve(
        record,
        depth,
    ):
        return candidates

    with pytest.raises(
        RuntimeError,
        match="candidate count",
    ):
        collect_reranked_candidates(
            [
                make_record(),
            ],
            retrieval_depth=20,
            reranker=DroppingReranker(),
            retrieve=retrieve,
        )


def test_collect_reranked_candidates_rejects_identity_change() -> None:
    """Reranking must not substitute different retrieval candidates."""

    class ReplacingReranker(
        Reranker
    ):
        def rerank(
            self,
            query,
            candidates,
            *,
            top_k=None,
        ):
            candidate_list = list(
                candidates
            )

            replacement = make_result(
                "point-x"
            )

            return [
                RerankedResult(
                    result=candidate_list[
                        0
                    ],
                    rerank_score=1.0,
                    original_rank=1,
                ),
                RerankedResult(
                    result=replacement,
                    rerank_score=0.9,
                    original_rank=2,
                ),
            ]

    candidates = [
        make_result(
            "point-a"
        ),
        make_result(
            "point-b"
        ),
    ]

    def retrieve(
        record,
        depth,
    ):
        return candidates

    with pytest.raises(
        RuntimeError,
        match="candidate identity",
    ):
        collect_reranked_candidates(
            [
                make_record(),
            ],
            retrieval_depth=20,
            reranker=ReplacingReranker(),
            retrieve=retrieve,
        )


def test_selected_token_count_sums_stored_chunk_tokens() -> None:
    """Context size should use indexed tokenizer-derived counts."""

    selected = [
        RerankedResult(
            result=make_result(
                "point-a",
                token_count=120,
            ),
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=make_result(
                "point-b",
                token_count=80,
            ),
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    assert (
        selected_token_count(
            selected
        )
        == 200
    )


def test_selected_token_count_handles_empty_selection() -> None:
    """Empty context should have zero passage tokens."""

    assert (
        selected_token_count([])
        == 0
    )


def test_selected_token_count_rejects_missing_token_metadata() -> None:
    """Evaluation should expose missing token metadata rather than guess."""

    selected = [
        RerankedResult(
            result=make_result(
                "point-a",
                token_count=None,
            ),
            rerank_score=0.9,
            original_rank=1,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="token_count",
    ):
        selected_token_count(
            selected
        )


def test_selected_token_count_rejects_negative_tokens() -> None:
    """Malformed negative token counts should fail explicitly."""

    selected = [
        RerankedResult(
            result=make_result(
                "point-a",
                token_count=-1,
            ),
            rerank_score=0.9,
            original_rank=1,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="negative token_count",
    ):
        selected_token_count(
            selected
        )


def test_count_adjacent_pairs_detects_same_document_neighbors() -> None:
    """Adjacent chunks from the same document should count as overlap."""

    first = make_result(
        "point-a",
        token_count=100,
    )

    second = RetrievalResult(
        point_id="point-b",
        score=0.5,
        chunk_id="chunk-point-b",
        document_id="test-document",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/test",
        chunk_text="Evidence point-b",
        chunk_index=2,
        token_count=100,
    )

    selected = [
        RerankedResult(
            result=first,
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=second,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    assert (
        count_adjacent_pairs(
            selected
        )
        == 1
    )


def test_count_adjacent_pairs_ignores_nonadjacent_chunks() -> None:
    """Separated chunks should not be counted as overlap."""

    first = make_result(
        "point-a",
        token_count=100,
    )

    second = RetrievalResult(
        point_id="point-b",
        score=0.5,
        chunk_id="chunk-point-b",
        document_id="test-document",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/test",
        chunk_text="Evidence point-b",
        chunk_index=5,
        token_count=100,
    )

    selected = [
        RerankedResult(
            result=first,
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=second,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    assert (
        count_adjacent_pairs(
            selected
        )
        == 0
    )


def test_count_adjacent_pairs_ignores_different_documents() -> None:
    """Equal or neighboring indexes across documents are unrelated."""

    first = make_result(
        "point-a",
        token_count=100,
    )

    second = RetrievalResult(
        point_id="point-b",
        score=0.5,
        chunk_id="chunk-point-b",
        document_id="different-document",
        title="Different Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/other",
        chunk_text="Evidence point-b",
        chunk_index=2,
        token_count=100,
    )

    selected = [
        RerankedResult(
            result=first,
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=second,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    assert (
        count_adjacent_pairs(
            selected
        )
        == 0
    )


def test_count_adjacent_pairs_ignores_missing_chunk_index() -> None:
    """Missing chunk position should not be treated as known adjacency."""

    first = make_result(
        "point-a",
        token_count=100,
    )

    second = RetrievalResult(
        point_id="point-b",
        score=0.5,
        chunk_id="chunk-point-b",
        document_id="test-document",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/test",
        chunk_text="Evidence point-b",
        chunk_index=None,
        token_count=100,
    )

    selected = [
        RerankedResult(
            result=first,
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=second,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    assert (
        count_adjacent_pairs(
            selected
        )
        == 0
    )


def test_count_adjacent_pairs_handles_empty_selection() -> None:
    """Empty context should contain no adjacent chunk pairs."""

    assert (
        count_adjacent_pairs([])
        == 0
    )


def test_evaluate_selector_measures_coverage_and_context_size() -> None:
    """Selector evaluation should report quality and context cost together."""

    record = make_record()

    reranked = {
        "q1": [
            RerankedResult(
                result=make_result(
                    "point-a",
                    token_count=100,
                ),
                rerank_score=0.9,
                original_rank=1,
            ),
            RerankedResult(
                result=make_result(
                    "point-b",
                    token_count=150,
                ),
                rerank_score=0.8,
                original_rank=2,
            ),
            RerankedResult(
                result=make_result(
                    "point-c",
                    token_count=200,
                ),
                rerank_score=0.7,
                original_rank=3,
            ),
        ]
    }

    metrics = evaluate_selector(
        [
            record,
        ],
        reranked,
        selector=(
            FixedTopKContextSelector(
                top_k=2
            )
        ),
    )

    assert (
        metrics.hit_rate
        == 1.0
    )

    assert (
        metrics.mrr
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics.recall
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics.average_passages
        == 2.0
    )

    assert (
        metrics.average_tokens
        == 250.0
    )

    # Both helper results use chunk_index=1 from make_result(), so they form
    # one same-document duplicate/adjacent pair under the overlap metric.
    assert (
        metrics.average_adjacent_pairs
        == 1.0
    )


def test_evaluate_selector_rejects_empty_records() -> None:
    """A benchmark without evaluation records is invalid."""

    with pytest.raises(
        ValueError,
        match="At least one evaluation record",
    ):
        evaluate_selector(
            [],
            {},
            selector=(
                FixedTopKContextSelector(
                    top_k=5
                )
            ),
        )