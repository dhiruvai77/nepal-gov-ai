"""Tests for the reranker benchmark orchestration."""

import pytest

from src.evaluation.compare_reranked_retrieval import (
    collect_rankings,
    validate_reranked_candidate_set,
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
) -> RetrievalResult:
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
        source_url="https://example.gov.np/test.pdf",
        chunk_text=f"Passage {point_id}",
        category="test",
        document_type="report",
    )


def make_record() -> EvaluationRecord:
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
        ),
        notes="Test record.",
    )


class ReverseReranker(
    Reranker
):
    """Deterministic test reranker that reverses candidate order."""

    def rerank(
        self,
        query,
        candidates,
        *,
        top_k=None,
    ):
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
                candidates,
                start=1,
            )
        ]

        ranked.reverse()

        if top_k is None:
            return ranked

        return ranked[
            :top_k
        ]


def test_validate_reranked_candidate_set_accepts_reordering() -> None:
    first_stage = [
        make_result(
            "point-a"
        ),
        make_result(
            "point-b"
        ),
    ]

    validate_reranked_candidate_set(
        first_stage,
        [
            first_stage[1],
            first_stage[0],
        ],
    )


def test_validate_reranked_candidate_set_rejects_identity_change() -> None:
    with pytest.raises(
        RuntimeError,
        match="identity",
    ):
        validate_reranked_candidate_set(
            [
                make_result(
                    "point-a"
                ),
                make_result(
                    "point-b"
                ),
            ],
            [
                make_result(
                    "point-a"
                ),
                make_result(
                    "point-c"
                ),
            ],
        )


def test_collect_rankings_retrieves_once_at_fixed_depth() -> None:
    calls: list[
        tuple[str, int]
    ] = []

    first_a = make_result(
        "point-a"
    )

    first_b = make_result(
        "point-b"
    )

    def retrieve(
        record,
        depth,
    ):
        calls.append(
            (
                record.question_id,
                depth,
            )
        )

        return [
            first_a,
            first_b,
        ]

    baseline, reranked = collect_rankings(
        [
            make_record()
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
        item.point_id
        for item in baseline[
            "q1"
        ]
    ] == [
        "point-a",
        "point-b",
    ]

    assert [
        item.point_id
        for item in reranked[
            "q1"
        ]
    ] == [
        "point-b",
        "point-a",
    ]