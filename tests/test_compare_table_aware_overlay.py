"""Tests for the selective table-aware contextual-vector overlay."""

from __future__ import annotations

import pytest

from src.evaluation.compare_table_aware_overlay import (
    TARGET_QUESTION_IDS,
    RankedPoint,
    aggregate_metrics,
    build_summary,
    build_table_aware_overlay_ranking,
    evaluate_ranking,
    normalized_dot_product,
    primary_rank,
    select_table_like_payloads,
    select_target_records,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)


def make_record(
    *,
    question_id: str,
    query_language: str = "en",
) -> EvaluationRecord:
    """Build one deterministic English-survey evaluation record."""

    return EvaluationRecord(
        question_id=question_id,
        query="Economic Survey question",
        query_language=query_language,
        target_language="en",
        category="finance_economy",
        expected_document_ids=(
            "economic_survey_2023_24_en",
        ),
        primary_relevant_chunk_ids=(
            "primary",
        ),
        relevant_chunk_ids=(
            "primary",
            "supporting",
        ),
        notes="Test.",
    )


def test_select_target_records_preserves_fixed_order() -> None:
    """The experiment must stay tied to the intended five questions."""

    records = [
        make_record(
            question_id=question_id,
            query_language=(
                "ne"
                if question_id
                == "ne_en_005"
                else "en"
            ),
        )
        for question_id in reversed(
            TARGET_QUESTION_IDS
        )
    ]

    selected = (
        select_target_records(
            records
        )
    )

    assert tuple(
        record.question_id
        for record in selected
    ) == (
        TARGET_QUESTION_IDS
    )


def test_select_target_records_rejects_missing_question() -> None:
    """Missing benchmark controls should fail explicitly."""

    records = [
        make_record(
            question_id=question_id
        )
        for question_id in (
            TARGET_QUESTION_IDS[
                :-1
            ]
        )
    ]

    with pytest.raises(
        ValueError,
        match=(
            "missing target questions"
        ),
    ):
        select_target_records(
            records
        )


def test_select_table_like_payloads_filters_document_and_heading() -> None:
    """Only detected table passages from the target survey should be replaced."""

    payloads = {
        "table": {
            "document_id": (
                "economic_survey_2023_24_en"
            ),
            "chunk_text": (
                "Annex 1.1: Annual Growth Rate of GDP"
            ),
        },
        "narrative": {
            "document_id": (
                "economic_survey_2023_24_en"
            ),
            "chunk_text": (
                "Economic growth increased."
            ),
        },
        "other-doc": {
            "document_id": (
                "constitution_nepal_current_en"
            ),
            "chunk_text": (
                "Table 1: Constitutional information"
            ),
        },
    }

    selected = (
        select_table_like_payloads(
            payloads
        )
    )

    assert set(
        selected
    ) == {
        "table",
    }


def test_normalized_dot_product() -> None:
    """Normalized embedding similarity should use the vector dot product."""

    assert (
        normalized_dot_product(
            [
                1.0,
                0.0,
            ],
            [
                0.5,
                0.5,
            ],
        )
        == pytest.approx(
            0.5
        )
    )


def test_normalized_dot_product_rejects_dimension_mismatch() -> None:
    """Replacement vectors must use the same embedding dimension."""

    with pytest.raises(
        ValueError,
        match=(
            "equal dimension"
        ),
    ):
        normalized_dot_product(
            [
                1.0,
            ],
            [
                1.0,
                0.0,
            ],
        )


def test_overlay_replaces_only_table_scores() -> None:
    """Non-table scores stay untouched while replacement vectors may reorder."""

    contextual = [
        RankedPoint(
            point_id="a",
            score=0.90,
        ),
        RankedPoint(
            point_id="table",
            score=0.20,
        ),
        RankedPoint(
            point_id="b",
            score=0.10,
        ),
    ]

    overlay = (
        build_table_aware_overlay_ranking(
            contextual,
            query_vector=[
                1.0,
                0.0,
            ],
            table_vectors_by_id={
                "table": [
                    0.95,
                    0.0,
                ],
            },
        )
    )

    assert [
        item.point_id
        for item in overlay
    ] == [
        "table",
        "a",
        "b",
    ]

    scores = {
        item.point_id: item.score
        for item in overlay
    }

    assert (
        scores[
            "a"
        ]
        == pytest.approx(
            0.90
        )
    )

    assert (
        scores[
            "b"
        ]
        == pytest.approx(
            0.10
        )
    )

    assert (
        scores[
            "table"
        ]
        == pytest.approx(
            0.95
        )
    )


def test_overlay_rejects_unknown_replacement_id() -> None:
    """A replacement vector must correspond to an indexed baseline point."""

    with pytest.raises(
        ValueError,
        match=(
            "outside the contextual ranking"
        ),
    ):
        build_table_aware_overlay_ranking(
            [
                RankedPoint(
                    point_id="a",
                    score=0.5,
                ),
            ],
            query_vector=[
                1.0,
            ],
            table_vectors_by_id={
                "missing": [
                    1.0,
                ],
            },
        )


def test_primary_rank() -> None:
    """Primary rank should use the manually verified primary set."""

    record = (
        make_record(
            question_id="en_en_011"
        )
    )

    ranking = [
        RankedPoint(
            point_id="other",
            score=0.9,
        ),
        RankedPoint(
            point_id="primary",
            score=0.8,
        ),
    ]

    assert (
        primary_rank(
            record,
            ranking,
        )
        == 2
    )


def test_evaluate_ranking_matches_standard_metric_semantics() -> None:
    """Hit, MRR, and recall should match the project retrieval evaluator."""

    record = (
        make_record(
            question_id="en_en_011"
        )
    )

    ranking = [
        RankedPoint(
            point_id="other",
            score=0.9,
        ),
        RankedPoint(
            point_id="primary",
            score=0.8,
        ),
        RankedPoint(
            point_id="supporting",
            score=0.7,
        ),
    ]

    metrics = (
        evaluate_ranking(
            record,
            ranking,
            cutoff=3,
        )
    )

    assert (
        metrics[
            "hit_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "mrr"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "recall"
        ]
        == 1.0
    )


def test_aggregate_metrics() -> None:
    """Aggregate metrics should average query-level values."""

    result = (
        aggregate_metrics(
            [
                {
                    "hit_rate": 1.0,
                    "mrr": 1.0,
                    "recall": 1.0,
                },
                {
                    "hit_rate": 0.0,
                    "mrr": 0.0,
                    "recall": 0.5,
                },
            ]
        )
    )

    assert (
        result[
            "hit_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        result[
            "mrr"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        result[
            "recall"
        ]
        == pytest.approx(
            0.75
        )
    )


def test_build_summary_records_rank_movement() -> None:
    """Summary should preserve baseline and overlay ranks and metrics."""

    records = [
        make_record(
            question_id=question_id,
            query_language=(
                "ne"
                if question_id
                == "ne_en_005"
                else "en"
            ),
        )
        for question_id in (
            TARGET_QUESTION_IDS
        )
    ]

    contextual = {
        record.question_id: [
            RankedPoint(
                point_id="other",
                score=0.9,
            ),
            RankedPoint(
                point_id="primary",
                score=0.8,
            ),
            RankedPoint(
                point_id="supporting",
                score=0.7,
            ),
        ]
        for record in records
    }

    overlay = {
        record.question_id: [
            RankedPoint(
                point_id="primary",
                score=0.95,
            ),
            RankedPoint(
                point_id="other",
                score=0.9,
            ),
            RankedPoint(
                point_id="supporting",
                score=0.7,
            ),
        ]
        for record in records
    }

    summary = (
        build_summary(
            records,
            contextual,
            overlay,
            english_corpus_count=100,
            target_document_count=20,
            table_like_count=5,
            table_hints_by_id={
                "primary": (
                    "Annex 1.1: Test table",
                ),
            },
            cutoffs=(
                1,
                3,
            ),
        )
    )

    assert (
        summary[
            "production_changed"
        ]
        is False
    )

    assert (
        summary[
            "primary_ranks"
        ][
            "en_en_011"
        ][
            "contextual"
        ]
        == 2
    )

    assert (
        summary[
            "primary_ranks"
        ][
            "en_en_011"
        ][
            "table_aware_overlay"
        ]
        == 1
    )

    assert (
        summary[
            "metrics"
        ][
            "1"
        ][
            "contextual"
        ][
            "hit_rate"
        ]
        == 0.0
    )

    assert (
        summary[
            "metrics"
        ][
            "1"
        ][
            "table_aware_overlay"
        ][
            "hit_rate"
        ]
        == 1.0
    )