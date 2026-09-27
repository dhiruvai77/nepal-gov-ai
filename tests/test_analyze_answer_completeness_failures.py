"""Tests for deterministic answer-completeness failure analysis."""

from __future__ import annotations

import pytest

from src.evaluation.analyze_answer_completeness_failures import (
    CAUSE_GENERATION,
    CAUSE_MIXED,
    CAUSE_PRIMARY_NOT_SELECTED,
    CAUSE_SUPPORTING_NOT_SELECTED,
    build_analysis,
    build_failure_record,
    calculate_selected_gold_coverage,
    classify_failure_cause,
)


def make_row(
    *,
    question_id: str = "en_en_001",
    completeness_label: str = "incomplete",
    primary_ids: list[str] | None = None,
    relevant_ids: list[str] | None = None,
    selected_ids: list[str] | None = None,
) -> dict:
    """Build one deterministic completed answer-quality row."""

    primary = (
        primary_ids
        if primary_ids is not None
        else [
            "primary",
        ]
    )

    relevant = (
        relevant_ids
        if relevant_ids is not None
        else [
            "primary",
            "supporting",
        ]
    )

    selected = (
        selected_ids
        if selected_ids is not None
        else [
            "primary",
            "supporting",
        ]
    )

    selected_relevant = (
        len(
            set(
                selected
            )
            & set(
                relevant
            )
        )
    )

    selected_primary_hit = int(
        bool(
            set(
                selected
            )
            & set(
                primary
            )
        )
    )

    return {
        "review_status": (
            "completed"
        ),
        "question_id": (
            question_id
        ),
        "query": (
            "Test question?"
        ),
        "query_language": (
            "en"
        ),
        "target_language": (
            "en"
        ),
        "category": (
            "test"
        ),
        "completeness_label": (
            completeness_label
        ),
        "factual_fidelity_label": (
            "fully_faithful"
        ),
        "primary_relevant_chunk_ids": (
            primary
        ),
        "relevant_chunk_ids": (
            relevant
        ),
        "selected_primary_hit": (
            selected_primary_hit
        ),
        "selected_relevant_recall": (
            selected_relevant
            / len(
                relevant
            )
        ),
        "selected_evidence": [
            {
                "point_id": (
                    point_id
                ),
            }
            for point_id in (
                selected
            )
        ],
        "review_notes": (
            "Test note."
        ),
    }


def test_calculate_selected_gold_coverage_full() -> None:
    """Full selected gold context should report perfect coverage."""

    coverage = (
        calculate_selected_gold_coverage(
            make_row()
        )
    )

    assert (
        coverage[
            "primary_recall"
        ]
        == 1.0
    )

    assert (
        coverage[
            "relevant_recall"
        ]
        == 1.0
    )

    assert (
        coverage[
            "all_primary_selected"
        ]
        is True
    )

    assert (
        coverage[
            "all_relevant_selected"
        ]
        is True
    )

    assert (
        coverage[
            "missing_relevant_ids"
        ]
        == []
    )


def test_primary_missing_is_primary_context_failure() -> None:
    """Missing central evidence should be classified before prompt behavior."""

    row = (
        make_row(
            selected_ids=[
                "supporting",
                "other",
            ],
        )
    )

    coverage = (
        calculate_selected_gold_coverage(
            row
        )
    )

    assert (
        classify_failure_cause(
            row,
            coverage,
        )
        == CAUSE_PRIMARY_NOT_SELECTED
    )


def test_full_context_incomplete_is_generation_failure() -> None:
    """Incomplete answer with all gold context isolates generation behavior."""

    row = (
        make_row(
            completeness_label=(
                "incomplete"
            ),
        )
    )

    coverage = (
        calculate_selected_gold_coverage(
            row
        )
    )

    assert (
        classify_failure_cause(
            row,
            coverage,
        )
        == CAUSE_GENERATION
    )


def test_partial_context_incomplete_is_mixed_failure() -> None:
    """Incomplete answer with primary but missing support is a mixed case."""

    row = (
        make_row(
            completeness_label=(
                "incomplete"
            ),
            selected_ids=[
                "primary",
                "other",
            ],
        )
    )

    coverage = (
        calculate_selected_gold_coverage(
            row
        )
    )

    assert (
        classify_failure_cause(
            row,
            coverage,
        )
        == CAUSE_MIXED
    )


def test_partial_context_mostly_complete_is_supporting_gap() -> None:
    """Mostly-complete answer with primary selected points to support coverage."""

    row = (
        make_row(
            completeness_label=(
                "mostly_complete"
            ),
            selected_ids=[
                "primary",
                "other",
            ],
        )
    )

    coverage = (
        calculate_selected_gold_coverage(
            row
        )
    )

    assert (
        classify_failure_cause(
            row,
            coverage,
        )
        == CAUSE_SUPPORTING_NOT_SELECTED
    )


def test_complete_answer_has_no_failure_record() -> None:
    """Complete answers should not enter the failure taxonomy."""

    row = (
        make_row(
            completeness_label=(
                "complete"
            ),
        )
    )

    assert (
        build_failure_record(
            row
        )
        is None
    )


def test_build_analysis_separates_intervention_candidates() -> None:
    """Analysis should distinguish clean prompt and context interventions."""

    rows = [
        make_row(
            question_id="complete",
            completeness_label="complete",
        ),
        make_row(
            question_id="generation",
            completeness_label="incomplete",
        ),
        make_row(
            question_id="mixed",
            completeness_label="incomplete",
            selected_ids=[
                "primary",
                "other",
            ],
        ),
        make_row(
            question_id="primary-gap",
            completeness_label="incomplete",
            selected_ids=[
                "supporting",
                "other",
            ],
        ),
        make_row(
            question_id="support-gap",
            completeness_label="mostly_complete",
            selected_ids=[
                "primary",
                "other",
            ],
        ),
    ]

    analysis = (
        build_analysis(
            rows
        )
    )

    assert (
        analysis[
            "question_count"
        ]
        == 5
    )

    assert (
        analysis[
            "fully_complete_count"
        ]
        == 1
    )

    assert (
        analysis[
            "non_complete_count"
        ]
        == 4
    )

    assert (
        analysis[
            "by_failure_cause"
        ][
            CAUSE_GENERATION
        ]
        == 1
    )

    assert (
        analysis[
            "by_failure_cause"
        ][
            CAUSE_MIXED
        ]
        == 1
    )

    assert (
        analysis[
            "by_failure_cause"
        ][
            CAUSE_PRIMARY_NOT_SELECTED
        ]
        == 1
    )

    assert (
        analysis[
            "by_failure_cause"
        ][
            CAUSE_SUPPORTING_NOT_SELECTED
        ]
        == 1
    )

    assert (
        analysis[
            "clean_generation_intervention_question_ids"
        ]
        == [
            "generation",
        ]
    )

    assert (
        analysis[
            "mixed_generation_intervention_question_ids"
        ]
        == [
            "mixed",
        ]
    )

    assert (
        analysis[
            "prompt_benchmark_question_ids"
        ]
        == [
            "generation",
            "mixed",
        ]
    )

    assert set(
        analysis[
            "retrieval_or_context_intervention_question_ids"
        ]
    ) == {
        "mixed",
        "primary-gap",
        "support-gap",
    }


def test_coverage_rejects_inconsistent_persisted_recall() -> None:
    """Persisted structural metrics must agree with evidence identities."""

    row = (
        make_row()
    )

    row[
        "selected_relevant_recall"
    ] = 0.5

    with pytest.raises(
        ValueError,
        match=(
            "selected_relevant_recall"
        ),
    ):
        calculate_selected_gold_coverage(
            row
        )


def test_coverage_rejects_duplicate_selected_ids() -> None:
    """Duplicate selected evidence should fail explicitly."""

    row = (
        make_row(
            selected_ids=[
                "primary",
                "primary",
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate point IDs"
        ),
    ):
        calculate_selected_gold_coverage(
            row
        )


def test_failure_record_preserves_human_review_notes() -> None:
    """Human notes should stay attached to the deterministic taxonomy."""

    row = (
        make_row(
            question_id="generation",
        )
    )

    row[
        "review_notes"
    ] = (
        "The selected passage contained "
        "the omitted numeric result."
    )

    result = (
        build_failure_record(
            row
        )
    )

    assert (
        result
        is not None
    )

    assert (
        result[
            "review_notes"
        ]
        == (
            "The selected passage contained "
            "the omitted numeric result."
        )
    )