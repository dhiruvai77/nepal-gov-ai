"""Tests for the second consolidated NepalGov AI quality checkpoint."""

from __future__ import annotations

import pytest

from src.evaluation.production_quality_checkpoint_v2 import (
    CHECKPOINT_CONFIG_ID,
    EXPECTED_NE_EN_RUN_CONFIG_ID,
    build_production_quality_checkpoint_v2,
    summarize_answer_quality,
    summarize_hard_semantic_judge,
    summarize_ne_en_retrieval_experiment,
    summarize_partial_mixed_evidence,
)


def test_partial_mixed_checkpoint_is_complete() -> None:
    """The persisted eight-case partial/mixed extension should remain perfect."""

    summary = (
        summarize_partial_mixed_evidence()
    )

    assert (
        summary[
            "question_count"
        ]
        == 8
    )

    structural = (
        summary[
            "structural"
        ][
            "overall"
        ]
    )

    assert (
        structural[
            "decision_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        structural[
            "partial_case_acceptance_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    response = (
        summary[
            "human_response_behavior"
        ][
            "overall"
        ]
    )

    assert (
        response[
            "completion_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        response[
            "behavior_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        response[
            "partial_response_success_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_hard_semantic_judge_checkpoint_matches_measured_result() -> None:
    """Hard-case ambiguity behavior should remain explicit in the checkpoint."""

    summary = (
        summarize_hard_semantic_judge()
    )

    assert (
        summary[
            "claim_count"
        ]
        == 12
    )

    overall = (
        summary[
            "overall"
        ]
    )

    assert (
        overall[
            "semantic_exact_accuracy"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        overall[
            "citation_requirement_accuracy"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        overall[
            "individual_exact_accuracy"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    unsupported = (
        summary[
            "by_human_semantic_label"
        ][
            "unsupported"
        ]
    )

    assert (
        unsupported[
            "semantic_exact_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    needs_review = (
        summary[
            "by_human_semantic_label"
        ][
            "needs_review"
        ]
    )

    assert (
        needs_review[
            "semantic_exact_accuracy"
        ]
        == pytest.approx(
            0.0
        )
    )

    unclear = (
        summary[
            "by_human_citation_requirement_label"
        ][
            "unclear"
        ]
    )

    assert (
        unclear[
            "citation_requirement_accuracy"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_targeted_ne_en_retrieval_checkpoint_remains_experimental() -> None:
    """The measured retrieval gain must not be represented as production."""

    summary = (
        summarize_ne_en_retrieval_experiment()
    )

    assert (
        summary[
            "run_config_id"
        ]
        == EXPECTED_NE_EN_RUN_CONFIG_ID
    )

    assert (
        summary[
            "production_changed"
        ]
        is False
    )

    production = (
        summary[
            "reranked_metrics"
        ][
            "10"
        ][
            "production_bge"
        ]
    )

    experimental = (
        summary[
            "reranked_metrics"
        ][
            "10"
        ][
            "translated_bge"
        ]
    )

    assert (
        production[
            "hit_rate"
        ]
        == pytest.approx(
            0.833
        )
    )

    assert (
        production[
            "recall"
        ]
        == pytest.approx(
            0.778
        )
    )

    assert (
        experimental[
            "hit_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        experimental[
            "recall"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_answer_quality_checkpoint_matches_completed_review() -> None:
    """The complete 30-question review should preserve measured quality rates."""

    summary = (
        summarize_answer_quality()
    )

    assert (
        summary[
            "question_count"
        ]
        == 30
    )

    overall = (
        summary[
            "overall"
        ]
    )

    assert (
        overall[
            "completion_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        overall[
            "fully_complete_rate"
        ]
        == pytest.approx(
            0.7
        )
    )

    assert (
        overall[
            "at_least_mostly_complete_rate"
        ]
        == pytest.approx(
            0.8
        )
    )

    assert (
        overall[
            "fully_faithful_rate"
        ]
        == pytest.approx(
            28 / 30
        )
    )

    assert (
        overall[
            "no_major_factual_error_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        overall[
            "minor_factual_issue_count"
        ]
        == 2
    )

    assert (
        overall[
            "major_factual_issue_count"
        ]
        == 0
    )

    assert (
        overall[
            "strong_answer_rate"
        ]
        == pytest.approx(
            20 / 30
        )
    )

    assert (
        overall[
            "acceptable_answer_rate"
        ]
        == pytest.approx(
            0.8
        )
    )


def test_checkpoint_v2_consolidates_all_four_cycle_extensions() -> None:
    """V2 should contain both the historical baseline and the new cycle."""

    checkpoint = (
        build_production_quality_checkpoint_v2()
    )

    assert (
        checkpoint[
            "checkpoint_config_id"
        ]
        == CHECKPOINT_CONFIG_ID
    )

    assert (
        checkpoint[
            "base_checkpoint_config_id"
        ]
        == "production-quality-checkpoint-v1"
    )

    extensions = (
        checkpoint[
            "cycle_extensions"
        ]
    )

    assert set(
        extensions
    ) == {
        "partial_mixed_evidence",
        "semantic_judge_hard_cases",
        "targeted_ne_en_retrieval",
        "human_answer_quality",
    }

    assert (
        checkpoint[
            "production_decision"
        ][
            "production_retrieval_changed"
        ]
        is False
    )

    assert (
        checkpoint[
            "production_decision"
        ][
            "official_production_benchmark_retained"
        ]
        is True
    )


def test_checkpoint_v2_headline_contains_new_quality_metrics() -> None:
    """The checkpoint headline should expose the new review dimensions."""

    checkpoint = (
        build_production_quality_checkpoint_v2()
    )

    headline = (
        checkpoint[
            "headline"
        ]
    )

    assert (
        headline[
            "partial_mixed_response_success_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "hard_case_judge_semantic_exact_accuracy"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        headline[
            "ne_en_experimental_bge_hit_at_10"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "answer_quality_fully_complete_rate"
        ]
        == pytest.approx(
            0.7
        )
    )

    assert (
        headline[
            "answer_quality_fully_faithful_rate"
        ]
        == pytest.approx(
            28 / 30
        )
    )

    assert (
        headline[
            "answer_quality_no_major_factual_error_rate"
        ]
        == pytest.approx(
            1.0
        )
    )