"""Tests for the third consolidated NepalGov AI quality checkpoint."""

from __future__ import annotations

import pytest

from src.evaluation.production_quality_checkpoint_v3 import (
    CHECKPOINT_CONFIG_ID,
    EXPECTED_AUTOMATIC_TRANSLATION_CONFIG_ID,
    EXPECTED_TABLE_AWARE_CONFIG_ID,
    build_production_quality_checkpoint_v3,
    summarize_automatic_translation_retrieval,
    summarize_completeness_failures,
    summarize_completeness_prompt_review,
    summarize_table_aware_overlay,
)


def test_automatic_translation_checkpoint_matches_measured_result() -> None:
    """Automatic translation should not be mistaken for the manual control."""

    summary = (
        summarize_automatic_translation_retrieval()
    )

    assert (
        summary[
            "run_config_id"
        ]
        == EXPECTED_AUTOMATIC_TRANSLATION_CONFIG_ID
    )

    assert (
        summary[
            "question_count"
        ]
        == 6
    )

    assert (
        summary[
            "production_changed"
        ]
        is False
    )

    automatic = (
        summary[
            "bge_at_10"
        ][
            "automatic_translation"
        ]
    )

    manual = (
        summary[
            "bge_at_10"
        ][
            "manual_control"
        ]
    )

    assert (
        automatic[
            "hit_rate"
        ]
        == pytest.approx(
            5 / 6
        )
    )

    assert (
        automatic[
            "recall"
        ]
        == pytest.approx(
            5 / 6
        )
    )

    assert (
        manual[
            "hit_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        manual[
            "recall"
        ]
        == pytest.approx(
            1.0
        )
    )

    hard_case = (
        summary[
            "ne_en_005"
        ]
    )

    assert (
        hard_case[
            "automatic_bge_rank"
        ]
        is None
    )

    assert (
        hard_case[
            "automatic_dense_rank"
        ]
        == 88
    )

    assert (
        hard_case[
            "manual_bge_rank"
        ]
        == 6
    )


def test_table_aware_checkpoint_preserves_mixed_result() -> None:
    """Table-aware recall gain and rank-quality regression should both remain."""

    summary = (
        summarize_table_aware_overlay()
    )

    assert (
        summary[
            "run_config_id"
        ]
        == EXPECTED_TABLE_AWARE_CONFIG_ID
    )

    assert (
        summary[
            "question_count"
        ]
        == 5
    )

    assert (
        summary[
            "table_like_point_count"
        ]
        == 204
    )

    metrics = (
        summary[
            "metrics_at_20"
        ]
    )

    assert (
        metrics[
            "contextual"
        ][
            "hit_rate"
        ]
        == pytest.approx(
            0.4
        )
    )

    assert (
        metrics[
            "table_aware_overlay"
        ][
            "hit_rate"
        ]
        == pytest.approx(
            0.6
        )
    )

    assert (
        metrics[
            "contextual"
        ][
            "recall"
        ]
        == pytest.approx(
            0.25
        )
    )

    assert (
        metrics[
            "table_aware_overlay"
        ][
            "recall"
        ]
        == pytest.approx(
            0.45
        )
    )

    assert (
        metrics[
            "table_aware_overlay"
        ][
            "mrr"
        ]
        < metrics[
            "contextual"
        ][
            "mrr"
        ]
    )

    assert (
        summary[
            "primary_ranks"
        ][
            "en_en_005"
        ][
            "contextual"
        ]
        == 38
    )

    assert (
        summary[
            "primary_ranks"
        ][
            "en_en_005"
        ][
            "table_aware_overlay"
        ]
        == 13
    )

    assert (
        summary[
            "primary_ranks"
        ][
            "ne_en_005"
        ][
            "contextual"
        ]
        == 840
    )

    assert (
        summary[
            "primary_ranks"
        ][
            "ne_en_005"
        ][
            "table_aware_overlay"
        ]
        == 936
    )


def test_completeness_failure_checkpoint_identifies_upstream_limit() -> None:
    """Most non-complete answers should remain context-coverage constrained."""

    summary = (
        summarize_completeness_failures()
    )

    assert (
        summary[
            "question_count"
        ]
        == 30
    )

    assert (
        summary[
            "fully_complete_count"
        ]
        == 21
    )

    assert (
        summary[
            "non_complete_count"
        ]
        == 9
    )

    causes = (
        summary[
            "by_failure_cause"
        ]
    )

    assert (
        causes[
            "primary_evidence_not_selected"
        ]
        == 5
    )

    assert (
        causes[
            "supporting_evidence_not_selected"
        ]
        == 2
    )

    assert (
        causes[
            "mixed_context_and_generation_omission"
        ]
        == 1
    )

    assert (
        causes[
            "generation_omission_despite_full_gold_context"
        ]
        == 1
    )

    assert (
        summary[
            "pure_context_limited_count"
        ]
        == 7
    )

    assert (
        summary[
            "retrieval_or_context_candidate_count"
        ]
        == 8
    )

    assert (
        summary[
            "clean_generation_intervention_count"
        ]
        == 1
    )


def test_completeness_prompt_checkpoint_matches_human_review() -> None:
    """Full human review should preserve the measured +1/30 improvement."""

    summary = (
        summarize_completeness_prompt_review()
    )

    assert (
        summary[
            "question_count"
        ]
        == 30
    )

    baseline = (
        summary[
            "baseline"
        ]
    )

    candidate = (
        summary[
            "candidate"
        ]
    )

    assert (
        baseline[
            "fully_complete_rate"
        ]
        == pytest.approx(
            21 / 30
        )
    )

    assert (
        candidate[
            "fully_complete_rate"
        ]
        == pytest.approx(
            22 / 30
        )
    )

    assert (
        baseline[
            "at_least_mostly_complete_rate"
        ]
        == pytest.approx(
            24 / 30
        )
    )

    assert (
        candidate[
            "at_least_mostly_complete_rate"
        ]
        == pytest.approx(
            25 / 30
        )
    )

    assert (
        baseline[
            "fully_faithful_rate"
        ]
        == pytest.approx(
            28 / 30
        )
    )

    assert (
        candidate[
            "fully_faithful_rate"
        ]
        == pytest.approx(
            28 / 30
        )
    )

    assert (
        candidate[
            "no_major_factual_error_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        candidate[
            "major_factual_issue_count"
        ]
        == 0
    )

    assert (
        summary[
            "delta"
        ][
            "fully_complete_rate"
        ]
        == pytest.approx(
            1 / 30
        )
    )

    en_ne = (
        summary[
            "by_language_pair"
        ][
            "en->ne"
        ]
    )

    assert (
        en_ne[
            "baseline"
        ][
            "fully_faithful_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        en_ne[
            "candidate"
        ][
            "fully_faithful_rate"
        ]
        == pytest.approx(
            5 / 6
        )
    )


def test_checkpoint_v3_consolidates_current_cycle() -> None:
    """V3 should preserve V2 and add all four current-cycle analyses."""

    checkpoint = (
        build_production_quality_checkpoint_v3()
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
        == "production-quality-checkpoint-v2"
    )

    assert set(
        checkpoint[
            "cycle_extensions"
        ]
    ) == {
        "automatic_ne_en_translation",
        "table_aware_passage_representation",
        "answer_completeness_failure_analysis",
        "answer_completeness_prompt",
    }

    decision = (
        checkpoint[
            "production_decision"
        ]
    )

    assert (
        decision[
            "production_retrieval_changed"
        ]
        is False
    )

    assert (
        decision[
            "production_prompt_changed"
        ]
        is False
    )

    assert (
        decision[
            "qdrant_reindexed"
        ]
        is False
    )

    assert (
        decision[
            "automatic_translation_promoted"
        ]
        is False
    )

    assert (
        decision[
            "table_aware_representation_promoted"
        ]
        is False
    )

    assert (
        decision[
            "completeness_prompt_promoted"
        ]
        is False
    )

    assert (
        decision[
            "official_production_benchmark_retained"
        ]
        is True
    )

    assert (
        checkpoint[
            "next_phase"
        ][
            "recommended"
        ]
        == "application_api_readiness"
    )


def test_checkpoint_v3_headline_matches_measured_cycle() -> None:
    """Headline metrics should expose the current production decision evidence."""

    checkpoint = (
        build_production_quality_checkpoint_v3()
    )

    headline = (
        checkpoint[
            "headline"
        ]
    )

    assert (
        headline[
            "automatic_translation_bge_hit_at_10"
        ]
        == pytest.approx(
            5 / 6
        )
    )

    assert (
        headline[
            "manual_control_bge_hit_at_10"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "table_contextual_hit_at_20"
        ]
        == pytest.approx(
            0.4
        )
    )

    assert (
        headline[
            "table_overlay_hit_at_20"
        ]
        == pytest.approx(
            0.6
        )
    )

    assert (
        headline[
            "non_complete_production_answer_count"
        ]
        == 9
    )

    assert (
        headline[
            "pure_context_limited_answer_count"
        ]
        == 7
    )

    assert (
        headline[
            "baseline_fully_complete_rate"
        ]
        == pytest.approx(
            0.7
        )
    )

    assert (
        headline[
            "candidate_fully_complete_rate"
        ]
        == pytest.approx(
            22 / 30
        )
    )

    assert (
        headline[
            "baseline_no_major_factual_error_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "candidate_no_major_factual_error_rate"
        ]
        == pytest.approx(
            1.0
        )
    )