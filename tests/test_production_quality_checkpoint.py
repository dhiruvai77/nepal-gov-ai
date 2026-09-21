"""Tests for the consolidated production-quality checkpoint."""

from __future__ import annotations

import json

import pytest

from src.evaluation.production_quality_checkpoint import (
    CHECKPOINT_CONFIG_ID,
    CHECKPOINT_SCHEMA_VERSION,
    build_production_quality_checkpoint,
    load_jsonl_rows,
    write_checkpoint,
)


def test_default_checkpoint_builds_from_committed_artifacts() -> None:
    """Current committed evaluation artifacts should form a complete checkpoint."""

    checkpoint = (
        build_production_quality_checkpoint()
    )

    assert (
        checkpoint[
            "schema_version"
        ]
        == CHECKPOINT_SCHEMA_VERSION
    )

    assert (
        checkpoint[
            "checkpoint_config_id"
        ]
        == CHECKPOINT_CONFIG_ID
    )

    assert (
        checkpoint[
            "production_rag"
        ][
            "question_count"
        ]
        == 30
    )

    assert (
        checkpoint[
            "human_semantic_citation_review"
        ][
            "claim_count"
        ]
        == 48
    )

    assert (
        checkpoint[
            "automated_semantic_judge"
        ][
            "claim_count"
        ]
        == 48
    )

    assert (
        checkpoint[
            "insufficient_evidence_structural"
        ][
            "question_count"
        ]
        == 16
    )

    assert (
        checkpoint[
            "human_response_behavior"
        ][
            "question_count"
        ]
        == 16
    )


def test_default_checkpoint_preserves_current_headline_metrics() -> None:
    """The checkpoint should reproduce the currently validated evaluation state."""

    checkpoint = (
        build_production_quality_checkpoint()
    )

    headline = (
        checkpoint[
            "headline"
        ]
    )

    assert (
        headline[
            "production_rag_selected_primary_hit_rate"
        ]
        == pytest.approx(
            0.8333333333
        )
    )

    assert (
        headline[
            "production_rag_valid_reference_ratio"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "human_semantic_fully_supported_rate"
        ]
        == pytest.approx(
            0.95
        )
    )

    assert (
        headline[
            "human_semantic_any_support_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "human_semantic_unsupported_rate"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        headline[
            "automated_judge_semantic_exact_accuracy"
        ]
        == pytest.approx(
            47 / 48
        )
    )

    assert (
        headline[
            "automated_judge_requirement_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "automated_judge_individual_exact_accuracy"
        ]
        == pytest.approx(
            0.9583333333
        )
    )

    assert (
        headline[
            "structural_withholding_success_rate"
        ]
        == pytest.approx(
            0.875
        )
    )

    assert (
        headline[
            "structural_false_accept_rate"
        ]
        == pytest.approx(
            0.125
        )
    )

    assert (
        headline[
            "human_response_behavior_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "human_semantic_abstention_success_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        headline[
            "unsafe_substantive_answer_rate"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_checkpoint_records_semantic_judge_reference_limits() -> None:
    checkpoint = (
        build_production_quality_checkpoint()
    )

    limits = (
        checkpoint[
            "automated_semantic_judge"
        ][
            "reference_limitations"
        ]
    )

    assert (
        limits[
            "human_unsupported_examples"
        ]
        == 0
    )

    assert (
        limits[
            "human_needs_review_examples"
        ]
        == 0
    )

    assert (
        limits[
            "human_unclear_requirement_examples"
        ]
        == 0
    )


def test_checkpoint_records_missing_insufficient_evidence_case_types() -> None:
    checkpoint = (
        build_production_quality_checkpoint()
    )

    missing = (
        checkpoint[
            "insufficient_evidence_structural"
        ][
            "not_yet_benchmarked_case_types"
        ]
    )

    assert (
        "partial_evidence"
        in missing
    )

    assert (
        "mixed_supported_unsupported"
        in missing
    )


def test_checkpoint_explicitly_records_no_hosted_rerun() -> None:
    checkpoint = (
        build_production_quality_checkpoint()
    )

    policy = (
        checkpoint[
            "rerun_policy"
        ]
    )

    assert (
        policy[
            "hosted_production_rerun_performed"
        ]
        is False
    )

    assert (
        policy[
            "official_production_benchmark_retained"
        ]
        is True
    )


def test_checkpoint_is_deterministic() -> None:
    first = (
        build_production_quality_checkpoint()
    )

    second = (
        build_production_quality_checkpoint()
    )

    assert (
        first
        == second
    )


def test_write_checkpoint_round_trips(
    tmp_path,
) -> None:
    checkpoint = (
        build_production_quality_checkpoint()
    )

    path = (
        tmp_path
        / "checkpoint.json"
    )

    write_checkpoint(
        checkpoint,
        path,
    )

    loaded = (
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    )

    assert (
        loaded
        == checkpoint
    )


def test_load_jsonl_rows_rejects_empty_file(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "empty.jsonl"
    )

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="no evaluation rows",
    ):
        load_jsonl_rows(
            path
        )


def test_load_jsonl_rows_rejects_non_object(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "invalid.jsonl"
    )

    path.write_text(
        "[1, 2, 3]\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        load_jsonl_rows(
            path
        )