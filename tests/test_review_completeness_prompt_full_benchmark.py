"""Tests for human review of the full completeness-prompt benchmark."""

from __future__ import annotations

import json

import pytest

from src.evaluation.review_completeness_prompt_full_benchmark import (
    PAIR_REVIEW_CONFIG_ID,
    PAIR_REVIEW_SCHEMA_VERSION,
    apply_candidate_labels,
    build_pair_review_row,
    candidate_as_standard_review_row,
    load_pair_review_rows,
    metric_delta,
    validate_pair_review_row,
)
from src.evaluation.run_completeness_prompt_full_benchmark import (
    FULL_RUN_CONFIG_ID,
)


def make_evidence(
    *,
    evidence_id: str | None,
    point_id: str,
    is_primary: bool,
) -> dict:
    """Build one review evidence fixture."""

    return {
        "evidence_id": (
            evidence_id
        ),
        "point_id": (
            point_id
        ),
        "chunk_id": (
            f"chunk-{point_id}"
        ),
        "document_id": "doc",
        "title": "Government Document",
        "organization": "Government",
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "source_url": (
            "https://example.gov.np/doc"
        ),
        "chunk_text": (
            f"Evidence for {point_id}."
        ),
        "is_primary": (
            is_primary
        ),
        "is_gold_relevant": True,
    }


def make_baseline_row() -> dict:
    """Build one completed production answer-quality row."""

    primary = (
        make_evidence(
            evidence_id="E1",
            point_id="primary",
            is_primary=True,
        )
    )

    support_selected = (
        make_evidence(
            evidence_id="E2",
            point_id="supporting",
            is_primary=False,
        )
    )

    support_gold = dict(
        support_selected
    )

    support_gold[
        "evidence_id"
    ] = None

    primary_gold = dict(
        primary
    )

    primary_gold[
        "evidence_id"
    ] = None

    return {
        "review_schema_version": 1,
        "review_config_id": (
            "production-rag-v2-answer-quality-v1"
        ),
        "source_run_config_id": (
            "production-rag-v2-interactions"
        ),
        "question_id": "en_en_001",
        "query": "What does the document report?",
        "query_language": "en",
        "target_language": "en",
        "answer_language": "en",
        "category": "test",
        "primary_relevant_chunk_ids": [
            "primary",
        ],
        "relevant_chunk_ids": [
            "primary",
            "supporting",
        ],
        "answer_text": (
            "Original production answer [E1]."
        ),
        "accepted": True,
        "withheld": False,
        "selected_primary_hit": 1,
        "selected_relevant_recall": 1.0,
        "selected_evidence": [
            primary,
            support_selected,
        ],
        "gold_reference_evidence": [
            primary_gold,
            support_gold,
        ],
        "completeness_label": (
            "mostly_complete"
        ),
        "factual_fidelity_label": (
            "fully_faithful"
        ),
        "review_notes": (
            "Original note."
        ),
        "review_status": "completed",
    }


def make_candidate_row() -> dict:
    """Build one full completeness-prompt result fixture."""

    return {
        "schema_version": 1,
        "run_config_id": (
            FULL_RUN_CONFIG_ID
        ),
        "source_review_config_id": (
            "production-rag-v2-answer-quality-v1"
        ),
        "source_pilot_config_id": (
            "production-rag-v2-completeness-prompt-v1"
        ),
        "question_id": "en_en_001",
        "query": "What does the document report?",
        "query_language": "en",
        "target_language": "en",
        "answer_language": "en",
        "variant": "completeness_v1",
        "prompt_sha256": (
            "a" * 64
        ),
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "selected_point_ids": [
            "primary",
            "supporting",
        ],
        "generated_answer_text": (
            "Candidate answer [E1] [E2]."
        ),
        "answer_text": (
            "Candidate answer [E1] [E2]."
        ),
        "accepted": True,
        "withheld": False,
        "guard_reason": None,
        "cited_evidence_ids": [
            "E1",
            "E2",
        ],
        "invalid_evidence_ids": [],
        "production_changed": False,
    }


def test_build_pair_review_row_is_pending() -> None:
    """New candidate human-review rows should begin unlabeled."""

    row = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    assert (
        row[
            "review_schema_version"
        ]
        == PAIR_REVIEW_SCHEMA_VERSION
    )

    assert (
        row[
            "review_config_id"
        ]
        == PAIR_REVIEW_CONFIG_ID
    )

    assert (
        row[
            "review_status"
        ]
        == "pending"
    )

    assert (
        row[
            "candidate_completeness_label"
        ]
        is None
    )

    assert (
        row[
            "candidate_factual_fidelity_label"
        ]
        is None
    )


def test_build_pair_review_row_preserves_baseline_labels() -> None:
    """Original human labels should remain available for final comparison."""

    row = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    assert (
        row[
            "baseline_completeness_label"
        ]
        == "mostly_complete"
    )

    assert (
        row[
            "baseline_factual_fidelity_label"
        ]
        == "fully_faithful"
    )


def test_build_pair_review_row_requires_same_selected_context() -> None:
    """Candidate comparison is invalid if selected evidence changed."""

    candidate = (
        make_candidate_row()
    )

    candidate[
        "selected_point_ids"
    ] = [
        "different",
    ]

    with pytest.raises(
        ValueError,
        match=(
            "selected context"
        ),
    ):
        build_pair_review_row(
            make_baseline_row(),
            candidate,
        )


def test_apply_candidate_labels_completes_row() -> None:
    """Human labels should transition one candidate row to completed."""

    pending = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    completed = (
        apply_candidate_labels(
            pending,
            completeness_label="complete",
            factual_fidelity_label=(
                "fully_faithful"
            ),
            review_notes=(
                "Covers all material evidence."
            ),
        )
    )

    assert (
        completed[
            "review_status"
        ]
        == "completed"
    )

    assert (
        completed[
            "candidate_completeness_label"
        ]
        == "complete"
    )

    assert (
        completed[
            "candidate_factual_fidelity_label"
        ]
        == "fully_faithful"
    )


def test_candidate_adapter_uses_existing_metric_contract() -> None:
    """Candidate labels should map onto the established review schema."""

    pending = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    completed = (
        apply_candidate_labels(
            pending,
            completeness_label="complete",
            factual_fidelity_label=(
                "fully_faithful"
            ),
            review_notes=None,
        )
    )

    standard = (
        candidate_as_standard_review_row(
            completed
        )
    )

    assert (
        standard[
            "review_config_id"
        ]
        == (
            "production-rag-v2-answer-quality-v1"
        )
    )

    assert (
        standard[
            "completeness_label"
        ]
        == "complete"
    )

    assert (
        standard[
            "factual_fidelity_label"
        ]
        == "fully_faithful"
    )


def test_validate_pending_rejects_candidate_labels() -> None:
    """Pending rows must not contain hidden completed labels."""

    row = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    row[
        "candidate_completeness_label"
    ] = "complete"

    with pytest.raises(
        ValueError,
        match=(
            "pending row"
        ),
    ):
        validate_pair_review_row(
            row
        )


def test_load_pair_review_rows_round_trip(
    tmp_path,
) -> None:
    """Persisted paired review rows should reload safely."""

    row = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    path = (
        tmp_path
        / "review.jsonl"
    )

    path.write_text(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = (
        load_pair_review_rows(
            path
        )
    )

    assert (
        len(
            loaded
        )
        == 1
    )

    assert (
        loaded[
            0
        ][
            "question_id"
        ]
        == "en_en_001"
    )


def test_load_pair_review_rows_rejects_duplicates(
    tmp_path,
) -> None:
    """Duplicate question IDs must fail rather than double-count metrics."""

    row = (
        build_pair_review_row(
            make_baseline_row(),
            make_candidate_row(),
        )
    )

    serialized = (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    path = (
        tmp_path
        / "review.jsonl"
    )

    path.write_text(
        serialized
        + "\n"
        + serialized
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate paired-review"
        ),
    ):
        load_pair_review_rows(
            path
        )


def test_metric_delta() -> None:
    """Comparison delta should be candidate minus baseline."""

    baseline = {
        "fully_complete_rate": 0.7,
    }

    candidate = {
        "fully_complete_rate": 0.8,
    }

    assert (
        metric_delta(
            candidate,
            baseline,
            "fully_complete_rate",
        )
        == pytest.approx(
            0.1
        )
    )