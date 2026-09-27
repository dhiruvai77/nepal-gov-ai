"""Tests for the full fixed-context completeness-prompt benchmark."""

from __future__ import annotations

import json

import pytest

from src.evaluation.run_completeness_prompt_full_benchmark import (
    FULL_RUN_CONFIG_ID,
    FULL_SCHEMA_VERSION,
    build_full_record,
    load_existing_full_output,
)
from src.evaluation.run_completeness_prompt_experiment import (
    VARIANT_COMPLETENESS,
    build_request_from_review_row,
)


def make_evidence(
    *,
    evidence_id: str = "E1",
    point_id: str = "point-1",
) -> dict:
    """Build one selected evidence fixture."""

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
            "The reported value is 42 [E1 is "
            "not source syntax; this is passage text]."
        ),
        "is_primary": True,
        "is_gold_relevant": True,
    }


def make_row(
    *,
    question_id: str = "en_en_001",
) -> dict:
    """Build one completed source review row."""

    return {
        "review_config_id": (
            "production-rag-v2-answer-quality-v1"
        ),
        "review_status": "completed",
        "question_id": question_id,
        "query": "What value is reported?",
        "query_language": "en",
        "target_language": "en",
        "answer_language": "en",
        "category": "test",
        "selected_evidence": [
            make_evidence(),
        ],
    }


def test_build_full_record_uses_full_run_identity() -> None:
    """Full benchmark should have its own immutable run identity."""

    row = (
        make_row()
    )

    record = (
        build_full_record(
            row,
            generated_answer_text=(
                "The reported value is 42 [E1]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    assert (
        record[
            "schema_version"
        ]
        == FULL_SCHEMA_VERSION
    )

    assert (
        record[
            "run_config_id"
        ]
        == FULL_RUN_CONFIG_ID
    )

    assert (
        record[
            "variant"
        ]
        == VARIANT_COMPLETENESS
    )

    assert (
        record[
            "production_changed"
        ]
        is False
    )


def test_build_full_record_preserves_selected_context() -> None:
    """Generation-only benchmark must not alter evidence identities."""

    row = (
        make_row()
    )

    request = (
        build_request_from_review_row(
            row
        )
    )

    record = (
        build_full_record(
            row,
            generated_answer_text=(
                "The reported value is 42 [E1]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    assert (
        record[
            "selected_point_ids"
        ]
        == [
            item.result.point_id
            for item in (
                request.context
            )
        ]
    )


def test_build_full_record_runs_structural_guard() -> None:
    """Full benchmark should preserve production citation gating."""

    row = (
        make_row()
    )

    record = (
        build_full_record(
            row,
            generated_answer_text=(
                "The reported value is 42 [E1]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    assert (
        record[
            "accepted"
        ]
        is True
    )

    assert (
        record[
            "invalid_evidence_ids"
        ]
        == []
    )


def test_load_existing_full_output_round_trip(
    tmp_path,
) -> None:
    """Persisted rows should be safely reusable."""

    row = (
        make_row()
    )

    record = (
        build_full_record(
            row,
            generated_answer_text=(
                "The reported value is 42 [E1]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    output = (
        tmp_path
        / "benchmark.jsonl"
    )

    output.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = (
        load_existing_full_output(
            output,
            [
                row,
            ],
        )
    )

    assert (
        loaded[
            "en_en_001"
        ][
            "answer_text"
        ]
        == (
            "The reported value is 42 [E1]."
        )
    )


def test_load_existing_full_output_rejects_duplicate(
    tmp_path,
) -> None:
    """Duplicate question rows must fail explicitly."""

    row = (
        make_row()
    )

    record = (
        build_full_record(
            row,
            generated_answer_text=(
                "The reported value is 42 [E1]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    serialized = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    output = (
        tmp_path
        / "benchmark.jsonl"
    )

    output.write_text(
        serialized
        + "\n"
        + serialized
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate question_id"
        ),
    ):
        load_existing_full_output(
            output,
            [
                row,
            ],
        )