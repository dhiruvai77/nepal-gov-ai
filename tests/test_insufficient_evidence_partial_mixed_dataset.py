"""Tests for the partial/mixed insufficient-evidence extension dataset."""

from __future__ import annotations

from collections import Counter

from src.evaluation.insufficient_evidence_evaluator import (
    EXPECTED_BEHAVIOR_PARTIAL,
    load_insufficient_evidence_records,
)


DATASET_PATH = (
    "data/evaluation/"
    "insufficient_evidence_partial_mixed_v1.jsonl"
)


def test_partial_mixed_dataset_has_eight_cases() -> None:
    records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    assert len(
        records
    ) == 8


def test_partial_mixed_dataset_is_balanced_by_language_pair() -> None:
    records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    counts = Counter(
        (
            record.query_language,
            record.target_language,
        )
        for record in records
    )

    assert counts == {
        ("en", "en"): 2,
        ("ne", "ne"): 2,
        ("en", "ne"): 2,
        ("ne", "en"): 2,
    }


def test_each_language_pair_has_one_partial_and_one_mixed_case() -> None:
    records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    counts = Counter(
        (
            record.query_language,
            record.target_language,
            record.case_type,
        )
        for record in records
    )

    for pair in (
        ("en", "en"),
        ("ne", "ne"),
        ("en", "ne"),
        ("ne", "en"),
    ):
        assert (
            counts[
                (
                    pair[0],
                    pair[1],
                    "partial_evidence",
                )
            ]
            == 1
        )

        assert (
            counts[
                (
                    pair[0],
                    pair[1],
                    "mixed_supported_unsupported",
                )
            ]
            == 1
        )


def test_all_extension_cases_expect_partial_behavior() -> None:
    records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    assert all(
        record.expected_behavior
        == EXPECTED_BEHAVIOR_PARTIAL
        for record in records
    )


def test_all_extension_cases_have_supported_document_target() -> None:
    records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    assert all(
        record.expected_document_ids
        for record in records
    )


def test_extension_question_ids_are_unique() -> None:
    records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    question_ids = [
        record.question_id
        for record in records
    ]

    assert len(
        question_ids
    ) == len(
        set(
            question_ids
        )
    )


def test_extension_does_not_reuse_base_question_ids() -> None:
    base_records = (
        load_insufficient_evidence_records(
            "data/evaluation/"
            "insufficient_evidence_questions.jsonl"
        )
    )

    extension_records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    base_ids = {
        record.question_id
        for record in base_records
    }

    extension_ids = {
        record.question_id
        for record in extension_records
    }

    assert (
        base_ids
        .isdisjoint(
            extension_ids
        )
    )


def test_combined_benchmark_family_has_twenty_four_cases() -> None:
    base_records = (
        load_insufficient_evidence_records(
            "data/evaluation/"
            "insufficient_evidence_questions.jsonl"
        )
    )

    extension_records = (
        load_insufficient_evidence_records(
            DATASET_PATH
        )
    )

    assert (
        len(
            base_records
        )
        + len(
            extension_records
        )
        == 24
    )