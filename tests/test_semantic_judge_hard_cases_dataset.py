"""Tests for the curated semantic-judge hard-case reference set."""

from __future__ import annotations

from collections import Counter

from src.evaluation.semantic_reference import (
    load_semantic_reference_rows,
)


DATASET_PATH = (
    "data/evaluation/semantic/"
    "semantic_judge_hard_cases_v1.jsonl"
)

REFERENCE_CONFIG_ID = (
    "semantic-judge-hard-cases-v1"
)

EXPECTED_PAIRS = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)

EXPECTED_STRATA = (
    "hard_unsupported",
    "hard_needs_review",
    "hard_citation_unclear",
)


def load_rows() -> list[
    dict
]:
    """Load the committed hard-case reference."""

    return (
        load_semantic_reference_rows(
            DATASET_PATH
        )
    )


def test_hard_case_dataset_has_twelve_rows() -> None:
    rows = (
        load_rows()
    )

    assert (
        len(
            rows
        )
        == 12
    )


def test_hard_case_dataset_has_unique_claim_ids() -> None:
    rows = (
        load_rows()
    )

    claim_ids = [
        row[
            "claim_id"
        ]
        for row in rows
    ]

    assert (
        len(
            claim_ids
        )
        == len(
            set(
                claim_ids
            )
        )
    )


def test_hard_case_dataset_uses_separate_reference_config() -> None:
    rows = (
        load_rows()
    )

    assert all(
        row[
            "review_sample_config_id"
        ]
        == REFERENCE_CONFIG_ID
        for row in rows
    )


def test_hard_case_dataset_is_completed() -> None:
    rows = (
        load_rows()
    )

    assert all(
        row[
            "review_status"
        ]
        == "completed"
        for row in rows
    )


def test_hard_case_dataset_is_balanced_by_language_pair() -> None:
    rows = (
        load_rows()
    )

    counts = Counter(
        (
            row[
                "query_language"
            ],
            row[
                "target_language"
            ],
        )
        for row in rows
    )

    assert counts == {
        pair: 3
        for pair in (
            EXPECTED_PAIRS
        )
    }


def test_each_language_pair_contains_all_three_hard_classes() -> None:
    rows = (
        load_rows()
    )

    counts = Counter(
        (
            row[
                "query_language"
            ],
            row[
                "target_language"
            ],
            row[
                "review_stratum"
            ],
        )
        for row in rows
    )

    for (
        query_language,
        target_language,
    ) in EXPECTED_PAIRS:
        for stratum in (
            EXPECTED_STRATA
        ):
            assert (
                counts[
                    (
                        query_language,
                        target_language,
                        stratum,
                    )
                ]
                == 1
            )


def test_hard_case_dataset_has_four_unsupported_cases() -> None:
    rows = (
        load_rows()
    )

    unsupported = [
        row
        for row in rows
        if (
            row[
                "review_stratum"
            ]
            == "hard_unsupported"
        )
    ]

    assert (
        len(
            unsupported
        )
        == 4
    )

    assert all(
        row[
            "semantic_support_label"
        ]
        == "unsupported"
        for row in unsupported
    )

    assert all(
        row[
            "citation_requirement_label"
        ]
        == "required"
        for row in unsupported
    )

    assert all(
        row[
            "cited_evidence"
        ][
            0
        ][
            "individual_support_label"
        ]
        == "unsupported"
        for row in unsupported
    )


def test_hard_case_dataset_has_four_needs_review_cases() -> None:
    rows = (
        load_rows()
    )

    needs_review = [
        row
        for row in rows
        if (
            row[
                "review_stratum"
            ]
            == "hard_needs_review"
        )
    ]

    assert (
        len(
            needs_review
        )
        == 4
    )

    assert all(
        row[
            "semantic_support_label"
        ]
        == "needs_review"
        for row in needs_review
    )

    assert all(
        row[
            "citation_requirement_label"
        ]
        == "required"
        for row in needs_review
    )

    assert all(
        row[
            "cited_evidence"
        ][
            0
        ][
            "individual_support_label"
        ]
        == "needs_review"
        for row in needs_review
    )


def test_hard_case_dataset_has_four_unclear_requirement_cases() -> None:
    rows = (
        load_rows()
    )

    unclear = [
        row
        for row in rows
        if (
            row[
                "review_stratum"
            ]
            == "hard_citation_unclear"
        )
    ]

    assert (
        len(
            unclear
        )
        == 4
    )

    assert all(
        row[
            "semantic_support_label"
        ]
        == "supported"
        for row in unclear
    )

    assert all(
        row[
            "citation_requirement_label"
        ]
        == "unclear"
        for row in unclear
    )

    assert all(
        row[
            "cited_evidence"
        ][
            0
        ][
            "individual_support_label"
        ]
        == "supported"
        for row in unclear
    )


def test_evidence_language_matches_target_language() -> None:
    rows = (
        load_rows()
    )

    for row in rows:
        assert (
            row[
                "cited_evidence"
            ][
                0
            ][
                "language"
            ]
            == row[
                "target_language"
            ]
        )


def test_all_hard_cases_preserve_source_reference_provenance() -> None:
    rows = (
        load_rows()
    )

    assert all(
        isinstance(
            row[
                "cited_evidence"
            ][
                0
            ].get(
                "source_reference_claim_id"
            ),
            str,
        )
        and row[
            "cited_evidence"
        ][
            0
        ][
            "source_reference_claim_id"
        ]
        for row in rows
    )


def test_all_hard_cases_have_exactly_one_evidence_passage() -> None:
    rows = (
        load_rows()
    )

    assert all(
        row[
            "evidence_ids"
        ]
        == [
            "E1",
        ]
        for row in rows
    )

    assert all(
        len(
            row[
                "cited_evidence"
            ]
        )
        == 1
        for row in rows
    )