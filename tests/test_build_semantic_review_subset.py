"""Tests for deterministic semantic-review subset sampling."""

from __future__ import annotations

import json

import pytest

from src.evaluation.build_semantic_review_subset import (
    EXPECTED_PER_PAIR,
    EXPECTED_TOTAL,
    PAIR_ORDER,
    SAMPLE_CONFIG_ID,
    STRATUM_QUOTAS,
    build_review_subset,
    build_semantic_review_subset,
    classify_claim_stratum,
    contains_numeric_content,
    summarize_subset,
)


def make_claim(
    claim_id: str,
    *,
    query_language: str = "en",
    target_language: str = "en",
    evidence_ids: list[str] | None = None,
    numeric: bool = False,
) -> dict:
    """Create one semantic claim fixture."""

    resolved_evidence_ids = (
        evidence_ids
        if evidence_ids is not None
        else [
            "E1",
        ]
    )

    claim_text = (
        "The literacy rate was 76.3%."
        if numeric
        else "The government provides this service."
    )

    return {
        "schema_version": 1,
        "source_run_config_id": (
            "production-rag-v2-interactions"
        ),
        "source_schema_version": 1,
        "question_id": (
            f"{query_language}_"
            f"{target_language}_question"
        ),
        "query": "Fixture question",
        "query_language": (
            query_language
        ),
        "target_language": (
            target_language
        ),
        "answer_language": (
            query_language
        ),
        "category": "test",
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "claim_id": claim_id,
        "claim_index": 1,
        "claim_text": claim_text,
        "raw_text": claim_text,
        "has_citation": bool(
            resolved_evidence_ids
        ),
        "evidence_ids": list(
            resolved_evidence_ids
        ),
        "cited_evidence": [],
        "semantic_support_label": None,
        "citation_requirement_label": None,
        "semantic_notes": None,
    }


def make_full_sampling_fixture() -> list[dict]:
    """Create enough rows to satisfy every production stratum quota."""

    rows: list[dict] = []

    for query_language, target_language in (
        PAIR_ORDER
    ):
        prefix = (
            f"{query_language}_"
            f"{target_language}"
        )

        # Add more candidates than required so deterministic selection matters.
        for index in range(
            4
        ):
            rows.append(
                make_claim(
                    f"{prefix}_multi_{index}",
                    query_language=(
                        query_language
                    ),
                    target_language=(
                        target_language
                    ),
                    evidence_ids=[
                        "E1",
                        "E2",
                    ],
                    numeric=(
                        index % 2 == 0
                    ),
                )
            )

        for index in range(
            5
        ):
            rows.append(
                make_claim(
                    f"{prefix}_single_num_{index}",
                    query_language=(
                        query_language
                    ),
                    target_language=(
                        target_language
                    ),
                    evidence_ids=[
                        "E1",
                    ],
                    numeric=True,
                )
            )

        for index in range(
            7
        ):
            rows.append(
                make_claim(
                    f"{prefix}_single_other_{index}",
                    query_language=(
                        query_language
                    ),
                    target_language=(
                        target_language
                    ),
                    evidence_ids=[
                        "E1",
                    ],
                    numeric=False,
                )
            )

        for index in range(
            3
        ):
            rows.append(
                make_claim(
                    f"{prefix}_uncited_num_{index}",
                    query_language=(
                        query_language
                    ),
                    target_language=(
                        target_language
                    ),
                    evidence_ids=[],
                    numeric=True,
                )
            )

        for index in range(
            3
        ):
            rows.append(
                make_claim(
                    f"{prefix}_uncited_other_{index}",
                    query_language=(
                        query_language
                    ),
                    target_language=(
                        target_language
                    ),
                    evidence_ids=[],
                    numeric=False,
                )
            )

    return rows


def test_detects_ascii_numeric_content() -> None:
    assert (
        contains_numeric_content(
            "Literacy was 76.3%."
        )
        is True
    )


def test_detects_devanagari_numeric_content() -> None:
    assert (
        contains_numeric_content(
            "साक्षरता दर ७६.३ प्रतिशत थियो।"
        )
        is True
    )


def test_classifies_multi_citation_claim() -> None:
    row = make_claim(
        "claim-1",
        evidence_ids=[
            "E1",
            "E2",
        ],
    )

    assert (
        classify_claim_stratum(
            row
        )
        == "multi_citation"
    )


def test_classifies_single_numeric_claim() -> None:
    row = make_claim(
        "claim-1",
        evidence_ids=[
            "E1",
        ],
        numeric=True,
    )

    assert (
        classify_claim_stratum(
            row
        )
        == "single_numeric"
    )


def test_classifies_single_other_claim() -> None:
    row = make_claim(
        "claim-1",
        evidence_ids=[
            "E1",
        ],
        numeric=False,
    )

    assert (
        classify_claim_stratum(
            row
        )
        == "single_other"
    )


def test_classifies_uncited_numeric_claim() -> None:
    row = make_claim(
        "claim-1",
        evidence_ids=[],
        numeric=True,
    )

    assert (
        classify_claim_stratum(
            row
        )
        == "uncited_numeric"
    )


def test_classifies_uncited_other_claim() -> None:
    row = make_claim(
        "claim-1",
        evidence_ids=[],
        numeric=False,
    )

    assert (
        classify_claim_stratum(
            row
        )
        == "uncited_other"
    )


def test_rejects_inconsistent_has_citation() -> None:
    row = make_claim(
        "claim-1",
        evidence_ids=[
            "E1",
        ],
    )

    row[
        "has_citation"
    ] = False

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        classify_claim_stratum(
            row
        )


def test_sampling_is_independent_of_source_order() -> None:
    rows = (
        make_full_sampling_fixture()
    )

    first = (
        build_review_subset(
            rows
        )
    )

    second = (
        build_review_subset(
            list(
                reversed(
                    rows
                )
            )
        )
    )

    assert [
        row[
            "claim_id"
        ]
        for row in first
    ] == [
        row[
            "claim_id"
        ]
        for row in second
    ]


def test_review_subset_has_expected_total_and_pair_counts() -> None:
    selected = (
        build_review_subset(
            make_full_sampling_fixture()
        )
    )

    assert len(
        selected
    ) == EXPECTED_TOTAL

    for pair in PAIR_ORDER:
        pair_rows = [
            row
            for row in selected
            if (
                row[
                    "query_language"
                ],
                row[
                    "target_language"
                ],
            )
            == pair
        ]

        assert len(
            pair_rows
        ) == EXPECTED_PER_PAIR


def test_review_subset_matches_all_stratum_quotas() -> None:
    selected = (
        build_review_subset(
            make_full_sampling_fixture()
        )
    )

    summary = (
        summarize_subset(
            selected
        )
    )

    for pair in PAIR_ORDER:
        assert (
            summary[
                pair
            ]
            == STRATUM_QUOTAS
        )


def test_review_subset_rejects_insufficient_stratum() -> None:
    rows = (
        make_full_sampling_fixture()
    )

    rows = [
        row
        for row in rows
        if not (
            row[
                "query_language"
            ]
            == "ne"
            and row[
                "target_language"
            ]
            == "en"
            and classify_claim_stratum(
                row
            )
            == "multi_citation"
        )
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Insufficient semantic claims"
        ),
    ):
        build_review_subset(
            rows
        )


def test_end_to_end_review_subset_is_written(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "semantic.jsonl"
    )

    output_path = (
        tmp_path
        / "review.jsonl"
    )

    rows = (
        make_full_sampling_fixture()
    )

    with source_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )

    selected = (
        build_semantic_review_subset(
            source_path,
            output_path=(
                output_path
            ),
        )
    )

    assert len(
        selected
    ) == EXPECTED_TOTAL

    assert output_path.exists()

    persisted = [
        json.loads(
            line
        )
        for line in (
            output_path.read_text(
                encoding="utf-8",
            )
            .splitlines()
        )
        if line.strip()
    ]

    assert len(
        persisted
    ) == EXPECTED_TOTAL

    assert all(
        row[
            "review_sample_config_id"
        ]
        == SAMPLE_CONFIG_ID
        for row in persisted
    )

    assert all(
        row[
            "review_status"
        ]
        == "pending"
        for row in persisted
    )