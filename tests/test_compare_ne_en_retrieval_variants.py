"""Tests for the targeted NE->EN retrieval-variant experiment."""

from __future__ import annotations

import pytest

from src.evaluation.compare_ne_en_retrieval_variants import (
    CONTROLLED_ENGLISH_QUERIES,
    DEFAULT_DATASET_PATH,
    DEFAULT_FINAL_CANDIDATE_DEPTH,
    TRANSLATED_HYBRID_COMPONENT_DEPTH,
    build_dual_route_fusion,
    build_translated_hybrid,
    calculate_metrics,
    first_primary_rank,
    format_rank,
    reciprocal_rank_fuse,
    select_ne_en_records,
    validate_controlled_queries,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


EXPECTED_NE_EN_IDS = (
    "ne_en_001",
    "ne_en_002",
    "ne_en_003",
    "ne_en_004",
    "ne_en_005",
    "ne_en_006",
)


def make_record(
    *,
    question_id: str = "ne_en_001",
    query_language: str = "ne",
    target_language: str = "en",
    primary_ids: tuple[str, ...] = (
        "primary",
    ),
    relevant_ids: tuple[str, ...] = (
        "primary",
        "supporting",
    ),
) -> EvaluationRecord:
    """Build one deterministic retrieval-evaluation record."""

    return EvaluationRecord(
        question_id=question_id,
        query=(
            "नेपालको संविधानले "
            "स्वास्थ्यसम्बन्धी अधिकारबारे "
            "के व्यवस्था गरेको छ?"
        ),
        query_language=query_language,
        target_language=target_language,
        category="constitution_law",
        expected_document_ids=(
            "constitution_nepal_current_en",
        ),
        primary_relevant_chunk_ids=primary_ids,
        relevant_chunk_ids=relevant_ids,
        notes="Test record.",
    )


def make_result(
    point_id: str,
    *,
    score: float = 1.0,
) -> RetrievalResult:
    """Build one deterministic normalized retrieval result."""

    return RetrievalResult(
        point_id=point_id,
        score=score,
        chunk_id=(
            f"chunk-{point_id}"
        ),
        document_id=(
            "constitution_nepal_current_en"
        ),
        title="Constitution of Nepal",
        organization=(
            "Nepal Law Commission"
        ),
        language="en",
        page_start=1,
        page_end=1,
        source_url=(
            "https://example.gov.np/"
        ),
        chunk_text=(
            f"Evidence for {point_id}."
        ),
        chunk_index=1,
        token_count=100,
        category=(
            "constitution_law"
        ),
        document_type=(
            "constitution"
        ),
        publication_date=None,
        section=None,
        subsection=None,
        article_number=None,
        article_title=None,
        extraction_method="native",
    )


def test_default_experiment_depths() -> None:
    """The measured translated-hybrid configuration should stay explicit."""

    assert (
        TRANSLATED_HYBRID_COMPONENT_DEPTH
        == 100
    )

    assert (
        DEFAULT_FINAL_CANDIDATE_DEPTH
        == 20
    )


def test_dataset_contains_expected_six_ne_en_records() -> None:
    """The controlled experiment must stay tied to the six NE->EN cases."""

    records = (
        load_evaluation_records(
            DEFAULT_DATASET_PATH
        )
    )

    selected = (
        select_ne_en_records(
            records
        )
    )

    assert tuple(
        record.question_id
        for record in selected
    ) == EXPECTED_NE_EN_IDS


def test_default_controlled_queries_match_ne_en_dataset() -> None:
    """Every production NE->EN record must have one controlled English query."""

    records = (
        load_evaluation_records(
            DEFAULT_DATASET_PATH
        )
    )

    selected = (
        select_ne_en_records(
            records
        )
    )

    validate_controlled_queries(
        selected
    )

    assert tuple(
        sorted(
            CONTROLLED_ENGLISH_QUERIES
        )
    ) == EXPECTED_NE_EN_IDS


def test_select_ne_en_records_filters_other_language_pairs() -> None:
    """Only Nepali-query to English-evidence records should be retained."""

    records = [
        make_record(
            question_id="ne_en_002",
        ),
        make_record(
            question_id="en_en_001",
            query_language="en",
            target_language="en",
        ),
        make_record(
            question_id="en_ne_001",
            query_language="en",
            target_language="ne",
        ),
        make_record(
            question_id="ne_en_001",
        ),
    ]

    selected = (
        select_ne_en_records(
            records
        )
    )

    assert [
        record.question_id
        for record in selected
    ] == [
        "ne_en_001",
        "ne_en_002",
    ]


def test_validate_controlled_queries_accepts_exact_mapping() -> None:
    """A complete one-to-one control mapping should validate."""

    records = [
        make_record(
            question_id="ne_en_001"
        ),
        make_record(
            question_id="ne_en_002"
        ),
    ]

    validate_controlled_queries(
        records,
        controlled_queries={
            "ne_en_001": (
                "English control one"
            ),
            "ne_en_002": (
                "English control two"
            ),
        },
    )


def test_validate_controlled_queries_rejects_missing_id() -> None:
    """Missing controlled counterparts must fail explicitly."""

    records = [
        make_record(
            question_id="ne_en_001"
        ),
        make_record(
            question_id="ne_en_002"
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Controlled English query IDs "
            "do not match"
        ),
    ):
        validate_controlled_queries(
            records,
            controlled_queries={
                "ne_en_001": (
                    "English control one"
                ),
            },
        )


def test_validate_controlled_queries_rejects_unexpected_id() -> None:
    """Controls that do not correspond to benchmark rows must fail."""

    records = [
        make_record(
            question_id="ne_en_001"
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Controlled English query IDs "
            "do not match"
        ),
    ):
        validate_controlled_queries(
            records,
            controlled_queries={
                "ne_en_001": (
                    "English control one"
                ),
                "ne_en_999": (
                    "Unexpected control"
                ),
            },
        )


def test_validate_controlled_queries_rejects_blank_query() -> None:
    """Controlled English queries must contain usable text."""

    records = [
        make_record(
            question_id="ne_en_001"
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "must contain non-whitespace text"
        ),
    ):
        validate_controlled_queries(
            records,
            controlled_queries={
                "ne_en_001": "   ",
            },
        )


def test_reciprocal_rank_fuse_rewards_overlap() -> None:
    """A point appearing highly in both lists should rise to the top."""

    first = [
        make_result(
            "a",
            score=0.9,
        ),
        make_result(
            "shared",
            score=0.8,
        ),
        make_result(
            "b",
            score=0.7,
        ),
    ]

    second = [
        make_result(
            "shared",
            score=0.95,
        ),
        make_result(
            "c",
            score=0.85,
        ),
        make_result(
            "d",
            score=0.75,
        ),
    ]

    fused = (
        reciprocal_rank_fuse(
            [
                first,
                second,
            ],
            top_k=5,
        )
    )

    assert (
        fused[
            0
        ].point_id
        == "shared"
    )

    assert (
        len(
            {
                item.point_id
                for item in fused
            }
        )
        == len(
            fused
        )
    )


def test_reciprocal_rank_fuse_uses_deterministic_tie_break() -> None:
    """Equal RRF scores should be ordered by stable point ID."""

    first = [
        make_result(
            "b"
        ),
    ]

    second = [
        make_result(
            "a"
        ),
    ]

    fused = (
        reciprocal_rank_fuse(
            [
                first,
                second,
            ],
            top_k=2,
            rrf_k=60,
        )
    )

    assert [
        item.point_id
        for item in fused
    ] == [
        "a",
        "b",
    ]


def test_reciprocal_rank_fuse_preserves_original_payload() -> None:
    """Fusion should alter rank score but retain source result metadata."""

    source = (
        make_result(
            "primary",
            score=0.91,
        )
    )

    fused = (
        reciprocal_rank_fuse(
            [
                [
                    source,
                ],
            ],
            top_k=1,
        )
    )

    assert (
        fused[
            0
        ].point_id
        == source.point_id
    )

    assert (
        fused[
            0
        ].chunk_id
        == source.chunk_id
    )

    assert (
        fused[
            0
        ].document_id
        == source.document_id
    )

    assert (
        fused[
            0
        ].chunk_text
        == source.chunk_text
    )

    assert (
        fused[
            0
        ].score
        == pytest.approx(
            1.0
            / 61.0
        )
    )


def test_reciprocal_rank_fuse_validates_arguments() -> None:
    """Invalid fusion configuration should fail before ranking."""

    with pytest.raises(
        ValueError,
        match=(
            "top_k must be greater than zero"
        ),
    ):
        reciprocal_rank_fuse(
            [],
            top_k=0,
        )

    with pytest.raises(
        ValueError,
        match=(
            "rrf_k must be zero or greater"
        ),
    ):
        reciprocal_rank_fuse(
            [],
            top_k=1,
            rrf_k=-1,
        )


def test_build_translated_hybrid_fuses_dense_and_sparse() -> None:
    """Translated hybrid should reward evidence found by both components."""

    dense = [
        make_result(
            "dense-only"
        ),
        make_result(
            "shared"
        ),
    ]

    sparse = [
        make_result(
            "shared"
        ),
        make_result(
            "sparse-only"
        ),
    ]

    result = (
        build_translated_hybrid(
            dense,
            sparse,
            component_depth=2,
            final_depth=2,
        )
    )

    assert (
        result[
            0
        ].point_id
        == "shared"
    )

    assert (
        len(
            result
        )
        == 2
    )


def test_build_translated_hybrid_respects_component_depth() -> None:
    """Candidates outside the configured component pool cannot enter fusion."""

    dense = [
        make_result(
            "dense-one"
        ),
        make_result(
            "outside"
        ),
    ]

    sparse = [
        make_result(
            "sparse-one"
        ),
        make_result(
            "outside"
        ),
    ]

    result = (
        build_translated_hybrid(
            dense,
            sparse,
            component_depth=1,
            final_depth=1,
        )
    )

    assert (
        result[
            0
        ].point_id
        in {
            "dense-one",
            "sparse-one",
        }
    )

    assert all(
        item.point_id
        != "outside"
        for item in result
    )


def test_build_translated_hybrid_validates_depths() -> None:
    """Translated-hybrid depth configuration should fail early when invalid."""

    with pytest.raises(
        ValueError,
        match=(
            "component_depth must be greater than zero"
        ),
    ):
        build_translated_hybrid(
            [],
            [],
            component_depth=0,
            final_depth=1,
        )

    with pytest.raises(
        ValueError,
        match=(
            "final_depth must be greater than zero"
        ),
    ):
        build_translated_hybrid(
            [],
            [],
            component_depth=1,
            final_depth=0,
        )

    with pytest.raises(
        ValueError,
        match=(
            "component_depth must be at least final_depth"
        ),
    ):
        build_translated_hybrid(
            [],
            [],
            component_depth=1,
            final_depth=2,
        )


def test_build_dual_route_fusion_rewards_cross_route_overlap() -> None:
    """Evidence retrieved by both routes should rise in final fusion."""

    production = [
        make_result(
            "production-only"
        ),
        make_result(
            "shared"
        ),
    ]

    translated = [
        make_result(
            "shared"
        ),
        make_result(
            "translated-only"
        ),
    ]

    result = (
        build_dual_route_fusion(
            production,
            translated,
            final_depth=2,
        )
    )

    assert (
        result[
            0
        ].point_id
        == "shared"
    )


def test_build_dual_route_fusion_rejects_invalid_depth() -> None:
    """Dual-route fusion requires a positive final candidate depth."""

    with pytest.raises(
        ValueError,
        match=(
            "final_depth must be greater than zero"
        ),
    ):
        build_dual_route_fusion(
            [],
            [],
            final_depth=0,
        )


def test_first_primary_rank_returns_first_primary_match() -> None:
    """Primary rank should use the strongest manually verified evidence set."""

    record = (
        make_record(
            primary_ids=(
                "primary-a",
                "primary-b",
            ),
            relevant_ids=(
                "primary-a",
                "primary-b",
                "supporting",
            ),
        )
    )

    results = [
        make_result(
            "irrelevant"
        ),
        make_result(
            "primary-b"
        ),
        make_result(
            "primary-a"
        ),
    ]

    assert (
        first_primary_rank(
            record,
            results,
        )
        == 2
    )


def test_first_primary_rank_returns_none_when_missing() -> None:
    """Missing primary evidence should be represented explicitly."""

    record = (
        make_record()
    )

    results = [
        make_result(
            "irrelevant"
        ),
        make_result(
            "supporting"
        ),
    ]

    assert (
        first_primary_rank(
            record,
            results,
        )
        is None
    )


def test_format_rank_handles_present_and_missing_values() -> None:
    """Rank rendering should make inspection-depth misses explicit."""

    assert (
        format_rank(
            7,
            depth=100,
        )
        == "7"
    )

    assert (
        format_rank(
            None,
            depth=100,
        )
        == ">100"
    )


def test_calculate_metrics_uses_existing_metric_contract() -> None:
    """Cached rankings should use the standard retrieval metrics."""

    record = (
        make_record()
    )

    rankings = {
        record.question_id: [
            make_result(
                "irrelevant"
            ),
            make_result(
                "primary"
            ),
            make_result(
                "supporting"
            ),
        ],
    }

    metrics = (
        calculate_metrics(
            [
                record,
            ],
            rankings,
            cutoff=3,
        )
    )

    assert (
        len(
            metrics
        )
        == 1
    )

    assert (
        metrics[
            0
        ].question_id
        == record.question_id
    )

    assert (
        metrics[
            0
        ].hit_rate
        == 1.0
    )

    assert (
        metrics[
            0
        ].reciprocal_rank
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            0
        ].recall
        == 1.0
    )