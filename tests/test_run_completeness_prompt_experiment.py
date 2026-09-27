"""Tests for the generation-only answer-completeness prompt experiment."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.evaluation.run_completeness_prompt_experiment import (
    COMPLETENESS_RULES,
    VARIANT_BASELINE,
    VARIANT_COMPLETENESS,
    CompletenessPromptBuilder,
    build_context_from_review_row,
    build_output_record,
    build_request_from_review_row,
    build_variant_builder,
    prompt_sha256,
)
from src.generation.base import (
    GenerationRequest,
)
from src.generation.grounded_prompt import (
    GroundedPromptBuilder,
)


def make_evidence(
    *,
    evidence_id: str,
    point_id: str,
    chunk_text: str,
) -> dict:
    """Build one selected-evidence fixture."""

    return {
        "evidence_id": evidence_id,
        "point_id": point_id,
        "chunk_id": (
            f"chunk-{point_id}"
        ),
        "document_id": (
            "economic_survey_2023_24_en"
        ),
        "title": (
            "Economic Survey 2023/24"
        ),
        "organization": (
            "Ministry of Finance"
        ),
        "language": "en",
        "page_start": 100,
        "page_end": 101,
        "source_url": (
            "https://example.gov.np/survey.pdf"
        ),
        "chunk_text": (
            chunk_text
        ),
        "is_primary": True,
        "is_gold_relevant": True,
    }


def make_row() -> dict:
    """Build one completed answer-quality source row."""

    return {
        "review_config_id": (
            "production-rag-v2-answer-quality-v1"
        ),
        "review_status": "completed",
        "question_id": "en_en_005",
        "query": (
            "What does the Economic Survey say "
            "about SEE examination results?"
        ),
        "query_language": "en",
        "target_language": "en",
        "answer_language": "en",
        "category": "education",
        "selected_evidence": [
            make_evidence(
                evidence_id="E1",
                point_id="point-1",
                chunk_text=(
                    "Year Appeared Number Passed Number "
                    "Passed Percentage 175418 81008 46.18 "
                    "Annex 11.10: Details of students "
                    "Appeared and passed in Secondary "
                    "Education Examination."
                ),
            ),
            make_evidence(
                evidence_id="E2",
                point_id="point-2",
                chunk_text=(
                    "Additional unrelated context."
                ),
            ),
        ],
    }


def test_completeness_prompt_preserves_production_prompt() -> None:
    """Variant should extend rather than replace production grounding rules."""

    request = (
        build_request_from_review_row(
            make_row()
        )
    )

    baseline = (
        GroundedPromptBuilder()(
            request
        )
    )

    experimental = (
        CompletenessPromptBuilder()(
            request
        )
    )

    assert (
        "using only the evidence passages"
        in experimental
    )

    assert (
        COMPLETENESS_RULES
        in experimental
    )

    assert (
        request.query
        in experimental
    )

    assert (
        request.context[
            0
        ].result.chunk_text
        in experimental
    )

    assert (
        len(
            experimental
        )
        > len(
            baseline
        )
    )


def test_completeness_prompt_requires_substantive_table_results() -> None:
    """The intervention should explicitly target the measured table omission."""

    request = (
        build_request_from_review_row(
            make_row()
        )
    )

    prompt = (
        CompletenessPromptBuilder()(
            request
        )
    )

    assert (
        "report the substantive values or results"
        in prompt
    )

    assert (
        "instead of merely stating that the source "
        "contains such a table or annex"
        in prompt
    )


def test_completeness_prompt_preserves_fidelity_constraint() -> None:
    """Completeness instructions must explicitly avoid guessing table mappings."""

    request = (
        build_request_from_review_row(
            make_row()
        )
    )

    prompt = (
        CompletenessPromptBuilder()(
            request
        )
    )

    assert (
        "Do not infer a value-to-label mapping"
        in prompt
    )

    assert (
        "State the limitation instead of guessing"
        in prompt
    )


def test_build_context_preserves_evidence_order() -> None:
    """Persisted selected context order must remain E1, E2, ..."""

    context = (
        build_context_from_review_row(
            make_row()
        )
    )

    assert [
        item.result.point_id
        for item in context
    ] == [
        "point-1",
        "point-2",
    ]

    assert [
        item.original_rank
        for item in context
    ] == [
        1,
        2,
    ]


def test_build_context_preserves_canonical_passage_text() -> None:
    """The experiment must not rewrite selected source passages."""

    row = (
        make_row()
    )

    context = (
        build_context_from_review_row(
            row
        )
    )

    assert (
        context[
            0
        ].result.chunk_text
        == row[
            "selected_evidence"
        ][
            0
        ][
            "chunk_text"
        ]
    )


def test_build_context_rejects_wrong_evidence_id_order() -> None:
    """Evidence labels must remain tied to persisted context order."""

    row = (
        make_row()
    )

    row[
        "selected_evidence"
    ][
        0
    ][
        "evidence_id"
    ] = "E2"

    with pytest.raises(
        ValueError,
        match=(
            "deterministic evidence IDs"
        ),
    ):
        build_context_from_review_row(
            row
        )


def test_build_request_uses_answer_language() -> None:
    """Generation should preserve the production answer-language contract."""

    request = (
        build_request_from_review_row(
            make_row()
        )
    )

    assert isinstance(
        request,
        GenerationRequest,
    )

    assert (
        request.answer_language
        == "en"
    )


def test_variant_builders_are_distinct() -> None:
    """Baseline and intervention variants should use different prompt policies."""

    baseline = (
        build_variant_builder(
            VARIANT_BASELINE
        )
    )

    experimental = (
        build_variant_builder(
            VARIANT_COMPLETENESS
        )
    )

    assert isinstance(
        baseline,
        GroundedPromptBuilder,
    )

    assert isinstance(
        experimental,
        CompletenessPromptBuilder,
    )


def test_prompt_hash_is_deterministic() -> None:
    """Persisted prompt provenance should be stable."""

    assert (
        prompt_sha256(
            "same prompt"
        )
        == prompt_sha256(
            "same prompt"
        )
    )

    assert (
        prompt_sha256(
            "same prompt"
        )
        != prompt_sha256(
            "different prompt"
        )
    )


def test_output_record_accepts_valid_citations() -> None:
    """A grounded generated answer should pass the structural evidence guard."""

    row = (
        make_row()
    )

    request = (
        build_request_from_review_row(
            row
        )
    )

    prompt = (
        GroundedPromptBuilder()(
            request
        )
    )

    output = (
        build_output_record(
            row,
            variant=(
                VARIANT_BASELINE
            ),
            request=request,
            prompt=prompt,
            generated_answer_text=(
                "The source reports an examination "
                "result of 46.18 percent [E1]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    assert (
        output[
            "accepted"
        ]
        is True
    )

    assert (
        output[
            "withheld"
        ]
        is False
    )

    assert (
        output[
            "cited_evidence_ids"
        ]
        == [
            "E1",
        ]
    )

    assert (
        output[
            "invalid_evidence_ids"
        ]
        == []
    )


def test_output_record_withholds_invalid_citation() -> None:
    """The experiment must use the same structural guard as production."""

    row = (
        make_row()
    )

    request = (
        build_request_from_review_row(
            row
        )
    )

    prompt = (
        GroundedPromptBuilder()(
            request
        )
    )

    output = (
        build_output_record(
            row,
            variant=(
                VARIANT_BASELINE
            ),
            request=request,
            prompt=prompt,
            generated_answer_text=(
                "Unsupported evidence reference [E99]."
            ),
            provider="gemini",
            model="gemini-3.8-flash",
        )
    )

    assert (
        output[
            "accepted"
        ]
        is False
    )

    assert (
        output[
            "withheld"
        ]
        is True
    )

    assert (
        output[
            "invalid_evidence_ids"
        ]
        == [
            "E99",
        ]
    )