"""Tests for automated semantic citation judging."""

import json

import pytest

from src.evaluation.semantic_judge import (
    aggregate_judge_agreement,
    aggregate_judge_agreement_by_language_pair,
    build_semantic_judge_prompt,
    compare_prediction_to_human,
    parse_semantic_judge_response,
    prediction_to_dict,
)


PRODUCTION_REFERENCE_CONFIG_ID = (
    "production-rag-v2-human-review-v1"
)

HARD_CASE_CONFIG_ID = (
    "semantic-judge-hard-cases-v1"
)


def _completed_row(
    *,
    claim_id: str = "q1_c001",
    cited: bool = True,
    config_id: str = (
        PRODUCTION_REFERENCE_CONFIG_ID
    ),
    query_language: str = "en",
    target_language: str = "en",
    semantic_support_label: str | None = None,
    citation_requirement_label: str | None = None,
    individual_support_label: str = (
        "supported"
    ),
) -> dict:
    """Build one completed semantic-reference fixture."""

    if (
        semantic_support_label
        is None
    ):
        semantic_support_label = (
            "supported"
            if cited
            else "not_a_factual_claim"
        )

    if (
        citation_requirement_label
        is None
    ):
        citation_requirement_label = (
            "required"
            if cited
            else "not_required"
        )

    cited_evidence = (
        [
            {
                "evidence_id": "E1",
                "chunk_text": (
                    "The law directly states "
                    "the relevant provision."
                ),
                "title": "Example Act",
                "page_start": 1,
                "page_end": 1,
                "individual_support_label": (
                    individual_support_label
                ),
                "individual_support_notes": None,
            }
        ]
        if cited
        else []
    )

    return {
        "schema_version": 1,
        "source_run_config_id": (
            "production-rag-v2-interactions"
        ),
        "source_schema_version": 1,
        "question_id": "q1",
        "query": "What does the law say?",
        "query_language": (
            query_language
        ),
        "target_language": (
            target_language
        ),
        "answer_language": (
            query_language
        ),
        "category": "law",
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "claim_id": (
            claim_id
        ),
        "claim_index": 1,
        "claim_text": (
            "The law states the provision."
        ),
        "raw_text": (
            "The law states the provision [E1]."
            if cited
            else (
                "Introductory information "
                "follows:"
            )
        ),
        "has_citation": (
            cited
        ),
        "evidence_ids": (
            [
                "E1",
            ]
            if cited
            else []
        ),
        "cited_evidence": (
            cited_evidence
        ),
        "semantic_support_label": (
            semantic_support_label
        ),
        "citation_requirement_label": (
            citation_requirement_label
        ),
        "semantic_notes": None,
        "review_sample_config_id": (
            config_id
        ),
        "review_language_pair": (
            f"{query_language}"
            f"->{target_language}"
        ),
        "review_stratum": (
            "single_other"
            if cited
            else "uncited_other"
        ),
        "review_status": "completed",
    }


def _response(
    *,
    semantic_support_label: str = (
        "supported"
    ),
    citation_requirement_label: str = (
        "required"
    ),
    cited: bool = True,
    individual_support_label: str = (
        "supported"
    ),
) -> str:
    """Build one valid automated-judge JSON response."""

    return (
        json.dumps(
            {
                "semantic_support_label": (
                    semantic_support_label
                ),
                "citation_requirement_label": (
                    citation_requirement_label
                ),
                "semantic_notes": (
                    "Diagnostic explanation."
                ),
                "individual_evidence": (
                    [
                        {
                            "evidence_id": "E1",
                            "support_label": (
                                individual_support_label
                            ),
                            "notes": (
                                "Evidence judgment."
                            ),
                        }
                    ]
                    if cited
                    else []
                ),
            }
        )
    )


def test_build_prompt_contains_claim_and_evidence() -> None:
    row = (
        _completed_row()
    )

    prompt = (
        build_semantic_judge_prompt(
            row
        )
    )

    assert (
        "The law states the provision."
        in prompt
    )

    assert (
        "The law directly states "
        "the relevant provision."
        in prompt
    )

    assert (
        '["E1"]'
        in prompt
    )


def test_build_prompt_for_uncited_claim() -> None:
    row = (
        _completed_row(
            cited=False
        )
    )

    prompt = (
        build_semantic_judge_prompt(
            row
        )
    )

    assert (
        "NO CITED EVIDENCE"
        in prompt
    )

    assert (
        "[]"
        in prompt
    )


def test_build_prompt_accepts_hard_case_reference_config() -> None:
    row = (
        _completed_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            )
        )
    )

    prompt = (
        build_semantic_judge_prompt(
            row
        )
    )

    assert (
        "The law states the provision."
        in prompt
    )


def test_parse_valid_prediction() -> None:
    row = (
        _completed_row()
    )

    prediction = (
        parse_semantic_judge_response(
            _response(),
            row=row,
        )
    )

    assert (
        prediction.claim_id
        == "q1_c001"
    )

    assert (
        prediction.semantic_support_label
        == "supported"
    )

    assert (
        prediction.individual_evidence[
            0
        ].evidence_id
        == "E1"
    )


def test_parse_unsupported_hard_case_prediction() -> None:
    row = (
        _completed_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            ),
            semantic_support_label=(
                "unsupported"
            ),
            individual_support_label=(
                "unsupported"
            ),
        )
    )

    prediction = (
        parse_semantic_judge_response(
            _response(
                semantic_support_label=(
                    "unsupported"
                ),
                individual_support_label=(
                    "unsupported"
                ),
            ),
            row=row,
        )
    )

    assert (
        prediction.semantic_support_label
        == "unsupported"
    )

    assert (
        prediction.individual_evidence[
            0
        ].support_label
        == "unsupported"
    )


def test_parse_needs_review_hard_case_prediction() -> None:
    row = (
        _completed_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            ),
            semantic_support_label=(
                "needs_review"
            ),
            individual_support_label=(
                "needs_review"
            ),
        )
    )

    prediction = (
        parse_semantic_judge_response(
            _response(
                semantic_support_label=(
                    "needs_review"
                ),
                individual_support_label=(
                    "needs_review"
                ),
            ),
            row=row,
        )
    )

    assert (
        prediction.semantic_support_label
        == "needs_review"
    )


def test_parse_unclear_citation_requirement() -> None:
    row = (
        _completed_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            ),
            citation_requirement_label=(
                "unclear"
            ),
        )
    )

    prediction = (
        parse_semantic_judge_response(
            _response(
                citation_requirement_label=(
                    "unclear"
                ),
            ),
            row=row,
        )
    )

    assert (
        prediction.citation_requirement_label
        == "unclear"
    )


def test_parse_rejects_non_json() -> None:
    row = (
        _completed_row()
    )

    with pytest.raises(
        ValueError,
        match="valid JSON",
    ):
        parse_semantic_judge_response(
            "supported",
            row=row,
        )


def test_parse_rejects_extra_fields() -> None:
    row = (
        _completed_row()
    )

    response = (
        json.dumps(
            {
                "semantic_support_label": (
                    "supported"
                ),
                "citation_requirement_label": (
                    "required"
                ),
                "semantic_notes": None,
                "individual_evidence": [
                    {
                        "evidence_id": "E1",
                        "support_label": (
                            "supported"
                        ),
                        "notes": None,
                    }
                ],
                "unexpected": True,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="unexpected JSON fields",
    ):
        parse_semantic_judge_response(
            response,
            row=row,
        )


def test_parse_rejects_wrong_evidence_order() -> None:
    row = (
        _completed_row()
    )

    response = (
        json.dumps(
            {
                "semantic_support_label": (
                    "supported"
                ),
                "citation_requirement_label": (
                    "required"
                ),
                "semantic_notes": None,
                "individual_evidence": [
                    {
                        "evidence_id": "E2",
                        "support_label": (
                            "supported"
                        ),
                        "notes": None,
                    }
                ],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="expected evidence",
    ):
        parse_semantic_judge_response(
            response,
            row=row,
        )


def test_uncited_claim_cannot_be_supported() -> None:
    row = (
        _completed_row(
            cited=False
        )
    )

    response = (
        json.dumps(
            {
                "semantic_support_label": (
                    "supported"
                ),
                "citation_requirement_label": (
                    "not_required"
                ),
                "semantic_notes": None,
                "individual_evidence": [],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="Uncited claims",
    ):
        parse_semantic_judge_response(
            response,
            row=row,
        )


def test_prediction_to_dict() -> None:
    row = (
        _completed_row()
    )

    prediction = (
        parse_semantic_judge_response(
            _response(),
            row=row,
        )
    )

    serialized = (
        prediction_to_dict(
            prediction
        )
    )

    assert (
        serialized[
            "judge_schema_version"
        ]
        == 1
    )

    assert (
        serialized[
            "claim_id"
        ]
        == "q1_c001"
    )


def test_compare_exact_prediction_to_human() -> None:
    row = (
        _completed_row()
    )

    prediction = (
        parse_semantic_judge_response(
            _response(),
            row=row,
        )
    )

    agreement = (
        compare_prediction_to_human(
            row,
            prediction,
        )
    )

    assert (
        agreement.semantic_exact
        is True
    )

    assert (
        agreement.citation_requirement_exact
        is True
    )

    assert (
        agreement.individual_exact_count
        == 1
    )


def test_compare_exact_hard_negative_prediction() -> None:
    row = (
        _completed_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            ),
            semantic_support_label=(
                "unsupported"
            ),
            individual_support_label=(
                "unsupported"
            ),
        )
    )

    prediction = (
        parse_semantic_judge_response(
            _response(
                semantic_support_label=(
                    "unsupported"
                ),
                individual_support_label=(
                    "unsupported"
                ),
            ),
            row=row,
        )
    )

    agreement = (
        compare_prediction_to_human(
            row,
            prediction,
        )
    )

    assert (
        agreement.semantic_exact
        is True
    )

    assert (
        agreement.individual_exact_count
        == 1
    )


def test_compare_detects_disagreement() -> None:
    row = (
        _completed_row()
    )

    prediction = (
        parse_semantic_judge_response(
            _response(
                semantic_support_label=(
                    "partially_supported"
                ),
                individual_support_label=(
                    "partially_supported"
                ),
            ),
            row=row,
        )
    )

    agreement = (
        compare_prediction_to_human(
            row,
            prediction,
        )
    )

    assert (
        agreement.semantic_exact
        is False
    )

    assert (
        agreement.individual_exact_count
        == 0
    )


def test_aggregate_agreement_metrics() -> None:
    first_row = (
        _completed_row(
            claim_id="q1_c001"
        )
    )

    second_row = (
        _completed_row(
            claim_id="q2_c001"
        )
    )

    first_prediction = (
        parse_semantic_judge_response(
            _response(),
            row=first_row,
        )
    )

    second_prediction = (
        parse_semantic_judge_response(
            _response(
                semantic_support_label=(
                    "partially_supported"
                ),
                individual_support_label=(
                    "partially_supported"
                ),
            ),
            row=second_row,
        )
    )

    agreements = [
        compare_prediction_to_human(
            first_row,
            first_prediction,
        ),
        compare_prediction_to_human(
            second_row,
            second_prediction,
        ),
    ]

    metrics = (
        aggregate_judge_agreement(
            agreements
        )
    )

    assert (
        metrics[
            "semantic_exact_accuracy"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "citation_requirement_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "individual_exact_accuracy"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_aggregate_by_language_pair() -> None:
    row = (
        _completed_row()
    )

    prediction = (
        parse_semantic_judge_response(
            _response(),
            row=row,
        )
    )

    agreement = (
        compare_prediction_to_human(
            row,
            prediction,
        )
    )

    summary = (
        aggregate_judge_agreement_by_language_pair(
            [
                agreement,
            ]
        )
    )

    assert (
        summary[
            "en->en"
        ][
            "semantic_exact_accuracy"
        ]
        == 1.0
    )


def test_aggregate_requires_predictions() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        aggregate_judge_agreement(
            []
        )