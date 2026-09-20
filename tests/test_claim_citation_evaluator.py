"""Tests for deterministic claim-to-citation alignment."""

import pytest

from src.evaluation.claim_citation_evaluator import (
    evaluate_claim_citations,
    evaluate_persisted_rag_row,
    extract_claim_citation_units,
)


def test_extracts_single_claim_and_citation() -> None:
    claims = extract_claim_citation_units(
        "Every citizen has the right to education [E1]."
    )

    assert len(claims) == 1
    assert (
        claims[0].claim_text
        == "Every citizen has the right to education."
    )
    assert claims[0].evidence_ids == ("E1",)
    assert claims[0].claim_index == 1
    assert claims[0].has_citation is True


def test_splits_multiple_english_sentences() -> None:
    claims = extract_claim_citation_units(
        "Basic education is free [E1]. "
        "Secondary education is also protected [E2]."
    )

    assert len(claims) == 2
    assert claims[0].evidence_ids == ("E1",)
    assert claims[1].evidence_ids == ("E2",)


def test_splits_nepali_danda_sentences() -> None:
    claims = extract_claim_citation_units(
        "प्रत्येक नागरिकलाई शिक्षाको हक हुनेछ [E1]। "
        "आधारभूत शिक्षा निःशुल्क हुनेछ [E2]।"
    )

    assert len(claims) == 2
    assert claims[0].evidence_ids == ("E1",)
    assert claims[1].evidence_ids == ("E2",)


def test_multiple_citations_preserve_order_and_deduplicate() -> None:
    claims = extract_claim_citation_units(
        "The rule applies [E2], [E1], [E2]."
    )

    assert claims[0].evidence_ids == (
        "E2",
        "E1",
    )

    assert (
        claims[0].claim_text
        == "The rule applies."
    )


def test_tracks_uncited_claim_units() -> None:
    claims = extract_claim_citation_units(
        "The Act applies nationally. "
        "Basic education is compulsory [E1]."
    )

    assert len(claims) == 2
    assert claims[0].evidence_ids == ()
    assert claims[0].has_citation is False
    assert claims[1].evidence_ids == ("E1",)


def test_ignores_markdown_headings() -> None:
    claims = extract_claim_citation_units(
        "**Education Rights**\n"
        "Every citizen has access to education [E1]."
    )

    assert len(claims) == 1
    assert (
        claims[0].claim_text
        == "Every citizen has access to education."
    )


def test_ignores_hash_markdown_heading() -> None:
    claims = extract_claim_citation_units(
        "## Education Rights\n"
        "Education is protected [E1]."
    )

    assert len(claims) == 1


def test_cleans_markdown_list_markup() -> None:
    claims = extract_claim_citation_units(
        "* **State Liability:** "
        "The State must provide education [E1]."
    )

    assert len(claims) == 1

    assert (
        claims[0].claim_text
        == "State Liability: The State must provide education."
    )


def test_keeps_uncited_introductory_prose() -> None:
    claims = extract_claim_citation_units(
        "According to the Act, the following rights apply:\n"
        "- Education is free [E1]."
    )

    assert len(claims) == 2

    assert (
        claims[0].claim_text
        == "According to the Act, the following rights apply:"
    )

    assert claims[0].evidence_ids == ()


def test_empty_answer_yields_no_claims() -> None:
    claims = extract_claim_citation_units(
        "   \n\n"
    )

    assert claims == ()


def test_mixed_claim_metrics() -> None:
    claims, metrics = evaluate_claim_citations(
        "Introductory statement. "
        "First supported claim [E1]. "
        "Second supported claim [E2].",
        available_evidence_ids={
            "E1",
            "E2",
        },
    )

    assert len(claims) == 3
    assert metrics.total_claim_count == 3
    assert metrics.cited_claim_count == 2
    assert metrics.uncited_claim_count == 1

    assert metrics.claim_citation_coverage == pytest.approx(
        2 / 3
    )

    assert metrics.citation_assignment_count == 2
    assert metrics.valid_citation_assignment_count == 2
    assert metrics.invalid_citation_assignment_count == 0
    assert metrics.valid_reference_ratio == 1.0
    assert metrics.unique_cited_evidence_count == 2

    assert metrics.avg_citations_per_cited_claim == 1.0


def test_invalid_citation_assignments_are_counted() -> None:
    _, metrics = evaluate_claim_citations(
        "Supported [E1]. Invented [E99].",
        available_evidence_ids={
            "E1",
            "E2",
        },
    )

    assert metrics.citation_assignment_count == 2
    assert metrics.valid_citation_assignment_count == 1
    assert metrics.invalid_citation_assignment_count == 1
    assert metrics.valid_reference_ratio == 0.5


def test_no_citations_produce_zero_reference_metrics() -> None:
    _, metrics = evaluate_claim_citations(
        "This answer contains no citation.",
        available_evidence_ids={
            "E1",
        },
    )

    assert metrics.total_claim_count == 1
    assert metrics.cited_claim_count == 0
    assert metrics.claim_citation_coverage == 0.0
    assert metrics.valid_reference_ratio == 0.0
    assert metrics.avg_citations_per_cited_claim == 0.0


def test_persisted_row_uses_selected_evidence_namespace() -> None:
    evaluation = evaluate_persisted_rag_row(
        {
            "question_id": "en_en_001",
            "generated_answer_text": (
                "Education is protected [E1]. "
                "Another claim cites an unknown passage [E9]."
            ),
            "answer_text": "unused",
            "selected_evidence": [
                {
                    "evidence_id": "E1",
                },
                {
                    "evidence_id": "E2",
                },
            ],
        }
    )

    assert evaluation.question_id == "en_en_001"
    assert len(evaluation.claims) == 2

    assert (
        evaluation.metrics.invalid_citation_assignment_count
        == 1
    )


def test_persisted_row_prefers_generated_answer_text() -> None:
    evaluation = evaluate_persisted_rag_row(
        {
            "question_id": "q1",
            "generated_answer_text": (
                "Original generated claim [E1]."
            ),
            "answer_text": (
                "Application replacement text."
            ),
            "selected_evidence": [
                {
                    "evidence_id": "E1",
                },
            ],
        }
    )

    assert len(evaluation.claims) == 1

    assert (
        evaluation.claims[0].claim_text
        == "Original generated claim."
    )


def test_persisted_row_rejects_missing_question_id() -> None:
    with pytest.raises(
        ValueError,
        match="valid question_id",
    ):
        evaluate_persisted_rag_row(
            {
                "generated_answer_text": "Claim [E1].",
                "selected_evidence": [
                    {
                        "evidence_id": "E1",
                    }
                ],
            }
        )


def test_persisted_row_rejects_invalid_selected_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="selected_evidence entries",
    ):
        evaluate_persisted_rag_row(
            {
                "question_id": "q1",
                "generated_answer_text": "Claim [E1].",
                "selected_evidence": [
                    "E1",
                ],
            }
        )