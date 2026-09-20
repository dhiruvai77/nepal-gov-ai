"""Tests for deterministic insufficient-evidence handling."""

import pytest

from src.citations.evidence import (
    process_answer_citations,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
    build_insufficient_evidence_message,
    build_no_evidence_result,
    guard_generated_answer,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_evidence(
    index: int = 1,
) -> RerankedResult:
    """Create one selected evidence passage."""

    result = RetrievalResult(
        point_id=f"point-{index}",
        score=0.1,
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        title=f"Government Document {index}",
        organization="Government of Nepal",
        language="en",
        page_start=index,
        page_end=index,
        source_url=(
            f"https://example.gov.np/"
            f"document-{index}.pdf"
        ),
        chunk_text=(
            f"Evidence passage {index}."
        ),
        chunk_index=index,
        token_count=200,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.95,
        original_rank=index,
    )


def test_build_insufficient_evidence_message_english() -> None:
    """English should have a deterministic application-owned message."""

    message = (
        build_insufficient_evidence_message(
            "en"
        )
    )

    assert message == (
        "The supplied government evidence is insufficient "
        "to provide a supported answer."
    )


def test_build_insufficient_evidence_message_nepali() -> None:
    """Nepali should have its own deterministic refusal message."""

    message = (
        build_insufficient_evidence_message(
            "ne"
        )
    )

    assert message == (
        "उपलब्ध गराइएको सरकारी प्रमाणका आधारमा "
        "पर्याप्त रूपमा समर्थित उत्तर दिन सकिएन।"
    )


def test_answer_language_is_normalized() -> None:
    """Language lookup should ignore surrounding whitespace and case."""

    assert (
        build_insufficient_evidence_message(
            " EN "
        )
        == build_insufficient_evidence_message(
            "en"
        )
    )


def test_unknown_language_uses_english_fallback() -> None:
    """Unsupported application messages should fall back deterministically."""

    assert (
        build_insufficient_evidence_message(
            "fr"
        )
        == build_insufficient_evidence_message(
            "en"
        )
    )


def test_blank_answer_language_is_rejected() -> None:
    """A refusal message still requires a valid language identifier."""

    with pytest.raises(
        ValueError,
        match=(
            "answer_language must contain "
            "non-whitespace text"
        ),
    ):
        build_insufficient_evidence_message(
            "   "
        )


def test_non_string_answer_language_is_rejected() -> None:
    """Language input should follow the generation string contract."""

    with pytest.raises(
        TypeError,
        match=(
            "answer_language must be a string"
        ),
    ):
        build_insufficient_evidence_message(
            None
        )


def test_no_selected_evidence_builds_withheld_result() -> None:
    """The future orchestrator should be able to stop before generation."""

    result = (
        build_no_evidence_result(
            answer_language="en",
        )
    )

    assert result.accepted is False

    assert result.withheld is True

    assert (
        result.reason
        == EvidenceGuardReason
        .NO_SELECTED_EVIDENCE
    )

    assert (
        result.citation_result
        is None
    )


def test_valid_cited_answer_is_accepted() -> None:
    """A generated answer with valid evidence references may proceed."""

    citation_result = (
        process_answer_citations(
            "Supported answer [E1].",
            [
                make_evidence(),
            ],
        )
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="en",
        )
    )

    assert result.accepted is True

    assert result.withheld is False

    assert result.reason is None

    assert (
        result.answer_text
        == "Supported answer [E1]."
    )


def test_accepted_answer_preserves_original_text() -> None:
    """Evidence gating must not rewrite accepted provider output."""

    answer = (
        "  First supported statement [E1].\n"
        "Second line.  "
    )

    citation_result = (
        process_answer_citations(
            answer,
            [
                make_evidence(),
            ],
        )
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="en",
        )
    )

    assert (
        result.answer_text
        == answer
    )


def test_accepted_result_preserves_citation_diagnostics() -> None:
    """Downstream orchestration should retain exact citation diagnostics."""

    citation_result = (
        process_answer_citations(
            "Supported answer [E1].",
            [
                make_evidence(),
            ],
        )
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="en",
        )
    )

    assert (
        result.citation_result
        is citation_result
    )


def test_uncited_generated_answer_is_withheld() -> None:
    """An answer that ignores the citation requirement should not pass."""

    citation_result = (
        process_answer_citations(
            "An answer without evidence references.",
            [
                make_evidence(),
            ],
        )
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="en",
        )
    )

    assert result.accepted is False

    assert result.withheld is True

    assert (
        result.reason
        == EvidenceGuardReason
        .MISSING_CITATIONS
    )

    assert result.answer_text == (
        "The supplied government evidence is insufficient "
        "to provide a supported answer."
    )


def test_invalid_evidence_reference_is_withheld() -> None:
    """Invented evidence identifiers should never reach the user."""

    citation_result = (
        process_answer_citations(
            (
                "One supported claim [E1]. "
                "One invented source [E9]."
            ),
            [
                make_evidence(),
            ],
        )
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="en",
        )
    )

    assert result.accepted is False

    assert (
        result.reason
        == EvidenceGuardReason
        .INVALID_CITATIONS
    )

    assert (
        citation_result
        .invalid_evidence_ids
        == (
            "E9",
        )
    )


def test_invalid_reference_reason_takes_precedence() -> None:
    """Invalid labels should be reported even when no valid citation exists."""

    citation_result = (
        process_answer_citations(
            "Invented source only [E99].",
            [
                make_evidence(),
            ],
        )
    )

    assert (
        citation_result.has_citations
        is False
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="en",
        )
    )

    assert (
        result.reason
        == EvidenceGuardReason
        .INVALID_CITATIONS
    )


def test_withheld_answer_uses_requested_nepali_message() -> None:
    """The deterministic refusal should respect the requested V1 language."""

    citation_result = (
        process_answer_citations(
            "उत्तरमा प्रमाण सन्दर्भ छैन।",
            [
                make_evidence(),
            ],
        )
    )

    result = (
        guard_generated_answer(
            citation_result,
            answer_language="ne",
        )
    )

    assert result.answer_text == (
        "उपलब्ध गराइएको सरकारी प्रमाणका आधारमा "
        "पर्याप्त रूपमा समर्थित उत्तर दिन सकिएन।"
    )