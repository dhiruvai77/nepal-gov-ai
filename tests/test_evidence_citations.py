"""Tests for generated-answer evidence citation processing."""

import pytest

from src.citations.evidence import (
    CitationProcessingResult,
    EvidenceCitation,
    build_evidence_map,
    extract_evidence_ids,
    format_evidence_citation,
    process_answer_citations,
    render_cited_sources,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_evidence(
    index: int,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> RerankedResult:
    """Create selected evidence with deterministic source metadata."""

    start = (
        page_start
        if page_start is not None
        else index
    )

    end = (
        page_end
        if page_end is not None
        else start
    )

    result = RetrievalResult(
        point_id=f"point-{index}",
        score=0.05,
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        title=f"Government Document {index}",
        organization="Government of Nepal",
        language="en",
        page_start=start,
        page_end=end,
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
        rerank_score=(
            1.0 - index / 100
        ),
        original_rank=index,
    )


def test_build_evidence_map_assigns_context_order_ids() -> None:
    """Evidence IDs should be determined only by selected-context order."""

    first = make_evidence(
        1
    )

    second = make_evidence(
        2
    )

    evidence_map = (
        build_evidence_map(
            [
                first,
                second,
            ]
        )
    )

    assert list(
        evidence_map
    ) == [
        "E1",
        "E2",
    ]

    assert (
        evidence_map[
            "E1"
        ]
        is first
    )

    assert (
        evidence_map[
            "E2"
        ]
        is second
    )


def test_build_evidence_map_rejects_empty_context() -> None:
    """Citation mapping requires the selected generation context."""

    with pytest.raises(
        ValueError,
        match=(
            "at least one selected evidence "
            "passage is required"
        ),
    ):
        build_evidence_map(
            []
        )


def test_extract_evidence_ids_preserves_first_appearance_order() -> None:
    """Citation extraction should follow generated-answer order."""

    evidence_ids = (
        extract_evidence_ids(
            (
                "First claim [E2]. "
                "Second claim [E1]."
            )
        )
    )

    assert evidence_ids == (
        "E2",
        "E1",
    )


def test_extract_evidence_ids_deduplicates_repeated_references() -> None:
    """Repeated references should map to one source-list entry."""

    evidence_ids = (
        extract_evidence_ids(
            (
                "Claim one [E1]. "
                "Claim two [E1]. "
                "Claim three [E2]."
            )
        )
    )

    assert evidence_ids == (
        "E1",
        "E2",
    )


def test_extract_evidence_ids_captures_invalid_numeric_ids() -> None:
    """Unsupported evidence numbers should be visible to validation."""

    evidence_ids = (
        extract_evidence_ids(
            (
                "Supported [E1], "
                "but also invalid [E99] "
                "and [E0]."
            )
        )
    )

    assert evidence_ids == (
        "E1",
        "E99",
        "E0",
    )


def test_extract_evidence_ids_returns_empty_when_none_are_present() -> None:
    """An uncited generated answer is valid parser input."""

    assert (
        extract_evidence_ids(
            "Answer without citations."
        )
        == ()
    )


def test_extract_evidence_ids_rejects_blank_answer() -> None:
    """Citation processing should not accept an empty generated answer."""

    with pytest.raises(
        ValueError,
        match=(
            "answer_text must contain "
            "non-whitespace text"
        ),
    ):
        extract_evidence_ids(
            "   "
        )


def test_extract_evidence_ids_rejects_non_string_answer() -> None:
    """Generated answer text must use the common string contract."""

    with pytest.raises(
        TypeError,
        match=(
            "answer_text must be a string"
        ),
    ):
        extract_evidence_ids(
            None
        )


def test_process_answer_citations_maps_valid_references() -> None:
    """Valid model references should map back to exact evidence objects."""

    first = make_evidence(
        1
    )

    second = make_evidence(
        2
    )

    processed = (
        process_answer_citations(
            (
                "First statement [E2]. "
                "Another statement [E1]."
            ),
            [
                first,
                second,
            ],
        )
    )

    assert isinstance(
        processed,
        CitationProcessingResult,
    )

    assert [
        citation.evidence_id
        for citation
        in processed.citations
    ] == [
        "E2",
        "E1",
    ]

    assert (
        processed.citations[
            0
        ].evidence
        is second
    )

    assert (
        processed.citations[
            1
        ].evidence
        is first
    )


def test_process_answer_citations_reports_invalid_references() -> None:
    """Invented evidence labels must never silently map to a source."""

    context = [
        make_evidence(
            1
        ),
        make_evidence(
            2
        ),
    ]

    processed = (
        process_answer_citations(
            (
                "Valid evidence [E1]. "
                "Invented evidence [E7]."
            ),
            context,
        )
    )

    assert [
        citation.evidence_id
        for citation
        in processed.citations
    ] == [
        "E1",
    ]

    assert (
        processed.invalid_evidence_ids
        == (
            "E7",
        )
    )

    assert (
        processed.all_references_valid
        is False
    )


def test_process_answer_citations_reports_valid_state() -> None:
    """All-valid generated references should expose a clean validation state."""

    processed = (
        process_answer_citations(
            "Supported answer [E1].",
            [
                make_evidence(
                    1
                ),
            ],
        )
    )

    assert (
        processed.has_citations
        is True
    )

    assert (
        processed.all_references_valid
        is True
    )


def test_process_answer_citations_handles_uncited_answer() -> None:
    """Citation presence and citation validity are separate concepts."""

    processed = (
        process_answer_citations(
            "Answer without evidence labels.",
            [
                make_evidence(
                    1
                ),
            ],
        )
    )

    assert (
        processed.has_citations
        is False
    )

    assert (
        processed.all_references_valid
        is True
    )

    assert (
        processed.citations
        == ()
    )


def test_process_answer_citations_preserves_answer_exactly() -> None:
    """Citation processing must not rewrite provider output."""

    answer = (
        "  First line [E1].\n"
        "Second line.  "
    )

    processed = (
        process_answer_citations(
            answer,
            [
                make_evidence(
                    1
                ),
            ],
        )
    )

    assert (
        processed.answer_text
        == answer
    )


def test_format_evidence_citation_formats_single_page() -> None:
    """User-facing source metadata should include a single page."""

    evidence = make_evidence(
        1,
        page_start=16,
        page_end=16,
    )

    citation = EvidenceCitation(
        evidence_id="E1",
        evidence=evidence,
    )

    rendered = (
        format_evidence_citation(
            citation
        )
    )

    assert rendered == (
        "[E1] Government Document 1 — "
        "Government of Nepal — "
        "p. 16 — "
        "https://example.gov.np/document-1.pdf"
    )


def test_format_evidence_citation_formats_page_range() -> None:
    """Multi-page evidence should preserve complete page provenance."""

    evidence = make_evidence(
        1,
        page_start=16,
        page_end=18,
    )

    citation = EvidenceCitation(
        evidence_id="E1",
        evidence=evidence,
    )

    rendered = (
        format_evidence_citation(
            citation
        )
    )

    assert (
        "pp. 16-18"
        in rendered
    )


def test_render_cited_sources_preserves_reference_order() -> None:
    """Source lists should follow first citation appearance in the answer."""

    first = make_evidence(
        1
    )

    second = make_evidence(
        2
    )

    processed = (
        process_answer_citations(
            (
                "Second source first [E2]. "
                "Then first source [E1]."
            ),
            [
                first,
                second,
            ],
        )
    )

    rendered = (
        render_cited_sources(
            processed
        )
    )

    assert len(
        rendered
    ) == 2

    assert rendered[
        0
    ].startswith(
        "[E2] Government Document 2"
    )

    assert rendered[
        1
    ].startswith(
        "[E1] Government Document 1"
    )