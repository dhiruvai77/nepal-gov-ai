"""Tests for contextual dense-passage representation."""

import pytest

from src.embeddings.contextual_passage import (
    build_contextualized_passage_text,
)


def test_build_contextualized_passage_includes_semantic_metadata() -> None:
    """Useful document and structural metadata should prefix the source text."""

    chunk = {
        "title": "Constitution of Nepal",
        "organization": "Nepal Law Commission",
        "document_type": "constitution",
        "section": "Fundamental Rights",
        "subsection": None,
        "article_number": "35",
        "article_title": "Right relating to Health",
        "chunk_text": (
            "Every citizen shall have the right to free basic health services."
        ),
    }

    result = build_contextualized_passage_text(
        chunk
    )

    assert result == (
        "Document: Constitution of Nepal\n"
        "Organization: Nepal Law Commission\n"
        "Document type: constitution\n"
        "Section: Fundamental Rights\n"
        "Article number: 35\n"
        "Article title: Right relating to Health\n"
        "Content:\n"
        "Every citizen shall have the right to free basic health services."
    )


def test_build_contextualized_passage_skips_missing_metadata() -> None:
    """Missing optional metadata should not create noisy placeholder text."""

    result = build_contextualized_passage_text(
        {
            "title": "Economic Survey 2023/24",
            "organization": "",
            "document_type": None,
            "section": None,
            "subsection": None,
            "article_number": None,
            "article_title": None,
            "chunk_text": "Gross Domestic Product grew during the year.",
        }
    )

    assert result == (
        "Document: Economic Survey 2023/24\n"
        "Content:\n"
        "Gross Domestic Product grew during the year."
    )

    assert "None" not in result


def test_build_contextualized_passage_rejects_blank_text() -> None:
    """A contextual representation still requires real source content."""

    with pytest.raises(
        ValueError,
        match="chunk_text must contain non-whitespace text",
    ):
        build_contextualized_passage_text(
            {
                "title": "Constitution of Nepal",
                "chunk_text": "   ",
            }
        )