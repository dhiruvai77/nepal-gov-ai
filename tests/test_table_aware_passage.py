"""Tests for deterministic table-aware passage representations."""

from __future__ import annotations

import pytest

from src.embeddings.table_aware_passage import (
    build_table_aware_passage_text,
    extract_table_hints,
    is_table_like_passage,
    normalize_inline_text,
)


def test_normalize_inline_text_collapses_whitespace() -> None:
    """Whitespace should not make extracted headings unstable."""

    assert (
        normalize_inline_text(
            "Annex 1.1:\n"
            "Annual   Growth Rate"
        )
        == (
            "Annex 1.1: "
            "Annual Growth Rate"
        )
    )


def test_extract_gdp_annex_heading() -> None:
    """The known GDP annex heading should be extracted without table values."""

    text = (
        "Gross Domestic Product (GDP) "
        "3.98 0.43 8.98 7.62 6.66 "
        "* Provisional Source: National Statistics Office, 2024 "
        "Annex 1.1: Annual Growth Rate of GDP by Economic Activities "
        "- 1 -"
    )

    hints = (
        extract_table_hints(
            text
        )
    )

    assert hints == (
        (
            "Annex 1.1: "
            "Annual Growth Rate of GDP "
            "by Economic Activities"
        ),
    )


def test_extract_multiple_annexes_from_one_chunk() -> None:
    """A chunk containing multiple annexes should preserve each heading."""

    text = (
        "Annex 11.9: Details of Scholarship for School Children "
        "Activities S.N. Source: Example. "
        "Annex 11.10: Details of students Appeared and passed in "
        "Secondary Education Examination, Regular SEE "
        "(SLC examination) *Letter grading system started later."
    )

    hints = (
        extract_table_hints(
            text,
            max_hints=3,
        )
    )

    assert (
        len(
            hints
        )
        == 2
    )

    assert (
        hints[
            0
        ].startswith(
            "Annex 11.9:"
        )
    )

    assert (
        hints[
            1
        ].startswith(
            "Annex 11.10:"
        )
    )

    assert (
        "*Letter grading"
        not in hints[
            1
        ]
    )


def test_extract_annex_stops_before_academic_year_body() -> None:
    """A table title should not absorb the first body column headings."""

    text = (
        "Annex 11.14: Number of Basic, Lower Secondary and "
        "Secondary Schools and Students "
        "(Students number in thousand) "
        "Academic Year Basic level(1-5) Basic Level(6-8)"
    )

    hints = (
        extract_table_hints(
            text
        )
    )

    assert (
        len(
            hints
        )
        == 1
    )

    assert (
        "Academic Year"
        not in hints[
            0
        ]
    )

    assert (
        "Number of Basic"
        in hints[
            0
        ]
    )


def test_non_table_passage_has_no_hints() -> None:
    """Ordinary narrative prose must not be incorrectly labeled as a table."""

    text = (
        "The current fiscal year is expected "
        "to record growth in the agricultural sector."
    )

    assert (
        extract_table_hints(
            text
        )
        == ()
    )

    assert (
        is_table_like_passage(
            text
        )
        is False
    )


def test_table_passage_is_detected() -> None:
    """An annex marker should identify a table-like passage."""

    assert (
        is_table_like_passage(
            "Annex 1.1: Annual Growth Rate of GDP"
        )
        is True
    )


def test_build_table_aware_passage_promotes_heading() -> None:
    """Representation should preserve metadata, table heading, and raw text."""

    chunk = {
        "title": (
            "Economic Survey 2023/24"
        ),
        "organization": (
            "Ministry of Finance"
        ),
        "document_type": (
            "economic_survey"
        ),
        "section": None,
        "subsection": None,
        "article_number": None,
        "article_title": None,
        "chunk_text": (
            "GDP values. "
            "Annex 1.1: Annual Growth Rate "
            "of GDP by Economic Activities - 1 -"
        ),
    }

    result = (
        build_table_aware_passage_text(
            chunk
        )
    )

    assert (
        "Document: Economic Survey 2023/24"
        in result
    )

    assert (
        "Organization: Ministry of Finance"
        in result
    )

    assert (
        "Table: Annex 1.1: Annual Growth Rate "
        "of GDP by Economic Activities"
        in result
    )

    assert (
        "Content:\nGDP values."
        in result
    )

    assert (
        chunk[
            "chunk_text"
        ]
        in result
    )


def test_build_non_table_passage_still_preserves_contextual_shape() -> None:
    """Narrative chunks should remain valid contextual representations."""

    chunk = {
        "title": "Economic Survey 2023/24",
        "organization": "Ministry of Finance",
        "document_type": "economic_survey",
        "chunk_text": (
            "Economic activity increased "
            "during the fiscal year."
        ),
    }

    result = (
        build_table_aware_passage_text(
            chunk
        )
    )

    assert (
        "Table:"
        not in result
    )

    assert (
        "Document: Economic Survey 2023/24"
        in result
    )

    assert (
        result.endswith(
            chunk[
                "chunk_text"
            ]
        )
    )


def test_build_table_aware_passage_rejects_blank_text() -> None:
    """Blank passage content must fail before embedding."""

    with pytest.raises(
        ValueError,
        match=(
            "chunk_text must contain"
        ),
    ):
        build_table_aware_passage_text(
            {
                "chunk_text": "   ",
            }
        )


def test_extract_table_hints_validates_limits() -> None:
    """Extractor limits should fail explicitly when invalid."""

    with pytest.raises(
        ValueError,
        match=(
            "max_hints"
        ),
    ):
        extract_table_hints(
            "Annex 1.1: Test",
            max_hints=0,
        )

    with pytest.raises(
        ValueError,
        match=(
            "max_hint_chars"
        ),
    ):
        extract_table_hints(
            "Annex 1.1: Test",
            max_hint_chars=0,
        )