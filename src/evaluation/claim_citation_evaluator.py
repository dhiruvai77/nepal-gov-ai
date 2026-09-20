"""Deterministic claim-to-citation alignment for NepalGov AI.

This module decomposes generated RAG answers into textual claim units and
associates each unit with the evidence identifiers cited inside that unit.

The resulting metrics measure structural claim-level citation coverage only.
They do not determine whether a cited passage semantically entails a claim.
Semantic citation correctness and faithfulness require a separate evaluation
layer using the original evidence text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


EVIDENCE_ID_PATTERN = re.compile(
    r"\[(E\d+)\]"
)

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?।])\s+"
)

LIST_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[-+*]\s+|[0-9०-९]+[.)।:]\s+)"
)

LIST_MARKER_ONLY_PATTERN = re.compile(
    r"^\s*(?:[-+*]|[0-9०-९]+[.)।:]?)\s*$"
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+"
)

HORIZONTAL_RULE_PATTERN = re.compile(
    r"^[-*_]{3,}$"
)

FULL_BOLD_HEADING_PATTERN = re.compile(
    r"^(?:\*\*|__).+(?:\*\*|__)$"
)

CITATION_WITH_SEPARATOR_PATTERN = re.compile(
    r"\s*(?:,\s*)?\[(?:E\d+)\]"
)


@dataclass(frozen=True)
class ClaimCitationUnit:
    """One deterministic textual claim unit and its cited evidence IDs."""

    claim_index: int
    claim_text: str
    evidence_ids: tuple[str, ...]
    raw_text: str

    @property
    def has_citation(self) -> bool:
        """Return whether this claim unit cites at least one evidence ID."""

        return bool(
            self.evidence_ids
        )


@dataclass(frozen=True)
class ClaimCitationMetrics:
    """Structural claim-level citation metrics for one generated answer."""

    total_claim_count: int
    cited_claim_count: int
    uncited_claim_count: int
    claim_citation_coverage: float

    citation_assignment_count: int
    valid_citation_assignment_count: int
    invalid_citation_assignment_count: int
    valid_reference_ratio: float

    unique_cited_evidence_count: int
    avg_citations_per_cited_claim: float


@dataclass(frozen=True)
class ClaimCitationEvaluation:
    """Claim-level structural evaluation for one benchmark question."""

    question_id: str
    claims: tuple[
        ClaimCitationUnit,
        ...,
    ]
    metrics: ClaimCitationMetrics


def _deduplicate_preserving_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Return unique strings while preserving first-appearance order."""

    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(
            value
        )

        ordered.append(
            value
        )

    return tuple(
        ordered
    )


def _remove_list_prefix(
    text: str,
) -> str:
    """Remove one Markdown or multilingual numbered-list prefix."""

    return LIST_PREFIX_PATTERN.sub(
        "",
        text,
        count=1,
    ).strip()


def _is_nonclaim_line(
    line: str,
) -> bool:
    """Return whether a line is structural Markdown rather than answer prose."""

    stripped = (
        line.strip()
    )

    if not stripped:
        return True

    # Standalone list markers such as:
    #
    # 1.
    # २.
    # -
    #
    # are formatting artifacts rather than semantic claims.
    if LIST_MARKER_ONLY_PATTERN.fullmatch(
        stripped
    ):
        return True

    if HORIZONTAL_RULE_PATTERN.fullmatch(
        stripped
    ):
        return True

    without_list_prefix = (
        _remove_list_prefix(
            stripped
        )
    )

    if MARKDOWN_HEADING_PATTERN.match(
        without_list_prefix
    ):
        return True

    if FULL_BOLD_HEADING_PATTERN.fullmatch(
        without_list_prefix
    ):
        return True

    return False


def _extract_evidence_ids(
    text: str,
) -> tuple[str, ...]:
    """Extract unique evidence IDs from one textual claim unit."""

    return (
        _deduplicate_preserving_order(
            EVIDENCE_ID_PATTERN.findall(
                text
            )
        )
    )


def _clean_claim_text(
    text: str,
) -> str:
    """Remove citation labels and lightweight Markdown from claim text."""

    cleaned = (
        _remove_list_prefix(
            text
        )
    )

    cleaned = (
        CITATION_WITH_SEPARATOR_PATTERN.sub(
            "",
            cleaned,
        )
    )

    cleaned = (
        cleaned.replace(
            "**",
            "",
        )
        .replace(
            "__",
            "",
        )
        .replace(
            "`",
            "",
        )
        .replace(
            "*",
            "",
        )
    )

    cleaned = re.sub(
        r"\s+([,.;:!?।])",
        r"\1",
        cleaned,
    )

    cleaned = re.sub(
        r"\s{2,}",
        " ",
        cleaned,
    )

    cleaned = (
        cleaned.strip()
    )

    return cleaned


def extract_claim_citation_units(
    answer_text: str,
) -> tuple[
    ClaimCitationUnit,
    ...,
]:
    """Split a generated answer into deterministic claim/citation units.

    English sentence punctuation and the Nepali danda are supported.

    Pure Markdown headings and standalone list markers are ignored.
    Bullet and numbered-list prefixes are removed while their prose content is
    retained.

    Citation IDs are removed from `claim_text` but preserved structurally in
    `evidence_ids`.
    """

    if not isinstance(
        answer_text,
        str,
    ):
        raise TypeError(
            "answer_text must be a string."
        )

    claims: list[
        ClaimCitationUnit
    ] = []

    for line in (
        answer_text.splitlines()
    ):
        stripped_line = (
            line.strip()
        )

        if _is_nonclaim_line(
            stripped_line
        ):
            continue

        segments = (
            SENTENCE_BOUNDARY_PATTERN.split(
                stripped_line
            )
        )

        for segment in segments:
            raw_text = (
                _remove_list_prefix(
                    segment
                )
            )

            if not raw_text:
                continue

            # A standalone marker can also appear after sentence splitting.
            if LIST_MARKER_ONLY_PATTERN.fullmatch(
                raw_text
            ):
                continue

            claim_text = (
                _clean_claim_text(
                    raw_text
                )
            )

            if not claim_text:
                continue

            claims.append(
                ClaimCitationUnit(
                    claim_index=(
                        len(claims)
                        + 1
                    ),
                    claim_text=(
                        claim_text
                    ),
                    evidence_ids=(
                        _extract_evidence_ids(
                            raw_text
                        )
                    ),
                    raw_text=(
                        raw_text
                    ),
                )
            )

    return tuple(
        claims
    )


def evaluate_claim_citations(
    answer_text: str,
    *,
    available_evidence_ids: Iterable[
        str
    ],
) -> tuple[
    tuple[
        ClaimCitationUnit,
        ...,
    ],
    ClaimCitationMetrics,
]:
    """Evaluate structural claim-level citation coverage for one answer."""

    claims = (
        extract_claim_citation_units(
            answer_text
        )
    )

    available_ids = set(
        available_evidence_ids
    )

    cited_claim_count = sum(
        claim.has_citation
        for claim in claims
    )

    total_claim_count = len(
        claims
    )

    uncited_claim_count = (
        total_claim_count
        - cited_claim_count
    )

    if total_claim_count:
        claim_citation_coverage = (
            cited_claim_count
            / total_claim_count
        )

    else:
        claim_citation_coverage = 0.0

    assignments = [
        evidence_id
        for claim in claims
        for evidence_id
        in claim.evidence_ids
    ]

    citation_assignment_count = len(
        assignments
    )

    valid_citation_assignment_count = sum(
        evidence_id
        in available_ids
        for evidence_id in assignments
    )

    invalid_citation_assignment_count = (
        citation_assignment_count
        - valid_citation_assignment_count
    )

    if citation_assignment_count:
        valid_reference_ratio = (
            valid_citation_assignment_count
            / citation_assignment_count
        )

    else:
        valid_reference_ratio = 0.0

    unique_cited_evidence_count = len(
        set(
            assignments
        )
    )

    if cited_claim_count:
        avg_citations_per_cited_claim = (
            citation_assignment_count
            / cited_claim_count
        )

    else:
        avg_citations_per_cited_claim = 0.0

    metrics = ClaimCitationMetrics(
        total_claim_count=(
            total_claim_count
        ),
        cited_claim_count=(
            cited_claim_count
        ),
        uncited_claim_count=(
            uncited_claim_count
        ),
        claim_citation_coverage=(
            claim_citation_coverage
        ),
        citation_assignment_count=(
            citation_assignment_count
        ),
        valid_citation_assignment_count=(
            valid_citation_assignment_count
        ),
        invalid_citation_assignment_count=(
            invalid_citation_assignment_count
        ),
        valid_reference_ratio=(
            valid_reference_ratio
        ),
        unique_cited_evidence_count=(
            unique_cited_evidence_count
        ),
        avg_citations_per_cited_claim=(
            avg_citations_per_cited_claim
        ),
    )

    return (
        claims,
        metrics,
    )


def evaluate_persisted_rag_row(
    row: Mapping[
        str,
        Any,
    ],
) -> ClaimCitationEvaluation:
    """Evaluate one persisted production RAG benchmark row offline.

    The original generated answer is preferred when available so withheld
    generation can also be inspected later.

    Selected evidence IDs define the valid evidence namespace for the answer.
    """

    question_id = (
        row.get(
            "question_id"
        )
    )

    if (
        not isinstance(
            question_id,
            str,
        )
        or not question_id.strip()
    ):
        raise ValueError(
            "Persisted RAG row must contain "
            "a valid question_id."
        )

    generated_answer = (
        row.get(
            "generated_answer_text"
        )
    )

    if isinstance(
        generated_answer,
        str,
    ):
        answer_text = (
            generated_answer
        )

    else:
        answer_text = (
            row.get(
                "answer_text"
            )
        )

    if not isinstance(
        answer_text,
        str,
    ):
        raise ValueError(
            "Persisted RAG row must contain "
            "answer text."
        )

    selected_evidence = (
        row.get(
            "selected_evidence"
        )
    )

    if not isinstance(
        selected_evidence,
        list,
    ):
        raise ValueError(
            "Persisted RAG row must contain "
            "selected_evidence as a list."
        )

    available_ids: list[
        str
    ] = []

    for item in selected_evidence:
        if not isinstance(
            item,
            Mapping,
        ):
            raise ValueError(
                "selected_evidence entries "
                "must be mappings."
            )

        evidence_id = (
            item.get(
                "evidence_id"
            )
        )

        if (
            not isinstance(
                evidence_id,
                str,
            )
            or not evidence_id.strip()
        ):
            raise ValueError(
                "selected_evidence entry "
                "has no valid evidence_id."
            )

        available_ids.append(
            evidence_id
        )

    claims, metrics = (
        evaluate_claim_citations(
            answer_text,
            available_evidence_ids=(
                available_ids
            ),
        )
    )

    return ClaimCitationEvaluation(
        question_id=(
            question_id
        ),
        claims=claims,
        metrics=metrics,
    )