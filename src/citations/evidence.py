"""Evidence citation parsing and source mapping for NepalGov AI.

Generated answers may refer to selected context passages through deterministic
evidence identifiers such as [E1], [E2], and [E3].

This module validates those references against the actual selected
RerankedResult objects. It does not ask the language model to invent source
metadata and does not alter generated answer text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.reranking.base import (
    RerankedResult,
)


# Capture syntactically recognizable evidence references. References such as
# E0 or E99 are still captured so they can be reported as invalid rather than
# silently ignored.
EVIDENCE_REFERENCE_PATTERN = re.compile(
    r"\[(E\d+)\]"
)


@dataclass(frozen=True)
class EvidenceCitation:
    """One validated generated-answer reference to selected evidence."""

    evidence_id: str
    evidence: RerankedResult


@dataclass(frozen=True)
class CitationProcessingResult:
    """Structured citation information extracted from one generated answer.

    `answer_text` is preserved exactly.

    `citations` contains valid evidence references in first-appearance order.

    `invalid_evidence_ids` contains labels used by the model that do not map to
    the selected context.
    """

    answer_text: str
    citations: tuple[
        EvidenceCitation,
        ...
    ]
    invalid_evidence_ids: tuple[
        str,
        ...
    ]

    @property
    def has_citations(
        self,
    ) -> bool:
        """Return whether the answer contains at least one valid citation."""

        return bool(
            self.citations
        )

    @property
    def all_references_valid(
        self,
    ) -> bool:
        """Return whether every recognized evidence reference was valid."""

        return not self.invalid_evidence_ids


def build_evidence_map(
    context: Sequence[
        RerankedResult
    ],
) -> dict[
    str,
    RerankedResult,
]:
    """Map deterministic evidence IDs onto selected context passages."""

    if not context:
        raise ValueError(
            "at least one selected evidence "
            "passage is required."
        )

    return {
        f"E{index}": evidence
        for index, evidence in enumerate(
            context,
            start=1,
        )
    }


def extract_evidence_ids(
    answer_text: str,
) -> tuple[
    str,
    ...
]:
    """Extract unique evidence references in first-appearance order."""

    if not isinstance(
        answer_text,
        str,
    ):
        raise TypeError(
            "answer_text must be a string."
        )

    if not answer_text.strip():
        raise ValueError(
            "answer_text must contain "
            "non-whitespace text."
        )

    seen: set[str] = set()
    ordered_ids: list[str] = []

    for match in (
        EVIDENCE_REFERENCE_PATTERN.finditer(
            answer_text
        )
    ):
        evidence_id = (
            match.group(
                1
            )
        )

        if evidence_id in seen:
            continue

        seen.add(
            evidence_id
        )

        ordered_ids.append(
            evidence_id
        )

    return tuple(
        ordered_ids
    )


def process_answer_citations(
    answer_text: str,
    context: Sequence[
        RerankedResult
    ],
) -> CitationProcessingResult:
    """Validate generated evidence references against selected context."""

    evidence_map = (
        build_evidence_map(
            context
        )
    )

    referenced_ids = (
        extract_evidence_ids(
            answer_text
        )
    )

    citations: list[
        EvidenceCitation
    ] = []

    invalid_ids: list[
        str
    ] = []

    for evidence_id in referenced_ids:
        evidence = evidence_map.get(
            evidence_id
        )

        if evidence is None:
            invalid_ids.append(
                evidence_id
            )
            continue

        citations.append(
            EvidenceCitation(
                evidence_id=evidence_id,
                evidence=evidence,
            )
        )

    return CitationProcessingResult(
        # Do not strip or otherwise rewrite generated answer text here.
        answer_text=answer_text,
        citations=tuple(
            citations
        ),
        invalid_evidence_ids=tuple(
            invalid_ids
        ),
    )


def format_evidence_citation(
    citation: EvidenceCitation,
) -> str:
    """Render one validated citation using canonical source metadata."""

    result = (
        citation.evidence.result
    )

    if (
        result.page_start
        == result.page_end
    ):
        page_text = (
            f"p. {result.page_start}"
        )

    else:
        page_text = (
            f"pp. {result.page_start}"
            f"-{result.page_end}"
        )

    return (
        f"[{citation.evidence_id}] "
        f"{result.title} — "
        f"{result.organization} — "
        f"{page_text} — "
        f"{result.source_url}"
    )


def render_cited_sources(
    result: CitationProcessingResult,
) -> tuple[
    str,
    ...
]:
    """Render validated sources in generated-answer citation order."""

    return tuple(
        format_evidence_citation(
            citation
        )
        for citation in result.citations
    )