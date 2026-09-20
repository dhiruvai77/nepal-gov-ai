"""Deterministic insufficient-evidence handling for NepalGov AI.

This layer decides whether a generated answer is safe to present based on
structural evidence-grounding signals that are already available in the RAG
pipeline.

It intentionally does not introduce retrieval-score, reranker-score, or
confidence thresholds. Such thresholds require dedicated evaluation before
they can become production policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.citations.evidence import (
    CitationProcessingResult,
)


INSUFFICIENT_EVIDENCE_MESSAGES = {
    "en": (
        "The supplied government evidence is insufficient "
        "to provide a supported answer."
    ),
    "ne": (
        "उपलब्ध गराइएको सरकारी प्रमाणका आधारमा "
        "पर्याप्त रूपमा समर्थित उत्तर दिन सकिएन।"
    ),
}


class EvidenceGuardReason(
    str,
    Enum,
):
    """Reason a generated answer was withheld."""

    NO_SELECTED_EVIDENCE = (
        "no_selected_evidence"
    )

    MISSING_CITATIONS = (
        "missing_citations"
    )

    INVALID_CITATIONS = (
        "invalid_citations"
    )


@dataclass(frozen=True)
class EvidenceGuardResult:
    """Final deterministic evidence-gating decision.

    `answer_text` contains either the accepted generated answer or a
    deterministic insufficient-evidence message.

    `accepted` indicates whether the generated model answer may be presented.

    `reason` is populated only when the generated answer is withheld.

    `citation_result` retains citation diagnostics when generation occurred.
    It is `None` when no evidence was selected and generation should not occur.
    """

    answer_text: str
    accepted: bool
    reason: EvidenceGuardReason | None
    citation_result: (
        CitationProcessingResult
        | None
    ) = None

    @property
    def withheld(
        self,
    ) -> bool:
        """Return whether the generated answer was withheld."""

        return not self.accepted


def build_insufficient_evidence_message(
    answer_language: str,
) -> str:
    """Return a deterministic user-facing insufficient-evidence message."""

    if not isinstance(
        answer_language,
        str,
    ):
        raise TypeError(
            "answer_language must be a string."
        )

    clean_language = (
        answer_language
        .strip()
        .lower()
    )

    if not clean_language:
        raise ValueError(
            "answer_language must contain "
            "non-whitespace text."
        )

    # English is the deterministic fallback for language identifiers that do
    # not yet have an application-owned refusal message. This avoids asking the
    # generation model to invent a refusal when evidence is already known to be
    # unavailable.
    return (
        INSUFFICIENT_EVIDENCE_MESSAGES.get(
            clean_language,
            INSUFFICIENT_EVIDENCE_MESSAGES[
                "en"
            ],
        )
    )


def build_no_evidence_result(
    *,
    answer_language: str,
) -> EvidenceGuardResult:
    """Build a refusal result when no evidence was selected upstream.

    This function allows future orchestration to stop before calling an LLM
    when retrieval/reranking/context selection produced no usable evidence.
    """

    return EvidenceGuardResult(
        answer_text=(
            build_insufficient_evidence_message(
                answer_language
            )
        ),
        accepted=False,
        reason=(
            EvidenceGuardReason
            .NO_SELECTED_EVIDENCE
        ),
        citation_result=None,
    )


def guard_generated_answer(
    citation_result: CitationProcessingResult,
    *,
    answer_language: str,
) -> EvidenceGuardResult:
    """Decide whether a generated answer may be presented.

    The policy is intentionally structural:

    1. Any invented/invalid evidence identifier causes withholding.
    2. An answer with no valid citation causes withholding.
    3. Otherwise the original generated answer is preserved exactly.

    This does not claim that citation presence proves semantic faithfulness.
    Claim-level support remains an evaluation concern.
    """

    fallback = (
        build_insufficient_evidence_message(
            answer_language
        )
    )

    # Invalid evidence identifiers are checked first. A response containing
    # [E99], for example, must not be presented even if another valid citation
    # also appears.
    if (
        citation_result
        .invalid_evidence_ids
    ):
        return EvidenceGuardResult(
            answer_text=fallback,
            accepted=False,
            reason=(
                EvidenceGuardReason
                .INVALID_CITATIONS
            ),
            citation_result=(
                citation_result
            ),
        )

    if not citation_result.has_citations:
        return EvidenceGuardResult(
            answer_text=fallback,
            accepted=False,
            reason=(
                EvidenceGuardReason
                .MISSING_CITATIONS
            ),
            citation_result=(
                citation_result
            ),
        )

    return EvidenceGuardResult(
        # Preserve accepted provider output exactly. Citation parsing already
        # retains the original answer text without rewriting it.
        answer_text=(
            citation_result
            .answer_text
        ),
        accepted=True,
        reason=None,
        citation_result=(
            citation_result
        ),
    )