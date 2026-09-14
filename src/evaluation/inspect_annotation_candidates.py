from __future__ import annotations

import sys
from dataclasses import dataclass

from src.retrieval.run_hybrid_retrieval import run_hybrid_retrieval


@dataclass(frozen=True)
class AnnotationQuery:
    """One constrained search used to discover candidate gold evidence."""

    question_id: str
    query: str
    document_id: str


# These searches are intentionally constrained to the expected source document.
# The goal is to discover candidate evidence for manual annotation, not to
# measure production retrieval performance.
QUESTIONS = [
    AnnotationQuery(
        question_id="en_en_007",
        # Use terminology likely to appear in the constitutional provision so
        # the search targets the substantive right-to-health article directly.
        query=(
            "right relating to health free basic health services "
            "emergency health services equal access health service"
        ),
        document_id="constitution_nepal_current_en",
    ),
    AnnotationQuery(
        question_id="en_en_008",
        # Include wording around access to public information so retrieval is
        # less likely to return generic occurrences of the word "information".
        query=(
            "right to information citizen demand receive information "
            "public importance confidentiality law"
        ),
        document_id="constitution_nepal_current_en",
    ),
    AnnotationQuery(
        question_id="en_en_009",
        query=(
            "What does the Public Health Service Act say about "
            "emergency health services?"
        ),
        document_id="public_health_service_act_2075_en",
    ),
    AnnotationQuery(
        question_id="en_en_010",
        query=(
            "What does the Public Health Service Act say about "
            "informed consent for treatment?"
        ),
        document_id="public_health_service_act_2075_en",
    ),
    AnnotationQuery(
        question_id="en_en_011",
        # Search for substantive GDP and growth statements rather than the
        # cover-page chart that can dominate a broad economic-growth query.
        query=(
            "economic growth rate GDP gross domestic product Nepal economy "
            "2023/24 estimated growth"
        ),
        document_id="economic_survey_2023_24_en",
    ),
    AnnotationQuery(
        question_id="en_en_012",
        query=(
            "What does the 2025/26 Budget Speech say about "
            "scholarships for students?"
        ),
        document_id="budget_speech_2025_26_en",
    ),
]


def main() -> None:
    """Print candidate passages for manual gold-label annotation."""

    # Government documents can contain Unicode ligatures and Nepali text that
    # Windows' default cp1252 console encoding cannot represent.
    sys.stdout.reconfigure(encoding="utf-8")

    for item in QUESTIONS:
        print("\n" + "=" * 90)
        print(f"{item.question_id}: {item.query}")
        print("=" * 90)

        results = run_hybrid_retrieval(
            item.query,
            top_k=8,
            filters={
                "language": "en",
                "document_id": item.document_id,
            },
        )

        # Print both identifiers:
        # - point_id is the deterministic Qdrant UUID used by evaluation.
        # - chunk_id is the human-readable source chunk identifier.
        #
        # A larger text preview helps us judge substantive relevance instead
        # of treating retrieval score alone as evidence of correctness.
        for rank, result in enumerate(results, start=1):
            preview = result.chunk_text[:1200].replace("\n", " ")

            print(
                f"\nRank {rank}"
                f"\npoint_id: {result.point_id}"
                f"\nchunk_id: {result.chunk_id}"
                f"\npages: {result.page_start}-{result.page_end}"
                f"\nscore: {result.score:.6f}"
                f"\ntext: {preview}"
            )


if __name__ == "__main__":
    main()