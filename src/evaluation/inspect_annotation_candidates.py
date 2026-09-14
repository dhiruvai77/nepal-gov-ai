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
    # -------------------------------------------------------------------------
    # English -> English
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Nepali -> Nepali
    # -------------------------------------------------------------------------
    AnnotationQuery(
        question_id="ne_ne_001",
        # Search for literacy indicators in the Nepali Economic Survey.
        query="साक्षरता दर नेपाल शिक्षा जनसंख्या",
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="ne_ne_002",
        # Search for school and student statistics in the Nepali survey.
        query="विद्यालय विद्यार्थी संख्या शिक्षा विद्यालय तह विद्यार्थी",
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="ne_ne_003",
        # Search for SEE examination results and pass statistics.
        query="एसईई परीक्षा नतिजा उत्तीर्ण विद्यार्थी परीक्षा परिणाम",
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="ne_ne_004",
        # Use wording aimed at national/sectoral GDP evidence rather than
        # loosely matching any occurrence of economic-growth terminology.
        query=(
            "नेपालको आर्थिक वृद्धिदर चालु आर्थिक वर्ष २०८१/८२ "
            "कुल गार्हस्थ्य उत्पादन वास्तविक आर्थिक वृद्धि अनुमान"
        ),
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="ne_ne_005",
        # Search for health-sector indicators and public-health information.
        query="स्वास्थ्य सेवा जनस्वास्थ्य स्वास्थ्य सूचक अस्पताल",
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="ne_ne_006",
        # Search for employment and labour-market indicators.
        query="रोजगारी बेरोजगारी श्रम रोजगार जनशक्ति",
        document_id="economic_survey_2081_82_ne",
    ),

    # -------------------------------------------------------------------------
    # English -> Nepali
    # -------------------------------------------------------------------------
    AnnotationQuery(
        question_id="en_ne_001",
        # Cross-lingual test: English query against Nepali literacy evidence.
        query="What does the Economic Survey report about literacy rates in Nepal?",
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="en_ne_002",
        # Cross-lingual test for school and student statistics.
        query=(
            "What does the Economic Survey report about schools "
            "and student numbers?"
        ),
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="en_ne_003",
        # Cross-lingual test for SEE examination results.
        query=(
            "What does the Economic Survey report about "
            "the SEE examination results?"
        ),
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="en_ne_004",
        # Cross-lingual test for sectoral output growth and GDP composition.
        query=(
            "What does the Economic Survey report about agricultural and "
            "non-agricultural growth and sector contributions to GDP?"
        ),
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="en_ne_005",
        # Cross-lingual test for health institutions and service indicators.
        query=(
            "What does the Economic Survey report about health services "
            "and health institutions in Nepal?"
        ),
        document_id="economic_survey_2081_82_ne",
    ),
    AnnotationQuery(
        question_id="en_ne_006",
        # Cross-lingual test for unemployment and employment programmes.
        query=(
            "What does the Economic Survey report about unemployment "
            "and employment programmes?"
        ),
        document_id="economic_survey_2081_82_ne",
    ),
]


def detect_target_language(question_id: str) -> str:
    """Infer the target corpus language from the evaluation question ID."""

    # Evaluation IDs follow:
    # query_language_target_language_NNN
    #
    # Examples:
    # en_en_001 -> English query, English target
    # ne_ne_001 -> Nepali query, Nepali target
    # en_ne_001 -> English query, Nepali target
    # ne_en_001 -> Nepali query, English target
    parts = question_id.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected question_id format: {question_id!r}. "
            "Expected a value such as 'en_en_001' or 'en_ne_001'."
        )

    target_language = parts[1]

    if target_language not in {"en", "ne"}:
        raise ValueError(
            f"Unsupported target language {target_language!r} "
            f"in question_id {question_id!r}."
        )

    return target_language


def main() -> None:
    """Print candidate passages for manual gold-label annotation."""

    # Government documents can contain Unicode ligatures and Nepali text that
    # Windows' default cp1252 console encoding cannot represent.
    sys.stdout.reconfigure(encoding="utf-8")

    for item in QUESTIONS:
        print("\n" + "=" * 90)
        print(f"{item.question_id}: {item.query}")
        print("=" * 90)

        # Derive the target document language from the question ID. This is
        # important for cross-lingual cases such as en_ne_*, where the query is
        # English but the evidence must come from Nepali documents.
        target_language = detect_target_language(item.question_id)

        results = run_hybrid_retrieval(
            item.query,
            # Annotation discovery deliberately searches deeper than the
            # production evaluation cutoff so we can manually inspect evidence
            # that may be relevant but poorly ranked.
            top_k=15,
            filters={
                "language": target_language,
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