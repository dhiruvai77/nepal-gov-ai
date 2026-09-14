from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.qdrant_setup import QDRANT_URL
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
)


# Search deeply enough to distinguish a top-ranking problem from a genuine
# retrieval failure. This is diagnostic only; production would not normally
# return 100 document chunks to the generation stage.
INSPECTION_DEPTH = 100


@dataclass(frozen=True)
class CrossLingualPair:
    """One Nepali query and its controlled English counterpart."""

    question_id: str
    nepali_query: str
    english_query: str
    target_language: str
    primary_ids: tuple[str, ...]


PAIRS = (
    CrossLingualPair(
        question_id="ne_en_001",
        nepali_query=(
            "नेपालको संविधानले स्वास्थ्यसम्बन्धी अधिकारबारे "
            "के व्यवस्था गरेको छ?"
        ),
        english_query=(
            "What does the Constitution of Nepal guarantee "
            "regarding the right to health?"
        ),
        target_language="en",
        primary_ids=(
            "1511cd29-5f86-502f-9530-e9a46e45a0a6",
        ),
    ),
    CrossLingualPair(
        question_id="ne_en_002",
        nepali_query=(
            "नेपालको संविधानले सूचनाको अधिकारबारे के भन्छ?"
        ),
        english_query=(
            "What does the Constitution say about "
            "the right to information?"
        ),
        target_language="en",
        primary_ids=(
            "d89d8791-b929-55f2-b73e-947c09fcecd0",
        ),
    ),
    CrossLingualPair(
        question_id="ne_en_003",
        nepali_query=(
            "जनस्वास्थ्य सेवा ऐनले आकस्मिक स्वास्थ्य सेवाबारे "
            "के व्यवस्था गरेको छ?"
        ),
        english_query=(
            "What does the Public Health Service Act say about "
            "emergency health services?"
        ),
        target_language="en",
        primary_ids=(
            "6fb81c34-02e8-5ebc-8bb4-c055ea1e02d9",
        ),
    ),
    CrossLingualPair(
        question_id="ne_en_004",
        nepali_query=(
            "जनस्वास्थ्य सेवा ऐनले उपचारका लागि "
            "सूचित सहमतिबारे के भन्छ?"
        ),
        english_query=(
            "What does the Public Health Service Act say about "
            "informed consent for treatment?"
        ),
        target_language="en",
        primary_ids=(
            "b5058102-94ca-5573-8c0e-d137d22b053a",
        ),
    ),
    CrossLingualPair(
        question_id="ne_en_005",
        nepali_query=(
            "आर्थिक सर्वेक्षणले नेपालको आर्थिक वृद्धिबारे "
            "के जानकारी दिएको छ?"
        ),
        english_query=(
            "What does the Economic Survey report about "
            "Nepal's economic growth?"
        ),
        target_language="en",
        primary_ids=(
            "332cdfc6-3b19-564d-ac21-64a6fae52238",
        ),
    ),
    CrossLingualPair(
        question_id="ne_en_006",
        nepali_query=(
            "आर्थिक वर्ष २०२५/२६ को बजेट वक्तव्यले "
            "विद्यार्थी छात्रवृत्तिबारे के भन्छ?"
        ),
        english_query=(
            "What does the 2025/26 Budget Speech say about "
            "scholarships for students?"
        ),
        target_language="en",
        primary_ids=(
            "dadb426b-2b3b-570e-a6c6-d08cc55c77a1",
        ),
    ),
)


def find_primary_rank(
    results: list[RetrievalResult],
    primary_ids: tuple[str, ...],
) -> int | None:
    """Return the first rank containing manually verified primary evidence."""

    target_ids = set(primary_ids)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result.point_id in target_ids:
            return rank

    return None


def format_rank(
    rank: int | None,
) -> str:
    """Render missing evidence consistently in the comparison table."""

    if rank is None:
        return f">{INSPECTION_DEPTH}"

    return str(rank)


def main() -> None:
    """Compare Nepali and English formulations against English evidence."""

    embedding_service = HuggingFaceE5EmbeddingService()

    client = QdrantClient(
        url=QDRANT_URL,
    )

    retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )

    print(
        "Nepali vs English query dense-rank comparison "
        f"(depth={INSPECTION_DEPTH})"
    )
    print("-" * 68)
    print(
        f"{'Question':<12}"
        f"{'Nepali rank':>14}"
        f"{'English rank':>16}"
    )
    print("-" * 68)

    for pair in PAIRS:
        filters = {
            "language": pair.target_language,
        }

        # Retrieve the same English corpus twice: once using the original
        # Nepali query and once using its controlled English equivalent.
        # The gold passage stays identical, isolating query-language effects.
        nepali_results = retriever.retrieve(
            query=pair.nepali_query,
            top_k=INSPECTION_DEPTH,
            filters=filters,
        )

        english_results = retriever.retrieve(
            query=pair.english_query,
            top_k=INSPECTION_DEPTH,
            filters=filters,
        )

        nepali_rank = find_primary_rank(
            nepali_results,
            pair.primary_ids,
        )

        english_rank = find_primary_rank(
            english_results,
            pair.primary_ids,
        )

        print(
            f"{pair.question_id:<12}"
            f"{format_rank(nepali_rank):>14}"
            f"{format_rank(english_rank):>16}"
        )


if __name__ == "__main__":
    main()