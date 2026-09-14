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


# Depth 100 matches our earlier failure inspection and makes the constrained
# and unconstrained ranks directly comparable.
INSPECTION_DEPTH = 100


@dataclass(frozen=True)
class DocumentConstraintCase:
    """One difficult query with an explicitly named target document."""

    question_id: str
    query: str
    document_id: str
    primary_point_id: str


CASES = (
    DocumentConstraintCase(
        question_id="en_en_004",
        query=(
            "What does the Economic Survey report "
            "about schools and students?"
        ),
        document_id="economic_survey_2023_24_en",
        primary_point_id=(
            "145add01-c0eb-5201-8c37-9ce095fac25f"
        ),
    ),
    DocumentConstraintCase(
        question_id="en_en_007",
        query=(
            "What does the Constitution of Nepal guarantee "
            "regarding the right to health?"
        ),
        document_id="constitution_nepal_current_en",
        primary_point_id=(
            "1511cd29-5f86-502f-9530-e9a46e45a0a6"
        ),
    ),
    DocumentConstraintCase(
        question_id="en_en_011",
        query=(
            "What does the Economic Survey report about "
            "Nepal's economic growth?"
        ),
        document_id="economic_survey_2023_24_en",
        primary_point_id=(
            "332cdfc6-3b19-564d-ac21-64a6fae52238"
        ),
    ),
)


def find_rank(
    results: list[RetrievalResult],
    point_id: str,
) -> int | None:
    """Return the one-based rank of the expected primary evidence."""

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result.point_id == point_id:
            return rank

    return None


def format_rank(
    rank: int | None,
) -> str:
    """Display results outside the inspection depth consistently."""

    return (
        str(rank)
        if rank is not None
        else f">{INSPECTION_DEPTH}"
    )


def main() -> None:
    """Compare normal dense retrieval with document-constrained retrieval."""

    embedding_service = (
        HuggingFaceE5EmbeddingService()
    )

    client = QdrantClient(
        url=QDRANT_URL,
    )

    retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )

    print(
        "Document-constrained dense retrieval comparison "
        f"(depth={INSPECTION_DEPTH})"
    )
    print("-" * 76)
    print(
        f"{'Question':<12}"
        f"{'Normal rank':>14}"
        f"{'Document rank':>18}"
    )
    print("-" * 76)

    for case in CASES:
        # First reproduce normal benchmark retrieval: only the target language
        # is constrained, so all English government documents compete.
        normal_results = retriever.retrieve(
            query=case.query,
            top_k=INSPECTION_DEPTH,
            filters={
                "language": "en",
            },
        )

        # Then constrain retrieval to the document explicitly named by the
        # question. This tests whether cross-document competition is the main
        # cause of the observed dense failures.
        document_results = retriever.retrieve(
            query=case.query,
            top_k=INSPECTION_DEPTH,
            filters={
                "language": "en",
                "document_id": case.document_id,
            },
        )

        normal_rank = find_rank(
            normal_results,
            case.primary_point_id,
        )

        document_rank = find_rank(
            document_results,
            case.primary_point_id,
        )

        print(
            f"{case.question_id:<12}"
            f"{format_rank(normal_rank):>14}"
            f"{format_rank(document_rank):>18}"
        )


if __name__ == "__main__":
    main()