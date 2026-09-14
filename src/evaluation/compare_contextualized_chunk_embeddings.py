from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    QDRANT_URL,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
)


# One hundred candidates are enough to tell whether contextualizing the chunk
# turns a severe retrieval failure into a realistically recoverable candidate.
INSPECTION_DEPTH = 100


@dataclass(frozen=True)
class RepresentationCase:
    """One difficult gold passage used for embedding-representation analysis."""

    question_id: str
    query: str
    primary_point_id: str


CASES = (
    RepresentationCase(
        question_id="en_en_004",
        query=(
            "What does the Economic Survey report "
            "about schools and students?"
        ),
        primary_point_id=(
            "145add01-c0eb-5201-8c37-9ce095fac25f"
        ),
    ),
    RepresentationCase(
        question_id="en_en_007",
        query=(
            "What does the Constitution of Nepal guarantee "
            "regarding the right to health?"
        ),
        primary_point_id=(
            "1511cd29-5f86-502f-9530-e9a46e45a0a6"
        ),
    ),
    RepresentationCase(
        question_id="en_en_011",
        query=(
            "What does the Economic Survey report about "
            "Nepal's economic growth?"
        ),
        primary_point_id=(
            "332cdfc6-3b19-564d-ac21-64a6fae52238"
        ),
    ),
)


def build_contextualized_text(
    payload: dict[str, Any],
) -> str:
    """Add useful document structure to the text sent to the embedder."""

    # Only metadata that can help semantic retrieval is included. Operational
    # fields such as URLs, hashes, and retrieval timestamps would add noise.
    metadata_fields = (
        ("Document", payload.get("title")),
        ("Organization", payload.get("organization")),
        ("Document type", payload.get("document_type")),
        ("Section", payload.get("section")),
        ("Subsection", payload.get("subsection")),
        ("Article number", payload.get("article_number")),
        ("Article title", payload.get("article_title")),
    )

    context_lines = [
        f"{label}: {value}"
        for label, value in metadata_fields
        if value is not None and str(value).strip()
    ]

    chunk_text = str(
        payload.get("chunk_text", "")
    ).strip()

    if not chunk_text:
        raise RuntimeError(
            "Gold Qdrant point is missing chunk_text."
        )

    # Keeping metadata before the content gives the embedding model the source
    # and structural context before it processes the body of the passage.
    return "\n".join(
        [
            *context_lines,
            "Content:",
            chunk_text,
        ]
    )


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    """Return cosine similarity for already-normalized E5 vectors."""

    if len(left) != len(right):
        raise ValueError(
            "Embedding vectors must have matching dimensions."
        )

    # Hosted E5 embeddings are normalized, so their dot product equals cosine
    # similarity and no NumPy dependency is required for this diagnostic.
    return sum(
        left_value * right_value
        for left_value, right_value in zip(
            left,
            right,
        )
    )


def estimate_rank(
    results: list[RetrievalResult],
    similarity: float,
    *,
    exclude_point_id: str,
) -> int | None:
    """Estimate where a newly embedded passage would rank in the top 100."""

    # Exclude the existing version of the gold point so we can estimate where
    # the replacement representation itself would enter the current ranking.
    competing_scores = [
        result.score
        for result in results
        if result.point_id != exclude_point_id
    ]

    # If at least 100 competitors still have a higher score, the candidate
    # remains outside the useful inspection window.
    higher_count = sum(
        score > similarity
        for score in competing_scores
    )

    if (
        len(competing_scores) >= INSPECTION_DEPTH - 1
        and higher_count >= INSPECTION_DEPTH - 1
    ):
        return None

    return higher_count + 1


def format_rank(
    rank: int | None,
) -> str:
    """Render estimated ranks consistently."""

    if rank is None:
        return f">{INSPECTION_DEPTH}"

    return str(rank)


def main() -> None:
    """Compare raw and metadata-contextualized gold-passage embeddings."""

    client = QdrantClient(
        url=QDRANT_URL,
    )

    embedding_service = (
        HuggingFaceE5EmbeddingService()
    )

    retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )

    print(
        "Raw vs contextualized chunk embedding comparison "
        f"(depth={INSPECTION_DEPTH})"
    )
    print("-" * 86)
    print(
        f"{'Question':<12}"
        f"{'Raw sim':>12}"
        f"{'Raw rank':>12}"
        f"{'Context sim':>14}"
        f"{'Context rank':>16}"
    )
    print("-" * 86)

    for case in CASES:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[
                case.primary_point_id,
            ],
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            raise RuntimeError(
                f"Gold point {case.primary_point_id} "
                "was not found in Qdrant."
            )

        payload = points[0].payload or {}

        raw_text = str(
            payload.get("chunk_text", "")
        ).strip()

        contextualized_text = (
            build_contextualized_text(
                payload
            )
        )

        # Embed the query exactly as production does, then compare it with two
        # alternative representations of the same gold evidence.
        query_vector = embedding_service.embed_query(
            case.query
        )

        passage_vectors = (
            embedding_service.embed_passages(
                [
                    raw_text,
                    contextualized_text,
                ]
            )
        )

        raw_similarity = cosine_similarity(
            query_vector,
            passage_vectors[0],
        )

        contextualized_similarity = cosine_similarity(
            query_vector,
            passage_vectors[1],
        )

        # The existing top-100 ranking supplies the score distribution against
        # which both alternative gold representations are compared.
        results = retriever.retrieve(
            query=case.query,
            top_k=INSPECTION_DEPTH,
            filters={
                "language": "en",
            },
        )

        raw_rank = estimate_rank(
            results,
            raw_similarity,
            exclude_point_id=case.primary_point_id,
        )

        contextualized_rank = estimate_rank(
            results,
            contextualized_similarity,
            exclude_point_id=case.primary_point_id,
        )

        print(
            f"{case.question_id:<12}"
            f"{raw_similarity:>12.6f}"
            f"{format_rank(raw_rank):>12}"
            f"{contextualized_similarity:>14.6f}"
            f"{format_rank(contextualized_rank):>16}"
        )


if __name__ == "__main__":
    main()