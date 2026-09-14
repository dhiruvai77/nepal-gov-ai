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
from src.retrieval.hybrid_retriever import (
    DEFAULT_RRF_K,
    HybridRetriever,
)
from src.retrieval.sparse_retriever import (
    SparseRetriever,
)


# We only inspect benchmark cases that remained unresolved at @20.
# Keeping this diagnostic targeted avoids needlessly spending hosted
# embedding requests on all 30 evaluation questions.
@dataclass(frozen=True)
class FailureCase:
    question_id: str
    query: str
    target_language: str
    primary_ids: tuple[str, ...]
    cross_lingual: bool


FAILURE_CASES = (
    FailureCase(
        question_id="en_en_004",
        query="What does the Economic Survey report about schools and students?",
        target_language="en",
        primary_ids=(
            "145add01-c0eb-5201-8c37-9ce095fac25f",
        ),
        cross_lingual=False,
    ),
    FailureCase(
        question_id="en_en_007",
        query=(
            "What does the Constitution of Nepal guarantee "
            "regarding the right to health?"
        ),
        target_language="en",
        primary_ids=(
            "1511cd29-5f86-502f-9530-e9a46e45a0a6",
        ),
        cross_lingual=False,
    ),
    FailureCase(
        question_id="en_en_011",
        query=(
            "What does the Economic Survey report about "
            "Nepal's economic growth?"
        ),
        target_language="en",
        primary_ids=(
            "332cdfc6-3b19-564d-ac21-64a6fae52238",
        ),
        cross_lingual=False,
    ),
    FailureCase(
        question_id="ne_en_001",
        query="नेपालको संविधानले स्वास्थ्यसम्बन्धी अधिकारबारे के व्यवस्था गरेको छ?",
        target_language="en",
        primary_ids=(
            "1511cd29-5f86-502f-9530-e9a46e45a0a6",
        ),
        cross_lingual=True,
    ),
    FailureCase(
        question_id="ne_en_002",
        query="नेपालको संविधानले सूचनाको अधिकारबारे के भन्छ?",
        target_language="en",
        primary_ids=(
            "d89d8791-b929-55f2-b73e-947c09fcecd0",
        ),
        cross_lingual=True,
    ),
    FailureCase(
        question_id="ne_en_004",
        query="जनस्वास्थ्य सेवा ऐनले उपचारका लागि सूचित सहमतिबारे के भन्छ?",
        target_language="en",
        primary_ids=(
            "b5058102-94ca-5573-8c0e-d137d22b053a",
        ),
        cross_lingual=True,
    ),
    FailureCase(
        question_id="ne_en_005",
        query="आर्थिक सर्वेक्षणले नेपालको आर्थिक वृद्धिबारे के जानकारी दिएको छ?",
        target_language="en",
        primary_ids=(
            "332cdfc6-3b19-564d-ac21-64a6fae52238",
        ),
        cross_lingual=True,
    ),
)


# A depth of 100 distinguishes a moderate ranking problem from a much more
# serious semantic-retrieval failure without scanning the entire collection.
INSPECTION_DEPTH = 100

# Same-language hybrid fusion needs a larger component candidate set than the
# final inspection depth so moderately ranked evidence can still be promoted.
HYBRID_CANDIDATE_DEPTH = 300


@dataclass
class CachedRetriever:
    """Expose cached candidates through HybridRetriever's retriever contract."""

    results: list[RetrievalResult]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return the requested prefix without issuing another retrieval call."""

        # Candidates already correspond to the current query and filters.
        del query, filters

        return self.results[:top_k]


def find_rank(
    results: list[RetrievalResult],
    target_ids: tuple[str, ...],
) -> int | None:
    """Return the first one-based rank containing primary gold evidence."""

    target_set = set(target_ids)

    for rank, result in enumerate(results, start=1):
        if result.point_id in target_set:
            return rank

    return None


def format_rank(rank: int | None) -> str:
    """Render missing evidence clearly in diagnostic output."""

    return str(rank) if rank is not None else ">100"


def main() -> None:
    """Inspect exact primary-evidence ranks for persistent benchmark failures."""

    embedding_service = HuggingFaceE5EmbeddingService()
    client = QdrantClient(
        url=QDRANT_URL
    )

    dense_retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )
    sparse_retriever = SparseRetriever(
        client=client,
    )

    print(
        f"Persistent failure inspection "
        f"(depth={INSPECTION_DEPTH})"
    )
    print("-" * 72)

    for index, case in enumerate(
        FAILURE_CASES,
        start=1,
    ):
        print(
            f"\n[{index}/{len(FAILURE_CASES)}] "
            f"{case.question_id}"
        )

        filters = {
            "language": case.target_language,
        }

        # Retrieve dense candidates once for every case. Cross-lingual
        # production retrieval uses only this path.
        dense_candidates = dense_retriever.retrieve(
            query=case.query,
            top_k=(
                HYBRID_CANDIDATE_DEPTH
                if not case.cross_lingual
                else INSPECTION_DEPTH
            ),
            filters=filters,
        )

        dense_rank = find_rank(
            dense_candidates[:INSPECTION_DEPTH],
            case.primary_ids,
        )

        print(
            f"  dense primary rank: "
            f"{format_rank(dense_rank)}"
        )

        if case.cross_lingual:
            # BM25/hybrid are intentionally not evaluated across languages
            # because production routing correctly uses multilingual dense
            # retrieval when query and evidence languages differ.
            continue

        sparse_candidates = sparse_retriever.retrieve(
            query=case.query,
            top_k=HYBRID_CANDIDATE_DEPTH,
            filters=filters,
        )

        sparse_rank = find_rank(
            sparse_candidates[:INSPECTION_DEPTH],
            case.primary_ids,
        )

        # Reuse the actual production RRF implementation while preventing
        # duplicate dense embedding calls through cached component rankings.
        hybrid_retriever = HybridRetriever(
            dense_retriever=CachedRetriever(
                dense_candidates
            ),
            sparse_retriever=CachedRetriever(
                sparse_candidates
            ),
            rrf_k=DEFAULT_RRF_K,
            # 300 cached component candidates / 100 final results.
            candidate_multiplier=3,
        )

        hybrid_results = hybrid_retriever.retrieve(
            query=case.query,
            top_k=INSPECTION_DEPTH,
            filters=filters,
        )

        hybrid_rank = find_rank(
            hybrid_results,
            case.primary_ids,
        )

        print(
            f"  bm25 primary rank:  "
            f"{format_rank(sparse_rank)}"
        )
        print(
            f"  hybrid primary rank:"
            f" {format_rank(hybrid_rank)}"
        )


if __name__ == "__main__":
    main()