from __future__ import annotations

from dataclasses import dataclass

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
)


# These are the three persistent English same-language failures that are most
# useful for understanding whether chunk content or dense semantics is at fault.
@dataclass(frozen=True)
class InspectionCase:
    question_id: str
    query: str
    primary_point_id: str


CASES = (
    InspectionCase(
        question_id="en_en_004",
        query=(
            "What does the Economic Survey report "
            "about schools and students?"
        ),
        primary_point_id=(
            "145add01-c0eb-5201-8c37-9ce095fac25f"
        ),
    ),
    InspectionCase(
        question_id="en_en_007",
        query=(
            "What does the Constitution of Nepal guarantee "
            "regarding the right to health?"
        ),
        primary_point_id=(
            "1511cd29-5f86-502f-9530-e9a46e45a0a6"
        ),
    ),
    InspectionCase(
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


# Search deeply enough to recover the known difficult passages when possible.
INSPECTION_DEPTH = 100


def main() -> None:
    """Compare each difficult gold chunk with the model's top dense results."""

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

    for case in CASES:
        print("\n" + "=" * 90)
        print(
            f"{case.question_id}: {case.query}"
        )
        print("=" * 90)

        # Retrieve the actual gold point directly so we can inspect its complete
        # indexed payload rather than relying on a truncated search preview.
        gold_points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[
                case.primary_point_id,
            ],
            with_payload=True,
            with_vectors=False,
        )

        if not gold_points:
            raise RuntimeError(
                f"Gold point {case.primary_point_id} "
                "was not found in Qdrant."
            )

        gold_payload = (
            gold_points[0].payload or {}
        )

        print("\nGOLD CHUNK")
        print(
            f"point_id: {case.primary_point_id}"
        )
        print(
            f"chunk_id: "
            f"{gold_payload.get('chunk_id')}"
        )
        print(
            f"document_id: "
            f"{gold_payload.get('document_id')}"
        )
        print(
            f"pages: "
            f"{gold_payload.get('page_start')}-"
            f"{gold_payload.get('page_end')}"
        )
        print(
            f"token_count: "
            f"{gold_payload.get('token_count')}"
        )
        print(
            "text:\n"
            f"{gold_payload.get('chunk_text')}"
        )

        # Restrict this diagnostic to English evidence so its ranking matches
        # the benchmark's target-language constraint.
        results = retriever.retrieve(
            query=case.query,
            top_k=INSPECTION_DEPTH,
            filters={
                "language": "en",
            },
        )

        gold_rank = next(
            (
                rank
                for rank, result in enumerate(
                    results,
                    start=1,
                )
                if result.point_id
                == case.primary_point_id
            ),
            None,
        )

        print(
            "\nDENSE GOLD RANK: "
            f"{gold_rank if gold_rank is not None else '>100'}"
        )

        print("\nTOP 5 DENSE RESULTS")

        for rank, result in enumerate(
            results[:5],
            start=1,
        ):
            # A moderate preview is enough to judge why the embedding model
            # prefers these candidates over the manually annotated gold chunk.
            preview = (
                result.chunk_text[:900]
                .replace("\n", " ")
            )

            print(
                f"\nRank {rank}"
                f"\npoint_id: {result.point_id}"
                f"\nchunk_id: {result.chunk_id}"
                f"\ndocument_id: {result.document_id}"
                f"\npages: {result.page_start}-{result.page_end}"
                f"\nscore: {result.score:.6f}"
                f"\ntext: {preview}"
            )


if __name__ == "__main__":
    main()