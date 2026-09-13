"""Smoke-test ingestion of NepalGov AI chunks into Qdrant.

This script validates the chunk-to-Qdrant payload pipeline using deterministic
placeholder vectors. Real E5 embeddings will replace these vectors later
without changing the collection schema or document metadata structure.
"""

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# Resolve repository-relative paths so the script works regardless of the
# current working directory.
# The smoke test lives directly under tests/, so parents[1] resolves to
# the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "child_chunks"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "nepal_gov_documents"
DENSE_VECTOR_NAME = "dense"
DENSE_VECTOR_SIZE = 1024

# Keep the initial smoke test deliberately small. We only need enough points
# to validate metadata mapping and Qdrant insertion behavior.
SMOKE_TEST_LIMIT = 5


def load_sample_chunks(limit: int) -> list[dict[str, Any]]:
    """Load a small deterministic sample of child chunks from processed JSON."""

    chunks: list[dict[str, Any]] = []

    # Sorting keeps the smoke test reproducible across runs and machines.
    for file_path in sorted(CHUNKS_DIR.glob("*.json")):
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            document_data = json.load(file)

        # The chunker output may be either a direct list or a document wrapper.
        # Supporting both forms makes this ingestion layer tolerant of the
        # current baseline format while we refine the pipeline.
        if isinstance(document_data, list):
            document_chunks = document_data
        else:
            document_chunks = document_data.get(
                "chunks",
                [],
            )

        for chunk in document_chunks:
            chunks.append(chunk)

            if len(chunks) >= limit:
                return chunks

    return chunks


def build_dummy_vector(point_index: int) -> list[float]:
    """Create a deterministic placeholder vector for infrastructure testing.

    These vectors are not semantic embeddings and must never be used to
    evaluate retrieval quality.
    """

    vector = [0.0] * DENSE_VECTOR_SIZE

    # Set one coordinate so each smoke-test point is non-zero and deterministic.
    vector[
        point_index % DENSE_VECTOR_SIZE
    ] = 1.0

    return vector


def build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    """Map chunk metadata into the payload stored alongside its vector."""

    return {
        "chunk_id": chunk.get("chunk_id"),
        "parent_chunk_id": chunk.get("parent_chunk_id"),
        "document_id": chunk.get("document_id"),
        "title": chunk.get("title"),
        "organization": chunk.get("organization"),
        "category": chunk.get("category"),
        "document_type": chunk.get("document_type"),
        "language": chunk.get("language"),
        "publication_date": chunk.get("publication_date"),
        "source_url": chunk.get("source_url"),
        "download_url": chunk.get("download_url"),
        "retrieved_at": chunk.get("retrieved_at"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "section": chunk.get("section"),
        "subsection": chunk.get("subsection"),
        "article_number": chunk.get("article_number"),
        "article_title": chunk.get("article_title"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_text": chunk.get("chunk_text"),
        "token_count": chunk.get("token_count"),
        "extraction_method": chunk.get("extraction_method"),
    }


def main() -> None:
    """Load sample chunks and insert them into the local Qdrant collection."""

    chunks = load_sample_chunks(
        SMOKE_TEST_LIMIT
    )

    if not chunks:
        raise RuntimeError(
            f"No chunk files found in {CHUNKS_DIR}"
        )

    client = QdrantClient(
        url=QDRANT_URL
    )

    points: list[PointStruct] = []

    for point_index, chunk in enumerate(chunks):
        # Integer point IDs are sufficient for this infrastructure smoke test.
        # Production ingestion will use stable deterministic IDs.
        point = PointStruct(
            id=point_index + 1,
            vector={
                DENSE_VECTOR_NAME: build_dummy_vector(
                    point_index
                )
            },
            payload=build_payload(chunk),
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(
        f"Inserted {len(points)} smoke-test points "
        f"into {COLLECTION_NAME}."
    )


if __name__ == "__main__":
    main()