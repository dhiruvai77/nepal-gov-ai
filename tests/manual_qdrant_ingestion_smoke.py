"""Smoke-test dense + BM25 ingestion into the local Qdrant collection.

This script validates the real Qdrant client/server path using deterministic
dummy dense vectors plus Qdrant's BM25 Document representation. The dense
vectors are infrastructure-only and must not be used for retrieval evaluation.
"""

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    PointStruct,
)

from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
)


# Resolve repository-relative paths so the smoke test works regardless of the
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "child_chunks"
)

# Qdrant BM25 model used by both production ingestion and sparse retrieval.
BM25_MODEL = "Qdrant/bm25"

# Keep the smoke test intentionally small because its purpose is API/schema
# validation rather than corpus ingestion or retrieval-quality evaluation.
SMOKE_TEST_LIMIT = 5


def load_sample_chunks(
    limit: int,
) -> list[dict[str, Any]]:
    """Load a deterministic sample of processed child chunks."""

    chunks: list[dict[str, Any]] = []

    for file_path in sorted(
        CHUNKS_DIR.glob("*.json")
    ):
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            document_data = json.load(
                file
            )

        # Support both the direct-list format and a possible wrapper object.
        if isinstance(
            document_data,
            list,
        ):
            document_chunks = document_data
        else:
            document_chunks = document_data.get(
                "chunks",
                [],
            )

        for chunk in document_chunks:
            chunks.append(
                chunk
            )

            if len(
                chunks
            ) >= limit:
                return chunks

    return chunks


def build_dummy_vector(
    point_index: int,
) -> list[float]:
    """Create a deterministic 1024-dimensional placeholder dense vector.

    These vectors only validate the named dense-vector schema. They are not
    semantic E5 embeddings and must never be used for quality evaluation.
    """

    vector = [
        0.0
    ] * DENSE_VECTOR_SIZE

    # Set one coordinate so the vector is non-zero and reproducible.
    vector[
        point_index % DENSE_VECTOR_SIZE
    ] = 1.0

    return vector


def build_payload(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """Preserve retrieval and citation metadata in the smoke-test payload."""

    return {
        "chunk_id": chunk.get(
            "chunk_id"
        ),
        "parent_chunk_id": chunk.get(
            "parent_chunk_id"
        ),
        "document_id": chunk.get(
            "document_id"
        ),
        "title": chunk.get(
            "title"
        ),
        "organization": chunk.get(
            "organization"
        ),
        "category": chunk.get(
            "category"
        ),
        "document_type": chunk.get(
            "document_type"
        ),
        "language": chunk.get(
            "language"
        ),
        "publication_date": chunk.get(
            "publication_date"
        ),
        "source_url": chunk.get(
            "source_url"
        ),
        "download_url": chunk.get(
            "download_url"
        ),
        "retrieved_at": chunk.get(
            "retrieved_at"
        ),
        "page_start": chunk.get(
            "page_start"
        ),
        "page_end": chunk.get(
            "page_end"
        ),
        "section": chunk.get(
            "section"
        ),
        "subsection": chunk.get(
            "subsection"
        ),
        "article_number": chunk.get(
            "article_number"
        ),
        "article_title": chunk.get(
            "article_title"
        ),
        "chunk_index": chunk.get(
            "chunk_index"
        ),
        "chunk_text": chunk.get(
            "chunk_text"
        ),
        "token_count": chunk.get(
            "token_count"
        ),
        "extraction_method": chunk.get(
            "extraction_method"
        ),
    }


def main() -> None:
    """Insert a few temporary dense + BM25 points into local Qdrant."""

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

    for point_index, chunk in enumerate(
        chunks
    ):
        chunk_text = str(
            chunk.get(
                "chunk_text",
                "",
            )
        ).strip()

        if not chunk_text:
            raise RuntimeError(
                "Smoke-test chunk contains blank text."
            )

        # Integer IDs 1-5 are deliberately reserved for this temporary smoke
        # test. Production ingestion uses deterministic UUID5 point IDs.
        point = PointStruct(
            id=point_index + 1,
            vector={
                DENSE_VECTOR_NAME: build_dummy_vector(
                    point_index
                ),
                # Let Qdrant generate the sparse BM25 representation from the
                # real chunk text during the upsert request.
                SPARSE_VECTOR_NAME: Document(
                    text=chunk_text,
                    model=BM25_MODEL,
                ),
            },
            payload=build_payload(
                chunk
            ),
        )

        points.append(
            point
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(
        f"Inserted {len(points)} dense + BM25 smoke-test points "
        f"into {COLLECTION_NAME}."
    )


if __name__ == "__main__":
    main()