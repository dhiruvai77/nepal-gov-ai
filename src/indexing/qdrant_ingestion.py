"""Production Qdrant ingestion utilities for NepalGov AI.

This module converts processed document chunks into Qdrant points. It depends
on the generic EmbeddingService interface rather than directly on E5, which
keeps ingestion independent from the model runtime.
"""

import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.embeddings.base import EmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "child_chunks"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "nepal_gov_documents"
DENSE_VECTOR_NAME = "dense"

# UUID5 gives every chunk a deterministic Qdrant-compatible identifier.
# Using a fixed namespace means the same chunk_id always resolves to the same
# point ID across machines and repeated indexing runs.
POINT_ID_NAMESPACE = uuid.UUID(
    "b96804e8-b3e4-4e92-b6bb-e32a18576851"
)

# Chunk identity and citation fields are required before data is allowed into
# the production vector collection.
REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "title",
    "organization",
    "language",
    "page_start",
    "page_end",
    "source_url",
    "chunk_text",
    "token_count",
}


def stable_point_id(chunk_id: str) -> str:
    """Create a deterministic UUID point ID from a chunk identifier."""

    clean_chunk_id = chunk_id.strip()

    if not clean_chunk_id:
        raise ValueError(
            "chunk_id must contain non-whitespace text."
        )

    # UUID5 hashes the chunk identifier within our project-specific namespace.
    # Returning its canonical string form matches Qdrant's supported UUID IDs.
    return str(
        uuid.uuid5(
            POINT_ID_NAMESPACE,
            clean_chunk_id,
        )
    )


def validate_chunk(chunk: dict[str, Any]) -> None:
    """Validate fields required for retrieval and source citation."""

    missing_fields = sorted(
        field
        for field in REQUIRED_CHUNK_FIELDS
        if field not in chunk or chunk[field] is None
    )

    if missing_fields:
        raise ValueError(
            "Chunk is missing required fields: "
            + ", ".join(missing_fields)
        )

    if not str(chunk["chunk_id"]).strip():
        raise ValueError(
            "chunk_id must contain non-whitespace text."
        )

    if not str(chunk["chunk_text"]).strip():
        raise ValueError(
            f"Chunk {chunk['chunk_id']} has empty chunk_text."
        )


def build_payload(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """Build the metadata payload stored alongside the dense vector."""

    validate_chunk(chunk)

    # Preserve retrieval, filtering, and citation metadata. Structural fields
    # may legitimately be null until structure-aware chunking is implemented.
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


def load_chunks(
    chunks_dir: Path = CHUNKS_DIR,
) -> list[dict[str, Any]]:
    """Load and validate every processed child chunk."""

    chunks: list[dict[str, Any]] = []

    for file_path in sorted(
        chunks_dir.glob("*.json")
    ):
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        # Current chunk outputs may be stored either directly as a list or
        # under a document-level "chunks" wrapper.
        document_chunks = (
            data
            if isinstance(data, list)
            else data.get("chunks", [])
        )

        for chunk in document_chunks:
            validate_chunk(chunk)
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            f"No processed chunks found in {chunks_dir}"
        )

    return chunks


def build_points(
    chunks: list[dict[str, Any]],
    embedding_service: EmbeddingService,
) -> list[PointStruct]:
    """Embed chunks and convert them into Qdrant points."""

    texts = [
        str(chunk["chunk_text"])
        for chunk in chunks
    ]

    vectors = embedding_service.embed_passages(
        texts
    )

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Embedding service returned a different number "
            "of vectors than input chunks."
        )

    points: list[PointStruct] = []

    for chunk, vector in zip(
        chunks,
        vectors,
        strict=True,
    ):
        if len(vector) != embedding_service.dimension:
            raise RuntimeError(
                f"Chunk {chunk['chunk_id']} produced "
                f"{len(vector)} dimensions; expected "
                f"{embedding_service.dimension}."
            )

        points.append(
            PointStruct(
                id=stable_point_id(
                    str(chunk["chunk_id"])
                ),
                vector={
                    DENSE_VECTOR_NAME: vector
                },
                payload=build_payload(chunk),
            )
        )

    return points


def ingest_chunks(
    client: QdrantClient,
    embedding_service: EmbeddingService,
    chunks: list[dict[str, Any]],
    batch_size: int = 64,
) -> int:
    """Embed and upsert chunks into Qdrant in bounded batches."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    total_ingested = 0

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start:start + batch_size
        ]

        points = build_points(
            batch,
            embedding_service,
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        total_ingested += len(points)

    return total_ingested