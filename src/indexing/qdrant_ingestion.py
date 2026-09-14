"""Production ingestion utilities for NepalGov AI.

This module loads processed document chunks, generates dense embeddings through
the generic EmbeddingService interface, attaches Qdrant's BM25 sparse document
representation, and upserts both representations with retrieval metadata.
"""

import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    PointStruct,
)

from src.embeddings.base import EmbeddingService
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "child_chunks"
)

QDRANT_URL = "http://localhost:6333"

# Qdrant sparse model used for BM25-style indexing and retrieval.
BM25_MODEL = "Qdrant/bm25"

# A fixed UUID namespace makes point IDs deterministic across repeated ingestion
# runs. Re-ingesting the same chunk therefore replaces the same Qdrant point
# instead of creating duplicates.
POINT_ID_NAMESPACE = uuid.UUID(
    "b96804e8-b3e4-4e92-b6bb-e32a18576851"
)

REQUIRED_CHUNK_FIELDS = (
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
)


def stable_point_id(
    chunk_id: str,
) -> str:
    """Create a deterministic Qdrant-compatible UUID for one chunk."""

    clean_chunk_id = chunk_id.strip()

    if not clean_chunk_id:
        raise ValueError(
            "chunk_id must not be blank."
        )

    return str(
        uuid.uuid5(
            POINT_ID_NAMESPACE,
            clean_chunk_id,
        )
    )


def validate_chunk(
    chunk: dict[str, Any],
) -> None:
    """Validate the minimum chunk contract required for retrieval."""

    missing_fields = [
        field_name
        for field_name in REQUIRED_CHUNK_FIELDS
        if chunk.get(field_name) is None
    ]

    if missing_fields:
        raise ValueError(
            "Chunk is missing required fields: "
            + ", ".join(missing_fields)
        )

    if not str(
        chunk["chunk_id"]
    ).strip():
        raise ValueError(
            "chunk_id must not be blank."
        )

    if not str(
        chunk["chunk_text"]
    ).strip():
        raise ValueError(
            "chunk_text must not be blank."
        )


def build_payload(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """Build the Qdrant payload used for retrieval and citation generation.

    The payload intentionally preserves the full chunk metadata rather than a
    reduced subset because later reranking, filtering, citation rendering, and
    evaluation stages may require different fields.
    """

    validate_chunk(
        chunk
    )

    return dict(
        chunk
    )


def load_chunks(
    chunks_dir: Path = CHUNKS_DIR,
) -> list[dict[str, Any]]:
    """Load and validate all processed child chunks in deterministic order."""

    chunks: list[dict[str, Any]] = []

    for path in sorted(
        chunks_dir.glob("*.json")
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

        # Support both the project's current list format and an optional
        # wrapper object if chunk serialization evolves later.
        if isinstance(
            data,
            dict,
        ):
            file_chunks = data.get(
                "chunks",
                [],
            )
        else:
            file_chunks = data

        if not isinstance(
            file_chunks,
            list,
        ):
            raise ValueError(
                f"Invalid chunk file format: {path}"
            )

        for chunk in file_chunks:
            validate_chunk(
                chunk
            )
            chunks.append(
                chunk
            )

    if not chunks:
        raise ValueError(
            f"No child chunks found in {chunks_dir}"
        )

    return chunks


def build_points(
    chunks: list[dict[str, Any]],
    embedding_service: EmbeddingService,
) -> list[PointStruct]:
    """Convert chunks into Qdrant points with dense and BM25 representations."""

    if not chunks:
        return []

    for chunk in chunks:
        validate_chunk(
            chunk
        )

    texts = [
        str(
            chunk["chunk_text"]
        ).strip()
        for chunk in chunks
    ]

    dense_vectors = (
        embedding_service.embed_passages(
            texts
        )
    )

    if len(
        dense_vectors
    ) != len(
        chunks
    ):
        raise RuntimeError(
            "Embedding service returned a different number "
            "of vectors than input chunks."
        )

    points: list[PointStruct] = []

    for chunk, text, dense_vector in zip(
        chunks,
        texts,
        dense_vectors,
        strict=True,
    ):
        if len(
            dense_vector
        ) != embedding_service.dimension:
            raise RuntimeError(
                "Embedding dimension mismatch for "
                f"chunk {chunk['chunk_id']}: "
                f"expected {embedding_service.dimension}, "
                f"received {len(dense_vector)}."
            )

        # Qdrant's Document representation tells the server to generate the
        # BM25 sparse vector from the same chunk text used by dense retrieval.
        # This avoids introducing a second local sparse-model runtime.
        sparse_document = Document(
            text=text,
            model=BM25_MODEL,
        )

        points.append(
            PointStruct(
                id=stable_point_id(
                    str(
                        chunk["chunk_id"]
                    )
                ),
                vector={
                    DENSE_VECTOR_NAME: dense_vector,
                    SPARSE_VECTOR_NAME: sparse_document,
                },
                payload=build_payload(
                    chunk
                ),
            )
        )

    return points


def ingest_chunks(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    embedding_service: EmbeddingService,
    batch_size: int = 64,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Embed and upsert chunks into Qdrant in bounded batches."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    inserted_count = 0

    for start_index in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start_index:
            start_index + batch_size
        ]

        points = build_points(
            chunks=batch,
            embedding_service=embedding_service,
        )

        if not points:
            continue

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        inserted_count += len(
            points
        )

    return inserted_count