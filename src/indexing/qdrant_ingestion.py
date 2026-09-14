"""Production ingestion utilities for NepalGov AI.

This module loads processed document chunks, generates both raw and
metadata-contextualized dense embeddings through the generic EmbeddingService
interface, attaches Qdrant's BM25 sparse document representation, and upserts
all retrieval representations with the original chunk metadata.
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
from src.embeddings.contextual_passage import (
    build_contextualized_passage_text,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
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

    The payload intentionally preserves the full original chunk metadata.
    Contextualized text is used only for one dense embedding representation;
    it does not replace source text used for citations or answer generation.
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
    """Convert chunks into raw dense, contextual dense, and BM25 vectors."""

    if not chunks:
        return []

    for chunk in chunks:
        validate_chunk(
            chunk
        )

    # Raw source text remains the canonical representation used by:
    # - the existing dense retrieval baseline;
    # - BM25 sparse retrieval;
    # - citations;
    # - downstream RAG generation.
    raw_texts = [
        str(
            chunk["chunk_text"]
        ).strip()
        for chunk in chunks
    ]

    # Contextual dense text enriches each chunk with useful document metadata
    # such as title, document type, section, and article information.
    #
    # This text exists only for semantic embedding. It does not replace the
    # original `chunk_text` stored in the point payload.
    contextual_texts = [
        build_contextualized_passage_text(
            chunk
        )
        for chunk in chunks
    ]

    # Embed both representations in one provider call. Hosted E5 performs its
    # own bounded batching internally, so this remains safe for ingestion.
    embedding_inputs = [
        *raw_texts,
        *contextual_texts,
    ]

    dense_vectors = (
        embedding_service.embed_passages(
            embedding_inputs
        )
    )

    expected_vector_count = (
        len(chunks) * 2
    )

    if (
        len(dense_vectors)
        != expected_vector_count
    ):
        raise RuntimeError(
            "Embedding service returned a different number "
            "of vectors than input passage representations."
        )

    # Inputs were ordered as all raw texts followed by all contextual texts.
    # Split the returned vectors at the same boundary.
    split_index = len(chunks)

    raw_dense_vectors = (
        dense_vectors[
            :split_index
        ]
    )

    contextual_dense_vectors = (
        dense_vectors[
            split_index:
        ]
    )

    points: list[PointStruct] = []

    for (
        chunk,
        raw_text,
        raw_dense_vector,
        contextual_dense_vector,
    ) in zip(
        chunks,
        raw_texts,
        raw_dense_vectors,
        contextual_dense_vectors,
        strict=True,
    ):
        # Both dense representations use the same E5 model and therefore must
        # satisfy the same vector-dimension contract.
        for vector_name, dense_vector in (
            (
                DENSE_VECTOR_NAME,
                raw_dense_vector,
            ),
            (
                CONTEXTUAL_DENSE_VECTOR_NAME,
                contextual_dense_vector,
            ),
        ):
            if (
                len(dense_vector)
                != embedding_service.dimension
            ):
                raise RuntimeError(
                    "Embedding dimension mismatch for "
                    f"{vector_name} vector of chunk "
                    f"{chunk['chunk_id']}: "
                    f"expected {embedding_service.dimension}, "
                    f"received {len(dense_vector)}."
                )

        # BM25 deliberately continues to use the original chunk text.
        #
        # This keeps the sparse retrieval baseline unchanged so subsequent
        # evaluation isolates the effect of contextual dense embeddings.
        sparse_document = Document(
            text=raw_text,
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
                    # Existing production dense baseline.
                    DENSE_VECTOR_NAME: (
                        raw_dense_vector
                    ),

                    # New metadata-enriched semantic representation.
                    CONTEXTUAL_DENSE_VECTOR_NAME: (
                        contextual_dense_vector
                    ),

                    # Existing lexical representation remains unchanged.
                    SPARSE_VECTOR_NAME: (
                        sparse_document
                    ),
                },

                # Preserve original chunk metadata and source text.
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