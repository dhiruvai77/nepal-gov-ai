"""Backfill metadata-contextualized dense vectors for existing Qdrant points.

This migration updates only the `dense_contextual` named vector. Existing raw
dense vectors, BM25 sparse vectors, payloads, and point identifiers are
preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointVectors

from src.embeddings.base import EmbeddingService
from src.embeddings.contextual_passage import (
    build_contextualized_passage_text,
)
from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.indexing.qdrant_ingestion import (
    load_chunks,
    stable_point_id,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    QDRANT_URL,
    SPARSE_VECTOR_NAME,
    ensure_contextual_dense_vector,
)


DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class ContextualBackfillStats:
    """Summary of one contextual-vector backfill run."""

    updated: int
    skipped: int


def _named_vectors(
    record: Any,
) -> dict[str, Any]:
    """Return named vectors from one retrieved Qdrant record."""

    vectors = getattr(
        record,
        "vector",
        None,
    )

    if vectors is None:
        return {}

    # This migration operates on a named-vector collection. An unnamed vector
    # response would indicate an unexpected collection configuration.
    if not isinstance(
        vectors,
        dict,
    ):
        raise RuntimeError(
            "Expected Qdrant point to contain named vectors."
        )

    return vectors


def _retrieve_vectors(
    client: QdrantClient,
    point_ids: list[str],
    collection_name: str,
) -> dict[str, Any]:
    """Retrieve all retrieval vectors needed for migration verification."""

    records = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=False,
        with_vectors=[
            DENSE_VECTOR_NAME,
            CONTEXTUAL_DENSE_VECTOR_NAME,
            SPARSE_VECTOR_NAME,
        ],
    )

    return {
        str(record.id): record
        for record in records
    }


def backfill_contextual_vectors(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    embedding_service: EmbeddingService,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_name: str = COLLECTION_NAME,
) -> ContextualBackfillStats:
    """Backfill contextual vectors while preserving existing representations.

    Existing contextual vectors are skipped, making the operation resumable.
    Each updated batch is read back from Qdrant so the migration verifies that
    the original dense and BM25 vectors were not changed.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    updated_count = 0
    skipped_count = 0

    for start_index in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start_index:
            start_index + batch_size
        ]

        point_ids = [
            stable_point_id(
                str(chunk["chunk_id"])
            )
            for chunk in batch
        ]

        # Read the current vector state before modifying anything. Besides
        # supporting resumability, this gives us a baseline for checking that
        # raw dense and BM25 representations survive the migration untouched.
        before_records = _retrieve_vectors(
            client=client,
            point_ids=point_ids,
            collection_name=collection_name,
        )

        missing_point_ids = [
            point_id
            for point_id in point_ids
            if point_id not in before_records
        ]

        if missing_point_ids:
            raise RuntimeError(
                "Cannot backfill contextual vectors because "
                "existing Qdrant points are missing: "
                + ", ".join(
                    missing_point_ids
                )
            )

        pending: list[
            tuple[
                dict[str, Any],
                str,
                dict[str, Any],
            ]
        ] = []

        for chunk, point_id in zip(
            batch,
            point_ids,
            strict=True,
        ):
            vectors = _named_vectors(
                before_records[
                    point_id
                ]
            )

            # Every existing indexed point must already have the two baseline
            # retrieval representations before this migration is allowed.
            if DENSE_VECTOR_NAME not in vectors:
                raise RuntimeError(
                    f"Point {point_id} is missing "
                    f"'{DENSE_VECTOR_NAME}'."
                )

            if SPARSE_VECTOR_NAME not in vectors:
                raise RuntimeError(
                    f"Point {point_id} is missing "
                    f"'{SPARSE_VECTOR_NAME}'."
                )

            # A non-null contextual vector means this point was already
            # migrated during an earlier run, so safely skip it.
            if (
                CONTEXTUAL_DENSE_VECTOR_NAME
                in vectors
                and vectors[
                    CONTEXTUAL_DENSE_VECTOR_NAME
                ]
                is not None
            ):
                skipped_count += 1
                continue

            pending.append(
                (
                    chunk,
                    point_id,
                    vectors,
                )
            )

        if not pending:
            continue

        contextual_texts = [
            build_contextualized_passage_text(
                chunk
            )
            for chunk, _, _ in pending
        ]

        contextual_vectors = (
            embedding_service.embed_passages(
                contextual_texts
            )
        )

        if (
            len(contextual_vectors)
            != len(pending)
        ):
            raise RuntimeError(
                "Embedding service returned a different "
                "number of vectors than contextual passages."
            )

        updates: list[PointVectors] = []

        for (
            (_, point_id, _),
            vector,
        ) in zip(
            pending,
            contextual_vectors,
            strict=True,
        ):
            if (
                len(vector)
                != embedding_service.dimension
            ):
                raise RuntimeError(
                    "Contextual embedding dimension mismatch "
                    f"for point {point_id}: expected "
                    f"{embedding_service.dimension}, "
                    f"received {len(vector)}."
                )

            # Only the new contextual vector is supplied here. Qdrant's vector
            # update operation therefore leaves raw dense and BM25 untouched.
            updates.append(
                PointVectors(
                    id=point_id,
                    vector={
                        CONTEXTUAL_DENSE_VECTOR_NAME: vector,
                    },
                )
            )

        client.update_vectors(
            collection_name=collection_name,
            points=updates,
            wait=True,
        )

        pending_ids = [
            point_id
            for _, point_id, _ in pending
        ]

        # Read the updated points back so the migration fails immediately if
        # any baseline representation was unexpectedly changed or removed.
        after_records = _retrieve_vectors(
            client=client,
            point_ids=pending_ids,
            collection_name=collection_name,
        )

        for (
            _,
            point_id,
            before_vectors,
        ) in pending:
            after_record = (
                after_records.get(
                    point_id
                )
            )

            if after_record is None:
                raise RuntimeError(
                    f"Updated point {point_id} "
                    "could not be retrieved."
                )

            after_vectors = _named_vectors(
                after_record
            )

            if (
                CONTEXTUAL_DENSE_VECTOR_NAME
                not in after_vectors
                or after_vectors[
                    CONTEXTUAL_DENSE_VECTOR_NAME
                ]
                is None
            ):
                raise RuntimeError(
                    f"Point {point_id} did not receive "
                    "a contextual dense vector."
                )

            # Exact equality is appropriate here because update_vectors should
            # not recalculate or rewrite either of these stored representations.
            if (
                after_vectors.get(
                    DENSE_VECTOR_NAME
                )
                != before_vectors.get(
                    DENSE_VECTOR_NAME
                )
            ):
                raise RuntimeError(
                    f"Raw dense vector changed for "
                    f"point {point_id}."
                )

            if (
                after_vectors.get(
                    SPARSE_VECTOR_NAME
                )
                != before_vectors.get(
                    SPARSE_VECTOR_NAME
                )
            ):
                raise RuntimeError(
                    f"BM25 vector changed for "
                    f"point {point_id}."
                )

        updated_count += len(
            pending
        )

        print(
            "Verified batch: "
            f"{len(pending)} contextual vector(s) updated; "
            "raw dense and BM25 preserved."
        )

    return ContextualBackfillStats(
        updated=updated_count,
        skipped=skipped_count,
    )


def run_contextual_backfill(
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedding_service: EmbeddingService | None = None,
    client: QdrantClient | None = None,
) -> ContextualBackfillStats:
    """Run the contextual-vector migration against processed project chunks."""

    if (
        limit is not None
        and limit <= 0
    ):
        raise ValueError(
            "limit must be greater than zero when provided."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    active_client = (
        client
        if client is not None
        else QdrantClient(
            url=QDRANT_URL
        )
    )

    active_embedding_service = (
        embedding_service
        if embedding_service is not None
        else HuggingFaceE5EmbeddingService()
    )

    # Ensure the named-vector schema exists before making any hosted embedding
    # requests or attempting to update existing Qdrant points.
    ensure_contextual_dense_vector(
        active_client
    )

    chunks = load_chunks()

    selected_chunks = (
        chunks[:limit]
        if limit is not None
        else chunks
    )

    print(
        f"Selected {len(selected_chunks)} "
        f"of {len(chunks)} chunks for contextual backfill."
    )

    stats = backfill_contextual_vectors(
        client=active_client,
        chunks=selected_chunks,
        embedding_service=active_embedding_service,
        batch_size=batch_size,
    )

    print(
        "Contextual backfill complete: "
        f"updated={stats.updated}, "
        f"skipped={stats.skipped}."
    )

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line options for smoke tests and full migrations."""

    parser = argparse.ArgumentParser(
        description=(
            "Backfill dense_contextual vectors "
            "without replacing existing point data."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N chunks. "
            "Use --limit 5 for the initial smoke test."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of existing points to process per migration batch."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute contextual-vector backfill from the command line."""

    args = parse_args()

    run_contextual_backfill(
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()