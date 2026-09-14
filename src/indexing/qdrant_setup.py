"""Create and configure the Qdrant collection used by NepalGov AI retrieval."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    DenseVectorConfig,
    DenseVectorNameConfig,
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorConfig,
    SparseVectorNameConfig,
    VectorParams,
)


# Local Qdrant instance started through Docker Compose.
QDRANT_URL = "http://localhost:6333"

# Stable collection name shared by ingestion and retrieval code.
COLLECTION_NAME = "nepal_gov_documents"

# multilingual-e5-large-instruct produces 1024-dimensional embeddings.
# Both raw and contextual dense representations use the same E5 model.
DENSE_VECTOR_SIZE = 1024

# Existing raw child-chunk embedding used by the production baseline.
DENSE_VECTOR_NAME = "dense"

# Experimental metadata-enriched dense representation.
#
# Keeping this separate from `dense` preserves the existing baseline so we can
# run a controlled raw-vs-contextual retrieval evaluation before promoting the
# contextual representation into production.
CONTEXTUAL_DENSE_VECTOR_NAME = "dense_contextual"

# Named sparse vector used for BM25-style lexical retrieval.
SPARSE_VECTOR_NAME = "bm25"

# These metadata fields are expected to be used frequently as exact-match
# retrieval filters, so they receive Qdrant keyword payload indexes.
KEYWORD_PAYLOAD_FIELDS = (
    "document_id",
    "language",
    "category",
    "document_type",
    "organization",
)


def create_client() -> QdrantClient:
    """Create a client connected to the local Qdrant service."""

    return QdrantClient(
        url=QDRANT_URL
    )


def ensure_sparse_vector(
    client: QdrantClient,
) -> None:
    """Ensure the collection contains the named BM25 sparse vector.

    Qdrant requires ``create_vector_name()`` when introducing a new named
    vector to an existing collection. The collection itself is never recreated.
    """

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    # Collections created before sparse retrieval support may report None here.
    # Treat that as an empty mapping so the existence check remains simple.
    sparse_vectors = (
        collection_info.config.params.sparse_vectors
        or {}
    )

    if SPARSE_VECTOR_NAME in sparse_vectors:
        print(
            "Sparse vector already exists: "
            f"{SPARSE_VECTOR_NAME}"
        )
        return

    # Modifier.IDF enables document-frequency weighting required by Qdrant's
    # BM25-style sparse retrieval.
    client.create_vector_name(
        collection_name=COLLECTION_NAME,
        vector_name=SPARSE_VECTOR_NAME,
        vector_name_config=SparseVectorNameConfig(
            sparse=SparseVectorConfig(
                modifier=Modifier.IDF,
            )
        ),
        wait=True,
    )

    print(
        f"Created sparse vector: "
        f"{SPARSE_VECTOR_NAME}"
    )


def ensure_contextual_dense_vector(
    client: QdrantClient,
) -> None:
    """Ensure the collection contains the contextual dense vector schema.

    Existing points do not automatically receive values for a newly created
    vector name. A separate backfill step will generate and attach the
    contextual embeddings after the schema has been validated.
    """

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    dense_vectors = (
        collection_info.config.params.vectors
    )

    # NepalGov AI uses named dense vectors. Failing explicitly here is safer
    # than attempting a migration against an unexpected unnamed-vector schema.
    if not isinstance(
        dense_vectors,
        dict,
    ):
        raise RuntimeError(
            "Expected Qdrant collection to use named dense vectors."
        )

    if (
        CONTEXTUAL_DENSE_VECTOR_NAME
        in dense_vectors
    ):
        print(
            "Contextual dense vector already exists: "
            f"{CONTEXTUAL_DENSE_VECTOR_NAME}"
        )
        return

    # Adding a named vector is non-destructive. Existing raw dense and BM25
    # representations remain untouched while this new vector is introduced.
    client.create_vector_name(
        collection_name=COLLECTION_NAME,
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
        vector_name_config=DenseVectorNameConfig(
            dense=DenseVectorConfig(
                size=DENSE_VECTOR_SIZE,
                distance=Distance.COSINE,
            )
        ),
        wait=True,
    )

    print(
        "Created contextual dense vector: "
        f"{CONTEXTUAL_DENSE_VECTOR_NAME}"
    )


def ensure_payload_indexes(
    client: QdrantClient,
) -> None:
    """Create keyword indexes used by metadata-filtered retrieval.

    Repeating these calls is intentional: setup should be safe to rerun and
    should also apply indexes to collections created by older project versions.
    """

    for field_name in KEYWORD_PAYLOAD_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )


def ensure_collection(
    client: QdrantClient,
) -> None:
    """Ensure the collection and all supporting retrieval schema exist."""

    collection_exists = client.collection_exists(
        collection_name=COLLECTION_NAME
    )

    if not collection_exists:
        # New collections begin with both dense representations so future fresh
        # environments do not require a separate schema-migration operation.
        #
        # BM25 is still created separately through create_vector_name() because
        # that is the schema-evolution path already used by this project.
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
                CONTEXTUAL_DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    else:
        # Never recreate an existing collection because it already contains
        # indexed points, payloads, and retrieval vectors.
        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )

    # Run these checks for both new and existing collections. This makes setup
    # idempotent and upgrades collections created by previous project versions.
    ensure_sparse_vector(
        client
    )

    ensure_contextual_dense_vector(
        client
    )

    # Metadata indexes support efficient exact-match filters for dense, sparse,
    # and future retrieval strategies.
    ensure_payload_indexes(
        client
    )

    print(
        "Payload indexes ensured for: "
        + ", ".join(KEYWORD_PAYLOAD_FIELDS)
    )


def main() -> None:
    """Validate connectivity and ensure the complete Qdrant schema exists."""

    client = create_client()

    # Listing collections gives us an explicit connectivity check before any
    # schema mutation is attempted.
    collections = client.get_collections()

    print(
        f"Connected to Qdrant. "
        f"Existing collections: "
        f"{len(collections.collections)}"
    )

    ensure_collection(
        client
    )


if __name__ == "__main__":
    main()