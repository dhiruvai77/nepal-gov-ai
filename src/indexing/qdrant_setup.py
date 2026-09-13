"""Create and configure the Qdrant collection used by NepalGov AI retrieval."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
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

# multilingual-e5-large-instruct produces 1024-dimensional dense embeddings.
# The embedding service verifies this dimension when the real model can load.
DENSE_VECTOR_SIZE = 1024

# Named dense vector used for semantic retrieval.
DENSE_VECTOR_NAME = "dense"

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

    Qdrant requires create_vector_name() when adding a completely new vector
    name to an existing collection. update_collection() can modify an existing
    vector configuration but cannot introduce an unknown vector name.
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
            f"Sparse vector already exists: "
            f"{SPARSE_VECTOR_NAME}"
        )
        return

    # SparseVectorNameConfig is the schema used specifically when adding a new
    # named sparse vector to an existing collection.
    #
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
        # The base collection starts with the dense semantic vector.
        # The BM25 sparse vector is added immediately afterward through the same
        # schema-evolution path used for older existing collections.
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                )
            },
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    else:
        # Never recreate an existing collection because it may already contain
        # indexed vectors and document payloads.
        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )

    # This must run for both new and existing collections because sparse
    # retrieval was introduced after the initial dense-only schema.
    ensure_sparse_vector(
        client
    )

    # Metadata indexes support efficient filters for both dense and sparse
    # retrieval paths.
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

    # Listing collections provides an explicit connectivity check before any
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