"""Create and configure the Qdrant collection used by NepalGov AI retrieval."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)


# Local Qdrant instance started through Docker Compose.
QDRANT_URL = "http://localhost:6333"

# Stable collection name used by ingestion and retrieval code.
COLLECTION_NAME = "nepal_gov_documents"

# multilingual-e5-large-instruct uses 1024-dimensional dense embeddings.
# The embedding service also validates this dimension when the model runtime
# becomes available.
DENSE_VECTOR_SIZE = 1024

# A named vector keeps the collection extensible for sparse/hybrid retrieval.
DENSE_VECTOR_NAME = "dense"

# These metadata fields are expected to appear frequently in retrieval filters.
# Keyword indexes improve filtering without indexing large text payloads such as
# chunk_text, which should remain stored but not treated as filter metadata.
KEYWORD_PAYLOAD_FIELDS = (
    "document_id",
    "language",
    "category",
    "document_type",
    "organization",
)


def create_client() -> QdrantClient:
    """Create a client connected to the local Qdrant instance."""

    return QdrantClient(
        url=QDRANT_URL
    )


def ensure_payload_indexes(
    client: QdrantClient,
) -> None:
    """Create keyword indexes used by metadata-filtered retrieval.

    Qdrant's payload-index operation is safe to request repeatedly for the same
    field/schema, which allows project setup to remain reproducible.
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
    """Ensure the retrieval collection and its payload indexes exist."""

    collection_exists = client.collection_exists(
        collection_name=COLLECTION_NAME
    )

    if not collection_exists:
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
            f"Created collection: {COLLECTION_NAME}"
        )
    else:
        print(
            f"Collection already exists: {COLLECTION_NAME}"
        )

    # Index configuration must run for both new and existing collections.
    # This is especially important during schema evolution because the current
    # NepalGov AI collection was created before payload indexes were introduced.
    ensure_payload_indexes(
        client
    )

    print(
        "Payload indexes ensured for: "
        + ", ".join(KEYWORD_PAYLOAD_FIELDS)
    )


def main() -> None:
    """Validate connectivity and ensure the retrieval schema exists."""

    client = create_client()

    # Fetching collections provides a lightweight connectivity check before
    # attempting collection or payload-schema operations.
    collections = client.get_collections()

    print(
        f"Connected to Qdrant. "
        f"Existing collections: {len(collections.collections)}"
    )

    ensure_collection(
        client
    )


if __name__ == "__main__":
    main()