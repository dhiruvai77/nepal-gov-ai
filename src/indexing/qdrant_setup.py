"""Create the Qdrant collection used by NepalGov AI retrieval."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# Local Qdrant instance started through Docker Compose.
QDRANT_URL = "http://localhost:6333"

# Stable collection name used by ingestion and retrieval code.
COLLECTION_NAME = "nepal_gov_documents"

# multilingual-e5-large-instruct uses 1024-dimensional dense embeddings.
# We will verify this again once model execution is available.
DENSE_VECTOR_SIZE = 1024

# A named vector keeps the schema extensible for sparse retrieval later.
DENSE_VECTOR_NAME = "dense"


def create_client() -> QdrantClient:
    """Create a client connected to the local Qdrant instance."""

    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection only if it does not already exist."""

    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection already exists: {COLLECTION_NAME}")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=Distance.COSINE,
            )
        },
    )

    print(f"Created collection: {COLLECTION_NAME}")


def main() -> None:
    """Validate connectivity and ensure the retrieval collection exists."""

    client = create_client()

    # Fetching collections gives us a simple connectivity check before
    # attempting any schema operation.
    collections = client.get_collections()

    print(
        f"Connected to Qdrant. "
        f"Existing collections: {len(collections.collections)}"
    )

    ensure_collection(client)


if __name__ == "__main__":
    main()