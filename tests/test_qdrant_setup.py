"""Unit tests for Qdrant collection and retrieval-schema configuration."""

from types import SimpleNamespace
from unittest.mock import Mock, call

from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorConfig,
    SparseVectorNameConfig,
)

from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    KEYWORD_PAYLOAD_FIELDS,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    ensure_payload_indexes,
    ensure_sparse_vector,
)


def collection_info_with_sparse_vectors(
    sparse_vectors: dict | None = None,
) -> SimpleNamespace:
    """Build the minimal collection-info structure required by setup tests."""

    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                sparse_vectors=sparse_vectors
            )
        )
    )


def test_ensure_payload_indexes_creates_keyword_indexes() -> None:
    """Every configured retrieval-filter field should receive a keyword index."""

    client = Mock()

    ensure_payload_indexes(
        client
    )

    expected_calls = [
        call(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        for field_name in KEYWORD_PAYLOAD_FIELDS
    ]

    assert (
        client.create_payload_index.call_count
        == len(KEYWORD_PAYLOAD_FIELDS)
    )

    client.create_payload_index.assert_has_calls(
        expected_calls
    )


def test_ensure_sparse_vector_adds_missing_bm25_schema() -> None:
    """A dense-only collection should gain BM25 without being recreated."""

    client = Mock()

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            {}
        )
    )

    ensure_sparse_vector(
        client
    )

    client.get_collection.assert_called_once_with(
        collection_name=COLLECTION_NAME
    )

    client.create_vector_name.assert_called_once()

    call_kwargs = (
        client.create_vector_name.call_args.kwargs
    )

    assert (
        call_kwargs["collection_name"]
        == COLLECTION_NAME
    )

    assert (
        call_kwargs["vector_name"]
        == SPARSE_VECTOR_NAME
    )

    assert (
        call_kwargs["wait"]
        is True
    )

    vector_config = (
        call_kwargs["vector_name_config"]
    )

    assert isinstance(
        vector_config,
        SparseVectorNameConfig,
    )

    assert isinstance(
        vector_config.sparse,
        SparseVectorConfig,
    )

    assert (
        vector_config.sparse.modifier
        == Modifier.IDF
    )


def test_ensure_sparse_vector_does_not_recreate_existing_schema() -> None:
    """Repeated setup should leave an existing BM25 sparse vector unchanged."""

    client = Mock()

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            {
                SPARSE_VECTOR_NAME: object(),
            }
        )
    )

    ensure_sparse_vector(
        client
    )

    client.create_vector_name.assert_not_called()


def test_ensure_collection_creates_missing_collection() -> None:
    """A missing collection should be created with the dense-vector schema."""

    client = Mock()

    client.collection_exists.return_value = False

    # After creation, setup inspects the collection for sparse-vector schema.
    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            {}
        )
    )

    ensure_collection(
        client
    )

    client.collection_exists.assert_called_once_with(
        collection_name=COLLECTION_NAME
    )

    client.create_collection.assert_called_once()

    create_call = (
        client.create_collection.call_args
    )

    assert (
        create_call.kwargs[
            "collection_name"
        ]
        == COLLECTION_NAME
    )

    vectors_config = (
        create_call.kwargs[
            "vectors_config"
        ]
    )

    assert (
        DENSE_VECTOR_NAME
        in vectors_config
    )

    assert (
        vectors_config[
            DENSE_VECTOR_NAME
        ].size
        == DENSE_VECTOR_SIZE
    )

    assert (
        vectors_config[
            DENSE_VECTOR_NAME
        ].distance
        == Distance.COSINE
    )

    # A fresh collection should also receive the sparse-vector schema.
    client.create_vector_name.assert_called_once()


def test_ensure_collection_does_not_recreate_existing_collection() -> None:
    """Existing collections must never be destructively recreated."""

    client = Mock()

    client.collection_exists.return_value = True

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            {
                SPARSE_VECTOR_NAME: object(),
            }
        )
    )

    ensure_collection(
        client
    )

    client.create_collection.assert_not_called()


def test_existing_collection_still_gets_payload_indexes() -> None:
    """Existing collections should still receive metadata-index checks."""

    client = Mock()

    client.collection_exists.return_value = True

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            {
                SPARSE_VECTOR_NAME: object(),
            }
        )
    )

    ensure_collection(
        client
    )

    assert (
        client.create_payload_index.call_count
        == len(KEYWORD_PAYLOAD_FIELDS)
    )


def test_new_collection_also_gets_payload_indexes() -> None:
    """Fresh collections should receive metadata indexes during setup."""

    client = Mock()

    client.collection_exists.return_value = False

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            {}
        )
    )

    ensure_collection(
        client
    )

    assert (
        client.create_payload_index.call_count
        == len(KEYWORD_PAYLOAD_FIELDS)
    )

    # New collections should receive the BM25 sparse vector too.
    client.create_vector_name.assert_called_once()