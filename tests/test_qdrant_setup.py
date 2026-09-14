"""Unit tests for Qdrant collection and retrieval-schema configuration."""

from types import SimpleNamespace
from unittest.mock import Mock, call

from qdrant_client.models import (
    DenseVectorConfig,
    DenseVectorNameConfig,
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorConfig,
    SparseVectorNameConfig,
)

from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    KEYWORD_PAYLOAD_FIELDS,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    ensure_contextual_dense_vector,
    ensure_payload_indexes,
    ensure_sparse_vector,
)


def collection_info_with_sparse_vectors(
    sparse_vectors: dict | None = None,
    dense_vectors: dict | None = None,
) -> SimpleNamespace:
    """Build the minimal collection-info structure required by setup tests.

    Most tests model the fully migrated dense schema. Individual migration tests
    explicitly provide a raw-only dense schema when they need to verify creation
    of the new contextual vector.
    """

    if dense_vectors is None:
        dense_vectors = {
            DENSE_VECTOR_NAME: object(),
            CONTEXTUAL_DENSE_VECTOR_NAME: object(),
        }

    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
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
            sparse_vectors={},
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
            sparse_vectors={
                SPARSE_VECTOR_NAME: object(),
            }
        )
    )

    ensure_sparse_vector(
        client
    )

    client.create_vector_name.assert_not_called()


def test_ensure_contextual_dense_vector_adds_missing_schema() -> None:
    """An older collection should gain contextual dense search safely."""

    client = Mock()

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            sparse_vectors={
                SPARSE_VECTOR_NAME: object(),
            },
            dense_vectors={
                # Model an existing project collection that only has the
                # original raw dense representation.
                DENSE_VECTOR_NAME: object(),
            },
        )
    )

    ensure_contextual_dense_vector(
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
        == CONTEXTUAL_DENSE_VECTOR_NAME
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
        DenseVectorNameConfig,
    )

    assert isinstance(
        vector_config.dense,
        DenseVectorConfig,
    )

    assert (
        vector_config.dense.size
        == DENSE_VECTOR_SIZE
    )

    assert (
        vector_config.dense.distance
        == Distance.COSINE
    )


def test_ensure_contextual_dense_vector_preserves_existing_schema() -> None:
    """Repeated setup should not recreate an existing contextual vector."""

    client = Mock()

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            sparse_vectors={
                SPARSE_VECTOR_NAME: object(),
            }
        )
    )

    ensure_contextual_dense_vector(
        client
    )

    client.create_vector_name.assert_not_called()


def test_ensure_collection_creates_missing_collection() -> None:
    """A missing collection should be created with both dense schemas."""

    client = Mock()

    client.collection_exists.return_value = False

    # The mocked collection returned after creation represents the intended new
    # dense schema. BM25 remains absent so ensure_collection() should add it.
    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            sparse_vectors={},
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

    # The original raw embedding remains available as the baseline.
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

    # Fresh collections also receive the contextual dense representation.
    assert (
        CONTEXTUAL_DENSE_VECTOR_NAME
        in vectors_config
    )

    assert (
        vectors_config[
            CONTEXTUAL_DENSE_VECTOR_NAME
        ].size
        == DENSE_VECTOR_SIZE
    )

    assert (
        vectors_config[
            CONTEXTUAL_DENSE_VECTOR_NAME
        ].distance
        == Distance.COSINE
    )

    # The fresh mocked collection already contains both dense names, so only
    # the missing BM25 vector should need a create_vector_name() call.
    assert (
        client.create_vector_name.call_count
        == 1
    )

    sparse_call = (
        client.create_vector_name.call_args.kwargs
    )

    assert (
        sparse_call["vector_name"]
        == SPARSE_VECTOR_NAME
    )


def test_ensure_collection_does_not_recreate_existing_collection() -> None:
    """Existing collections must never be destructively recreated."""

    client = Mock()

    client.collection_exists.return_value = True

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            sparse_vectors={
                SPARSE_VECTOR_NAME: object(),
            }
        )
    )

    ensure_collection(
        client
    )

    client.create_collection.assert_not_called()

    # Both BM25 and contextual dense already exist in the mocked collection.
    client.create_vector_name.assert_not_called()


def test_existing_collection_adds_missing_contextual_dense_vector() -> None:
    """Existing raw-only collections should receive contextual dense schema."""

    client = Mock()

    client.collection_exists.return_value = True

    # Both setup checks call get_collection(). The collection has BM25 already
    # but does not yet contain the contextual dense representation.
    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            sparse_vectors={
                SPARSE_VECTOR_NAME: object(),
            },
            dense_vectors={
                DENSE_VECTOR_NAME: object(),
            },
        )
    )

    ensure_collection(
        client
    )

    client.create_collection.assert_not_called()

    # BM25 already exists, so the one schema mutation should be contextual dense.
    assert (
        client.create_vector_name.call_count
        == 1
    )

    vector_call = (
        client.create_vector_name.call_args.kwargs
    )

    assert (
        vector_call["vector_name"]
        == CONTEXTUAL_DENSE_VECTOR_NAME
    )


def test_existing_collection_still_gets_payload_indexes() -> None:
    """Existing collections should still receive metadata-index checks."""

    client = Mock()

    client.collection_exists.return_value = True

    client.get_collection.return_value = (
        collection_info_with_sparse_vectors(
            sparse_vectors={
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
            sparse_vectors={},
        )
    )

    ensure_collection(
        client
    )

    assert (
        client.create_payload_index.call_count
        == len(KEYWORD_PAYLOAD_FIELDS)
    )

    # New collections already contain both dense vector names, so BM25 is the
    # only additional vector-name operation required by this mocked state.
    assert (
        client.create_vector_name.call_count
        == 1
    )

    assert (
        client.create_vector_name.call_args.kwargs[
            "vector_name"
        ]
        == SPARSE_VECTOR_NAME
    )