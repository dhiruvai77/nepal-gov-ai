"""Unit tests for Qdrant collection and payload-index configuration."""

from unittest.mock import Mock, call

from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
)

from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    KEYWORD_PAYLOAD_FIELDS,
    ensure_collection,
    ensure_payload_indexes,
)


def test_ensure_payload_indexes_creates_keyword_indexes() -> None:
    """Every configured filter field should receive a keyword index."""

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

    assert client.create_payload_index.call_count == len(
        KEYWORD_PAYLOAD_FIELDS
    )

    client.create_payload_index.assert_has_calls(
        expected_calls
    )


def test_ensure_collection_creates_missing_collection() -> None:
    """A missing collection should be created with the dense-vector schema."""

    client = Mock()

    client.collection_exists.return_value = False

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

    assert create_call.kwargs[
        "collection_name"
    ] == COLLECTION_NAME

    vectors_config = create_call.kwargs[
        "vectors_config"
    ]

    assert DENSE_VECTOR_NAME in vectors_config
    assert (
        vectors_config[DENSE_VECTOR_NAME].size
        == DENSE_VECTOR_SIZE
    )
    assert (
        vectors_config[DENSE_VECTOR_NAME].distance
        == Distance.COSINE
    )


def test_ensure_collection_does_not_recreate_existing_collection() -> None:
    """An existing collection should not be destructively recreated."""

    client = Mock()

    client.collection_exists.return_value = True

    ensure_collection(
        client
    )

    client.create_collection.assert_not_called()


def test_existing_collection_still_gets_payload_indexes() -> None:
    """Schema evolution must apply indexes to collections already on disk."""

    client = Mock()

    client.collection_exists.return_value = True

    ensure_collection(
        client
    )

    assert client.create_payload_index.call_count == len(
        KEYWORD_PAYLOAD_FIELDS
    )


def test_new_collection_also_gets_payload_indexes() -> None:
    """Fresh installations should receive indexes immediately after creation."""

    client = Mock()

    client.collection_exists.return_value = False

    ensure_collection(
        client
    )

    assert client.create_payload_index.call_count == len(
        KEYWORD_PAYLOAD_FIELDS
    )