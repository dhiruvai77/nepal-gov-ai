"""Tests for production dependency-readiness checks."""

from __future__ import annotations

from src.api.readiness import (
    check_production_dependencies,
)
from src.embeddings.hf_e5_service import (
    HF_TOKEN_ENV,
)
from src.generation.gemini_service import (
    GEMINI_API_KEY_ENV,
)
from src.reranking.hf_bge_reranker import (
    HF_RERANKER_ENDPOINT_ENV,
)


class FakeQdrantClient:
    """Minimal deterministic client for readiness tests."""

    def __init__(
        self,
        *,
        collection_exists: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.collection_exists_value = (
            collection_exists
        )

        self.error = (
            error
        )

        self.closed = False

    def get_collections(
        self,
    ):
        """Simulate one non-mutating connectivity check."""

        if (
            self.error
            is not None
        ):
            raise self.error

        return object()

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """Return configured collection state."""

        assert (
            collection_name
            == "nepal_gov_documents"
        )

        return (
            self.collection_exists_value
        )

    def close(
        self,
    ) -> None:
        """Record cleanup."""

        self.closed = True


def configure_environment(
    monkeypatch,
) -> None:
    """Set non-secret placeholder values for readiness tests."""

    monkeypatch.setenv(
        GEMINI_API_KEY_ENV,
        "configured",
    )

    monkeypatch.setenv(
        HF_TOKEN_ENV,
        "configured",
    )

    monkeypatch.setenv(
        HF_RERANKER_ENDPOINT_ENV,
        "https://example.invalid",
    )


def test_all_dependencies_ready(
    monkeypatch,
) -> None:
    """Configured providers and a healthy collection should be ready."""

    configure_environment(
        monkeypatch
    )

    client = (
        FakeQdrantClient()
    )

    result = (
        check_production_dependencies(
            client_factory=lambda: client,
        )
    )

    assert (
        result.ready
        is True
    )

    assert (
        result.gemini_configured
        is True
    )

    assert (
        result.hf_token_configured
        is True
    )

    assert (
        result.reranker_endpoint_configured
        is True
    )

    assert (
        result.qdrant_reachable
        is True
    )

    assert (
        result.qdrant_collection_ready
        is True
    )

    assert (
        client.closed
        is True
    )


def test_missing_provider_configuration_is_not_ready(
    monkeypatch,
) -> None:
    """Missing credentials should be reported without exposing values."""

    configure_environment(
        monkeypatch
    )

    monkeypatch.delenv(
        GEMINI_API_KEY_ENV,
        raising=False,
    )

    result = (
        check_production_dependencies(
            client_factory=(
                FakeQdrantClient
            ),
        )
    )

    assert (
        result.ready
        is False
    )

    assert (
        result.gemini_configured
        is False
    )

    assert (
        result.hf_token_configured
        is True
    )

    assert (
        result.qdrant_reachable
        is True
    )


def test_unreachable_qdrant_is_not_ready(
    monkeypatch,
) -> None:
    """Connectivity failure should not escape the readiness boundary."""

    configure_environment(
        monkeypatch
    )

    client = (
        FakeQdrantClient(
            error=RuntimeError(
                "connection failed"
            ),
        )
    )

    result = (
        check_production_dependencies(
            client_factory=lambda: client,
        )
    )

    assert (
        result.ready
        is False
    )

    assert (
        result.qdrant_reachable
        is False
    )

    assert (
        result.qdrant_collection_ready
        is False
    )

    assert (
        client.closed
        is True
    )


def test_missing_collection_is_distinct_from_unreachable_qdrant(
    monkeypatch,
) -> None:
    """Reachable Qdrant without the production collection is not ready."""

    configure_environment(
        monkeypatch
    )

    result = (
        check_production_dependencies(
            client_factory=lambda: (
                FakeQdrantClient(
                    collection_exists=False,
                )
            ),
        )
    )

    assert (
        result.ready
        is False
    )

    assert (
        result.qdrant_reachable
        is True
    )

    assert (
        result.qdrant_collection_ready
        is False
    )