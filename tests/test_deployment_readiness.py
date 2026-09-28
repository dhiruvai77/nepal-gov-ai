"""Tests for NepalGov AI container and deployment configuration."""

from __future__ import annotations

from pathlib import (
    Path,
)
from unittest.mock import (
    Mock,
    patch,
)

from src.indexing.qdrant_setup import (
    DEFAULT_QDRANT_URL,
    QDRANT_API_KEY_ENV,
    QDRANT_URL_ENV,
    create_client,
    resolve_qdrant_url,
)
from src.retrieval.run_hybrid_retrieval import (
    run_hybrid_retrieval,
)


REPOSITORY_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)


def test_resolve_qdrant_url_uses_local_default(
    monkeypatch,
) -> None:
    """Local development should retain the established localhost default."""

    monkeypatch.delenv(
        QDRANT_URL_ENV,
        raising=False,
    )

    assert (
        resolve_qdrant_url()
        == DEFAULT_QDRANT_URL
    )


def test_resolve_qdrant_url_uses_environment(
    monkeypatch,
) -> None:
    """Deployment should be able to redirect Qdrant without code changes."""

    monkeypatch.setenv(
        QDRANT_URL_ENV,
        "  http://qdrant:6333/  ",
    )

    assert (
        resolve_qdrant_url()
        == "http://qdrant:6333"
    )


def test_create_client_uses_environment_configuration(
    monkeypatch,
) -> None:
    """The Qdrant factory should pass URL and optional credentials safely."""

    monkeypatch.setenv(
        QDRANT_URL_ENV,
        "http://qdrant:6333",
    )

    monkeypatch.setenv(
        QDRANT_API_KEY_ENV,
        "test-qdrant-key",
    )

    with patch(
        "src.indexing.qdrant_setup.QdrantClient"
    ) as client_class_mock:
        create_client()

    client_class_mock.assert_called_once_with(
        url="http://qdrant:6333",
        api_key="test-qdrant-key",
    )


def test_owned_qdrant_client_is_closed_after_retrieval() -> None:
    """Production retrieval should release clients it constructs itself."""

    client = (
        Mock()
    )

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval.create_client",
            return_value=client,
        ) as create_client_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ),
        patch(
            "src.retrieval.run_hybrid_retrieval.SparseRetriever"
        ),
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        hybrid_class_mock.return_value.retrieve.return_value = []

        run_hybrid_retrieval(
            query="health services",
            embedding_service=Mock(),
        )

    create_client_mock.assert_called_once_with()

    client.close.assert_called_once_with()


def test_injected_qdrant_client_is_not_closed_by_retrieval() -> None:
    """Callers should retain ownership of explicitly injected clients."""

    client = (
        Mock()
    )

    with (
        patch(
            "src.retrieval.run_hybrid_retrieval.create_client"
        ) as create_client_mock,
        patch(
            "src.retrieval.run_hybrid_retrieval.DenseRetriever"
        ),
        patch(
            "src.retrieval.run_hybrid_retrieval.SparseRetriever"
        ),
        patch(
            "src.retrieval.run_hybrid_retrieval.HybridRetriever"
        ) as hybrid_class_mock,
    ):
        hybrid_class_mock.return_value.retrieve.return_value = []

        run_hybrid_retrieval(
            query="health services",
            embedding_service=Mock(),
            client=client,
        )

    create_client_mock.assert_not_called()

    client.close.assert_not_called()


def test_runtime_requirements_cover_production_without_local_torch_stack(
) -> None:
    """The container should install only dependencies needed by the API path."""

    requirements = (
        REPOSITORY_ROOT
        / "requirements-runtime.txt"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "qdrant-client[fastembed]==1.19.0"
        in requirements
    )

    assert (
        "huggingface-hub==1.31.0"
        in requirements
    )

    assert (
        "google-genai==2.24.0"
        in requirements
    )

    assert (
        "fastapi==0.116.1"
        in requirements
    )

    assert (
        "uvicorn==0.35.0"
        in requirements
    )

    assert (
        "torch=="
        not in requirements
    )

    assert (
        "sentence-transformers=="
        not in requirements
    )

    assert (
        "transformers=="
        not in requirements
    )

    assert (
        "pytest=="
        not in requirements
    )


def test_dockerfile_runs_the_api_as_non_root() -> None:
    """The production image should use a lean runtime and non-root user."""

    dockerfile = (
        REPOSITORY_ROOT
        / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "FROM python:3.12-slim"
        in dockerfile
    )

    assert (
        "requirements-runtime.txt"
        in dockerfile
    )

    assert (
        "USER appuser"
        in dockerfile
    )

    assert (
        "src.api.app:app"
        in dockerfile
    )

    assert (
        "--host 0.0.0.0"
        in dockerfile
    )


def test_compose_wires_app_to_qdrant_and_external_providers() -> None:
    """Compose should preserve Qdrant storage and keep secrets external."""

    compose = (
        REPOSITORY_ROOT
        / "docker-compose.yml"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "qdrant/qdrant:v1.19.1"
        in compose
    )

    assert (
        "qdrant_storage:/qdrant/storage"
        in compose
    )

    assert (
        "QDRANT_URL: http://qdrant:6333"
        in compose
    )

    assert (
        "${GEMINI_API_KEY:?"
        in compose
    )

    assert (
        "${HF_TOKEN:?"
        in compose
    )

    assert (
        "${HF_RERANKER_ENDPOINT_URL:?"
        in compose
    )

    assert (
        "${APP_PORT:-8000}:8000"
        in compose
    )


def test_environment_example_contains_no_secret_values() -> None:
    """Tracked deployment configuration should never contain credentials."""

    environment_example = (
        REPOSITORY_ROOT
        / ".env.example"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "GEMINI_API_KEY=\n"
        in environment_example
    )

    assert (
        "HF_TOKEN=\n"
        in environment_example
    )

    assert (
        "HF_RERANKER_ENDPOINT_URL=\n"
        in environment_example
    )

    assert (
        "QDRANT_API_KEY=\n"
        in environment_example
    )

    assert (
        "QDRANT_URL=http://localhost:6333"
        in environment_example
    )


def test_dockerignore_excludes_local_and_secret_material() -> None:
    """Docker build context should exclude development and secret material."""

    dockerignore = (
        REPOSITORY_ROOT
        / ".dockerignore"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        ".venv"
        in dockerignore
    )

    assert (
        ".env"
        in dockerignore
    )

    assert (
        "data"
        in dockerignore
    )

    assert (
        "tests"
        in dockerignore
    )

    assert (
        "!.env.example"
        in dockerignore
    )