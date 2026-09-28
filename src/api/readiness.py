"""Production dependency-readiness checks for NepalGov AI.

Readiness checks are intentionally lightweight.

They verify:

- required provider configuration is present,
- Qdrant is reachable,
- the production Qdrant collection exists.

They do not invoke hosted embedding, reranking, or generation models. Health
checks should not incur model inference cost or mutate external systems.
"""

from __future__ import annotations

import os
from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
)
from typing import (
    Protocol,
)

from src.embeddings.hf_e5_service import (
    HF_TOKEN_ENV,
)
from src.generation.gemini_service import (
    GEMINI_API_KEY_ENV,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    create_client,
)
from src.reranking.hf_bge_reranker import (
    HF_RERANKER_ENDPOINT_ENV,
)


class QdrantReadinessClient(
    Protocol
):
    """Minimal Qdrant client contract required by readiness checks."""

    def get_collections(
        self,
    ):
        """Return collection metadata."""

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """Return whether one collection exists."""

    def close(
        self,
    ) -> None:
        """Release client resources."""


QdrantClientFactory = Callable[
    [],
    QdrantReadinessClient,
]


@dataclass(
    frozen=True
)
class DependencyReadiness:
    """Non-secret dependency readiness state."""

    gemini_configured: bool
    hf_token_configured: bool
    reranker_endpoint_configured: bool
    qdrant_reachable: bool
    qdrant_collection_ready: bool

    @property
    def ready(
        self,
    ) -> bool:
        """Return whether every required production dependency is ready."""

        return all(
            (
                self.gemini_configured,
                self.hf_token_configured,
                self.reranker_endpoint_configured,
                self.qdrant_reachable,
                self.qdrant_collection_ready,
            )
        )


def _environment_value_is_configured(
    name: str,
) -> bool:
    """Return whether an environment variable contains usable text."""

    value = os.getenv(
        name
    )

    return (
        isinstance(
            value,
            str,
        )
        and bool(
            value.strip()
        )
    )


def check_production_dependencies(
    *,
    client_factory: QdrantClientFactory = (
        create_client
    ),
) -> DependencyReadiness:
    """Check production configuration and Qdrant without model inference."""

    gemini_configured = (
        _environment_value_is_configured(
            GEMINI_API_KEY_ENV
        )
    )

    hf_token_configured = (
        _environment_value_is_configured(
            HF_TOKEN_ENV
        )
    )

    reranker_endpoint_configured = (
        _environment_value_is_configured(
            HF_RERANKER_ENDPOINT_ENV
        )
    )

    client: (
        QdrantReadinessClient
        | None
    ) = None

    qdrant_reachable = False
    qdrant_collection_ready = False

    try:
        client = (
            client_factory()
        )

        # An explicit read verifies connectivity without modifying schema.
        client.get_collections()

        qdrant_reachable = True

        qdrant_collection_ready = bool(
            client.collection_exists(
                COLLECTION_NAME
            )
        )

    except Exception:
        # Readiness is intentionally represented as state rather than allowing
        # infrastructure exceptions to prevent the health endpoint responding.
        qdrant_reachable = False
        qdrant_collection_ready = False

    finally:
        if (
            client
            is not None
        ):
            close_method = getattr(
                client,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                try:
                    close_method()

                except Exception:
                    # Failure while closing a short-lived health client should
                    # not replace the actual dependency readiness result.
                    pass

    return DependencyReadiness(
        gemini_configured=(
            gemini_configured
        ),
        hf_token_configured=(
            hf_token_configured
        ),
        reranker_endpoint_configured=(
            reranker_endpoint_configured
        ),
        qdrant_reachable=(
            qdrant_reachable
        ),
        qdrant_collection_ready=(
            qdrant_collection_ready
        ),
    )