"""Hosted multilingual E5 embedding service using Hugging Face inference.

This provider implements the same EmbeddingService contract as the local
Sentence Transformers implementation, but model execution happens remotely.
That keeps the NepalGov AI pipeline usable on Windows systems where local
PyTorch native libraries cannot be loaded.
"""

import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from src.embeddings.base import (
    EmbeddingService,
    EmbeddingVector,
)
from src.embeddings.e5_service import (
    EXPECTED_DIMENSION,
    MODEL_NAME,
    QUERY_INSTRUCTION,
)


HF_TOKEN_ENV = "HF_TOKEN"

# Explicitly select Hugging Face inference rather than relying on a local model
# runtime. Provider routing can be revisited later if evaluation or cost
# requirements justify another hosted backend.
HF_PROVIDER = "hf-inference"

# Three total attempts provide resilience against short-lived network or
# provider interruptions without allowing requests to retry indefinitely.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0

# These HTTP status codes represent failures that are commonly temporary and
# therefore reasonable to retry after a short delay.
RETRYABLE_HTTP_STATUS_CODES = {
    408,  # Request Timeout
    425,  # Too Early
    429,  # Too Many Requests
}


class HuggingFaceE5EmbeddingService(
    EmbeddingService
):
    """Remote embedding provider for multilingual-e5-large-instruct."""

    def __init__(
        self,
        token: str | None = None,
        model_name: str = MODEL_NAME,
        batch_size: int = 16,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_base_delay_seconds: float = (
            DEFAULT_RETRY_BASE_DELAY_SECONDS
        ),
    ) -> None:
        """Configure remote E5 inference without making a network request."""

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be greater than zero."
            )

        if retry_base_delay_seconds < 0:
            raise ValueError(
                "retry_base_delay_seconds cannot be negative."
            )

        # Allow explicit dependency injection in tests while using HF_TOKEN for
        # the production command-line workflow.
        resolved_token = (
            token
            if token is not None
            else os.getenv(HF_TOKEN_ENV)
        )

        if not resolved_token:
            raise ValueError(
                f"{HF_TOKEN_ENV} must be set for hosted embeddings."
            )

        self.token = resolved_token
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.retry_base_delay_seconds = (
            retry_base_delay_seconds
        )

        # Keep the client lazy so constructing the service remains cheap and
        # tests can replace it without performing external network requests.
        self._client: Any | None = None

    @property
    def dimension(self) -> int:
        """Return the dimensionality required by the Qdrant dense schema."""

        return EXPECTED_DIMENSION

    @staticmethod
    def format_query(
        query: str,
    ) -> str:
        """Format retrieval queries using the E5-instruct convention."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Query must contain non-whitespace text."
            )

        # Match the local E5 service exactly so local and hosted embeddings use
        # the same query semantics.
        return (
            f"Instruct: {QUERY_INSTRUCTION}\n"
            f"Query: {clean_query}"
        )

    @staticmethod
    def _validate_passages(
        texts: Sequence[str],
    ) -> list[str]:
        """Validate and normalize passage inputs before remote inference."""

        if not texts:
            raise ValueError(
                "At least one passage is required."
            )

        cleaned_texts: list[str] = []

        for index, text in enumerate(
            texts
        ):
            clean_text = text.strip()

            if not clean_text:
                raise ValueError(
                    f"Passage at index {index} is empty."
                )

            cleaned_texts.append(
                clean_text
            )

        return cleaned_texts

    def _get_client(
        self,
    ) -> Any:
        """Create the Hugging Face inference client only when required."""

        if self._client is not None:
            return self._client

        try:
            # Import lazily so modules depending on the embedding abstraction
            # do not require hosted-inference dependencies during import.
            from huggingface_hub import InferenceClient

        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for hosted embeddings."
            ) from exc

        self._client = InferenceClient(
            provider=HF_PROVIDER,
            token=self.token,
        )

        return self._client

    @staticmethod
    def _convert_embeddings(
        result: Any,
    ) -> list[EmbeddingVector]:
        """Convert hosted inference output into plain Python float lists."""

        # InferenceClient currently returns a NumPy array for feature
        # extraction. Using tolist() keeps NumPy out of the public interface.
        if hasattr(
            result,
            "tolist",
        ):
            result = result.tolist()

        if not isinstance(
            result,
            list,
        ):
            raise RuntimeError(
                "Unexpected embedding response type."
            )

        # A one-text response may occasionally be represented as one flat
        # vector, so normalize it into the same batch shape used elsewhere.
        if result and isinstance(
            result[0],
            (int, float),
        ):
            result = [
                result
            ]

        return result

    @staticmethod
    def _extract_status_code(
        exc: BaseException,
    ) -> int | None:
        """Extract an HTTP status code from an exception or its cause chain."""

        current: BaseException | None = exc

        # Hugging Face HTTP exceptions typically expose a response object.
        # Inspecting the chained causes also handles wrapped HTTP failures.
        while current is not None:
            response = getattr(
                current,
                "response",
                None,
            )

            status_code = getattr(
                response,
                "status_code",
                None,
            )

            if isinstance(
                status_code,
                int,
            ):
                return status_code

            current = current.__cause__

        return None

    @classmethod
    def _is_retryable_exception(
        cls,
        exc: BaseException,
    ) -> bool:
        """Return whether a hosted inference failure should be retried."""

        status_code = cls._extract_status_code(
            exc
        )

        if status_code is not None:
            # Retry rate limits and server-side failures. Client/authentication
            # failures such as 400, 401, 403, and 404 should fail immediately.
            return (
                status_code
                in RETRYABLE_HTTP_STATUS_CODES
                or 500 <= status_code <= 599
            )

        # Transport failures have no HTTP response because the request failed
        # before a valid response was received. The RemoteProtocolError seen
        # during full ingestion is one example and is safe to retry.
        current: BaseException | None = exc

        while current is not None:
            if isinstance(
                current,
                httpx.TransportError,
            ):
                return True

            current = current.__cause__

        # Unknown application/provider errors are not automatically retried.
        # This avoids repeating requests that are unlikely to succeed.
        return False

    def _request_embeddings(
        self,
        texts: list[str],
    ) -> Any:
        """Execute one hosted request with bounded transient-error retries."""

        client = self._get_client()
        last_exception: Exception | None = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            try:
                return client.feature_extraction(
                    text=texts,
                    model=self.model_name,
                    normalize=True,
                )

            except Exception as exc:
                last_exception = exc

                is_retryable = (
                    self._is_retryable_exception(
                        exc
                    )
                )

                # Permanent failures and the final allowed attempt terminate
                # immediately and preserve the original exception as the cause.
                if (
                    not is_retryable
                    or attempt == self.max_attempts
                ):
                    break

                # Exponential backoff produces 1s, 2s, 4s... delays depending
                # on the configured base delay and number of attempts.
                delay_seconds = (
                    self.retry_base_delay_seconds
                    * (2 ** (attempt - 1))
                )

                time.sleep(
                    delay_seconds
                )

        raise RuntimeError(
            "Hosted E5 embedding request failed "
            f"after {self.max_attempts if self._is_retryable_exception(last_exception) else 1} "
            "attempt(s)."
        ) from last_exception

    def _embed_batch(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """Request one normalized batch of embeddings from Hugging Face."""

        result = self._request_embeddings(
            texts
        )

        embeddings = self._convert_embeddings(
            result
        )

        if len(
            embeddings
        ) != len(
            texts
        ):
            raise RuntimeError(
                "Hosted embedding service returned a different "
                "number of vectors than input texts."
            )

        for vector in embeddings:
            if len(
                vector
            ) != EXPECTED_DIMENSION:
                raise RuntimeError(
                    "Unexpected hosted embedding dimension: "
                    f"expected {EXPECTED_DIMENSION}, "
                    f"received {len(vector)}."
                )

        return embeddings

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Embed passages remotely in bounded batches."""

        passages = self._validate_passages(
            texts
        )

        embeddings: list[
            EmbeddingVector
        ] = []

        # Batching prevents one request containing the complete corpus and
        # provides a controllable unit for retrying transient provider errors.
        for start_index in range(
            0,
            len(passages),
            self.batch_size,
        ):
            batch = passages[
                start_index:
                start_index + self.batch_size
            ]

            embeddings.extend(
                self._embed_batch(
                    batch
                )
            )

        return embeddings

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Embed one E5-formatted retrieval query remotely."""

        formatted_query = self.format_query(
            query
        )

        embeddings = self._embed_batch(
            [
                formatted_query,
            ]
        )

        return embeddings[0]