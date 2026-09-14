"""Hosted multilingual E5 embedding service using Hugging Face inference.

This provider implements the same EmbeddingService contract as the local
Sentence Transformers implementation, but model execution happens remotely.
That keeps the NepalGov AI pipeline usable on Windows systems where local
PyTorch native libraries cannot be loaded.
"""

import os
from collections.abc import Sequence
from typing import Any

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


class HuggingFaceE5EmbeddingService(
    EmbeddingService
):
    """Remote embedding provider for multilingual-e5-large-instruct."""

    def __init__(
        self,
        token: str | None = None,
        model_name: str = MODEL_NAME,
        batch_size: int = 16,
    ) -> None:
        """Configure remote E5 inference without making a network request."""

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
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

    def _embed_batch(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        """Request one normalized batch of embeddings from Hugging Face."""

        client = self._get_client()

        try:
            result = client.feature_extraction(
                text=texts,
                model=self.model_name,
                normalize=True,
            )

        except Exception as exc:
            # Convert provider-specific failures into a stable service-level
            # error so callers do not depend on Hugging Face exception types.
            raise RuntimeError(
                "Hosted E5 embedding request failed."
            ) from exc

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

        # Batching prevents one request containing the complete 2,276-chunk
        # corpus and gives us a controllable request size.
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