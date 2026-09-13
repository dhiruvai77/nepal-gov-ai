"""Common embedding interface for NepalGov AI.

Retrieval and indexing code should depend on this abstraction rather than on a
specific model library. This keeps the pipeline portable across local models,
containerized runtimes, or future hosted embedding providers.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


# Keep the vector type simple at the service boundary. Qdrant accepts regular
# Python float lists, so downstream code does not need to know about NumPy or
# PyTorch tensor types.
EmbeddingVector = list[float]


class EmbeddingService(ABC):
    """Interface implemented by all dense embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of vectors produced by the service."""

        raise NotImplementedError

    @abstractmethod
    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Embed document passages for storage in the vector database."""

        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Embed one user query for dense retrieval."""

        raise NotImplementedError