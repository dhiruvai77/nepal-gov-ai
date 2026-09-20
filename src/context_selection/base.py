"""Provider-independent context-selection interfaces for NepalGov AI.

Context selection receives the reranked candidate pool and chooses which
evidence passages should be supplied to the generation layer.

Keeping selection separate from retrieval and reranking allows strategies such
as fixed top-k, token budgets, redundancy control, and source diversity to be
evaluated independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.reranking.base import (
    RerankedResult,
)


class ContextSelector(ABC):
    """Abstract interface implemented by context-selection strategies."""

    @abstractmethod
    def select(
        self,
        candidates: Sequence[
            RerankedResult
        ],
    ) -> list[RerankedResult]:
        """Select generation context from an already reranked candidate pool."""