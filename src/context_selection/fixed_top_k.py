"""Fixed top-k context selection for NepalGov AI.

This is the simplest context-selection baseline: preserve reranker order and
select the first k passages.

It intentionally performs no token budgeting, deduplication, source balancing,
or score thresholding. Those strategies should be compared against this
baseline rather than being introduced without measured evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.context_selection.base import (
    ContextSelector,
)
from src.reranking.base import (
    RerankedResult,
)


DEFAULT_CONTEXT_COUNT = 5


class FixedTopKContextSelector(
    ContextSelector
):
    """Select the first k passages from an already reranked candidate list."""

    def __init__(
        self,
        top_k: int = DEFAULT_CONTEXT_COUNT,
    ) -> None:
        """Configure the maximum number of passages supplied as context."""

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.top_k = top_k

    def select(
        self,
        candidates: Sequence[
            RerankedResult
        ],
    ) -> list[RerankedResult]:
        """Return up to k candidates while preserving reranker order."""

        if not candidates:
            return []

        return list(
            candidates[
                :self.top_k
            ]
        )