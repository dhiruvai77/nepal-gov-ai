"""Token-budget context selection for NepalGov AI.

The selector walks the reranked candidate list in relevance order and keeps
adding passages while the configured evidence-token budget permits.

It uses the tokenizer-derived token counts already stored during chunking and
preserved through Qdrant retrieval. No downstream retokenization is required.

This selector intentionally does not perform redundancy filtering yet. Token
budgeting and redundancy control should be evaluated independently.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.context_selection.base import (
    ContextSelector,
)
from src.reranking.base import (
    RerankedResult,
)


DEFAULT_CONTEXT_TOKEN_BUDGET = 1800


class TokenBudgetContextSelector(
    ContextSelector
):
    """Select reranked passages without exceeding an evidence-token budget."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        """Configure the maximum number of source-passage tokens."""

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self.max_tokens = max_tokens

    def select(
        self,
        candidates: Sequence[
            RerankedResult
        ],
    ) -> list[RerankedResult]:
        """Select the highest-ranked prefix that fits within the token budget.

        Selection stops when the next ranked passage would exceed the budget.
        Lower-ranked passages are not allowed to leapfrog that candidate merely
        because they happen to contain fewer tokens.

        This keeps context selection faithful to the reranker's relevance
        ordering and makes the strategy deterministic.
        """

        selected: list[
            RerankedResult
        ] = []

        used_tokens = 0

        for item in candidates:
            candidate_tokens = (
                item.result.token_count
            )

            if candidate_tokens is None:
                raise ValueError(
                    "token_count is required for token-budget "
                    f"context selection: {item.result.chunk_id}"
                )

            if candidate_tokens < 0:
                raise ValueError(
                    "token_count must not be negative: "
                    f"{item.result.chunk_id}"
                )

            if (
                used_tokens + candidate_tokens
                > self.max_tokens
            ):
                break

            selected.append(
                item
            )

            used_tokens += candidate_tokens

        return selected