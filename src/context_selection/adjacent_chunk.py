"""Adjacent-chunk redundancy control for NepalGov AI context selection.

Baseline retrieval chunks intentionally contain overlap with neighboring chunks.
When adjacent chunks from the same document both rank highly, sending both to
the generation model can duplicate evidence.

This selector preserves reranker order while skipping candidates that are
immediately adjacent to an already selected chunk from the same document. It
continues down the reranked list until the requested number of passages has
been selected or the candidate pool is exhausted.

The strategy intentionally addresses only known chunk-boundary overlap.
Semantic deduplication is not performed here.
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


class AdjacentChunkContextSelector(
    ContextSelector
):
    """Select reranked passages while suppressing adjacent chunk overlap."""

    def __init__(
        self,
        top_k: int = DEFAULT_CONTEXT_COUNT,
    ) -> None:
        """Configure the maximum number of selected evidence passages."""

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.top_k = top_k

    @staticmethod
    def _is_adjacent_to_selected(
        candidate: RerankedResult,
        selected: Sequence[
            RerankedResult
        ],
    ) -> bool:
        """Return True when a candidate overlaps an adjacent selected chunk.

        Adjacency is considered only inside the same source document.

        Missing chunk-index metadata does not cause a candidate to be removed,
        because redundancy cannot be established reliably in that case.
        """

        candidate_result = (
            candidate.result
        )

        candidate_index = (
            candidate_result.chunk_index
        )

        if candidate_index is None:
            return False

        for selected_item in selected:
            selected_result = (
                selected_item.result
            )

            if (
                selected_result.document_id
                != candidate_result.document_id
            ):
                continue

            selected_index = (
                selected_result.chunk_index
            )

            if selected_index is None:
                continue

            # Consecutive baseline chunks may share the configured overlap
            # introduced during chunk construction.
            if (
                abs(
                    candidate_index
                    - selected_index
                )
                <= 1
            ):
                return True

        return False

    def select(
        self,
        candidates: Sequence[
            RerankedResult
        ],
    ) -> list[RerankedResult]:
        """Select non-adjacent evidence while preserving reranker priority."""

        selected: list[
            RerankedResult
        ] = []

        for candidate in candidates:
            if self._is_adjacent_to_selected(
                candidate,
                selected,
            ):
                continue

            selected.append(
                candidate
            )

            if (
                len(selected)
                >= self.top_k
            ):
                break

        return selected