"""Evaluate context-selection strategies on reranked NepalGov AI evidence.

The benchmark retrieves and reranks each evaluation query exactly once, then
applies multiple context-selection strategies to the cached reranked candidate
pool.

This separation is important because context-selection experiments should not
change first-stage retrieval or reranking behavior.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path

from src.context_selection.adjacent_chunk import (
    AdjacentChunkContextSelector,
)
from src.context_selection.base import (
    ContextSelector,
)
from src.context_selection.fixed_top_k import (
    FixedTopKContextSelector,
)
from src.context_selection.token_budget import (
    TokenBudgetContextSelector,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    aggregate_metrics,
    evaluate_ranked_results,
    load_evaluation_records,
)
from src.evaluation.run_retrieval_evaluation import (
    DEFAULT_DATASET_PATH,
    LANGUAGE_PAIR_ORDER,
    format_language_pair,
    retrieve_for_evaluation,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.reranking.hf_bge_reranker import (
    HuggingFaceBGEReranker,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


DEFAULT_RETRIEVAL_DEPTH = 20

DEFAULT_CONTEXT_COUNTS = (
    3,
    5,
    8,
)

DEFAULT_TOKEN_BUDGETS = (
    1400,
    1800,
    2200,
)


RetrieveFunction = Callable[
    [
        EvaluationRecord,
        int,
    ],
    Sequence[
        RetrievalResult
    ],
]


@dataclass(frozen=True)
class ContextSelectionMetrics:
    """Aggregate quality and context-size metrics for one selector."""

    hit_rate: float
    mrr: float
    recall: float
    average_passages: float
    average_tokens: float
    average_adjacent_pairs: float


def validate_context_counts(
    context_counts: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize fixed context-size experiment values."""

    if not context_counts:
        raise ValueError(
            "At least one context count is required."
        )

    normalized = tuple(
        context_counts
    )

    if any(
        count <= 0
        for count in normalized
    ):
        raise ValueError(
            "Context counts must be greater than zero."
        )

    if len(
        set(normalized)
    ) != len(normalized):
        raise ValueError(
            "Context counts must not contain duplicates."
        )

    return normalized


def validate_token_budgets(
    token_budgets: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize token-budget experiment values."""

    if not token_budgets:
        raise ValueError(
            "At least one token budget is required."
        )

    normalized = tuple(
        token_budgets
    )

    if any(
        budget <= 0
        for budget in normalized
    ):
        raise ValueError(
            "Token budgets must be greater than zero."
        )

    if len(
        set(normalized)
    ) != len(normalized):
        raise ValueError(
            "Token budgets must not contain duplicates."
        )

    return normalized


def collect_reranked_candidates(
    records: Sequence[
        EvaluationRecord
    ],
    *,
    retrieval_depth: int,
    reranker: Reranker,
    retrieve: RetrieveFunction = retrieve_for_evaluation,
) -> dict[
    str,
    list[
        RerankedResult
    ],
]:
    """Retrieve and rerank each question exactly once."""

    if retrieval_depth <= 0:
        raise ValueError(
            "retrieval_depth must be greater than zero."
        )

    collected: dict[
        str,
        list[
            RerankedResult
        ],
    ] = {}

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record.question_id}"
        )

        candidates = list(
            retrieve(
                record,
                retrieval_depth,
            )
        )

        if not candidates:
            raise RuntimeError(
                f"{record.question_id}: first-stage retrieval "
                "returned no candidates."
            )

        reranked = reranker.rerank(
            record.query,
            candidates,
            top_k=None,
        )

        if len(
            reranked
        ) != len(
            candidates
        ):
            raise RuntimeError(
                f"{record.question_id}: reranking changed "
                "candidate count."
            )

        first_stage_ids = {
            candidate.point_id
            for candidate in candidates
        }

        reranked_ids = {
            item.result.point_id
            for item in reranked
        }

        if (
            first_stage_ids
            != reranked_ids
        ):
            raise RuntimeError(
                f"{record.question_id}: reranking changed "
                "candidate identity."
            )

        if len(
            reranked_ids
        ) != len(
            reranked
        ):
            raise RuntimeError(
                f"{record.question_id}: reranking returned "
                "duplicate candidates."
            )

        collected[
            record.question_id
        ] = list(
            reranked
        )

    return collected


def selected_token_count(
    selected: Sequence[
        RerankedResult
    ],
) -> int:
    """Return total stored chunk tokens for selected evidence."""

    total = 0

    for item in selected:
        token_count = (
            item.result.token_count
        )

        if token_count is None:
            raise RuntimeError(
                "Context-selection evaluation requires "
                f"token_count for chunk {item.result.chunk_id}."
            )

        if token_count < 0:
            raise RuntimeError(
                "Context-selection evaluation received "
                f"negative token_count for chunk "
                f"{item.result.chunk_id}."
            )

        total += token_count

    return total


def count_adjacent_pairs(
    selected: Sequence[
        RerankedResult
    ],
) -> int:
    """Count adjacent same-document chunk pairs in selected context."""

    count = 0

    for first_index, first_item in enumerate(
        selected
    ):
        first_result = (
            first_item.result
        )

        if (
            first_result.chunk_index
            is None
        ):
            continue

        for second_item in selected[
            first_index + 1:
        ]:
            second_result = (
                second_item.result
            )

            if (
                second_result.chunk_index
                is None
            ):
                continue

            if (
                first_result.document_id
                != second_result.document_id
            ):
                continue

            if (
                abs(
                    first_result.chunk_index
                    - second_result.chunk_index
                )
                <= 1
            ):
                count += 1

    return count


def evaluate_selector(
    records: Sequence[
        EvaluationRecord
    ],
    reranked_candidates: dict[
        str,
        list[
            RerankedResult
        ],
    ],
    *,
    selector: ContextSelector,
) -> ContextSelectionMetrics:
    """Evaluate evidence coverage and context size for one selector."""

    if not records:
        raise ValueError(
            "At least one evaluation record is required."
        )

    query_metrics = []

    total_passages = 0
    total_tokens = 0
    total_adjacent_pairs = 0

    for record in records:
        candidates = reranked_candidates[
            record.question_id
        ]

        selected = selector.select(
            candidates
        )

        selected_results = [
            item.result
            for item in selected
        ]

        # evaluate_ranked_results safely handles a shorter result sequence.
        # k=1 is used for an empty selection so the metric becomes a genuine
        # zero rather than passing an invalid cutoff.
        metric = evaluate_ranked_results(
            record,
            selected_results,
            k=max(
                1,
                len(
                    selected_results
                ),
            ),
        )

        query_metrics.append(
            metric
        )

        total_passages += len(
            selected
        )

        total_tokens += (
            selected_token_count(
                selected
            )
        )

        total_adjacent_pairs += (
            count_adjacent_pairs(
                selected
            )
        )

    aggregate = aggregate_metrics(
        query_metrics
    )

    count = len(
        records
    )

    return ContextSelectionMetrics(
        hit_rate=aggregate[
            "hit_rate"
        ],
        mrr=aggregate[
            "mrr"
        ],
        recall=aggregate[
            "recall"
        ],
        average_passages=(
            total_passages
            / count
        ),
        average_tokens=(
            total_tokens
            / count
        ),
        average_adjacent_pairs=(
            total_adjacent_pairs
            / count
        ),
    )


def evaluate_selector_by_language_pair(
    records: Sequence[
        EvaluationRecord
    ],
    reranked_candidates: dict[
        str,
        list[
            RerankedResult
        ],
    ],
    *,
    selector: ContextSelector,
) -> dict[
    tuple[str, str],
    ContextSelectionMetrics,
]:
    """Evaluate one selector independently for each language pair."""

    grouped_records: dict[
        tuple[str, str],
        list[
            EvaluationRecord
        ],
    ] = {}

    for record in records:
        pair = (
            record.query_language,
            record.target_language,
        )

        grouped_records.setdefault(
            pair,
            [],
        ).append(
            record
        )

    return {
        pair: evaluate_selector(
            pair_records,
            reranked_candidates,
            selector=selector,
        )
        for pair, pair_records
        in grouped_records.items()
    }


def print_metrics_row(
    label: str,
    count: int,
    metrics: ContextSelectionMetrics,
) -> None:
    """Print one context-selection metric row."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{metrics.hit_rate:>7.3f} "
        f"{metrics.mrr:>7.3f} "
        f"{metrics.recall:>7.3f}  "
        f"{metrics.average_passages:>8.2f} "
        f"{metrics.average_tokens:>9.1f} "
        f"{metrics.average_adjacent_pairs:>8.2f}"
    )


def print_selector_results(
    records: Sequence[
        EvaluationRecord
    ],
    reranked_candidates: dict[
        str,
        list[
            RerankedResult
        ],
    ],
    *,
    label: str,
    selector: ContextSelector,
) -> None:
    """Print overall and multilingual context-selection metrics."""

    overall = evaluate_selector(
        records,
        reranked_candidates,
        selector=selector,
    )

    by_language = (
        evaluate_selector_by_language_pair(
            records,
            reranked_candidates,
            selector=selector,
        )
    )

    print()
    print(
        "=" * 82
    )

    print(
        label
    )

    print(
        "=" * 82
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>4}  "
        f"{'Hit':>7} "
        f"{'MRR':>7} "
        f"{'Recall':>7}  "
        f"{'Passages':>8} "
        f"{'Tokens':>9} "
        f"{'AdjPairs':>8}"
    )

    print(
        "-" * 82
    )

    for pair in LANGUAGE_PAIR_ORDER:
        metrics = by_language.get(
            pair
        )

        if metrics is None:
            continue

        pair_count = sum(
            1
            for record in records
            if (
                record.query_language,
                record.target_language,
            )
            == pair
        )

        print_metrics_row(
            format_language_pair(
                pair
            ),
            pair_count,
            metrics,
        )

    print(
        "-" * 82
    )

    print_metrics_row(
        "overall",
        len(
            records
        ),
        overall,
    )


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH,
    context_counts: Sequence[int] = DEFAULT_CONTEXT_COUNTS,
    token_budgets: Sequence[int] = DEFAULT_TOKEN_BUDGETS,
    reranker: Reranker | None = None,
) -> None:
    """Compare fixed-count, token-budget, and overlap-aware strategies."""

    normalized_counts = (
        validate_context_counts(
            context_counts
        )
    )

    normalized_budgets = (
        validate_token_budgets(
            token_budgets
        )
    )

    if retrieval_depth <= 0:
        raise ValueError(
            "retrieval_depth must be greater than zero."
        )

    if max(
        normalized_counts
    ) > retrieval_depth:
        raise ValueError(
            "Context count cannot exceed retrieval depth."
        )

    records = load_evaluation_records(
        dataset_path
    )

    if not records:
        raise ValueError(
            "Evaluation dataset contains no records."
        )

    active_reranker = (
        reranker
        if reranker is not None
        else HuggingFaceBGEReranker()
    )

    print(
        "NepalGov AI context-selection benchmark"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"First-stage/reranker depth: "
        f"{retrieval_depth}"
    )

    print(
        "Fixed context counts: "
        + ", ".join(
            str(count)
            for count in normalized_counts
        )
    )

    print(
        "Token budgets: "
        + ", ".join(
            str(budget)
            for budget in normalized_budgets
        )
    )

    # Retrieval and hosted BGE reranking happen only here. Every selector below
    # evaluates this exact cached reranked pool.
    reranked_candidates = (
        collect_reranked_candidates(
            records,
            retrieval_depth=retrieval_depth,
            reranker=active_reranker,
        )
    )

    for count in normalized_counts:
        print_selector_results(
            records,
            reranked_candidates,
            label=(
                f"Fixed top-{count} context"
            ),
            selector=(
                FixedTopKContextSelector(
                    top_k=count
                )
            ),
        )

    for budget in normalized_budgets:
        print_selector_results(
            records,
            reranked_candidates,
            label=(
                f"Token budget {budget}"
            ),
            selector=(
                TokenBudgetContextSelector(
                    max_tokens=budget
                )
            ),
        )

    print_selector_results(
        records,
        reranked_candidates,
        label="Adjacent-aware top-5 context",
        selector=(
            AdjacentChunkContextSelector(
                top_k=5
            )
        ),
    )


if __name__ == "__main__":
    run_comparison()