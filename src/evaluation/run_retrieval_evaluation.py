from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    QueryMetrics,
    aggregate_metrics,
    evaluate_ranked_results,
    load_evaluation_records,
)
from src.retrieval.dense_retriever import RetrievalResult
from src.retrieval.run_hybrid_retrieval import run_hybrid_retrieval


DEFAULT_DATASET_PATH = Path("data/evaluation/retrieval_questions.jsonl")

# Use the same retrieved ranking for every evaluation cutoff. This avoids
# changing the retrieval candidate pool when comparing @5, @10, and @20.
DEFAULT_CUTOFFS = (5, 10, 20)

# Keep slice output in a stable order so benchmark runs are easy to compare.
LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)


def retrieve_for_evaluation(
    record: EvaluationRecord,
    k: int,
) -> Sequence[RetrievalResult]:
    """Run the same language-aware retrieval routing used in production."""

    # Evaluation explicitly defines the language in which the gold evidence
    # must exist. This also drives dense-only routing for cross-lingual cases.
    filters = {
        "language": record.target_language,
    }

    return run_hybrid_retrieval(
        record.query,
        top_k=k,
        filters=filters,
    )


def validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize requested retrieval evaluation cutoffs."""

    # Remove duplicates while keeping the resulting benchmark deterministic.
    normalized = tuple(sorted(set(cutoffs)))

    if not normalized:
        raise ValueError("At least one evaluation cutoff is required.")

    if any(k <= 0 for k in normalized):
        raise ValueError("All evaluation cutoffs must be greater than zero.")

    return normalized


def group_metrics_by_language_pair(
    records: Sequence[EvaluationRecord],
    metrics: Sequence[QueryMetrics],
) -> dict[tuple[str, str], list[QueryMetrics]]:
    """Group per-query metrics into query-language -> target-language slices."""

    metrics_by_question = {
        metric.question_id: metric
        for metric in metrics
    }

    grouped: dict[tuple[str, str], list[QueryMetrics]] = defaultdict(list)

    for record in records:
        language_pair = (
            record.query_language,
            record.target_language,
        )

        grouped[language_pair].append(
            metrics_by_question[record.question_id]
        )

    return dict(grouped)


def format_language_pair(language_pair: tuple[str, str]) -> str:
    """Return a compact human-readable language-pair label."""

    query_language, target_language = language_pair
    return f"{query_language}->{target_language}"


def print_summary_row(
    label: str,
    count: int,
    summary: dict[str, float],
) -> None:
    """Print one aligned aggregate benchmark row."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{summary['hit_rate']:>8.3f}  "
        f"{summary['mrr']:>8.3f}  "
        f"{summary['recall']:>8.3f}"
    )


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> None:
    """Evaluate production retrieval at multiple cutoffs and language slices."""

    normalized_cutoffs = validate_cutoffs(cutoffs)
    records = load_evaluation_records(dataset_path)

    if not records:
        raise ValueError("Evaluation dataset contains no records.")

    # Retrieve once at the deepest requested cutoff. Reusing this ranking for
    # smaller cutoffs makes @5/@10/@20 directly comparable and also avoids
    # repeating hosted embedding requests three times per evaluation query.
    retrieval_depth = max(normalized_cutoffs)

    retrieved_results = {
        record.question_id: retrieve_for_evaluation(
            record,
            retrieval_depth,
        )
        for record in records
    }

    print(
        "Retrieval evaluation "
        f"(fixed retrieval depth = {retrieval_depth})"
    )
    print(
        "Each query is retrieved once; @k metrics are calculated "
        "from the same ranked result list."
    )

    for k in normalized_cutoffs:
        # Evaluate every record against the same cached ranking, truncating
        # only at the requested metric cutoff.
        per_query = [
            evaluate_ranked_results(
                record,
                retrieved_results[record.question_id],
                k=k,
            )
            for record in records
        ]

        overall_summary = aggregate_metrics(per_query)
        grouped_metrics = group_metrics_by_language_pair(
            records,
            per_query,
        )

        print("\n" + "=" * 62)
        print(f"Retrieval evaluation @ {k}")
        print("=" * 62)
        print(
            f"{'Slice':<10}"
            f"{'N':>4}  "
            f"{'Hit Rate':>8}  "
            f"{'MRR':>8}  "
            f"{'Recall':>8}"
        )
        print("-" * 62)

        # Print the four multilingual evaluation slices separately so dense,
        # hybrid, and cross-lingual behavior can be diagnosed independently.
        for language_pair in LANGUAGE_PAIR_ORDER:
            slice_metrics = grouped_metrics.get(language_pair)

            if not slice_metrics:
                continue

            slice_summary = aggregate_metrics(slice_metrics)

            print_summary_row(
                format_language_pair(language_pair),
                len(slice_metrics),
                slice_summary,
            )

        print("-" * 62)
        print_summary_row(
            "overall",
            len(per_query),
            overall_summary,
        )

        # Listing primary-evidence misses lets us see whether a query moves
        # from failure at @5 into the relevant range at @10 or @20.
        missed_primary = [
            metric.question_id
            for metric in per_query
            if metric.hit_rate == 0.0
        ]

        print()
        print(
            f"Primary evidence misses @{k}: "
            f"{len(missed_primary)}/{len(per_query)}"
        )

        if missed_primary:
            print(", ".join(missed_primary))
        else:
            print("None")


if __name__ == "__main__":
    run_evaluation()