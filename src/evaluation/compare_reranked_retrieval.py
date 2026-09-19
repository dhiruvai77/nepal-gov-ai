"""Benchmark production retrieval against hosted BGE reranking."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    QueryMetrics,
    aggregate_metrics,
    evaluate_ranked_results,
    load_evaluation_records,
)
from src.evaluation.run_retrieval_evaluation import (
    DEFAULT_CUTOFFS,
    DEFAULT_DATASET_PATH,
    LANGUAGE_PAIR_ORDER,
    format_language_pair,
    group_metrics_by_language_pair,
    retrieve_for_evaluation,
    validate_cutoffs,
)
from src.reranking.base import Reranker
from src.reranking.hf_bge_reranker import HuggingFaceBGEReranker
from src.retrieval.dense_retriever import RetrievalResult


RetrieveFunction = Callable[
    [EvaluationRecord, int],
    Sequence[RetrievalResult],
]


def validate_reranked_candidate_set(
    first_stage: Sequence[RetrievalResult],
    reranked: Sequence[RetrievalResult],
) -> None:
    """Ensure reranking changes only order, never the candidate pool."""

    first_ids = [
        item.point_id
        for item in first_stage
    ]

    reranked_ids = [
        item.point_id
        for item in reranked
    ]

    if len(first_ids) != len(reranked_ids):
        raise RuntimeError(
            "Reranking changed the number of candidates."
        )

    if set(first_ids) != set(reranked_ids):
        raise RuntimeError(
            "Reranking changed candidate identity."
        )

    if len(set(reranked_ids)) != len(reranked_ids):
        raise RuntimeError(
            "Reranking returned duplicate candidates."
        )


def collect_rankings(
    records: Sequence[EvaluationRecord],
    *,
    retrieval_depth: int,
    reranker: Reranker,
    retrieve: RetrieveFunction = retrieve_for_evaluation,
) -> tuple[
    dict[str, list[RetrievalResult]],
    dict[str, list[RetrievalResult]],
]:
    """Retrieve once per question and rerank that exact fixed-depth list."""

    if retrieval_depth <= 0:
        raise ValueError(
            "retrieval_depth must be greater than zero."
        )

    first_stage: dict[
        str,
        list[RetrievalResult],
    ] = {}

    reranked: dict[
        str,
        list[RetrievalResult],
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

        # Do not truncate here. @5/@10/@20 must all be calculated from
        # the same fixed 20-candidate reranked list.
        reranked_items = reranker.rerank(
            record.query,
            candidates,
            top_k=None,
        )

        reranked_results = [
            item.result
            for item in reranked_items
        ]

        validate_reranked_candidate_set(
            candidates,
            reranked_results,
        )

        first_stage[
            record.question_id
        ] = candidates

        reranked[
            record.question_id
        ] = reranked_results

    return (
        first_stage,
        reranked,
    )


def calculate_metrics(
    records: Sequence[EvaluationRecord],
    rankings: dict[
        str,
        list[RetrievalResult],
    ],
    *,
    cutoff: int,
) -> list[QueryMetrics]:
    """Calculate existing NepalGov retrieval metrics for cached rankings."""

    return [
        evaluate_ranked_results(
            record,
            rankings[
                record.question_id
            ],
            k=cutoff,
        )
        for record in records
    ]


def print_comparison_row(
    label: str,
    count: int,
    baseline: dict[str, float],
    reranked: dict[str, float],
) -> None:
    """Print baseline, reranked, and delta metrics for one slice."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{baseline['hit_rate']:>7.3f} "
        f"{reranked['hit_rate']:>7.3f} "
        f"{reranked['hit_rate'] - baseline['hit_rate']:>+7.3f}  "
        f"{baseline['mrr']:>7.3f} "
        f"{reranked['mrr']:>7.3f} "
        f"{reranked['mrr'] - baseline['mrr']:>+7.3f}  "
        f"{baseline['recall']:>7.3f} "
        f"{reranked['recall']:>7.3f} "
        f"{reranked['recall'] - baseline['recall']:>+7.3f}"
    )


def print_cutoff_comparison(
    records: Sequence[EvaluationRecord],
    baseline: Sequence[QueryMetrics],
    reranked: Sequence[QueryMetrics],
    *,
    cutoff: int,
) -> None:
    """Print multilingual metric changes and hit-level regressions."""

    baseline_grouped = (
        group_metrics_by_language_pair(
            records,
            baseline,
        )
    )

    reranked_grouped = (
        group_metrics_by_language_pair(
            records,
            reranked,
        )
    )

    print(
        "\n"
        + "=" * 116
    )

    print(
        f"First-stage vs BGE reranked @ {cutoff}"
    )

    print(
        "=" * 116
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>4}  "
        f"{'BaseHit':>7} "
        f"{'RerHit':>7} "
        f"{'ΔHit':>7}  "
        f"{'BaseMRR':>7} "
        f"{'RerMRR':>7} "
        f"{'ΔMRR':>7}  "
        f"{'BaseRec':>7} "
        f"{'RerRec':>7} "
        f"{'ΔRec':>7}"
    )

    print(
        "-" * 116
    )

    for pair in LANGUAGE_PAIR_ORDER:
        baseline_slice = (
            baseline_grouped.get(
                pair
            )
        )

        reranked_slice = (
            reranked_grouped.get(
                pair
            )
        )

        if (
            not baseline_slice
            or not reranked_slice
        ):
            continue

        print_comparison_row(
            format_language_pair(
                pair
            ),
            len(
                baseline_slice
            ),
            aggregate_metrics(
                baseline_slice
            ),
            aggregate_metrics(
                reranked_slice
            ),
        )

    print(
        "-" * 116
    )

    print_comparison_row(
        "overall",
        len(
            records
        ),
        aggregate_metrics(
            baseline
        ),
        aggregate_metrics(
            reranked
        ),
    )

    baseline_by_id = {
        item.question_id: item
        for item in baseline
    }

    reranked_by_id = {
        item.question_id: item
        for item in reranked
    }

    recovered = sorted(
        question_id
        for question_id, metric
        in baseline_by_id.items()
        if (
            metric.hit_rate == 0.0
            and reranked_by_id[
                question_id
            ].hit_rate == 1.0
        )
    )

    regressed = sorted(
        question_id
        for question_id, metric
        in baseline_by_id.items()
        if (
            metric.hit_rate == 1.0
            and reranked_by_id[
                question_id
            ].hit_rate == 0.0
        )
    )

    print()

    print(
        f"Recovered by reranking @{cutoff}: "
        f"{len(recovered)}"
    )

    print(
        ", ".join(
            recovered
        )
        if recovered
        else "None"
    )

    print()

    print(
        f"Regressed under reranking @{cutoff}: "
        f"{len(regressed)}"
    )

    print(
        ", ".join(
            regressed
        )
        if regressed
        else "None"
    )


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    reranker: Reranker | None = None,
) -> None:
    """Run the 30-question fixed-depth reranker benchmark."""

    normalized_cutoffs = validate_cutoffs(
        cutoffs
    )

    records = load_evaluation_records(
        dataset_path
    )

    if not records:
        raise ValueError(
            "Evaluation dataset contains no records."
        )

    retrieval_depth = max(
        normalized_cutoffs
    )

    active_reranker = (
        reranker
        or HuggingFaceBGEReranker()
    )

    print(
        "Production retrieval vs BGE reranking"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Fixed first-stage depth: {retrieval_depth}"
    )

    (
        baseline_rankings,
        reranked_rankings,
    ) = collect_rankings(
        records,
        retrieval_depth=retrieval_depth,
        reranker=active_reranker,
    )

    for cutoff in normalized_cutoffs:
        baseline_metrics = (
            calculate_metrics(
                records,
                baseline_rankings,
                cutoff=cutoff,
            )
        )

        reranked_metrics = (
            calculate_metrics(
                records,
                reranked_rankings,
                cutoff=cutoff,
            )
        )

        print_cutoff_comparison(
            records,
            baseline_metrics,
            reranked_metrics,
            cutoff=cutoff,
        )


if __name__ == "__main__":
    run_comparison()