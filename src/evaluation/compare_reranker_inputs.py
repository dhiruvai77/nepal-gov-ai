"""Compare plain vs document-title-aware BGE reranker inputs.

This experiment keeps first-stage retrieval fixed and tests whether adding the
source document title to each passage helps the reranker respect source intent
without changing production reranking behavior.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

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


TRACKED_QUESTION_IDS = (
    "en_en_002",
    "en_en_007",
    "ne_ne_002",
    "ne_en_003",
)


class TitleAwareReranker(Reranker):
    """Evaluation-only wrapper that prefixes document titles to passages."""

    def __init__(
        self,
        inner: Reranker,
    ) -> None:
        self.inner = inner

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Rerank using title-enriched text while preserving original results."""

        original_candidates = list(
            candidates
        )

        originals_by_id = {
            candidate.point_id: candidate
            for candidate in original_candidates
        }

        if (
            len(originals_by_id)
            != len(original_candidates)
        ):
            raise ValueError(
                "Candidates must have unique point IDs."
            )

        title_aware_candidates = [
            replace(
                candidate,
                chunk_text=(
                    f"Document: {candidate.title}\n\n"
                    f"{candidate.chunk_text}"
                ),
            )
            for candidate in original_candidates
        ]

        ranked = self.inner.rerank(
            query,
            title_aware_candidates,
            top_k=top_k,
        )

        # Return the untouched original RetrievalResult payload. Only the text
        # supplied to the cross-encoder is modified for this experiment.
        return [
            RerankedResult(
                result=originals_by_id[
                    item.result.point_id
                ],
                rerank_score=item.rerank_score,
                original_rank=item.original_rank,
            )
            for item in ranked
        ]


def validate_candidate_identity(
    baseline: Sequence[RetrievalResult],
    reranked: Sequence[RerankedResult],
) -> None:
    """Ensure reranking only changes order, not candidate membership."""

    baseline_ids = [
        item.point_id
        for item in baseline
    ]

    reranked_ids = [
        item.result.point_id
        for item in reranked
    ]

    if len(
        baseline_ids
    ) != len(
        reranked_ids
    ):
        raise RuntimeError(
            "Reranking changed candidate count."
        )

    if set(
        baseline_ids
    ) != set(
        reranked_ids
    ):
        raise RuntimeError(
            "Reranking changed candidate identity."
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
    """Evaluate cached rankings with the project's existing metric logic."""

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


def print_three_way_row(
    label: str,
    count: int,
    baseline: dict[str, float],
    plain: dict[str, float],
    title: dict[str, float],
) -> None:
    """Print baseline, plain-BGE, and title-aware-BGE metrics."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{baseline['hit_rate']:>6.3f} "
        f"{plain['hit_rate']:>6.3f} "
        f"{title['hit_rate']:>6.3f}  "
        f"{baseline['mrr']:>6.3f} "
        f"{plain['mrr']:>6.3f} "
        f"{title['mrr']:>6.3f}  "
        f"{baseline['recall']:>6.3f} "
        f"{plain['recall']:>6.3f} "
        f"{title['recall']:>6.3f}"
    )


def first_primary_rank(
    record: EvaluationRecord,
    results: Sequence[RetrievalResult],
) -> int | None:
    """Return the rank of the first primary-relevant result."""

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result.point_id in primary_ids:
            return rank

    return None


def print_cutoff_comparison(
    records: Sequence[EvaluationRecord],
    baseline_metrics: Sequence[QueryMetrics],
    plain_metrics: Sequence[QueryMetrics],
    title_metrics: Sequence[QueryMetrics],
    *,
    cutoff: int,
) -> None:
    """Print three-way metrics and plain-vs-title hit transitions."""

    baseline_grouped = (
        group_metrics_by_language_pair(
            records,
            baseline_metrics,
        )
    )

    plain_grouped = (
        group_metrics_by_language_pair(
            records,
            plain_metrics,
        )
    )

    title_grouped = (
        group_metrics_by_language_pair(
            records,
            title_metrics,
        )
    )

    print(
        "\n"
        + "=" * 105
    )

    print(
        f"Reranker input comparison @ {cutoff}"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>4}  "
        f"{'BaseH':>6} "
        f"{'PlainH':>6} "
        f"{'TitleH':>6}  "
        f"{'BaseM':>6} "
        f"{'PlainM':>6} "
        f"{'TitleM':>6}  "
        f"{'BaseR':>6} "
        f"{'PlainR':>6} "
        f"{'TitleR':>6}"
    )

    print(
        "-" * 105
    )

    for pair in LANGUAGE_PAIR_ORDER:
        baseline_slice = (
            baseline_grouped.get(
                pair
            )
        )

        plain_slice = (
            plain_grouped.get(
                pair
            )
        )

        title_slice = (
            title_grouped.get(
                pair
            )
        )

        if (
            not baseline_slice
            or not plain_slice
            or not title_slice
        ):
            continue

        print_three_way_row(
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
                plain_slice
            ),
            aggregate_metrics(
                title_slice
            ),
        )

    print(
        "-" * 105
    )

    print_three_way_row(
        "overall",
        len(
            records
        ),
        aggregate_metrics(
            baseline_metrics
        ),
        aggregate_metrics(
            plain_metrics
        ),
        aggregate_metrics(
            title_metrics
        ),
    )

    plain_by_id = {
        item.question_id: item
        for item in plain_metrics
    }

    title_by_id = {
        item.question_id: item
        for item in title_metrics
    }

    recovered = sorted(
        question_id
        for question_id, plain_metric
        in plain_by_id.items()
        if (
            plain_metric.hit_rate == 0.0
            and title_by_id[
                question_id
            ].hit_rate == 1.0
        )
    )

    regressed = sorted(
        question_id
        for question_id, plain_metric
        in plain_by_id.items()
        if (
            plain_metric.hit_rate == 1.0
            and title_by_id[
                question_id
            ].hit_rate == 0.0
        )
    )

    print()

    print(
        f"Recovered by title-aware vs plain @{cutoff}: "
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
        f"Regressed by title-aware vs plain @{cutoff}: "
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
) -> None:
    """Compare plain and title-aware BGE on identical retrieved candidates."""

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

    base_reranker = (
        HuggingFaceBGEReranker()
    )

    title_reranker = (
        TitleAwareReranker(
            base_reranker
        )
    )

    baseline_rankings: dict[
        str,
        list[RetrievalResult],
    ] = {}

    plain_rankings: dict[
        str,
        list[RetrievalResult],
    ] = {}

    title_rankings: dict[
        str,
        list[RetrievalResult],
    ] = {}

    print(
        "Plain vs document-title-aware BGE reranking"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Fixed first-stage depth: {retrieval_depth}"
    )

    print(
        "Each question is retrieved once, then the identical "
        "candidate pool is reranked by both variants."
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record.question_id}"
        )

        candidates = list(
            retrieve_for_evaluation(
                record,
                retrieval_depth,
            )
        )

        if not candidates:
            raise RuntimeError(
                f"{record.question_id}: retrieval returned no candidates."
            )

        plain_items = (
            base_reranker.rerank(
                record.query,
                candidates,
                top_k=None,
            )
        )

        title_items = (
            title_reranker.rerank(
                record.query,
                candidates,
                top_k=None,
            )
        )

        validate_candidate_identity(
            candidates,
            plain_items,
        )

        validate_candidate_identity(
            candidates,
            title_items,
        )

        baseline_rankings[
            record.question_id
        ] = candidates

        plain_rankings[
            record.question_id
        ] = [
            item.result
            for item in plain_items
        ]

        title_rankings[
            record.question_id
        ] = [
            item.result
            for item in title_items
        ]

    for cutoff in normalized_cutoffs:
        baseline_metrics = calculate_metrics(
            records,
            baseline_rankings,
            cutoff=cutoff,
        )

        plain_metrics = calculate_metrics(
            records,
            plain_rankings,
            cutoff=cutoff,
        )

        title_metrics = calculate_metrics(
            records,
            title_rankings,
            cutoff=cutoff,
        )

        print_cutoff_comparison(
            records,
            baseline_metrics,
            plain_metrics,
            title_metrics,
            cutoff=cutoff,
        )

    print(
        "\nTracked primary-evidence ranks"
    )

    print(
        "=" * 60
    )

    records_by_id = {
        record.question_id: record
        for record in records
    }

    for question_id in TRACKED_QUESTION_IDS:
        record = records_by_id[
            question_id
        ]

        print(
            f"{question_id}: "
            f"baseline="
            f"{first_primary_rank(record, baseline_rankings[question_id])}, "
            f"plain="
            f"{first_primary_rank(record, plain_rankings[question_id])}, "
            f"title="
            f"{first_primary_rank(record, title_rankings[question_id])}"
        )


if __name__ == "__main__":
    run_comparison()