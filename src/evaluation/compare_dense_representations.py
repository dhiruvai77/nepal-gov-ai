"""Compare raw and contextual dense retrieval on the full evaluation benchmark.

This diagnostic embeds each evaluation query once, searches both Qdrant dense
representations with the same query vector and filters, and evaluates both
rankings at fixed @5, @10, and @20 cutoffs.

Production retrieval is not modified by this experiment.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    QueryMetrics,
    aggregate_metrics,
    evaluate_ranked_results,
    load_evaluation_records,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    QDRANT_URL,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
    build_metadata_filter,
    normalize_search_result,
)


DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

# Retrieve once at the deepest cutoff so @5, @10, and @20 all come from the
# same ranking. This avoids changing candidate depth between metric cutoffs.
DEFAULT_CUTOFFS = (
    5,
    10,
    20,
)

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)


def retrieve_with_vector(
    client: QdrantClient,
    query_vector: list[float],
    *,
    vector_name: str,
    target_language: str,
    top_k: int,
) -> list[RetrievalResult]:
    """Search one named dense representation with a precomputed query vector."""

    query_filter = build_metadata_filter(
        {
            "language": target_language,
        }
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=vector_name,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        normalize_search_result(
            point
        )
        for point in response.points
    ]


def primary_rank(
    record: EvaluationRecord,
    results: Sequence[RetrievalResult],
) -> int | None:
    """Return the first rank containing manually verified primary evidence."""

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


def format_rank(
    rank: int | None,
    depth: int,
) -> str:
    """Format evidence outside the retrieved depth consistently."""

    if rank is None:
        return f">{depth}"

    return str(rank)


def group_metrics_by_language_pair(
    records: Sequence[EvaluationRecord],
    metrics: Sequence[QueryMetrics],
) -> dict[
    tuple[str, str],
    list[QueryMetrics],
]:
    """Group metrics into query-language -> evidence-language slices."""

    metrics_by_question = {
        metric.question_id: metric
        for metric in metrics
    }

    grouped: dict[
        tuple[str, str],
        list[QueryMetrics],
    ] = defaultdict(list)

    for record in records:
        pair = (
            record.query_language,
            record.target_language,
        )

        grouped[pair].append(
            metrics_by_question[
                record.question_id
            ]
        )

    return dict(
        grouped
    )


def format_pair(
    pair: tuple[str, str],
) -> str:
    """Format a language-pair tuple for benchmark output."""

    return (
        f"{pair[0]}->{pair[1]}"
    )


def print_metric_row(
    label: str,
    count: int,
    raw_summary: dict[str, float],
    contextual_summary: dict[str, float],
) -> None:
    """Print raw/contextual metrics plus contextual-minus-raw deltas."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{raw_summary['hit_rate']:>7.3f} "
        f"{contextual_summary['hit_rate']:>7.3f} "
        f"{contextual_summary['hit_rate'] - raw_summary['hit_rate']:>+7.3f}  "
        f"{raw_summary['mrr']:>7.3f} "
        f"{contextual_summary['mrr']:>7.3f} "
        f"{contextual_summary['mrr'] - raw_summary['mrr']:>+7.3f}  "
        f"{raw_summary['recall']:>7.3f} "
        f"{contextual_summary['recall']:>7.3f} "
        f"{contextual_summary['recall'] - raw_summary['recall']:>+7.3f}"
    )


def print_cutoff_comparison(
    records: Sequence[EvaluationRecord],
    raw_results: dict[
        str,
        list[RetrievalResult],
    ],
    contextual_results: dict[
        str,
        list[RetrievalResult],
    ],
    *,
    k: int,
) -> None:
    """Print aggregate raw/contextual metrics at one cutoff."""

    raw_metrics = [
        evaluate_ranked_results(
            record,
            raw_results[
                record.question_id
            ],
            k=k,
        )
        for record in records
    ]

    contextual_metrics = [
        evaluate_ranked_results(
            record,
            contextual_results[
                record.question_id
            ],
            k=k,
        )
        for record in records
    ]

    raw_grouped = (
        group_metrics_by_language_pair(
            records,
            raw_metrics,
        )
    )

    contextual_grouped = (
        group_metrics_by_language_pair(
            records,
            contextual_metrics,
        )
    )

    print(
        "\n"
        + "=" * 116
    )

    print(
        f"Dense representation comparison @ {k}"
    )

    print(
        "=" * 116
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>4}  "
        f"{'RawHit':>7} "
        f"{'CtxHit':>7} "
        f"{'ΔHit':>7}  "
        f"{'RawMRR':>7} "
        f"{'CtxMRR':>7} "
        f"{'ΔMRR':>7}  "
        f"{'RawRec':>7} "
        f"{'CtxRec':>7} "
        f"{'ΔRec':>7}"
    )

    print(
        "-" * 116
    )

    for pair in LANGUAGE_PAIR_ORDER:
        raw_slice = raw_grouped.get(
            pair
        )

        contextual_slice = (
            contextual_grouped.get(
                pair
            )
        )

        if (
            not raw_slice
            or not contextual_slice
        ):
            continue

        print_metric_row(
            format_pair(
                pair
            ),
            len(
                raw_slice
            ),
            aggregate_metrics(
                raw_slice
            ),
            aggregate_metrics(
                contextual_slice
            ),
        )

    print(
        "-" * 116
    )

    print_metric_row(
        "overall",
        len(
            records
        ),
        aggregate_metrics(
            raw_metrics
        ),
        aggregate_metrics(
            contextual_metrics
        ),
    )

    raw_misses = {
        metric.question_id
        for metric in raw_metrics
        if metric.hit_rate == 0.0
    }

    contextual_misses = {
        metric.question_id
        for metric in contextual_metrics
        if metric.hit_rate == 0.0
    }

    recovered = sorted(
        raw_misses
        - contextual_misses
    )

    regressions = sorted(
        contextual_misses
        - raw_misses
    )

    print()

    print(
        f"Raw primary misses @{k}: "
        f"{len(raw_misses)}/{len(records)}"
    )

    print(
        ", ".join(
            sorted(
                raw_misses
            )
        )
        if raw_misses
        else "None"
    )

    print()

    print(
        f"Contextual primary misses @{k}: "
        f"{len(contextual_misses)}/{len(records)}"
    )

    print(
        ", ".join(
            sorted(
                contextual_misses
            )
        )
        if contextual_misses
        else "None"
    )

    print()

    print(
        f"Recovered by contextual @{k}: "
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
        f"Regressed under contextual @{k}: "
        f"{len(regressions)}"
    )

    print(
        ", ".join(
            regressions
        )
        if regressions
        else "None"
    )


def print_rank_changes(
    records: Sequence[EvaluationRecord],
    raw_results: dict[
        str,
        list[RetrievalResult],
    ],
    contextual_results: dict[
        str,
        list[RetrievalResult],
    ],
    *,
    depth: int,
) -> None:
    """Print primary-evidence rank movement for every benchmark question."""

    print(
        "\n"
        + "=" * 72
    )

    print(
        "Primary evidence rank comparison"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Question':<14}"
        f"{'Pair':<9}"
        f"{'Raw':>8}"
        f"{'Context':>10}"
        f"{'Change':>12}"
    )

    print(
        "-" * 72
    )

    for record in records:
        raw_rank = primary_rank(
            record,
            raw_results[
                record.question_id
            ],
        )

        contextual_rank = primary_rank(
            record,
            contextual_results[
                record.question_id
            ],
        )

        if (
            raw_rank is not None
            and contextual_rank is not None
        ):
            # Positive values mean contextual retrieval moved the primary
            # evidence closer to rank 1.
            rank_change = (
                raw_rank
                - contextual_rank
            )

            change_text = (
                f"{rank_change:+d}"
            )

        elif (
            raw_rank is None
            and contextual_rank is not None
        ):
            change_text = "recovered"

        elif (
            raw_rank is not None
            and contextual_rank is None
        ):
            change_text = "lost"

        else:
            change_text = "both >depth"

        pair = format_pair(
            (
                record.query_language,
                record.target_language,
            )
        )

        print(
            f"{record.question_id:<14}"
            f"{pair:<9}"
            f"{format_rank(raw_rank, depth):>8}"
            f"{format_rank(contextual_rank, depth):>10}"
            f"{change_text:>12}"
        )


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> None:
    """Run the full raw-vs-contextual dense benchmark."""

    normalized_cutoffs = tuple(
        sorted(
            set(
                cutoffs
            )
        )
    )

    if not normalized_cutoffs:
        raise ValueError(
            "At least one evaluation cutoff is required."
        )

    if any(
        cutoff <= 0
        for cutoff in normalized_cutoffs
    ):
        raise ValueError(
            "All evaluation cutoffs must be greater than zero."
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

    client = QdrantClient(
        url=QDRANT_URL
    )

    embedding_service = (
        HuggingFaceE5EmbeddingService()
    )

    raw_results: dict[
        str,
        list[RetrievalResult],
    ] = {}

    contextual_results: dict[
        str,
        list[RetrievalResult],
    ] = {}

    print(
        "Raw dense vs contextual dense evaluation"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Fixed retrieval depth: {retrieval_depth}"
    )

    print(
        "Each query is embedded once and used against both dense vectors."
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record.question_id}"
        )

        # The exact same E5 query embedding is used against both passage
        # representations. This isolates the effect of passage contextualization.
        query_vector = (
            embedding_service.embed_query(
                record.query
            )
        )

        if (
            len(query_vector)
            != embedding_service.dimension
        ):
            raise RuntimeError(
                "Unexpected query embedding dimension "
                f"for {record.question_id}: "
                f"{len(query_vector)}."
            )

        raw_results[
            record.question_id
        ] = retrieve_with_vector(
            client,
            query_vector,
            vector_name=DENSE_VECTOR_NAME,
            target_language=record.target_language,
            top_k=retrieval_depth,
        )

        contextual_results[
            record.question_id
        ] = retrieve_with_vector(
            client,
            query_vector,
            vector_name=(
                CONTEXTUAL_DENSE_VECTOR_NAME
            ),
            target_language=record.target_language,
            top_k=retrieval_depth,
        )

    for cutoff in normalized_cutoffs:
        print_cutoff_comparison(
            records,
            raw_results,
            contextual_results,
            k=cutoff,
        )

    print_rank_changes(
        records,
        raw_results,
        contextual_results,
        depth=retrieval_depth,
    )


if __name__ == "__main__":
    run_comparison()