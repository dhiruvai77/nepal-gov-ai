"""Compare production retrieval using raw vs contextual dense representations.

Same-language queries use BM25 + dense + RRF exactly like production.
Cross-lingual queries use dense only, matching NepalGov AI routing.

Both variants use fixed retrieval depth so @5, @10, and @20 metrics are derived
from one stable ranking per query.
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
    CONTEXTUAL_DENSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    QDRANT_URL,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.retrieval.run_hybrid_retrieval import (
    detect_query_language,
)
from src.retrieval.sparse_retriever import (
    SparseRetriever,
)


DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

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


def retrieve_variant(
    *,
    record: EvaluationRecord,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    top_k: int,
) -> list[RetrievalResult]:
    """Run the same language-aware routing used by production."""

    filters = {
        "language": record.target_language,
    }

    query_language = detect_query_language(
        record.query
    )

    # Production skips BM25 when query and evidence languages differ because
    # sparse lexical overlap is unreliable across English and Nepali.
    if (
        query_language
        != record.target_language
    ):
        return dense_retriever.retrieve(
            query=record.query,
            top_k=top_k,
            filters=filters,
        )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    return hybrid_retriever.retrieve(
        query=record.query,
        top_k=top_k,
        filters=filters,
    )


def group_metrics(
    records: Sequence[EvaluationRecord],
    metrics: Sequence[QueryMetrics],
) -> dict[
    tuple[str, str],
    list[QueryMetrics],
]:
    """Group metrics by query-language -> target-language pair."""

    by_question = {
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
            by_question[
                record.question_id
            ]
        )

    return dict(
        grouped
    )


def format_pair(
    pair: tuple[str, str],
) -> str:
    """Return a compact language-pair label."""

    return (
        f"{pair[0]}->{pair[1]}"
    )


def print_row(
    label: str,
    count: int,
    raw: dict[str, float],
    contextual: dict[str, float],
) -> None:
    """Print raw/contextual production metrics and their differences."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{raw['hit_rate']:>7.3f} "
        f"{contextual['hit_rate']:>7.3f} "
        f"{contextual['hit_rate'] - raw['hit_rate']:>+7.3f}  "
        f"{raw['mrr']:>7.3f} "
        f"{contextual['mrr']:>7.3f} "
        f"{contextual['mrr'] - raw['mrr']:>+7.3f}  "
        f"{raw['recall']:>7.3f} "
        f"{contextual['recall']:>7.3f} "
        f"{contextual['recall'] - raw['recall']:>+7.3f}"
    )


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> None:
    """Compare the two dense representations under production routing."""

    normalized_cutoffs = tuple(
        sorted(
            set(
                cutoffs
            )
        )
    )

    if not normalized_cutoffs:
        raise ValueError(
            "At least one cutoff is required."
        )

    if any(
        cutoff <= 0
        for cutoff in normalized_cutoffs
    ):
        raise ValueError(
            "All cutoffs must be greater than zero."
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

    raw_dense = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
        vector_name=DENSE_VECTOR_NAME,
    )

    contextual_dense = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
        vector_name=CONTEXTUAL_DENSE_VECTOR_NAME,
    )

    sparse = SparseRetriever(
        client=client,
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
        "Production-routing retrieval comparison"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Fixed retrieval depth: {retrieval_depth}"
    )

    print(
        "Same-language: dense + BM25 + RRF"
    )

    print(
        "Cross-lingual: dense only"
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record.question_id}"
        )

        raw_results[
            record.question_id
        ] = retrieve_variant(
            record=record,
            dense_retriever=raw_dense,
            sparse_retriever=sparse,
            top_k=retrieval_depth,
        )

        contextual_results[
            record.question_id
        ] = retrieve_variant(
            record=record,
            dense_retriever=contextual_dense,
            sparse_retriever=sparse,
            top_k=retrieval_depth,
        )

    for cutoff in normalized_cutoffs:
        raw_metrics = [
            evaluate_ranked_results(
                record,
                raw_results[
                    record.question_id
                ],
                k=cutoff,
            )
            for record in records
        ]

        contextual_metrics = [
            evaluate_ranked_results(
                record,
                contextual_results[
                    record.question_id
                ],
                k=cutoff,
            )
            for record in records
        ]

        raw_grouped = group_metrics(
            records,
            raw_metrics,
        )

        contextual_grouped = group_metrics(
            records,
            contextual_metrics,
        )

        print(
            "\n"
            + "=" * 116
        )

        print(
            f"Production-routing comparison @ {cutoff}"
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

            print_row(
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

        print_row(
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
            f"Recovered by contextual @{cutoff}: "
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
            f"Regressed under contextual @{cutoff}: "
            f"{len(regressions)}"
        )

        print(
            ", ".join(
                regressions
            )
            if regressions
            else "None"
        )


if __name__ == "__main__":
    run_comparison()