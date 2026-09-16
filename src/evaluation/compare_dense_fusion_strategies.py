"""Compare contextual-only retrieval with dual-dense reciprocal-rank fusion.

This experiment tests whether the original raw E5 representation still adds
useful complementary evidence after contextual embeddings are introduced.

Same-language:
    contextual dense + BM25
    versus
    raw dense + contextual dense + BM25

Cross-lingual:
    contextual dense only
    versus
    raw dense + contextual dense

Production code is not modified by this experiment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
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

# Use the same conventional RRF constant already used by NepalGov AI.
RRF_K = 60

# Retrieve substantially more candidates than the final evaluation depth so a
# passage supported by one representation can still enter the fused ranking.
CANDIDATE_MULTIPLIER = 3

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)


def rrf_fuse(
    ranked_lists: Sequence[
        Sequence[RetrievalResult]
    ],
    *,
    top_k: int,
) -> list[RetrievalResult]:
    """Fuse any number of ranked retrieval lists using standard RRF."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    results_by_id: dict[
        str,
        RetrievalResult,
    ] = {}

    fused_scores: dict[
        str,
        float,
    ] = {}

    for results in ranked_lists:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            results_by_id.setdefault(
                result.point_id,
                result,
            )

            fused_scores[
                result.point_id
            ] = (
                fused_scores.get(
                    result.point_id,
                    0.0,
                )
                + 1.0 / (
                    RRF_K + rank
                )
            )

    ranked_ids = sorted(
        fused_scores,
        key=lambda point_id: (
            -fused_scores[
                point_id
            ],
            point_id,
        ),
    )

    return [
        replace(
            results_by_id[
                point_id
            ],
            score=fused_scores[
                point_id
            ],
        )
        for point_id in ranked_ids[
            :top_k
        ]
    ]


def retrieve_candidates(
    record: EvaluationRecord,
    *,
    raw_dense: DenseRetriever,
    contextual_dense: DenseRetriever,
    sparse: SparseRetriever,
    final_depth: int,
) -> tuple[
    list[RetrievalResult],
    list[RetrievalResult],
]:
    """Return contextual-only and dual-dense rankings for one query."""

    filters = {
        "language": record.target_language,
    }

    candidate_depth = (
        final_depth
        * CANDIDATE_MULTIPLIER
    )

    query_language = detect_query_language(
        record.query
    )

    raw_results = raw_dense.retrieve(
        query=record.query,
        top_k=candidate_depth,
        filters=filters,
    )

    contextual_results = (
        contextual_dense.retrieve(
            query=record.query,
            top_k=candidate_depth,
            filters=filters,
        )
    )

    # Cross-lingual routing deliberately excludes BM25. Compare contextual-only
    # dense retrieval against fusion of the two multilingual dense views.
    if (
        query_language
        != record.target_language
    ):
        contextual_only = (
            contextual_results[
                :final_depth
            ]
        )

        dual_dense = rrf_fuse(
            [
                raw_results,
                contextual_results,
            ],
            top_k=final_depth,
        )

        return (
            contextual_only,
            dual_dense,
        )

    sparse_results = sparse.retrieve(
        query=record.query,
        top_k=candidate_depth,
        filters=filters,
    )

    # This reproduces the proposed contextual production path: contextual E5
    # plus the existing lexical BM25 ranking.
    contextual_only = rrf_fuse(
        [
            contextual_results,
            sparse_results,
        ],
        top_k=final_depth,
    )

    # The experimental path preserves both semantic views while keeping BM25.
    dual_dense = rrf_fuse(
        [
            raw_results,
            contextual_results,
            sparse_results,
        ],
        top_k=final_depth,
    )

    return (
        contextual_only,
        dual_dense,
    )


def group_metrics(
    records: Sequence[EvaluationRecord],
    metrics: Sequence[QueryMetrics],
) -> dict[
    tuple[str, str],
    list[QueryMetrics],
]:
    """Group benchmark metrics into language-pair slices."""

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

        grouped[
            pair
        ].append(
            by_question[
                record.question_id
            ]
        )

    return dict(
        grouped
    )


def print_row(
    label: str,
    count: int,
    contextual: dict[str, float],
    fused: dict[str, float],
) -> None:
    """Print contextual-only and dual-dense fusion metrics."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{contextual['hit_rate']:>7.3f} "
        f"{fused['hit_rate']:>7.3f} "
        f"{fused['hit_rate'] - contextual['hit_rate']:>+7.3f}  "
        f"{contextual['mrr']:>7.3f} "
        f"{fused['mrr']:>7.3f} "
        f"{fused['mrr'] - contextual['mrr']:>+7.3f}  "
        f"{contextual['recall']:>7.3f} "
        f"{fused['recall']:>7.3f} "
        f"{fused['recall'] - contextual['recall']:>+7.3f}"
    )


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> None:
    """Evaluate whether raw dense adds value beside contextual dense."""

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
        vector_name=(
            CONTEXTUAL_DENSE_VECTOR_NAME
        ),
    )

    sparse = SparseRetriever(
        client=client,
    )

    contextual_rankings: dict[
        str,
        list[RetrievalResult],
    ] = {}

    fused_rankings: dict[
        str,
        list[RetrievalResult],
    ] = {}

    print(
        "Contextual vs dual-dense fusion benchmark"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Fixed evaluation depth: {retrieval_depth}"
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record.question_id}"
        )

        contextual, fused = (
            retrieve_candidates(
                record,
                raw_dense=raw_dense,
                contextual_dense=contextual_dense,
                sparse=sparse,
                final_depth=retrieval_depth,
            )
        )

        contextual_rankings[
            record.question_id
        ] = contextual

        fused_rankings[
            record.question_id
        ] = fused

    for cutoff in normalized_cutoffs:
        contextual_metrics = [
            evaluate_ranked_results(
                record,
                contextual_rankings[
                    record.question_id
                ],
                k=cutoff,
            )
            for record in records
        ]

        fused_metrics = [
            evaluate_ranked_results(
                record,
                fused_rankings[
                    record.question_id
                ],
                k=cutoff,
            )
            for record in records
        ]

        contextual_grouped = group_metrics(
            records,
            contextual_metrics,
        )

        fused_grouped = group_metrics(
            records,
            fused_metrics,
        )

        print(
            "\n"
            + "=" * 116
        )

        print(
            f"Contextual vs dual-dense fusion @ {cutoff}"
        )

        print(
            "=" * 116
        )

        print(
            f"{'Slice':<10}"
            f"{'N':>4}  "
            f"{'CtxHit':>7} "
            f"{'FuseHit':>7} "
            f"{'ΔHit':>7}  "
            f"{'CtxMRR':>7} "
            f"{'FuseMRR':>7} "
            f"{'ΔMRR':>7}  "
            f"{'CtxRec':>7} "
            f"{'FuseRec':>7} "
            f"{'ΔRec':>7}"
        )

        print(
            "-" * 116
        )

        for pair in LANGUAGE_PAIR_ORDER:
            contextual_slice = (
                contextual_grouped.get(
                    pair
                )
            )

            fused_slice = (
                fused_grouped.get(
                    pair
                )
            )

            if (
                not contextual_slice
                or not fused_slice
            ):
                continue

            print_row(
                (
                    f"{pair[0]}"
                    f"->{pair[1]}"
                ),
                len(
                    contextual_slice
                ),
                aggregate_metrics(
                    contextual_slice
                ),
                aggregate_metrics(
                    fused_slice
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
                contextual_metrics
            ),
            aggregate_metrics(
                fused_metrics
            ),
        )

        contextual_misses = {
            metric.question_id
            for metric in contextual_metrics
            if metric.hit_rate == 0.0
        }

        fused_misses = {
            metric.question_id
            for metric in fused_metrics
            if metric.hit_rate == 0.0
        }

        recovered = sorted(
            contextual_misses
            - fused_misses
        )

        regressions = sorted(
            fused_misses
            - contextual_misses
        )

        print()

        print(
            f"Recovered by fusion @{cutoff}: "
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
            f"Regressed under fusion @{cutoff}: "
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