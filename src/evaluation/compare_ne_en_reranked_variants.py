"""Compare NE->EN production and translated-hybrid candidate pools after BGE.

This targeted experiment evaluates whether the multilingual BGE reranker can
turn the improved candidate recall of the translated-hybrid retrieval variant
into better final top-k rankings.

The experiment is restricted to the six manually verified Nepali-query ->
English-evidence benchmark cases.

Candidate generation:

Production:
    original Nepali query -> contextual English dense top 20

Translated hybrid:
    controlled English counterpart ->
    contextual English dense top 100 + English BM25 top 100 ->
    Reciprocal Rank Fusion -> final top 20

Both candidate pools are reranked using the ORIGINAL NEPALI USER QUERY. This
keeps the reranker condition identical and isolates the effect of candidate
generation.

The controlled English queries are evaluation controls only. This module does
not add translation to the production RAG pipeline.

No production retrieval or reranking code is modified.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

from qdrant_client import (
    QdrantClient,
)

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.evaluation.compare_ne_en_retrieval_variants import (
    CONTROLLED_ENGLISH_QUERIES,
    DEFAULT_FINAL_CANDIDATE_DEPTH,
    TRANSLATED_HYBRID_COMPONENT_DEPTH,
    build_translated_hybrid,
    select_ne_en_records,
    validate_controlled_queries,
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
    QDRANT_URL,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.reranking.hf_bge_reranker import (
    HuggingFaceBGEReranker,
)
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
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


def validate_candidate_identity(
    candidates: Sequence[
        RetrievalResult
    ],
    reranked: Sequence[
        RerankedResult
    ],
) -> None:
    """Ensure reranking changes order only, never candidate membership."""

    candidate_ids = [
        item.point_id
        for item in candidates
    ]

    reranked_ids = [
        item.result.point_id
        for item in reranked
    ]

    if (
        len(
            candidate_ids
        )
        != len(
            reranked_ids
        )
    ):
        raise RuntimeError(
            "Reranking changed candidate count."
        )

    if (
        set(
            candidate_ids
        )
        != set(
            reranked_ids
        )
    ):
        raise RuntimeError(
            "Reranking changed candidate identity."
        )

    if (
        len(
            set(
                reranked_ids
            )
        )
        != len(
            reranked_ids
        )
    ):
        raise RuntimeError(
            "Reranking returned duplicate candidates."
        )


def reranked_results_only(
    items: Sequence[
        RerankedResult
    ],
) -> list[
    RetrievalResult
]:
    """Extract normalized retrieval results from reranker output."""

    return [
        item.result
        for item in items
    ]


def first_primary_rank(
    record: EvaluationRecord,
    results: Sequence[
        RetrievalResult
    ],
) -> int | None:
    """Return the first rank containing manually verified primary evidence."""

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    for (
        rank,
        result,
    ) in enumerate(
        results,
        start=1,
    ):
        if (
            result.point_id
            in primary_ids
        ):
            return rank

    return None


def format_rank(
    rank: int | None,
    *,
    depth: int,
) -> str:
    """Render missing primary evidence consistently."""

    if rank is None:
        return (
            f">{depth}"
        )

    return str(
        rank
    )


def calculate_metrics(
    records: Sequence[
        EvaluationRecord
    ],
    rankings: Mapping[
        str,
        Sequence[
            RetrievalResult
        ],
    ],
    *,
    cutoff: int,
) -> list[
    QueryMetrics
]:
    """Evaluate cached rankings with the standard retrieval metric contract."""

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


def print_metric_row(
    label: str,
    metrics: Mapping[
        str,
        float,
    ],
) -> None:
    """Print one aggregate metric row."""

    print(
        f"{label:<28}"
        f"{metrics['hit_rate']:>10.3f}"
        f"{metrics['mrr']:>10.3f}"
        f"{metrics['recall']:>10.3f}"
    )


def run_comparison(
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    *,
    candidate_depth: int = (
        DEFAULT_FINAL_CANDIDATE_DEPTH
    ),
    translated_component_depth: int = (
        TRANSLATED_HYBRID_COMPONENT_DEPTH
    ),
    cutoffs: Sequence[
        int
    ] = DEFAULT_CUTOFFS,
    reranker: Reranker | None = None,
) -> None:
    """Compare production and translated-hybrid pools after identical BGE."""

    if candidate_depth <= 0:
        raise ValueError(
            "candidate_depth must be greater than zero."
        )

    if translated_component_depth <= 0:
        raise ValueError(
            "translated_component_depth must be greater than zero."
        )

    if (
        translated_component_depth
        < candidate_depth
    ):
        raise ValueError(
            "translated_component_depth must be at least candidate_depth."
        )

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

    if (
        max(
            normalized_cutoffs
        )
        > candidate_depth
    ):
        raise ValueError(
            "candidate_depth must be at least "
            "the largest evaluation cutoff."
        )

    records = (
        select_ne_en_records(
            load_evaluation_records(
                dataset_path
            )
        )
    )

    if not records:
        raise ValueError(
            "No NE->EN evaluation records found."
        )

    validate_controlled_queries(
        records
    )

    client = (
        QdrantClient(
            url=QDRANT_URL
        )
    )

    embedding_service = (
        HuggingFaceE5EmbeddingService()
    )

    dense_retriever = (
        DenseRetriever(
            client=client,
            embedding_service=(
                embedding_service
            ),
            vector_name=(
                CONTEXTUAL_DENSE_VECTOR_NAME
            ),
        )
    )

    sparse_retriever = (
        SparseRetriever(
            client=client
        )
    )

    active_reranker = (
        reranker
        if reranker is not None
        else HuggingFaceBGEReranker()
    )

    production_first_stage: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    translated_first_stage: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    production_reranked: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    translated_reranked: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    print(
        "NE->EN candidate-pool reranker benchmark"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Final candidate depth: {candidate_depth}"
    )

    print(
        "Translated-hybrid component depth: "
        f"{translated_component_depth}"
    )

    print(
        "Both candidate pools are reranked with "
        "the original Nepali query."
    )

    for (
        index,
        record,
    ) in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{index}/{len(records)}] "
            f"{record.question_id}"
        )

        filters = {
            "language": "en",
        }

        controlled_query = (
            CONTROLLED_ENGLISH_QUERIES[
                record.question_id
            ]
        )

        production_candidates = (
            dense_retriever.retrieve(
                query=(
                    record.query
                ),
                top_k=(
                    candidate_depth
                ),
                filters=(
                    filters
                ),
            )
        )

        translated_dense = (
            dense_retriever.retrieve(
                query=(
                    controlled_query
                ),
                top_k=(
                    translated_component_depth
                ),
                filters=(
                    filters
                ),
            )
        )

        translated_sparse = (
            sparse_retriever.retrieve(
                query=(
                    controlled_query
                ),
                top_k=(
                    translated_component_depth
                ),
                filters=(
                    filters
                ),
            )
        )

        translated_candidates = (
            build_translated_hybrid(
                translated_dense,
                translated_sparse,
                component_depth=(
                    translated_component_depth
                ),
                final_depth=(
                    candidate_depth
                ),
            )
        )

        production_items = (
            active_reranker.rerank(
                record.query,
                production_candidates,
                top_k=None,
            )
        )

        translated_items = (
            active_reranker.rerank(
                record.query,
                translated_candidates,
                top_k=None,
            )
        )

        validate_candidate_identity(
            production_candidates,
            production_items,
        )

        validate_candidate_identity(
            translated_candidates,
            translated_items,
        )

        production_first_stage[
            record.question_id
        ] = (
            production_candidates
        )

        translated_first_stage[
            record.question_id
        ] = (
            translated_candidates
        )

        production_reranked[
            record.question_id
        ] = (
            reranked_results_only(
                production_items
            )
        )

        translated_reranked[
            record.question_id
        ] = (
            reranked_results_only(
                translated_items
            )
        )

    print()
    print(
        "=" * 104
    )

    print(
        "Primary-evidence rank movement"
    )

    print(
        "=" * 104
    )

    print(
        f"{'Question':<12}"
        f"{'ProdFirst':>14}"
        f"{'ProdBGE':>14}"
        f"{'TransFirst':>14}"
        f"{'TransBGE':>14}"
    )

    print(
        "-" * 104
    )

    for record in records:
        question_id = (
            record.question_id
        )

        production_first_rank = (
            first_primary_rank(
                record,
                production_first_stage[
                    question_id
                ],
            )
        )

        production_bge_rank = (
            first_primary_rank(
                record,
                production_reranked[
                    question_id
                ],
            )
        )

        translated_first_rank = (
            first_primary_rank(
                record,
                translated_first_stage[
                    question_id
                ],
            )
        )

        translated_bge_rank = (
            first_primary_rank(
                record,
                translated_reranked[
                    question_id
                ],
            )
        )

        print(
            f"{question_id:<12}"
            f"{format_rank(production_first_rank, depth=candidate_depth):>14}"
            f"{format_rank(production_bge_rank, depth=candidate_depth):>14}"
            f"{format_rank(translated_first_rank, depth=candidate_depth):>14}"
            f"{format_rank(translated_bge_rank, depth=candidate_depth):>14}"
        )

    for cutoff in (
        normalized_cutoffs
    ):
        production_first_metrics = (
            calculate_metrics(
                records,
                production_first_stage,
                cutoff=cutoff,
            )
        )

        production_bge_metrics = (
            calculate_metrics(
                records,
                production_reranked,
                cutoff=cutoff,
            )
        )

        translated_first_metrics = (
            calculate_metrics(
                records,
                translated_first_stage,
                cutoff=cutoff,
            )
        )

        translated_bge_metrics = (
            calculate_metrics(
                records,
                translated_reranked,
                cutoff=cutoff,
            )
        )

        print()
        print(
            "=" * 74
        )

        print(
            f"NE->EN candidate-pool + BGE metrics @ {cutoff}"
        )

        print(
            "=" * 74
        )

        print(
            f"{'Variant':<28}"
            f"{'Hit':>10}"
            f"{'MRR':>10}"
            f"{'Recall':>10}"
        )

        print(
            "-" * 74
        )

        print_metric_row(
            "production_first_stage",
            aggregate_metrics(
                production_first_metrics
            ),
        )

        print_metric_row(
            "production_bge",
            aggregate_metrics(
                production_bge_metrics
            ),
        )

        print_metric_row(
            "translated_first_stage",
            aggregate_metrics(
                translated_first_metrics
            ),
        )

        print_metric_row(
            "translated_bge",
            aggregate_metrics(
                translated_bge_metrics
            ),
        )


if __name__ == "__main__":
    run_comparison()