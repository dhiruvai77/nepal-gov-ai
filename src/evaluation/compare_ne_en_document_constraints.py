"""Diagnose cross-document competition in NE->EN contextual retrieval.

This evaluation targets the six Nepali-query -> English-evidence retrieval
questions in NepalGov AI.

For each question, the Nepali query is embedded exactly once. The same query
vector is then searched against the contextual dense representation in two
ways:

1. production-style:
   filter only by target language,

2. oracle document-constrained:
   additionally filter to the manually verified target document recorded in
   the retrieval benchmark.

The document-constrained variant is diagnostic only. It uses gold benchmark
metadata and must not be interpreted as a deployable production configuration.

Its purpose is to distinguish two failure modes:

- cross-document competition:
  the gold passage ranks well once the correct document is isolated,

- within-document semantic/chunk failure:
  the gold passage still ranks poorly even inside the correct document.

No production retrieval code is modified.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from qdrant_client import (
    QdrantClient,
)

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

NE_EN_PAIR = (
    "ne",
    "en",
)

DEFAULT_RETRIEVAL_DEPTH = 100

DEFAULT_CUTOFFS = (
    5,
    10,
    20,
)


def select_ne_en_records(
    records: Sequence[
        EvaluationRecord
    ],
) -> list[
    EvaluationRecord
]:
    """Return the controlled NE->EN retrieval slice in stable ID order."""

    selected = [
        record
        for record in records
        if (
            record.query_language,
            record.target_language,
        )
        == NE_EN_PAIR
    ]

    selected.sort(
        key=lambda record: (
            record.question_id
        )
    )

    return selected


def expected_document_id(
    record: EvaluationRecord,
) -> str:
    """Return the single gold document required by this diagnostic."""

    if (
        len(
            record.expected_document_ids
        )
        != 1
    ):
        raise ValueError(
            f"{record.question_id}: document-constraint "
            "diagnostic requires exactly one "
            "expected_document_id."
        )

    document_id = (
        record.expected_document_ids[
            0
        ].strip()
    )

    if not document_id:
        raise ValueError(
            f"{record.question_id}: expected document ID "
            "must contain non-whitespace text."
        )

    return document_id


def retrieve_contextual_with_vector(
    client: QdrantClient,
    query_vector: list[
        float
    ],
    *,
    filters: Mapping[
        str,
        str,
    ],
    top_k: int,
) -> list[
    RetrievalResult
]:
    """Search contextual dense vectors with an already-computed query vector."""

    if not query_vector:
        raise ValueError(
            "query_vector cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    query_filter = (
        build_metadata_filter(
            dict(
                filters
            )
        )
    )

    response = (
        client.query_points(
            collection_name=(
                COLLECTION_NAME
            ),
            query=(
                query_vector
            ),
            using=(
                CONTEXTUAL_DENSE_VECTOR_NAME
            ),
            query_filter=(
                query_filter
            ),
            limit=(
                top_k
            ),
            with_payload=True,
        )
    )

    return [
        normalize_search_result(
            point
        )
        for point in (
            response.points
        )
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
    """Render a missing primary passage consistently."""

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
    """Evaluate cached rankings with the project's standard metric contract."""

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
    """Print one aggregate retrieval metric row."""

    print(
        f"{label:<24}"
        f"{metrics['hit_rate']:>10.3f}"
        f"{metrics['mrr']:>10.3f}"
        f"{metrics['recall']:>10.3f}"
    )


def run_comparison(
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    *,
    retrieval_depth: int = (
        DEFAULT_RETRIEVAL_DEPTH
    ),
    cutoffs: Sequence[
        int
    ] = DEFAULT_CUTOFFS,
) -> None:
    """Run the six-question contextual document-constraint diagnostic."""

    if retrieval_depth <= 0:
        raise ValueError(
            "retrieval_depth must be greater than zero."
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
        > retrieval_depth
    ):
        raise ValueError(
            "retrieval_depth must be at least "
            "the largest evaluation cutoff."
        )

    all_records = (
        load_evaluation_records(
            dataset_path
        )
    )

    records = (
        select_ne_en_records(
            all_records
        )
    )

    if not records:
        raise ValueError(
            "No NE->EN evaluation records found."
        )

    # Validate the oracle-document assumption before making hosted
    # embedding calls.
    for record in records:
        expected_document_id(
            record
        )

    client = (
        QdrantClient(
            url=QDRANT_URL
        )
    )

    embedding_service = (
        HuggingFaceE5EmbeddingService()
    )

    production_rankings: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    constrained_rankings: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    print(
        "NE->EN contextual document-constraint diagnostic"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Retrieval depth: {retrieval_depth}"
    )

    print(
        "Production diagnostic = language filter only"
    )

    print(
        "Oracle diagnostic = language + gold document filter"
    )

    print(
        "Each Nepali query is embedded once and reused for both searches."
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

        query_vector = (
            embedding_service.embed_query(
                record.query
            )
        )

        if (
            len(
                query_vector
            )
            != embedding_service.dimension
        ):
            raise RuntimeError(
                "Unexpected query embedding dimension "
                f"for {record.question_id}: "
                f"{len(query_vector)}."
            )

        production_rankings[
            record.question_id
        ] = (
            retrieve_contextual_with_vector(
                client,
                query_vector,
                filters={
                    "language": "en",
                },
                top_k=(
                    retrieval_depth
                ),
            )
        )

        constrained_rankings[
            record.question_id
        ] = (
            retrieve_contextual_with_vector(
                client,
                query_vector,
                filters={
                    "language": "en",
                    "document_id": (
                        expected_document_id(
                            record
                        )
                    ),
                },
                top_k=(
                    retrieval_depth
                ),
            )
        )

    print()
    print(
        "=" * 72
    )

    print(
        "Primary-evidence ranks"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Question':<12}"
        f"{'Production':>16}"
        f"{'Gold-document':>18}"
        f"{'Movement':>14}"
    )

    print(
        "-" * 72
    )

    for record in records:
        production_rank = (
            first_primary_rank(
                record,
                production_rankings[
                    record.question_id
                ],
            )
        )

        constrained_rank = (
            first_primary_rank(
                record,
                constrained_rankings[
                    record.question_id
                ],
            )
        )

        if (
            production_rank is not None
            and constrained_rank is not None
        ):
            movement = (
                production_rank
                - constrained_rank
            )

            movement_text = (
                f"{movement:+d}"
            )

        elif (
            production_rank is None
            and constrained_rank is not None
        ):
            movement_text = (
                "recovered"
            )

        elif (
            production_rank is not None
            and constrained_rank is None
        ):
            movement_text = (
                "lost"
            )

        else:
            movement_text = (
                "both >depth"
            )

        print(
            f"{record.question_id:<12}"
            f"{format_rank(production_rank, depth=retrieval_depth):>16}"
            f"{format_rank(constrained_rank, depth=retrieval_depth):>18}"
            f"{movement_text:>14}"
        )

    for cutoff in (
        normalized_cutoffs
    ):
        production_metrics = (
            calculate_metrics(
                records,
                production_rankings,
                cutoff=cutoff,
            )
        )

        constrained_metrics = (
            calculate_metrics(
                records,
                constrained_rankings,
                cutoff=cutoff,
            )
        )

        production_summary = (
            aggregate_metrics(
                production_metrics
            )
        )

        constrained_summary = (
            aggregate_metrics(
                constrained_metrics
            )
        )

        print()
        print(
            "=" * 64
        )

        print(
            f"NE->EN document-constraint metrics @ {cutoff}"
        )

        print(
            "=" * 64
        )

        print(
            f"{'Variant':<24}"
            f"{'Hit':>10}"
            f"{'MRR':>10}"
            f"{'Recall':>10}"
        )

        print(
            "-" * 64
        )

        print_metric_row(
            "production",
            production_summary,
        )

        print_metric_row(
            "gold_document_oracle",
            constrained_summary,
        )


if __name__ == "__main__":
    run_comparison()