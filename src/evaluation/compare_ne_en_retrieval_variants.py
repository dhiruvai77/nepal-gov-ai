"""Targeted NE->EN retrieval experiment for NepalGov AI.

This evaluation isolates the weakest production retrieval direction:
Nepali queries retrieving English evidence.

It compares four strategies over the same six manually verified NE->EN
benchmark questions:

1. production:
   original Nepali query -> English contextual dense retrieval,

2. controlled English dense:
   manually controlled English counterpart -> the same contextual dense corpus,

3. translated hybrid:
   controlled English query -> contextual dense + BM25 -> RRF, using a deeper
   fixed component pool of 100 results and returning a final top-20 ranking,

4. dual route:
   RRF of the original Nepali contextual ranking and the translated-hybrid
   ranking.

The controlled English formulations are evaluation controls. This experiment
does not add translation to the production application.

The translated-hybrid component depth of 100 was selected after a targeted
candidate-depth diagnostic on the persistent ne_en_005 failure. At final
top-20, component depth 60 did not recover the gold evidence, while depth 100
recovered it at rank 13. Deeper pools were not monotonically better under RRF.

No production retrieval code is modified.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

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
    QDRANT_URL,
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

NE_EN_PAIR = (
    "ne",
    "en",
)

DEFAULT_CUTOFFS = (
    5,
    10,
    20,
)

# Deep enough to inspect persistent dense failures while allowing all @5/@10/
# @20 metrics to come from one stable ranking.
DEFAULT_RETRIEVAL_DEPTH = 100

# The proposed translated-hybrid experiment still returns only the 20
# candidates available to the production reranker. The deeper value applies
# only to the component rankings fused before that final truncation.
TRANSLATED_HYBRID_COMPONENT_DEPTH = 100

DEFAULT_FINAL_CANDIDATE_DEPTH = 20

RRF_K = 60


CONTROLLED_ENGLISH_QUERIES = {
    "ne_en_001": (
        "What does the Constitution of Nepal guarantee "
        "regarding the right to health?"
    ),
    "ne_en_002": (
        "What does the Constitution say about "
        "the right to information?"
    ),
    "ne_en_003": (
        "What does the Public Health Service Act say about "
        "emergency health services?"
    ),
    "ne_en_004": (
        "What does the Public Health Service Act say about "
        "informed consent for treatment?"
    ),
    "ne_en_005": (
        "What does the Economic Survey report about "
        "Nepal's economic growth?"
    ),
    "ne_en_006": (
        "What does the 2025/26 Budget Speech say about "
        "scholarships for students?"
    ),
}


def select_ne_en_records(
    records: Sequence[
        EvaluationRecord
    ],
) -> list[
    EvaluationRecord
]:
    """Return the six controlled Nepali-query -> English-evidence records."""

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
        key=lambda item: (
            item.question_id
        )
    )

    return selected


def validate_controlled_queries(
    records: Sequence[
        EvaluationRecord
    ],
    controlled_queries: Mapping[
        str,
        str,
    ] = CONTROLLED_ENGLISH_QUERIES,
) -> None:
    """Ensure every NE->EN record has exactly one controlled English query."""

    expected_ids = {
        record.question_id
        for record in records
    }

    controlled_ids = set(
        controlled_queries
    )

    if (
        controlled_ids
        != expected_ids
    ):
        missing = sorted(
            expected_ids
            - controlled_ids
        )

        unexpected = sorted(
            controlled_ids
            - expected_ids
        )

        raise ValueError(
            "Controlled English query IDs do not match "
            "the NE->EN benchmark records. "
            f"Missing={missing}; unexpected={unexpected}."
        )

    for (
        question_id,
        query,
    ) in controlled_queries.items():
        if (
            not isinstance(
                query,
                str,
            )
            or not query.strip()
        ):
            raise ValueError(
                "Controlled English query "
                f"{question_id!r} must contain "
                "non-whitespace text."
            )


def reciprocal_rank_fuse(
    ranked_lists: Sequence[
        Sequence[
            RetrievalResult
        ]
    ],
    *,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[
    RetrievalResult
]:
    """Fuse ranked result lists using deterministic Reciprocal Rank Fusion."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if rrf_k < 0:
        raise ValueError(
            "rrf_k must be zero or greater."
        )

    results_by_id: dict[
        str,
        RetrievalResult,
    ] = {}

    scores: dict[
        str,
        float,
    ] = {}

    for results in ranked_lists:
        for (
            rank,
            result,
        ) in enumerate(
            results,
            start=1,
        ):
            results_by_id.setdefault(
                result.point_id,
                result,
            )

            scores[
                result.point_id
            ] = (
                scores.get(
                    result.point_id,
                    0.0,
                )
                + (
                    1.0
                    / (
                        rrf_k
                        + rank
                    )
                )
            )

    ranked_ids = sorted(
        scores,
        key=lambda point_id: (
            -scores[
                point_id
            ],
            point_id,
        ),
    )

    fused: list[
        RetrievalResult
    ] = []

    for point_id in ranked_ids[
        :top_k
    ]:
        source = (
            results_by_id[
                point_id
            ]
        )

        fused.append(
            RetrievalResult(
                point_id=(
                    source.point_id
                ),
                score=(
                    scores[
                        point_id
                    ]
                ),
                chunk_id=(
                    source.chunk_id
                ),
                document_id=(
                    source.document_id
                ),
                title=(
                    source.title
                ),
                organization=(
                    source.organization
                ),
                language=(
                    source.language
                ),
                page_start=(
                    source.page_start
                ),
                page_end=(
                    source.page_end
                ),
                source_url=(
                    source.source_url
                ),
                chunk_text=(
                    source.chunk_text
                ),
                chunk_index=(
                    source.chunk_index
                ),
                token_count=(
                    source.token_count
                ),
                category=(
                    source.category
                ),
                document_type=(
                    source.document_type
                ),
                publication_date=(
                    source.publication_date
                ),
                section=(
                    source.section
                ),
                subsection=(
                    source.subsection
                ),
                article_number=(
                    source.article_number
                ),
                article_title=(
                    source.article_title
                ),
                extraction_method=(
                    source.extraction_method
                ),
            )
        )

    return fused


def build_translated_hybrid(
    controlled_dense: Sequence[
        RetrievalResult
    ],
    sparse: Sequence[
        RetrievalResult
    ],
    *,
    component_depth: int = (
        TRANSLATED_HYBRID_COMPONENT_DEPTH
    ),
    final_depth: int = (
        DEFAULT_FINAL_CANDIDATE_DEPTH
    ),
) -> list[
    RetrievalResult
]:
    """Fuse translated-query dense and BM25 component rankings.

    Component retrieval may search more deeply than the final returned pool.
    The final candidate count remains fixed so a later reranker benchmark can
    compare candidate sets at the same production-facing depth.
    """

    if component_depth <= 0:
        raise ValueError(
            "component_depth must be greater than zero."
        )

    if final_depth <= 0:
        raise ValueError(
            "final_depth must be greater than zero."
        )

    if component_depth < final_depth:
        raise ValueError(
            "component_depth must be at least final_depth."
        )

    return reciprocal_rank_fuse(
        [
            controlled_dense[
                :component_depth
            ],
            sparse[
                :component_depth
            ],
        ],
        top_k=final_depth,
    )


def build_dual_route_fusion(
    production: Sequence[
        RetrievalResult
    ],
    translated_hybrid: Sequence[
        RetrievalResult
    ],
    *,
    final_depth: int = (
        DEFAULT_FINAL_CANDIDATE_DEPTH
    ),
) -> list[
    RetrievalResult
]:
    """Fuse original cross-lingual and translated-hybrid rankings."""

    if final_depth <= 0:
        raise ValueError(
            "final_depth must be greater than zero."
        )

    return reciprocal_rank_fuse(
        [
            production,
            translated_hybrid,
        ],
        top_k=final_depth,
    )


def first_primary_rank(
    record: EvaluationRecord,
    results: Sequence[
        RetrievalResult
    ],
) -> int | None:
    """Return rank of the first manually verified primary evidence passage."""

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
    """Render absent primary evidence consistently."""

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
    """Evaluate cached rankings with the existing project metric contract."""

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
    """Print one compact aggregate retrieval row."""

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
    translated_component_depth: int = (
        TRANSLATED_HYBRID_COMPONENT_DEPTH
    ),
    final_candidate_depth: int = (
        DEFAULT_FINAL_CANDIDATE_DEPTH
    ),
    cutoffs: Sequence[
        int
    ] = DEFAULT_CUTOFFS,
) -> None:
    """Run the targeted six-question NE->EN retrieval experiment."""

    if retrieval_depth <= 0:
        raise ValueError(
            "retrieval_depth must be greater than zero."
        )

    if translated_component_depth <= 0:
        raise ValueError(
            "translated_component_depth must be greater than zero."
        )

    if final_candidate_depth <= 0:
        raise ValueError(
            "final_candidate_depth must be greater than zero."
        )

    if (
        translated_component_depth
        < final_candidate_depth
    ):
        raise ValueError(
            "translated_component_depth must be at least "
            "final_candidate_depth."
        )

    if (
        retrieval_depth
        < translated_component_depth
    ):
        raise ValueError(
            "retrieval_depth must be at least "
            "translated_component_depth."
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
        > final_candidate_depth
    ):
        raise ValueError(
            "final_candidate_depth must be at least "
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

    contextual_dense = (
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

    production_rankings: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    english_dense_rankings: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    translated_hybrid_rankings: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    dual_route_rankings: dict[
        str,
        list[
            RetrievalResult
        ],
    ] = {}

    print(
        "Targeted NE->EN retrieval benchmark"
    )

    print(
        f"Questions: {len(records)}"
    )

    print(
        f"Dense diagnostic depth: {retrieval_depth}"
    )

    print(
        "Translated-hybrid component depth: "
        f"{translated_component_depth}"
    )

    print(
        "Final candidate depth: "
        f"{final_candidate_depth}"
    )

    print(
        "Production = Nepali query -> contextual English dense"
    )

    print(
        "English dense = controlled English query -> contextual dense"
    )

    print(
        "Translated hybrid = controlled English contextual dense "
        "+ English BM25 -> RRF"
    )

    print(
        "Dual route = original Nepali contextual ranking "
        "+ translated hybrid -> RRF"
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

        production = (
            contextual_dense.retrieve(
                query=(
                    record.query
                ),
                top_k=(
                    retrieval_depth
                ),
                filters=(
                    filters
                ),
            )
        )

        controlled_dense = (
            contextual_dense.retrieve(
                query=(
                    controlled_query
                ),
                top_k=(
                    retrieval_depth
                ),
                filters=(
                    filters
                ),
            )
        )

        sparse = (
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

        translated_hybrid = (
            build_translated_hybrid(
                controlled_dense,
                sparse,
                component_depth=(
                    translated_component_depth
                ),
                final_depth=(
                    final_candidate_depth
                ),
            )
        )

        dual_route = (
            build_dual_route_fusion(
                production[
                    :translated_component_depth
                ],
                translated_hybrid,
                final_depth=(
                    final_candidate_depth
                ),
            )
        )

        production_rankings[
            record.question_id
        ] = production

        english_dense_rankings[
            record.question_id
        ] = controlled_dense

        translated_hybrid_rankings[
            record.question_id
        ] = translated_hybrid

        dual_route_rankings[
            record.question_id
        ] = dual_route

    print()
    print(
        "=" * 96
    )

    print(
        "Primary-evidence ranks"
    )

    print(
        "=" * 96
    )

    print(
        f"{'Question':<12}"
        f"{'Production':>14}"
        f"{'EnglishDense':>14}"
        f"{'TransHybrid':>14}"
        f"{'DualRoute':>14}"
    )

    print(
        "-" * 96
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

        english_rank = (
            first_primary_rank(
                record,
                english_dense_rankings[
                    record.question_id
                ],
            )
        )

        translated_rank = (
            first_primary_rank(
                record,
                translated_hybrid_rankings[
                    record.question_id
                ],
            )
        )

        dual_route_rank = (
            first_primary_rank(
                record,
                dual_route_rankings[
                    record.question_id
                ],
            )
        )

        print(
            f"{record.question_id:<12}"
            f"{format_rank(production_rank, depth=retrieval_depth):>14}"
            f"{format_rank(english_rank, depth=retrieval_depth):>14}"
            f"{format_rank(translated_rank, depth=final_candidate_depth):>14}"
            f"{format_rank(dual_route_rank, depth=final_candidate_depth):>14}"
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

        english_metrics = (
            calculate_metrics(
                records,
                english_dense_rankings,
                cutoff=cutoff,
            )
        )

        translated_metrics = (
            calculate_metrics(
                records,
                translated_hybrid_rankings,
                cutoff=cutoff,
            )
        )

        dual_route_metrics = (
            calculate_metrics(
                records,
                dual_route_rankings,
                cutoff=cutoff,
            )
        )

        print()
        print(
            "=" * 68
        )

        print(
            f"NE->EN retrieval metrics @ {cutoff}"
        )

        print(
            "=" * 68
        )

        print(
            f"{'Variant':<24}"
            f"{'Hit':>10}"
            f"{'MRR':>10}"
            f"{'Recall':>10}"
        )

        print(
            "-" * 68
        )

        print_metric_row(
            "production",
            aggregate_metrics(
                production_metrics
            ),
        )

        print_metric_row(
            "controlled_english",
            aggregate_metrics(
                english_metrics
            ),
        )

        print_metric_row(
            "translated_hybrid",
            aggregate_metrics(
                translated_metrics
            ),
        )

        print_metric_row(
            "dual_route_rrf",
            aggregate_metrics(
                dual_route_metrics
            ),
        )


if __name__ == "__main__":
    run_comparison()