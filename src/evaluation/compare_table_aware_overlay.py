"""Benchmark a table-aware contextual-vector overlay without modifying Qdrant.

The experiment targets English Economic Survey evidence because previous
diagnostics identified a persistent table-heavy retrieval failure in that
document.

Current production behavior remains unchanged.

Simulation:

1. retrieve the complete current English contextual-dense ranking,
2. identify Economic Survey passages containing explicit Annex/Table/Chart
   headings,
3. build and embed deterministic table-aware representations for those passages,
4. locally replace only those contextual similarity scores,
5. rerank the complete English corpus,
6. compare production contextual retrieval against the table-aware overlay.

Because every non-table passage keeps its current contextual score, this
simulates selectively replacing the contextual vector for detected table-like
Economic Survey passages without performing a Qdrant backfill.

The benchmark includes:

- en_en_004: schools and students,
- en_en_005: SEE results,
- en_en_006: literacy,
- en_en_011: economic growth,
- ne_en_005: Nepali query for the same English economic-growth evidence.

No production vectors are changed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import (
    QdrantClient,
)

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.embeddings.table_aware_passage import (
    build_table_aware_passage_text,
    extract_table_hints,
    is_table_like_passage,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    CONTEXTUAL_DENSE_VECTOR_NAME,
    QDRANT_URL,
)
from src.retrieval.dense_retriever import (
    build_metadata_filter,
)


BENCHMARK_SCHEMA_VERSION = 1

BENCHMARK_CONFIG_ID = (
    "table-aware-economic-survey-overlay-v1"
)

DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "table_aware_economic_survey_overlay_v1.json"
)

TARGET_DOCUMENT_ID = (
    "economic_survey_2023_24_en"
)

TARGET_LANGUAGE = "en"

TARGET_QUESTION_IDS = (
    "en_en_004",
    "en_en_005",
    "en_en_006",
    "en_en_011",
    "ne_en_005",
)

DEFAULT_CUTOFFS = (
    5,
    10,
    20,
)

DEFAULT_SCROLL_PAGE_SIZE = 256

DEFAULT_INSPECTION_LIMIT = 12


@dataclass(frozen=True)
class RankedPoint:
    """One point ID and its dense retrieval similarity score."""

    point_id: str
    score: float


def select_target_records(
    records: Sequence[
        EvaluationRecord
    ],
) -> list[
    EvaluationRecord
]:
    """Return the fixed English Economic Survey diagnostic slice."""

    records_by_id = {
        record.question_id: record
        for record in records
    }

    missing = [
        question_id
        for question_id
        in TARGET_QUESTION_IDS
        if (
            question_id
            not in records_by_id
        )
    ]

    if missing:
        raise ValueError(
            "Retrieval benchmark is missing "
            "target questions: "
            + ", ".join(
                missing
            )
        )

    selected = [
        records_by_id[
            question_id
        ]
        for question_id
        in TARGET_QUESTION_IDS
    ]

    for record in selected:
        if (
            record.target_language
            != TARGET_LANGUAGE
        ):
            raise ValueError(
                f"{record.question_id}: "
                "target language is not English."
            )

        if (
            TARGET_DOCUMENT_ID
            not in record.expected_document_ids
        ):
            raise ValueError(
                f"{record.question_id}: "
                "target Economic Survey document "
                "is not an expected document."
            )

    return selected


def scroll_payloads(
    client: Any,
    *,
    filters: Mapping[
        str,
        str,
    ],
    page_size: int = (
        DEFAULT_SCROLL_PAGE_SIZE
    ),
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load all matching Qdrant payloads without retrieving vectors."""

    if page_size <= 0:
        raise ValueError(
            "page_size must be greater than zero."
        )

    query_filter = (
        build_metadata_filter(
            dict(
                filters
            )
        )
    )

    payloads: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    offset: Any | None = None

    while True:
        (
            points,
            next_offset,
        ) = client.scroll(
            collection_name=(
                COLLECTION_NAME
            ),
            scroll_filter=(
                query_filter
            ),
            limit=(
                page_size
            ),
            offset=(
                offset
            ),
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            point_id = str(
                point.id
            )

            if (
                point_id
                in payloads
            ):
                raise RuntimeError(
                    "Qdrant scroll returned "
                    f"duplicate point ID "
                    f"{point_id!r}."
                )

            payload = (
                point.payload
                or {}
            )

            if not isinstance(
                payload,
                Mapping,
            ):
                raise RuntimeError(
                    "Qdrant point payload must "
                    "be a mapping."
                )

            payloads[
                point_id
            ] = dict(
                payload
            )

        if (
            next_offset
            is None
        ):
            break

        offset = (
            next_offset
        )

    if not payloads:
        raise RuntimeError(
            "Qdrant scroll returned no points."
        )

    return payloads


def select_table_like_payloads(
    payloads_by_id: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Return detected table-like passages from the target survey document."""

    selected: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for (
        point_id,
        payload,
    ) in payloads_by_id.items():
        if (
            payload.get(
                "document_id"
            )
            != TARGET_DOCUMENT_ID
        ):
            continue

        chunk_text = str(
            payload.get(
                "chunk_text"
            )
            or ""
        )

        if not (
            is_table_like_passage(
                chunk_text
            )
        ):
            continue

        selected[
            point_id
        ] = dict(
            payload
        )

    return selected


def normalized_dot_product(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate cosine-equivalent score for normalized E5 vectors."""

    if (
        len(
            left
        )
        != len(
            right
        )
    ):
        raise ValueError(
            "Vectors must have equal dimension."
        )

    if not left:
        raise ValueError(
            "Vectors cannot be empty."
        )

    return float(
        sum(
            a * b
            for (
                a,
                b,
            ) in zip(
                left,
                right,
            )
        )
    )


def retrieve_complete_contextual_ranking(
    client: Any,
    query_vector: list[float],
    *,
    corpus_size: int,
) -> list[
    RankedPoint
]:
    """Retrieve every English point using the current contextual vector."""

    if corpus_size <= 0:
        raise ValueError(
            "corpus_size must be greater than zero."
        )

    query_filter = (
        build_metadata_filter(
            {
                "language": (
                    TARGET_LANGUAGE
                ),
            }
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
                corpus_size
            ),
            with_payload=False,
        )
    )

    ranking = [
        RankedPoint(
            point_id=str(
                point.id
            ),
            score=float(
                point.score
            ),
        )
        for point in (
            response.points
        )
    ]

    if (
        len(
            ranking
        )
        != corpus_size
    ):
        raise RuntimeError(
            "Complete contextual retrieval returned "
            f"{len(ranking)} points; expected "
            f"{corpus_size}."
        )

    return ranking


def build_table_aware_overlay_ranking(
    contextual_ranking: Sequence[
        RankedPoint
    ],
    *,
    query_vector: Sequence[float],
    table_vectors_by_id: Mapping[
        str,
        Sequence[float],
    ],
) -> list[
    RankedPoint
]:
    """Replace table-like contextual scores and deterministically rerank."""

    baseline_ids = {
        item.point_id
        for item in contextual_ranking
    }

    missing_table_ids = (
        set(
            table_vectors_by_id
        )
        - baseline_ids
    )

    if missing_table_ids:
        raise ValueError(
            "Table-aware vectors contain IDs "
            "outside the contextual ranking: "
            + ", ".join(
                sorted(
                    missing_table_ids
                )
            )
        )

    overlay: list[
        RankedPoint
    ] = []

    for item in contextual_ranking:
        replacement = (
            table_vectors_by_id.get(
                item.point_id
            )
        )

        if replacement is None:
            score = (
                item.score
            )

        else:
            score = (
                normalized_dot_product(
                    query_vector,
                    replacement,
                )
            )

        overlay.append(
            RankedPoint(
                point_id=(
                    item.point_id
                ),
                score=(
                    score
                ),
            )
        )

    return sorted(
        overlay,
        key=lambda item: (
            -item.score,
            item.point_id,
        ),
    )


def primary_rank(
    record: EvaluationRecord,
    ranking: Sequence[
        RankedPoint
    ],
) -> int | None:
    """Return the first manually verified primary-evidence rank."""

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    for (
        rank,
        item,
    ) in enumerate(
        ranking,
        start=1,
    ):
        if (
            item.point_id
            in primary_ids
        ):
            return rank

    return None


def evaluate_ranking(
    record: EvaluationRecord,
    ranking: Sequence[
        RankedPoint
    ],
    *,
    cutoff: int,
) -> dict[
    str,
    float,
]:
    """Calculate Hit, MRR, and Recall using the existing benchmark semantics."""

    if cutoff <= 0:
        raise ValueError(
            "cutoff must be greater than zero."
        )

    top_ids = [
        item.point_id
        for item in (
            ranking[
                :cutoff
            ]
        )
    ]

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    relevant_ids = set(
        record.relevant_chunk_ids
    )

    hit_rate = float(
        any(
            point_id
            in primary_ids
            for point_id
            in top_ids
        )
    )

    reciprocal_rank = 0.0

    for (
        rank,
        point_id,
    ) in enumerate(
        top_ids,
        start=1,
    ):
        if (
            point_id
            in primary_ids
        ):
            reciprocal_rank = (
                1.0
                / rank
            )

            break

    retrieved_relevant = len(
        set(
            top_ids
        )
        & relevant_ids
    )

    recall = (
        retrieved_relevant
        / len(
            relevant_ids
        )
    )

    return {
        "hit_rate": (
            hit_rate
        ),
        "mrr": (
            reciprocal_rank
        ),
        "recall": (
            recall
        ),
    }


def aggregate_metrics(
    metrics: Sequence[
        Mapping[
            str,
            float,
        ]
    ],
) -> dict[
    str,
    float,
]:
    """Average one list of per-query retrieval metrics."""

    if not metrics:
        raise ValueError(
            "At least one metric row is required."
        )

    count = len(
        metrics
    )

    return {
        "hit_rate": sum(
            row[
                "hit_rate"
            ]
            for row in metrics
        )
        / count,
        "mrr": sum(
            row[
                "mrr"
            ]
            for row in metrics
        )
        / count,
        "recall": sum(
            row[
                "recall"
            ]
            for row in metrics
        )
        / count,
    }


def build_summary(
    records: Sequence[
        EvaluationRecord
    ],
    contextual_rankings: Mapping[
        str,
        Sequence[
            RankedPoint
        ],
    ],
    overlay_rankings: Mapping[
        str,
        Sequence[
            RankedPoint
        ],
    ],
    *,
    english_corpus_count: int,
    target_document_count: int,
    table_like_count: int,
    table_hints_by_id: Mapping[
        str,
        Sequence[str],
    ],
    cutoffs: Sequence[int] = (
        DEFAULT_CUTOFFS
    ),
) -> dict[
    str,
    Any,
]:
    """Build one deterministic benchmark result artifact."""

    expected_ids = {
        record.question_id
        for record in records
    }

    if (
        set(
            contextual_rankings
        )
        != expected_ids
        or set(
            overlay_rankings
        )
        != expected_ids
    ):
        raise ValueError(
            "Benchmark rankings are incomplete."
        )

    primary_ranks: dict[
        str,
        dict[
            str,
            int | None,
        ],
    ] = {}

    for record in records:
        primary_ranks[
            record.question_id
        ] = {
            "contextual": (
                primary_rank(
                    record,
                    contextual_rankings[
                        record.question_id
                    ],
                )
            ),
            "table_aware_overlay": (
                primary_rank(
                    record,
                    overlay_rankings[
                        record.question_id
                    ],
                )
            ),
        }

    metrics_by_cutoff: dict[
        str,
        dict[
            str,
            dict[
                str,
                float,
            ],
        ],
    ] = {}

    for cutoff in cutoffs:
        contextual_metrics = [
            evaluate_ranking(
                record,
                contextual_rankings[
                    record.question_id
                ],
                cutoff=cutoff,
            )
            for record in records
        ]

        overlay_metrics = [
            evaluate_ranking(
                record,
                overlay_rankings[
                    record.question_id
                ],
                cutoff=cutoff,
            )
            for record in records
        ]

        metrics_by_cutoff[
            str(
                cutoff
            )
        ] = {
            "contextual": (
                aggregate_metrics(
                    contextual_metrics
                )
            ),
            "table_aware_overlay": (
                aggregate_metrics(
                    overlay_metrics
                )
            ),
        }

    return {
        "schema_version": (
            BENCHMARK_SCHEMA_VERSION
        ),
        "run_config_id": (
            BENCHMARK_CONFIG_ID
        ),
        "target_document_id": (
            TARGET_DOCUMENT_ID
        ),
        "target_language": (
            TARGET_LANGUAGE
        ),
        "question_ids": [
            record.question_id
            for record in records
        ],
        "question_count": len(
            records
        ),
        "english_corpus_point_count": (
            english_corpus_count
        ),
        "target_document_point_count": (
            target_document_count
        ),
        "table_like_point_count": (
            table_like_count
        ),
        "simulation": {
            "baseline_vector": (
                CONTEXTUAL_DENSE_VECTOR_NAME
            ),
            "replacement_scope": (
                "detected table-like passages "
                "inside economic_survey_2023_24_en"
            ),
            "non_table_vectors_changed": False,
            "qdrant_modified": False,
        },
        "primary_ranks": (
            primary_ranks
        ),
        "metrics": (
            metrics_by_cutoff
        ),
        "table_hints_for_primary_evidence": {
            point_id: list(
                hints
            )
            for (
                point_id,
                hints,
            ) in (
                table_hints_by_id.items()
            )
        },
        "production_changed": False,
    }


def write_json_atomic(
    path: str | Path,
    data: Mapping[
        str,
        Any,
    ],
) -> None:
    """Write benchmark output atomically."""

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        destination.with_suffix(
            destination.suffix
            + ".tmp"
        )
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            dict(
                data
            ),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    temporary.replace(
        destination
    )


def print_table_candidate_inspection(
    table_payloads: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
    *,
    limit: int = (
        DEFAULT_INSPECTION_LIMIT
    ),
) -> None:
    """Print a small deterministic sample before any embedding call."""

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero."
        )

    print()
    print(
        "=" * 100
    )

    print(
        "Detected table-like Economic Survey passages"
    )

    print(
        "=" * 100
    )

    for (
        index,
        point_id,
    ) in enumerate(
        sorted(
            table_payloads
        )[
            :limit
        ],
        start=1,
    ):
        payload = (
            table_payloads[
                point_id
            ]
        )

        hints = (
            extract_table_hints(
                str(
                    payload.get(
                        "chunk_text"
                    )
                    or ""
                )
            )
        )

        print()
        print(
            f"{index}. {point_id}"
        )

        print(
            "Pages: "
            f"{payload.get('page_start')}"
            "-"
            f"{payload.get('page_end')}"
        )

        for hint in hints:
            print(
                f"  - {hint}"
            )


def _format_rank(
    rank: int | None,
    *,
    depth: int,
) -> str:
    """Render missing ranks explicitly."""

    if rank is None:
        return (
            f">{depth}"
        )

    return str(
        rank
    )


def _print_metric_row(
    label: str,
    metrics: Mapping[
        str,
        float,
    ],
) -> None:
    """Print one aggregate metric row."""

    print(
        f"{label:<24}"
        f"{metrics['hit_rate']:>10.3f}"
        f"{metrics['mrr']:>10.3f}"
        f"{metrics['recall']:>10.3f}"
    )


def print_summary(
    summary: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print benchmark ranks and metrics."""

    print()
    print(
        "=" * 86
    )

    print(
        "Table-aware Economic Survey overlay benchmark"
    )

    print(
        "=" * 86
    )

    print(
        "English corpus points: "
        f"{summary['english_corpus_point_count']}"
    )

    print(
        "Economic Survey points: "
        f"{summary['target_document_point_count']}"
    )

    print(
        "Detected table-like points: "
        f"{summary['table_like_point_count']}"
    )

    print()
    print(
        f"{'Question':<14}"
        f"{'Contextual':>14}"
        f"{'TableAware':>14}"
        f"{'Movement':>14}"
    )

    print(
        "-" * 58
    )

    primary_ranks = (
        summary[
            "primary_ranks"
        ]
    )

    depth = (
        summary[
            "english_corpus_point_count"
        ]
    )

    for question_id in (
        summary[
            "question_ids"
        ]
    ):
        ranks = (
            primary_ranks[
                question_id
            ]
        )

        contextual = (
            ranks[
                "contextual"
            ]
        )

        table_aware = (
            ranks[
                "table_aware_overlay"
            ]
        )

        if (
            contextual is not None
            and table_aware is not None
        ):
            movement = (
                contextual
                - table_aware
            )

            movement_text = (
                f"{movement:+d}"
            )

        elif (
            contextual is None
            and table_aware is not None
        ):
            movement_text = (
                "recovered"
            )

        elif (
            contextual is not None
            and table_aware is None
        ):
            movement_text = (
                "lost"
            )

        else:
            movement_text = (
                "both >depth"
            )

        print(
            f"{question_id:<14}"
            f"{_format_rank(contextual, depth=depth):>14}"
            f"{_format_rank(table_aware, depth=depth):>14}"
            f"{movement_text:>14}"
        )

    for cutoff in (
        DEFAULT_CUTOFFS
    ):
        metrics = (
            summary[
                "metrics"
            ][
                str(
                    cutoff
                )
            ]
        )

        print()
        print(
            "=" * 66
        )

        print(
            f"Retrieval metrics @ {cutoff}"
        )

        print(
            "=" * 66
        )

        print(
            f"{'Variant':<24}"
            f"{'Hit':>10}"
            f"{'MRR':>10}"
            f"{'Recall':>10}"
        )

        print(
            "-" * 66
        )

        _print_metric_row(
            "contextual",
            metrics[
                "contextual"
            ],
        )

        _print_metric_row(
            "table_aware_overlay",
            metrics[
                "table_aware_overlay"
            ],
        )


def run_benchmark(
    *,
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    inspect_only: bool = False,
) -> dict[
    str,
    Any,
] | None:
    """Run the selective table-aware contextual-vector simulation."""

    records = (
        select_target_records(
            load_evaluation_records(
                dataset_path
            )
        )
    )

    client = (
        QdrantClient(
            url=QDRANT_URL
        )
    )

    try:
        english_payloads = (
            scroll_payloads(
                client,
                filters={
                    "language": (
                        TARGET_LANGUAGE
                    ),
                },
            )
        )

        target_payloads = {
            point_id: payload
            for (
                point_id,
                payload,
            ) in (
                english_payloads.items()
            )
            if (
                payload.get(
                    "document_id"
                )
                == TARGET_DOCUMENT_ID
            )
        }

        if not target_payloads:
            raise RuntimeError(
                "No English Economic Survey "
                "points were found."
            )

        table_payloads = (
            select_table_like_payloads(
                english_payloads
            )
        )

        if not table_payloads:
            raise RuntimeError(
                "No table-like Economic Survey "
                "passages were detected."
            )

        print(
            "Table-aware Economic Survey experiment"
        )

        print(
            "Questions: "
            f"{len(records)}"
        )

        print(
            "English corpus points: "
            f"{len(english_payloads)}"
        )

        print(
            "Economic Survey points: "
            f"{len(target_payloads)}"
        )

        print(
            "Detected table-like points: "
            f"{len(table_payloads)}"
        )

        print_table_candidate_inspection(
            table_payloads
        )

        if inspect_only:
            print()
            print(
                "Inspection only: no embedding "
                "requests were made."
            )

            return None

        embedding_service = (
            HuggingFaceE5EmbeddingService()
        )

        ordered_table_ids = sorted(
            table_payloads
        )

        table_texts = [
            build_table_aware_passage_text(
                table_payloads[
                    point_id
                ]
            )
            for point_id in (
                ordered_table_ids
            )
        ]

        print()
        print(
            "Embedding table-aware passages: "
            f"{len(table_texts)}"
        )

        table_vectors = (
            embedding_service.embed_passages(
                table_texts
            )
        )

        table_vectors_by_id = {
            point_id: vector
            for (
                point_id,
                vector,
            ) in zip(
                ordered_table_ids,
                table_vectors,
                strict=True,
            )
        }

        contextual_rankings: dict[
            str,
            list[
                RankedPoint
            ],
        ] = {}

        overlay_rankings: dict[
            str,
            list[
                RankedPoint
            ],
        ] = {}

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

            contextual = (
                retrieve_complete_contextual_ranking(
                    client,
                    query_vector,
                    corpus_size=(
                        len(
                            english_payloads
                        )
                    ),
                )
            )

            overlay = (
                build_table_aware_overlay_ranking(
                    contextual,
                    query_vector=(
                        query_vector
                    ),
                    table_vectors_by_id=(
                        table_vectors_by_id
                    ),
                )
            )

            contextual_rankings[
                record.question_id
            ] = (
                contextual
            )

            overlay_rankings[
                record.question_id
            ] = (
                overlay
            )

        primary_ids = {
            point_id
            for record in records
            for point_id in (
                record.primary_relevant_chunk_ids
            )
        }

        table_hints_by_primary = {
            point_id: (
                extract_table_hints(
                    str(
                        table_payloads[
                            point_id
                        ].get(
                            "chunk_text"
                        )
                        or ""
                    )
                )
            )
            for point_id in (
                sorted(
                    primary_ids
                    & set(
                        table_payloads
                    )
                )
            )
        }

        summary = (
            build_summary(
                records,
                contextual_rankings,
                overlay_rankings,
                english_corpus_count=(
                    len(
                        english_payloads
                    )
                ),
                target_document_count=(
                    len(
                        target_payloads
                    )
                ),
                table_like_count=(
                    len(
                        table_payloads
                    )
                ),
                table_hints_by_id=(
                    table_hints_by_primary
                ),
            )
        )

        write_json_atomic(
            output_path,
            summary,
        )

        print_summary(
            summary
        )

        print()
        print(
            f"Output: {output_path}"
        )

        return summary

    finally:
        close_method = getattr(
            client,
            "close",
            None,
        )

        if callable(
            close_method
        ):
            close_method()


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the table-aware benchmark CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Benchmark a selective table-aware "
                "contextual-vector overlay."
            )
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            DEFAULT_DATASET_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help=(
            "Inspect detected table-like passages "
            "without making embedding requests."
        ),
    )

    return parser


def main(
) -> None:
    """Run the requested table-aware evaluation."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    run_benchmark(
        dataset_path=(
            args.dataset
        ),
        output_path=(
            args.output
        ),
        inspect_only=(
            args.inspect_only
        ),
    )


if __name__ == "__main__":
    main()