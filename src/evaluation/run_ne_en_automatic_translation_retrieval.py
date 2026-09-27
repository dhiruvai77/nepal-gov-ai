"""Evaluate automatic NE->EN query translation as a retrieval signal.

This evaluation tests whether automatically generated English translations of
the six Nepali-query -> English-evidence benchmark questions reproduce the
retrieval gains previously observed with manually controlled English queries.

The previous production/manual-control experiment is loaded from its persisted
artifact and is NOT rerun.

Automatic path:

    Gemini-generated English translation
        -> contextual dense top 100
        + English BM25 top 100
        -> Reciprocal Rank Fusion
        -> final top 20
        -> BGE reranking with ORIGINAL Nepali query

This means only the automatic-translation candidate path requires new hosted
retrieval/reranking work.

Important boundaries:

- production retrieval is unchanged,
- automatic translations remain evaluation-only,
- the manually controlled queries are not used to generate translations,
- BGE receives the original Nepali query for fair comparison,
- completed per-question results are persisted immediately and reused on resume.
"""

from __future__ import annotations

import argparse
import json
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
from src.evaluation.compare_ne_en_reranked_variants import (
    first_primary_rank,
    reranked_results_only,
    validate_candidate_identity,
)
from src.evaluation.compare_ne_en_retrieval_variants import (
    DEFAULT_FINAL_CANDIDATE_DEPTH,
    TRANSLATED_HYBRID_COMPONENT_DEPTH,
    build_translated_hybrid,
    select_ne_en_records,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.evaluation.run_ne_en_query_translation import (
    DEFAULT_OUTPUT_PATH as DEFAULT_TRANSLATION_PATH,
    TRANSLATION_CONFIG_ID,
    load_existing_output as load_translation_output,
)
from src.generation.gemini_service import (
    MODEL_NAME as TRANSLATION_MODEL,
    PROVIDER_NAME as TRANSLATION_PROVIDER,
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


BENCHMARK_SCHEMA_VERSION = 1

BENCHMARK_CONFIG_ID = (
    "ne-en-automatic-translation-retrieval-v1"
)

EXPECTED_BASELINE_RUN_CONFIG_ID = (
    "ne-en-targeted-retrieval-v1"
)

DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

DEFAULT_BASELINE_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "ne_en_targeted_retrieval_v1.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "ne_en_automatic_translation_retrieval_v1.jsonl"
)

DEFAULT_SUMMARY_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "ne_en_automatic_translation_retrieval_v1_summary.json"
)

DEFAULT_CUTOFFS = (
    5,
    10,
    20,
)


def _load_json_object(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Load one non-empty JSON object."""

    source = Path(
        path
    )

    if not source.exists():
        raise FileNotFoundError(
            f"JSON artifact does not exist: {source}"
        )

    try:
        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON artifact: {source}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"{source} must contain one JSON object."
        )

    if not data:
        raise ValueError(
            f"{source} contains an empty JSON object."
        )

    return data


def validate_baseline_artifact(
    baseline: Mapping[
        str,
        Any,
    ],
    records: Sequence[
        EvaluationRecord
    ],
) -> None:
    """Validate the prior manual-control retrieval benchmark."""

    if (
        baseline.get(
            "run_config_id"
        )
        != EXPECTED_BASELINE_RUN_CONFIG_ID
    ):
        raise ValueError(
            "Unexpected baseline run_config_id."
        )

    if (
        baseline.get(
            "production_changed"
        )
        is not False
    ):
        raise ValueError(
            "Baseline retrieval artifact must "
            "remain evaluation-only."
        )

    evaluation_slice = (
        baseline.get(
            "evaluation_slice"
        )
    )

    if not isinstance(
        evaluation_slice,
        Mapping,
    ):
        raise ValueError(
            "Baseline contains no valid "
            "evaluation_slice."
        )

    expected_ids = [
        record.question_id
        for record in records
    ]

    actual_ids = (
        evaluation_slice.get(
            "question_ids"
        )
    )

    if (
        actual_ids
        != expected_ids
    ):
        raise ValueError(
            "Baseline question IDs do not "
            "match the current NE->EN slice."
        )

    if (
        evaluation_slice.get(
            "question_count"
        )
        != len(
            records
        )
    ):
        raise ValueError(
            "Baseline question count does "
            "not match the current NE->EN slice."
        )

    candidate_generation = (
        baseline.get(
            "candidate_generation"
        )
    )

    if not isinstance(
        candidate_generation,
        Mapping,
    ):
        raise ValueError(
            "Baseline contains no valid "
            "candidate_generation object."
        )

    translated = (
        candidate_generation.get(
            "translated_hybrid"
        )
    )

    if not isinstance(
        translated,
        Mapping,
    ):
        raise ValueError(
            "Baseline contains no translated "
            "hybrid candidate configuration."
        )

    if (
        translated.get(
            "dense_component_depth"
        )
        != TRANSLATED_HYBRID_COMPONENT_DEPTH
    ):
        raise ValueError(
            "Baseline dense component depth "
            "does not match this experiment."
        )

    if (
        translated.get(
            "bm25_component_depth"
        )
        != TRANSLATED_HYBRID_COMPONENT_DEPTH
    ):
        raise ValueError(
            "Baseline BM25 component depth "
            "does not match this experiment."
        )

    if (
        translated.get(
            "final_candidate_depth"
        )
        != DEFAULT_FINAL_CANDIDATE_DEPTH
    ):
        raise ValueError(
            "Baseline final candidate depth "
            "does not match this experiment."
        )


def _point_ids(
    results: Sequence[
        RetrievalResult
    ],
) -> list[str]:
    """Extract point IDs while preserving ranking order."""

    return [
        result.point_id
        for result in results
    ]


def _serialize_reranked(
    results: Sequence[
        RerankedResult
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Serialize reranker output without duplicating source passage text."""

    return [
        {
            "point_id": (
                item.result.point_id
            ),
            "rerank_score": (
                item.rerank_score
            ),
            "original_rank": (
                item.original_rank
            ),
        }
        for item in results
    ]


def _bge_point_ids(
    row: Mapping[
        str,
        Any,
    ],
) -> list[str]:
    """Extract persisted reranked point IDs."""

    reranked = (
        row.get(
            "automatic_bge"
        )
    )

    if not isinstance(
        reranked,
        list,
    ):
        raise ValueError(
            "Persisted benchmark row contains "
            "no automatic_bge list."
        )

    point_ids: list[str] = []

    for item in reranked:
        if not isinstance(
            item,
            Mapping,
        ):
            raise ValueError(
                "Persisted automatic_bge item "
                "must be an object."
            )

        point_id = (
            item.get(
                "point_id"
            )
        )

        if (
            not isinstance(
                point_id,
                str,
            )
            or not point_id
        ):
            raise ValueError(
                "Persisted automatic_bge item "
                "has no valid point_id."
            )

        point_ids.append(
            point_id
        )

    return point_ids


def primary_rank_from_ids(
    record: EvaluationRecord,
    point_ids: Sequence[str],
) -> int | None:
    """Return first primary-evidence rank from persisted point IDs."""

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    for (
        rank,
        point_id,
    ) in enumerate(
        point_ids,
        start=1,
    ):
        if (
            point_id
            in primary_ids
        ):
            return rank

    return None


def evaluate_point_ids(
    record: EvaluationRecord,
    point_ids: Sequence[str],
    *,
    cutoff: int,
) -> dict[
    str,
    float,
]:
    """Calculate standard Hit, MRR, and Recall from persisted IDs."""

    if cutoff <= 0:
        raise ValueError(
            "cutoff must be greater than zero."
        )

    top_ids = list(
        point_ids[
            :cutoff
        ]
    )

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


def aggregate_point_id_metrics(
    records: Sequence[
        EvaluationRecord
    ],
    rows_by_id: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
    *,
    ranking_field: str,
    cutoff: int,
) -> dict[
    str,
    float,
]:
    """Aggregate standard retrieval metrics from persisted rankings."""

    metrics: list[
        dict[
            str,
            float,
        ]
    ] = []

    for record in records:
        row = (
            rows_by_id[
                record.question_id
            ]
        )

        if (
            ranking_field
            == "automatic_bge"
        ):
            point_ids = (
                _bge_point_ids(
                    row
                )
            )

        else:
            value = (
                row.get(
                    ranking_field
                )
            )

            if not isinstance(
                value,
                list,
            ):
                raise ValueError(
                    f"{record.question_id}: "
                    f"{ranking_field} must be a list."
                )

            if not all(
                isinstance(
                    item,
                    str,
                )
                and item
                for item in value
            ):
                raise ValueError(
                    f"{record.question_id}: "
                    f"{ranking_field} contains "
                    "invalid point IDs."
                )

            point_ids = list(
                value
            )

        metrics.append(
            evaluate_point_ids(
                record,
                point_ids,
                cutoff=cutoff,
            )
        )

    count = len(
        metrics
    )

    if count == 0:
        raise ValueError(
            "At least one metric row is required."
        )

    return {
        "hit_rate": sum(
            item[
                "hit_rate"
            ]
            for item in metrics
        )
        / count,
        "mrr": sum(
            item[
                "mrr"
            ]
            for item in metrics
        )
        / count,
        "recall": sum(
            item[
                "recall"
            ]
            for item in metrics
        )
        / count,
    }


def build_benchmark_row(
    record: EvaluationRecord,
    translation_row: Mapping[
        str,
        Any,
    ],
    *,
    dense_results: Sequence[
        RetrievalResult
    ],
    hybrid_results: Sequence[
        RetrievalResult
    ],
    reranked_results: Sequence[
        RerankedResult
    ],
) -> dict[
    str,
    Any,
]:
    """Build one persisted automatic-translation retrieval result."""

    validate_candidate_identity(
        hybrid_results,
        reranked_results,
    )

    automatic_query = (
        translation_row.get(
            "translated_query"
        )
    )

    if (
        not isinstance(
            automatic_query,
            str,
        )
        or not automatic_query.strip()
    ):
        raise ValueError(
            f"{record.question_id}: automatic "
            "translation is missing."
        )

    prompt_hash = (
        translation_row.get(
            "prompt_sha256"
        )
    )

    if (
        not isinstance(
            prompt_hash,
            str,
        )
        or not prompt_hash
    ):
        raise ValueError(
            f"{record.question_id}: translation "
            "prompt hash is missing."
        )

    dense_ids = (
        _point_ids(
            dense_results
        )
    )

    hybrid_ids = (
        _point_ids(
            hybrid_results
        )
    )

    reranked_only = (
        reranked_results_only(
            reranked_results
        )
    )

    return {
        "schema_version": (
            BENCHMARK_SCHEMA_VERSION
        ),
        "run_config_id": (
            BENCHMARK_CONFIG_ID
        ),
        "translation_config_id": (
            TRANSLATION_CONFIG_ID
        ),
        "question_id": (
            record.question_id
        ),
        "query_language": (
            record.query_language
        ),
        "target_language": (
            record.target_language
        ),
        "original_query": (
            record.query
        ),
        "automatic_query": (
            automatic_query
        ),
        "translation_prompt_sha256": (
            prompt_hash
        ),
        "component_depth": (
            TRANSLATED_HYBRID_COMPONENT_DEPTH
        ),
        "final_candidate_depth": (
            DEFAULT_FINAL_CANDIDATE_DEPTH
        ),
        "automatic_dense_point_ids": (
            dense_ids
        ),
        "automatic_hybrid_point_ids": (
            hybrid_ids
        ),
        "automatic_bge": (
            _serialize_reranked(
                reranked_results
            )
        ),
        "automatic_dense_primary_rank": (
            first_primary_rank(
                record,
                dense_results,
            )
        ),
        "automatic_hybrid_primary_rank": (
            first_primary_rank(
                record,
                hybrid_results,
            )
        ),
        "automatic_bge_primary_rank": (
            first_primary_rank(
                record,
                reranked_only,
            )
        ),
    }


def validate_persisted_benchmark_row(
    row: Mapping[
        str,
        Any,
    ],
    record: EvaluationRecord,
    translation_row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Reject stale or incompatible automatic retrieval rows."""

    expected = (
        (
            "schema_version",
            BENCHMARK_SCHEMA_VERSION,
        ),
        (
            "run_config_id",
            BENCHMARK_CONFIG_ID,
        ),
        (
            "translation_config_id",
            TRANSLATION_CONFIG_ID,
        ),
        (
            "question_id",
            record.question_id,
        ),
        (
            "query_language",
            record.query_language,
        ),
        (
            "target_language",
            record.target_language,
        ),
        (
            "original_query",
            record.query,
        ),
        (
            "automatic_query",
            translation_row[
                "translated_query"
            ],
        ),
        (
            "translation_prompt_sha256",
            translation_row[
                "prompt_sha256"
            ],
        ),
        (
            "component_depth",
            TRANSLATED_HYBRID_COMPONENT_DEPTH,
        ),
        (
            "final_candidate_depth",
            DEFAULT_FINAL_CANDIDATE_DEPTH,
        ),
    )

    for (
        field_name,
        expected_value,
    ) in expected:
        if (
            row.get(
                field_name
            )
            != expected_value
        ):
            raise ValueError(
                f"{record.question_id}: persisted "
                f"benchmark field {field_name!r} "
                "does not match the current experiment."
            )

    dense_ids = (
        row.get(
            "automatic_dense_point_ids"
        )
    )

    hybrid_ids = (
        row.get(
            "automatic_hybrid_point_ids"
        )
    )

    if (
        not isinstance(
            dense_ids,
            list,
        )
        or not dense_ids
    ):
        raise ValueError(
            f"{record.question_id}: automatic "
            "dense ranking is missing."
        )

    if (
        not isinstance(
            hybrid_ids,
            list,
        )
        or len(
            hybrid_ids
        )
        != DEFAULT_FINAL_CANDIDATE_DEPTH
    ):
        raise ValueError(
            f"{record.question_id}: automatic "
            "hybrid ranking has unexpected depth."
        )

    if (
        len(
            hybrid_ids
        )
        != len(
            set(
                hybrid_ids
            )
        )
    ):
        raise ValueError(
            f"{record.question_id}: automatic "
            "hybrid ranking contains duplicates."
        )

    bge_ids = (
        _bge_point_ids(
            row
        )
    )

    if (
        len(
            bge_ids
        )
        != DEFAULT_FINAL_CANDIDATE_DEPTH
    ):
        raise ValueError(
            f"{record.question_id}: automatic "
            "BGE ranking has unexpected depth."
        )

    if (
        set(
            bge_ids
        )
        != set(
            hybrid_ids
        )
    ):
        raise ValueError(
            f"{record.question_id}: BGE ranking "
            "changed candidate identity."
        )


def load_existing_benchmark_rows(
    path: str | Path,
    records: Sequence[
        EvaluationRecord
    ],
    translations_by_id: Mapping[
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
    """Load reusable per-question automatic retrieval results."""

    output_path = Path(
        path
    )

    if not output_path.exists():
        return {}

    records_by_id = {
        record.question_id: record
        for record in records
    }

    completed: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for (
            line_number,
            raw_line,
        ) in enumerate(
            handle,
            start=1,
        ):
            line = (
                raw_line.strip()
            )

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid automatic retrieval "
                    f"JSON on line {line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Automatic retrieval row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            question_id = (
                row.get(
                    "question_id"
                )
            )

            if (
                not isinstance(
                    question_id,
                    str,
                )
                or not question_id
            ):
                raise ValueError(
                    "Automatic retrieval row "
                    f"{line_number} has no "
                    "valid question_id."
                )

            record = (
                records_by_id.get(
                    question_id
                )
            )

            translation = (
                translations_by_id.get(
                    question_id
                )
            )

            if (
                record is None
                or translation is None
            ):
                raise ValueError(
                    "Automatic retrieval output "
                    "contains unexpected "
                    f"question_id {question_id!r}."
                )

            if (
                question_id
                in completed
            ):
                raise ValueError(
                    "Automatic retrieval output "
                    "contains duplicate question_id "
                    f"{question_id!r}."
                )

            validate_persisted_benchmark_row(
                row,
                record,
                translation,
            )

            completed[
                question_id
            ] = row

    return completed


def append_benchmark_row(
    path: str | Path,
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Persist one completed automatic retrieval case immediately."""

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                dict(
                    row
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        handle.flush()


def build_summary(
    records: Sequence[
        EvaluationRecord
    ],
    rows_by_id: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
    baseline: Mapping[
        str,
        Any,
    ],
    *,
    cutoffs: Sequence[int] = (
        DEFAULT_CUTOFFS
    ),
) -> dict[
    str,
    Any,
]:
    """Build the deterministic automatic-vs-manual retrieval summary."""

    if (
        set(
            rows_by_id
        )
        != {
            record.question_id
            for record in records
        }
    ):
        raise ValueError(
            "Automatic retrieval benchmark "
            "is incomplete."
        )

    automatic_metrics: dict[
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
        automatic_metrics[
            str(
                cutoff
            )
        ] = {
            "automatic_first_stage": (
                aggregate_point_id_metrics(
                    records,
                    rows_by_id,
                    ranking_field=(
                        "automatic_hybrid_point_ids"
                    ),
                    cutoff=cutoff,
                )
            ),
            "automatic_bge": (
                aggregate_point_id_metrics(
                    records,
                    rows_by_id,
                    ranking_field=(
                        "automatic_bge"
                    ),
                    cutoff=cutoff,
                )
            ),
        }

    question_ranks: dict[
        str,
        dict[
            str,
            int | None,
        ],
    ] = {}

    baseline_ranks = (
        baseline[
            "reranked_primary_ranks"
        ]
    )

    for record in records:
        automatic_row = (
            rows_by_id[
                record.question_id
            ]
        )

        previous = (
            baseline_ranks[
                record.question_id
            ]
        )

        question_ranks[
            record.question_id
        ] = {
            "production_first_stage": (
                previous[
                    "production_first_stage"
                ]
            ),
            "production_bge": (
                previous[
                    "production_bge"
                ]
            ),
            "manual_first_stage": (
                previous[
                    "translated_first_stage"
                ]
            ),
            "manual_bge": (
                previous[
                    "translated_bge"
                ]
            ),
            "automatic_dense": (
                automatic_row[
                    "automatic_dense_primary_rank"
                ]
            ),
            "automatic_first_stage": (
                automatic_row[
                    "automatic_hybrid_primary_rank"
                ]
            ),
            "automatic_bge": (
                automatic_row[
                    "automatic_bge_primary_rank"
                ]
            ),
        }

    return {
        "schema_version": (
            BENCHMARK_SCHEMA_VERSION
        ),
        "run_config_id": (
            BENCHMARK_CONFIG_ID
        ),
        "translation_config_id": (
            TRANSLATION_CONFIG_ID
        ),
        "baseline_run_config_id": (
            EXPECTED_BASELINE_RUN_CONFIG_ID
        ),
        "question_count": len(
            records
        ),
        "question_ids": [
            record.question_id
            for record in records
        ],
        "candidate_generation": {
            "automatic_query_source": (
                "gemini_translation"
            ),
            "dense_vector": (
                CONTEXTUAL_DENSE_VECTOR_NAME
            ),
            "dense_component_depth": (
                TRANSLATED_HYBRID_COMPONENT_DEPTH
            ),
            "bm25_component_depth": (
                TRANSLATED_HYBRID_COMPONENT_DEPTH
            ),
            "fusion": (
                "reciprocal_rank_fusion"
            ),
            "final_candidate_depth": (
                DEFAULT_FINAL_CANDIDATE_DEPTH
            ),
            "reranker_query": (
                "original_nepali_query"
            ),
        },
        "primary_ranks": (
            question_ranks
        ),
        "automatic_metrics": (
            automatic_metrics
        ),
        "baseline_metrics": {
            "first_stage": (
                baseline[
                    "first_stage_metrics"
                ]
            ),
            "reranked": (
                baseline[
                    "reranked_metrics"
                ]
            ),
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
    """Write one JSON summary atomically."""

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


def _rank_text(
    value: int | None,
    *,
    depth: int,
) -> str:
    """Render persisted missing ranks consistently."""

    if value is None:
        return (
            f">{depth}"
        )

    return str(
        value
    )


def _print_metric_row(
    label: str,
    metrics: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print one retrieval metric row."""

    print(
        f"{label:<26}"
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
    """Print production/manual/automatic comparison."""

    print()
    print(
        "=" * 110
    )

    print(
        "NE->EN automatic translation retrieval benchmark"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Question':<12}"
        f"{'ProdBGE':>12}"
        f"{'ManualFirst':>14}"
        f"{'ManualBGE':>12}"
        f"{'AutoDense':>12}"
        f"{'AutoFirst':>12}"
        f"{'AutoBGE':>12}"
    )

    print(
        "-" * 110
    )

    primary_ranks = (
        summary[
            "primary_ranks"
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

        print(
            f"{question_id:<12}"
            f"{_rank_text(ranks['production_bge'], depth=20):>12}"
            f"{_rank_text(ranks['manual_first_stage'], depth=20):>14}"
            f"{_rank_text(ranks['manual_bge'], depth=20):>12}"
            f"{_rank_text(ranks['automatic_dense'], depth=100):>12}"
            f"{_rank_text(ranks['automatic_first_stage'], depth=20):>12}"
            f"{_rank_text(ranks['automatic_bge'], depth=20):>12}"
        )

    for cutoff in (
        DEFAULT_CUTOFFS
    ):
        key = str(
            cutoff
        )

        baseline_reranked = (
            summary[
                "baseline_metrics"
            ][
                "reranked"
            ][
                key
            ]
        )

        automatic = (
            summary[
                "automatic_metrics"
            ][
                key
            ]
        )

        print()
        print(
            "=" * 76
        )

        print(
            f"NE->EN BGE comparison @ {cutoff}"
        )

        print(
            "=" * 76
        )

        print(
            f"{'Variant':<26}"
            f"{'Hit':>10}"
            f"{'MRR':>10}"
            f"{'Recall':>10}"
        )

        print(
            "-" * 76
        )

        _print_metric_row(
            "production_bge",
            baseline_reranked[
                "production_bge"
            ],
        )

        _print_metric_row(
            "manual_translation_bge",
            baseline_reranked[
                "translated_bge"
            ],
        )

        _print_metric_row(
            "automatic_translation_bge",
            automatic[
                "automatic_bge"
            ],
        )

        print()
        print(
            "Automatic first-stage hybrid:"
        )

        _print_metric_row(
            "automatic_first_stage",
            automatic[
                "automatic_first_stage"
            ],
        )


def run_benchmark(
    *,
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    translation_path: str | Path = (
        DEFAULT_TRANSLATION_PATH
    ),
    baseline_path: str | Path = (
        DEFAULT_BASELINE_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    summary_path: str | Path = (
        DEFAULT_SUMMARY_PATH
    ),
    reranker: Reranker | None = None,
) -> dict[
    str,
    Any,
]:
    """Run missing automatic retrieval cases and persist the final summary."""

    records = (
        select_ne_en_records(
            load_evaluation_records(
                dataset_path
            )
        )
    )

    if not records:
        raise ValueError(
            "No NE->EN benchmark records found."
        )

    translation_rows = (
        load_translation_output(
            translation_path,
            records,
            model=(
                TRANSLATION_MODEL
            ),
            provider=(
                TRANSLATION_PROVIDER
            ),
        )
    )

    if (
        set(
            translation_rows
        )
        != {
            record.question_id
            for record in records
        }
    ):
        raise ValueError(
            "Automatic translation artifact "
            "is incomplete."
        )

    baseline = (
        _load_json_object(
            baseline_path
        )
    )

    validate_baseline_artifact(
        baseline,
        records,
    )

    completed = (
        load_existing_benchmark_rows(
            output_path,
            records,
            translation_rows,
        )
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

    try:
        for (
            index,
            record,
        ) in enumerate(
            records,
            start=1,
        ):
            if (
                record.question_id
                in completed
            ):
                print(
                    f"[{index}/{len(records)}] "
                    f"{record.question_id}: reuse"
                )

                continue

            print(
                f"[{index}/{len(records)}] "
                f"{record.question_id}: retrieve + rerank"
            )

            translation = (
                translation_rows[
                    record.question_id
                ]
            )

            automatic_query = (
                translation[
                    "translated_query"
                ]
            )

            filters = {
                "language": "en",
            }

            dense_results = (
                dense_retriever.retrieve(
                    query=(
                        automatic_query
                    ),
                    top_k=(
                        TRANSLATED_HYBRID_COMPONENT_DEPTH
                    ),
                    filters=(
                        filters
                    ),
                )
            )

            sparse_results = (
                sparse_retriever.retrieve(
                    query=(
                        automatic_query
                    ),
                    top_k=(
                        TRANSLATED_HYBRID_COMPONENT_DEPTH
                    ),
                    filters=(
                        filters
                    ),
                )
            )

            hybrid_results = (
                build_translated_hybrid(
                    dense_results,
                    sparse_results,
                    component_depth=(
                        TRANSLATED_HYBRID_COMPONENT_DEPTH
                    ),
                    final_depth=(
                        DEFAULT_FINAL_CANDIDATE_DEPTH
                    ),
                )
            )

            reranked = (
                active_reranker.rerank(
                    record.query,
                    hybrid_results,
                    top_k=None,
                )
            )

            row = (
                build_benchmark_row(
                    record,
                    translation,
                    dense_results=(
                        dense_results
                    ),
                    hybrid_results=(
                        hybrid_results
                    ),
                    reranked_results=(
                        reranked
                    ),
                )
            )

            append_benchmark_row(
                output_path,
                row,
            )

            completed[
                record.question_id
            ] = row

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

    summary = (
        build_summary(
            records,
            completed,
            baseline,
        )
    )

    write_json_atomic(
        summary_path,
        summary,
    )

    print_summary(
        summary
    )

    print()
    print(
        f"Rows: {output_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    return summary


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the automatic retrieval benchmark CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Benchmark automatic English "
                "query translation on the six "
                "NE->EN retrieval cases."
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
        "--translations",
        type=Path,
        default=(
            DEFAULT_TRANSLATION_PATH
        ),
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            DEFAULT_BASELINE_PATH
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
        "--summary",
        type=Path,
        default=(
            DEFAULT_SUMMARY_PATH
        ),
    )

    return parser


def main(
) -> None:
    """Run the automatic NE->EN retrieval benchmark."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    run_benchmark(
        dataset_path=(
            args.dataset
        ),
        translation_path=(
            args.translations
        ),
        baseline_path=(
            args.baseline
        ),
        output_path=(
            args.output
        ),
        summary_path=(
            args.summary
        ),
    )


if __name__ == "__main__":
    main()