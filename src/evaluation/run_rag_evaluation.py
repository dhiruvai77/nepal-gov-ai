"""Production end-to-end RAG evaluation runner for NepalGov AI.

This runner executes the real production RAG pipeline over the manually
verified multilingual evaluation dataset.

Results are persisted incrementally as JSONL. Completed questions are reused on
later runs after their configuration and dataset fields are validated, avoiding
unnecessary repeated hosted retrieval, reranking, and Gemini calls.

The deterministic metrics in this runner evaluate evidence selection, citation
structure, and withholding behavior. They do not by themselves establish
claim-level semantic faithfulness.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import (
    asdict,
)
from pathlib import Path
from typing import (
    Any,
    Protocol,
)

from src.evaluation.rag_evaluator import (
    RAGQueryMetrics,
    aggregate_rag_metrics,
    evaluate_rag_result,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.rag.pipeline import (
    RAGResult,
    build_production_rag_pipeline,
)


DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v1.jsonl"
)

RESULT_SCHEMA_VERSION = 1

# Increment this whenever the production RAG configuration changes in a way
# that makes previous generated benchmark rows no longer directly comparable.
RUN_CONFIG_ID = "production-rag-v1"

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)


class EvaluationPipeline(
    Protocol
):
    """Minimal pipeline interface required by the evaluation runner."""

    def answer(
        self,
        query: str,
        *,
        answer_language: str,
        filters: dict[
            str,
            str,
        ]
        | None = None,
    ) -> RAGResult:
        """Answer one evaluation query."""

    def close(
        self,
    ) -> None:
        """Release provider resources."""


PipelineFactory = Callable[
    [],
    EvaluationPipeline,
]


def serialize_selected_evidence(
    result: RAGResult,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Serialize selected context with canonical source identifiers."""

    serialized: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, item in enumerate(
        result.selected_context,
        start=1,
    ):
        evidence = (
            item.result
        )

        serialized.append(
            {
                "evidence_id": (
                    f"E{index}"
                ),
                "point_id": (
                    evidence.point_id
                ),
                "chunk_id": (
                    evidence.chunk_id
                ),
                "document_id": (
                    evidence.document_id
                ),
                "title": (
                    evidence.title
                ),
                "organization": (
                    evidence.organization
                ),
                "language": (
                    evidence.language
                ),
                "page_start": (
                    evidence.page_start
                ),
                "page_end": (
                    evidence.page_end
                ),
                "source_url": (
                    evidence.source_url
                ),
                "rerank_score": (
                    item.rerank_score
                ),
                "original_rank": (
                    item.original_rank
                ),
            }
        )

    return serialized


def serialize_cited_evidence(
    result: RAGResult,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Serialize validated evidence references from generated output."""

    citation_result = (
        result.citation_result
    )

    if citation_result is None:
        return []

    serialized: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for citation in (
        citation_result.citations
    ):
        evidence = (
            citation.evidence.result
        )

        serialized.append(
            {
                "evidence_id": (
                    citation.evidence_id
                ),
                "point_id": (
                    evidence.point_id
                ),
                "chunk_id": (
                    evidence.chunk_id
                ),
                "document_id": (
                    evidence.document_id
                ),
                "title": (
                    evidence.title
                ),
                "page_start": (
                    evidence.page_start
                ),
                "page_end": (
                    evidence.page_end
                ),
                "source_url": (
                    evidence.source_url
                ),
            }
        )

    return serialized


def build_output_record(
    record: EvaluationRecord,
    result: RAGResult,
    metrics: RAGQueryMetrics,
) -> dict[
    str,
    Any,
]:
    """Build one durable JSON-serializable benchmark record."""

    citation_result = (
        result.citation_result
    )

    generated_answer_text = (
        citation_result.answer_text
        if citation_result
        is not None
        else None
    )

    invalid_evidence_ids = (
        list(
            citation_result
            .invalid_evidence_ids
        )
        if citation_result
        is not None
        else []
    )

    reason = (
        result.reason.value
        if result.reason
        is not None
        else None
    )

    return {
        "schema_version": (
            RESULT_SCHEMA_VERSION
        ),
        "run_config_id": (
            RUN_CONFIG_ID
        ),
        "question_id": (
            record.question_id
        ),
        "query": (
            record.query
        ),
        "query_language": (
            record.query_language
        ),
        "target_language": (
            record.target_language
        ),
        # User-facing answers follow the user's query language, including
        # cross-lingual retrieval cases.
        "answer_language": (
            record.query_language
        ),
        "category": (
            record.category
        ),
        "expected_document_ids": list(
            record.expected_document_ids
        ),
        "primary_relevant_chunk_ids": list(
            record.primary_relevant_chunk_ids
        ),
        "relevant_chunk_ids": list(
            record.relevant_chunk_ids
        ),
        "accepted": (
            result.accepted
        ),
        "withheld": (
            result.withheld
        ),
        "reason": reason,
        # answer_text is the application-visible result. If evidence guarding
        # withheld generation, this contains the deterministic fallback.
        "answer_text": (
            result.answer_text
        ),
        # Preserve the original provider output separately for failure analysis.
        "generated_answer_text": (
            generated_answer_text
        ),
        "provider": (
            result.provider
        ),
        "model": (
            result.model
        ),
        "selected_evidence": (
            serialize_selected_evidence(
                result
            )
        ),
        "cited_evidence": (
            serialize_cited_evidence(
                result
            )
        ),
        "invalid_evidence_ids": (
            invalid_evidence_ids
        ),
        "sources": list(
            result.sources
        ),
        "metrics": asdict(
            metrics
        ),
    }


def metric_from_output_record(
    row: dict[
        str,
        Any,
    ],
) -> RAGQueryMetrics:
    """Reconstruct deterministic metrics from one persisted row."""

    metrics = row.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict,
    ):
        raise ValueError(
            "Persisted RAG evaluation row "
            "contains no valid metrics object."
        )

    try:
        return RAGQueryMetrics(
            **metrics
        )

    except TypeError as exc:
        raise ValueError(
            "Persisted RAG evaluation metrics "
            "do not match the current schema."
        ) from exc


def append_output_record(
    output_path: str | Path,
    row: dict[
        str,
        Any,
    ],
) -> None:
    """Append one completed question immediately to durable JSONL output."""

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        # The context manager flushes on close. Explicit flushing documents the
        # checkpoint intent and ensures each completed question reaches the file
        # before the next expensive hosted request begins.
        handle.flush()


def validate_persisted_row(
    row: dict[
        str,
        Any,
    ],
    record: EvaluationRecord,
    *,
    line_number: int,
) -> None:
    """Reject stale or incompatible persisted evaluation rows."""

    if (
        row.get(
            "schema_version"
        )
        != RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Persisted RAG evaluation row "
            f"on line {line_number} uses an "
            "incompatible schema_version."
        )

    if (
        row.get(
            "run_config_id"
        )
        != RUN_CONFIG_ID
    ):
        raise ValueError(
            "Persisted RAG evaluation row "
            f"on line {line_number} uses a "
            "different run_config_id. Use --reset "
            "for an intentional fresh run."
        )

    expected_fields = {
        "query": (
            record.query
        ),
        "query_language": (
            record.query_language
        ),
        "target_language": (
            record.target_language
        ),
        "answer_language": (
            record.query_language
        ),
    }

    for field, expected_value in (
        expected_fields.items()
    ):
        if (
            row.get(
                field
            )
            != expected_value
        ):
            raise ValueError(
                "Persisted RAG evaluation row "
                f"for {record.question_id!r} "
                f"does not match current {field!r}. "
                "Use --reset for an intentional "
                "fresh run."
            )

    # Validate that saved metrics remain readable under the current evaluator.
    metric_from_output_record(
        row
    )


def load_existing_output(
    output_path: str | Path,
    records: Sequence[
        EvaluationRecord
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load and validate reusable completed evaluation rows."""

    path = Path(
        output_path
    )

    if not path.exists():
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

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = (
                raw_line.strip()
            )

            if not line:
                continue

            row = json.loads(
                line
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
                    "Persisted RAG evaluation row "
                    f"on line {line_number} has no "
                    "valid question_id."
                )

            if question_id not in records_by_id:
                raise ValueError(
                    "Persisted RAG evaluation row "
                    f"on line {line_number} contains "
                    f"unknown question_id "
                    f"{question_id!r}."
                )

            if question_id in completed:
                raise ValueError(
                    "Persisted RAG evaluation output "
                    f"contains duplicate question_id "
                    f"{question_id!r}."
                )

            validate_persisted_row(
                row,
                records_by_id[
                    question_id
                ],
                line_number=(
                    line_number
                ),
            )

            completed[
                question_id
            ] = row

    return completed


def validate_limit(
    limit: int | None,
) -> int | None:
    """Validate an optional maximum number of new questions to execute."""

    if (
        limit is not None
        and limit <= 0
    ):
        raise ValueError(
            "limit must be greater than zero "
            "when provided."
        )

    return limit


def group_metrics_by_language_pair(
    records: Sequence[
        EvaluationRecord
    ],
    metrics: Sequence[
        RAGQueryMetrics
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    list[
        RAGQueryMetrics
    ],
]:
    """Group completed RAG metrics by query -> target language pair."""

    metrics_by_question = {
        metric.question_id: metric
        for metric in metrics
    }

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[
            RAGQueryMetrics
        ],
    ] = defaultdict(
        list
    )

    for record in records:
        metric = (
            metrics_by_question.get(
                record.question_id
            )
        )

        if metric is None:
            continue

        pair = (
            record.query_language,
            record.target_language,
        )

        grouped[
            pair
        ].append(
            metric
        )

    return dict(
        grouped
    )


def format_language_pair(
    pair: tuple[
        str,
        str,
    ],
) -> str:
    """Return a compact language-slice label."""

    return (
        f"{pair[0]}->{pair[1]}"
    )


def print_summary_row(
    label: str,
    count: int,
    summary: dict[
        str,
        float,
    ],
) -> None:
    """Print one compact deterministic RAG evaluation summary row."""

    print(
        f"{label:<10}"
        f"{count:>4}  "
        f"{summary['acceptance_rate']:>7.3f}  "
        f"{summary['selected_primary_hit_rate']:>7.3f}  "
        f"{summary['selected_relevant_recall']:>7.3f}  "
        f"{summary['cited_primary_hit_rate']:>7.3f}  "
        f"{summary['cited_relevant_precision']:>7.3f}  "
        f"{summary['cited_relevant_recall']:>7.3f}  "
        f"{summary['valid_reference_ratio']:>7.3f}"
    )


def print_evaluation_summary(
    records: Sequence[
        EvaluationRecord
    ],
    metrics: Sequence[
        RAGQueryMetrics
    ],
) -> None:
    """Print overall and multilingual summaries for completed questions."""

    summary = (
        aggregate_rag_metrics(
            metrics
        )
    )

    grouped = (
        group_metrics_by_language_pair(
            records,
            metrics,
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "Production RAG evaluation"
    )

    print(
        "=" * 78
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>4}  "
        f"{'Accept':>7}  "
        f"{'SelHit':>7}  "
        f"{'SelRec':>7}  "
        f"{'CitHit':>7}  "
        f"{'CitPrec':>7}  "
        f"{'CitRec':>7}  "
        f"{'Valid':>7}"
    )

    print(
        "-" * 78
    )

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_metrics = (
            grouped.get(
                pair
            )
        )

        if not pair_metrics:
            continue

        print_summary_row(
            format_language_pair(
                pair
            ),
            len(
                pair_metrics
            ),
            aggregate_rag_metrics(
                pair_metrics
            ),
        )

    print(
        "-" * 78
    )

    print_summary_row(
        "overall",
        len(
            metrics
        ),
        summary,
    )


def run_evaluation(
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    *,
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    limit: int | None = None,
    reset: bool = False,
    pipeline_factory: PipelineFactory = (
        build_production_rag_pipeline
    ),
) -> tuple[
    list[
        RAGQueryMetrics
    ],
    dict[
        str,
        float,
    ],
]:
    """Run or resume production RAG evaluation.

    Every completed question is appended immediately to the output JSONL file.
    On subsequent runs, compatible saved rows are validated and reused.

    `limit` restricts only the number of new hosted evaluations executed during
    the current invocation. Previously completed rows remain part of the
    printed aggregate summary.

    `reset=True` intentionally deletes the existing output before execution.
    """

    normalized_limit = (
        validate_limit(
            limit
        )
    )

    records = (
        load_evaluation_records(
            dataset_path
        )
    )

    if not records:
        raise ValueError(
            "Evaluation dataset contains no records."
        )

    path = Path(
        output_path
    )

    if (
        reset
        and path.exists()
    ):
        path.unlink()

    completed = (
        load_existing_output(
            path,
            records,
        )
    )

    missing_records = [
        record
        for record in records
        if record.question_id
        not in completed
    ]

    if (
        normalized_limit
        is not None
    ):
        records_to_run = (
            missing_records[
                :normalized_limit
            ]
        )

    else:
        records_to_run = (
            missing_records
        )

    if records_to_run:
        pipeline = (
            pipeline_factory()
        )

        try:
            for index, record in enumerate(
                records_to_run,
                start=1,
            ):
                print(
                    f"[{index}/"
                    f"{len(records_to_run)}] "
                    f"{record.question_id}: "
                    f"{record.query_language}"
                    f"->{record.target_language}"
                )

                # target_language controls the corpus slice used for retrieval.
                # answer_language follows the user's query language.
                result = (
                    pipeline.answer(
                        record.query,
                        answer_language=(
                            record.query_language
                        ),
                        filters={
                            "language": (
                                record.target_language
                            ),
                        },
                    )
                )

                metrics = (
                    evaluate_rag_result(
                        record,
                        result,
                    )
                )

                row = (
                    build_output_record(
                        record,
                        result,
                        metrics,
                    )
                )

                append_output_record(
                    path,
                    row,
                )

                completed[
                    record.question_id
                ] = row

        finally:
            pipeline.close()

    ordered_metrics = [
        metric_from_output_record(
            completed[
                record.question_id
            ]
        )
        for record in records
        if record.question_id
        in completed
    ]

    print()
    print(
        "Completed: "
        f"{len(ordered_metrics)}/"
        f"{len(records)}"
    )

    remaining = (
        len(records)
        - len(
            ordered_metrics
        )
    )

    print(
        f"Remaining: {remaining}"
    )

    print(
        f"Output: {path}"
    )

    if not ordered_metrics:
        raise ValueError(
            "No RAG evaluation questions "
            "have been completed."
        )

    print_evaluation_summary(
        records,
        ordered_metrics,
    )

    summary = (
        aggregate_rag_metrics(
            ordered_metrics
        )
    )

    return (
        ordered_metrics,
        summary,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the production RAG evaluation command-line interface."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run or resume the NepalGov AI "
                "production RAG benchmark."
            )
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            DEFAULT_DATASET_PATH
        ),
        help=(
            "Evaluation JSONL dataset path."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Persistent JSONL output path."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new questions "
            "to execute in this invocation."
        ),
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete compatible/incompatible "
            "existing output and start fresh."
        ),
    )

    return parser


def main() -> None:
    """Run the production RAG benchmark from the command line."""

    parser = (
        build_argument_parser()
    )

    args = (
        parser.parse_args()
    )

    run_evaluation(
        dataset_path=(
            args.dataset
        ),
        output_path=(
            args.output
        ),
        limit=(
            args.limit
        ),
        reset=(
            args.reset
        ),
    )


if __name__ == "__main__":
    main()