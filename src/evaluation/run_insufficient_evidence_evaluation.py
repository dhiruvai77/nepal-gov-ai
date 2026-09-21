"""Production insufficient-evidence evaluation runner for NepalGov AI.

This runner executes the dedicated answer-vs-withhold benchmark through the
real production RAG pipeline.

Every completed result is persisted immediately so hosted retrieval, reranking,
and Gemini calls are not repeated after interruption.

The primary metric here is the structural application decision:

- accepted
- withheld

Important limitation:

An accepted RAGResult can still contain provider text that verbally says the
evidence is insufficient. Therefore structural false acceptance must later be
inspected together with generated answer text before changing production policy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Protocol,
)

from src.evaluation.insufficient_evidence_evaluator import (
    EXPECTED_BEHAVIOR_ANSWER,
    InsufficientEvidenceQueryMetrics,
    InsufficientEvidenceRecord,
    aggregate_insufficient_evidence_metrics,
    evaluate_insufficient_evidence_result,
    load_insufficient_evidence_records,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.evaluation.run_rag_evaluation import (
    serialize_cited_evidence,
    serialize_selected_evidence,
)
from src.rag.pipeline import (
    RAGResult,
    build_production_rag_pipeline,
)


DEFAULT_DATASET_PATH = Path(
    "data/evaluation/"
    "insufficient_evidence_questions.jsonl"
)

DEFAULT_REFERENCE_DATASET_PATH = Path(
    "data/evaluation/"
    "retrieval_questions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_v1.jsonl"
)

RESULT_SCHEMA_VERSION = 1

RUN_CONFIG_ID = (
    "insufficient-evidence-v1"
)

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)

CASE_TYPE_ORDER = (
    "answerable_control",
    "out_of_corpus_document",
    "out_of_corpus_period",
    "partial_evidence",
    "mixed_supported_unsupported",
)


class EvaluationPipeline(
    Protocol
):
    """Minimal RAG pipeline interface required by this benchmark."""

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
        """Answer one benchmark question."""

    def close(
        self,
    ) -> None:
        """Release provider resources."""


PipelineFactory = Callable[
    [],
    EvaluationPipeline,
]


def validate_limit(
    limit: int | None,
) -> int | None:
    """Validate an optional maximum number of new hosted calls."""

    if (
        limit is not None
        and limit <= 0
    ):
        raise ValueError(
            "limit must be greater "
            "than zero when provided."
        )

    return limit


def validate_answerable_controls(
    records: Sequence[
        InsufficientEvidenceRecord
    ],
    reference_records: Sequence[
        EvaluationRecord
    ],
) -> None:
    """Validate answerable controls against the existing verified benchmark.

    This prevents copied control questions from silently drifting away from the
    manually verified retrieval benchmark on which they are based.
    """

    reference_by_id = {
        record.question_id: record
        for record in (
            reference_records
        )
    }

    for record in records:
        if (
            record.expected_behavior
            != EXPECTED_BEHAVIOR_ANSWER
        ):
            continue

        reference_id = (
            record.reference_question_id
        )

        if reference_id is None:
            raise ValueError(
                f"Answerable control "
                f"{record.question_id!r} "
                "contains no reference_question_id."
            )

        reference = (
            reference_by_id.get(
                reference_id
            )
        )

        if reference is None:
            raise ValueError(
                f"Answerable control "
                f"{record.question_id!r} "
                "references unknown verified "
                f"question {reference_id!r}."
            )

        expected_fields = {
            "query": (
                reference.query
            ),
            "query_language": (
                reference.query_language
            ),
            "target_language": (
                reference.target_language
            ),
            "category": (
                reference.category
            ),
        }

        for (
            field,
            expected_value,
        ) in (
            expected_fields.items()
        ):
            if (
                getattr(
                    record,
                    field,
                )
                != expected_value
            ):
                raise ValueError(
                    f"Answerable control "
                    f"{record.question_id!r} "
                    f"does not match verified "
                    f"{field!r} from "
                    f"{reference_id!r}."
                )

        if (
            set(
                record.expected_document_ids
            )
            != set(
                reference.expected_document_ids
            )
        ):
            raise ValueError(
                f"Answerable control "
                f"{record.question_id!r} "
                "does not match verified "
                "expected_document_ids from "
                f"{reference_id!r}."
            )


def build_output_record(
    record: InsufficientEvidenceRecord,
    result: RAGResult,
    metrics: InsufficientEvidenceQueryMetrics,
) -> dict[
    str,
    Any,
]:
    """Build one durable JSON-serializable benchmark result."""

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
        "answer_language": (
            record.query_language
        ),
        "category": (
            record.category
        ),
        "expected_behavior": (
            record.expected_behavior
        ),
        "case_type": (
            record.case_type
        ),
        "expected_document_ids": list(
            record.expected_document_ids
        ),
        "reference_question_id": (
            record.reference_question_id
        ),
        "notes": (
            record.notes
        ),
        "accepted": (
            result.accepted
        ),
        "withheld": (
            result.withheld
        ),
        "reason": reason,
        # Application-visible text. This is either accepted generation or the
        # deterministic evidence-guard fallback.
        "answer_text": (
            result.answer_text
        ),
        # Preserve provider output separately when generation occurred.
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
) -> InsufficientEvidenceQueryMetrics:
    """Reconstruct structural decision metrics from one persisted row."""

    metrics = (
        row.get(
            "metrics"
        )
    )

    if not isinstance(
        metrics,
        dict,
    ):
        raise ValueError(
            "Persisted insufficient-"
            "evidence row contains no "
            "valid metrics object."
        )

    try:
        return (
            InsufficientEvidenceQueryMetrics(
                **metrics
            )
        )

    except TypeError as exc:
        raise ValueError(
            "Persisted insufficient-"
            "evidence metrics do not "
            "match the current schema."
        ) from exc


def append_output_record(
    output_path: str | Path,
    row: dict[
        str,
        Any,
    ],
) -> None:
    """Append one completed hosted evaluation immediately."""

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

        handle.flush()


def validate_persisted_row(
    row: dict[
        str,
        Any,
    ],
    record: InsufficientEvidenceRecord,
    *,
    line_number: int,
) -> None:
    """Reject stale or incompatible persisted benchmark rows."""

    if (
        row.get(
            "schema_version"
        )
        != RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Persisted insufficient-"
            "evidence row on line "
            f"{line_number} uses an "
            "incompatible schema_version."
        )

    if (
        row.get(
            "run_config_id"
        )
        != RUN_CONFIG_ID
    ):
        raise ValueError(
            "Persisted insufficient-"
            "evidence row on line "
            f"{line_number} uses a "
            "different run_config_id. "
            "Use --reset for a fresh run."
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
        "category": (
            record.category
        ),
        "expected_behavior": (
            record.expected_behavior
        ),
        "case_type": (
            record.case_type
        ),
        "expected_document_ids": list(
            record.expected_document_ids
        ),
        "reference_question_id": (
            record.reference_question_id
        ),
    }

    for (
        field,
        expected_value,
    ) in (
        expected_fields.items()
    ):
        if (
            row.get(
                field
            )
            != expected_value
        ):
            raise ValueError(
                "Persisted insufficient-"
                "evidence row for "
                f"{record.question_id!r} "
                "does not match current "
                f"{field!r}. Use --reset "
                "for a fresh run."
            )

    metric_from_output_record(
        row
    )


def load_existing_output(
    output_path: str | Path,
    records: Sequence[
        InsufficientEvidenceRecord
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load reusable completed benchmark rows."""

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
                    "Invalid JSON in "
                    "insufficient-evidence "
                    "output on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Persisted insufficient-"
                    "evidence row "
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
                    "Persisted insufficient-"
                    "evidence row on line "
                    f"{line_number} has no "
                    "valid question_id."
                )

            if (
                question_id
                not in records_by_id
            ):
                raise ValueError(
                    "Persisted insufficient-"
                    "evidence output contains "
                    "unknown question_id "
                    f"{question_id!r}."
                )

            if (
                question_id
                in completed
            ):
                raise ValueError(
                    "Persisted insufficient-"
                    "evidence output contains "
                    "duplicate question_id "
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


def group_metrics_by_language_pair(
    records: Sequence[
        InsufficientEvidenceRecord
    ],
    metrics: Sequence[
        InsufficientEvidenceQueryMetrics
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    list[
        InsufficientEvidenceQueryMetrics
    ],
]:
    """Group completed metrics by query -> target language pair."""

    metrics_by_id = {
        metric.question_id: metric
        for metric in metrics
    }

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[
            InsufficientEvidenceQueryMetrics
        ],
    ] = defaultdict(
        list
    )

    for record in records:
        metric = (
            metrics_by_id.get(
                record.question_id
            )
        )

        if metric is None:
            continue

        grouped[
            (
                record.query_language,
                record.target_language,
            )
        ].append(
            metric
        )

    return dict(
        grouped
    )


def group_metrics_by_case_type(
    records: Sequence[
        InsufficientEvidenceRecord
    ],
    metrics: Sequence[
        InsufficientEvidenceQueryMetrics
    ],
) -> dict[
    str,
    list[
        InsufficientEvidenceQueryMetrics
    ],
]:
    """Group completed metrics by diagnostic case type."""

    metrics_by_id = {
        metric.question_id: metric
        for metric in metrics
    }

    grouped: dict[
        str,
        list[
            InsufficientEvidenceQueryMetrics
        ],
    ] = defaultdict(
        list
    )

    for record in records:
        metric = (
            metrics_by_id.get(
                record.question_id
            )
        )

        if metric is None:
            continue

        grouped[
            record.case_type
        ].append(
            metric
        )

    return dict(
        grouped
    )


def print_summary_row(
    label: str,
    metrics: Sequence[
        InsufficientEvidenceQueryMetrics
    ],
) -> None:
    """Print one compact answer-vs-withhold summary row."""

    summary = (
        aggregate_insufficient_evidence_metrics(
            metrics
        )
    )

    print(
        f"{label:<24}"
        f"{summary['question_count']:>5}"
        f"{summary['decision_accuracy']:>10.3f}"
        f"{summary['answer_acceptance_rate']:>10.3f}"
        f"{summary['withholding_success_rate']:>10.3f}"
        f"{summary['guard_false_accept_rate']:>10.3f}"
        f"{summary['guard_false_withhold_rate']:>10.3f}"
    )


def print_evaluation_summary(
    records: Sequence[
        InsufficientEvidenceRecord
    ],
    metrics: Sequence[
        InsufficientEvidenceQueryMetrics
    ],
) -> None:
    """Print overall, language-pair, and case-type summaries."""

    if not metrics:
        print(
            "No completed insufficient-"
            "evidence evaluations."
        )
        return

    grouped_pairs = (
        group_metrics_by_language_pair(
            records,
            metrics,
        )
    )

    grouped_cases = (
        group_metrics_by_case_type(
            records,
            metrics,
        )
    )

    print()
    print(
        "=" * 79
    )

    print(
        "Insufficient-evidence RAG evaluation"
    )

    print(
        "=" * 79
    )

    print(
        f"{'Slice':<24}"
        f"{'N':>5}"
        f"{'Decision':>10}"
        f"{'AnsAcc':>10}"
        f"{'Withhold':>10}"
        f"{'FAcc':>10}"
        f"{'FWith':>10}"
    )

    print(
        "-" * 79
    )

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_metrics = (
            grouped_pairs.get(
                pair
            )
        )

        if not pair_metrics:
            continue

        print_summary_row(
            (
                f"{pair[0]}"
                f"->{pair[1]}"
            ),
            pair_metrics,
        )

    print(
        "-" * 79
    )

    print_summary_row(
        "overall",
        metrics,
    )

    print()
    print(
        "Case type breakdown"
    )
    print(
        "-" * 79
    )

    for case_type in (
        CASE_TYPE_ORDER
    ):
        case_metrics = (
            grouped_cases.get(
                case_type
            )
        )

        if not case_metrics:
            continue

        print_summary_row(
            case_type,
            case_metrics,
        )


def run_evaluation(
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    *,
    reference_dataset_path: str | Path = (
        DEFAULT_REFERENCE_DATASET_PATH
    ),
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
        InsufficientEvidenceQueryMetrics
    ],
    dict[
        str,
        float | int,
    ],
]:
    """Run or resume the production insufficient-evidence benchmark."""

    normalized_limit = (
        validate_limit(
            limit
        )
    )

    records = (
        load_insufficient_evidence_records(
            dataset_path
        )
    )

    reference_records = (
        load_evaluation_records(
            reference_dataset_path
        )
    )

    validate_answerable_controls(
        records,
        reference_records,
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
        if (
            record.question_id
            not in completed
        )
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
            for (
                index,
                record,
            ) in enumerate(
                records_to_run,
                start=1,
            ):
                print(
                    "Evaluating "
                    f"{index}/"
                    f"{len(records_to_run)}: "
                    f"{record.question_id} "
                    f"({record.expected_behavior})"
                )

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
                    evaluate_insufficient_evidence_result(
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

        finally:
            pipeline.close()

    completed = (
        load_existing_output(
            path,
            records,
        )
    )

    metrics = [
        metric_from_output_record(
            completed[
                record.question_id
            ]
        )
        for record in records
        if (
            record.question_id
            in completed
        )
    ]

    if metrics:
        overall = (
            aggregate_insufficient_evidence_metrics(
                metrics
            )
        )

    else:
        overall = {
            "question_count": 0,
        }

    return (
        metrics,
        overall,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the production benchmark command-line interface."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run NepalGov AI's "
                "answer-vs-withhold "
                "insufficient-evidence "
                "benchmark."
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
        "--reference-dataset",
        type=Path,
        default=(
            DEFAULT_REFERENCE_DATASET_PATH
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
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reset",
        action="store_true",
    )

    return parser


def main(
) -> None:
    """Run or resume insufficient-evidence evaluation."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    metrics, _ = (
        run_evaluation(
            args.dataset,
            reference_dataset_path=(
                args.reference_dataset
            ),
            output_path=(
                args.output
            ),
            limit=args.limit,
            reset=args.reset,
        )
    )

    records = (
        load_insufficient_evidence_records(
            args.dataset
        )
    )

    print_evaluation_summary(
        records,
        metrics,
    )

    print()

    print(
        "Completed questions: "
        f"{len(metrics)}/"
        f"{len(records)}"
    )


if __name__ == "__main__":
    main()