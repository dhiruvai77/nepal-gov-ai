"""Consolidated production-quality checkpoint for NepalGov AI.

This module combines the project's persisted evaluation layers into one
deterministic, machine-readable checkpoint:

1. structural production RAG evaluation,
2. human semantic citation review,
3. automated semantic-judge agreement,
4. structural insufficient-evidence evaluation,
5. human response-level answer/abstention review.

No hosted model, retrieval, reranking, embedding, or generation calls are made.

The official 30-question production RAG benchmark is intentionally reused
because the production retrieval/generation stack did not change during this
evaluation cycle. Re-running a stochastic hosted generator without a production
change would not measure an intervention.
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

from src.evaluation.insufficient_evidence_evaluator import (
    SUPPORTED_CASE_TYPES,
    aggregate_insufficient_evidence_metrics,
    load_insufficient_evidence_records,
)
from src.evaluation.rag_evaluator import (
    aggregate_rag_metrics,
)
from src.evaluation.response_behavior_review import (
    aggregate_response_behavior_review,
    load_review_rows as load_response_review_rows,
)
from src.evaluation.run_insufficient_evidence_evaluation import (
    load_existing_output as load_insufficient_output,
    metric_from_output_record as insufficient_metric_from_output_record,
)
from src.evaluation.run_rag_evaluation import (
    metric_from_output_record as rag_metric_from_output_record,
)
from src.evaluation.run_semantic_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    build_agreements,
    load_existing_output as load_semantic_judge_output,
)
from src.evaluation.semantic_judge import (
    aggregate_judge_agreement,
    aggregate_judge_agreement_by_language_pair,
)
from src.evaluation.semantic_review import (
    aggregate_semantic_review,
    load_review_rows as load_semantic_review_rows,
)


CHECKPOINT_SCHEMA_VERSION = 1

CHECKPOINT_CONFIG_ID = (
    "production-quality-checkpoint-v1"
)

DEFAULT_PRODUCTION_RAG_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_interactions.jsonl"
)

DEFAULT_HUMAN_SEMANTIC_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1_labeled.jsonl"
)

DEFAULT_SEMANTIC_JUDGE_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1_judge.jsonl"
)

DEFAULT_INSUFFICIENT_DATASET_PATH = Path(
    "data/evaluation/"
    "insufficient_evidence_questions.jsonl"
)

DEFAULT_INSUFFICIENT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_v1.jsonl"
)

DEFAULT_RESPONSE_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_v1_response_review.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/"
    "production_quality_checkpoint_v1.json"
)

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)


def _pair_label(
    query_language: str,
    target_language: str,
) -> str:
    """Return canonical query->target language-pair label."""

    return (
        f"{query_language}"
        f"->{target_language}"
    )


def load_jsonl_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load one non-empty JSONL artifact containing object rows."""

    source_path = Path(
        path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Evaluation artifact does "
            f"not exist: {source_path}"
        )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    with source_path.open(
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
                    f"{source_path} on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    f"{source_path} line "
                    f"{line_number} must be "
                    "a JSON object."
                )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            f"{source_path} contains "
            "no evaluation rows."
        )

    return rows


def summarize_production_rag(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Summarize the persisted official production RAG benchmark."""

    rows = (
        load_jsonl_rows(
            path
        )
    )

    metrics = [
        rag_metric_from_output_record(
            row
        )
        for row in rows
    ]

    by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for (
        query_language,
        target_language,
    ) in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_metrics = [
            metric
            for row, metric in zip(
                rows,
                metrics,
                strict=True,
            )
            if (
                row.get(
                    "query_language"
                )
                == query_language
                and row.get(
                    "target_language"
                )
                == target_language
            )
        ]

        if not pair_metrics:
            continue

        by_language_pair[
            _pair_label(
                query_language,
                target_language,
            )
        ] = {
            "question_count": len(
                pair_metrics
            ),
            **aggregate_rag_metrics(
                pair_metrics
            ),
        }

    run_config_ids = {
        row.get(
            "run_config_id"
        )
        for row in rows
    }

    if len(
        run_config_ids
    ) != 1:
        raise ValueError(
            "Production RAG artifact "
            "contains multiple run "
            "configuration IDs."
        )

    return {
        "question_count": len(
            rows
        ),
        "run_config_id": (
            next(
                iter(
                    run_config_ids
                )
            )
        ),
        "overall": (
            aggregate_rag_metrics(
                metrics
            )
        ),
        "by_language_pair": (
            by_language_pair
        ),
    }


def summarize_human_semantic(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Summarize the completed human semantic reference set."""

    rows = (
        load_semantic_review_rows(
            path
        )
    )

    by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for (
        query_language,
        target_language,
    ) in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_rows = [
            row
            for row in rows
            if (
                row.get(
                    "query_language"
                )
                == query_language
                and row.get(
                    "target_language"
                )
                == target_language
            )
        ]

        if not pair_rows:
            continue

        by_language_pair[
            _pair_label(
                query_language,
                target_language,
            )
        ] = (
            aggregate_semantic_review(
                pair_rows
            )
        )

    sample_config_ids = {
        row.get(
            "review_sample_config_id"
        )
        for row in rows
    }

    if len(
        sample_config_ids
    ) != 1:
        raise ValueError(
            "Human semantic artifact "
            "contains multiple sample "
            "configuration IDs."
        )

    return {
        "claim_count": len(
            rows
        ),
        "sample_config_id": (
            next(
                iter(
                    sample_config_ids
                )
            )
        ),
        "overall": (
            aggregate_semantic_review(
                rows
            )
        ),
        "by_language_pair": (
            by_language_pair
        ),
        "sampling_note": (
            "The human semantic set is a "
            "deterministic stratified "
            "diagnostic sample, not a "
            "population-proportional random "
            "sample of all production claims."
        ),
    }


def summarize_semantic_judge(
    human_path: str | Path,
    judge_path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Summarize automated semantic-judge agreement with human labels."""

    human_rows = (
        load_semantic_review_rows(
            human_path
        )
    )

    completed = (
        load_semantic_judge_output(
            judge_path,
            human_rows,
            provider=(
                DEFAULT_JUDGE_PROVIDER
            ),
            model=(
                DEFAULT_JUDGE_MODEL
            ),
        )
    )

    agreements = (
        build_agreements(
            human_rows,
            completed,
        )
    )

    if (
        len(
            agreements
        )
        != len(
            human_rows
        )
    ):
        raise ValueError(
            "Automated semantic judge "
            "checkpoint is incomplete."
        )

    return {
        "claim_count": len(
            agreements
        ),
        "judge_provider": (
            DEFAULT_JUDGE_PROVIDER
        ),
        "judge_model": (
            DEFAULT_JUDGE_MODEL
        ),
        "overall": (
            aggregate_judge_agreement(
                agreements
            )
        ),
        "by_language_pair": (
            aggregate_judge_agreement_by_language_pair(
                agreements
            )
        ),
        "reference_limitations": {
            "human_unsupported_examples": 0,
            "human_needs_review_examples": 0,
            "human_unclear_requirement_examples": 0,
            "note": (
                "High exact agreement does "
                "not validate judge behavior "
                "for classes absent from the "
                "48-claim human reference set."
            ),
        },
    }


def summarize_insufficient_evidence(
    dataset_path: str | Path,
    output_path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Summarize structural answer-vs-withhold benchmark behavior."""

    records = (
        load_insufficient_evidence_records(
            dataset_path
        )
    )

    completed = (
        load_insufficient_output(
            output_path,
            records,
        )
    )

    if (
        len(
            completed
        )
        != len(
            records
        )
    ):
        raise ValueError(
            "Insufficient-evidence "
            "benchmark is incomplete."
        )

    metrics = [
        insufficient_metric_from_output_record(
            completed[
                record.question_id
            ]
        )
        for record in records
    ]

    by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for (
        query_language,
        target_language,
    ) in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_metrics = [
            metric
            for record, metric in zip(
                records,
                metrics,
                strict=True,
            )
            if (
                record.query_language
                == query_language
                and record.target_language
                == target_language
            )
        ]

        if not pair_metrics:
            continue

        by_language_pair[
            _pair_label(
                query_language,
                target_language,
            )
        ] = (
            aggregate_insufficient_evidence_metrics(
                pair_metrics
            )
        )

    by_case_type: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for case_type in (
        SUPPORTED_CASE_TYPES
    ):
        case_metrics = [
            metric
            for record, metric in zip(
                records,
                metrics,
                strict=True,
            )
            if (
                record.case_type
                == case_type
            )
        ]

        if not case_metrics:
            continue

        by_case_type[
            case_type
        ] = (
            aggregate_insufficient_evidence_metrics(
                case_metrics
            )
        )

    present_case_types = sorted(
        {
            record.case_type
            for record in records
        }
    )

    missing_case_types = [
        case_type
        for case_type in (
            SUPPORTED_CASE_TYPES
        )
        if (
            case_type
            not in present_case_types
        )
    ]

    return {
        "question_count": len(
            records
        ),
        "overall": (
            aggregate_insufficient_evidence_metrics(
                metrics
            )
        ),
        "by_language_pair": (
            by_language_pair
        ),
        "by_case_type": (
            by_case_type
        ),
        "present_case_types": (
            present_case_types
        ),
        "not_yet_benchmarked_case_types": (
            missing_case_types
        ),
    }


def summarize_response_behavior(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Summarize human-reviewed user-visible answer/abstention behavior."""

    rows = (
        load_response_review_rows(
            path
        )
    )

    by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for (
        query_language,
        target_language,
    ) in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_rows = [
            row
            for row in rows
            if (
                row.get(
                    "query_language"
                )
                == query_language
                and row.get(
                    "target_language"
                )
                == target_language
            )
        ]

        if not pair_rows:
            continue

        by_language_pair[
            _pair_label(
                query_language,
                target_language,
            )
        ] = (
            aggregate_response_behavior_review(
                pair_rows
            )
        )

    return {
        "question_count": len(
            rows
        ),
        "overall": (
            aggregate_response_behavior_review(
                rows
            )
        ),
        "by_language_pair": (
            by_language_pair
        ),
    }


def build_production_quality_checkpoint(
    *,
    production_rag_path: str | Path = (
        DEFAULT_PRODUCTION_RAG_PATH
    ),
    human_semantic_path: str | Path = (
        DEFAULT_HUMAN_SEMANTIC_PATH
    ),
    semantic_judge_path: str | Path = (
        DEFAULT_SEMANTIC_JUDGE_PATH
    ),
    insufficient_dataset_path: str | Path = (
        DEFAULT_INSUFFICIENT_DATASET_PATH
    ),
    insufficient_output_path: str | Path = (
        DEFAULT_INSUFFICIENT_OUTPUT_PATH
    ),
    response_review_path: str | Path = (
        DEFAULT_RESPONSE_REVIEW_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Build one deterministic consolidated production-quality checkpoint."""

    production_rag = (
        summarize_production_rag(
            production_rag_path
        )
    )

    human_semantic = (
        summarize_human_semantic(
            human_semantic_path
        )
    )

    semantic_judge = (
        summarize_semantic_judge(
            human_semantic_path,
            semantic_judge_path,
        )
    )

    insufficient_evidence = (
        summarize_insufficient_evidence(
            insufficient_dataset_path,
            insufficient_output_path,
        )
    )

    response_behavior = (
        summarize_response_behavior(
            response_review_path
        )
    )

    return {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "checkpoint_config_id": (
            CHECKPOINT_CONFIG_ID
        ),
        "production_rag": (
            production_rag
        ),
        "human_semantic_citation_review": (
            human_semantic
        ),
        "automated_semantic_judge": (
            semantic_judge
        ),
        "insufficient_evidence_structural": (
            insufficient_evidence
        ),
        "human_response_behavior": (
            response_behavior
        ),
        "headline": {
            "production_rag_selected_primary_hit_rate": (
                production_rag[
                    "overall"
                ][
                    "selected_primary_hit_rate"
                ]
            ),
            "production_rag_valid_reference_ratio": (
                production_rag[
                    "overall"
                ][
                    "valid_reference_ratio"
                ]
            ),
            "human_semantic_fully_supported_rate": (
                human_semantic[
                    "overall"
                ][
                    "fully_supported_rate"
                ]
            ),
            "human_semantic_any_support_rate": (
                human_semantic[
                    "overall"
                ][
                    "at_least_partially_supported_rate"
                ]
            ),
            "human_semantic_unsupported_rate": (
                human_semantic[
                    "overall"
                ][
                    "unsupported_rate"
                ]
            ),
            "automated_judge_semantic_exact_accuracy": (
                semantic_judge[
                    "overall"
                ][
                    "semantic_exact_accuracy"
                ]
            ),
            "automated_judge_requirement_accuracy": (
                semantic_judge[
                    "overall"
                ][
                    "citation_requirement_accuracy"
                ]
            ),
            "automated_judge_individual_exact_accuracy": (
                semantic_judge[
                    "overall"
                ][
                    "individual_exact_accuracy"
                ]
            ),
            "structural_withholding_success_rate": (
                insufficient_evidence[
                    "overall"
                ][
                    "withholding_success_rate"
                ]
            ),
            "structural_false_accept_rate": (
                insufficient_evidence[
                    "overall"
                ][
                    "guard_false_accept_rate"
                ]
            ),
            "human_response_behavior_accuracy": (
                response_behavior[
                    "overall"
                ][
                    "behavior_accuracy"
                ]
            ),
            "human_semantic_abstention_success_rate": (
                response_behavior[
                    "overall"
                ][
                    "semantic_abstention_success_rate"
                ]
            ),
            "unsafe_substantive_answer_rate": (
                response_behavior[
                    "overall"
                ][
                    "unsafe_substantive_answer_rate"
                ]
            ),
        },
        "rerun_policy": {
            "hosted_production_rerun_performed": False,
            "official_production_benchmark_retained": True,
            "reason": (
                "The production retrieval, "
                "reranking, context-selection, "
                "prompt, Gemini generation, "
                "citation-processing, and "
                "evidence-guard stack did not "
                "change during this evaluation "
                "cycle. A new stochastic hosted "
                "generation run would therefore "
                "not measure a production "
                "intervention."
            ),
        },
        "measured_remaining_work": [
            (
                "NE->EN retrieval remains the "
                "weakest production language "
                "direction in the existing "
                "structural benchmark."
            ),
            (
                "The automated semantic judge "
                "has not yet been validated "
                "against human unsupported, "
                "needs-review, or unclear "
                "citation-requirement examples."
            ),
            (
                "The insufficient-evidence "
                "benchmark does not yet contain "
                "partial-evidence or mixed "
                "supported/unsupported cases."
            ),
            (
                "The structural evidence guard "
                "can classify a semantically "
                "abstaining response as accepted "
                "when that response contains a "
                "valid citation."
            ),
            (
                "Human semantic review identified "
                "numerical fidelity, truncated "
                "evidence, OCR corruption, and "
                "individual over-citation as "
                "remaining diagnostic failure "
                "modes."
            ),
        ],
    }


def write_checkpoint(
    checkpoint: Mapping[
        str,
        Any,
    ],
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
) -> None:
    """Write the consolidated checkpoint atomically."""

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            checkpoint,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    temporary_path.replace(
        path
    )


def print_checkpoint_summary(
    checkpoint: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print the checkpoint's principal evaluation results."""

    headline = (
        checkpoint[
            "headline"
        ]
    )

    print()
    print(
        "=" * 78
    )

    print(
        "NepalGov AI consolidated production-quality checkpoint"
    )

    print(
        "=" * 78
    )

    print(
        "Official production RAG benchmark:"
    )

    print(
        "  Selected primary hit: "
        f"{headline['production_rag_selected_primary_hit_rate']:.3f}"
    )

    print(
        "  Valid citation references: "
        f"{headline['production_rag_valid_reference_ratio']:.3f}"
    )

    print()
    print(
        "Human semantic citation review:"
    )

    print(
        "  Fully supported: "
        f"{headline['human_semantic_fully_supported_rate']:.3f}"
    )

    print(
        "  At least partially supported: "
        f"{headline['human_semantic_any_support_rate']:.3f}"
    )

    print(
        "  Unsupported: "
        f"{headline['human_semantic_unsupported_rate']:.3f}"
    )

    print()
    print(
        "Automated semantic judge vs human:"
    )

    print(
        "  Semantic exact agreement: "
        f"{headline['automated_judge_semantic_exact_accuracy']:.3f}"
    )

    print(
        "  Citation-requirement agreement: "
        f"{headline['automated_judge_requirement_accuracy']:.3f}"
    )

    print(
        "  Individual-evidence agreement: "
        f"{headline['automated_judge_individual_exact_accuracy']:.3f}"
    )

    print()
    print(
        "Insufficient-evidence structural evaluation:"
    )

    print(
        "  Withholding success: "
        f"{headline['structural_withholding_success_rate']:.3f}"
    )

    print(
        "  Structural false-accept rate: "
        f"{headline['structural_false_accept_rate']:.3f}"
    )

    print()
    print(
        "Human response-level behavior:"
    )

    print(
        "  Behavior accuracy: "
        f"{headline['human_response_behavior_accuracy']:.3f}"
    )

    print(
        "  Semantic abstention success: "
        f"{headline['human_semantic_abstention_success_rate']:.3f}"
    )

    print(
        "  Unsafe substantive-answer rate: "
        f"{headline['unsafe_substantive_answer_rate']:.3f}"
    )

    print()
    print(
        "Hosted production rerun performed: no"
    )

    print(
        "Official production benchmark retained: yes"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build checkpoint CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Build the deterministic "
                "NepalGov AI consolidated "
                "production-quality checkpoint."
            )
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser


def main(
) -> None:
    """Build, persist, and print the production-quality checkpoint."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    checkpoint = (
        build_production_quality_checkpoint()
    )

    write_checkpoint(
        checkpoint,
        args.output,
    )

    print_checkpoint_summary(
        checkpoint
    )


if __name__ == "__main__":
    main()