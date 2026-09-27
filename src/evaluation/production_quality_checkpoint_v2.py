"""Second consolidated production-quality checkpoint for NepalGov AI.

This checkpoint preserves the original production-quality V1 checkpoint and
adds the four evaluation milestones completed in the following development
cycle:

1. partial/mixed insufficient-evidence evaluation,
2. semantic-judge hard-case challenge evaluation,
3. targeted NE->EN retrieval experiments,
4. full production answer-completeness and factual-fidelity review.

The V2 checkpoint is deterministic over persisted evaluation artifacts.

It performs no:

- hosted generation,
- embedding,
- retrieval,
- reranking,
- OCR,
- ingestion,
- vector backfill.

The official 30-question production RAG run remains unchanged because no
experimental retrieval strategy was promoted to production during this cycle.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from src.evaluation.answer_quality_review import (
    aggregate_answer_quality,
    load_review_rows as load_answer_quality_rows,
)
from src.evaluation.insufficient_evidence_evaluator import (
    aggregate_insufficient_evidence_metrics,
    load_insufficient_evidence_records,
)
from src.evaluation.production_quality_checkpoint import (
    build_production_quality_checkpoint as build_v1_checkpoint,
    load_jsonl_rows,
)
from src.evaluation.response_behavior_review import (
    aggregate_response_behavior_review,
    load_review_rows as load_response_review_rows,
)
from src.evaluation.run_insufficient_evidence_evaluation import (
    metric_from_output_record as insufficient_metric_from_output_record,
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
from src.evaluation.semantic_reference import (
    load_semantic_reference_rows,
)


CHECKPOINT_SCHEMA_VERSION = 1

CHECKPOINT_CONFIG_ID = (
    "production-quality-checkpoint-v2"
)

DEFAULT_PARTIAL_MIXED_DATASET_PATH = Path(
    "data/evaluation/"
    "insufficient_evidence_partial_mixed_v1.jsonl"
)

DEFAULT_PARTIAL_MIXED_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_partial_mixed_v1.jsonl"
)

DEFAULT_PARTIAL_MIXED_RESPONSE_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_partial_mixed_v1_response_review.jsonl"
)

DEFAULT_HARD_REFERENCE_PATH = Path(
    "data/evaluation/semantic/"
    "semantic_judge_hard_cases_v1.jsonl"
)

DEFAULT_HARD_JUDGE_PATH = Path(
    "data/evaluation/semantic/"
    "semantic_judge_hard_cases_v1_judge.jsonl"
)

HARD_JUDGE_RUN_CONFIG_ID = (
    "semantic-judge-hard-cases-v1-judge"
)

DEFAULT_NE_EN_RETRIEVAL_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "ne_en_targeted_retrieval_v1.json"
)

EXPECTED_NE_EN_RUN_CONFIG_ID = (
    "ne-en-targeted-retrieval-v1"
)

DEFAULT_ANSWER_QUALITY_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_answer_quality_review_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/"
    "production_quality_checkpoint_v2.json"
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
    """Return one canonical query->evidence language-pair label."""

    return (
        f"{query_language}"
        f"->{target_language}"
    )


def _load_json_object(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Load one persisted non-empty JSON object."""

    source_path = Path(
        path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Evaluation artifact does "
            f"not exist: {source_path}"
        )

    try:
        with source_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON artifact: "
            f"{source_path}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"{source_path} must contain "
            "one JSON object."
        )

    if not data:
        raise ValueError(
            f"{source_path} contains "
            "an empty JSON object."
        )

    return data


def _group_rows_by_language_pair(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    list[
        Mapping[
            str,
            Any,
        ]
    ],
]:
    """Group persisted rows by query and evidence language."""

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[
            Mapping[
                str,
                Any,
            ]
        ],
    ] = defaultdict(
        list
    )

    for row in rows:
        grouped[
            (
                str(
                    row[
                        "query_language"
                    ]
                ),
                str(
                    row[
                        "target_language"
                    ]
                ),
            )
        ].append(
            row
        )

    return dict(
        grouped
    )


def summarize_partial_mixed_evidence(
    *,
    dataset_path: str | Path = (
        DEFAULT_PARTIAL_MIXED_DATASET_PATH
    ),
    output_path: str | Path = (
        DEFAULT_PARTIAL_MIXED_OUTPUT_PATH
    ),
    response_review_path: str | Path = (
        DEFAULT_PARTIAL_MIXED_RESPONSE_REVIEW_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the eight-case partial/mixed evidence extension."""

    records = (
        load_insufficient_evidence_records(
            dataset_path
        )
    )

    output_rows = (
        load_jsonl_rows(
            output_path
        )
    )

    records_by_id = {
        record.question_id: record
        for record in records
    }

    output_by_id: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for row in output_rows:
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
                "Partial/mixed output row "
                "has no valid question_id."
            )

        if (
            question_id
            in output_by_id
        ):
            raise ValueError(
                "Partial/mixed output contains "
                "duplicate question_id "
                f"{question_id!r}."
            )

        output_by_id[
            question_id
        ] = row

    if (
        set(
            output_by_id
        )
        != set(
            records_by_id
        )
    ):
        raise ValueError(
            "Partial/mixed persisted output "
            "does not match its benchmark."
        )

    metrics_by_id = {
        question_id: (
            insufficient_metric_from_output_record(
                output_by_id[
                    question_id
                ]
            )
        )
        for question_id in (
            records_by_id
        )
    }

    structural_overall = (
        aggregate_insufficient_evidence_metrics(
            metrics_by_id.values()
        )
    )

    structural_by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_metrics = [
            metrics_by_id[
                record.question_id
            ]
            for record in records
            if (
                record.query_language,
                record.target_language,
            )
            == pair
        ]

        if not pair_metrics:
            continue

        structural_by_language_pair[
            _pair_label(
                *pair
            )
        ] = (
            aggregate_insufficient_evidence_metrics(
                pair_metrics
            )
        )

    structural_by_case_type: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for case_type in (
        "partial_evidence",
        "mixed_supported_unsupported",
    ):
        case_metrics = [
            metrics_by_id[
                record.question_id
            ]
            for record in records
            if (
                record.case_type
                == case_type
            )
        ]

        if not case_metrics:
            continue

        structural_by_case_type[
            case_type
        ] = (
            aggregate_insufficient_evidence_metrics(
                case_metrics
            )
        )

    response_rows = (
        load_response_review_rows(
            response_review_path
        )
    )

    response_overall = (
        aggregate_response_behavior_review(
            response_rows
        )
    )

    response_grouped = (
        _group_rows_by_language_pair(
            response_rows
        )
    )

    response_by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_rows = (
            response_grouped.get(
                pair
            )
        )

        if not pair_rows:
            continue

        response_by_language_pair[
            _pair_label(
                *pair
            )
        ] = (
            aggregate_response_behavior_review(
                pair_rows
            )
        )

    return {
        "question_count": len(
            records
        ),
        "structural": {
            "overall": (
                structural_overall
            ),
            "by_language_pair": (
                structural_by_language_pair
            ),
            "by_case_type": (
                structural_by_case_type
            ),
        },
        "human_response_behavior": {
            "overall": (
                response_overall
            ),
            "by_language_pair": (
                response_by_language_pair
            ),
        },
        "interpretation": (
            "The extension evaluates partial_evidence "
            "and mixed_supported_unsupported cases. "
            "Structural acceptance is expected so the "
            "supported portion remains visible; human "
            "review determines whether unsupported "
            "portions are explicitly limited."
        ),
    }


def summarize_hard_semantic_judge(
    *,
    reference_path: str | Path = (
        DEFAULT_HARD_REFERENCE_PATH
    ),
    judge_path: str | Path = (
        DEFAULT_HARD_JUDGE_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the hard negative and ambiguity judge challenge set."""

    human_rows = (
        load_semantic_reference_rows(
            reference_path
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
            judge_run_config_id=(
                HARD_JUDGE_RUN_CONFIG_ID
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
            "Hard semantic-judge output "
            "is incomplete."
        )

    by_human_semantic_label: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for label in (
        "unsupported",
        "needs_review",
        "supported",
    ):
        label_agreements = [
            item
            for item in agreements
            if (
                item.semantic_human_label
                == label
            )
        ]

        if not label_agreements:
            continue

        by_human_semantic_label[
            label
        ] = (
            aggregate_judge_agreement(
                label_agreements
            )
        )

    by_requirement_label: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for label in (
        "required",
        "unclear",
    ):
        label_agreements = [
            item
            for item in agreements
            if (
                item.citation_requirement_human_label
                == label
            )
        ]

        if not label_agreements:
            continue

        by_requirement_label[
            label
        ] = (
            aggregate_judge_agreement(
                label_agreements
            )
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
        "by_human_semantic_label": (
            by_human_semantic_label
        ),
        "by_human_citation_requirement_label": (
            by_requirement_label
        ),
        "interpretation": (
            "The judge correctly handles clear unsupported "
            "hard negatives but is over-decisive on ambiguity: "
            "human needs_review cases are mapped to "
            "partially_supported and human unclear citation "
            "requirements are mapped to required."
        ),
    }


def summarize_ne_en_retrieval_experiment(
    path: str | Path = (
        DEFAULT_NE_EN_RETRIEVAL_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Load and validate the persisted targeted NE->EN experiment."""

    data = (
        _load_json_object(
            path
        )
    )

    if (
        data.get(
            "run_config_id"
        )
        != EXPECTED_NE_EN_RUN_CONFIG_ID
    ):
        raise ValueError(
            "Unexpected targeted NE->EN "
            "run_config_id."
        )

    if (
        data.get(
            "production_changed"
        )
        is not False
    ):
        raise ValueError(
            "Targeted NE->EN artifact must "
            "remain evaluation-only."
        )

    evaluation_slice = (
        data.get(
            "evaluation_slice"
        )
    )

    if not isinstance(
        evaluation_slice,
        Mapping,
    ):
        raise ValueError(
            "Targeted NE->EN artifact contains "
            "no valid evaluation_slice."
        )

    if (
        evaluation_slice.get(
            "question_count"
        )
        != 6
    ):
        raise ValueError(
            "Targeted NE->EN experiment must "
            "contain six questions."
        )

    first_stage_metrics = (
        data.get(
            "first_stage_metrics"
        )
    )

    reranked_metrics = (
        data.get(
            "reranked_metrics"
        )
    )

    conclusion = (
        data.get(
            "conclusion"
        )
    )

    if not isinstance(
        first_stage_metrics,
        Mapping,
    ):
        raise ValueError(
            "Targeted NE->EN artifact contains "
            "no first-stage metrics."
        )

    if not isinstance(
        reranked_metrics,
        Mapping,
    ):
        raise ValueError(
            "Targeted NE->EN artifact contains "
            "no reranked metrics."
        )

    if not isinstance(
        conclusion,
        Mapping,
    ):
        raise ValueError(
            "Targeted NE->EN artifact contains "
            "no conclusion."
        )

    return data


def summarize_answer_quality(
    path: str | Path = (
        DEFAULT_ANSWER_QUALITY_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the complete production answer-quality human review."""

    rows = (
        load_answer_quality_rows(
            path
        )
    )

    grouped = (
        _group_rows_by_language_pair(
            rows
        )
    )

    by_language_pair: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        pair_rows = (
            grouped.get(
                pair
            )
        )

        if not pair_rows:
            continue

        by_language_pair[
            _pair_label(
                *pair
            )
        ] = (
            aggregate_answer_quality(
                pair_rows
            )
        )

    overall = (
        aggregate_answer_quality(
            rows
        )
    )

    if (
        overall[
            "completion_rate"
        ]
        != 1.0
    ):
        raise ValueError(
            "Answer-quality human review "
            "is not complete."
        )

    return {
        "question_count": len(
            rows
        ),
        "overall": (
            overall
        ),
        "by_language_pair": (
            by_language_pair
        ),
        "interpretation": (
            "Completeness is evaluated against the "
            "manually verified gold-relevant evidence. "
            "Factual fidelity is evaluated against exact "
            "selected and gold passages. The two dimensions "
            "are intentionally reported separately."
        ),
    }


def build_production_quality_checkpoint_v2(
) -> dict[
    str,
    Any,
]:
    """Build the deterministic second consolidated quality checkpoint."""

    base_checkpoint = (
        build_v1_checkpoint()
    )

    partial_mixed = (
        summarize_partial_mixed_evidence()
    )

    hard_judge = (
        summarize_hard_semantic_judge()
    )

    ne_en_retrieval = (
        summarize_ne_en_retrieval_experiment()
    )

    answer_quality = (
        summarize_answer_quality()
    )

    base_headline = dict(
        base_checkpoint[
            "headline"
        ]
    )

    answer_overall = (
        answer_quality[
            "overall"
        ]
    )

    partial_response_overall = (
        partial_mixed[
            "human_response_behavior"
        ][
            "overall"
        ]
    )

    hard_overall = (
        hard_judge[
            "overall"
        ]
    )

    translated_bge_10 = (
        ne_en_retrieval[
            "reranked_metrics"
        ][
            "10"
        ][
            "translated_bge"
        ]
    )

    production_bge_10 = (
        ne_en_retrieval[
            "reranked_metrics"
        ][
            "10"
        ][
            "production_bge"
        ]
    )

    headline = {
        **base_headline,
        "partial_mixed_behavior_accuracy": (
            partial_response_overall[
                "behavior_accuracy"
            ]
        ),
        "partial_mixed_response_success_rate": (
            partial_response_overall[
                "partial_response_success_rate"
            ]
        ),
        "hard_case_judge_semantic_exact_accuracy": (
            hard_overall[
                "semantic_exact_accuracy"
            ]
        ),
        "hard_case_judge_requirement_accuracy": (
            hard_overall[
                "citation_requirement_accuracy"
            ]
        ),
        "hard_case_judge_individual_exact_accuracy": (
            hard_overall[
                "individual_exact_accuracy"
            ]
        ),
        "ne_en_production_bge_hit_at_10": (
            production_bge_10[
                "hit_rate"
            ]
        ),
        "ne_en_production_bge_recall_at_10": (
            production_bge_10[
                "recall"
            ]
        ),
        "ne_en_experimental_bge_hit_at_10": (
            translated_bge_10[
                "hit_rate"
            ]
        ),
        "ne_en_experimental_bge_recall_at_10": (
            translated_bge_10[
                "recall"
            ]
        ),
        "answer_quality_fully_complete_rate": (
            answer_overall[
                "fully_complete_rate"
            ]
        ),
        "answer_quality_at_least_mostly_complete_rate": (
            answer_overall[
                "at_least_mostly_complete_rate"
            ]
        ),
        "answer_quality_fully_faithful_rate": (
            answer_overall[
                "fully_faithful_rate"
            ]
        ),
        "answer_quality_no_major_factual_error_rate": (
            answer_overall[
                "no_major_factual_error_rate"
            ]
        ),
        "answer_quality_strong_answer_rate": (
            answer_overall[
                "strong_answer_rate"
            ]
        ),
        "answer_quality_acceptable_answer_rate": (
            answer_overall[
                "acceptable_answer_rate"
            ]
        ),
    }

    return {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "checkpoint_config_id": (
            CHECKPOINT_CONFIG_ID
        ),
        "base_checkpoint_config_id": (
            base_checkpoint[
                "checkpoint_config_id"
            ]
        ),
        "base_production_quality": (
            base_checkpoint
        ),
        "cycle_extensions": {
            "partial_mixed_evidence": (
                partial_mixed
            ),
            "semantic_judge_hard_cases": (
                hard_judge
            ),
            "targeted_ne_en_retrieval": (
                ne_en_retrieval
            ),
            "human_answer_quality": (
                answer_quality
            ),
        },
        "headline": (
            headline
        ),
        "production_decision": {
            "production_retrieval_changed": False,
            "hosted_production_rag_rerun_performed": False,
            "official_production_benchmark_retained": True,
            "reason": (
                "The measured NE->EN retrieval improvement "
                "depends on manually controlled English query "
                "counterparts. Automatic translation quality, "
                "latency, cost, and full-benchmark regression "
                "behavior have not yet been evaluated, so the "
                "experimental route was not promoted."
            ),
        },
        "measured_remaining_work": [
            (
                "Evaluate an automatic translation or "
                "cross-lingual reformulation mechanism before "
                "considering the measured NE->EN translated-hybrid "
                "candidate path for production."
            ),
            (
                "Investigate table-aware passage representations "
                "for table-heavy Economic Survey evidence."
            ),
            (
                "Improve answer completeness: only 70.0% of the "
                "30 reviewed production answers were fully complete "
                "and 80.0% were at least mostly complete."
            ),
            (
                "Preserve the strong factual-safety result while "
                "eliminating the two observed localized factual "
                "issues; no reviewed answer contained a major "
                "factual error."
            ),
            (
                "Treat automated-judge ambiguity decisions "
                "cautiously because all four needs_review and all "
                "four unclear-requirement challenge cases were "
                "resolved too aggressively by the judge."
            ),
        ],
    }


def write_checkpoint(
    checkpoint: Mapping[
        str,
        Any,
    ],
    *,
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

    temporary = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    with temporary.open(
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

    temporary.replace(
        path
    )


def print_checkpoint_summary(
    checkpoint: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print the new-cycle headline metrics."""

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
        "NepalGov AI production-quality checkpoint V2"
    )

    print(
        "=" * 78
    )

    print(
        "Partial/mixed response behavior:"
    )

    print(
        "  Behavior accuracy: "
        f"{headline['partial_mixed_behavior_accuracy']:.3f}"
    )

    print(
        "  Partial response success: "
        f"{headline['partial_mixed_response_success_rate']:.3f}"
    )

    print()

    print(
        "Hard semantic-judge challenge:"
    )

    print(
        "  Semantic exact: "
        f"{headline['hard_case_judge_semantic_exact_accuracy']:.3f}"
    )

    print(
        "  Requirement exact: "
        f"{headline['hard_case_judge_requirement_accuracy']:.3f}"
    )

    print(
        "  Individual exact: "
        f"{headline['hard_case_judge_individual_exact_accuracy']:.3f}"
    )

    print()

    print(
        "Targeted NE->EN BGE @10:"
    )

    print(
        "  Production hit/recall: "
        f"{headline['ne_en_production_bge_hit_at_10']:.3f} / "
        f"{headline['ne_en_production_bge_recall_at_10']:.3f}"
    )

    print(
        "  Experimental hit/recall: "
        f"{headline['ne_en_experimental_bge_hit_at_10']:.3f} / "
        f"{headline['ne_en_experimental_bge_recall_at_10']:.3f}"
    )

    print()

    print(
        "Human answer quality:"
    )

    print(
        "  Fully complete: "
        f"{headline['answer_quality_fully_complete_rate']:.3f}"
    )

    print(
        "  At least mostly complete: "
        f"{headline['answer_quality_at_least_mostly_complete_rate']:.3f}"
    )

    print(
        "  Fully faithful: "
        f"{headline['answer_quality_fully_faithful_rate']:.3f}"
    )

    print(
        "  No major factual error: "
        f"{headline['answer_quality_no_major_factual_error_rate']:.3f}"
    )

    print(
        "  Strong answers: "
        f"{headline['answer_quality_strong_answer_rate']:.3f}"
    )

    print(
        "  Acceptable answers: "
        f"{headline['answer_quality_acceptable_answer_rate']:.3f}"
    )

    print()

    production_decision = (
        checkpoint[
            "production_decision"
        ]
    )

    print(
        "Production retrieval changed: "
        f"{production_decision['production_retrieval_changed']}"
    )

    print(
        "Official production benchmark retained: "
        f"{production_decision['official_production_benchmark_retained']}"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the V2 checkpoint command-line interface."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Build the second deterministic "
                "NepalGov AI production-quality checkpoint."
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
    """Build, persist, and summarize checkpoint V2."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    checkpoint = (
        build_production_quality_checkpoint_v2()
    )

    write_checkpoint(
        checkpoint,
        output_path=(
            args.output
        ),
    )

    print_checkpoint_summary(
        checkpoint
    )

    print()

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()