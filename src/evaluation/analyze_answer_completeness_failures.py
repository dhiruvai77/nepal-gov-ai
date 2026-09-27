"""Analyze answer-completeness failures in the production RAG benchmark.

This diagnostic operates only on the completed human answer-quality review.

It separates non-complete answers according to whether the manually verified
gold evidence was actually present in the five selected production passages.

The purpose is to distinguish failures that can plausibly be addressed by
generation/prompt changes from failures that require retrieval or context
selection improvements.

Failure categories:

primary_evidence_not_selected
    At least one manually verified primary passage was absent from selected
    production evidence. The generator therefore did not receive the central
    benchmark evidence.

supporting_evidence_not_selected
    All primary evidence was selected, but one or more broader gold-relevant
    supporting passages were absent. The answer is otherwise mostly complete.

mixed_context_and_generation_omission
    Primary evidence was selected, but broader gold evidence was incomplete,
    and the human review still judged the answer materially incomplete.
    Existing selected primary evidence may contain omitted answer material, so
    both context coverage and generation behavior may contribute.

generation_omission_despite_full_gold_context
    All manually verified gold evidence was present in selected context, but
    the human review still judged the answer incomplete or mostly complete.
    This is the cleanest prompt/generation intervention target.

No retrieval, generation, embedding, reranking, Qdrant, or hosted API calls are
made by this module.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from src.evaluation.answer_quality_review import (
    REVIEW_STATUS_COMPLETED,
    load_review_rows,
)


ANALYSIS_SCHEMA_VERSION = 1

ANALYSIS_CONFIG_ID = (
    "production-rag-v2-completeness-failure-analysis-v1"
)

DEFAULT_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_answer_quality_review_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_failure_analysis_v1.json"
)

CAUSE_PRIMARY_NOT_SELECTED = (
    "primary_evidence_not_selected"
)

CAUSE_SUPPORTING_NOT_SELECTED = (
    "supporting_evidence_not_selected"
)

CAUSE_MIXED = (
    "mixed_context_and_generation_omission"
)

CAUSE_GENERATION = (
    "generation_omission_despite_full_gold_context"
)

CAUSE_NEEDS_REVIEW = (
    "needs_review"
)

SUPPORTED_CAUSES = (
    CAUSE_PRIMARY_NOT_SELECTED,
    CAUSE_SUPPORTING_NOT_SELECTED,
    CAUSE_MIXED,
    CAUSE_GENERATION,
    CAUSE_NEEDS_REVIEW,
)

NON_COMPLETE_LABELS = (
    "mostly_complete",
    "incomplete",
    "needs_review",
)

FLOAT_TOLERANCE = 1e-9


def _ordered_unique(
    values: Sequence[str],
) -> list[str]:
    """Return unique values while preserving first appearance."""

    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(
            value
        )

        ordered.append(
            value
        )

    return ordered


def _require_string_list(
    value: Any,
    *,
    field_name: str,
    question_id: str,
) -> list[str]:
    """Return one validated list of non-empty strings."""

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            f"{question_id}: {field_name} "
            "must be a list."
        )

    result: list[str] = []

    for item in value:
        if (
            not isinstance(
                item,
                str,
            )
            or not item.strip()
        ):
            raise ValueError(
                f"{question_id}: {field_name} "
                "contains an invalid point ID."
            )

        result.append(
            item
        )

    if (
        len(
            result
        )
        != len(
            set(
                result
            )
        )
    ):
        raise ValueError(
            f"{question_id}: {field_name} "
            "contains duplicate point IDs."
        )

    return result


def selected_point_ids(
    row: Mapping[
        str,
        Any,
    ],
) -> list[str]:
    """Return selected production point IDs in evidence order."""

    question_id = str(
        row.get(
            "question_id"
        )
        or ""
    )

    selected = (
        row.get(
            "selected_evidence"
        )
    )

    if not isinstance(
        selected,
        list,
    ):
        raise ValueError(
            f"{question_id}: selected_evidence "
            "must be a list."
        )

    point_ids: list[str] = []

    for evidence in selected:
        if not isinstance(
            evidence,
            Mapping,
        ):
            raise ValueError(
                f"{question_id}: selected evidence "
                "entries must be objects."
            )

        point_id = (
            evidence.get(
                "point_id"
            )
        )

        if (
            not isinstance(
                point_id,
                str,
            )
            or not point_id.strip()
        ):
            raise ValueError(
                f"{question_id}: selected evidence "
                "contains an invalid point_id."
            )

        point_ids.append(
            point_id
        )

    if (
        len(
            point_ids
        )
        != len(
            set(
                point_ids
            )
        )
    ):
        raise ValueError(
            f"{question_id}: selected evidence "
            "contains duplicate point IDs."
        )

    return point_ids


def calculate_selected_gold_coverage(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """Recalculate selected primary/relevant coverage from persisted IDs."""

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
            "question_id must contain "
            "non-whitespace text."
        )

    primary_ids = (
        _require_string_list(
            row.get(
                "primary_relevant_chunk_ids"
            ),
            field_name=(
                "primary_relevant_chunk_ids"
            ),
            question_id=(
                question_id
            ),
        )
    )

    relevant_ids = (
        _require_string_list(
            row.get(
                "relevant_chunk_ids"
            ),
            field_name=(
                "relevant_chunk_ids"
            ),
            question_id=(
                question_id
            ),
        )
    )

    if not relevant_ids:
        raise ValueError(
            f"{question_id}: relevant evidence "
            "cannot be empty."
        )

    if not primary_ids:
        raise ValueError(
            f"{question_id}: primary evidence "
            "cannot be empty."
        )

    if not set(
        primary_ids
    ).issubset(
        set(
            relevant_ids
        )
    ):
        raise ValueError(
            f"{question_id}: primary evidence "
            "must be part of relevant evidence."
        )

    selected_ids = (
        selected_point_ids(
            row
        )
    )

    selected_set = set(
        selected_ids
    )

    primary_set = set(
        primary_ids
    )

    relevant_set = set(
        relevant_ids
    )

    selected_primary_ids = [
        point_id
        for point_id in primary_ids
        if (
            point_id
            in selected_set
        )
    ]

    missing_primary_ids = [
        point_id
        for point_id in primary_ids
        if (
            point_id
            not in selected_set
        )
    ]

    selected_relevant_ids = [
        point_id
        for point_id in relevant_ids
        if (
            point_id
            in selected_set
        )
    ]

    missing_relevant_ids = [
        point_id
        for point_id in relevant_ids
        if (
            point_id
            not in selected_set
        )
    ]

    primary_recall = (
        len(
            selected_primary_ids
        )
        / len(
            primary_set
        )
    )

    relevant_recall = (
        len(
            selected_relevant_ids
        )
        / len(
            relevant_set
        )
    )

    persisted_primary_hit = (
        row.get(
            "selected_primary_hit"
        )
    )

    if (
        persisted_primary_hit
        is not None
    ):
        expected_hit = int(
            bool(
                selected_primary_ids
            )
        )

        if (
            persisted_primary_hit
            != expected_hit
        ):
            raise ValueError(
                f"{question_id}: persisted "
                "selected_primary_hit does not "
                "match selected evidence IDs."
            )

    persisted_recall = (
        row.get(
            "selected_relevant_recall"
        )
    )

    if (
        persisted_recall
        is not None
    ):
        if not isinstance(
            persisted_recall,
            (int, float),
        ):
            raise ValueError(
                f"{question_id}: persisted "
                "selected_relevant_recall must "
                "be numeric."
            )

        if (
            abs(
                float(
                    persisted_recall
                )
                - relevant_recall
            )
            > FLOAT_TOLERANCE
        ):
            raise ValueError(
                f"{question_id}: persisted "
                "selected_relevant_recall does "
                "not match selected evidence IDs."
            )

    return {
        "selected_point_ids": (
            selected_ids
        ),
        "selected_primary_ids": (
            selected_primary_ids
        ),
        "missing_primary_ids": (
            missing_primary_ids
        ),
        "selected_relevant_ids": (
            selected_relevant_ids
        ),
        "missing_relevant_ids": (
            missing_relevant_ids
        ),
        "primary_recall": (
            primary_recall
        ),
        "relevant_recall": (
            relevant_recall
        ),
        "all_primary_selected": (
            len(
                missing_primary_ids
            )
            == 0
        ),
        "all_relevant_selected": (
            len(
                missing_relevant_ids
            )
            == 0
        ),
    }


def classify_failure_cause(
    row: Mapping[
        str,
        Any,
    ],
    coverage: Mapping[
        str,
        Any,
    ],
) -> str | None:
    """Classify one reviewed answer using evidence availability and label."""

    completeness = (
        row.get(
            "completeness_label"
        )
    )

    if (
        completeness
        == "complete"
    ):
        return None

    if (
        completeness
        == "needs_review"
    ):
        return (
            CAUSE_NEEDS_REVIEW
        )

    if (
        completeness
        not in {
            "mostly_complete",
            "incomplete",
        }
    ):
        raise ValueError(
            "Unsupported completeness label "
            f"{completeness!r}."
        )

    all_primary = bool(
        coverage[
            "all_primary_selected"
        ]
    )

    all_relevant = bool(
        coverage[
            "all_relevant_selected"
        ]
    )

    if not all_primary:
        return (
            CAUSE_PRIMARY_NOT_SELECTED
        )

    if all_relevant:
        return (
            CAUSE_GENERATION
        )

    if (
        completeness
        == "incomplete"
    ):
        return (
            CAUSE_MIXED
        )

    return (
        CAUSE_SUPPORTING_NOT_SELECTED
    )


def build_failure_record(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
] | None:
    """Build one deterministic non-complete-answer diagnostic record."""

    if (
        row.get(
            "review_status"
        )
        != REVIEW_STATUS_COMPLETED
    ):
        raise ValueError(
            f"{row.get('question_id')}: "
            "answer-quality review is not completed."
        )

    coverage = (
        calculate_selected_gold_coverage(
            row
        )
    )

    cause = (
        classify_failure_cause(
            row,
            coverage,
        )
    )

    if cause is None:
        return None

    question_id = (
        row[
            "question_id"
        ]
    )

    notes = (
        row.get(
            "review_notes"
        )
    )

    if (
        notes is not None
        and not isinstance(
            notes,
            str,
        )
    ):
        raise ValueError(
            f"{question_id}: review_notes "
            "must be a string or null."
        )

    return {
        "question_id": (
            question_id
        ),
        "query": (
            row[
                "query"
            ]
        ),
        "query_language": (
            row[
                "query_language"
            ]
        ),
        "target_language": (
            row[
                "target_language"
            ]
        ),
        "category": (
            row[
                "category"
            ]
        ),
        "completeness_label": (
            row[
                "completeness_label"
            ]
        ),
        "factual_fidelity_label": (
            row[
                "factual_fidelity_label"
            ]
        ),
        "failure_cause": (
            cause
        ),
        "primary_recall": (
            coverage[
                "primary_recall"
            ]
        ),
        "relevant_recall": (
            coverage[
                "relevant_recall"
            ]
        ),
        "all_primary_selected": (
            coverage[
                "all_primary_selected"
            ]
        ),
        "all_relevant_selected": (
            coverage[
                "all_relevant_selected"
            ]
        ),
        "selected_primary_ids": (
            coverage[
                "selected_primary_ids"
            ]
        ),
        "missing_primary_ids": (
            coverage[
                "missing_primary_ids"
            ]
        ),
        "selected_relevant_ids": (
            coverage[
                "selected_relevant_ids"
            ]
        ),
        "missing_relevant_ids": (
            coverage[
                "missing_relevant_ids"
            ]
        ),
        "review_notes": (
            notes
        ),
    }


def build_analysis(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:
    """Build the full deterministic answer-completeness failure analysis."""

    if not rows:
        raise ValueError(
            "At least one answer-quality "
            "review row is required."
        )

    failure_records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    complete_count = 0

    for row in rows:
        if (
            row.get(
                "review_status"
            )
            != REVIEW_STATUS_COMPLETED
        ):
            raise ValueError(
                "Answer-quality review "
                "must be fully completed."
            )

        failure_record = (
            build_failure_record(
                row
            )
        )

        if (
            failure_record
            is None
        ):
            complete_count += 1

            continue

        failure_records.append(
            failure_record
        )

    cause_counts = Counter(
        record[
            "failure_cause"
        ]
        for record in (
            failure_records
        )
    )

    completeness_counts = Counter(
        record[
            "completeness_label"
        ]
        for record in (
            failure_records
        )
    )

    # A prompt-only benchmark is strongest when all gold evidence was selected.
    # Mixed cases are also useful prompt experiments because the selected
    # primary passage may already contain material information that generation
    # omitted, but they cannot establish that prompt changes alone solve the
    # complete answer-quality problem.
    clean_generation_candidates = [
        record[
            "question_id"
        ]
        for record in (
            failure_records
        )
        if (
            record[
                "failure_cause"
            ]
            == CAUSE_GENERATION
        )
    ]

    mixed_generation_candidates = [
        record[
            "question_id"
        ]
        for record in (
            failure_records
        )
        if (
            record[
                "failure_cause"
            ]
            == CAUSE_MIXED
        )
    ]

    context_candidates = [
        record[
            "question_id"
        ]
        for record in (
            failure_records
        )
        if (
            record[
                "failure_cause"
            ]
            in {
                CAUSE_PRIMARY_NOT_SELECTED,
                CAUSE_SUPPORTING_NOT_SELECTED,
                CAUSE_MIXED,
            }
        )
    ]

    prompt_benchmark_candidates = (
        _ordered_unique(
            [
                *clean_generation_candidates,
                *mixed_generation_candidates,
            ]
        )
    )

    return {
        "schema_version": (
            ANALYSIS_SCHEMA_VERSION
        ),
        "analysis_config_id": (
            ANALYSIS_CONFIG_ID
        ),
        "source_review_config_id": (
            "production-rag-v2-answer-quality-v1"
        ),
        "question_count": len(
            rows
        ),
        "fully_complete_count": (
            complete_count
        ),
        "non_complete_count": len(
            failure_records
        ),
        "non_complete_rate": (
            len(
                failure_records
            )
            / len(
                rows
            )
        ),
        "by_completeness_label": {
            label: (
                completeness_counts.get(
                    label,
                    0,
                )
            )
            for label in (
                NON_COMPLETE_LABELS
            )
        },
        "by_failure_cause": {
            cause: (
                cause_counts.get(
                    cause,
                    0,
                )
            )
            for cause in (
                SUPPORTED_CAUSES
            )
        },
        "clean_generation_intervention_question_ids": (
            clean_generation_candidates
        ),
        "mixed_generation_intervention_question_ids": (
            mixed_generation_candidates
        ),
        "prompt_benchmark_question_ids": (
            prompt_benchmark_candidates
        ),
        "retrieval_or_context_intervention_question_ids": (
            context_candidates
        ),
        "failures": (
            failure_records
        ),
        "interpretation": {
            "primary_evidence_not_selected": (
                "The generator never received all manually "
                "verified primary evidence, so retrieval or "
                "context selection is the first limiting layer."
            ),
            "supporting_evidence_not_selected": (
                "Primary evidence was available, but one or more "
                "supporting gold passages were absent. Improving "
                "context coverage may address the remaining omission."
            ),
            "mixed_context_and_generation_omission": (
                "Primary evidence was present but broader gold "
                "coverage was incomplete, while the answer was still "
                "materially incomplete. Both generation behavior and "
                "context coverage may contribute."
            ),
            "generation_omission_despite_full_gold_context": (
                "All manually verified gold evidence was selected, "
                "so generation behavior can be isolated without "
                "changing retrieval."
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
    """Write the analysis artifact atomically."""

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


def print_analysis(
    analysis: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print one compact diagnostic summary."""

    print()
    print(
        "=" * 100
    )

    print(
        "Production answer-completeness failure analysis"
    )

    print(
        "=" * 100
    )

    print(
        "Questions: "
        f"{analysis['question_count']}"
    )

    print(
        "Fully complete: "
        f"{analysis['fully_complete_count']}"
    )

    print(
        "Non-complete: "
        f"{analysis['non_complete_count']}"
    )

    print()

    print(
        "Failure causes:"
    )

    for (
        cause,
        count,
    ) in (
        analysis[
            "by_failure_cause"
        ].items()
    ):
        print(
            f"  {cause:<48}"
            f"{count:>3}"
        )

    print()
    print(
        f"{'Question':<14}"
        f"{'Pair':<9}"
        f"{'Label':<18}"
        f"{'PrimRec':>9}"
        f"{'RelRec':>9}  "
        f"{'Cause'}"
    )

    print(
        "-" * 100
    )

    for failure in (
        analysis[
            "failures"
        ]
    ):
        pair = (
            f"{failure['query_language']}"
            f"->{failure['target_language']}"
        )

        print(
            f"{failure['question_id']:<14}"
            f"{pair:<9}"
            f"{failure['completeness_label']:<18}"
            f"{failure['primary_recall']:>9.3f}"
            f"{failure['relevant_recall']:>9.3f}  "
            f"{failure['failure_cause']}"
        )

    print()
    print(
        "Clean generation-intervention candidates:"
    )

    clean_candidates = (
        analysis[
            "clean_generation_intervention_question_ids"
        ]
    )

    print(
        ", ".join(
            clean_candidates
        )
        if clean_candidates
        else "None"
    )

    print()
    print(
        "Mixed generation-intervention candidates:"
    )

    mixed_candidates = (
        analysis[
            "mixed_generation_intervention_question_ids"
        ]
    )

    print(
        ", ".join(
            mixed_candidates
        )
        if mixed_candidates
        else "None"
    )

    print()
    print(
        "Retrieval/context intervention candidates:"
    )

    context_candidates = (
        analysis[
            "retrieval_or_context_intervention_question_ids"
        ]
    )

    print(
        ", ".join(
            context_candidates
        )
        if context_candidates
        else "None"
    )


def run_analysis(
    *,
    review_path: str | Path = (
        DEFAULT_REVIEW_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Load the completed review, analyze it, and persist the result."""

    rows = (
        load_review_rows(
            review_path
        )
    )

    analysis = (
        build_analysis(
            rows
        )
    )

    write_json_atomic(
        output_path,
        analysis,
    )

    print_analysis(
        analysis
    )

    print()
    print(
        f"Output: {output_path}"
    )

    return analysis


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the failure-analysis CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Classify production answer-completeness "
                "failures by evidence availability."
            )
        )
    )

    parser.add_argument(
        "--review",
        type=Path,
        default=(
            DEFAULT_REVIEW_PATH
        ),
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
    """Run the deterministic failure analysis."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    run_analysis(
        review_path=(
            args.review
        ),
        output_path=(
            args.output
        ),
    )


if __name__ == "__main__":
    main()