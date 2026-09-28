"""Third consolidated production-quality checkpoint for NepalGov AI.

This checkpoint extends production-quality checkpoint V2 with the three
evaluation milestones completed in the following development cycle:

1. automatic NE->EN query translation retrieval,
2. table-aware Economic Survey passage representation,
3. answer-completeness failure analysis and prompt intervention.

The checkpoint is deterministic over persisted evaluation artifacts.

It performs no:

- hosted generation,
- embedding,
- retrieval,
- reranking,
- OCR,
- ingestion,
- vector backfill,
- Qdrant writes.

None of the experimental interventions in this cycle is promoted to
production. The existing production retrieval stack, production prompt, and
official 30-question production RAG benchmark therefore remain unchanged.
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
    REVIEW_STATUS_COMPLETED,
    aggregate_answer_quality,
)
from src.evaluation.production_quality_checkpoint_v2 import (
    CHECKPOINT_CONFIG_ID as V2_CHECKPOINT_CONFIG_ID,
    build_production_quality_checkpoint_v2,
)
from src.evaluation.review_completeness_prompt_full_benchmark import (
    PAIR_REVIEW_CONFIG_ID,
    candidate_as_standard_review_row,
    load_pair_review_rows,
)


CHECKPOINT_SCHEMA_VERSION = 1

CHECKPOINT_CONFIG_ID = (
    "production-quality-checkpoint-v3"
)

DEFAULT_AUTOMATIC_TRANSLATION_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "ne_en_automatic_translation_retrieval_v1_summary.json"
)

EXPECTED_AUTOMATIC_TRANSLATION_CONFIG_ID = (
    "ne-en-automatic-translation-retrieval-v1"
)

DEFAULT_TABLE_AWARE_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "table_aware_economic_survey_overlay_v1.json"
)

EXPECTED_TABLE_AWARE_CONFIG_ID = (
    "table-aware-economic-survey-overlay-v1"
)

DEFAULT_COMPLETENESS_FAILURE_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_failure_analysis_v1.json"
)

EXPECTED_COMPLETENESS_FAILURE_CONFIG_ID = (
    "production-rag-v2-completeness-failure-analysis-v1"
)

DEFAULT_COMPLETENESS_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_prompt_full_v1_"
    "answer_quality_review.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/"
    "production_quality_checkpoint_v3.json"
)

EXPECTED_QUESTION_COUNT = 30

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)

QUALITY_METRIC_NAMES = (
    "fully_complete_rate",
    "at_least_mostly_complete_rate",
    "fully_faithful_rate",
    "no_major_factual_error_rate",
    "strong_answer_rate",
    "acceptable_answer_rate",
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
            "Evaluation artifact does not exist: "
            f"{source_path}"
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
            f"Invalid JSON artifact: {source_path}"
        ) from exc

    if (
        not isinstance(
            data,
            dict,
        )
        or not data
    ):
        raise ValueError(
            f"{source_path} must contain "
            "one non-empty JSON object."
        )

    return data


def _pair_label(
    query_language: str,
    target_language: str,
) -> str:
    """Return one canonical query->evidence language label."""

    return (
        f"{query_language}"
        f"->{target_language}"
    )


def _group_by_language_pair(
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
    """Group review rows by query and evidence language."""

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
        pair = (
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

        grouped[
            pair
        ].append(
            row
        )

    return dict(
        grouped
    )


def summarize_automatic_translation_retrieval(
    path: str | Path = (
        DEFAULT_AUTOMATIC_TRANSLATION_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the automatic NE->EN translation retrieval experiment."""

    data = (
        _load_json_object(
            path
        )
    )

    if (
        data.get(
            "run_config_id"
        )
        != EXPECTED_AUTOMATIC_TRANSLATION_CONFIG_ID
    ):
        raise ValueError(
            "Unexpected automatic-translation "
            "retrieval run_config_id."
        )

    if (
        data.get(
            "question_count"
        )
        != 6
    ):
        raise ValueError(
            "Automatic-translation retrieval "
            "must contain six questions."
        )

    if (
        data.get(
            "production_changed"
        )
        is not False
    ):
        raise ValueError(
            "Automatic-translation retrieval "
            "must remain evaluation-only."
        )

    automatic_metrics = (
        data.get(
            "automatic_metrics"
        )
    )

    baseline_metrics = (
        data.get(
            "baseline_metrics"
        )
    )

    primary_ranks = (
        data.get(
            "primary_ranks"
        )
    )

    if not isinstance(
        automatic_metrics,
        Mapping,
    ):
        raise ValueError(
            "Automatic-translation artifact "
            "contains no automatic_metrics."
        )

    if not isinstance(
        baseline_metrics,
        Mapping,
    ):
        raise ValueError(
            "Automatic-translation artifact "
            "contains no baseline_metrics."
        )

    if not isinstance(
        primary_ranks,
        Mapping,
    ):
        raise ValueError(
            "Automatic-translation artifact "
            "contains no primary_ranks."
        )

    automatic_bge_10 = (
        automatic_metrics[
            "10"
        ][
            "automatic_bge"
        ]
    )

    production_bge_10 = (
        baseline_metrics[
            "reranked"
        ][
            "10"
        ][
            "production_bge"
        ]
    )

    manual_bge_10 = (
        baseline_metrics[
            "reranked"
        ][
            "10"
        ][
            "translated_bge"
        ]
    )

    hard_case = (
        primary_ranks.get(
            "ne_en_005"
        )
    )

    if not isinstance(
        hard_case,
        Mapping,
    ):
        raise ValueError(
            "Automatic-translation artifact "
            "is missing ne_en_005."
        )

    return {
        "run_config_id": (
            data[
                "run_config_id"
            ]
        ),
        "question_count": 6,
        "production_changed": False,
        "bge_at_10": {
            "production": (
                production_bge_10
            ),
            "manual_control": (
                manual_bge_10
            ),
            "automatic_translation": (
                automatic_bge_10
            ),
        },
        "ne_en_005": {
            "production_bge_rank": (
                hard_case.get(
                    "production_bge"
                )
            ),
            "manual_bge_rank": (
                hard_case.get(
                    "manual_bge"
                )
            ),
            "automatic_bge_rank": (
                hard_case.get(
                    "automatic_bge"
                )
            ),
            "automatic_dense_rank": (
                hard_case.get(
                    "automatic_dense"
                )
            ),
        },
        "interpretation": (
            "Automatic translation did not reproduce the "
            "controlled manual-query gain. At BGE @10 it "
            "matched the production hit rate while improving "
            "recall only modestly, and the key ne_en_005 GDP "
            "passage remained absent from the final candidate "
            "pool. The route was therefore not promoted."
        ),
    }


def summarize_table_aware_overlay(
    path: str | Path = (
        DEFAULT_TABLE_AWARE_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the table-aware Economic Survey overlay experiment."""

    data = (
        _load_json_object(
            path
        )
    )

    if (
        data.get(
            "run_config_id"
        )
        != EXPECTED_TABLE_AWARE_CONFIG_ID
    ):
        raise ValueError(
            "Unexpected table-aware "
            "run_config_id."
        )

    if (
        data.get(
            "question_count"
        )
        != 5
    ):
        raise ValueError(
            "Table-aware experiment must "
            "contain five questions."
        )

    if (
        data.get(
            "production_changed"
        )
        is not False
    ):
        raise ValueError(
            "Table-aware experiment must "
            "remain evaluation-only."
        )

    metrics = (
        data.get(
            "metrics"
        )
    )

    primary_ranks = (
        data.get(
            "primary_ranks"
        )
    )

    simulation = (
        data.get(
            "simulation"
        )
    )

    if not isinstance(
        metrics,
        Mapping,
    ):
        raise ValueError(
            "Table-aware artifact contains "
            "no metrics."
        )

    if not isinstance(
        primary_ranks,
        Mapping,
    ):
        raise ValueError(
            "Table-aware artifact contains "
            "no primary_ranks."
        )

    if not isinstance(
        simulation,
        Mapping,
    ):
        raise ValueError(
            "Table-aware artifact contains "
            "no simulation metadata."
        )

    if (
        simulation.get(
            "qdrant_modified"
        )
        is not False
    ):
        raise ValueError(
            "Table-aware overlay must not "
            "modify Qdrant."
        )

    return {
        "run_config_id": (
            data[
                "run_config_id"
            ]
        ),
        "question_count": 5,
        "english_corpus_point_count": (
            data.get(
                "english_corpus_point_count"
            )
        ),
        "target_document_point_count": (
            data.get(
                "target_document_point_count"
            )
        ),
        "table_like_point_count": (
            data.get(
                "table_like_point_count"
            )
        ),
        "production_changed": False,
        "metrics_at_20": (
            metrics[
                "20"
            ]
        ),
        "primary_ranks": {
            "en_en_005": (
                primary_ranks[
                    "en_en_005"
                ]
            ),
            "en_en_011": (
                primary_ranks[
                    "en_en_011"
                ]
            ),
            "ne_en_005": (
                primary_ranks[
                    "ne_en_005"
                ]
            ),
        },
        "interpretation": (
            "The table-aware overlay increased broad recall "
            "at @20 and substantially improved the English "
            "SEE table case, but overall MRR fell and the "
            "critical cross-language GDP case ne_en_005 "
            "regressed from contextual rank 840 to rank 936. "
            "No representation or Qdrant change was promoted."
        ),
    }


def summarize_completeness_failures(
    path: str | Path = (
        DEFAULT_COMPLETENESS_FAILURE_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the production answer-completeness failure taxonomy."""

    data = (
        _load_json_object(
            path
        )
    )

    if (
        data.get(
            "analysis_config_id"
        )
        != EXPECTED_COMPLETENESS_FAILURE_CONFIG_ID
    ):
        raise ValueError(
            "Unexpected completeness-failure "
            "analysis_config_id."
        )

    if (
        data.get(
            "question_count"
        )
        != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError(
            "Completeness failure analysis must "
            "cover all 30 production questions."
        )

    if (
        data.get(
            "production_changed"
        )
        is not False
    ):
        raise ValueError(
            "Completeness failure analysis "
            "must remain diagnostic."
        )

    causes = (
        data.get(
            "by_failure_cause"
        )
    )

    if not isinstance(
        causes,
        Mapping,
    ):
        raise ValueError(
            "Completeness failure analysis "
            "contains no failure taxonomy."
        )

    expected_non_complete = (
        causes.get(
            "primary_evidence_not_selected",
            0,
        )
        + causes.get(
            "supporting_evidence_not_selected",
            0,
        )
        + causes.get(
            "mixed_context_and_generation_omission",
            0,
        )
        + causes.get(
            "generation_omission_despite_full_gold_context",
            0,
        )
        + causes.get(
            "needs_review",
            0,
        )
    )

    if (
        expected_non_complete
        != data.get(
            "non_complete_count"
        )
    ):
        raise ValueError(
            "Completeness failure counts "
            "do not sum to non_complete_count."
        )

    pure_context_limited = (
        causes[
            "primary_evidence_not_selected"
        ]
        + causes[
            "supporting_evidence_not_selected"
        ]
    )

    retrieval_or_context_ids = (
        data.get(
            "retrieval_or_context_intervention_question_ids"
        )
    )

    if not isinstance(
        retrieval_or_context_ids,
        list,
    ):
        raise ValueError(
            "Completeness analysis contains "
            "no retrieval/context candidate list."
        )

    return {
        "analysis_config_id": (
            data[
                "analysis_config_id"
            ]
        ),
        "question_count": (
            data[
                "question_count"
            ]
        ),
        "fully_complete_count": (
            data[
                "fully_complete_count"
            ]
        ),
        "non_complete_count": (
            data[
                "non_complete_count"
            ]
        ),
        "by_failure_cause": (
            dict(
                causes
            )
        ),
        "pure_context_limited_count": (
            pure_context_limited
        ),
        "retrieval_or_context_candidate_count": (
            len(
                retrieval_or_context_ids
            )
        ),
        "clean_generation_intervention_count": (
            len(
                data.get(
                    "clean_generation_intervention_question_ids",
                    [],
                )
            )
        ),
        "interpretation": (
            "Seven of nine non-complete production answers "
            "are pure retrieval/context-coverage failures, "
            "one is mixed context plus generation, and only "
            "one cleanly isolates generation behavior. "
            "Upstream evidence coverage is therefore the "
            "dominant remaining answer-completeness limit."
        ),
    }


def _baseline_standard_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """Adapt stored baseline labels to the established quality metric schema."""

    standard = (
        candidate_as_standard_review_row(
            row
        )
    )

    standard[
        "completeness_label"
    ] = (
        row[
            "baseline_completeness_label"
        ]
    )

    standard[
        "factual_fidelity_label"
    ] = (
        row[
            "baseline_factual_fidelity_label"
        ]
    )

    standard[
        "review_notes"
    ] = (
        row.get(
            "baseline_review_notes"
        )
    )

    return standard


def summarize_completeness_prompt_review(
    path: str | Path = (
        DEFAULT_COMPLETENESS_REVIEW_PATH
    ),
) -> dict[
    str,
    Any,
]:
    """Summarize the full human review of the completeness prompt."""

    rows = (
        load_pair_review_rows(
            path
        )
    )

    if (
        len(
            rows
        )
        != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError(
            "Completeness-prompt review must "
            "contain 30 questions."
        )

    if any(
        row.get(
            "review_status"
        )
        != REVIEW_STATUS_COMPLETED
        for row in rows
    ):
        raise ValueError(
            "Completeness-prompt human "
            "review is incomplete."
        )

    if any(
        row.get(
            "review_config_id"
        )
        != PAIR_REVIEW_CONFIG_ID
        for row in rows
    ):
        raise ValueError(
            "Unexpected completeness-prompt "
            "review_config_id."
        )

    candidate_rows = [
        candidate_as_standard_review_row(
            row
        )
        for row in rows
    ]

    baseline_rows = [
        _baseline_standard_review_row(
            row
        )
        for row in rows
    ]

    baseline_overall = (
        aggregate_answer_quality(
            baseline_rows
        )
    )

    candidate_overall = (
        aggregate_answer_quality(
            candidate_rows
        )
    )

    baseline_grouped = (
        _group_by_language_pair(
            baseline_rows
        )
    )

    candidate_grouped = (
        _group_by_language_pair(
            candidate_rows
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
        baseline_pair = (
            baseline_grouped.get(
                pair
            )
        )

        candidate_pair = (
            candidate_grouped.get(
                pair
            )
        )

        if (
            not baseline_pair
            or not candidate_pair
        ):
            continue

        baseline_metrics = (
            aggregate_answer_quality(
                baseline_pair
            )
        )

        candidate_metrics = (
            aggregate_answer_quality(
                candidate_pair
            )
        )

        by_language_pair[
            _pair_label(
                *pair
            )
        ] = {
            "baseline": (
                baseline_metrics
            ),
            "candidate": (
                candidate_metrics
            ),
            "fully_complete_delta": (
                candidate_metrics[
                    "fully_complete_rate"
                ]
                - baseline_metrics[
                    "fully_complete_rate"
                ]
            ),
            "fully_faithful_delta": (
                candidate_metrics[
                    "fully_faithful_rate"
                ]
                - baseline_metrics[
                    "fully_faithful_rate"
                ]
            ),
        }

    deltas = {
        metric_name: (
            candidate_overall[
                metric_name
            ]
            - baseline_overall[
                metric_name
            ]
        )
        for metric_name in (
            QUALITY_METRIC_NAMES
        )
    }

    return {
        "review_config_id": (
            PAIR_REVIEW_CONFIG_ID
        ),
        "question_count": (
            len(
                rows
            )
        ),
        "baseline": (
            baseline_overall
        ),
        "candidate": (
            candidate_overall
        ),
        "delta": (
            deltas
        ),
        "by_language_pair": (
            by_language_pair
        ),
        "production_changed": False,
        "interpretation": (
            "The completeness prompt improved fully complete, "
            "at-least-mostly-complete, strong, and acceptable "
            "answer rates by one question out of 30 (+0.033). "
            "Overall fully-faithful performance remained 0.933 "
            "and the no-major-error rate remained 1.000, but "
            "the en->ne fully-faithful slice declined from "
            "1.000 to 0.833. The measured gain is too small "
            "and uneven to justify production promotion."
        ),
    }


def build_production_quality_checkpoint_v3(
) -> dict[
    str,
    Any,
]:
    """Build the deterministic third production-quality checkpoint."""

    base_checkpoint = (
        build_production_quality_checkpoint_v2()
    )

    automatic_translation = (
        summarize_automatic_translation_retrieval()
    )

    table_aware = (
        summarize_table_aware_overlay()
    )

    completeness_failures = (
        summarize_completeness_failures()
    )

    completeness_prompt = (
        summarize_completeness_prompt_review()
    )

    automatic_10 = (
        automatic_translation[
            "bge_at_10"
        ]
    )

    table_20 = (
        table_aware[
            "metrics_at_20"
        ]
    )

    baseline_quality = (
        completeness_prompt[
            "baseline"
        ]
    )

    candidate_quality = (
        completeness_prompt[
            "candidate"
        ]
    )

    return {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "checkpoint_config_id": (
            CHECKPOINT_CONFIG_ID
        ),
        "base_checkpoint_config_id": (
            V2_CHECKPOINT_CONFIG_ID
        ),
        "base_production_quality": (
            base_checkpoint
        ),
        "cycle_extensions": {
            "automatic_ne_en_translation": (
                automatic_translation
            ),
            "table_aware_passage_representation": (
                table_aware
            ),
            "answer_completeness_failure_analysis": (
                completeness_failures
            ),
            "answer_completeness_prompt": (
                completeness_prompt
            ),
        },
        "headline": {
            "automatic_translation_bge_hit_at_10": (
                automatic_10[
                    "automatic_translation"
                ][
                    "hit_rate"
                ]
            ),
            "automatic_translation_bge_recall_at_10": (
                automatic_10[
                    "automatic_translation"
                ][
                    "recall"
                ]
            ),
            "manual_control_bge_hit_at_10": (
                automatic_10[
                    "manual_control"
                ][
                    "hit_rate"
                ]
            ),
            "manual_control_bge_recall_at_10": (
                automatic_10[
                    "manual_control"
                ][
                    "recall"
                ]
            ),
            "table_contextual_hit_at_20": (
                table_20[
                    "contextual"
                ][
                    "hit_rate"
                ]
            ),
            "table_overlay_hit_at_20": (
                table_20[
                    "table_aware_overlay"
                ][
                    "hit_rate"
                ]
            ),
            "table_contextual_recall_at_20": (
                table_20[
                    "contextual"
                ][
                    "recall"
                ]
            ),
            "table_overlay_recall_at_20": (
                table_20[
                    "table_aware_overlay"
                ][
                    "recall"
                ]
            ),
            "table_contextual_mrr_at_20": (
                table_20[
                    "contextual"
                ][
                    "mrr"
                ]
            ),
            "table_overlay_mrr_at_20": (
                table_20[
                    "table_aware_overlay"
                ][
                    "mrr"
                ]
            ),
            "non_complete_production_answer_count": (
                completeness_failures[
                    "non_complete_count"
                ]
            ),
            "pure_context_limited_answer_count": (
                completeness_failures[
                    "pure_context_limited_count"
                ]
            ),
            "baseline_fully_complete_rate": (
                baseline_quality[
                    "fully_complete_rate"
                ]
            ),
            "candidate_fully_complete_rate": (
                candidate_quality[
                    "fully_complete_rate"
                ]
            ),
            "baseline_at_least_mostly_complete_rate": (
                baseline_quality[
                    "at_least_mostly_complete_rate"
                ]
            ),
            "candidate_at_least_mostly_complete_rate": (
                candidate_quality[
                    "at_least_mostly_complete_rate"
                ]
            ),
            "baseline_fully_faithful_rate": (
                baseline_quality[
                    "fully_faithful_rate"
                ]
            ),
            "candidate_fully_faithful_rate": (
                candidate_quality[
                    "fully_faithful_rate"
                ]
            ),
            "baseline_no_major_factual_error_rate": (
                baseline_quality[
                    "no_major_factual_error_rate"
                ]
            ),
            "candidate_no_major_factual_error_rate": (
                candidate_quality[
                    "no_major_factual_error_rate"
                ]
            ),
        },
        "production_decision": {
            "production_retrieval_changed": False,
            "production_prompt_changed": False,
            "qdrant_reindexed": False,
            "hosted_production_rag_rerun_performed": False,
            "official_production_benchmark_retained": True,
            "automatic_translation_promoted": False,
            "table_aware_representation_promoted": False,
            "completeness_prompt_promoted": False,
            "reason": (
                "None of the three measured interventions "
                "demonstrated a sufficiently robust production "
                "improvement. Automatic NE->EN translation did "
                "not recover the key GDP retrieval failure; the "
                "table-aware overlay improved recall for selected "
                "English table queries but reduced MRR and "
                "regressed the cross-language GDP case; and the "
                "completeness prompt improved overall completeness "
                "by only one answer out of 30 while producing an "
                "uneven language-slice fidelity result. The "
                "existing production retrieval and prompt are "
                "therefore retained."
            ),
        },
        "next_phase": {
            "recommended": (
                "application_api_readiness"
            ),
            "reason": (
                "The core RAG architecture now has retrieval, "
                "generation, citation, insufficient-evidence, "
                "cross-language, hard-case, and human answer-"
                "quality evaluation coverage. Further retrieval "
                "research can continue later, but the measured "
                "results support moving the evaluated production "
                "stack into an application/API layer without "
                "promoting the current experimental variants."
            ),
        },
        "measured_remaining_work": [
            (
                "Improve retrieval/context coverage for the "
                "remaining incomplete answers: seven of nine "
                "non-complete production answers are pure "
                "context-coverage failures and one is mixed."
            ),
            (
                "Preserve the official production prompt until "
                "a broader prompt intervention demonstrates a "
                "larger completeness gain without language-slice "
                "fidelity regression."
            ),
            (
                "Build the application/API layer around the "
                "currently retained production RAG pipeline."
            ),
            (
                "Add application-level observability, error "
                "handling, health checks, configuration, and "
                "end-to-end tests."
            ),
            (
                "Expand the official government-document corpus "
                "after the application path is stable, then rerun "
                "retrieval and answer-quality evaluation."
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
    """Write the V3 checkpoint atomically."""

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
            dict(
                checkpoint
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
        path
    )


def print_checkpoint_summary(
    checkpoint: Mapping[
        str,
        Any,
    ],
) -> None:
    """Print the V3 decision headline."""

    headline = (
        checkpoint[
            "headline"
        ]
    )

    print()
    print(
        "=" * 88
    )

    print(
        "NepalGov AI production-quality checkpoint V3"
    )

    print(
        "=" * 88
    )

    print(
        "Automatic NE->EN BGE @10:"
    )

    print(
        "  Automatic hit/recall: "
        f"{headline['automatic_translation_bge_hit_at_10']:.3f} / "
        f"{headline['automatic_translation_bge_recall_at_10']:.3f}"
    )

    print(
        "  Manual-control hit/recall: "
        f"{headline['manual_control_bge_hit_at_10']:.3f} / "
        f"{headline['manual_control_bge_recall_at_10']:.3f}"
    )

    print()

    print(
        "Table-aware overlay @20:"
    )

    print(
        "  Contextual hit/recall/MRR: "
        f"{headline['table_contextual_hit_at_20']:.3f} / "
        f"{headline['table_contextual_recall_at_20']:.3f} / "
        f"{headline['table_contextual_mrr_at_20']:.3f}"
    )

    print(
        "  Overlay hit/recall/MRR: "
        f"{headline['table_overlay_hit_at_20']:.3f} / "
        f"{headline['table_overlay_recall_at_20']:.3f} / "
        f"{headline['table_overlay_mrr_at_20']:.3f}"
    )

    print()

    print(
        "Production completeness failures:"
    )

    print(
        "  Non-complete answers: "
        f"{headline['non_complete_production_answer_count']}"
    )

    print(
        "  Pure context-limited: "
        f"{headline['pure_context_limited_answer_count']}"
    )

    print()

    print(
        "Completeness prompt human review:"
    )

    print(
        "  Fully complete: "
        f"{headline['baseline_fully_complete_rate']:.3f}"
        " -> "
        f"{headline['candidate_fully_complete_rate']:.3f}"
    )

    print(
        "  At least mostly complete: "
        f"{headline['baseline_at_least_mostly_complete_rate']:.3f}"
        " -> "
        f"{headline['candidate_at_least_mostly_complete_rate']:.3f}"
    )

    print(
        "  Fully faithful: "
        f"{headline['baseline_fully_faithful_rate']:.3f}"
        " -> "
        f"{headline['candidate_fully_faithful_rate']:.3f}"
    )

    print(
        "  No major factual error: "
        f"{headline['baseline_no_major_factual_error_rate']:.3f}"
        " -> "
        f"{headline['candidate_no_major_factual_error_rate']:.3f}"
    )

    print()

    decision = (
        checkpoint[
            "production_decision"
        ]
    )

    print(
        "Production retrieval changed: "
        f"{decision['production_retrieval_changed']}"
    )

    print(
        "Production prompt changed: "
        f"{decision['production_prompt_changed']}"
    )

    print(
        "Experimental variants promoted: False"
    )

    print(
        "Official production benchmark retained: "
        f"{decision['official_production_benchmark_retained']}"
    )

    print()

    print(
        "Next phase: "
        f"{checkpoint['next_phase']['recommended']}"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the V3 checkpoint CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Build the third deterministic "
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
    """Build, persist, and print checkpoint V3."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    checkpoint = (
        build_production_quality_checkpoint_v3()
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