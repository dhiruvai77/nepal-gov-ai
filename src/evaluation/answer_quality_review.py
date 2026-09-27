"""Human response-level completeness and factual-fidelity review.

This evaluation reviews the complete 30-question production RAG benchmark.

It keeps two answer-quality dimensions separate:

Completeness
    Whether the application-visible answer covers the material information in
    the benchmark's manually verified relevant evidence. Primary evidence
    represents the central answer target; the broader relevant set captures
    additional verified answer material.

Factual fidelity
    Whether factual statements in the visible answer are consistent with the
    exact source passages available for review. This includes selected RAG
    evidence and the manually verified gold-relevant evidence.

The review is deliberately response-level. Existing claim-level semantic
evaluation continues to measure claim/citation support separately.

No generation, embedding, retrieval, or reranking calls are made. Qdrant is
used only when building the review artifact so exact original passage text can
be attached to the benchmark rows.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from src.evaluation.build_semantic_evaluation_dataset import (
    fetch_qdrant_payloads,
    load_rag_run_rows,
    validate_evidence_payload,
    write_jsonl_atomic,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    QDRANT_URL,
)


REVIEW_SCHEMA_VERSION = 1

REVIEW_CONFIG_ID = (
    "production-rag-v2-answer-quality-v1"
)

DEFAULT_RAG_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_interactions.jsonl"
)

DEFAULT_BENCHMARK_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_answer_quality_review_v1.jsonl"
)

EXPECTED_SOURCE_RUN_CONFIG_ID = (
    "production-rag-v2-interactions"
)

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_COMPLETED = "completed"

SUPPORTED_REVIEW_STATUSES = (
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_COMPLETED,
)

COMPLETENESS_LABELS = (
    "complete",
    "mostly_complete",
    "incomplete",
    "needs_review",
)

FIDELITY_LABELS = (
    "fully_faithful",
    "minor_issue",
    "major_issue",
    "needs_review",
)

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)

COMPLETENESS_CHOICES = {
    "1": "complete",
    "2": "mostly_complete",
    "3": "incomplete",
    "4": "needs_review",
}

FIDELITY_CHOICES = {
    "1": "fully_faithful",
    "2": "minor_issue",
    "3": "major_issue",
    "4": "needs_review",
}


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Return one validated non-empty string."""

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must contain "
            "non-whitespace text."
        )

    return value


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


def _validate_rag_record_alignment(
    rag_row: Mapping[
        str,
        Any,
    ],
    record: EvaluationRecord,
) -> None:
    """Ensure the persisted RAG row still matches its benchmark definition."""

    question_id = (
        record.question_id
    )

    if (
        rag_row.get(
            "question_id"
        )
        != question_id
    ):
        raise ValueError(
            "RAG row question_id does not "
            f"match benchmark {question_id!r}."
        )

    comparisons = (
        (
            "query",
            rag_row.get(
                "query"
            ),
            record.query,
        ),
        (
            "query_language",
            rag_row.get(
                "query_language"
            ),
            record.query_language,
        ),
        (
            "target_language",
            rag_row.get(
                "target_language"
            ),
            record.target_language,
        ),
        (
            "category",
            rag_row.get(
                "category"
            ),
            record.category,
        ),
        (
            "expected_document_ids",
            tuple(
                rag_row.get(
                    "expected_document_ids",
                    [],
                )
            ),
            tuple(
                record.expected_document_ids
            ),
        ),
        (
            "primary_relevant_chunk_ids",
            tuple(
                rag_row.get(
                    "primary_relevant_chunk_ids",
                    [],
                )
            ),
            tuple(
                record.primary_relevant_chunk_ids
            ),
        ),
        (
            "relevant_chunk_ids",
            tuple(
                rag_row.get(
                    "relevant_chunk_ids",
                    [],
                )
            ),
            tuple(
                record.relevant_chunk_ids
            ),
        ),
    )

    for (
        field_name,
        persisted_value,
        benchmark_value,
    ) in comparisons:
        if (
            persisted_value
            != benchmark_value
        ):
            raise ValueError(
                f"{question_id}: persisted RAG field "
                f"{field_name!r} does not match "
                "the retrieval benchmark."
            )

    if (
        rag_row.get(
            "run_config_id"
        )
        != EXPECTED_SOURCE_RUN_CONFIG_ID
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "source run_config_id."
        )


def collect_required_point_ids(
    rag_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    records_by_id: Mapping[
        str,
        EvaluationRecord,
    ],
) -> tuple[str, ...]:
    """Collect selected and gold-relevant point IDs in stable order."""

    point_ids: list[str] = []

    for rag_row in rag_rows:
        question_id = (
            _require_nonempty_string(
                rag_row.get(
                    "question_id"
                ),
                field_name=(
                    "question_id"
                ),
            )
        )

        record = (
            records_by_id.get(
                question_id
            )
        )

        if record is None:
            raise ValueError(
                "RAG benchmark contains "
                f"unknown question_id "
                f"{question_id!r}."
            )

        selected_evidence = (
            rag_row.get(
                "selected_evidence"
            )
        )

        if not isinstance(
            selected_evidence,
            list,
        ):
            raise ValueError(
                f"{question_id}: selected_evidence "
                "must be a list."
            )

        for evidence in selected_evidence:
            if not isinstance(
                evidence,
                Mapping,
            ):
                raise ValueError(
                    f"{question_id}: selected evidence "
                    "entries must be mappings."
                )

            point_ids.append(
                _require_nonempty_string(
                    evidence.get(
                        "point_id"
                    ),
                    field_name=(
                        "selected evidence point_id"
                    ),
                )
            )

        point_ids.extend(
            record.relevant_chunk_ids
        )

    return tuple(
        _ordered_unique(
            point_ids
        )
    )


def _payload_evidence(
    *,
    point_id: str,
    payload: Mapping[
        str,
        Any,
    ],
    evidence_id: str | None,
    is_primary: bool,
    is_gold_relevant: bool,
) -> dict[
    str,
    Any,
]:
    """Build one review evidence object from an exact Qdrant payload."""

    chunk_text = (
        payload.get(
            "chunk_text"
        )
    )

    if not isinstance(
        chunk_text,
        str,
    ) or not chunk_text.strip():
        raise ValueError(
            f"Qdrant point {point_id!r} "
            "has no usable chunk_text."
        )

    return {
        "evidence_id": (
            evidence_id
        ),
        "point_id": (
            point_id
        ),
        "chunk_id": (
            payload.get(
                "chunk_id"
            )
        ),
        "document_id": (
            payload.get(
                "document_id"
            )
        ),
        "title": (
            payload.get(
                "title"
            )
        ),
        "organization": (
            payload.get(
                "organization"
            )
        ),
        "language": (
            payload.get(
                "language"
            )
        ),
        "page_start": (
            payload.get(
                "page_start"
            )
        ),
        "page_end": (
            payload.get(
                "page_end"
            )
        ),
        "source_url": (
            payload.get(
                "source_url"
            )
        ),
        "chunk_text": (
            chunk_text
        ),
        "is_primary": (
            is_primary
        ),
        "is_gold_relevant": (
            is_gold_relevant
        ),
    }


def build_review_row(
    rag_row: Mapping[
        str,
        Any,
    ],
    record: EvaluationRecord,
    *,
    payloads_by_point_id: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> dict[
    str,
    Any,
]:
    """Build one answer-quality review row."""

    _validate_rag_record_alignment(
        rag_row,
        record,
    )

    question_id = (
        record.question_id
    )

    answer_text = (
        rag_row.get(
            "answer_text"
        )
    )

    if not isinstance(
        answer_text,
        str,
    ):
        raise ValueError(
            f"{question_id}: answer_text "
            "must be a string."
        )

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    relevant_ids = set(
        record.relevant_chunk_ids
    )

    selected_evidence = (
        rag_row.get(
            "selected_evidence"
        )
    )

    if not isinstance(
        selected_evidence,
        list,
    ):
        raise ValueError(
            f"{question_id}: selected_evidence "
            "must be a list."
        )

    selected_review_evidence: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for persisted in selected_evidence:
        if not isinstance(
            persisted,
            Mapping,
        ):
            raise ValueError(
                f"{question_id}: selected evidence "
                "entry must be a mapping."
            )

        point_id = (
            _require_nonempty_string(
                persisted.get(
                    "point_id"
                ),
                field_name=(
                    "selected evidence point_id"
                ),
            )
        )

        payload = (
            payloads_by_point_id.get(
                point_id
            )
        )

        if payload is None:
            raise ValueError(
                f"{question_id}: missing Qdrant "
                f"payload for {point_id!r}."
            )

        validate_evidence_payload(
            persisted,
            payload,
        )

        evidence_id = (
            _require_nonempty_string(
                persisted.get(
                    "evidence_id"
                ),
                field_name=(
                    "evidence_id"
                ),
            )
        )

        selected_review_evidence.append(
            _payload_evidence(
                point_id=point_id,
                payload=payload,
                evidence_id=(
                    evidence_id
                ),
                is_primary=(
                    point_id
                    in primary_ids
                ),
                is_gold_relevant=(
                    point_id
                    in relevant_ids
                ),
            )
        )

    gold_reference_evidence: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for point_id in (
        record.relevant_chunk_ids
    ):
        payload = (
            payloads_by_point_id.get(
                point_id
            )
        )

        if payload is None:
            raise ValueError(
                f"{question_id}: missing gold "
                f"Qdrant payload for "
                f"{point_id!r}."
            )

        gold_reference_evidence.append(
            _payload_evidence(
                point_id=point_id,
                payload=payload,
                evidence_id=None,
                is_primary=(
                    point_id
                    in primary_ids
                ),
                is_gold_relevant=True,
            )
        )

    return {
        "review_schema_version": (
            REVIEW_SCHEMA_VERSION
        ),
        "review_config_id": (
            REVIEW_CONFIG_ID
        ),
        "source_run_config_id": (
            rag_row[
                "run_config_id"
            ]
        ),
        "question_id": (
            question_id
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
            rag_row.get(
                "answer_language"
            )
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
        "answer_text": (
            answer_text
        ),
        "accepted": bool(
            rag_row.get(
                "accepted"
            )
        ),
        "withheld": bool(
            rag_row.get(
                "withheld"
            )
        ),
        "selected_primary_hit": (
            rag_row.get(
                "metrics",
                {},
            ).get(
                "selected_primary_hit"
            )
            if isinstance(
                rag_row.get(
                    "metrics"
                ),
                Mapping,
            )
            else None
        ),
        "selected_relevant_recall": (
            rag_row.get(
                "metrics",
                {},
            ).get(
                "selected_relevant_recall"
            )
            if isinstance(
                rag_row.get(
                    "metrics"
                ),
                Mapping,
            )
            else None
        ),
        "selected_evidence": (
            selected_review_evidence
        ),
        "gold_reference_evidence": (
            gold_reference_evidence
        ),
        "completeness_label": None,
        "factual_fidelity_label": None,
        "review_notes": None,
        "review_status": (
            REVIEW_STATUS_PENDING
        ),
    }


def validate_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Validate one pending or completed answer-quality review row."""

    question_id = (
        _require_nonempty_string(
            row.get(
                "question_id"
            ),
            field_name=(
                "question_id"
            ),
        )
    )

    if (
        row.get(
            "review_schema_version"
        )
        != REVIEW_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "review_schema_version."
        )

    if (
        row.get(
            "review_config_id"
        )
        != REVIEW_CONFIG_ID
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "review_config_id."
        )

    status = (
        row.get(
            "review_status"
        )
    )

    if (
        status
        not in SUPPORTED_REVIEW_STATUSES
    ):
        raise ValueError(
            f"{question_id}: unsupported "
            f"review_status {status!r}."
        )

    gold_evidence = (
        row.get(
            "gold_reference_evidence"
        )
    )

    if (
        not isinstance(
            gold_evidence,
            list,
        )
        or not gold_evidence
    ):
        raise ValueError(
            f"{question_id}: gold_reference_evidence "
            "must be a non-empty list."
        )

    relevant_ids = (
        row.get(
            "relevant_chunk_ids"
        )
    )

    primary_ids = (
        row.get(
            "primary_relevant_chunk_ids"
        )
    )

    if not isinstance(
        relevant_ids,
        list,
    ):
        raise ValueError(
            f"{question_id}: relevant_chunk_ids "
            "must be a list."
        )

    if not isinstance(
        primary_ids,
        list,
    ):
        raise ValueError(
            f"{question_id}: primary_relevant_chunk_ids "
            "must be a list."
        )

    gold_ids = [
        evidence.get(
            "point_id"
        )
        for evidence
        in gold_evidence
        if isinstance(
            evidence,
            Mapping,
        )
    ]

    if (
        gold_ids
        != relevant_ids
    ):
        raise ValueError(
            f"{question_id}: gold evidence order "
            "does not match relevant_chunk_ids."
        )

    if (
        not set(
            primary_ids
        ).issubset(
            set(
                relevant_ids
            )
        )
    ):
        raise ValueError(
            f"{question_id}: primary evidence "
            "must be part of relevant evidence."
        )

    selected_evidence = (
        row.get(
            "selected_evidence"
        )
    )

    if not isinstance(
        selected_evidence,
        list,
    ):
        raise ValueError(
            f"{question_id}: selected_evidence "
            "must be a list."
        )

    selected_ids = [
        evidence.get(
            "point_id"
        )
        for evidence
        in selected_evidence
        if isinstance(
            evidence,
            Mapping,
        )
    ]

    if (
        len(
            selected_ids
        )
        != len(
            set(
                selected_ids
            )
        )
    ):
        raise ValueError(
            f"{question_id}: duplicate "
            "selected evidence point IDs."
        )

    completeness = (
        row.get(
            "completeness_label"
        )
    )

    fidelity = (
        row.get(
            "factual_fidelity_label"
        )
    )

    if (
        status
        == REVIEW_STATUS_PENDING
    ):
        if (
            completeness is not None
            or fidelity is not None
        ):
            raise ValueError(
                f"{question_id}: pending row "
                "cannot contain review labels."
            )

        return

    if (
        completeness
        not in COMPLETENESS_LABELS
    ):
        raise ValueError(
            f"{question_id}: unsupported "
            "completeness label "
            f"{completeness!r}."
        )

    if (
        fidelity
        not in FIDELITY_LABELS
    ):
        raise ValueError(
            f"{question_id}: unsupported "
            "factual-fidelity label "
            f"{fidelity!r}."
        )


def apply_review_labels(
    row: Mapping[
        str,
        Any,
    ],
    *,
    completeness_label: str,
    factual_fidelity_label: str,
    review_notes: str | None = None,
) -> dict[
    str,
    Any,
]:
    """Return one completed answer-quality review row."""

    if (
        completeness_label
        not in COMPLETENESS_LABELS
    ):
        raise ValueError(
            "Unsupported completeness label "
            f"{completeness_label!r}."
        )

    if (
        factual_fidelity_label
        not in FIDELITY_LABELS
    ):
        raise ValueError(
            "Unsupported factual-fidelity label "
            f"{factual_fidelity_label!r}."
        )

    completed = (
        copy.deepcopy(
            dict(
                row
            )
        )
    )

    completed[
        "completeness_label"
    ] = (
        completeness_label
    )

    completed[
        "factual_fidelity_label"
    ] = (
        factual_fidelity_label
    )

    completed[
        "review_notes"
    ] = (
        review_notes.strip()
        if (
            isinstance(
                review_notes,
                str,
            )
            and review_notes.strip()
        )
        else None
    )

    completed[
        "review_status"
    ] = (
        REVIEW_STATUS_COMPLETED
    )

    validate_review_row(
        completed
    )

    return completed


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """Return zero when a metric denominator is empty."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def aggregate_answer_quality(
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
    """Aggregate human completeness and factual-fidelity metrics."""

    if not rows:
        raise ValueError(
            "At least one answer-quality "
            "review row is required."
        )

    for row in rows:
        validate_review_row(
            row
        )

    reviewed = [
        row
        for row in rows
        if (
            row[
                "review_status"
            ]
            == REVIEW_STATUS_COMPLETED
        )
    ]

    scored = [
        row
        for row in reviewed
        if (
            row[
                "completeness_label"
            ]
            != "needs_review"
            and row[
                "factual_fidelity_label"
            ]
            != "needs_review"
        )
    ]

    complete_count = sum(
        row[
            "completeness_label"
        ]
        == "complete"
        for row in scored
    )

    mostly_complete_count = sum(
        row[
            "completeness_label"
        ]
        == "mostly_complete"
        for row in scored
    )

    incomplete_count = sum(
        row[
            "completeness_label"
        ]
        == "incomplete"
        for row in scored
    )

    fully_faithful_count = sum(
        row[
            "factual_fidelity_label"
        ]
        == "fully_faithful"
        for row in scored
    )

    minor_issue_count = sum(
        row[
            "factual_fidelity_label"
        ]
        == "minor_issue"
        for row in scored
    )

    major_issue_count = sum(
        row[
            "factual_fidelity_label"
        ]
        == "major_issue"
        for row in scored
    )

    strong_answer_count = sum(
        (
            row[
                "completeness_label"
            ]
            == "complete"
            and row[
                "factual_fidelity_label"
            ]
            == "fully_faithful"
        )
        for row in scored
    )

    acceptable_answer_count = sum(
        (
            row[
                "completeness_label"
            ]
            in {
                "complete",
                "mostly_complete",
            }
            and row[
                "factual_fidelity_label"
            ]
            in {
                "fully_faithful",
                "minor_issue",
            }
        )
        for row in scored
    )

    needs_review_count = sum(
        (
            row[
                "completeness_label"
            ]
            == "needs_review"
            or row[
                "factual_fidelity_label"
            ]
            == "needs_review"
        )
        for row in reviewed
    )

    return {
        "question_count": len(
            rows
        ),
        "reviewed_count": len(
            reviewed
        ),
        "scored_count": len(
            scored
        ),
        "completion_rate": (
            _safe_ratio(
                len(
                    reviewed
                ),
                len(
                    rows
                ),
            )
        ),
        "scorable_rate": (
            _safe_ratio(
                len(
                    scored
                ),
                len(
                    reviewed
                ),
            )
        ),
        "needs_review_count": (
            needs_review_count
        ),
        "complete_count": (
            complete_count
        ),
        "mostly_complete_count": (
            mostly_complete_count
        ),
        "incomplete_count": (
            incomplete_count
        ),
        "fully_complete_rate": (
            _safe_ratio(
                complete_count,
                len(
                    scored
                ),
            )
        ),
        "at_least_mostly_complete_rate": (
            _safe_ratio(
                (
                    complete_count
                    + mostly_complete_count
                ),
                len(
                    scored
                ),
            )
        ),
        "incomplete_rate": (
            _safe_ratio(
                incomplete_count,
                len(
                    scored
                ),
            )
        ),
        "fully_faithful_count": (
            fully_faithful_count
        ),
        "minor_factual_issue_count": (
            minor_issue_count
        ),
        "major_factual_issue_count": (
            major_issue_count
        ),
        "fully_faithful_rate": (
            _safe_ratio(
                fully_faithful_count,
                len(
                    scored
                ),
            )
        ),
        "no_major_factual_error_rate": (
            _safe_ratio(
                (
                    fully_faithful_count
                    + minor_issue_count
                ),
                len(
                    scored
                ),
            )
        ),
        "major_factual_issue_rate": (
            _safe_ratio(
                major_issue_count,
                len(
                    scored
                ),
            )
        ),
        "strong_answer_count": (
            strong_answer_count
        ),
        "strong_answer_rate": (
            _safe_ratio(
                strong_answer_count,
                len(
                    scored
                ),
            )
        ),
        "acceptable_answer_count": (
            acceptable_answer_count
        ),
        "acceptable_answer_rate": (
            _safe_ratio(
                acceptable_answer_count,
                len(
                    scored
                ),
            )
        ),
    }


def load_review_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load and validate one answer-quality review artifact."""

    input_path = Path(
        path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "Answer-quality review file "
            f"does not exist: {input_path}"
        )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen_question_ids: set[
        str
    ] = set()

    with input_path.open(
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
                    "Invalid answer-quality JSON "
                    f"on line {line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Answer-quality review row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            validate_review_row(
                row
            )

            question_id = (
                row[
                    "question_id"
                ]
            )

            if (
                question_id
                in seen_question_ids
            ):
                raise ValueError(
                    "Duplicate answer-quality "
                    f"question_id "
                    f"{question_id!r}."
                )

            seen_question_ids.add(
                question_id
            )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            "Answer-quality review "
            "contains no rows."
        )

    return rows


def build_review_dataset(
    *,
    rag_path: str | Path = (
        DEFAULT_RAG_PATH
    ),
    benchmark_path: str | Path = (
        DEFAULT_BENCHMARK_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    client: Any | None = None,
    collection_name: str = (
        COLLECTION_NAME
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Build the complete 30-question pending human-review dataset."""

    destination = Path(
        output_path
    )

    if destination.exists():
        raise FileExistsError(
            "Answer-quality output already "
            f"exists: {destination}. "
            "Refusing to overwrite human "
            "review progress."
        )

    rag_rows = (
        load_rag_run_rows(
            rag_path
        )
    )

    records = (
        load_evaluation_records(
            benchmark_path
        )
    )

    records_by_id = {
        record.question_id: record
        for record in records
    }

    rag_ids = [
        _require_nonempty_string(
            row.get(
                "question_id"
            ),
            field_name=(
                "question_id"
            ),
        )
        for row in rag_rows
    ]

    benchmark_ids = [
        record.question_id
        for record in records
    ]

    if (
        set(
            rag_ids
        )
        != set(
            benchmark_ids
        )
    ):
        missing_from_rag = sorted(
            set(
                benchmark_ids
            )
            - set(
                rag_ids
            )
        )

        unexpected_in_rag = sorted(
            set(
                rag_ids
            )
            - set(
                benchmark_ids
            )
        )

        raise ValueError(
            "Production RAG and retrieval "
            "benchmark question IDs differ. "
            f"Missing from RAG={missing_from_rag}; "
            f"unexpected in RAG={unexpected_in_rag}."
        )

    point_ids = (
        collect_required_point_ids(
            rag_rows,
            records_by_id,
        )
    )

    owns_client = (
        client is None
    )

    active_client = (
        client
        if client is not None
        else QdrantClient(
            url=QDRANT_URL
        )
    )

    try:
        payloads = (
            fetch_qdrant_payloads(
                active_client,
                point_ids,
                collection_name=(
                    collection_name
                ),
            )
        )

    finally:
        if owns_client:
            close_method = getattr(
                active_client,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                close_method()

    review_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for rag_row in rag_rows:
        question_id = (
            str(
                rag_row[
                    "question_id"
                ]
            )
        )

        record = (
            records_by_id[
                question_id
            ]
        )

        row = (
            build_review_row(
                rag_row,
                record,
                payloads_by_point_id=(
                    payloads
                ),
            )
        )

        validate_review_row(
            row
        )

        review_rows.append(
            row
        )

    write_jsonl_atomic(
        destination,
        review_rows,
    )

    return review_rows


def _page_label(
    evidence: Mapping[
        str,
        Any,
    ],
) -> str:
    """Format one evidence page range for console review."""

    page_start = (
        evidence.get(
            "page_start"
        )
    )

    page_end = (
        evidence.get(
            "page_end"
        )
    )

    if (
        page_start is None
        and page_end is None
    ):
        return "page unknown"

    if (
        page_start
        == page_end
    ):
        return (
            f"p. {page_start}"
        )

    return (
        f"pp. {page_start}-{page_end}"
    )


def _print_evidence(
    evidence: Mapping[
        str,
        Any,
    ],
    *,
    index: int,
    show_evidence_id: bool,
) -> None:
    """Print one exact evidence passage."""

    markers: list[str] = []

    if (
        evidence.get(
            "is_primary"
        )
    ):
        markers.append(
            "PRIMARY"
        )

    if (
        evidence.get(
            "is_gold_relevant"
        )
    ):
        markers.append(
            "GOLD"
        )

    marker_text = (
        ", ".join(
            markers
        )
        if markers
        else "SELECTED"
    )

    evidence_id = (
        evidence.get(
            "evidence_id"
        )
    )

    identifier = (
        f"{evidence_id} / "
        if (
            show_evidence_id
            and evidence_id
        )
        else ""
    )

    print()
    print(
        f"[{index}] "
        f"{identifier}"
        f"{marker_text}"
    )

    print(
        f"{evidence.get('title')} — "
        f"{_page_label(evidence)}"
    )

    print(
        f"point_id: "
        f"{evidence.get('point_id')}"
    )

    print()
    print(
        evidence.get(
            "chunk_text"
        )
    )


def _prompt_choice(
    prompt: str,
    choices: Mapping[
        str,
        str,
    ],
) -> str:
    """Prompt until the reviewer chooses a valid label or quits."""

    while True:
        value = (
            input(
                prompt
            )
            .strip()
            .lower()
        )

        if (
            value
            == "q"
        ):
            return "__quit__"

        label = (
            choices.get(
                value
            )
        )

        if label is not None:
            return label

        print(
            "Invalid choice."
        )


def review_interactively(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    output_path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Review pending responses interactively and save after every answer."""

    total = len(
        rows
    )

    for (
        index,
        row,
    ) in enumerate(
        rows
    ):
        if (
            row[
                "review_status"
            ]
            == REVIEW_STATUS_COMPLETED
        ):
            continue

        print()
        print(
            "=" * 100
        )

        print(
            f"Question {index + 1}/{total}: "
            f"{row['question_id']}"
        )

        print(
            "=" * 100
        )

        print(
            "Language pair: "
            f"{row['query_language']}"
            "->"
            f"{row['target_language']}"
        )

        print(
            "Selected primary hit: "
            f"{row['selected_primary_hit']}"
        )

        print(
            "Selected relevant recall: "
            f"{row['selected_relevant_recall']}"
        )

        print()
        print(
            "QUERY:"
        )

        print(
            row[
                "query"
            ]
        )

        print()
        print(
            "APPLICATION-VISIBLE ANSWER:"
        )

        print(
            row[
                "answer_text"
            ]
        )

        print()
        print(
            "-" * 100
        )

        print(
            "GOLD REFERENCE EVIDENCE"
        )

        print(
            "Use this evidence primarily "
            "to judge COMPLETENESS."
        )

        for (
            evidence_index,
            evidence,
        ) in enumerate(
            row[
                "gold_reference_evidence"
            ],
            start=1,
        ):
            _print_evidence(
                evidence,
                index=(
                    evidence_index
                ),
                show_evidence_id=False,
            )

        print()
        print(
            "-" * 100
        )

        print(
            "SELECTED RAG EVIDENCE"
        )

        print(
            "Use this together with the gold "
            "evidence to judge FACTUAL FIDELITY."
        )

        for (
            evidence_index,
            evidence,
        ) in enumerate(
            row[
                "selected_evidence"
            ],
            start=1,
        ):
            _print_evidence(
                evidence,
                index=(
                    evidence_index
                ),
                show_evidence_id=True,
            )

        print()
        print(
            "=" * 100
        )

        print(
            "COMPLETENESS"
        )

        print(
            "  1 = complete"
        )

        print(
            "      Covers the central primary "
            "answer and all material information "
            "needed from the verified relevant evidence."
        )

        print(
            "  2 = mostly_complete"
        )

        print(
            "      Covers the central answer but "
            "omits only secondary material detail."
        )

        print(
            "  3 = incomplete"
        )

        print(
            "      Misses the central answer, "
            "primary evidence, or another major "
            "material part of the requested information."
        )

        print(
            "  4 = needs_review"
        )

        print(
            "      Completeness cannot be "
            "classified confidently."
        )

        print(
            "  q = quit and preserve progress"
        )

        completeness_label = (
            _prompt_choice(
                "Completeness choice: ",
                COMPLETENESS_CHOICES,
            )
        )

        if (
            completeness_label
            == "__quit__"
        ):
            break

        print()
        print(
            "FACTUAL FIDELITY"
        )

        print(
            "  1 = fully_faithful"
        )

        print(
            "      No material factual error, "
            "unsupported factual assertion, "
            "or materially wrong number/qualification."
        )

        print(
            "  2 = minor_issue"
        )

        print(
            "      A localized factual imprecision "
            "or small error exists but does not "
            "change the central answer."
        )

        print(
            "  3 = major_issue"
        )

        print(
            "      A material factual claim is "
            "unsupported, contradicted, or "
            "substantially incorrect."
        )

        print(
            "  4 = needs_review"
        )

        print(
            "      Factual fidelity cannot be "
            "classified confidently."
        )

        print(
            "  q = quit and preserve progress"
        )

        fidelity_label = (
            _prompt_choice(
                "Fidelity choice: ",
                FIDELITY_CHOICES,
            )
        )

        if (
            fidelity_label
            == "__quit__"
        ):
            break

        notes = input(
            "Optional notes "
            "(press Enter for none): "
        )

        rows[
            index
        ] = (
            apply_review_labels(
                row,
                completeness_label=(
                    completeness_label
                ),
                factual_fidelity_label=(
                    fidelity_label
                ),
                review_notes=(
                    notes
                ),
            )
        )

        write_jsonl_atomic(
            output_path,
            rows,
        )

        completed_count = sum(
            item[
                "review_status"
            ]
            == REVIEW_STATUS_COMPLETED
            for item in rows
        )

        print(
            f"Saved. Completed "
            f"{completed_count}/{total}."
        )

    return rows


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
    """Group review rows by query->evidence language pair."""

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


def print_answer_quality_summary(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Print overall and language-pair answer-quality metrics."""

    grouped = (
        _group_by_language_pair(
            rows
        )
    )

    print()
    print(
        "=" * 112
    )

    print(
        "Human answer completeness and factual-fidelity evaluation"
    )

    print(
        "=" * 112
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>5}"
        f"{'Done':>7}"
        f"{'Scored':>8}"
        f"{'FullComp':>10}"
        f"{'>=Mostly':>10}"
        f"{'Faithful':>10}"
        f"{'NoMajor':>10}"
        f"{'Strong':>10}"
        f"{'Accept':>10}"
    )

    print(
        "-" * 112
    )

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

        metrics = (
            aggregate_answer_quality(
                pair_rows
            )
        )

        print(
            f"{pair[0]}->{pair[1]:<6}"
            f"{metrics['question_count']:>5}"
            f"{metrics['completion_rate']:>7.3f}"
            f"{metrics['scorable_rate']:>8.3f}"
            f"{metrics['fully_complete_rate']:>10.3f}"
            f"{metrics['at_least_mostly_complete_rate']:>10.3f}"
            f"{metrics['fully_faithful_rate']:>10.3f}"
            f"{metrics['no_major_factual_error_rate']:>10.3f}"
            f"{metrics['strong_answer_rate']:>10.3f}"
            f"{metrics['acceptable_answer_rate']:>10.3f}"
        )

    print(
        "-" * 112
    )

    overall = (
        aggregate_answer_quality(
            rows
        )
    )

    print(
        f"{'overall':<10}"
        f"{overall['question_count']:>5}"
        f"{overall['completion_rate']:>7.3f}"
        f"{overall['scorable_rate']:>8.3f}"
        f"{overall['fully_complete_rate']:>10.3f}"
        f"{overall['at_least_mostly_complete_rate']:>10.3f}"
        f"{overall['fully_faithful_rate']:>10.3f}"
        f"{overall['no_major_factual_error_rate']:>10.3f}"
        f"{overall['strong_answer_rate']:>10.3f}"
        f"{overall['acceptable_answer_rate']:>10.3f}"
    )

    print()

    print(
        "Reviewed: "
        f"{overall['reviewed_count']}/"
        f"{overall['question_count']}"
    )

    print(
        "Scored: "
        f"{overall['scored_count']}/"
        f"{overall['reviewed_count']}"
    )

    print(
        "Needs review: "
        f"{overall['needs_review_count']}"
    )

    print()

    print(
        "Completeness counts:"
    )

    print(
        "  complete: "
        f"{overall['complete_count']}"
    )

    print(
        "  mostly_complete: "
        f"{overall['mostly_complete_count']}"
    )

    print(
        "  incomplete: "
        f"{overall['incomplete_count']}"
    )

    print()

    print(
        "Factual-fidelity counts:"
    )

    print(
        "  fully_faithful: "
        f"{overall['fully_faithful_count']}"
    )

    print(
        "  minor_issue: "
        f"{overall['minor_factual_issue_count']}"
    )

    print(
        "  major_issue: "
        f"{overall['major_factual_issue_count']}"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the answer-quality review CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Build, review, or summarize "
                "production answer completeness "
                "and factual fidelity."
            )
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    build_parser = (
        subparsers.add_parser(
            "build",
            help=(
                "Build the pending 30-question "
                "answer-quality review artifact."
            ),
        )
    )

    build_parser.add_argument(
        "--rag",
        type=Path,
        default=(
            DEFAULT_RAG_PATH
        ),
    )

    build_parser.add_argument(
        "--benchmark",
        type=Path,
        default=(
            DEFAULT_BENCHMARK_PATH
        ),
    )

    build_parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    review_parser = (
        subparsers.add_parser(
            "review",
            help=(
                "Human-review pending "
                "answer-quality rows."
            ),
        )
    )

    review_parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    summary_parser = (
        subparsers.add_parser(
            "summary",
            help=(
                "Summarize an answer-quality "
                "review artifact."
            ),
        )
    )

    summary_parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser


def main(
) -> None:
    """Run the requested answer-quality workflow."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    if (
        args.command
        == "build"
    ):
        rows = (
            build_review_dataset(
                rag_path=(
                    args.rag
                ),
                benchmark_path=(
                    args.benchmark
                ),
                output_path=(
                    args.output
                ),
            )
        )

        print(
            "Built answer-quality "
            f"review dataset: {len(rows)} rows"
        )

        print(
            f"Output: {args.output}"
        )

        return

    rows = (
        load_review_rows(
            args.input
        )
    )

    if (
        args.command
        == "review"
    ):
        review_interactively(
            rows,
            output_path=(
                args.input
            ),
        )

        rows = (
            load_review_rows(
                args.input
            )
        )

    print_answer_quality_summary(
        rows
    )


if __name__ == "__main__":
    main()