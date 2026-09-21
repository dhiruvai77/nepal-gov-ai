"""Insufficient-evidence benchmark contracts and metrics for NepalGov AI.

This module defines a dedicated evaluation dataset for determining whether the
production RAG pipeline answers when evidence should be sufficient and withholds
when the indexed corpus should be insufficient.

The existing retrieval benchmark cannot represent unanswerable questions
because it requires at least one relevant passage. This evaluator therefore uses
a separate schema.

The metrics here evaluate the structural pipeline decision:

- accepted
- withheld

They do not by themselves determine whether an accepted provider response
verbally states that evidence is insufficient. Generated answer text must remain
available in the persisted benchmark output for later semantic inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import (
    Iterable,
    Mapping,
)

from src.rag.pipeline import (
    RAGResult,
)


EXPECTED_BEHAVIOR_ANSWER = "answer"
EXPECTED_BEHAVIOR_WITHHOLD = "withhold"

SUPPORTED_EXPECTED_BEHAVIORS = (
    EXPECTED_BEHAVIOR_ANSWER,
    EXPECTED_BEHAVIOR_WITHHOLD,
)

SUPPORTED_CASE_TYPES = (
    "answerable_control",
    "out_of_corpus_document",
    "out_of_corpus_period",
    "partial_evidence",
    "mixed_supported_unsupported",
)

SUPPORTED_LANGUAGES = (
    "en",
    "ne",
)


@dataclass(frozen=True)
class InsufficientEvidenceRecord:
    """One manually designed evidence-sufficiency benchmark case."""

    question_id: str
    query: str
    query_language: str
    target_language: str
    category: str
    expected_behavior: str
    case_type: str
    expected_document_ids: tuple[
        str,
        ...,
    ]
    reference_question_id: str | None
    notes: str


@dataclass(frozen=True)
class InsufficientEvidenceQueryMetrics:
    """Structural decision metrics for one benchmark case."""

    question_id: str
    expected_behavior: str
    accepted: bool
    withheld: bool
    correct_decision: bool
    false_accept: bool
    false_withhold: bool
    reason: str | None


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

    return (
        value.strip()
    )


def _parse_string_list(
    value: Any,
    *,
    field_name: str,
) -> tuple[
    str,
    ...,
]:
    """Validate and normalize one JSON list of non-empty strings."""

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            f"{field_name} must be a list."
        )

    normalized: list[
        str
    ] = []

    for index, item in enumerate(
        value,
        start=1,
    ):
        normalized.append(
            _require_nonempty_string(
                item,
                field_name=(
                    f"{field_name}[{index}]"
                ),
            )
        )

    if (
        len(
            normalized
        )
        != len(
            set(
                normalized
            )
        )
    ):
        raise ValueError(
            f"{field_name} must not "
            "contain duplicates."
        )

    return tuple(
        normalized
    )


def _parse_optional_string(
    value: Any,
    *,
    field_name: str,
) -> str | None:
    """Validate one optional non-empty string."""

    if value is None:
        return None

    return (
        _require_nonempty_string(
            value,
            field_name=field_name,
        )
    )


def parse_insufficient_evidence_record(
    data: Mapping[
        str,
        Any,
    ],
    *,
    line_number: int | None = None,
) -> InsufficientEvidenceRecord:
    """Validate one benchmark JSON object."""

    prefix = (
        f"Line {line_number}: "
        if line_number is not None
        else ""
    )

    question_id = (
        _require_nonempty_string(
            data.get(
                "question_id"
            ),
            field_name=(
                f"{prefix}question_id"
            ),
        )
    )

    query = (
        _require_nonempty_string(
            data.get(
                "query"
            ),
            field_name=(
                f"{prefix}query"
            ),
        )
    )

    query_language = (
        _require_nonempty_string(
            data.get(
                "query_language"
            ),
            field_name=(
                f"{prefix}query_language"
            ),
        )
        .lower()
    )

    target_language = (
        _require_nonempty_string(
            data.get(
                "target_language"
            ),
            field_name=(
                f"{prefix}target_language"
            ),
        )
        .lower()
    )

    if (
        query_language
        not in SUPPORTED_LANGUAGES
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "unsupported query_language "
            f"{query_language!r}."
        )

    if (
        target_language
        not in SUPPORTED_LANGUAGES
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "unsupported target_language "
            f"{target_language!r}."
        )

    category = (
        _require_nonempty_string(
            data.get(
                "category"
            ),
            field_name=(
                f"{prefix}category"
            ),
        )
    )

    expected_behavior = (
        _require_nonempty_string(
            data.get(
                "expected_behavior"
            ),
            field_name=(
                f"{prefix}expected_behavior"
            ),
        )
    )

    if (
        expected_behavior
        not in SUPPORTED_EXPECTED_BEHAVIORS
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "unsupported expected_behavior "
            f"{expected_behavior!r}."
        )

    case_type = (
        _require_nonempty_string(
            data.get(
                "case_type"
            ),
            field_name=(
                f"{prefix}case_type"
            ),
        )
    )

    if (
        case_type
        not in SUPPORTED_CASE_TYPES
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "unsupported case_type "
            f"{case_type!r}."
        )

    expected_document_ids = (
        _parse_string_list(
            data.get(
                "expected_document_ids"
            ),
            field_name=(
                f"{prefix}"
                "expected_document_ids"
            ),
        )
    )

    if (
        expected_behavior
        == EXPECTED_BEHAVIOR_ANSWER
        and not expected_document_ids
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "answerable controls require "
            "at least one expected document."
        )

    if (
        case_type
        == "answerable_control"
        and expected_behavior
        != EXPECTED_BEHAVIOR_ANSWER
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "answerable_control must use "
            "expected_behavior='answer'."
        )

    if (
        case_type
        != "answerable_control"
        and expected_behavior
        == EXPECTED_BEHAVIOR_ANSWER
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "only answerable_control cases "
            "may currently expect an answer."
        )

    reference_question_id = (
        _parse_optional_string(
            data.get(
                "reference_question_id"
            ),
            field_name=(
                f"{prefix}"
                "reference_question_id"
            ),
        )
    )

    if (
        case_type
        == "answerable_control"
        and reference_question_id
        is None
    ):
        raise ValueError(
            f"{prefix}{question_id}: "
            "answerable controls require "
            "reference_question_id."
        )

    notes = (
        _require_nonempty_string(
            data.get(
                "notes"
            ),
            field_name=(
                f"{prefix}notes"
            ),
        )
    )

    return (
        InsufficientEvidenceRecord(
            question_id=question_id,
            query=query,
            query_language=(
                query_language
            ),
            target_language=(
                target_language
            ),
            category=category,
            expected_behavior=(
                expected_behavior
            ),
            case_type=(
                case_type
            ),
            expected_document_ids=(
                expected_document_ids
            ),
            reference_question_id=(
                reference_question_id
            ),
            notes=notes,
        )
    )


def load_insufficient_evidence_records(
    path: str | Path,
) -> list[
    InsufficientEvidenceRecord
]:
    """Load and validate the insufficient-evidence JSONL benchmark."""

    dataset_path = Path(
        path
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Insufficient-evidence "
            "benchmark does not exist: "
            f"{dataset_path}"
        )

    records: list[
        InsufficientEvidenceRecord
    ] = []

    seen_question_ids: set[
        str
    ] = set()

    with dataset_path.open(
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
                data = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in "
                    "insufficient-evidence "
                    "benchmark on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                data,
                dict,
            ):
                raise ValueError(
                    "Insufficient-evidence "
                    "benchmark row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            record = (
                parse_insufficient_evidence_record(
                    data,
                    line_number=(
                        line_number
                    ),
                )
            )

            if (
                record.question_id
                in seen_question_ids
            ):
                raise ValueError(
                    "Duplicate question_id "
                    f"{record.question_id!r} "
                    f"on line {line_number}."
                )

            seen_question_ids.add(
                record.question_id
            )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            "Insufficient-evidence "
            "benchmark contains no records."
        )

    return records


def evaluate_insufficient_evidence_result(
    record: InsufficientEvidenceRecord,
    result: RAGResult,
) -> InsufficientEvidenceQueryMetrics:
    """Evaluate the structural accept/withhold decision for one case."""

    expected_answer = (
        record.expected_behavior
        == EXPECTED_BEHAVIOR_ANSWER
    )

    expected_withhold = (
        record.expected_behavior
        == EXPECTED_BEHAVIOR_WITHHOLD
    )

    correct_decision = (
        (
            expected_answer
            and result.accepted
        )
        or (
            expected_withhold
            and result.withheld
        )
    )

    false_accept = (
        expected_withhold
        and result.accepted
    )

    false_withhold = (
        expected_answer
        and result.withheld
    )

    reason = (
        result.reason.value
        if result.reason
        is not None
        else None
    )

    return (
        InsufficientEvidenceQueryMetrics(
            question_id=(
                record.question_id
            ),
            expected_behavior=(
                record.expected_behavior
            ),
            accepted=(
                result.accepted
            ),
            withheld=(
                result.withheld
            ),
            correct_decision=(
                correct_decision
            ),
            false_accept=(
                false_accept
            ),
            false_withhold=(
                false_withhold
            ),
            reason=reason,
        )
    )


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """Return a deterministic zero-safe ratio."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def aggregate_insufficient_evidence_metrics(
    metrics: Iterable[
        InsufficientEvidenceQueryMetrics
    ],
) -> dict[
    str,
    float | int,
]:
    """Aggregate structural evidence-sufficiency decision metrics."""

    items = list(
        metrics
    )

    if not items:
        raise ValueError(
            "At least one insufficient-"
            "evidence metric is required."
        )

    answer_items = [
        item
        for item in items
        if (
            item.expected_behavior
            == EXPECTED_BEHAVIOR_ANSWER
        )
    ]

    withhold_items = [
        item
        for item in items
        if (
            item.expected_behavior
            == EXPECTED_BEHAVIOR_WITHHOLD
        )
    ]

    correct_count = sum(
        item.correct_decision
        for item in items
    )

    accepted_answer_count = sum(
        item.accepted
        for item in answer_items
    )

    withheld_withhold_count = sum(
        item.withheld
        for item in withhold_items
    )

    false_accept_count = sum(
        item.false_accept
        for item in items
    )

    false_withhold_count = sum(
        item.false_withhold
        for item in items
    )

    return {
        "question_count": len(
            items
        ),
        "expected_answer_count": len(
            answer_items
        ),
        "expected_withhold_count": len(
            withhold_items
        ),
        "correct_decision_count": (
            correct_count
        ),
        "decision_accuracy": (
            _safe_ratio(
                correct_count,
                len(
                    items
                ),
            )
        ),
        "answer_acceptance_rate": (
            _safe_ratio(
                accepted_answer_count,
                len(
                    answer_items
                ),
            )
        ),
        "withholding_success_rate": (
            _safe_ratio(
                withheld_withhold_count,
                len(
                    withhold_items
                ),
            )
        ),
        "guard_false_accept_count": (
            false_accept_count
        ),
        "guard_false_accept_rate": (
            _safe_ratio(
                false_accept_count,
                len(
                    withhold_items
                ),
            )
        ),
        "guard_false_withhold_count": (
            false_withhold_count
        ),
        "guard_false_withhold_rate": (
            _safe_ratio(
                false_withhold_count,
                len(
                    answer_items
                ),
            )
        ),
    }