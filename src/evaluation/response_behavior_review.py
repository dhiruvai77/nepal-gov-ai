"""Human response-behavior review for the insufficient-evidence benchmark.

This module evaluates the user-visible behavior of persisted RAG responses
separately from the structural evidence-guard decision.

The distinction matters because a generated response may correctly abstain
semantically while still passing the structural evidence guard because it
contains a valid citation.

Human review labels:

- substantive_answer
- abstained
- partial_answer_with_limitation
- needs_review

The application-visible `answer_text` is the primary object being labeled.
`generated_answer_text` is shown as diagnostic context.

No hosted model calls are performed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from src.evaluation.insufficient_evidence_evaluator import (
    EXPECTED_BEHAVIOR_ANSWER,
    EXPECTED_BEHAVIOR_WITHHOLD,
)


DEFAULT_SOURCE_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "insufficient_evidence_v1_response_review.jsonl"
)

REVIEW_SCHEMA_VERSION = 1

REVIEW_CONFIG_ID = (
    "insufficient-evidence-v1-"
    "response-review-v1"
)

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_COMPLETED = "completed"

SUPPORTED_REVIEW_STATUSES = (
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_COMPLETED,
)

RESPONSE_BEHAVIOR_LABELS = (
    "substantive_answer",
    "abstained",
    "partial_answer_with_limitation",
    "needs_review",
)

RESPONSE_BEHAVIOR_CHOICES = {
    "1": "substantive_answer",
    "2": "abstained",
    "3": "partial_answer_with_limitation",
    "4": "needs_review",
}

LANGUAGE_PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)


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


def source_record_sha256(
    row: Mapping[
        str,
        Any,
    ],
) -> str:
    """Return a deterministic fingerprint for one persisted source row."""

    canonical = (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )

    return (
        hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def load_source_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load persisted insufficient-evidence benchmark results."""

    source_path = Path(
        path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Insufficient-evidence "
            "benchmark output does not exist: "
            f"{source_path}"
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
                    "Invalid JSON in response "
                    "review source on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Response review source "
                    f"row {line_number} must "
                    "be a JSON object."
                )

            question_id = (
                _require_nonempty_string(
                    row.get(
                        "question_id"
                    ),
                    field_name=(
                        f"question_id on "
                        f"line {line_number}"
                    ),
                )
            )

            if (
                question_id
                in seen_question_ids
            ):
                raise ValueError(
                    "Response review source "
                    "contains duplicate "
                    f"question_id "
                    f"{question_id!r}."
                )

            seen_question_ids.add(
                question_id
            )

            expected_behavior = (
                row.get(
                    "expected_behavior"
                )
            )

            if (
                expected_behavior
                not in {
                    EXPECTED_BEHAVIOR_ANSWER,
                    EXPECTED_BEHAVIOR_WITHHOLD,
                }
            ):
                raise ValueError(
                    f"Source question "
                    f"{question_id!r} has "
                    "invalid expected_behavior."
                )

            accepted = (
                row.get(
                    "accepted"
                )
            )

            withheld = (
                row.get(
                    "withheld"
                )
            )

            if (
                not isinstance(
                    accepted,
                    bool,
                )
                or not isinstance(
                    withheld,
                    bool,
                )
            ):
                raise ValueError(
                    f"Source question "
                    f"{question_id!r} must "
                    "contain boolean accepted "
                    "and withheld values."
                )

            if (
                accepted
                == withheld
            ):
                raise ValueError(
                    f"Source question "
                    f"{question_id!r} has "
                    "inconsistent structural "
                    "decision state."
                )

            _require_nonempty_string(
                row.get(
                    "answer_text"
                ),
                field_name=(
                    f"answer_text for "
                    f"{question_id}"
                ),
            )

            _require_nonempty_string(
                row.get(
                    "query"
                ),
                field_name=(
                    f"query for "
                    f"{question_id}"
                ),
            )

            _require_nonempty_string(
                row.get(
                    "query_language"
                ),
                field_name=(
                    f"query_language for "
                    f"{question_id}"
                ),
            )

            _require_nonempty_string(
                row.get(
                    "target_language"
                ),
                field_name=(
                    f"target_language for "
                    f"{question_id}"
                ),
            )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            "Response review source "
            "contains no rows."
        )

    return rows


def _selected_document_ids(
    row: Mapping[
        str,
        Any,
    ],
) -> list[
    str
]:
    """Extract unique selected document IDs preserving order."""

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
            "selected_evidence must "
            "be a list."
        )

    document_ids: list[
        str
    ] = []

    for evidence in (
        selected
    ):
        if not isinstance(
            evidence,
            Mapping,
        ):
            raise ValueError(
                "selected_evidence items "
                "must be mappings."
            )

        document_id = (
            _require_nonempty_string(
                evidence.get(
                    "document_id"
                ),
                field_name="document_id",
            )
        )

        if (
            document_id
            not in document_ids
        ):
            document_ids.append(
                document_id
            )

    return document_ids


def _cited_evidence_ids(
    row: Mapping[
        str,
        Any,
    ],
) -> list[
    str
]:
    """Extract cited evidence identifiers preserving order."""

    cited = (
        row.get(
            "cited_evidence"
        )
    )

    if not isinstance(
        cited,
        list,
    ):
        raise ValueError(
            "cited_evidence must "
            "be a list."
        )

    evidence_ids: list[
        str
    ] = []

    for evidence in (
        cited
    ):
        if not isinstance(
            evidence,
            Mapping,
        ):
            raise ValueError(
                "cited_evidence items "
                "must be mappings."
            )

        evidence_id = (
            _require_nonempty_string(
                evidence.get(
                    "evidence_id"
                ),
                field_name="evidence_id",
            )
        )

        evidence_ids.append(
            evidence_id
        )

    return evidence_ids


def build_review_row(
    source_row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """Build one pending human response-behavior review row."""

    question_id = (
        _require_nonempty_string(
            source_row.get(
                "question_id"
            ),
            field_name="question_id",
        )
    )

    return {
        "review_schema_version": (
            REVIEW_SCHEMA_VERSION
        ),
        "review_config_id": (
            REVIEW_CONFIG_ID
        ),
        "source_record_sha256": (
            source_record_sha256(
                source_row
            )
        ),
        "question_id": (
            question_id
        ),
        "query": (
            source_row[
                "query"
            ]
        ),
        "query_language": (
            source_row[
                "query_language"
            ]
        ),
        "target_language": (
            source_row[
                "target_language"
            ]
        ),
        "expected_behavior": (
            source_row[
                "expected_behavior"
            ]
        ),
        "case_type": (
            source_row[
                "case_type"
            ]
        ),
        "structural_accepted": (
            source_row[
                "accepted"
            ]
        ),
        "structural_withheld": (
            source_row[
                "withheld"
            ]
        ),
        "structural_reason": (
            source_row.get(
                "reason"
            )
        ),
        "answer_text": (
            source_row[
                "answer_text"
            ]
        ),
        "generated_answer_text": (
            source_row.get(
                "generated_answer_text"
            )
        ),
        "selected_document_ids": (
            _selected_document_ids(
                source_row
            )
        ),
        "cited_evidence_ids": (
            _cited_evidence_ids(
                source_row
            )
        ),
        "response_behavior_label": None,
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
    """Validate one response-behavior review row."""

    question_id = (
        _require_nonempty_string(
            row.get(
                "question_id"
            ),
            field_name="question_id",
        )
    )

    if (
        row.get(
            "review_schema_version"
        )
        != REVIEW_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Question {question_id!r} "
            "uses an incompatible review "
            "schema version."
        )

    if (
        row.get(
            "review_config_id"
        )
        != REVIEW_CONFIG_ID
    ):
        raise ValueError(
            f"Question {question_id!r} "
            "uses an unexpected review "
            "configuration."
        )

    _require_nonempty_string(
        row.get(
            "source_record_sha256"
        ),
        field_name=(
            f"source_record_sha256 for "
            f"{question_id}"
        ),
    )

    _require_nonempty_string(
        row.get(
            "query"
        ),
        field_name=(
            f"query for {question_id}"
        ),
    )

    _require_nonempty_string(
        row.get(
            "answer_text"
        ),
        field_name=(
            f"answer_text for "
            f"{question_id}"
        ),
    )

    expected_behavior = (
        row.get(
            "expected_behavior"
        )
    )

    if (
        expected_behavior
        not in {
            EXPECTED_BEHAVIOR_ANSWER,
            EXPECTED_BEHAVIOR_WITHHOLD,
        }
    ):
        raise ValueError(
            f"Question {question_id!r} "
            "has invalid expected_behavior."
        )

    accepted = (
        row.get(
            "structural_accepted"
        )
    )

    withheld = (
        row.get(
            "structural_withheld"
        )
    )

    if (
        not isinstance(
            accepted,
            bool,
        )
        or not isinstance(
            withheld,
            bool,
        )
        or accepted
        == withheld
    ):
        raise ValueError(
            f"Question {question_id!r} "
            "has inconsistent structural "
            "decision state."
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
            f"Question {question_id!r} "
            "has invalid review_status."
        )

    if (
        status
        == REVIEW_STATUS_PENDING
    ):
        if (
            row.get(
                "response_behavior_label"
            )
            is not None
        ):
            raise ValueError(
                f"Pending question "
                f"{question_id!r} must not "
                "contain a response label."
            )

        return

    response_label = (
        row.get(
            "response_behavior_label"
        )
    )

    if (
        response_label
        not in RESPONSE_BEHAVIOR_LABELS
    ):
        raise ValueError(
            f"Completed question "
            f"{question_id!r} has invalid "
            "response behavior label."
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
            f"Question {question_id!r} "
            "review_notes must be "
            "a string or null."
        )


def write_jsonl_atomic(
    path: str | Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Write review state atomically."""

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
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

    temporary_path.replace(
        output_path
    )


def load_review_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load and validate response-behavior review rows."""

    review_path = Path(
        path
    )

    if not review_path.exists():
        raise FileNotFoundError(
            "Response review file "
            f"does not exist: "
            f"{review_path}"
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

    with review_path.open(
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
                    "Invalid JSON in response "
                    "review file on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Response review row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            validate_review_row(
                row
            )

            question_id = str(
                row[
                    "question_id"
                ]
            )

            if (
                question_id
                in seen_question_ids
            ):
                raise ValueError(
                    "Response review contains "
                    "duplicate question_id "
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
            "Response review file "
            "contains no rows."
        )

    return rows


def initialize_review_output(
    source_path: str | Path = (
        DEFAULT_SOURCE_PATH
    ),
    *,
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Create or resume the response-behavior review artifact."""

    source_rows = (
        load_source_rows(
            source_path
        )
    )

    expected_rows = [
        build_review_row(
            source_row
        )
        for source_row in (
            source_rows
        )
    ]

    destination = Path(
        output_path
    )

    if not destination.exists():
        write_jsonl_atomic(
            destination,
            expected_rows,
        )

        return (
            expected_rows
        )

    existing_rows = (
        load_review_rows(
            destination
        )
    )

    if (
        len(
            existing_rows
        )
        != len(
            expected_rows
        )
    ):
        raise ValueError(
            "Existing response review "
            "does not match current "
            "benchmark output."
        )

    expected_by_id = {
        row[
            "question_id"
        ]: row
        for row in (
            expected_rows
        )
    }

    for row in (
        existing_rows
    ):
        question_id = (
            row[
                "question_id"
            ]
        )

        expected = (
            expected_by_id.get(
                question_id
            )
        )

        if expected is None:
            raise ValueError(
                "Existing response review "
                "contains unknown question_id "
                f"{question_id!r}."
            )

        if (
            row[
                "source_record_sha256"
            ]
            != expected[
                "source_record_sha256"
            ]
        ):
            raise ValueError(
                "Existing response review "
                f"for {question_id!r} was "
                "created from different "
                "benchmark output."
            )

    return (
        existing_rows
    )


def apply_response_behavior_label(
    row: Mapping[
        str,
        Any,
    ],
    *,
    response_behavior_label: str,
    review_notes: str | None = None,
) -> dict[
    str,
    Any,
]:
    """Return one completed human response-behavior review row."""

    validate_review_row(
        row
    )

    if (
        response_behavior_label
        not in RESPONSE_BEHAVIOR_LABELS
    ):
        raise ValueError(
            "Unsupported response "
            "behavior label: "
            f"{response_behavior_label!r}."
        )

    completed = dict(
        row
    )

    completed[
        "response_behavior_label"
    ] = (
        response_behavior_label
    )

    cleaned_notes = (
        review_notes.strip()
        if isinstance(
            review_notes,
            str,
        )
        else ""
    )

    completed[
        "review_notes"
    ] = (
        cleaned_notes
        or None
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


def aggregate_response_behavior_review(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    float | int,
]:
    """Aggregate human response-behavior metrics."""

    if not rows:
        raise ValueError(
            "At least one response "
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

    answer_rows = [
        row
        for row in reviewed
        if (
            row[
                "expected_behavior"
            ]
            == EXPECTED_BEHAVIOR_ANSWER
        )
    ]

    withhold_rows = [
        row
        for row in reviewed
        if (
            row[
                "expected_behavior"
            ]
            == EXPECTED_BEHAVIOR_WITHHOLD
        )
    ]

    substantive_answers = sum(
        row[
            "response_behavior_label"
        ]
        == "substantive_answer"
        for row in reviewed
    )

    abstentions = sum(
        row[
            "response_behavior_label"
        ]
        == "abstained"
        for row in reviewed
    )

    partial_answers = sum(
        row[
            "response_behavior_label"
        ]
        == "partial_answer_with_limitation"
        for row in reviewed
    )

    needs_review = sum(
        row[
            "response_behavior_label"
        ]
        == "needs_review"
        for row in reviewed
    )

    answer_delivered = sum(
        row[
            "response_behavior_label"
        ]
        == "substantive_answer"
        for row in answer_rows
    )

    successful_abstention = sum(
        row[
            "response_behavior_label"
        ]
        == "abstained"
        for row in withhold_rows
    )

    unsafe_substantive_answers = sum(
        row[
            "response_behavior_label"
        ]
        == "substantive_answer"
        for row in withhold_rows
    )

    unnecessary_abstentions = sum(
        row[
            "response_behavior_label"
        ]
        == "abstained"
        for row in answer_rows
    )

    partial_on_withhold = sum(
        row[
            "response_behavior_label"
        ]
        == "partial_answer_with_limitation"
        for row in withhold_rows
    )

    behavior_correct = (
        answer_delivered
        + successful_abstention
    )

    structural_behavior_agreement = sum(
        (
            row[
                "structural_accepted"
            ]
            and row[
                "response_behavior_label"
            ]
            == "substantive_answer"
        )
        or (
            row[
                "structural_withheld"
            ]
            and row[
                "response_behavior_label"
            ]
            == "abstained"
        )
        for row in reviewed
    )

    safe_abstention_despite_accept = sum(
        (
            row[
                "expected_behavior"
            ]
            == EXPECTED_BEHAVIOR_WITHHOLD
        )
        and row[
            "structural_accepted"
        ]
        and (
            row[
                "response_behavior_label"
            ]
            == "abstained"
        )
        for row in reviewed
    )

    structural_false_accepts = sum(
        (
            row[
                "expected_behavior"
            ]
            == EXPECTED_BEHAVIOR_WITHHOLD
        )
        and row[
            "structural_accepted"
        ]
        for row in reviewed
    )

    return {
        "question_count": len(
            rows
        ),
        "reviewed_count": len(
            reviewed
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
        "substantive_answer_count": (
            substantive_answers
        ),
        "abstention_count": (
            abstentions
        ),
        "partial_answer_count": (
            partial_answers
        ),
        "needs_review_count": (
            needs_review
        ),
        "behavior_accuracy": (
            _safe_ratio(
                behavior_correct,
                len(
                    reviewed
                ),
            )
        ),
        "answer_delivery_rate": (
            _safe_ratio(
                answer_delivered,
                len(
                    answer_rows
                ),
            )
        ),
        "semantic_abstention_success_rate": (
            _safe_ratio(
                successful_abstention,
                len(
                    withhold_rows
                ),
            )
        ),
        "unsafe_substantive_answer_count": (
            unsafe_substantive_answers
        ),
        "unsafe_substantive_answer_rate": (
            _safe_ratio(
                unsafe_substantive_answers,
                len(
                    withhold_rows
                ),
            )
        ),
        "unnecessary_abstention_count": (
            unnecessary_abstentions
        ),
        "unnecessary_abstention_rate": (
            _safe_ratio(
                unnecessary_abstentions,
                len(
                    answer_rows
                ),
            )
        ),
        "partial_answer_on_withhold_count": (
            partial_on_withhold
        ),
        "partial_answer_on_withhold_rate": (
            _safe_ratio(
                partial_on_withhold,
                len(
                    withhold_rows
                ),
            )
        ),
        "structural_behavior_agreement_rate": (
            _safe_ratio(
                structural_behavior_agreement,
                len(
                    reviewed
                ),
            )
        ),
        "structural_false_accept_count": (
            structural_false_accepts
        ),
        "safe_abstention_despite_structural_accept_count": (
            safe_abstention_despite_accept
        ),
        "safe_abstention_despite_structural_accept_rate": (
            _safe_ratio(
                safe_abstention_despite_accept,
                structural_false_accepts,
            )
        ),
    }


def _prompt_choice(
    prompt: str,
) -> str:
    """Prompt until a valid review choice or quit command is entered."""

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
            RESPONSE_BEHAVIOR_CHOICES.get(
                value
            )
        )

        if (
            label
            is not None
        ):
            return label

        print(
            "Invalid choice. "
            "Enter 1, 2, 3, 4, or q."
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
    """Human-review pending application-visible response behavior."""

    total = len(
        rows
    )

    for index, row in enumerate(
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
            "=" * 88
        )

        print(
            f"Question "
            f"{index + 1}/{total}: "
            f"{row['question_id']}"
        )

        print(
            "=" * 88
        )

        print(
            "Expected behavior: "
            f"{row['expected_behavior']}"
        )

        print(
            "Case type: "
            f"{row['case_type']}"
        )

        print(
            "Language pair: "
            f"{row['query_language']}"
            "->"
            f"{row['target_language']}"
        )

        print(
            "Structural result: "
            + (
                "accepted"
                if row[
                    "structural_accepted"
                ]
                else (
                    "withheld "
                    f"({row['structural_reason']})"
                )
            )
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

        generated = (
            row.get(
                "generated_answer_text"
            )
        )

        if (
            isinstance(
                generated,
                str,
            )
            and generated.strip()
            and generated.strip()
            != row[
                "answer_text"
            ].strip()
        ):
            print()
            print(
                "ORIGINAL GENERATED ANSWER "
                "(diagnostic only):"
            )

            print(
                generated
            )

        print()
        print(
            "Selected documents: "
            + (
                ", ".join(
                    row[
                        "selected_document_ids"
                    ]
                )
                or "(none)"
            )
        )

        print(
            "Cited evidence IDs: "
            + (
                ", ".join(
                    row[
                        "cited_evidence_ids"
                    ]
                )
                or "(none)"
            )
        )

        print()
        print(
            "Response behavior:"
        )

        print(
            "  1 = substantive_answer"
        )

        print(
            "      The visible response "
            "actually answers the requested "
            "question with substantive content."
        )

        print(
            "  2 = abstained"
        )

        print(
            "      The visible response says "
            "the requested answer cannot be "
            "supported from the supplied evidence."
        )

        print(
            "  3 = partial_answer_with_limitation"
        )

        print(
            "      The visible response answers "
            "part of the requested substance but "
            "explicitly limits another material part."
        )

        print(
            "  4 = needs_review"
        )

        print(
            "  q = quit and preserve progress"
        )

        label = (
            _prompt_choice(
                "Choice: "
            )
        )

        if (
            label
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
            apply_response_behavior_label(
                row,
                response_behavior_label=(
                    label
                ),
                review_notes=notes,
            )
        )

        write_jsonl_atomic(
            output_path,
            rows,
        )

        completed = sum(
            item[
                "review_status"
            ]
            == REVIEW_STATUS_COMPLETED
            for item in rows
        )

        print(
            f"Saved. Completed "
            f"{completed}/{total}."
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
    """Group response-review rows by language pair."""

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


def print_response_behavior_summary(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Print overall and language-pair human response metrics."""

    grouped = (
        _group_by_language_pair(
            rows
        )
    )

    print()
    print(
        "=" * 88
    )

    print(
        "Human response-level abstention evaluation"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>5}"
        f"{'Done':>7}"
        f"{'BehAcc':>9}"
        f"{'Answer':>9}"
        f"{'Abstain':>9}"
        f"{'Unsafe':>9}"
        f"{'GuardEq':>9}"
    )

    print(
        "-" * 88
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
            aggregate_response_behavior_review(
                pair_rows
            )
        )

        print(
            f"{pair[0]}->{pair[1]:<6}"
            f"{metrics['question_count']:>5}"
            f"{metrics['completion_rate']:>7.3f}"
            f"{metrics['behavior_accuracy']:>9.3f}"
            f"{metrics['answer_delivery_rate']:>9.3f}"
            f"{metrics['semantic_abstention_success_rate']:>9.3f}"
            f"{metrics['unsafe_substantive_answer_rate']:>9.3f}"
            f"{metrics['structural_behavior_agreement_rate']:>9.3f}"
        )

    print(
        "-" * 88
    )

    overall = (
        aggregate_response_behavior_review(
            rows
        )
    )

    print(
        f"{'overall':<10}"
        f"{overall['question_count']:>5}"
        f"{overall['completion_rate']:>7.3f}"
        f"{overall['behavior_accuracy']:>9.3f}"
        f"{overall['answer_delivery_rate']:>9.3f}"
        f"{overall['semantic_abstention_success_rate']:>9.3f}"
        f"{overall['unsafe_substantive_answer_rate']:>9.3f}"
        f"{overall['structural_behavior_agreement_rate']:>9.3f}"
    )

    print()

    print(
        "Reviewed responses: "
        f"{overall['reviewed_count']}/"
        f"{overall['question_count']}"
    )

    print(
        "Partial answers: "
        f"{overall['partial_answer_count']}"
    )

    print(
        "Needs-review responses: "
        f"{overall['needs_review_count']}"
    )

    print(
        "Unsafe substantive answers on "
        "expected-withhold cases: "
        f"{overall['unsafe_substantive_answer_count']}"
    )

    print(
        "Unnecessary abstentions on "
        "answerable controls: "
        f"{overall['unnecessary_abstention_count']}"
    )

    print(
        "Structural false accepts: "
        f"{overall['structural_false_accept_count']}"
    )

    print(
        "Safe semantic abstentions among "
        "structural false accepts: "
        f"{overall['safe_abstention_despite_structural_accept_count']}"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build response-behavior review CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Human-review user-visible "
                "answer vs abstention behavior "
                "for the insufficient-evidence "
                "benchmark."
            )
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    review_parser = (
        subparsers.add_parser(
            "review",
            help=(
                "Review pending response "
                "behavior interactively."
            ),
        )
    )

    review_parser.add_argument(
        "--source",
        type=Path,
        default=(
            DEFAULT_SOURCE_PATH
        ),
    )

    review_parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    summary_parser = (
        subparsers.add_parser(
            "summary",
            help=(
                "Summarize completed "
                "response-behavior labels."
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
    """Run interactive response review or print its summary."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    if (
        args.command
        == "review"
    ):
        rows = (
            initialize_review_output(
                args.source,
                output_path=(
                    args.output
                ),
            )
        )

        review_interactively(
            rows,
            output_path=(
                args.output
            ),
        )

        final_rows = (
            load_review_rows(
                args.output
            )
        )

        print_response_behavior_summary(
            final_rows
        )

        return

    rows = (
        load_review_rows(
            args.input
        )
    )

    print_response_behavior_summary(
        rows
    )


if __name__ == "__main__":
    main()