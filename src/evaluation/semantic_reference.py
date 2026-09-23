"""Generic completed semantic-reference validation for NepalGov AI.

The original human semantic-review module is intentionally tied to the
deterministic 48-claim production review sample. Automated semantic-judge
evaluation, however, may also use separately curated challenge/reference sets.

This module defines the common completed-reference contract needed by the
semantic judge without claiming that every reference row belongs to the
original production review sample.

It validates:

1. completed semantic-support labels,
2. citation-requirement labels,
3. claim/citation consistency,
4. individual evidence-support labels,
5. evidence identity and ordering,
6. language-pair metadata,
7. reference-set provenance metadata.

No model calls are performed here.
"""

from __future__ import annotations

import json
from collections.abc import (
    Mapping,
)
from pathlib import Path
from typing import Any

from src.evaluation.build_semantic_evaluation_dataset import (
    SUPPORTED_CITATION_REQUIREMENT_LABELS,
    SUPPORTED_SEMANTIC_LABELS,
)


REFERENCE_STATUS_COMPLETED = (
    "completed"
)

SUPPORTED_INDIVIDUAL_SUPPORT_LABELS = (
    "supported",
    "partially_supported",
    "unsupported",
    "needs_review",
)


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Return one normalized non-empty string."""

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


def _validate_optional_string(
    value: Any,
    *,
    field_name: str,
) -> None:
    """Validate optional free-text metadata."""

    if (
        value is not None
        and not isinstance(
            value,
            str,
        )
    ):
        raise ValueError(
            f"{field_name} must be "
            "a string or null."
        )


def validate_semantic_reference_row(
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Validate one completed semantic-judge reference row."""

    claim_id = (
        _require_nonempty_string(
            row.get(
                "claim_id"
            ),
            field_name="claim_id",
        )
    )

    _require_nonempty_string(
        row.get(
            "question_id"
        ),
        field_name=(
            f"question_id for "
            f"{claim_id}"
        ),
    )

    _require_nonempty_string(
        row.get(
            "query"
        ),
        field_name=(
            f"query for "
            f"{claim_id}"
        ),
    )

    _require_nonempty_string(
        row.get(
            "claim_text"
        ),
        field_name=(
            f"claim_text for "
            f"{claim_id}"
        ),
    )

    query_language = (
        _require_nonempty_string(
            row.get(
                "query_language"
            ),
            field_name=(
                f"query_language for "
                f"{claim_id}"
            ),
        )
    )

    target_language = (
        _require_nonempty_string(
            row.get(
                "target_language"
            ),
            field_name=(
                f"target_language for "
                f"{claim_id}"
            ),
        )
    )

    reference_config_id = (
        _require_nonempty_string(
            row.get(
                "review_sample_config_id"
            ),
            field_name=(
                "review_sample_config_id "
                f"for {claim_id}"
            ),
        )
    )

    if not reference_config_id:
        raise ValueError(
            f"Claim {claim_id!r} has no "
            "reference configuration."
        )

    review_language_pair = (
        _require_nonempty_string(
            row.get(
                "review_language_pair"
            ),
            field_name=(
                "review_language_pair "
                f"for {claim_id}"
            ),
        )
    )

    expected_language_pair = (
        f"{query_language}"
        f"->{target_language}"
    )

    if (
        review_language_pair
        != expected_language_pair
    ):
        raise ValueError(
            f"Claim {claim_id!r} has "
            "inconsistent review_language_pair."
        )

    review_status = (
        row.get(
            "review_status"
        )
    )

    if (
        review_status
        != REFERENCE_STATUS_COMPLETED
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} must be completed."
        )

    semantic_label = (
        row.get(
            "semantic_support_label"
        )
    )

    if (
        semantic_label
        not in SUPPORTED_SEMANTIC_LABELS
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} has invalid "
            "semantic_support_label "
            f"{semantic_label!r}."
        )

    citation_requirement = (
        row.get(
            "citation_requirement_label"
        )
    )

    if (
        citation_requirement
        not in (
            SUPPORTED_CITATION_REQUIREMENT_LABELS
        )
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} has invalid "
            "citation_requirement_label "
            f"{citation_requirement!r}."
        )

    _validate_optional_string(
        row.get(
            "semantic_notes"
        ),
        field_name=(
            f"semantic_notes for "
            f"{claim_id}"
        ),
    )

    has_citation = (
        row.get(
            "has_citation"
        )
    )

    if not isinstance(
        has_citation,
        bool,
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} must contain "
            "boolean has_citation."
        )

    evidence_ids = (
        row.get(
            "evidence_ids"
        )
    )

    if not isinstance(
        evidence_ids,
        list,
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} must contain "
            "evidence_ids as a list."
        )

    normalized_evidence_ids: list[
        str
    ] = []

    for (
        evidence_index,
        evidence_id,
    ) in enumerate(
        evidence_ids,
        start=1,
    ):
        normalized_id = (
            _require_nonempty_string(
                evidence_id,
                field_name=(
                    "evidence_id "
                    f"{evidence_index} for "
                    f"{claim_id}"
                ),
            )
        )

        normalized_evidence_ids.append(
            normalized_id
        )

    if (
        len(
            normalized_evidence_ids
        )
        != len(
            set(
                normalized_evidence_ids
            )
        )
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} contains "
            "duplicate evidence IDs."
        )

    if (
        has_citation
        != bool(
            normalized_evidence_ids
        )
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} has inconsistent "
            "citation state."
        )

    cited_evidence = (
        row.get(
            "cited_evidence"
        )
    )

    if not isinstance(
        cited_evidence,
        list,
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} must contain "
            "cited_evidence as a list."
        )

    if (
        len(
            cited_evidence
        )
        != len(
            normalized_evidence_ids
        )
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} has inconsistent "
            "evidence counts."
        )

    cited_ids: list[
        str
    ] = []

    for (
        evidence_index,
        evidence,
    ) in enumerate(
        cited_evidence,
        start=1,
    ):
        if not isinstance(
            evidence,
            Mapping,
        ):
            raise ValueError(
                f"Semantic reference claim "
                f"{claim_id!r} evidence "
                f"{evidence_index} must "
                "be a mapping."
            )

        evidence_id = (
            _require_nonempty_string(
                evidence.get(
                    "evidence_id"
                ),
                field_name=(
                    "cited evidence_id "
                    f"{evidence_index} for "
                    f"{claim_id}"
                ),
            )
        )

        cited_ids.append(
            evidence_id
        )

        _require_nonempty_string(
            evidence.get(
                "title"
            ),
            field_name=(
                f"title for "
                f"{claim_id} "
                f"{evidence_id}"
            ),
        )

        _require_nonempty_string(
            evidence.get(
                "chunk_text"
            ),
            field_name=(
                f"chunk_text for "
                f"{claim_id} "
                f"{evidence_id}"
            ),
        )

        individual_label = (
            evidence.get(
                "individual_support_label"
            )
        )

        if (
            individual_label
            not in (
                SUPPORTED_INDIVIDUAL_SUPPORT_LABELS
            )
        ):
            raise ValueError(
                f"Semantic reference claim "
                f"{claim_id!r} evidence "
                f"{evidence_id!r} has invalid "
                "individual_support_label "
                f"{individual_label!r}."
            )

        _validate_optional_string(
            evidence.get(
                "individual_support_notes"
            ),
            field_name=(
                "individual_support_notes "
                f"for {claim_id} "
                f"{evidence_id}"
            ),
        )

    if (
        cited_ids
        != normalized_evidence_ids
    ):
        raise ValueError(
            f"Semantic reference claim "
            f"{claim_id!r} cited evidence "
            "IDs do not match evidence_ids "
            "in the same order."
        )

    if (
        not has_citation
        and semantic_label
        in {
            "supported",
            "partially_supported",
        }
    ):
        raise ValueError(
            f"Uncited semantic reference "
            f"claim {claim_id!r} cannot be "
            "supported or partially_supported."
        )


def load_semantic_reference_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load and validate a completed semantic reference JSONL file."""

    reference_path = Path(
        path
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            "Semantic reference file "
            f"does not exist: "
            f"{reference_path}"
        )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen_claim_ids: set[
        str
    ] = set()

    with reference_path.open(
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
                row = (
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in semantic "
                    "reference file on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Semantic reference row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            validate_semantic_reference_row(
                row
            )

            claim_id = str(
                row[
                    "claim_id"
                ]
            )

            if (
                claim_id
                in seen_claim_ids
            ):
                raise ValueError(
                    "Semantic reference "
                    "contains duplicate "
                    f"claim_id "
                    f"{claim_id!r}."
                )

            seen_claim_ids.add(
                claim_id
            )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            "Semantic reference file "
            "contains no rows."
        )

    return rows