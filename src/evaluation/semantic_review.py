"""Human semantic citation review and evaluation for NepalGov AI.

This module supports the final human-verified semantic evaluation stage.

It provides:

1. resumable review of the deterministic 48-claim subset,
2. atomic persistence after every completed review,
3. validation of semantic and citation-requirement labels,
4. individual claim-to-passage support labels,
5. aggregate and language-pair semantic metrics.

No LLM judge is used here. The resulting labels are intended to form a small
human-verified reference set for later automated evaluator experiments.
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

from src.evaluation.build_semantic_evaluation_dataset import (
    SUPPORTED_CITATION_REQUIREMENT_LABELS,
    SUPPORTED_SEMANTIC_LABELS,
)
from src.evaluation.build_semantic_review_subset import (
    PAIR_ORDER,
    SAMPLE_CONFIG_ID,
)


DEFAULT_SOURCE_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1_labeled.jsonl"
)

INDIVIDUAL_SUPPORT_LABELS = (
    "supported",
    "partially_supported",
    "unsupported",
    "needs_review",
)

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_COMPLETED = "completed"

SUPPORTED_REVIEW_STATUSES = (
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_COMPLETED,
)

SEMANTIC_SUPPORT_CHOICES = {
    "1": "supported",
    "2": "partially_supported",
    "3": "unsupported",
    "4": "not_a_factual_claim",
    "5": "needs_review",
}

UNCITED_SUPPORT_CHOICES = {
    "3": "unsupported",
    "4": "not_a_factual_claim",
    "5": "needs_review",
}

CITATION_REQUIREMENT_CHOICES = {
    "1": "required",
    "2": "not_required",
    "3": "unclear",
}

INDIVIDUAL_SUPPORT_CHOICES = {
    "1": "supported",
    "2": "partially_supported",
    "3": "unsupported",
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


def validate_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Validate one semantic review row."""

    claim_id = (
        _require_nonempty_string(
            row.get(
                "claim_id"
            ),
            field_name="claim_id",
        )
    )

    if (
        row.get(
            "review_sample_config_id"
        )
        != SAMPLE_CONFIG_ID
    ):
        raise ValueError(
            f"Claim {claim_id!r} has an "
            "unexpected review sample "
            "configuration."
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
            f"Claim {claim_id!r} has invalid "
            f"review_status {status!r}."
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
            f"Claim {claim_id!r} must contain "
            "evidence_ids as a list."
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
            f"Claim {claim_id!r} must contain "
            "boolean has_citation."
        )

    if (
        has_citation
        != bool(
            evidence_ids
        )
    ):
        raise ValueError(
            f"Claim {claim_id!r} has "
            "inconsistent citation state."
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
            f"Claim {claim_id!r} must contain "
            "cited_evidence as a list."
        )

    if (
        len(
            cited_evidence
        )
        != len(
            evidence_ids
        )
    ):
        raise ValueError(
            f"Claim {claim_id!r} has "
            "inconsistent evidence counts."
        )

    for index, evidence in enumerate(
        cited_evidence,
        start=1,
    ):
        if not isinstance(
            evidence,
            Mapping,
        ):
            raise ValueError(
                f"Claim {claim_id!r} evidence "
                f"{index} must be a mapping."
            )

        _require_nonempty_string(
            evidence.get(
                "evidence_id"
            ),
            field_name=(
                f"evidence_id for "
                f"{claim_id}"
            ),
        )

        _require_nonempty_string(
            evidence.get(
                "chunk_text"
            ),
            field_name=(
                f"chunk_text for "
                f"{claim_id}"
            ),
        )

    if (
        status
        == REVIEW_STATUS_PENDING
    ):
        return

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
            f"Completed claim {claim_id!r} "
            "has invalid semantic support "
            f"label {semantic_label!r}."
        )

    citation_requirement = (
        row.get(
            "citation_requirement_label"
        )
    )

    if (
        citation_requirement
        not in SUPPORTED_CITATION_REQUIREMENT_LABELS
    ):
        raise ValueError(
            f"Completed claim {claim_id!r} "
            "has invalid citation requirement "
            f"label {citation_requirement!r}."
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
            f"Uncited claim {claim_id!r} "
            "cannot be semantically supported "
            "by cited evidence."
        )

    for index, evidence in enumerate(
        cited_evidence,
        start=1,
    ):
        individual_label = (
            evidence.get(
                "individual_support_label"
            )
        )

        if (
            individual_label
            not in INDIVIDUAL_SUPPORT_LABELS
        ):
            raise ValueError(
                f"Completed claim {claim_id!r} "
                f"evidence {index} has invalid "
                "individual support label "
                f"{individual_label!r}."
            )


def load_review_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load and validate semantic review rows."""

    review_path = Path(
        path
    )

    if not review_path.exists():
        raise FileNotFoundError(
            "Semantic review file does not "
            f"exist: {review_path}"
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

    with review_path.open(
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

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in semantic "
                    f"review file on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Semantic review row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            validate_review_row(
                row
            )

            claim_id = str(
                row[
                    "claim_id"
                ]
            )

            if claim_id in seen_claim_ids:
                raise ValueError(
                    "Semantic review contains "
                    "duplicate claim_id "
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
            "Semantic review file "
            "contains no rows."
        )

    return rows


def write_jsonl_atomic(
    path: str | Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Persist semantic review state atomically."""

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
    """Create or resume the labeled review file."""

    source_rows = (
        load_review_rows(
            source_path
        )
    )

    destination = Path(
        output_path
    )

    if not destination.exists():
        write_jsonl_atomic(
            destination,
            source_rows,
        )

        return source_rows

    output_rows = (
        load_review_rows(
            destination
        )
    )

    source_ids = [
        row[
            "claim_id"
        ]
        for row in (
            source_rows
        )
    ]

    output_ids = [
        row[
            "claim_id"
        ]
        for row in (
            output_rows
        )
    ]

    if (
        output_ids
        != source_ids
    ):
        raise ValueError(
            "Existing semantic review output "
            "does not match the current "
            "review subset."
        )

    return output_rows


def apply_review_labels(
    row: Mapping[
        str,
        Any,
    ],
    *,
    semantic_support_label: str,
    citation_requirement_label: str,
    individual_support_labels: Sequence[
        str
    ] = (),
    semantic_notes: str | None = None,
) -> dict[
    str,
    Any,
]:
    """Return one completed review row with validated human labels."""

    validate_review_row(
        row
    )

    claim_id = str(
        row[
            "claim_id"
        ]
    )

    if (
        semantic_support_label
        not in SUPPORTED_SEMANTIC_LABELS
    ):
        raise ValueError(
            "Unsupported semantic support "
            f"label: "
            f"{semantic_support_label!r}."
        )

    if (
        citation_requirement_label
        not in SUPPORTED_CITATION_REQUIREMENT_LABELS
    ):
        raise ValueError(
            "Unsupported citation requirement "
            f"label: "
            f"{citation_requirement_label!r}."
        )

    cited_evidence = (
        row[
            "cited_evidence"
        ]
    )

    if (
        len(
            individual_support_labels
        )
        != len(
            cited_evidence
        )
    ):
        raise ValueError(
            f"Claim {claim_id!r} requires "
            f"{len(cited_evidence)} "
            "individual evidence labels."
        )

    for label in (
        individual_support_labels
    ):
        if (
            label
            not in INDIVIDUAL_SUPPORT_LABELS
        ):
            raise ValueError(
                "Unsupported individual "
                f"support label: "
                f"{label!r}."
            )

    if (
        not row[
            "has_citation"
        ]
        and semantic_support_label
        in {
            "supported",
            "partially_supported",
        }
    ):
        raise ValueError(
            "An uncited claim cannot receive "
            "supported or partially_supported "
            "semantic support."
        )

    completed = dict(
        row
    )

    completed_evidence: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for evidence, label in zip(
        cited_evidence,
        individual_support_labels,
        strict=True,
    ):
        evidence_copy = dict(
            evidence
        )

        evidence_copy[
            "individual_support_label"
        ] = label

        completed_evidence.append(
            evidence_copy
        )

    completed[
        "cited_evidence"
    ] = completed_evidence

    completed[
        "semantic_support_label"
    ] = semantic_support_label

    completed[
        "citation_requirement_label"
    ] = citation_requirement_label

    cleaned_notes = (
        semantic_notes.strip()
        if isinstance(
            semantic_notes,
            str,
        )
        else ""
    )

    completed[
        "semantic_notes"
    ] = (
        cleaned_notes
        or None
    )

    completed[
        "review_status"
    ] = REVIEW_STATUS_COMPLETED

    validate_review_row(
        completed
    )

    return completed


def aggregate_semantic_review(
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
    """Aggregate human semantic-review metrics."""

    if not rows:
        raise ValueError(
            "At least one semantic "
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

    supported = sum(
        row[
            "semantic_support_label"
        ]
        == "supported"
        for row in reviewed
    )

    partially_supported = sum(
        row[
            "semantic_support_label"
        ]
        == "partially_supported"
        for row in reviewed
    )

    unsupported = sum(
        row[
            "semantic_support_label"
        ]
        == "unsupported"
        for row in reviewed
    )

    not_factual = sum(
        row[
            "semantic_support_label"
        ]
        == "not_a_factual_claim"
        for row in reviewed
    )

    semantic_needs_review = sum(
        row[
            "semantic_support_label"
        ]
        == "needs_review"
        for row in reviewed
    )

    semantic_decided = (
        supported
        + partially_supported
        + unsupported
    )

    required_rows = [
        row
        for row in reviewed
        if (
            row[
                "citation_requirement_label"
            ]
            == "required"
        )
    ]

    required_with_citation = sum(
        bool(
            row[
                "has_citation"
            ]
        )
        for row in (
            required_rows
        )
    )

    not_required_rows = [
        row
        for row in reviewed
        if (
            row[
                "citation_requirement_label"
            ]
            == "not_required"
        )
    ]

    unnecessary_citations = sum(
        bool(
            row[
                "has_citation"
            ]
        )
        for row in (
            not_required_rows
        )
    )

    unclear_requirement = sum(
        row[
            "citation_requirement_label"
        ]
        == "unclear"
        for row in reviewed
    )

    individual_labels = [
        evidence[
            "individual_support_label"
        ]
        for row in reviewed
        for evidence
        in row[
            "cited_evidence"
        ]
    ]

    individual_supported = sum(
        label
        == "supported"
        for label in (
            individual_labels
        )
    )

    individual_partial = sum(
        label
        == "partially_supported"
        for label in (
            individual_labels
        )
    )

    individual_unsupported = sum(
        label
        == "unsupported"
        for label in (
            individual_labels
        )
    )

    individual_needs_review = sum(
        label
        == "needs_review"
        for label in (
            individual_labels
        )
    )

    individual_decided = (
        individual_supported
        + individual_partial
        + individual_unsupported
    )

    return {
        "claim_count": len(
            rows
        ),
        "reviewed_claim_count": len(
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
        "supported_claim_count": (
            supported
        ),
        "partially_supported_claim_count": (
            partially_supported
        ),
        "unsupported_claim_count": (
            unsupported
        ),
        "not_factual_claim_count": (
            not_factual
        ),
        "semantic_needs_review_count": (
            semantic_needs_review
        ),
        "semantic_decided_claim_count": (
            semantic_decided
        ),
        "fully_supported_rate": (
            _safe_ratio(
                supported,
                semantic_decided,
            )
        ),
        "at_least_partially_supported_rate": (
            _safe_ratio(
                supported
                + partially_supported,
                semantic_decided,
            )
        ),
        "unsupported_rate": (
            _safe_ratio(
                unsupported,
                semantic_decided,
            )
        ),
        "citation_required_claim_count": len(
            required_rows
        ),
        "citation_required_with_citation_count": (
            required_with_citation
        ),
        "required_citation_coverage": (
            _safe_ratio(
                required_with_citation,
                len(
                    required_rows
                ),
            )
        ),
        "citation_not_required_claim_count": len(
            not_required_rows
        ),
        "unnecessary_citation_count": (
            unnecessary_citations
        ),
        "unnecessary_citation_rate": (
            _safe_ratio(
                unnecessary_citations,
                len(
                    not_required_rows
                ),
            )
        ),
        "citation_requirement_unclear_count": (
            unclear_requirement
        ),
        "individual_citation_count": len(
            individual_labels
        ),
        "individual_supported_count": (
            individual_supported
        ),
        "individual_partially_supported_count": (
            individual_partial
        ),
        "individual_unsupported_count": (
            individual_unsupported
        ),
        "individual_needs_review_count": (
            individual_needs_review
        ),
        "individual_decided_count": (
            individual_decided
        ),
        "individual_full_support_rate": (
            _safe_ratio(
                individual_supported,
                individual_decided,
            )
        ),
        "individual_at_least_partial_support_rate": (
            _safe_ratio(
                individual_supported
                + individual_partial,
                individual_decided,
            )
        ),
        "individual_unsupported_rate": (
            _safe_ratio(
                individual_unsupported,
                individual_decided,
            )
        ),
    }


def _prompt_choice(
    prompt: str,
    choices: Mapping[
        str,
        str,
    ],
    *,
    allow_skip: bool = False,
) -> str:
    """Read one validated interactive review choice."""

    while True:
        value = input(
            prompt
        ).strip().lower()

        if value == "q":
            return "__quit__"

        if (
            allow_skip
            and value == "s"
        ):
            return "__skip__"

        if value in choices:
            return choices[
                value
            ]

        print(
            "Invalid choice. "
            "Please try again."
        )


def _print_claim_for_review(
    row: Mapping[
        str,
        Any,
    ],
    *,
    position: int,
    total: int,
) -> None:
    """Display one claim and its exact evidence passages."""

    print()
    print(
        "=" * 78
    )

    print(
        f"Review {position}/{total}: "
        f"{row['claim_id']}"
    )

    print(
        "=" * 78
    )

    print(
        f"Pair: "
        f"{row['review_language_pair']}"
    )

    print(
        f"Stratum: "
        f"{row['review_stratum']}"
    )

    print()
    print(
        "QUESTION:"
    )
    print(
        row[
            "query"
        ]
    )

    print()
    print(
        "CLAIM:"
    )
    print(
        row[
            "claim_text"
        ]
    )

    evidence = (
        row[
            "cited_evidence"
        ]
    )

    if not evidence:
        print()
        print(
            "CITED EVIDENCE: none"
        )

        return

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        print()
        print(
            "-" * 78
        )

        print(
            f"EVIDENCE {index}: "
            f"[{item['evidence_id']}]"
        )

        print(
            f"Source: "
            f"{item['title']}"
        )

        print(
            "Pages: "
            f"{item['page_start']}"
            f"-{item['page_end']}"
        )

        print()

        print(
            item[
                "chunk_text"
            ]
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
    """Interactively review pending claims and checkpoint after each one."""

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

        _print_claim_for_review(
            row,
            position=(
                index
                + 1
            ),
            total=total,
        )

        print()
        print(
            "Citation requirement:"
        )
        print(
            "  1 = required"
        )
        print(
            "  2 = not_required"
        )
        print(
            "  3 = unclear"
        )
        print(
            "  s = skip this claim"
        )
        print(
            "  q = quit and keep progress"
        )

        requirement = (
            _prompt_choice(
                "Choice: ",
                CITATION_REQUIREMENT_CHOICES,
                allow_skip=True,
            )
        )

        if requirement == "__quit__":
            break

        if requirement == "__skip__":
            continue

        print()
        print(
            "Joint semantic support:"
        )

        if row[
            "has_citation"
        ]:
            print(
                "  1 = supported"
            )
            print(
                "  2 = partially_supported"
            )

            support_choices = (
                SEMANTIC_SUPPORT_CHOICES
            )

        else:
            support_choices = (
                UNCITED_SUPPORT_CHOICES
            )

        print(
            "  3 = unsupported"
        )
        print(
            "  4 = not_a_factual_claim"
        )
        print(
            "  5 = needs_review"
        )
        print(
            "  q = quit and keep progress"
        )

        semantic_support = (
            _prompt_choice(
                "Choice: ",
                support_choices,
            )
        )

        if semantic_support == "__quit__":
            break

        individual_labels: list[
            str
        ] = []

        quit_requested = False

        for evidence_index, evidence in enumerate(
            row[
                "cited_evidence"
            ],
            start=1,
        ):
            print()
            print(
                "Individual support for "
                f"[{evidence['evidence_id']}] "
                f"({evidence_index}/"
                f"{len(row['cited_evidence'])}):"
            )

            print(
                "  1 = supported"
            )
            print(
                "  2 = partially_supported"
            )
            print(
                "  3 = unsupported"
            )
            print(
                "  4 = needs_review"
            )
            print(
                "  q = quit and keep progress"
            )

            individual_label = (
                _prompt_choice(
                    "Choice: ",
                    INDIVIDUAL_SUPPORT_CHOICES,
                )
            )

            if (
                individual_label
                == "__quit__"
            ):
                quit_requested = True
                break

            individual_labels.append(
                individual_label
            )

        if quit_requested:
            break

        print()

        notes = input(
            "Optional notes "
            "(press Enter for none): "
        )

        rows[
            index
        ] = (
            apply_review_labels(
                row,
                semantic_support_label=(
                    semantic_support
                ),
                citation_requirement_label=(
                    requirement
                ),
                individual_support_labels=(
                    individual_labels
                ),
                semantic_notes=notes,
            )
        )

        # Persist immediately so manual work survives interruption.
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


def print_semantic_summary(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Print overall and language-pair human-review metrics."""

    print()
    print(
        "=" * 88
    )

    print(
        "Human semantic citation evaluation"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>5}"
        f"{'Done':>7}"
        f"{'Full':>9}"
        f"{'AnySup':>9}"
        f"{'Unsup':>9}"
        f"{'ReqCov':>9}"
        f"{'IndSup':>9}"
    )

    print(
        "-" * 88
    )

    for pair in (
        PAIR_ORDER
    ):
        pair_rows = [
            row
            for row in rows
            if (
                row[
                    "query_language"
                ],
                row[
                    "target_language"
                ],
            )
            == pair
        ]

        if not pair_rows:
            continue

        metrics = (
            aggregate_semantic_review(
                pair_rows
            )
        )

        print(
            f"{pair[0]}->{pair[1]:<6}"
            f"{metrics['claim_count']:>5}"
            f"{metrics['completion_rate']:>7.3f}"
            f"{metrics['fully_supported_rate']:>9.3f}"
            f"{metrics['at_least_partially_supported_rate']:>9.3f}"
            f"{metrics['unsupported_rate']:>9.3f}"
            f"{metrics['required_citation_coverage']:>9.3f}"
            f"{metrics['individual_at_least_partial_support_rate']:>9.3f}"
        )

    print(
        "-" * 88
    )

    overall = (
        aggregate_semantic_review(
            rows
        )
    )

    print(
        f"{'overall':<10}"
        f"{overall['claim_count']:>5}"
        f"{overall['completion_rate']:>7.3f}"
        f"{overall['fully_supported_rate']:>9.3f}"
        f"{overall['at_least_partially_supported_rate']:>9.3f}"
        f"{overall['unsupported_rate']:>9.3f}"
        f"{overall['required_citation_coverage']:>9.3f}"
        f"{overall['individual_at_least_partial_support_rate']:>9.3f}"
    )

    print()

    print(
        "Reviewed claims: "
        f"{overall['reviewed_claim_count']}/"
        f"{overall['claim_count']}"
    )

    print(
        "Semantic needs-review claims: "
        f"{overall['semantic_needs_review_count']}"
    )

    print(
        "Not-a-factual-claim labels: "
        f"{overall['not_factual_claim_count']}"
    )

    print(
        "Citation requirement unclear: "
        f"{overall['citation_requirement_unclear_count']}"
    )

    print(
        "Individual evidence needs-review: "
        f"{overall['individual_needs_review_count']}"
    )

    print(
        "Unnecessary citation rate: "
        f"{overall['unnecessary_citation_rate']:.3f}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the semantic-review command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Human-review NepalGov AI "
            "claim-level citation support."
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
                "Review pending semantic "
                "claims interactively."
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
                "semantic labels."
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


def main() -> None:
    """Run interactive review or semantic summary."""

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

        print_semantic_summary(
            final_rows
        )

        return

    rows = (
        load_review_rows(
            args.input
        )
    )

    print_semantic_summary(
        rows
    )


if __name__ == "__main__":
    main()