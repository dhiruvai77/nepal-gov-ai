"""Human answer-quality review for the full completeness-prompt experiment.

This module evaluates the 30 completeness-prompt answers using the same
completeness and factual-fidelity rubric as the original production review.

Important review design:

- the candidate answer is reviewed against the exact same gold evidence,
- the selected production evidence is unchanged,
- the original production answer is not shown during interactive candidate
  labeling, reducing direct comparison bias,
- the original human labels are retained only for final metric comparison,
- progress is persisted after every reviewed answer,
- no generation, retrieval, embedding, reranking, Qdrant, or hosted calls occur.

After all 30 candidate answers are labeled, the summary compares:

- fully complete rate,
- at-least-mostly-complete rate,
- fully faithful rate,
- no-major-factual-error rate,
- strong-answer rate,
- acceptable-answer rate,

against the original production human review.
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

from src.evaluation.answer_quality_review import (
    COMPLETENESS_CHOICES,
    COMPLETENESS_LABELS,
    FIDELITY_CHOICES,
    FIDELITY_LABELS,
    LANGUAGE_PAIR_ORDER,
    REVIEW_CONFIG_ID,
    REVIEW_SCHEMA_VERSION,
    REVIEW_STATUS_COMPLETED,
    REVIEW_STATUS_PENDING,
    aggregate_answer_quality,
    load_review_rows as load_baseline_review_rows,
)
from src.evaluation.run_completeness_prompt_full_benchmark import (
    FULL_RUN_CONFIG_ID,
    load_existing_full_output,
)


PAIR_REVIEW_SCHEMA_VERSION = 1

PAIR_REVIEW_CONFIG_ID = (
    "production-rag-v2-completeness-prompt-full-review-v1"
)

DEFAULT_BASELINE_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_answer_quality_review_v1.jsonl"
)

DEFAULT_CANDIDATE_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_prompt_full_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_prompt_full_v1_"
    "answer_quality_review.jsonl"
)

EXPECTED_QUESTION_COUNT = 30


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
    question_id: str,
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
            f"{question_id}: {field_name} "
            "must contain non-whitespace text."
        )

    return value


def _selected_point_ids(
    evidence: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> list[str]:
    """Return selected evidence point IDs in stable order."""

    point_ids: list[str] = []

    for item in evidence:
        point_id = (
            item.get(
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
                "Selected evidence contains "
                "an invalid point_id."
            )

        point_ids.append(
            point_id
        )

    return point_ids


def build_pair_review_row(
    baseline_row: Mapping[
        str,
        Any,
    ],
    candidate_row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """Build one pending candidate review paired with baseline metadata."""

    question_id = (
        _require_nonempty_string(
            baseline_row.get(
                "question_id"
            ),
            field_name="question_id",
            question_id="unknown",
        )
    )

    if (
        candidate_row.get(
            "question_id"
        )
        != question_id
    ):
        raise ValueError(
            f"{question_id}: candidate "
            "question_id does not match baseline."
        )

    if (
        candidate_row.get(
            "run_config_id"
        )
        != FULL_RUN_CONFIG_ID
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "candidate run_config_id."
        )

    if (
        candidate_row.get(
            "accepted"
        )
        is not True
    ):
        raise ValueError(
            f"{question_id}: candidate answer "
            "was not structurally accepted."
        )

    invalid_ids = (
        candidate_row.get(
            "invalid_evidence_ids"
        )
    )

    if (
        not isinstance(
            invalid_ids,
            list,
        )
        or invalid_ids
    ):
        raise ValueError(
            f"{question_id}: candidate contains "
            "invalid evidence references."
        )

    selected_evidence = (
        baseline_row.get(
            "selected_evidence"
        )
    )

    if not isinstance(
        selected_evidence,
        list,
    ):
        raise ValueError(
            f"{question_id}: baseline "
            "selected_evidence must be a list."
        )

    selected_ids = (
        _selected_point_ids(
            selected_evidence
        )
    )

    if (
        candidate_row.get(
            "selected_point_ids"
        )
        != selected_ids
    ):
        raise ValueError(
            f"{question_id}: candidate selected "
            "context does not match production context."
        )

    candidate_answer = (
        _require_nonempty_string(
            candidate_row.get(
                "answer_text"
            ),
            field_name=(
                "candidate answer_text"
            ),
            question_id=question_id,
        )
    )

    baseline_answer = (
        _require_nonempty_string(
            baseline_row.get(
                "answer_text"
            ),
            field_name=(
                "baseline answer_text"
            ),
            question_id=question_id,
        )
    )

    return {
        "review_schema_version": (
            PAIR_REVIEW_SCHEMA_VERSION
        ),
        "review_config_id": (
            PAIR_REVIEW_CONFIG_ID
        ),
        "source_baseline_review_config_id": (
            REVIEW_CONFIG_ID
        ),
        "source_candidate_run_config_id": (
            FULL_RUN_CONFIG_ID
        ),
        "question_id": (
            question_id
        ),
        "query": (
            baseline_row[
                "query"
            ]
        ),
        "query_language": (
            baseline_row[
                "query_language"
            ]
        ),
        "target_language": (
            baseline_row[
                "target_language"
            ]
        ),
        "answer_language": (
            baseline_row.get(
                "answer_language"
            )
        ),
        "category": (
            baseline_row[
                "category"
            ]
        ),
        "primary_relevant_chunk_ids": copy.deepcopy(
            baseline_row[
                "primary_relevant_chunk_ids"
            ]
        ),
        "relevant_chunk_ids": copy.deepcopy(
            baseline_row[
                "relevant_chunk_ids"
            ]
        ),
        "selected_primary_hit": (
            baseline_row.get(
                "selected_primary_hit"
            )
        ),
        "selected_relevant_recall": (
            baseline_row.get(
                "selected_relevant_recall"
            )
        ),
        "selected_evidence": copy.deepcopy(
            selected_evidence
        ),
        "gold_reference_evidence": copy.deepcopy(
            baseline_row[
                "gold_reference_evidence"
            ]
        ),
        "baseline_answer_text": (
            baseline_answer
        ),
        "baseline_completeness_label": (
            baseline_row[
                "completeness_label"
            ]
        ),
        "baseline_factual_fidelity_label": (
            baseline_row[
                "factual_fidelity_label"
            ]
        ),
        "baseline_review_notes": (
            baseline_row.get(
                "review_notes"
            )
        ),
        "candidate_answer_text": (
            candidate_answer
        ),
        "candidate_completeness_label": None,
        "candidate_factual_fidelity_label": None,
        "candidate_review_notes": None,
        "review_status": (
            REVIEW_STATUS_PENDING
        ),
    }


def validate_pair_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Validate one pending or completed paired review row."""

    question_id = (
        _require_nonempty_string(
            row.get(
                "question_id"
            ),
            field_name="question_id",
            question_id="unknown",
        )
    )

    if (
        row.get(
            "review_schema_version"
        )
        != PAIR_REVIEW_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "review_schema_version."
        )

    if (
        row.get(
            "review_config_id"
        )
        != PAIR_REVIEW_CONFIG_ID
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "review_config_id."
        )

    if (
        row.get(
            "source_baseline_review_config_id"
        )
        != REVIEW_CONFIG_ID
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "baseline review config."
        )

    if (
        row.get(
            "source_candidate_run_config_id"
        )
        != FULL_RUN_CONFIG_ID
    ):
        raise ValueError(
            f"{question_id}: unexpected "
            "candidate run config."
        )

    for field_name in (
        "query",
        "query_language",
        "target_language",
        "candidate_answer_text",
        "baseline_answer_text",
    ):
        _require_nonempty_string(
            row.get(
                field_name
            ),
            field_name=field_name,
            question_id=question_id,
        )

    primary_ids = (
        row.get(
            "primary_relevant_chunk_ids"
        )
    )

    relevant_ids = (
        row.get(
            "relevant_chunk_ids"
        )
    )

    if (
        not isinstance(
            primary_ids,
            list,
        )
        or not primary_ids
    ):
        raise ValueError(
            f"{question_id}: primary evidence "
            "must be a non-empty list."
        )

    if (
        not isinstance(
            relevant_ids,
            list,
        )
        or not relevant_ids
    ):
        raise ValueError(
            f"{question_id}: relevant evidence "
            "must be a non-empty list."
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
            f"{question_id}: gold reference "
            "evidence must be non-empty."
        )

    gold_ids = [
        item.get(
            "point_id"
        )
        for item in gold_evidence
        if isinstance(
            item,
            Mapping,
        )
    ]

    if (
        gold_ids
        != relevant_ids
    ):
        raise ValueError(
            f"{question_id}: gold evidence order "
            "does not match relevant IDs."
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

    selected_ids = (
        _selected_point_ids(
            selected_evidence
        )
    )

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
            f"{question_id}: selected evidence "
            "contains duplicate point IDs."
        )

    if (
        row.get(
            "baseline_completeness_label"
        )
        not in COMPLETENESS_LABELS
    ):
        raise ValueError(
            f"{question_id}: invalid baseline "
            "completeness label."
        )

    if (
        row.get(
            "baseline_factual_fidelity_label"
        )
        not in FIDELITY_LABELS
    ):
        raise ValueError(
            f"{question_id}: invalid baseline "
            "factual-fidelity label."
        )

    status = (
        row.get(
            "review_status"
        )
    )

    if (
        status
        not in {
            REVIEW_STATUS_PENDING,
            REVIEW_STATUS_COMPLETED,
        }
    ):
        raise ValueError(
            f"{question_id}: unsupported "
            f"review_status {status!r}."
        )

    completeness = (
        row.get(
            "candidate_completeness_label"
        )
    )

    fidelity = (
        row.get(
            "candidate_factual_fidelity_label"
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
                "cannot contain candidate labels."
            )

        return

    if (
        completeness
        not in COMPLETENESS_LABELS
    ):
        raise ValueError(
            f"{question_id}: invalid candidate "
            "completeness label."
        )

    if (
        fidelity
        not in FIDELITY_LABELS
    ):
        raise ValueError(
            f"{question_id}: invalid candidate "
            "factual-fidelity label."
        )


def apply_candidate_labels(
    row: Mapping[
        str,
        Any,
    ],
    *,
    completeness_label: str,
    factual_fidelity_label: str,
    review_notes: str | None,
) -> dict[
    str,
    Any,
]:
    """Return one completed candidate review row."""

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

    completed = copy.deepcopy(
        dict(
            row
        )
    )

    completed[
        "candidate_completeness_label"
    ] = (
        completeness_label
    )

    completed[
        "candidate_factual_fidelity_label"
    ] = (
        factual_fidelity_label
    )

    completed[
        "candidate_review_notes"
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

    validate_pair_review_row(
        completed
    )

    return completed


def write_rows_atomic(
    path: str | Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Write the complete review artifact atomically."""

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
        for row in rows:
            handle.write(
                json.dumps(
                    dict(
                        row
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            handle.write(
                "\n"
            )

    temporary.replace(
        destination
    )


def load_pair_review_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load and validate the paired human-review artifact."""

    input_path = Path(
        path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Review artifact does not exist: {input_path}"
        )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen: set[str] = set()

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
                    "Invalid paired-review JSON "
                    f"on line {line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    f"Review row {line_number} "
                    "must be a JSON object."
                )

            validate_pair_review_row(
                row
            )

            question_id = (
                row[
                    "question_id"
                ]
            )

            if (
                question_id
                in seen
            ):
                raise ValueError(
                    "Duplicate paired-review "
                    f"question_id {question_id!r}."
                )

            seen.add(
                question_id
            )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            "Paired-review artifact contains no rows."
        )

    return rows


def build_review_dataset(
    *,
    baseline_review_path: str | Path = (
        DEFAULT_BASELINE_REVIEW_PATH
    ),
    candidate_path: str | Path = (
        DEFAULT_CANDIDATE_PATH
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Build 30 pending candidate-review rows."""

    baseline_rows = (
        load_baseline_review_rows(
            baseline_review_path
        )
    )

    if (
        len(
            baseline_rows
        )
        != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError(
            "Baseline human review must contain "
            f"{EXPECTED_QUESTION_COUNT} rows."
        )

    if any(
        row.get(
            "review_status"
        )
        != REVIEW_STATUS_COMPLETED
        for row in baseline_rows
    ):
        raise ValueError(
            "Baseline human review must be complete."
        )

    candidate_rows = (
        load_existing_full_output(
            candidate_path,
            baseline_rows,
        )
    )

    if (
        len(
            candidate_rows
        )
        != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError(
            "Completeness-prompt benchmark must "
            f"contain {EXPECTED_QUESTION_COUNT} rows."
        )

    review_rows = [
        build_pair_review_row(
            baseline_row,
            candidate_rows[
                baseline_row[
                    "question_id"
                ]
            ],
        )
        for baseline_row in baseline_rows
    ]

    for row in review_rows:
        validate_pair_review_row(
            row
        )

    return review_rows


def _page_label(
    evidence: Mapping[
        str,
        Any,
    ],
) -> str:
    """Render one evidence page range."""

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
        page_start
        == page_end
    ):
        return str(
            page_start
        )

    return (
        f"{page_start}-{page_end}"
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
    """Print one evidence passage for human review."""

    evidence_id = (
        evidence.get(
            "evidence_id"
        )
    )

    prefix = (
        f"[{evidence_id}] "
        if (
            show_evidence_id
            and evidence_id
        )
        else ""
    )

    print()
    print(
        f"{index}. {prefix}"
        f"{evidence.get('title')}"
    )

    print(
        "Pages: "
        f"{_page_label(evidence)}"
    )

    print(
        "Point ID: "
        f"{evidence.get('point_id')}"
    )

    print(
        "Primary: "
        f"{bool(evidence.get('is_primary'))}"
    )

    print(
        "Gold relevant: "
        f"{bool(evidence.get('is_gold_relevant'))}"
    )

    print()

    print(
        evidence.get(
            "chunk_text"
        )
    )


def _prompt_choice(
    prompt_text: str,
    choices: Mapping[
        str,
        str,
    ],
) -> str:
    """Read one validated numeric review choice or quit request."""

    while True:
        value = (
            input(
                prompt_text
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
            "Invalid choice. Enter one of "
            f"{', '.join(choices)} or q."
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
    """Blind-review candidate answers and persist progress after every row."""

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
            f"Candidate question "
            f"{index + 1}/{total}: "
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
            "CANDIDATE APPLICATION-VISIBLE ANSWER:"
        )

        print(
            row[
                "candidate_answer_text"
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
            "Use this primarily to judge COMPLETENESS."
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
                index=evidence_index,
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
            "Use this together with gold evidence "
            "to judge FACTUAL FIDELITY."
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
                index=evidence_index,
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
            "      Covers the central primary answer "
            "and all material information needed from "
            "the verified relevant evidence."
        )

        print(
            "  2 = mostly_complete"
        )

        print(
            "      Covers the central answer but omits "
            "only secondary material detail."
        )

        print(
            "  3 = incomplete"
        )

        print(
            "      Misses the central answer, primary "
            "evidence, or another major material part."
        )

        print(
            "  4 = needs_review"
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
            "unsupported assertion, or materially "
            "wrong number/qualification."
        )

        print(
            "  2 = minor_issue"
        )

        print(
            "      Localized factual imprecision or "
            "small error that does not change the "
            "central answer."
        )

        print(
            "  3 = major_issue"
        )

        print(
            "      Material unsupported, contradicted, "
            "or substantially incorrect content."
        )

        print(
            "  4 = needs_review"
        )

        print(
            "  q = quit and preserve progress"
        )

        fidelity_label = (
            _prompt_choice(
                "Factual fidelity choice: ",
                FIDELITY_CHOICES,
            )
        )

        if (
            fidelity_label
            == "__quit__"
        ):
            break

        print()

        notes = input(
            "Review notes "
            "(optional, press Enter for none): "
        )

        rows[
            index
        ] = (
            apply_candidate_labels(
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

        write_rows_atomic(
            output_path,
            rows,
        )

        print(
            "Saved."
        )

    return rows


def candidate_as_standard_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """Adapt one candidate review to the established quality metric contract."""

    return {
        "review_schema_version": (
            REVIEW_SCHEMA_VERSION
        ),
        "review_config_id": (
            REVIEW_CONFIG_ID
        ),
        "question_id": (
            row[
                "question_id"
            ]
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
        "primary_relevant_chunk_ids": copy.deepcopy(
            row[
                "primary_relevant_chunk_ids"
            ]
        ),
        "relevant_chunk_ids": copy.deepcopy(
            row[
                "relevant_chunk_ids"
            ]
        ),
        "selected_evidence": copy.deepcopy(
            row[
                "selected_evidence"
            ]
        ),
        "gold_reference_evidence": copy.deepcopy(
            row[
                "gold_reference_evidence"
            ]
        ),
        "completeness_label": (
            row[
                "candidate_completeness_label"
            ]
        ),
        "factual_fidelity_label": (
            row[
                "candidate_factual_fidelity_label"
            ]
        ),
        "review_notes": (
            row.get(
                "candidate_review_notes"
            )
        ),
        "review_status": (
            row[
                "review_status"
            ]
        ),
    }


def _group_by_pair(
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
        ],
    ],
]:
    """Group rows by query -> evidence language pair."""

    grouped: dict[
        tuple[
            str,
            str,
        ],
        list[
            Mapping[
                str,
                Any,
            ],
        ],
    ] = defaultdict(
        list
    )

    for row in rows:
        grouped[
            (
                row[
                    "query_language"
                ],
                row[
                    "target_language"
                ],
            )
        ].append(
            row
        )

    return dict(
        grouped
    )


def metric_delta(
    candidate: Mapping[
        str,
        Any,
    ],
    baseline: Mapping[
        str,
        Any,
    ],
    metric_name: str,
) -> float:
    """Return candidate minus baseline metric value."""

    return float(
        candidate[
            metric_name
        ]
    ) - float(
        baseline[
            metric_name
        ]
    )


def print_comparison_summary(
    candidate_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    baseline_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Compare human-reviewed candidate quality against production baseline."""

    if (
        len(
            candidate_rows
        )
        != len(
            baseline_rows
        )
    ):
        raise ValueError(
            "Candidate and baseline review sizes differ."
        )

    if any(
        row[
            "review_status"
        ]
        != REVIEW_STATUS_COMPLETED
        for row in (
            candidate_rows
        )
    ):
        reviewed_count = sum(
            row[
                "review_status"
            ]
            == REVIEW_STATUS_COMPLETED
            for row in (
                candidate_rows
            )
        )

        print()
        print(
            "Candidate review is incomplete: "
            f"{reviewed_count}/"
            f"{len(candidate_rows)} reviewed."
        )

        return

    candidate_standard = [
        candidate_as_standard_review_row(
            row
        )
        for row in (
            candidate_rows
        )
    ]

    baseline_by_pair = (
        _group_by_pair(
            baseline_rows
        )
    )

    candidate_by_pair = (
        _group_by_pair(
            candidate_standard
        )
    )

    print()
    print(
        "=" * 122
    )

    print(
        "Human answer-quality comparison: "
        "production vs completeness prompt"
    )

    print(
        "=" * 122
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>5}"
        f"{'BaseFull':>10}"
        f"{'CandFull':>10}"
        f"{'ΔFull':>9}"
        f"{'BaseFaith':>11}"
        f"{'CandFaith':>11}"
        f"{'BaseNoMaj':>11}"
        f"{'CandNoMaj':>11}"
    )

    print(
        "-" * 122
    )

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        baseline_pair = (
            baseline_by_pair.get(
                pair
            )
        )

        candidate_pair = (
            candidate_by_pair.get(
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

        print(
            f"{pair[0]}->{pair[1]:<6}"
            f"{len(candidate_pair):>5}"
            f"{baseline_metrics['fully_complete_rate']:>10.3f}"
            f"{candidate_metrics['fully_complete_rate']:>10.3f}"
            f"{metric_delta(candidate_metrics, baseline_metrics, 'fully_complete_rate'):>+9.3f}"
            f"{baseline_metrics['fully_faithful_rate']:>11.3f}"
            f"{candidate_metrics['fully_faithful_rate']:>11.3f}"
            f"{baseline_metrics['no_major_factual_error_rate']:>11.3f}"
            f"{candidate_metrics['no_major_factual_error_rate']:>11.3f}"
        )

    baseline_overall = (
        aggregate_answer_quality(
            baseline_rows
        )
    )

    candidate_overall = (
        aggregate_answer_quality(
            candidate_standard
        )
    )

    print(
        "-" * 122
    )

    print(
        f"{'overall':<10}"
        f"{len(candidate_standard):>5}"
        f"{baseline_overall['fully_complete_rate']:>10.3f}"
        f"{candidate_overall['fully_complete_rate']:>10.3f}"
        f"{metric_delta(candidate_overall, baseline_overall, 'fully_complete_rate'):>+9.3f}"
        f"{baseline_overall['fully_faithful_rate']:>11.3f}"
        f"{candidate_overall['fully_faithful_rate']:>11.3f}"
        f"{baseline_overall['no_major_factual_error_rate']:>11.3f}"
        f"{candidate_overall['no_major_factual_error_rate']:>11.3f}"
    )

    print()
    print(
        "Overall metric comparison:"
    )

    metric_labels = (
        (
            "fully_complete_rate",
            "Fully complete",
        ),
        (
            "at_least_mostly_complete_rate",
            "At least mostly complete",
        ),
        (
            "fully_faithful_rate",
            "Fully faithful",
        ),
        (
            "no_major_factual_error_rate",
            "No major factual error",
        ),
        (
            "strong_answer_rate",
            "Strong answer",
        ),
        (
            "acceptable_answer_rate",
            "Acceptable answer",
        ),
    )

    for (
        metric_name,
        label,
    ) in (
        metric_labels
    ):
        baseline_value = (
            baseline_overall[
                metric_name
            ]
        )

        candidate_value = (
            candidate_overall[
                metric_name
            ]
        )

        delta = (
            candidate_value
            - baseline_value
        )

        print(
            f"  {label:<28}"
            f"{baseline_value:.3f}"
            f" -> {candidate_value:.3f}"
            f"   ({delta:+.3f})"
        )

    print()
    print(
        "Candidate completeness counts:"
    )

    print(
        "  complete: "
        f"{candidate_overall['complete_count']}"
    )

    print(
        "  mostly_complete: "
        f"{candidate_overall['mostly_complete_count']}"
    )

    print(
        "  incomplete: "
        f"{candidate_overall['incomplete_count']}"
    )

    print()

    print(
        "Candidate factual-fidelity counts:"
    )

    print(
        "  fully_faithful: "
        f"{candidate_overall['fully_faithful_count']}"
    )

    print(
        "  minor_issue: "
        f"{candidate_overall['minor_factual_issue_count']}"
    )

    print(
        "  major_issue: "
        f"{candidate_overall['major_factual_issue_count']}"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the paired-review CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Build, human-review, and summarize "
                "the full completeness-prompt benchmark."
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
        )
    )

    build_parser.add_argument(
        "--baseline-review",
        type=Path,
        default=(
            DEFAULT_BASELINE_REVIEW_PATH
        ),
    )

    build_parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            DEFAULT_CANDIDATE_PATH
        ),
    )

    build_parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    build_parser.add_argument(
        "--reset",
        action="store_true",
    )

    review_parser = (
        subparsers.add_parser(
            "review",
        )
    )

    review_parser.add_argument(
        "--review",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    summary_parser = (
        subparsers.add_parser(
            "summary",
        )
    )

    summary_parser.add_argument(
        "--review",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    summary_parser.add_argument(
        "--baseline-review",
        type=Path,
        default=(
            DEFAULT_BASELINE_REVIEW_PATH
        ),
    )

    return parser


def main(
) -> None:
    """Execute the requested paired human-review operation."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    if (
        args.command
        == "build"
    ):
        output_path = Path(
            args.output
        )

        if (
            output_path.exists()
            and not args.reset
        ):
            raise FileExistsError(
                "Review artifact already exists. "
                "Use --reset only for an intentional "
                "fresh human review."
            )

        rows = (
            build_review_dataset(
                baseline_review_path=(
                    args.baseline_review
                ),
                candidate_path=(
                    args.candidate
                ),
            )
        )

        write_rows_atomic(
            output_path,
            rows,
        )

        print(
            "Built pending completeness-prompt "
            f"human review: {len(rows)} rows"
        )

        print(
            f"Output: {output_path}"
        )

        return

    if (
        args.command
        == "review"
    ):
        rows = (
            load_pair_review_rows(
                args.review
            )
        )

        reviewed = (
            review_interactively(
                rows,
                output_path=(
                    args.review
                ),
            )
        )

        completed = sum(
            row[
                "review_status"
            ]
            == REVIEW_STATUS_COMPLETED
            for row in (
                reviewed
            )
        )

        print()
        print(
            "Review progress: "
            f"{completed}/{len(reviewed)}"
        )

        return

    if (
        args.command
        == "summary"
    ):
        candidate_rows = (
            load_pair_review_rows(
                args.review
            )
        )

        baseline_rows = (
            load_baseline_review_rows(
                args.baseline_review
            )
        )

        print_comparison_summary(
            candidate_rows,
            baseline_rows,
        )

        return

    raise RuntimeError(
        f"Unsupported command {args.command!r}."
    )


if __name__ == "__main__":
    main()