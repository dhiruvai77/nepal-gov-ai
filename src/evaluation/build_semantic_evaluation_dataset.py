"""Build the claim-level semantic citation evaluation dataset.

The production RAG benchmark persists selected evidence metadata but deliberately
does not duplicate the complete original passage text. Semantic claim-support
evaluation needs that text.

This module:

1. loads the committed production RAG benchmark,
2. deterministically extracts claim/citation units,
3. resolves selected Qdrant point IDs,
4. validates persisted evidence metadata against the current corpus payload,
5. attaches the exact original chunk_text,
6. writes an unlabeled claim-level JSONL dataset.

No LLM calls are made here. Human/model semantic labels belong to a later
evaluation milestone.
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

from qdrant_client import QdrantClient

from src.evaluation.claim_citation_evaluator import (
    evaluate_persisted_rag_row,
)
from src.indexing.qdrant_setup import (
    COLLECTION_NAME,
    QDRANT_URL,
)


DEFAULT_SOURCE_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_interactions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_claims.jsonl"
)

SOURCE_RUN_CONFIG_ID = (
    "production-rag-v2-interactions"
)

SEMANTIC_SCHEMA_VERSION = 1

DEFAULT_BATCH_SIZE = 64

SUPPORTED_SEMANTIC_LABELS = (
    "supported",
    "partially_supported",
    "unsupported",
    "not_a_factual_claim",
    "needs_review",
)

SUPPORTED_CITATION_REQUIREMENT_LABELS = (
    "required",
    "not_required",
    "unclear",
)


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Return a validated non-empty string."""

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


def load_rag_run_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load and validate the source production RAG benchmark."""

    source_path = Path(
        path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"RAG benchmark does not exist: "
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
                    "Invalid JSON in RAG benchmark "
                    f"on line {line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "RAG benchmark row "
                    f"{line_number} must be "
                    "a JSON object."
                )

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
                    "run_config_id"
                )
                != SOURCE_RUN_CONFIG_ID
            ):
                raise ValueError(
                    "RAG benchmark row "
                    f"{question_id!r} has "
                    "an unexpected run_config_id."
                )

            if (
                question_id
                in seen_question_ids
            ):
                raise ValueError(
                    "RAG benchmark contains "
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
            "RAG benchmark contains no rows."
        )

    return rows


def collect_selected_point_ids(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> tuple[
    str,
    ...,
]:
    """Collect unique selected Qdrant point IDs in first-appearance order."""

    seen: set[str] = set()
    ordered: list[str] = []

    for row in rows:
        question_id = (
            row.get(
                "question_id",
                "<unknown>",
            )
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
                "RAG benchmark row "
                f"{question_id!r} must contain "
                "selected_evidence as a list."
            )

        for evidence in (
            selected_evidence
        ):
            if not isinstance(
                evidence,
                Mapping,
            ):
                raise ValueError(
                    "selected_evidence entries "
                    "must be mappings."
                )

            point_id = (
                _require_nonempty_string(
                    evidence.get(
                        "point_id"
                    ),
                    field_name=(
                        "selected evidence point_id"
                    ),
                )
            )

            if point_id in seen:
                continue

            seen.add(
                point_id
            )

            ordered.append(
                point_id
            )

    return tuple(
        ordered
    )


def fetch_qdrant_payloads(
    client: Any,
    point_ids: Sequence[str],
    *,
    collection_name: str = (
        COLLECTION_NAME
    ),
    batch_size: int = (
        DEFAULT_BATCH_SIZE
    ),
) -> dict[
    str,
    Mapping[
        str,
        Any,
    ],
]:
    """Fetch exact Qdrant payloads for the selected production evidence."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be "
            "greater than zero."
        )

    payloads: dict[
        str,
        Mapping[
            str,
            Any,
        ],
    ] = {}

    for start in range(
        0,
        len(
            point_ids
        ),
        batch_size,
    ):
        batch = (
            point_ids[
                start:
                start + batch_size
            ]
        )

        records = (
            client.retrieve(
                collection_name=(
                    collection_name
                ),
                ids=list(
                    batch
                ),
                with_payload=True,
                with_vectors=False,
            )
        )

        for record in records:
            point_id = str(
                record.id
            )

            if point_id in payloads:
                raise ValueError(
                    "Qdrant returned duplicate "
                    f"point_id {point_id!r}."
                )

            payload = (
                record.payload
                or {}
            )

            if not isinstance(
                payload,
                Mapping,
            ):
                raise ValueError(
                    "Qdrant payload for "
                    f"{point_id!r} is invalid."
                )

            chunk_text = (
                payload.get(
                    "chunk_text"
                )
            )

            if not isinstance(
                chunk_text,
                str,
            ):
                raise ValueError(
                    "Qdrant payload for "
                    f"{point_id!r} contains no "
                    "chunk_text."
                )

            payloads[
                point_id
            ] = payload

    missing_ids = [
        point_id
        for point_id in point_ids
        if point_id
        not in payloads
    ]

    if missing_ids:
        raise ValueError(
            "Qdrant is missing selected "
            "benchmark point IDs: "
            + ", ".join(
                missing_ids
            )
        )

    return payloads


def validate_evidence_payload(
    persisted: Mapping[
        str,
        Any,
    ],
    payload: Mapping[
        str,
        Any,
    ],
) -> None:
    """Ensure the current Qdrant corpus matches persisted benchmark evidence."""

    point_id = (
        persisted.get(
            "point_id"
        )
    )

    fields = (
        "chunk_id",
        "document_id",
        "title",
        "organization",
        "language",
        "page_start",
        "page_end",
        "source_url",
    )

    for field_name in fields:
        persisted_value = (
            persisted.get(
                field_name
            )
        )

        payload_value = (
            payload.get(
                field_name
            )
        )

        if (
            persisted_value
            != payload_value
        ):
            raise ValueError(
                "Persisted benchmark evidence "
                f"{point_id!r} does not match "
                f"Qdrant field "
                f"{field_name!r}."
            )

    if not isinstance(
        payload.get(
            "chunk_text"
        ),
        str,
    ):
        raise ValueError(
            "Qdrant evidence "
            f"{point_id!r} has no "
            "original chunk_text."
        )


def _build_selected_evidence_index(
    row: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Mapping[
        str,
        Any,
    ],
]:
    """Map persisted evidence IDs to selected-evidence metadata."""

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
            "selected_evidence must "
            "be a list."
        )

    index: dict[
        str,
        Mapping[
            str,
            Any,
        ],
    ] = {}

    for item in selected_evidence:
        if not isinstance(
            item,
            Mapping,
        ):
            raise ValueError(
                "selected_evidence entries "
                "must be mappings."
            )

        evidence_id = (
            _require_nonempty_string(
                item.get(
                    "evidence_id"
                ),
                field_name=(
                    "evidence_id"
                ),
            )
        )

        if evidence_id in index:
            raise ValueError(
                "Duplicate selected "
                f"evidence_id "
                f"{evidence_id!r}."
            )

        index[
            evidence_id
        ] = item

    return index


def build_semantic_claim_rows(
    rag_row: Mapping[
        str,
        Any,
    ],
    *,
    payloads_by_point_id: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Build semantic-label-ready rows for one RAG benchmark question."""

    evaluation = (
        evaluate_persisted_rag_row(
            rag_row
        )
    )

    evidence_index = (
        _build_selected_evidence_index(
            rag_row
        )
    )

    question_id = (
        evaluation.question_id
    )

    output_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for claim in (
        evaluation.claims
    ):
        cited_passages: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for evidence_id in (
            claim.evidence_ids
        ):
            persisted_evidence = (
                evidence_index.get(
                    evidence_id
                )
            )

            if (
                persisted_evidence
                is None
            ):
                raise ValueError(
                    "Claim "
                    f"{question_id}:"
                    f"{claim.claim_index} "
                    "references unknown "
                    f"evidence_id "
                    f"{evidence_id!r}."
                )

            point_id = (
                _require_nonempty_string(
                    persisted_evidence.get(
                        "point_id"
                    ),
                    field_name=(
                        "point_id"
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
                    "No Qdrant payload was "
                    "loaded for point_id "
                    f"{point_id!r}."
                )

            validate_evidence_payload(
                persisted_evidence,
                payload,
            )

            cited_passages.append(
                {
                    "evidence_id": (
                        evidence_id
                    ),
                    "point_id": (
                        point_id
                    ),
                    "chunk_id": (
                        persisted_evidence[
                            "chunk_id"
                        ]
                    ),
                    "document_id": (
                        persisted_evidence[
                            "document_id"
                        ]
                    ),
                    "title": (
                        persisted_evidence[
                            "title"
                        ]
                    ),
                    "organization": (
                        persisted_evidence[
                            "organization"
                        ]
                    ),
                    "language": (
                        persisted_evidence[
                            "language"
                        ]
                    ),
                    "page_start": (
                        persisted_evidence[
                            "page_start"
                        ]
                    ),
                    "page_end": (
                        persisted_evidence[
                            "page_end"
                        ]
                    ),
                    "source_url": (
                        persisted_evidence[
                            "source_url"
                        ]
                    ),
                    "chunk_text": (
                        payload[
                            "chunk_text"
                        ]
                    ),
                    # These are intentionally unlabeled.
                    "individual_support_label": (
                        None
                    ),
                    "individual_support_notes": (
                        None
                    ),
                }
            )

        output_rows.append(
            {
                "schema_version": (
                    SEMANTIC_SCHEMA_VERSION
                ),
                "source_run_config_id": (
                    rag_row.get(
                        "run_config_id"
                    )
                ),
                "source_schema_version": (
                    rag_row.get(
                        "schema_version"
                    )
                ),
                "question_id": (
                    question_id
                ),
                "query": (
                    rag_row.get(
                        "query"
                    )
                ),
                "query_language": (
                    rag_row.get(
                        "query_language"
                    )
                ),
                "target_language": (
                    rag_row.get(
                        "target_language"
                    )
                ),
                "answer_language": (
                    rag_row.get(
                        "answer_language"
                    )
                ),
                "category": (
                    rag_row.get(
                        "category"
                    )
                ),
                "provider": (
                    rag_row.get(
                        "provider"
                    )
                ),
                "model": (
                    rag_row.get(
                        "model"
                    )
                ),
                "claim_id": (
                    f"{question_id}_c"
                    f"{claim.claim_index:03d}"
                ),
                "claim_index": (
                    claim.claim_index
                ),
                "claim_text": (
                    claim.claim_text
                ),
                "raw_text": (
                    claim.raw_text
                ),
                "has_citation": (
                    claim.has_citation
                ),
                "evidence_ids": list(
                    claim.evidence_ids
                ),
                "cited_evidence": (
                    cited_passages
                ),
                # Joint semantic support from all cited evidence.
                "semantic_support_label": (
                    None
                ),
                # Whether this claim should require a citation at all.
                "citation_requirement_label": (
                    None
                ),
                "semantic_notes": (
                    None
                ),
            }
        )

    return output_rows


def write_jsonl_atomic(
    path: str | Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Write JSONL through a temporary file and atomically replace output."""

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


def build_semantic_evaluation_dataset(
    source_path: str | Path = (
        DEFAULT_SOURCE_PATH
    ),
    *,
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    client: Any | None = None,
    collection_name: str = (
        COLLECTION_NAME
    ),
    batch_size: int = (
        DEFAULT_BATCH_SIZE
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Materialize exact evidence text for every extracted benchmark claim."""

    rag_rows = (
        load_rag_run_rows(
            source_path
        )
    )

    point_ids = (
        collect_selected_point_ids(
            rag_rows
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
                batch_size=(
                    batch_size
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

    semantic_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for rag_row in rag_rows:
        semantic_rows.extend(
            build_semantic_claim_rows(
                rag_row,
                payloads_by_point_id=(
                    payloads
                ),
            )
        )

    if not semantic_rows:
        raise ValueError(
            "No claim rows were produced."
        )

    write_jsonl_atomic(
        output_path,
        semantic_rows,
    )

    return semantic_rows


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the semantic-dataset command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build claim-level semantic "
            "citation evaluation data "
            "from the production RAG run."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=(
            DEFAULT_SOURCE_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            DEFAULT_BATCH_SIZE
        ),
    )

    return parser


def main() -> None:
    """Materialize the semantic citation dataset."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    rows = (
        build_semantic_evaluation_dataset(
            source_path=(
                args.source
            ),
            output_path=(
                args.output
            ),
            batch_size=(
                args.batch_size
            ),
        )
    )

    cited_claim_count = sum(
        bool(
            row[
                "has_citation"
            ]
        )
        for row in rows
    )

    citation_assignment_count = sum(
        len(
            row[
                "cited_evidence"
            ]
        )
        for row in rows
    )

    question_count = len(
        {
            row[
                "question_id"
            ]
            for row in rows
        }
    )

    print(
        "Semantic citation dataset"
    )

    print(
        f"Questions: "
        f"{question_count}"
    )

    print(
        f"Claims: "
        f"{len(rows)}"
    )

    print(
        f"Cited claims: "
        f"{cited_claim_count}"
    )

    print(
        "Citation assignments: "
        f"{citation_assignment_count}"
    )

    print(
        f"Output: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()