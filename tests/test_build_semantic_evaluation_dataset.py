"""Tests for semantic citation dataset materialization."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.evaluation.build_semantic_evaluation_dataset import (
    SOURCE_RUN_CONFIG_ID,
    build_semantic_claim_rows,
    build_semantic_evaluation_dataset,
    collect_selected_point_ids,
    fetch_qdrant_payloads,
    load_rag_run_rows,
    validate_evidence_payload,
)


def make_selected_evidence(
    *,
    evidence_id: str = "E1",
    point_id: str = "point-1",
) -> dict:
    """Create persisted selected-evidence metadata."""

    return {
        "evidence_id": evidence_id,
        "point_id": point_id,
        "chunk_id": f"chunk-{point_id}",
        "document_id": "document-1",
        "title": "Government Document",
        "organization": "Government of Nepal",
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "source_url": "https://example.gov.np/document",
    }


def make_payload(
    *,
    point_id: str = "point-1",
    chunk_text: str = "Exact government evidence passage.",
) -> dict:
    """Create one matching Qdrant payload."""

    return {
        "chunk_id": f"chunk-{point_id}",
        "document_id": "document-1",
        "title": "Government Document",
        "organization": "Government of Nepal",
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "source_url": "https://example.gov.np/document",
        "chunk_text": chunk_text,
    }


def make_rag_row(
    *,
    question_id: str = "q1",
    answer: str = "Education is protected [E1].",
    selected_evidence: list[dict] | None = None,
) -> dict:
    """Create one production-style persisted RAG row."""

    return {
        "schema_version": 1,
        "run_config_id": SOURCE_RUN_CONFIG_ID,
        "question_id": question_id,
        "query": "What right is protected?",
        "query_language": "en",
        "target_language": "en",
        "answer_language": "en",
        "category": "test",
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "generated_answer_text": answer,
        "answer_text": answer,
        "selected_evidence": (
            selected_evidence
            if selected_evidence is not None
            else [
                make_selected_evidence()
            ]
        ),
    }


class FakeQdrantClient:
    """Minimal Qdrant retrieve client for deterministic tests."""

    def __init__(
        self,
        payloads: dict[
            str,
            dict,
        ],
    ) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    def retrieve(
        self,
        *,
        collection_name,
        ids,
        with_payload,
        with_vectors,
    ):
        self.calls.append(
            {
                "collection_name": collection_name,
                "ids": ids,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )

        return [
            SimpleNamespace(
                id=point_id,
                payload=(
                    self.payloads[
                        point_id
                    ]
                ),
            )
            for point_id in ids
            if point_id
            in self.payloads
        ]


def write_rows(
    path,
    rows,
) -> None:
    """Write test JSONL rows."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            handle.write(
                "\n"
            )


def test_loads_valid_production_rows(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "run.jsonl"
    )

    write_rows(
        path,
        [
            make_rag_row(),
        ],
    )

    rows = load_rag_run_rows(
        path
    )

    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"


def test_rejects_wrong_source_run_config(
    tmp_path,
) -> None:
    row = make_rag_row()
    row["run_config_id"] = "old-run"

    path = (
        tmp_path
        / "run.jsonl"
    )

    write_rows(
        path,
        [
            row,
        ],
    )

    with pytest.raises(
        ValueError,
        match="unexpected run_config_id",
    ):
        load_rag_run_rows(
            path
        )


def test_rejects_duplicate_question_ids(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "run.jsonl"
    )

    write_rows(
        path,
        [
            make_rag_row(),
            make_rag_row(),
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicate question_id",
    ):
        load_rag_run_rows(
            path
        )


def test_collect_point_ids_preserves_order_and_deduplicates() -> None:
    first = make_rag_row(
        selected_evidence=[
            make_selected_evidence(
                evidence_id="E1",
                point_id="point-1",
            ),
            make_selected_evidence(
                evidence_id="E2",
                point_id="point-2",
            ),
        ]
    )

    second = make_rag_row(
        question_id="q2",
        selected_evidence=[
            make_selected_evidence(
                evidence_id="E1",
                point_id="point-2",
            ),
            make_selected_evidence(
                evidence_id="E2",
                point_id="point-3",
            ),
        ],
    )

    assert collect_selected_point_ids(
        [
            first,
            second,
        ]
    ) == (
        "point-1",
        "point-2",
        "point-3",
    )


def test_fetches_payloads_without_vectors() -> None:
    client = FakeQdrantClient(
        {
            "point-1": make_payload(),
        }
    )

    payloads = fetch_qdrant_payloads(
        client,
        [
            "point-1",
        ],
        collection_name="test-collection",
    )

    assert (
        payloads[
            "point-1"
        ][
            "chunk_text"
        ]
        == "Exact government evidence passage."
    )

    assert client.calls == [
        {
            "collection_name": "test-collection",
            "ids": [
                "point-1",
            ],
            "with_payload": True,
            "with_vectors": False,
        }
    ]


def test_fetch_rejects_missing_qdrant_points() -> None:
    client = FakeQdrantClient(
        {}
    )

    with pytest.raises(
        ValueError,
        match="missing selected benchmark point IDs",
    ):
        fetch_qdrant_payloads(
            client,
            [
                "point-1",
            ],
        )


def test_fetch_rejects_non_positive_batch_size() -> None:
    client = FakeQdrantClient(
        {}
    )

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        fetch_qdrant_payloads(
            client,
            [],
            batch_size=0,
        )


def test_evidence_payload_validation_detects_stale_corpus() -> None:
    evidence = (
        make_selected_evidence()
    )

    payload = (
        make_payload()
    )

    payload[
        "document_id"
    ] = "changed-document"

    with pytest.raises(
        ValueError,
        match="does not match Qdrant field",
    ):
        validate_evidence_payload(
            evidence,
            payload,
        )


def test_claim_row_attaches_exact_original_passage() -> None:
    rows = (
        build_semantic_claim_rows(
            make_rag_row(),
            payloads_by_point_id={
                "point-1": make_payload(
                    chunk_text=(
                        "Canonical original passage."
                    )
                ),
            },
        )
    )

    assert len(rows) == 1

    assert (
        rows[0][
            "claim_id"
        ]
        == "q1_c001"
    )

    assert (
        rows[0][
            "cited_evidence"
        ][0][
            "chunk_text"
        ]
        == "Canonical original passage."
    )

    assert (
        rows[0][
            "semantic_support_label"
        ]
        is None
    )

    assert (
        rows[0][
            "cited_evidence"
        ][0][
            "individual_support_label"
        ]
        is None
    )


def test_multiple_citations_keep_claim_citation_order() -> None:
    first = make_selected_evidence(
        evidence_id="E1",
        point_id="point-1",
    )

    second = make_selected_evidence(
        evidence_id="E2",
        point_id="point-2",
    )

    row = make_rag_row(
        answer=(
            "The claim has joint support "
            "[E2], [E1]."
        ),
        selected_evidence=[
            first,
            second,
        ],
    )

    result = (
        build_semantic_claim_rows(
            row,
            payloads_by_point_id={
                "point-1": make_payload(
                    point_id="point-1",
                ),
                "point-2": make_payload(
                    point_id="point-2",
                ),
            },
        )
    )

    assert result[0]["evidence_ids"] == [
        "E2",
        "E1",
    ]

    assert [
        evidence[
            "evidence_id"
        ]
        for evidence
        in result[0][
            "cited_evidence"
        ]
    ] == [
        "E2",
        "E1",
    ]


def test_uncited_claim_is_retained_for_completeness_review() -> None:
    row = make_rag_row(
        answer=(
            "According to the Act, "
            "the following applies:"
        )
    )

    result = (
        build_semantic_claim_rows(
            row,
            payloads_by_point_id={
                "point-1": make_payload(),
            },
        )
    )

    assert len(result) == 1
    assert result[0]["has_citation"] is False
    assert result[0]["evidence_ids"] == []
    assert result[0]["cited_evidence"] == []


def test_unknown_claim_evidence_id_is_rejected() -> None:
    row = make_rag_row(
        answer="Unsupported reference [E9]."
    )

    with pytest.raises(
        ValueError,
        match="references unknown evidence_id",
    ):
        build_semantic_claim_rows(
            row,
            payloads_by_point_id={
                "point-1": make_payload(),
            },
        )


def test_end_to_end_materialization_writes_claim_jsonl(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "run.jsonl"
    )

    output_path = (
        tmp_path
        / "semantic.jsonl"
    )

    write_rows(
        source_path,
        [
            make_rag_row(
                answer=(
                    "Supported claim [E1]. "
                    "Uncited introduction."
                )
            ),
        ],
    )

    client = FakeQdrantClient(
        {
            "point-1": make_payload(),
        }
    )

    rows = (
        build_semantic_evaluation_dataset(
            source_path,
            output_path=output_path,
            client=client,
        )
    )

    assert len(rows) == 2
    assert output_path.exists()

    persisted = [
        json.loads(
            line
        )
        for line in (
            output_path.read_text(
                encoding="utf-8",
            )
            .splitlines()
        )
        if line.strip()
    ]

    assert len(persisted) == 2
    assert (
        persisted[0][
            "question_id"
        ]
        == "q1"
    )
    assert (
        persisted[0][
            "cited_evidence"
        ][0][
            "chunk_text"
        ]
        == "Exact government evidence passage."
    )