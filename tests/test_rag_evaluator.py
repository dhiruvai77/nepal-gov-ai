"""Tests for deterministic end-to-end RAG evaluation metrics."""

import pytest

from src.citations.evidence import (
    process_answer_citations,
)
from src.evaluation.rag_evaluator import (
    aggregate_rag_metrics,
    evaluate_rag_dataset,
    evaluate_rag_result,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)
from src.rag.pipeline import (
    RAGResult,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


PRIMARY_ID = "primary-chunk"
SECONDARY_ID = "secondary-chunk"
IRRELEVANT_ID = "irrelevant-chunk"


def make_record() -> EvaluationRecord:
    """Create one answerable evaluation record."""

    return EvaluationRecord(
        question_id="q1",
        query="What does the evidence say?",
        query_language="en",
        target_language="en",
        category="test",
        expected_document_ids=(
            "document-primary",
        ),
        primary_relevant_chunk_ids=(
            PRIMARY_ID,
        ),
        relevant_chunk_ids=(
            PRIMARY_ID,
            SECONDARY_ID,
        ),
        notes="Test fixture.",
    )


def make_evidence(
    point_id: str,
    *,
    index: int,
) -> RerankedResult:
    """Create deterministic selected evidence."""

    result = RetrievalResult(
        point_id=point_id,
        score=0.1,
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        title=f"Government Document {index}",
        organization="Government of Nepal",
        language="en",
        page_start=index,
        page_end=index,
        source_url=(
            f"https://example.gov.np/"
            f"document-{index}.pdf"
        ),
        chunk_text=(
            f"Evidence passage {index}."
        ),
        chunk_index=index,
        token_count=200,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.9,
        original_rank=index,
    )


def make_result(
    *,
    selected_context: tuple[
        RerankedResult,
        ...
    ],
    answer_text: str,
    accepted: bool = True,
) -> RAGResult:
    """Create a RAG result with real citation processing."""

    citation_result = (
        process_answer_citations(
            answer_text,
            selected_context,
        )
        if selected_context
        else None
    )

    sources = (
        tuple(
            citation.evidence_id
            for citation
            in citation_result.citations
        )
        if (
            accepted
            and citation_result
            is not None
        )
        else ()
    )

    return RAGResult(
        answer_text=answer_text,
        accepted=accepted,
        reason=None,
        sources=sources,
        selected_context=(
            selected_context
        ),
        citation_result=(
            citation_result
        ),
        provider="fake",
        model="fake-model",
    )


def test_primary_selected_and_cited_scores_full_hits() -> None:
    """Primary evidence in both context and citations should score as hits."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    result = make_result(
        selected_context=(
            primary,
        ),
        answer_text=(
            "Supported statement [E1]."
        ),
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.selected_primary_hit
        == 1.0
    )

    assert (
        metrics.cited_primary_hit
        == 1.0
    )

    assert (
        metrics.cited_relevant_precision
        == 1.0
    )


def test_selected_relevant_recall_uses_all_gold_evidence() -> None:
    """Context coverage should use the broader manually verified evidence set."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    result = make_result(
        selected_context=(
            primary,
        ),
        answer_text="Answer [E1].",
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.selected_relevant_recall
        == 0.5
    )


def test_cited_relevant_recall_uses_all_gold_evidence() -> None:
    """Cited evidence recall should measure broader gold-evidence coverage."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    secondary = make_evidence(
        SECONDARY_ID,
        index=2,
    )

    result = make_result(
        selected_context=(
            primary,
            secondary,
        ),
        answer_text="Answer [E1].",
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.cited_relevant_recall
        == 0.5
    )


def test_citing_all_relevant_evidence_has_full_recall() -> None:
    """Citing all manually relevant passages should produce full recall."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    secondary = make_evidence(
        SECONDARY_ID,
        index=2,
    )

    result = make_result(
        selected_context=(
            primary,
            secondary,
        ),
        answer_text=(
            "First [E1]. "
            "Second [E2]."
        ),
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.cited_relevant_recall
        == 1.0
    )

    assert (
        metrics.cited_relevant_precision
        == 1.0
    )


def test_irrelevant_citation_reduces_precision() -> None:
    """Citing selected but non-gold evidence should reduce evidence precision."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    irrelevant = make_evidence(
        IRRELEVANT_ID,
        index=2,
    )

    result = make_result(
        selected_context=(
            primary,
            irrelevant,
        ),
        answer_text=(
            "Relevant [E1]. "
            "Irrelevant [E2]."
        ),
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.cited_relevant_precision
        == 0.5
    )


def test_secondary_relevant_citation_is_not_primary_hit() -> None:
    """Broader useful evidence should not be mistaken for primary evidence."""

    secondary = make_evidence(
        SECONDARY_ID,
        index=1,
    )

    result = make_result(
        selected_context=(
            secondary,
        ),
        answer_text="Answer [E1].",
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.cited_primary_hit
        == 0.0
    )

    assert (
        metrics.cited_relevant_precision
        == 1.0
    )


def test_invalid_reference_reduces_valid_reference_ratio() -> None:
    """Invented evidence IDs should be reflected in structural citation quality."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    result = make_result(
        selected_context=(
            primary,
        ),
        answer_text=(
            "Valid [E1]. "
            "Invented [E99]."
        ),
        accepted=False,
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.valid_citation_count
        == 1
    )

    assert (
        metrics.invalid_citation_count
        == 1
    )

    assert (
        metrics.valid_reference_ratio
        == 0.5
    )


def test_uncited_answer_has_zero_reference_metrics() -> None:
    """An answer without citations should not receive citation credit."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    result = make_result(
        selected_context=(
            primary,
        ),
        answer_text=(
            "Answer without citation."
        ),
        accepted=False,
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.valid_citation_count
        == 0
    )

    assert (
        metrics.invalid_citation_count
        == 0
    )

    assert (
        metrics.valid_reference_ratio
        == 0.0
    )

    assert (
        metrics.cited_relevant_precision
        == 0.0
    )


def test_no_evidence_result_has_zero_coverage() -> None:
    """No-evidence withholding should produce zero evidence metrics."""

    result = RAGResult(
        answer_text=(
            "Insufficient evidence."
        ),
        accepted=False,
        reason=None,
        sources=(),
        selected_context=(),
        citation_result=None,
        provider=None,
        model=None,
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.selected_primary_hit
        == 0.0
    )

    assert (
        metrics.selected_relevant_recall
        == 0.0
    )

    assert (
        metrics.cited_primary_hit
        == 0.0
    )

    assert (
        metrics.cited_relevant_recall
        == 0.0
    )


def test_acceptance_and_withholding_are_complements() -> None:
    """Application acceptance should be represented explicitly."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    accepted_result = (
        make_result(
            selected_context=(
                primary,
            ),
            answer_text="Answer [E1].",
            accepted=True,
        )
    )

    withheld_result = (
        make_result(
            selected_context=(
                primary,
            ),
            answer_text="No citation.",
            accepted=False,
        )
    )

    accepted_metrics = (
        evaluate_rag_result(
            make_record(),
            accepted_result,
        )
    )

    withheld_metrics = (
        evaluate_rag_result(
            make_record(),
            withheld_result,
        )
    )

    assert (
        accepted_metrics.accepted
        == 1.0
    )

    assert (
        accepted_metrics.withheld
        == 0.0
    )

    assert (
        withheld_metrics.accepted
        == 0.0
    )

    assert (
        withheld_metrics.withheld
        == 1.0
    )


def test_evaluator_uses_point_ids_matching_existing_gold_contract() -> None:
    """RAG evaluation must align with the retrieval benchmark's point IDs."""

    evidence = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    assert (
        evidence.result.chunk_id
        != PRIMARY_ID
    )

    result = make_result(
        selected_context=(
            evidence,
        ),
        answer_text="Answer [E1].",
    )

    metrics = evaluate_rag_result(
        make_record(),
        result,
    )

    assert (
        metrics.selected_primary_hit
        == 1.0
    )


def test_aggregate_rag_metrics_macro_averages_queries() -> None:
    """Aggregate metrics should average per-query results consistently."""

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    good = evaluate_rag_result(
        make_record(),
        make_result(
            selected_context=(
                primary,
            ),
            answer_text="Answer [E1].",
        ),
    )

    bad = evaluate_rag_result(
        make_record(),
        make_result(
            selected_context=(
                primary,
            ),
            answer_text="No citation.",
            accepted=False,
        ),
    )

    summary = aggregate_rag_metrics(
        [
            good,
            bad,
        ]
    )

    assert (
        summary[
            "acceptance_rate"
        ]
        == 0.5
    )

    assert (
        summary[
            "withholding_rate"
        ]
        == 0.5
    )

    assert (
        summary[
            "cited_primary_hit_rate"
        ]
        == 0.5
    )


def test_aggregate_rag_metrics_rejects_empty_input() -> None:
    """A benchmark summary requires at least one evaluated question."""

    with pytest.raises(
        ValueError,
        match=(
            "At least one RAG query metric "
            "is required"
        ),
    ):
        aggregate_rag_metrics(
            []
        )


def test_evaluate_rag_dataset_runs_all_records() -> None:
    """Dataset evaluation should invoke the answer function once per record."""

    first = make_record()

    second = EvaluationRecord(
        question_id="q2",
        query="Second question",
        query_language="en",
        target_language="en",
        category="test",
        expected_document_ids=(
            "document-primary",
        ),
        primary_relevant_chunk_ids=(
            PRIMARY_ID,
        ),
        relevant_chunk_ids=(
            PRIMARY_ID,
            SECONDARY_ID,
        ),
        notes="Second fixture.",
    )

    primary = make_evidence(
        PRIMARY_ID,
        index=1,
    )

    calls: list[str] = []

    def answer(
        record: EvaluationRecord,
    ) -> RAGResult:
        calls.append(
            record.question_id
        )

        return make_result(
            selected_context=(
                primary,
            ),
            answer_text="Answer [E1].",
        )

    per_query, summary = (
        evaluate_rag_dataset(
            [
                first,
                second,
            ],
            answer,
        )
    )

    assert calls == [
        "q1",
        "q2",
    ]

    assert len(
        per_query
    ) == 2

    assert (
        summary[
            "acceptance_rate"
        ]
        == 1.0
    )


def test_evaluate_rag_dataset_rejects_empty_records() -> None:
    """Dataset evaluation should fail clearly for an empty benchmark."""

    with pytest.raises(
        ValueError,
        match=(
            "At least one evaluation record "
            "is required"
        ),
    ):
        evaluate_rag_dataset(
            [],
            lambda record: None,
        )