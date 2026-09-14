import json

import pytest

from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    aggregate_metrics,
    evaluate_dataset,
    evaluate_ranked_results,
    load_evaluation_records,
)
from src.retrieval.dense_retriever import RetrievalResult


def make_result(point_id: str) -> RetrievalResult:
    """Create a minimal RetrievalResult compatible with the production dataclass."""

    return RetrievalResult(
        point_id=point_id,
        score=1.0,
        chunk_id=point_id,
        document_id="doc",
        title="Title",
        organization="Organization",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.com",
        chunk_text="Example text",
        category="education",
        document_type="report",
        publication_date=None,
        section=None,
        subsection=None,
        article_number=None,
        article_title=None,
        extraction_method="native",
    )


@pytest.fixture
def record() -> EvaluationRecord:
    """Return one record with a primary result and two relevant chunks."""

    return EvaluationRecord(
        question_id="q1",
        query="Example question?",
        query_language="en",
        target_language="en",
        category="education",
        expected_document_ids=("doc",),
        primary_relevant_chunk_ids=("primary",),
        relevant_chunk_ids=("primary", "supporting"),
        notes="Test record.",
    )


def test_evaluate_ranked_results_calculates_metrics(record):
    results = [
        make_result("irrelevant"),
        make_result("primary"),
        make_result("supporting"),
    ]

    metrics = evaluate_ranked_results(record, results, k=3)

    assert metrics.hit_rate == 1.0
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert metrics.recall == 1.0


def test_evaluate_ranked_results_respects_cutoff(record):
    results = [
        make_result("irrelevant"),
        make_result("primary"),
        make_result("supporting"),
    ]

    metrics = evaluate_ranked_results(record, results, k=1)

    assert metrics.hit_rate == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.recall == 0.0


def test_aggregate_metrics_returns_means():
    from src.evaluation.retrieval_evaluator import QueryMetrics

    metrics = [
        QueryMetrics("q1", 1.0, 1.0, 0.5),
        QueryMetrics("q2", 0.0, 0.0, 1.0),
    ]

    summary = aggregate_metrics(metrics)

    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.75)


def test_load_evaluation_records_reads_valid_jsonl(tmp_path):
    path = tmp_path / "questions.jsonl"
    row = {
        "question_id": "q1",
        "query": "Question?",
        "query_language": "en",
        "target_language": "en",
        "category": "education",
        "expected_document_ids": ["doc"],
        "primary_relevant_chunk_ids": ["chunk-1"],
        "relevant_chunk_ids": ["chunk-1", "chunk-2"],
        "notes": "Verified evidence.",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    records = load_evaluation_records(path)

    assert len(records) == 1
    assert records[0].question_id == "q1"
    assert records[0].primary_relevant_chunk_ids == ("chunk-1",)


def test_load_evaluation_records_rejects_invalid_primary_subset(tmp_path):
    path = tmp_path / "questions.jsonl"
    row = {
        "question_id": "q1",
        "query": "Question?",
        "query_language": "en",
        "target_language": "en",
        "category": "education",
        "expected_document_ids": ["doc"],
        "primary_relevant_chunk_ids": ["missing"],
        "relevant_chunk_ids": ["chunk-1"],
        "notes": "Invalid example.",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="subset"):
        load_evaluation_records(path)


def test_evaluate_dataset_runs_retriever_for_each_record(record):
    calls = []

    def retrieve(current_record, k):
        # Record the call so the test verifies orchestration as well as metrics.
        calls.append((current_record.question_id, k))
        return [make_result("primary")]

    per_query, summary = evaluate_dataset([record], retrieve, k=5)

    assert calls == [("q1", 5)]
    assert per_query[0].hit_rate == 1.0
    assert summary["mrr"] == 1.0