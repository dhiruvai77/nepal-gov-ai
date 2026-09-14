from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from src.retrieval.dense_retriever import RetrievalResult


@dataclass(frozen=True)
class EvaluationRecord:
    """One manually curated retrieval-evaluation example."""

    question_id: str
    query: str
    query_language: str
    target_language: str
    category: str
    expected_document_ids: tuple[str, ...]
    primary_relevant_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class QueryMetrics:
    """Per-query retrieval metrics at one cutoff k."""

    question_id: str
    hit_rate: float
    reciprocal_rank: float
    recall: float


def load_evaluation_records(path: str | Path) -> list[EvaluationRecord]:
    """Load and validate JSONL retrieval-evaluation records."""

    records: list[EvaluationRecord] = []
    seen_question_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            # Blank lines are ignored so manual editing does not break the dataset.
            if not line:
                continue

            data = json.loads(line)

            question_id = data["question_id"]
            if question_id in seen_question_ids:
                raise ValueError(
                    f"Duplicate question_id {question_id!r} on line {line_number}."
                )
            seen_question_ids.add(question_id)

            primary_ids = tuple(data["primary_relevant_chunk_ids"])
            relevant_ids = tuple(data["relevant_chunk_ids"])

            # Primary evidence must always be part of the full relevant set.
            if not set(primary_ids).issubset(relevant_ids):
                raise ValueError(
                    f"{question_id}: primary_relevant_chunk_ids must be a subset "
                    "of relevant_chunk_ids."
                )

            if not relevant_ids:
                raise ValueError(f"{question_id}: relevant_chunk_ids cannot be empty.")

            records.append(
                EvaluationRecord(
                    question_id=question_id,
                    query=data["query"],
                    query_language=data["query_language"],
                    target_language=data["target_language"],
                    category=data["category"],
                    expected_document_ids=tuple(data["expected_document_ids"]),
                    primary_relevant_chunk_ids=primary_ids,
                    relevant_chunk_ids=relevant_ids,
                    notes=data["notes"],
                )
            )

    return records


def evaluate_ranked_results(
    record: EvaluationRecord,
    results: Sequence[RetrievalResult],
    *,
    k: int,
) -> QueryMetrics:
    """Calculate Hit Rate@k, MRR@k, and Recall@k for one query."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_results = results[:k]
    retrieved_ids = [result.point_id for result in top_results]

    primary_ids = set(record.primary_relevant_chunk_ids)
    relevant_ids = set(record.relevant_chunk_ids)

    # Hit Rate uses the strongest, primary evidence set.
    hit_rate = float(any(chunk_id in primary_ids for chunk_id in retrieved_ids))

    # MRR is based on the rank of the first primary-relevant result.
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in primary_ids:
            reciprocal_rank = 1.0 / rank
            break

    # Recall measures how much of the broader manually verified evidence set
    # was recovered within the requested cutoff.
    retrieved_relevant = sum(
        1 for chunk_id in set(retrieved_ids) if chunk_id in relevant_ids
    )
    recall = retrieved_relevant / len(relevant_ids)

    return QueryMetrics(
        question_id=record.question_id,
        hit_rate=hit_rate,
        reciprocal_rank=reciprocal_rank,
        recall=recall,
    )


def aggregate_metrics(metrics: Iterable[QueryMetrics]) -> dict[str, float]:
    """Average per-query metrics into one evaluation summary."""

    metric_list = list(metrics)
    if not metric_list:
        raise ValueError("At least one query metric is required.")

    count = len(metric_list)

    return {
        "hit_rate": sum(item.hit_rate for item in metric_list) / count,
        "mrr": sum(item.reciprocal_rank for item in metric_list) / count,
        "recall": sum(item.recall for item in metric_list) / count,
    }


def evaluate_dataset(
    records: Sequence[EvaluationRecord],
    retrieve: Callable[[EvaluationRecord, int], Sequence[RetrievalResult]],
    *,
    k: int,
) -> tuple[list[QueryMetrics], dict[str, float]]:
    """Run a retriever over all records and return per-query and aggregate metrics."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    per_query: list[QueryMetrics] = []

    for record in records:
        results = retrieve(record, k)
        per_query.append(evaluate_ranked_results(record, results, k=k))

    return per_query, aggregate_metrics(per_query)