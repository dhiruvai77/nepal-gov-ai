"""Deterministic end-to-end RAG evaluation metrics for NepalGov AI.

The evaluator reuses the manually verified retrieval benchmark and compares
selected/cited evidence against its gold chunk annotations.

These metrics measure evidence coverage and citation structure. They do not
claim to measure semantic entailment between every generated claim and its
citation. Claim-level faithfulness requires a separate evaluation layer.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Sequence,
)
from dataclasses import dataclass

from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)
from src.rag.pipeline import (
    RAGResult,
)


@dataclass(frozen=True)
class RAGQueryMetrics:
    """Deterministic metrics for one end-to-end RAG evaluation question."""

    question_id: str

    accepted: float
    withheld: float

    selected_primary_hit: float
    selected_relevant_recall: float

    cited_primary_hit: float
    cited_relevant_precision: float
    cited_relevant_recall: float

    valid_reference_ratio: float

    selected_context_count: int
    valid_citation_count: int
    invalid_citation_count: int
    rendered_source_count: int


def evaluate_rag_result(
    record: EvaluationRecord,
    result: RAGResult,
) -> RAGQueryMetrics:
    """Evaluate one RAG result against manually verified evidence annotations."""

    primary_ids = set(
        record.primary_relevant_chunk_ids
    )

    relevant_ids = set(
        record.relevant_chunk_ids
    )

    # The existing retrieval benchmark uses Qdrant point_id values as its gold
    # chunk identifiers, so end-to-end evaluation must use the same identity.
    selected_ids = {
        item.result.point_id
        for item
        in result.selected_context
    }

    selected_primary_hit = float(
        bool(
            selected_ids
            & primary_ids
        )
    )

    selected_relevant_recall = (
        len(
            selected_ids
            & relevant_ids
        )
        / len(
            relevant_ids
        )
    )

    citation_result = (
        result.citation_result
    )

    if citation_result is None:
        cited_ids: set[str] = set()
        valid_citation_count = 0
        invalid_citation_count = 0

    else:
        cited_ids = {
            citation.evidence.result.point_id
            for citation
            in citation_result.citations
        }

        valid_citation_count = len(
            citation_result.citations
        )

        invalid_citation_count = len(
            citation_result.invalid_evidence_ids
        )

    cited_primary_hit = float(
        bool(
            cited_ids
            & primary_ids
        )
    )

    if cited_ids:
        cited_relevant_precision = (
            len(
                cited_ids
                & relevant_ids
            )
            / len(
                cited_ids
            )
        )

    else:
        cited_relevant_precision = 0.0

    cited_relevant_recall = (
        len(
            cited_ids
            & relevant_ids
        )
        / len(
            relevant_ids
        )
    )

    total_reference_count = (
        valid_citation_count
        + invalid_citation_count
    )

    if total_reference_count:
        valid_reference_ratio = (
            valid_citation_count
            / total_reference_count
        )

    else:
        valid_reference_ratio = 0.0

    return RAGQueryMetrics(
        question_id=(
            record.question_id
        ),
        accepted=float(
            result.accepted
        ),
        withheld=float(
            result.withheld
        ),
        selected_primary_hit=(
            selected_primary_hit
        ),
        selected_relevant_recall=(
            selected_relevant_recall
        ),
        cited_primary_hit=(
            cited_primary_hit
        ),
        cited_relevant_precision=(
            cited_relevant_precision
        ),
        cited_relevant_recall=(
            cited_relevant_recall
        ),
        valid_reference_ratio=(
            valid_reference_ratio
        ),
        selected_context_count=len(
            result.selected_context
        ),
        valid_citation_count=(
            valid_citation_count
        ),
        invalid_citation_count=(
            invalid_citation_count
        ),
        rendered_source_count=len(
            result.sources
        ),
    )


def aggregate_rag_metrics(
    metrics: Iterable[
        RAGQueryMetrics
    ],
) -> dict[
    str,
    float,
]:
    """Macro-average deterministic end-to-end RAG metrics."""

    metric_list = list(
        metrics
    )

    if not metric_list:
        raise ValueError(
            "At least one RAG query metric is required."
        )

    count = len(
        metric_list
    )

    return {
        "acceptance_rate": (
            sum(
                item.accepted
                for item
                in metric_list
            )
            / count
        ),
        "withholding_rate": (
            sum(
                item.withheld
                for item
                in metric_list
            )
            / count
        ),
        "selected_primary_hit_rate": (
            sum(
                item.selected_primary_hit
                for item
                in metric_list
            )
            / count
        ),
        "selected_relevant_recall": (
            sum(
                item.selected_relevant_recall
                for item
                in metric_list
            )
            / count
        ),
        "cited_primary_hit_rate": (
            sum(
                item.cited_primary_hit
                for item
                in metric_list
            )
            / count
        ),
        "cited_relevant_precision": (
            sum(
                item.cited_relevant_precision
                for item
                in metric_list
            )
            / count
        ),
        "cited_relevant_recall": (
            sum(
                item.cited_relevant_recall
                for item
                in metric_list
            )
            / count
        ),
        "valid_reference_ratio": (
            sum(
                item.valid_reference_ratio
                for item
                in metric_list
            )
            / count
        ),
        "avg_selected_context": (
            sum(
                item.selected_context_count
                for item
                in metric_list
            )
            / count
        ),
        "avg_valid_citations": (
            sum(
                item.valid_citation_count
                for item
                in metric_list
            )
            / count
        ),
        "avg_invalid_citations": (
            sum(
                item.invalid_citation_count
                for item
                in metric_list
            )
            / count
        ),
        "avg_rendered_sources": (
            sum(
                item.rendered_source_count
                for item
                in metric_list
            )
            / count
        ),
    }


def evaluate_rag_dataset(
    records: Sequence[
        EvaluationRecord
    ],
    answer: Callable[
        [
            EvaluationRecord,
        ],
        RAGResult,
    ],
) -> tuple[
    list[
        RAGQueryMetrics
    ],
    dict[
        str,
        float,
    ],
]:
    """Evaluate an answer function over the complete annotated dataset."""

    if not records:
        raise ValueError(
            "At least one evaluation record is required."
        )

    per_query: list[
        RAGQueryMetrics
    ] = []

    for record in records:
        result = answer(
            record
        )

        per_query.append(
            evaluate_rag_result(
                record,
                result,
            )
        )

    return (
        per_query,
        aggregate_rag_metrics(
            per_query
        ),
    )