from __future__ import annotations

from pathlib import Path

from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    evaluate_dataset,
    load_evaluation_records,
)
from src.retrieval.run_hybrid_retrieval import run_hybrid_retrieval


DEFAULT_DATASET_PATH = Path("data/evaluation/retrieval_questions.jsonl")


def retrieve_for_evaluation(
    record: EvaluationRecord,
    k: int,
):
    """Run the same retrieval routing used by the production application."""

    # The explicit target-language filter is important because the evaluation
    # dataset defines which document language should contain the gold evidence.
    filters = {
        "language": record.target_language,
    }

    return run_hybrid_retrieval(
        record.query,
        top_k=k,
        filters=filters,
    )


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    k: int = 5,
) -> None:
    """Evaluate the production retrieval route against the curated gold set."""

    records = load_evaluation_records(dataset_path)

    per_query, summary = evaluate_dataset(
        records,
        retrieve_for_evaluation,
        k=k,
    )

    print(f"Retrieval evaluation @ {k}")
    print("-" * 60)

    # Per-query output makes failures visible instead of hiding them
    # behind only aggregate metrics.
    for metric in per_query:
        print(
            f"{metric.question_id}: "
            f"hit={metric.hit_rate:.3f} "
            f"mrr={metric.reciprocal_rank:.3f} "
            f"recall={metric.recall:.3f}"
        )

    print("-" * 60)
    print(f"Hit Rate@{k}: {summary['hit_rate']:.3f}")
    print(f"MRR@{k}:      {summary['mrr']:.3f}")
    print(f"Recall@{k}:   {summary['recall']:.3f}")


if __name__ == "__main__":
    run_evaluation()