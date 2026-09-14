from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient

from src.embeddings.hf_e5_service import (
    HuggingFaceE5EmbeddingService,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    QueryMetrics,
    aggregate_metrics,
    evaluate_ranked_results,
    load_evaluation_records,
)
from src.indexing.qdrant_setup import QDRANT_URL
from src.retrieval.dense_retriever import (
    DenseRetriever,
    RetrievalResult,
)
from src.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    HybridRetriever,
)
from src.retrieval.sparse_retriever import SparseRetriever


DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

# Evaluate several cutoffs from one fixed ranking so the result at @5 is
# always the prefix of @10, which is itself the prefix of @20.
DEFAULT_CUTOFFS = (5, 10, 20)
RETRIEVAL_DEPTH = max(DEFAULT_CUTOFFS)

# Only same-language slices are compared here. Cross-lingual production
# retrieval intentionally skips BM25 and therefore has no meaningful
# dense-vs-BM25-vs-hybrid comparison.
SAME_LANGUAGE_PAIRS = (
    ("en", "en"),
    ("ne", "ne"),
)

RETRIEVAL_MODES = (
    "dense",
    "bm25",
    "hybrid",
)


@dataclass
class CachedRetriever:
    """Expose already-retrieved candidates through the Retriever protocol."""

    results: Sequence[RetrievalResult]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return a prefix without performing another external retrieval."""

        # query and filters are intentionally unused here. The candidates were
        # already retrieved with the correct query and metadata filter.
        del query, filters

        return list(
            self.results[:top_k]
        )


def evaluate_rankings(
    records: Sequence[EvaluationRecord],
    rankings: dict[str, list[RetrievalResult]],
    *,
    k: int,
) -> list[QueryMetrics]:
    """Evaluate cached rankings for one retrieval mode at cutoff k."""

    return [
        evaluate_ranked_results(
            record,
            rankings[record.question_id],
            k=k,
        )
        for record in records
    ]


def print_metric_row(
    language_pair: tuple[str, str],
    mode: str,
    count: int,
    summary: dict[str, float],
) -> None:
    """Print one aligned retrieval-mode comparison row."""

    query_language, target_language = language_pair
    pair_label = f"{query_language}->{target_language}"

    print(
        f"{pair_label:<8}"
        f"{mode:<10}"
        f"{count:>4}  "
        f"{summary['hit_rate']:>8.3f}  "
        f"{summary['mrr']:>8.3f}  "
        f"{summary['recall']:>8.3f}"
    )


def build_rankings(
    records: Sequence[EvaluationRecord],
) -> dict[str, dict[str, list[RetrievalResult]]]:
    """Retrieve dense/BM25 candidates once and derive the hybrid rankings."""

    # Reuse one hosted embedding service and one Qdrant client across the
    # benchmark. This avoids rebuilding dependencies for every query.
    embedding_service = HuggingFaceE5EmbeddingService()
    client = QdrantClient(
        url=QDRANT_URL
    )

    dense_retriever = DenseRetriever(
        client=client,
        embedding_service=embedding_service,
    )
    sparse_retriever = SparseRetriever(
        client=client,
    )

    rankings: dict[
        str,
        dict[str, list[RetrievalResult]],
    ] = {
        "dense": {},
        "bm25": {},
        "hybrid": {},
    }

    # Production hybrid retrieval requests top_k * candidate_multiplier from
    # each component retriever before RRF. For a fixed final depth of 20, that
    # means we need 60 candidates from dense and 60 from BM25.
    candidate_depth = (
        RETRIEVAL_DEPTH
        * DEFAULT_CANDIDATE_MULTIPLIER
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"Retrieving {index}/{len(records)}: "
            f"{record.question_id}"
        )

        filters = {
            "language": record.target_language,
        }

        # Retrieve each component only once. Dense retrieval uses the hosted
        # multilingual E5 service, while BM25 runs through Qdrant.
        dense_candidates = dense_retriever.retrieve(
            query=record.query,
            top_k=candidate_depth,
            filters=filters,
        )

        sparse_candidates = sparse_retriever.retrieve(
            query=record.query,
            top_k=candidate_depth,
            filters=filters,
        )

        rankings["dense"][record.question_id] = list(
            dense_candidates[:RETRIEVAL_DEPTH]
        )
        rankings["bm25"][record.question_id] = list(
            sparse_candidates[:RETRIEVAL_DEPTH]
        )

        # Feed cached component rankings into the real production
        # HybridRetriever. This reproduces the existing RRF implementation
        # without making a second hosted embedding request for the query.
        cached_hybrid = HybridRetriever(
            dense_retriever=CachedRetriever(
                dense_candidates
            ),
            sparse_retriever=CachedRetriever(
                sparse_candidates
            ),
            rrf_k=DEFAULT_RRF_K,
            candidate_multiplier=(
                DEFAULT_CANDIDATE_MULTIPLIER
            ),
        )

        rankings["hybrid"][record.question_id] = (
            cached_hybrid.retrieve(
                query=record.query,
                top_k=RETRIEVAL_DEPTH,
                filters=filters,
            )
        )

    return rankings


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> None:
    """Compare dense, BM25, and hybrid retrieval on same-language slices."""

    all_records = load_evaluation_records(
        dataset_path
    )

    # Cross-lingual records are deliberately excluded because production
    # routing correctly uses dense-only retrieval for those queries.
    records = [
        record
        for record in all_records
        if (
            record.query_language,
            record.target_language,
        )
        in SAME_LANGUAGE_PAIRS
    ]

    if not records:
        raise ValueError(
            "No same-language evaluation records were found."
        )

    rankings = build_rankings(
        records
    )

    for k in DEFAULT_CUTOFFS:
        print("\n" + "=" * 68)
        print(
            f"Dense vs BM25 vs Hybrid @ {k}"
        )
        print("=" * 68)
        print(
            f"{'Slice':<8}"
            f"{'Mode':<10}"
            f"{'N':>4}  "
            f"{'Hit Rate':>8}  "
            f"{'MRR':>8}  "
            f"{'Recall':>8}"
        )
        print("-" * 68)

        for language_pair in SAME_LANGUAGE_PAIRS:
            pair_records = [
                record
                for record in records
                if (
                    record.query_language,
                    record.target_language,
                )
                == language_pair
            ]

            for mode in RETRIEVAL_MODES:
                metrics = evaluate_rankings(
                    pair_records,
                    rankings[mode],
                    k=k,
                )

                summary = aggregate_metrics(
                    metrics
                )

                print_metric_row(
                    language_pair,
                    mode,
                    len(pair_records),
                    summary,
                )

        print("-" * 68)

    print("\nPersistent primary-evidence misses @20")
    print("=" * 68)

    # Print the exact failures for each retrieval method. This tells us whether
    # hybrid retrieval is rescuing dense/BM25 failures or suppressing otherwise
    # successful component results.
    for language_pair in SAME_LANGUAGE_PAIRS:
        pair_records = [
            record
            for record in records
            if (
                record.query_language,
                record.target_language,
            )
            == language_pair
        ]

        pair_label = (
            f"{language_pair[0]}"
            f"->{language_pair[1]}"
        )

        for mode in RETRIEVAL_MODES:
            metrics = evaluate_rankings(
                pair_records,
                rankings[mode],
                k=20,
            )

            missed_ids = [
                metric.question_id
                for metric in metrics
                if metric.hit_rate == 0.0
            ]

            misses = (
                ", ".join(missed_ids)
                if missed_ids
                else "None"
            )

            print(
                f"{pair_label:<8}"
                f"{mode:<10}"
                f"{misses}"
            )


if __name__ == "__main__":
    run_comparison()