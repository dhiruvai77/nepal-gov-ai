"""Hosted BGE reranker using a Hugging Face TEI-compatible endpoint.

The provider sends one query and a list of first-stage retrieval candidates to
a Text Embeddings Inference `/rerank` endpoint. The endpoint is expected to
serve BAAI/bge-reranker-v2-m3 or another compatible cross-encoder reranker.

Keeping the endpoint URL configurable allows NepalGov AI to use a managed
Hugging Face Inference Endpoint, another TEI deployment, or a local TEI service
without changing the retrieval pipeline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from src.reranking.base import (
    RerankedResult,
    Reranker,
    select_top_reranked_results,
    validate_rerank_request,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


MODEL_NAME = "BAAI/bge-reranker-v2-m3"

HF_TOKEN_ENV = "HF_TOKEN"
HF_RERANKER_ENDPOINT_ENV = (
    "HF_RERANKER_ENDPOINT_URL"
)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0

RETRYABLE_HTTP_STATUS_CODES = {
    408,
    425,
    429,
}


class HuggingFaceBGEReranker(
    Reranker
):
    """Hosted multilingual BGE reranker backed by TEI."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        token: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_base_delay_seconds: float = (
            DEFAULT_RETRY_BASE_DELAY_SECONDS
        ),
    ) -> None:
        """Configure the hosted reranker without making a network request."""

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be greater than zero."
            )

        if retry_base_delay_seconds < 0:
            raise ValueError(
                "retry_base_delay_seconds cannot be negative."
            )

        resolved_endpoint = (
            endpoint_url
            if endpoint_url is not None
            else os.getenv(
                HF_RERANKER_ENDPOINT_ENV
            )
        )

        if not resolved_endpoint:
            raise ValueError(
                f"{HF_RERANKER_ENDPOINT_ENV} must be set "
                "for hosted reranking."
            )

        clean_endpoint = (
            resolved_endpoint
            .strip()
            .rstrip("/")
        )

        if not clean_endpoint:
            raise ValueError(
                "endpoint_url must contain non-whitespace text."
            )

        # Accept either a TEI base URL or the full `/rerank` URL so deployment
        # configuration remains simple across local and hosted environments.
        self.rerank_url = (
            clean_endpoint
            if clean_endpoint.endswith(
                "/rerank"
            )
            else (
                f"{clean_endpoint}/rerank"
            )
        )

        self.token = (
            token
            if token is not None
            else os.getenv(
                HF_TOKEN_ENV
            )
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self.max_attempts = (
            max_attempts
        )

        self.retry_base_delay_seconds = (
            retry_base_delay_seconds
        )

        # Lazily constructing the HTTP client keeps unit tests isolated from
        # external services and makes provider construction inexpensive.
        self._client: Any | None = None

    def _get_client(
        self,
    ) -> Any:
        """Create the HTTP client only when reranking is actually requested."""

        if self._client is not None:
            return self._client

        headers = {}

        # Protected Hugging Face Inference Endpoints normally require an
        # authorization token, while a local TEI deployment may not.
        if self.token:
            headers[
                "Authorization"
            ] = (
                f"Bearer {self.token}"
            )

        self._client = httpx.Client(
            timeout=self.timeout_seconds,
            headers=headers,
        )

        return self._client

    @staticmethod
    def _format_candidate_text(
        candidate: RetrievalResult,
    ) -> str:
        """Build the evaluated source-aware reranker passage representation.

        Evaluation showed that adding the document title helps the cross-encoder
        respect explicit source intent while retaining the original evidence
        passage. The richer contextual dense-embedding representation is not
        passed to the reranker.
        """

        return (
            f"Document: {candidate.title}\n\n"
            f"{candidate.chunk_text}"
        )

    @staticmethod
    def _extract_status_code(
        exc: BaseException,
    ) -> int | None:
        """Extract an HTTP status code from an exception or its cause chain."""

        current: BaseException | None = exc

        while current is not None:
            response = getattr(
                current,
                "response",
                None,
            )

            status_code = getattr(
                response,
                "status_code",
                None,
            )

            if isinstance(
                status_code,
                int,
            ):
                return status_code

            current = current.__cause__

        return None

    @classmethod
    def _is_retryable_exception(
        cls,
        exc: BaseException,
    ) -> bool:
        """Return whether a hosted reranking failure is likely temporary."""

        status_code = cls._extract_status_code(
            exc
        )

        if status_code is not None:
            return (
                status_code
                in RETRYABLE_HTTP_STATUS_CODES
                or 500 <= status_code <= 599
            )

        # Transport failures happen before a valid HTTP response is received
        # and are generally safe to retry with bounded exponential backoff.
        current: BaseException | None = exc

        while current is not None:
            if isinstance(
                current,
                httpx.TransportError,
            ):
                return True

            current = current.__cause__

        return False

    def _request_rerank(
        self,
        query: str,
        texts: list[str],
    ) -> Any:
        """Send one TEI rerank request with bounded transient-error retries."""

        client = self._get_client()

        # TEI's native reranking contract accepts one query and a list of texts.
        # Normalized scores are used because only ranking order matters here.
        payload = {
            "query": query,
            "texts": texts,
            "truncate": True,
            "raw_scores": False,
            "return_text": False,
        }

        last_exception: Exception | None = None
        attempts_used = 0

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            attempts_used = attempt

            try:
                response = client.post(
                    self.rerank_url,
                    json=payload,
                )

                response.raise_for_status()

                return response.json()

            except Exception as exc:
                last_exception = exc

                if (
                    not self._is_retryable_exception(
                        exc
                    )
                    or attempt
                    == self.max_attempts
                ):
                    break

                delay_seconds = (
                    self.retry_base_delay_seconds
                    * (
                        2 ** (
                            attempt - 1
                        )
                    )
                )

                time.sleep(
                    delay_seconds
                )

        raise RuntimeError(
            "Hosted BGE reranker request failed "
            f"after {attempts_used} attempt(s)."
        ) from last_exception

    @staticmethod
    def _parse_ranked_results(
        response_data: Any,
        candidates: Sequence[
            RetrievalResult
        ],
    ) -> list[
        RerankedResult
    ]:
        """Map TEI rank records back onto their original retrieval candidates."""

        if not isinstance(
            response_data,
            list,
        ):
            raise RuntimeError(
                "Unexpected reranker response type."
            )

        if len(
            response_data
        ) != len(
            candidates
        ):
            raise RuntimeError(
                "Hosted reranker returned a different "
                "number of ranks than input candidates."
            )

        seen_indices: set[int] = set()

        results: list[
            RerankedResult
        ] = []

        for item in response_data:
            if not isinstance(
                item,
                dict,
            ):
                raise RuntimeError(
                    "Unexpected reranker rank item type."
                )

            index = item.get(
                "index"
            )

            score = item.get(
                "score"
            )

            if (
                not isinstance(
                    index,
                    int,
                )
                or isinstance(
                    index,
                    bool,
                )
            ):
                raise RuntimeError(
                    "Reranker response is missing a valid candidate index."
                )

            if (
                index < 0
                or index >= len(
                    candidates
                )
            ):
                raise RuntimeError(
                    "Reranker returned a candidate index "
                    "outside the input range."
                )

            if index in seen_indices:
                raise RuntimeError(
                    "Reranker returned a duplicate candidate index."
                )

            if (
                not isinstance(
                    score,
                    (int, float),
                )
                or isinstance(
                    score,
                    bool,
                )
            ):
                raise RuntimeError(
                    "Reranker response is missing a numeric score."
                )

            seen_indices.add(
                index
            )

            candidate = candidates[
                index
            ]

            results.append(
                RerankedResult(
                    result=candidate,
                    rerank_score=float(
                        score
                    ),
                    # TEI's index points back to the original first-stage list.
                    original_rank=(
                        index + 1
                    ),
                )
            )

        return results

    def rerank(
        self,
        query: str,
        candidates: Sequence[
            RetrievalResult
        ],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Rerank first-stage candidates using multilingual BGE relevance."""

        clean_query = validate_rerank_request(
            query,
            candidates,
            top_k=top_k,
        )

        candidate_list = list(
            candidates
        )

        # Controlled evaluation showed that adding only the document title
        # substantially improves source-sensitive reranking while preserving
        # the original passage text for citations and downstream RAG stages.
        #
        # The richer metadata representation used for contextual dense
        # embeddings remains retrieval-only and is not passed to BGE.
        texts = [
            self._format_candidate_text(
                candidate
            )
            for candidate in candidate_list
        ]

        response_data = (
            self._request_rerank(
                clean_query,
                texts,
            )
        )

        results = (
            self._parse_ranked_results(
                response_data,
                candidate_list,
            )
        )

        # TEI already returns ranked results, but applying our shared helper
        # gives deterministic tie handling and one consistent top_k contract.
        return select_top_reranked_results(
            results,
            top_k=top_k,
        )