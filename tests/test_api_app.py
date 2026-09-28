"""Tests for the hardened NepalGov AI FastAPI application boundary."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from fastapi.testclient import (
    TestClient,
)

from src.api.app import (
    PROCESS_TIME_HEADER,
    REQUEST_ID_HEADER,
    create_app,
)
from src.api.readiness import (
    DependencyReadiness,
)
from src.citations.evidence import (
    process_answer_citations,
    render_cited_sources,
)
from src.generation.base import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
)
from src.rag.pipeline import (
    RAGPipeline,
    RAGResult,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def ready_dependencies(
) -> DependencyReadiness:
    """Return deterministic healthy dependency state."""

    return DependencyReadiness(
        gemini_configured=True,
        hf_token_configured=True,
        reranker_endpoint_configured=True,
        qdrant_reachable=True,
        qdrant_collection_ready=True,
    )


def unavailable_dependencies(
) -> DependencyReadiness:
    """Return one deterministic unhealthy dependency state."""

    return DependencyReadiness(
        gemini_configured=True,
        hf_token_configured=True,
        reranker_endpoint_configured=True,
        qdrant_reachable=False,
        qdrant_collection_ready=False,
    )


def make_evidence(
    index: int = 1,
) -> RerankedResult:
    """Create one deterministic selected evidence passage."""

    result = RetrievalResult(
        point_id=(
            f"point-{index}"
        ),
        score=0.9,
        chunk_id=(
            f"chunk-{index}"
        ),
        document_id=(
            f"document-{index}"
        ),
        title=(
            f"Government Document {index}"
        ),
        organization=(
            "Government of Nepal"
        ),
        language="en",
        page_start=index,
        page_end=index,
        source_url=(
            "https://example.gov.np/"
            f"document-{index}.pdf"
        ),
        chunk_text=(
            f"Evidence passage {index}."
        ),
        chunk_index=index,
        token_count=100,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.95,
        original_rank=index,
    )


def make_result(
    *,
    accepted: bool = True,
) -> RAGResult:
    """Build one deterministic application-level RAG result."""

    if not accepted:
        return RAGResult(
            answer_text=(
                "The supplied government evidence is insufficient "
                "to provide a supported answer."
            ),
            accepted=False,
            reason=(
                EvidenceGuardReason
                .NO_SELECTED_EVIDENCE
            ),
            sources=(),
            selected_context=(),
            citation_result=None,
            provider=None,
            model=None,
        )

    evidence = (
        make_evidence(),
    )

    citation_result = (
        process_answer_citations(
            "Supported government answer [E1].",
            evidence,
        )
    )

    return RAGResult(
        answer_text=(
            "Supported government answer [E1]."
        ),
        accepted=True,
        reason=None,
        sources=(
            render_cited_sources(
                citation_result
            )
        ),
        selected_context=(
            evidence
        ),
        citation_result=(
            citation_result
        ),
        provider="fake",
        model="fake-model",
    )


@dataclass
class FakePipeline:
    """Deterministic application pipeline used by API boundary tests."""

    result: RAGResult
    error: Exception | None = None
    calls: list[
        dict
    ] = field(
        default_factory=list
    )
    closed: bool = False

    def answer(
        self,
        query: str,
        *,
        answer_language: str,
        filters: dict[
            str,
            str,
        ]
        | None = None,
    ) -> RAGResult:
        """Record one API call and return the configured result."""

        self.calls.append(
            {
                "query": query,
                "answer_language": (
                    answer_language
                ),
                "filters": filters,
            }
        )

        if (
            self.error
            is not None
        ):
            raise self.error

        return self.result

    def close(
        self,
    ) -> None:
        """Record application shutdown cleanup."""

        self.closed = True


class FakeGenerationService(
    GenerationService
):
    """Deterministic generator for an API-to-RAG integration test."""

    def __init__(
        self,
        answer_text: str,
    ) -> None:
        self.answer_text = (
            answer_text
        )

        self.requests: list[
            GenerationRequest
        ] = []

        self.closed = False

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Record generation input and return deterministic cited text."""

        self.requests.append(
            request
        )

        return GenerationResult(
            answer_text=(
                self.answer_text
            ),
            provider="integration-fake",
            model="integration-model",
        )

    def close(
        self,
    ) -> None:
        """Record cleanup."""

        self.closed = True


def test_health_reports_ready_pipeline() -> None:
    """Health should confirm successful pipeline initialization."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.get(
                "/health"
            )
        )

        assert (
            response.status_code
            == 200
        )

        assert (
            response.json()
            == {
                "status": "ok",
                "service": "nepal-gov-ai",
                "api_version": "v1",
                "pipeline_ready": True,
            }
        )


def test_readiness_reports_all_dependencies_ready() -> None:
    """Ready endpoint should expose non-secret dependency state."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.get(
                "/health/ready"
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()
        == {
            "status": "ready",
            "service": "nepal-gov-ai",
            "api_version": "v1",
            "checks": {
                "pipeline_initialized": True,
                "gemini_configured": True,
                "hf_token_configured": True,
                "reranker_endpoint_configured": True,
                "qdrant_reachable": True,
                "qdrant_collection_ready": True,
            },
        }
    )


def test_readiness_returns_503_when_dependency_is_unavailable() -> None:
    """A live process should remain explicitly not-ready on dependency failure."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                unavailable_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.get(
                "/health/ready"
            )
        )

    assert (
        response.status_code
        == 503
    )

    body = (
        response.json()
    )

    assert (
        body[
            "status"
        ]
        == "not_ready"
    )

    assert (
        body[
            "checks"
        ][
            "pipeline_initialized"
        ]
        is True
    )

    assert (
        body[
            "checks"
        ][
            "qdrant_reachable"
        ]
        is False
    )


def test_answer_endpoint_returns_structured_sources() -> None:
    """Accepted output should expose canonical source fields, not display text."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": (
                        "What does the Constitution say?"
                    ),
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 200
    )

    body = (
        response.json()
    )

    assert (
        body[
            "answer_text"
        ]
        == "Supported government answer [E1]."
    )

    assert (
        body[
            "accepted"
        ]
        is True
    )

    assert (
        body[
            "withheld"
        ]
        is False
    )

    assert (
        body[
            "reason"
        ]
        is None
    )

    assert (
        body[
            "provider"
        ]
        == "fake"
    )

    assert (
        body[
            "model"
        ]
        == "fake-model"
    )

    assert (
        body[
            "selected_context_count"
        ]
        == 1
    )

    assert (
        body[
            "sources"
        ]
        == [
            {
                "evidence_id": "E1",
                "document_id": "document-1",
                "chunk_id": "chunk-1",
                "title": "Government Document 1",
                "organization": "Government of Nepal",
                "language": "en",
                "page_start": 1,
                "page_end": 1,
                "source_url": (
                    "https://example.gov.np/"
                    "document-1.pdf"
                ),
            }
        ]
    )


def test_answer_endpoint_normalizes_application_input() -> None:
    """The API should normalize language, query, and retrieval filters."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "  प्रश्न के हो?  ",
                    "answer_language": " NE ",
                    "filters": {
                        " language ": " ne ",
                        " category ": " law ",
                    },
                },
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        pipeline.calls
        == [
            {
                "query": "प्रश्न के हो?",
                "answer_language": "ne",
                "filters": {
                    "language": "ne",
                    "category": "law",
                },
            }
        ]
    )


def test_answer_endpoint_preserves_withholding_decision() -> None:
    """Evidence-guard withholding must remain visible through the API."""

    pipeline = (
        FakePipeline(
            make_result(
                accepted=False,
            )
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "Question",
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 200
    )

    body = (
        response.json()
    )

    assert (
        body[
            "accepted"
        ]
        is False
    )

    assert (
        body[
            "withheld"
        ]
        is True
    )

    assert (
        body[
            "reason"
        ]
        == "no_selected_evidence"
    )

    assert (
        body[
            "sources"
        ]
        == []
    )


def test_blank_query_is_rejected_by_api_validation() -> None:
    """Blank user input should never reach the production pipeline."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "   ",
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 422
    )

    assert (
        pipeline.calls
        == []
    )


def test_unsupported_answer_language_is_rejected() -> None:
    """V1 should explicitly support only English and Nepali answers."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "Question",
                    "answer_language": "fr",
                },
            )
        )

    assert (
        response.status_code
        == 422
    )

    assert (
        pipeline.calls
        == []
    )


def test_pipeline_validation_error_becomes_422() -> None:
    """Application-level invalid requests should not become server errors."""

    pipeline = (
        FakePipeline(
            make_result(),
            error=ValueError(
                "invalid retrieval filter"
            ),
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "Question",
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 422
    )

    assert (
        response.json()
        == {
            "detail": (
                "invalid retrieval filter"
            )
        }
    )


def test_unexpected_pipeline_failure_becomes_503() -> None:
    """Provider/infrastructure failures should receive a stable API error."""

    pipeline = (
        FakePipeline(
            make_result(),
            error=RuntimeError(
                "provider unavailable"
            ),
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "Question",
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 503
    )

    assert (
        response.json()
        == {
            "detail": (
                "RAG service is temporarily unavailable."
            )
        }
    )


def test_caller_request_id_is_preserved() -> None:
    """Safe caller correlation IDs should be echoed in body and headers."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                headers={
                    REQUEST_ID_HEADER: (
                        "client-request-123"
                    ),
                },
                json={
                    "query": "Question",
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.headers[
            REQUEST_ID_HEADER
        ]
        == "client-request-123"
    )

    assert (
        response.json()[
            "request_id"
        ]
        == "client-request-123"
    )

    assert (
        float(
            response.headers[
                PROCESS_TIME_HEADER
            ]
        )
        >= 0.0
    )


def test_missing_request_id_is_generated() -> None:
    """Requests without correlation metadata should receive an opaque ID."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "Question",
                    "answer_language": "en",
                },
            )
        )

    request_id = (
        response.headers[
            REQUEST_ID_HEADER
        ]
    )

    assert (
        len(
            request_id
        )
        == 32
    )

    assert (
        response.json()[
            "request_id"
        ]
        == request_id
    )


def test_pipeline_startup_failure_keeps_health_endpoint_available() -> None:
    """Bad production configuration should mark the service not-ready."""

    def failing_factory(
    ):
        raise RuntimeError(
            "configuration unavailable"
        )

    app = (
        create_app(
            pipeline_factory=(
                failing_factory
            ),
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        health_response = (
            client.get(
                "/health"
            )
        )

        ready_response = (
            client.get(
                "/health/ready"
            )
        )

        answer_response = (
            client.post(
                "/v1/answer",
                json={
                    "query": "Question",
                    "answer_language": "en",
                },
            )
        )

    assert (
        health_response.status_code
        == 200
    )

    assert (
        health_response.json()[
            "pipeline_ready"
        ]
        is False
    )

    assert (
        ready_response.status_code
        == 503
    )

    assert (
        ready_response.json()[
            "checks"
        ][
            "pipeline_initialized"
        ]
        is False
    )

    assert (
        answer_response.status_code
        == 503
    )

    assert (
        answer_response.json()
        == {
            "detail": (
                "RAG service is not ready."
            )
        }
    )


def test_application_shutdown_closes_pipeline() -> None:
    """FastAPI shutdown should release RAG provider resources."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    assert (
        pipeline.closed
        is False
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.get(
                "/health"
            )
        )

        assert (
            response.status_code
            == 200
        )

        assert (
            pipeline.closed
            is False
        )

    assert (
        pipeline.closed
        is True
    )


def test_api_integrates_with_real_rag_orchestration() -> None:
    """Exercise API -> RAG -> citations -> guard without external services."""

    evidence = (
        make_evidence()
    )

    generation_service = (
        FakeGenerationService(
            "Integrated supported answer [E1]."
        )
    )

    pipeline = RAGPipeline(
        context_provider=(
            lambda query, filters=None: [
                evidence,
            ]
        ),
        generation_service=(
            generation_service
        ),
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    with TestClient(
        app
    ) as client:
        response = (
            client.post(
                "/v1/answer",
                json={
                    "query": (
                        "What does the government document say?"
                    ),
                    "answer_language": "en",
                },
            )
        )

    assert (
        response.status_code
        == 200
    )

    body = (
        response.json()
    )

    assert (
        body[
            "accepted"
        ]
        is True
    )

    assert (
        body[
            "answer_text"
        ]
        == "Integrated supported answer [E1]."
    )

    assert (
        body[
            "provider"
        ]
        == "integration-fake"
    )

    assert (
        body[
            "sources"
        ][
            0
        ][
            "evidence_id"
        ]
        == "E1"
    )

    assert (
        body[
            "sources"
        ][
            0
        ][
            "document_id"
        ]
        == "document-1"
    )

    assert (
        len(
            generation_service.requests
        )
        == 1
    )

    request = (
        generation_service.requests[
            0
        ]
    )

    assert (
        request.context
        == (
            evidence,
        )
    )

    assert (
        generation_service.closed
        is True
    )