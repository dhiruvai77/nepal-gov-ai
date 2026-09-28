"""Tests for the NepalGov AI FastAPI application boundary."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from fastapi.testclient import (
    TestClient,
)

from src.api.app import (
    create_app,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
)
from src.rag.pipeline import (
    RAGResult,
)


@dataclass
class FakePipeline:
    """Deterministic application pipeline used by API tests."""

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


def make_result(
    *,
    accepted: bool = True,
) -> RAGResult:
    """Build one deterministic application-level RAG result."""

    if accepted:
        return RAGResult(
            answer_text=(
                "Supported government answer [E1]."
            ),
            accepted=True,
            reason=None,
            sources=(
                "[E1] Government Document — "
                "Government of Nepal — p. 1 — "
                "https://example.gov.np/document.pdf",
            ),
            selected_context=(),
            citation_result=None,
            provider="fake",
            model="fake-model",
        )

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


def test_health_reports_ready_pipeline() -> None:
    """Health should confirm that startup initialized the pipeline."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
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


def test_answer_endpoint_returns_accepted_rag_result() -> None:
    """Accepted production output should map to the stable API schema."""

    pipeline = (
        FakePipeline(
            make_result()
        )
    )

    app = (
        create_app(
            pipeline_factory=lambda: pipeline,
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
        len(
            body[
                "sources"
            ]
        )
        == 1
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