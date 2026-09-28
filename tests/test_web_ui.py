"""Tests for the NepalGov AI browser application."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)

from fastapi.testclient import (
    TestClient,
)

from src.api.app import (
    create_app,
)
from src.api.readiness import (
    DependencyReadiness,
)
from src.rag.pipeline import (
    RAGResult,
)


def ready_dependencies(
) -> DependencyReadiness:
    """Return deterministic healthy readiness state."""

    return DependencyReadiness(
        gemini_configured=True,
        hf_token_configured=True,
        reranker_endpoint_configured=True,
        qdrant_reachable=True,
        qdrant_collection_ready=True,
    )


@dataclass
class FakePipeline:
    """Minimal pipeline dependency for frontend route tests."""

    closed: bool = False
    answer_calls: int = 0

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
        """Return a deterministic result if a test unexpectedly calls answer."""

        self.answer_calls += 1

        return RAGResult(
            answer_text="Answer [E1].",
            accepted=True,
            reason=None,
            sources=(),
            selected_context=(),
            citation_result=None,
            provider="fake",
            model="fake",
        )

    def close(
        self,
    ) -> None:
        """Record shutdown."""

        self.closed = True


def build_client(
) -> tuple[
    FakePipeline,
    TestClient,
]:
    """Create one frontend test client."""

    pipeline = (
        FakePipeline()
    )

    application = (
        create_app(
            pipeline_factory=lambda: pipeline,
            readiness_checker=(
                ready_dependencies
            ),
        )
    )

    return (
        pipeline,
        TestClient(
            application
        ),
    )


def test_root_serves_browser_application() -> None:
    """The FastAPI root should serve the NepalGov AI interface."""

    pipeline, client = (
        build_client()
    )

    with client:
        response = (
            client.get(
                "/"
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.headers[
            "content-type"
        ].startswith(
            "text/html"
        )
    )

    assert (
        "NepalGov AI"
        in response.text
    )

    assert (
        'id="ask-form"'
        in response.text
    )

    assert (
        'id="answer-language"'
        in response.text
    )

    assert (
        'id="document-filter"'
        in response.text
    )

    assert (
        pipeline.answer_calls
        == 0
    )


def test_root_has_browser_security_headers() -> None:
    """The UI entry point should receive basic browser hardening headers."""

    _, client = (
        build_client()
    )

    with client:
        response = (
            client.get(
                "/"
            )
        )

    assert (
        response.headers[
            "x-content-type-options"
        ]
        == "nosniff"
    )

    assert (
        response.headers[
            "referrer-policy"
        ]
        == "no-referrer"
    )

    assert (
        "default-src 'self'"
        in response.headers[
            "content-security-policy"
        ]
    )


def test_stylesheet_is_served() -> None:
    """The same FastAPI service should expose the UI stylesheet."""

    _, client = (
        build_client()
    )

    with client:
        response = (
            client.get(
                "/static/styles.css"
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        "text/css"
        in response.headers[
            "content-type"
        ]
    )

    assert (
        "--accent:"
        in response.text
    )


def test_browser_javascript_is_served() -> None:
    """The browser application logic should be available same-origin."""

    _, client = (
        build_client()
    )

    with client:
        response = (
            client.get(
                "/static/app.js"
            )
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        "javascript"
        in response.headers[
            "content-type"
        ]
    )

    assert (
        '"/health/ready"'
        in response.text
    )

    assert (
        '"/v1/answer"'
        in response.text
    )


def test_browser_renderer_does_not_inject_generated_html() -> None:
    """Generated answer/source text should be assigned through safe DOM APIs."""

    _, client = (
        build_client()
    )

    with client:
        response = (
            client.get(
                "/static/app.js"
            )
        )

    assert (
        "textContent"
        in response.text
    )

    assert (
        "innerHTML"
        not in response.text
    )


def test_ui_lists_current_six_document_scopes() -> None:
    """The first interface should expose the six indexed V1 documents."""

    _, client = (
        build_client()
    )

    with client:
        response = (
            client.get(
                "/"
            )
        )

    expected_document_ids = (
        "constitution_nepal_current_en",
        "public_health_service_act_2075_en",
        "compulsory_free_education_act_2075_en",
        "economic_survey_2023_24_en",
        "economic_survey_2081_82_ne",
        "budget_speech_2025_26_en",
    )

    for document_id in (
        expected_document_ids
    ):
        assert (
            document_id
            in response.text
        )


def test_frontend_remains_available_when_pipeline_startup_fails() -> None:
    """Users should still receive the UI when backend readiness is degraded."""

    def failing_factory(
    ):
        raise RuntimeError(
            "pipeline unavailable"
        )

    application = (
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
        application
    ) as client:
        root_response = (
            client.get(
                "/"
            )
        )

        ready_response = (
            client.get(
                "/health/ready"
            )
        )

    assert (
        root_response.status_code
        == 200
    )

    assert (
        "NepalGov AI"
        in root_response.text
    )

    assert (
        ready_response.status_code
        == 503
    )