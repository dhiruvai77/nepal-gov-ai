"""FastAPI application boundary for the NepalGov AI production RAG pipeline.

The API delegates retrieval, context selection, generation, citation
processing, and evidence guarding to the existing production RAG pipeline.

Application responsibilities are limited to:

- request validation,
- lifecycle management,
- structured source serialization,
- request observability,
- dependency readiness,
- stable HTTP error behavior.

This module does not alter production RAG behavior.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import (
    Callable,
)
from contextlib import (
    asynccontextmanager,
)
from typing import (
    Any,
    Protocol,
)
from uuid import (
    uuid4,
)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from src.api.readiness import (
    DependencyReadiness,
    check_production_dependencies,
)
from src.rag.pipeline import (
    RAGResult,
    build_production_rag_pipeline,
)


LOGGER = logging.getLogger(
    __name__
)

SERVICE_NAME = (
    "nepal-gov-ai"
)

API_VERSION = (
    "v1"
)

SUPPORTED_ANSWER_LANGUAGES = {
    "en",
    "ne",
}

REQUEST_ID_HEADER = (
    "X-Request-ID"
)

PROCESS_TIME_HEADER = (
    "X-Process-Time-Ms"
)

REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{1,128}$"
)


class AnswerPipeline(
    Protocol
):
    """Application-facing contract implemented by RAGPipeline."""

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
        """Return one evidence-grounded RAG result."""

    def close(
        self,
    ) -> None:
        """Release pipeline resources."""


PipelineFactory = Callable[
    [],
    AnswerPipeline,
]

ReadinessChecker = Callable[
    [],
    DependencyReadiness,
]


class AnswerRequest(
    BaseModel
):
    """Validated API request for one grounded answer."""

    query: str = Field(
        min_length=1,
        max_length=2000,
    )

    answer_language: str = "en"

    filters: dict[
        str,
        str,
    ] | None = None

    @field_validator(
        "query"
    )
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate the user query."""

        clean_value = (
            value.strip()
        )

        if not clean_value:
            raise ValueError(
                "query must contain non-whitespace text."
            )

        return clean_value

    @field_validator(
        "answer_language",
        mode="before",
    )
    @classmethod
    def validate_answer_language(
        cls,
        value: Any,
    ) -> str:
        """Normalize the supported English/Nepali answer language."""

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "answer_language must be a string."
            )

        clean_value = (
            value
            .strip()
            .lower()
        )

        if (
            clean_value
            not in SUPPORTED_ANSWER_LANGUAGES
        ):
            raise ValueError(
                "answer_language must be "
                "'en' or 'ne'."
            )

        return clean_value

    @field_validator(
        "filters"
    )
    @classmethod
    def validate_filters(
        cls,
        value: dict[
            str,
            str,
        ]
        | None,
    ) -> dict[
        str,
        str,
    ] | None:
        """Normalize optional retrieval-filter keys and values."""

        if value is None:
            return None

        normalized: dict[
            str,
            str,
        ] = {}

        for (
            raw_key,
            raw_value,
        ) in value.items():
            clean_key = (
                raw_key.strip()
            )

            clean_value = (
                raw_value.strip()
            )

            if not clean_key:
                raise ValueError(
                    "filter names must contain "
                    "non-whitespace text."
                )

            if not clean_value:
                raise ValueError(
                    "filter values must contain "
                    "non-whitespace text."
                )

            if (
                clean_key
                in normalized
            ):
                raise ValueError(
                    "filters contain duplicate "
                    "normalized names."
                )

            normalized[
                clean_key
            ] = (
                clean_value
            )

        return (
            normalized
            if normalized
            else None
        )


class SourceResponse(
    BaseModel
):
    """Structured canonical metadata for one cited evidence passage."""

    evidence_id: str
    document_id: str
    chunk_id: str
    title: str
    organization: str
    language: str
    page_start: int
    page_end: int
    source_url: str


class AnswerResponse(
    BaseModel
):
    """Stable application response returned by the answer endpoint."""

    request_id: str
    answer_text: str
    accepted: bool
    withheld: bool
    reason: str | None
    sources: list[
        SourceResponse
    ]
    provider: str | None
    model: str | None
    selected_context_count: int


class HealthResponse(
    BaseModel
):
    """Lightweight application lifecycle health response."""

    status: str
    service: str
    api_version: str
    pipeline_ready: bool


class ReadinessChecks(
    BaseModel
):
    """Non-secret production dependency status."""

    pipeline_initialized: bool
    gemini_configured: bool
    hf_token_configured: bool
    reranker_endpoint_configured: bool
    qdrant_reachable: bool
    qdrant_collection_ready: bool


class ReadinessResponse(
    BaseModel
):
    """Production dependency-readiness response."""

    status: str
    service: str
    api_version: str
    checks: ReadinessChecks


def _resolve_request_id(
    supplied_value: str | None,
) -> str:
    """Accept a safe caller request ID or generate a new opaque ID."""

    if (
        isinstance(
            supplied_value,
            str,
        )
    ):
        clean_value = (
            supplied_value.strip()
        )

        if (
            REQUEST_ID_PATTERN.fullmatch(
                clean_value
            )
        ):
            return clean_value

    return uuid4().hex


def _pipeline_is_ready(
    request: Request,
) -> bool:
    """Return whether application startup initialized the RAG pipeline."""

    return (
        getattr(
            request.app.state,
            "pipeline",
            None,
        )
        is not None
    )


def _build_structured_sources(
    result: RAGResult,
) -> list[
    SourceResponse
]:
    """Serialize only citations validated by the production citation layer."""

    if (
        not result.accepted
        or result.citation_result
        is None
    ):
        return []

    sources: list[
        SourceResponse
    ] = []

    for citation in (
        result
        .citation_result
        .citations
    ):
        evidence = (
            citation.evidence.result
        )

        sources.append(
            SourceResponse(
                evidence_id=(
                    citation.evidence_id
                ),
                document_id=(
                    evidence.document_id
                ),
                chunk_id=(
                    evidence.chunk_id
                ),
                title=(
                    evidence.title
                ),
                organization=(
                    evidence.organization
                ),
                language=(
                    evidence.language
                ),
                page_start=(
                    evidence.page_start
                ),
                page_end=(
                    evidence.page_end
                ),
                source_url=(
                    evidence.source_url
                ),
            )
        )

    return sources


def _result_to_response(
    result: RAGResult,
    *,
    request_id: str,
) -> AnswerResponse:
    """Convert one immutable internal RAG result into the API contract."""

    reason = (
        result.reason.value
        if result.reason
        is not None
        else None
    )

    return AnswerResponse(
        request_id=(
            request_id
        ),
        answer_text=(
            result.answer_text
        ),
        accepted=(
            result.accepted
        ),
        withheld=(
            result.withheld
        ),
        reason=reason,
        sources=(
            _build_structured_sources(
                result
            )
        ),
        provider=(
            result.provider
        ),
        model=(
            result.model
        ),
        selected_context_count=len(
            result.selected_context
        ),
    )


def create_app(
    *,
    pipeline_factory: PipelineFactory = (
        build_production_rag_pipeline
    ),
    readiness_checker: ReadinessChecker = (
        check_production_dependencies
    ),
) -> FastAPI:
    """Create the API with injectable production dependencies."""

    if not callable(
        pipeline_factory
    ):
        raise TypeError(
            "pipeline_factory must be callable."
        )

    if not callable(
        readiness_checker
    ):
        raise TypeError(
            "readiness_checker must be callable."
        )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        """Initialize production resources without killing health endpoints."""

        pipeline: (
            AnswerPipeline
            | None
        ) = None

        app.state.pipeline = None

        try:
            pipeline = (
                pipeline_factory()
            )

            app.state.pipeline = (
                pipeline
            )

        except Exception:
            # Starting the HTTP process while marking it not-ready provides
            # useful diagnostics to orchestrators instead of making both
            # liveness and readiness disappear on configuration failures.
            LOGGER.exception(
                "Production RAG pipeline initialization failed."
            )

        try:
            yield

        finally:
            if (
                pipeline
                is not None
            ):
                pipeline.close()

            app.state.pipeline = None

    application = FastAPI(
        title="NepalGov AI",
        description=(
            "Evidence-grounded question answering "
            "over official Nepal government documents."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.middleware(
        "http"
    )
    async def request_observability(
        request: Request,
        call_next,
    ):
        """Attach request correlation and processing-time metadata."""

        request_id = (
            _resolve_request_id(
                request.headers.get(
                    REQUEST_ID_HEADER
                )
            )
        )

        request.state.request_id = (
            request_id
        )

        started_at = (
            time.perf_counter()
        )

        try:
            response = (
                await call_next(
                    request
                )
            )

        except Exception:
            elapsed_ms = (
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000.0
            )

            LOGGER.exception(
                "api_request_failed "
                "request_id=%s method=%s path=%s "
                "duration_ms=%.3f",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )

            raise

        elapsed_ms = (
            (
                time.perf_counter()
                - started_at
            )
            * 1000.0
        )

        response.headers[
            REQUEST_ID_HEADER
        ] = (
            request_id
        )

        response.headers[
            PROCESS_TIME_HEADER
        ] = (
            f"{elapsed_ms:.3f}"
        )

        LOGGER.info(
            "api_request "
            "request_id=%s method=%s path=%s "
            "status=%s duration_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=[
            "system",
        ],
    )
    def health(
        request: Request,
    ) -> HealthResponse:
        """Report application lifecycle and pipeline initialization."""

        pipeline_ready = (
            _pipeline_is_ready(
                request
            )
        )

        return HealthResponse(
            status=(
                "ok"
                if pipeline_ready
                else "not_ready"
            ),
            service=SERVICE_NAME,
            api_version=API_VERSION,
            pipeline_ready=(
                pipeline_ready
            ),
        )

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=[
            "system",
        ],
    )
    def readiness(
        request: Request,
        response: Response,
    ) -> ReadinessResponse:
        """Check production configuration and Qdrant readiness."""

        dependency_state = (
            readiness_checker()
        )

        pipeline_initialized = (
            _pipeline_is_ready(
                request
            )
        )

        ready = (
            pipeline_initialized
            and dependency_state.ready
        )

        if not ready:
            response.status_code = 503

        return ReadinessResponse(
            status=(
                "ready"
                if ready
                else "not_ready"
            ),
            service=SERVICE_NAME,
            api_version=API_VERSION,
            checks=ReadinessChecks(
                pipeline_initialized=(
                    pipeline_initialized
                ),
                gemini_configured=(
                    dependency_state
                    .gemini_configured
                ),
                hf_token_configured=(
                    dependency_state
                    .hf_token_configured
                ),
                reranker_endpoint_configured=(
                    dependency_state
                    .reranker_endpoint_configured
                ),
                qdrant_reachable=(
                    dependency_state
                    .qdrant_reachable
                ),
                qdrant_collection_ready=(
                    dependency_state
                    .qdrant_collection_ready
                ),
            ),
        )

    @application.post(
        "/v1/answer",
        response_model=AnswerResponse,
        tags=[
            "rag",
        ],
    )
    def answer(
        payload: AnswerRequest,
        request: Request,
    ) -> AnswerResponse:
        """Run one question through the retained production RAG stack."""

        pipeline = getattr(
            request.app.state,
            "pipeline",
            None,
        )

        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "RAG service is not ready."
                ),
            )

        try:
            result = (
                pipeline.answer(
                    payload.query,
                    answer_language=(
                        payload.answer_language
                    ),
                    filters=(
                        payload.filters
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(
                    exc
                ),
            ) from exc

        except Exception as exc:
            LOGGER.exception(
                "Unhandled production RAG request failure. "
                "request_id=%s",
                request.state.request_id,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "RAG service is temporarily unavailable."
                ),
            ) from exc

        return (
            _result_to_response(
                result,
                request_id=(
                    request.state.request_id
                ),
            )
        )

    return application


app = (
    create_app()
)