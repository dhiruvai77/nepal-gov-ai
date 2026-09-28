"""FastAPI application boundary for the NepalGov AI production RAG pipeline.

The API layer intentionally delegates all retrieval, context selection,
generation, citation processing, and evidence guarding to the existing
production RAG pipeline.

This module does not alter production RAG behavior.

The production pipeline is constructed during application startup rather than
module import. Tests can inject a deterministic pipeline factory without
loading hosted generation providers or retrieval infrastructure.
"""

from __future__ import annotations

import logging
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

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
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


class AnswerResponse(
    BaseModel
):
    """Stable application response returned by the answer endpoint."""

    answer_text: str
    accepted: bool
    withheld: bool
    reason: str | None
    sources: list[str]
    provider: str | None
    model: str | None
    selected_context_count: int


class HealthResponse(
    BaseModel
):
    """Minimal application readiness response."""

    status: str
    service: str
    api_version: str
    pipeline_ready: bool


def _result_to_response(
    result: RAGResult,
) -> AnswerResponse:
    """Convert the internal immutable RAG result to the API contract."""

    reason = (
        result.reason.value
        if result.reason
        is not None
        else None
    )

    return AnswerResponse(
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
        sources=list(
            result.sources
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
) -> FastAPI:
    """Create the API with an injectable production-pipeline factory."""

    if not callable(
        pipeline_factory
    ):
        raise TypeError(
            "pipeline_factory must be callable."
        )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        """Construct and release the application pipeline."""

        pipeline = (
            pipeline_factory()
        )

        app.state.pipeline = (
            pipeline
        )

        try:
            yield

        finally:
            pipeline.close()

    application = FastAPI(
        title="NepalGov AI",
        description=(
            "Evidence-grounded question answering "
            "over official Nepal government documents."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

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
        """Report whether the application pipeline is initialized."""

        pipeline_ready = (
            hasattr(
                request.app.state,
                "pipeline",
            )
            and request.app.state.pipeline
            is not None
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
                "Unhandled production RAG request failure."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "RAG service is temporarily unavailable."
                ),
            ) from exc

        return (
            _result_to_response(
                result
            )
        )

    return application


app = (
    create_app()
)