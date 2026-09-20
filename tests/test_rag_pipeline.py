"""Tests for end-to-end NepalGov AI RAG orchestration."""

from __future__ import annotations

from unittest.mock import (
    Mock,
)

import pytest

from src.generation.base import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
)
from src.generation.gemini_service import (
    GeminiGenerationService,
)
from src.generation.grounded_prompt import (
    GroundedPromptBuilder,
)
from src.rag.pipeline import (
    RAGPipeline,
    build_production_rag_pipeline,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_evidence(
    index: int = 1,
) -> RerankedResult:
    """Create deterministic selected evidence."""

    result = RetrievalResult(
        point_id=f"point-{index}",
        score=0.1,
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        title=f"Government Document {index}",
        organization="Government of Nepal",
        language="en",
        page_start=index,
        page_end=index,
        source_url=(
            f"https://example.gov.np/"
            f"document-{index}.pdf"
        ),
        chunk_text=(
            f"Evidence passage {index}."
        ),
        chunk_index=index,
        token_count=200,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.95,
        original_rank=index,
    )


class FakeGenerationService(
    GenerationService
):
    """Deterministic generation provider used by orchestration tests."""

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
        """Record the request and return configured output."""

        self.requests.append(
            request
        )

        return GenerationResult(
            answer_text=(
                self.answer_text
            ),
            provider="fake",
            model="fake-model",
        )

    def close(
        self,
    ) -> None:
        """Record resource cleanup."""

        self.closed = True


def test_pipeline_accepts_valid_cited_answer() -> None:
    """A valid generated answer should pass through the complete pipeline."""

    evidence = (
        make_evidence()
    )

    context_provider = Mock(
        return_value=[
            evidence,
        ]
    )

    generation_service = (
        FakeGenerationService(
            "Supported answer [E1]."
        )
    )

    pipeline = RAGPipeline(
        context_provider=(
            context_provider
        ),
        generation_service=(
            generation_service
        ),
    )

    result = pipeline.answer(
        "What does the document say?",
        answer_language="en",
    )

    assert result.accepted is True
    assert result.withheld is False
    assert result.reason is None

    assert (
        result.answer_text
        == "Supported answer [E1]."
    )

    assert result.provider == "fake"
    assert result.model == "fake-model"


def test_pipeline_builds_generation_request_from_selected_context() -> None:
    """Generation should receive exactly the selected context."""

    first = make_evidence(
        1
    )

    second = make_evidence(
        2
    )

    generation_service = (
        FakeGenerationService(
            "Answer [E1] [E2]."
        )
    )

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[
                first,
                second,
            ]
        ),
        generation_service=(
            generation_service
        ),
    )

    pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert len(
        generation_service.requests
    ) == 1

    request = (
        generation_service.requests[
            0
        ]
    )

    assert request.query == "Question"

    assert request.context == (
        first,
        second,
    )

    assert (
        request.context[
            0
        ]
        is first
    )

    assert (
        request.context[
            1
        ]
        is second
    )


def test_pipeline_normalizes_query_and_answer_language() -> None:
    """Application input normalization should happen before downstream stages."""

    context_provider = Mock(
        return_value=[
            make_evidence(),
        ]
    )

    generation_service = (
        FakeGenerationService(
            "Answer [E1]."
        )
    )

    pipeline = RAGPipeline(
        context_provider=(
            context_provider
        ),
        generation_service=(
            generation_service
        ),
    )

    pipeline.answer(
        "  Question  ",
        answer_language=" EN ",
    )

    context_provider.assert_called_once_with(
        "Question",
        filters=None,
    )

    request = (
        generation_service.requests[
            0
        ]
    )

    assert request.query == "Question"
    assert request.answer_language == "en"


def test_pipeline_passes_filters_to_context_provider() -> None:
    """Document filters should remain available to retrieval orchestration."""

    context_provider = Mock(
        return_value=[
            make_evidence(),
        ]
    )

    pipeline = RAGPipeline(
        context_provider=(
            context_provider
        ),
        generation_service=(
            FakeGenerationService(
                "Answer [E1]."
            )
        ),
    )

    filters = {
        "language": "en",
        "category": "law",
    }

    pipeline.answer(
        "Question",
        answer_language="en",
        filters=filters,
    )

    context_provider.assert_called_once_with(
        "Question",
        filters=filters,
    )


def test_pipeline_renders_validated_sources() -> None:
    """Accepted answers should expose canonical source metadata."""

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[
                make_evidence(
                    1
                ),
            ]
        ),
        generation_service=(
            FakeGenerationService(
                "Supported answer [E1]."
            )
        ),
    )

    result = pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert len(
        result.sources
    ) == 1

    assert result.sources[
        0
    ] == (
        "[E1] Government Document 1 — "
        "Government of Nepal — "
        "p. 1 — "
        "https://example.gov.np/document-1.pdf"
    )


def test_pipeline_source_order_follows_answer_citations() -> None:
    """Rendered sources should follow citation appearance order."""

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[
                make_evidence(
                    1
                ),
                make_evidence(
                    2
                ),
            ]
        ),
        generation_service=(
            FakeGenerationService(
                "Second first [E2]. "
                "First second [E1]."
            )
        ),
    )

    result = pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert result.sources[
        0
    ].startswith(
        "[E2] Government Document 2"
    )

    assert result.sources[
        1
    ].startswith(
        "[E1] Government Document 1"
    )


def test_no_selected_evidence_skips_generation() -> None:
    """The LLM must not be called when context selection returns nothing."""

    generation_service = (
        FakeGenerationService(
            "This must never be generated."
        )
    )

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[]
        ),
        generation_service=(
            generation_service
        ),
    )

    result = pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert (
        generation_service.requests
        == []
    )

    assert result.accepted is False

    assert (
        result.reason
        == EvidenceGuardReason
        .NO_SELECTED_EVIDENCE
    )

    assert result.provider is None
    assert result.model is None
    assert result.sources == ()
    assert result.selected_context == ()


def test_missing_citation_withholds_generated_answer() -> None:
    """Model output without evidence references should not reach the user."""

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[
                make_evidence(),
            ]
        ),
        generation_service=(
            FakeGenerationService(
                "Unsupported uncited answer."
            )
        ),
    )

    result = pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert result.accepted is False

    assert (
        result.reason
        == EvidenceGuardReason
        .MISSING_CITATIONS
    )

    assert result.sources == ()

    assert result.answer_text == (
        "The supplied government evidence is insufficient "
        "to provide a supported answer."
    )

    # Generation still occurred, so provenance remains available.
    assert result.provider == "fake"
    assert result.model == "fake-model"


def test_invalid_citation_withholds_generated_answer() -> None:
    """Invented source identifiers should be blocked end to end."""

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[
                make_evidence(),
            ]
        ),
        generation_service=(
            FakeGenerationService(
                "Claim [E99]."
            )
        ),
    )

    result = pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert result.accepted is False

    assert (
        result.reason
        == EvidenceGuardReason
        .INVALID_CITATIONS
    )

    assert result.sources == ()


def test_selected_context_is_preserved_in_result() -> None:
    """The final result should retain exact evidence objects for diagnostics."""

    evidence = (
        make_evidence()
    )

    pipeline = RAGPipeline(
        context_provider=Mock(
            return_value=[
                evidence,
            ]
        ),
        generation_service=(
            FakeGenerationService(
                "Answer [E1]."
            )
        ),
    )

    result = pipeline.answer(
        "Question",
        answer_language="en",
    )

    assert result.selected_context == (
        evidence,
    )

    assert (
        result.selected_context[
            0
        ]
        is evidence
    )


def test_blank_query_is_rejected_before_context_retrieval() -> None:
    """Invalid queries should not trigger retrieval or provider calls."""

    context_provider = Mock()

    generation_service = (
        FakeGenerationService(
            "Answer [E1]."
        )
    )

    pipeline = RAGPipeline(
        context_provider=(
            context_provider
        ),
        generation_service=(
            generation_service
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "query must contain non-whitespace text"
        ),
    ):
        pipeline.answer(
            "   ",
            answer_language="en",
        )

    context_provider.assert_not_called()

    assert (
        generation_service.requests
        == []
    )


def test_non_string_query_is_rejected() -> None:
    """Queries should fail clearly rather than producing attribute errors."""

    pipeline = RAGPipeline(
        context_provider=Mock(),
        generation_service=(
            FakeGenerationService(
                "Answer."
            )
        ),
    )

    with pytest.raises(
        TypeError,
        match="query must be a string",
    ):
        pipeline.answer(
            None,
            answer_language="en",
        )


def test_blank_answer_language_is_rejected() -> None:
    """Answer-language validation should happen before retrieval."""

    context_provider = Mock()

    pipeline = RAGPipeline(
        context_provider=(
            context_provider
        ),
        generation_service=(
            FakeGenerationService(
                "Answer."
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "answer_language must contain "
            "non-whitespace text"
        ),
    ):
        pipeline.answer(
            "Question",
            answer_language="   ",
        )

    context_provider.assert_not_called()


def test_non_string_answer_language_is_rejected() -> None:
    """Answer languages should fail with a clear type error."""

    pipeline = RAGPipeline(
        context_provider=Mock(),
        generation_service=(
            FakeGenerationService(
                "Answer."
            )
        ),
    )

    with pytest.raises(
        TypeError,
        match=(
            "answer_language must be a string"
        ),
    ):
        pipeline.answer(
            "Question",
            answer_language=None,
        )


def test_pipeline_requires_callable_context_provider() -> None:
    """Pipeline construction should reject invalid retrieval dependencies."""

    with pytest.raises(
        TypeError,
        match=(
            "context_provider must be callable"
        ),
    ):
        RAGPipeline(
            context_provider=None,
            generation_service=(
                FakeGenerationService(
                    "Answer."
                )
            ),
        )


def test_pipeline_close_delegates_to_generation_service() -> None:
    """Application shutdown should release provider resources."""

    generation_service = (
        FakeGenerationService(
            "Answer."
        )
    )

    pipeline = RAGPipeline(
        context_provider=Mock(),
        generation_service=(
            generation_service
        ),
    )

    pipeline.close()

    assert (
        generation_service.closed
        is True
    )


def test_production_pipeline_wires_gemini_and_grounded_prompt(
    monkeypatch,
) -> None:
    """Production construction should connect the selected generation stack."""

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    pipeline = (
        build_production_rag_pipeline()
    )

    assert isinstance(
        pipeline.generation_service,
        GeminiGenerationService,
    )

    assert isinstance(
        pipeline
        .generation_service
        .prompt_builder,
        GroundedPromptBuilder,
    )

    pipeline.close()