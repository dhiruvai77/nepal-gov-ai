"""End-to-end evidence-grounded RAG orchestration for NepalGov AI.

This module composes the independently tested retrieval, context-selection,
generation, citation-processing, and evidence-guard stages into one application
pipeline.

It deliberately contains orchestration rather than duplicating stage-specific
logic.
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from dataclasses import dataclass
from typing import (
    Protocol,
)

from src.citations.evidence import (
    CitationProcessingResult,
    process_answer_citations,
    render_cited_sources,
)
from src.context_selection.run_context_selection import (
    run_context_selection,
)
from src.generation.base import (
    GenerationService,
    build_generation_request,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
    build_no_evidence_result,
    guard_generated_answer,
)
from src.generation.gemini_service import (
    GeminiGenerationService,
)
from src.generation.grounded_prompt import (
    GroundedPromptBuilder,
)
from src.reranking.base import (
    RerankedResult,
)


class ContextProvider(
    Protocol
):
    """Callable contract for obtaining selected generation context."""

    def __call__(
        self,
        query: str,
        *,
        filters: dict[
            str,
            str,
        ]
        | None = None,
    ) -> Sequence[
        RerankedResult
    ]:
        """Return selected evidence for one normalized query."""


@dataclass(frozen=True)
class RAGResult:
    """Final application-level result returned by the RAG pipeline.

    `answer_text` contains either the accepted generated answer or the
    deterministic insufficient-evidence response.

    `sources` contains rendered canonical source metadata only for evidence
    identifiers that were validated against the selected context.

    `selected_context` preserves the exact evidence supplied to generation.

    `citation_result` preserves machine-readable citation diagnostics for
    evaluation and application observability. It is None when generation was
    skipped because no evidence was selected.

    `provider` and `model` remain None when generation was skipped because no
    evidence was selected.
    """

    answer_text: str
    accepted: bool
    reason: (
        EvidenceGuardReason
        | None
    )
    sources: tuple[
        str,
        ...
    ]
    selected_context: tuple[
        RerankedResult,
        ...
    ]
    citation_result: (
        CitationProcessingResult
        | None
    ) = None
    provider: str | None = None
    model: str | None = None

    @property
    def withheld(
        self,
    ) -> bool:
        """Return whether a generated answer was withheld or skipped."""

        return not self.accepted


class RAGPipeline:
    """Compose selected evidence, generation, citations, and evidence gating."""

    def __init__(
        self,
        *,
        context_provider: ContextProvider,
        generation_service: GenerationService,
    ) -> None:
        """Store independently testable RAG stage dependencies."""

        if not callable(
            context_provider
        ):
            raise TypeError(
                "context_provider must be callable."
            )

        self.context_provider = (
            context_provider
        )

        self.generation_service = (
            generation_service
        )

    @staticmethod
    def _normalize_query(
        query: str,
    ) -> str:
        """Validate and normalize one user query before retrieval."""

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        clean_query = (
            query.strip()
        )

        if not clean_query:
            raise ValueError(
                "query must contain non-whitespace text."
            )

        return clean_query

    @staticmethod
    def _normalize_answer_language(
        answer_language: str,
    ) -> str:
        """Validate the requested answer language."""

        if not isinstance(
            answer_language,
            str,
        ):
            raise TypeError(
                "answer_language must be a string."
            )

        clean_language = (
            answer_language
            .strip()
            .lower()
        )

        if not clean_language:
            raise ValueError(
                "answer_language must contain "
                "non-whitespace text."
            )

        return clean_language

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
        """Run the complete evidence-grounded answer pipeline."""

        clean_query = (
            self._normalize_query(
                query
            )
        )

        clean_answer_language = (
            self._normalize_answer_language(
                answer_language
            )
        )

        selected_context = tuple(
            self.context_provider(
                clean_query,
                filters=filters,
            )
        )

        # Retrieval with no selected evidence is a valid pipeline outcome.
        # Stop here rather than sending an evidence-free request to an LLM.
        if not selected_context:
            guarded = (
                build_no_evidence_result(
                    answer_language=(
                        clean_answer_language
                    ),
                )
            )

            return RAGResult(
                answer_text=(
                    guarded.answer_text
                ),
                accepted=(
                    guarded.accepted
                ),
                reason=(
                    guarded.reason
                ),
                sources=(),
                selected_context=(),
                citation_result=None,
                provider=None,
                model=None,
            )

        generation_request = (
            build_generation_request(
                clean_query,
                selected_context,
                answer_language=(
                    clean_answer_language
                ),
            )
        )

        generation_result = (
            self.generation_service.generate(
                generation_request
            )
        )

        citation_result = (
            process_answer_citations(
                generation_result.answer_text,
                generation_request.context,
            )
        )

        guarded = (
            guard_generated_answer(
                citation_result,
                answer_language=(
                    clean_answer_language
                ),
            )
        )

        # A withheld answer should not expose a source list for model output
        # that failed citation validation.
        sources = (
            render_cited_sources(
                citation_result
            )
            if guarded.accepted
            else ()
        )

        return RAGResult(
            answer_text=(
                guarded.answer_text
            ),
            accepted=(
                guarded.accepted
            ),
            reason=(
                guarded.reason
            ),
            sources=sources,
            selected_context=(
                generation_request.context
            ),
            citation_result=(
                citation_result
            ),
            provider=(
                generation_result.provider
            ),
            model=(
                generation_result.model
            ),
        )

    def close(
        self,
    ) -> None:
        """Release provider resources when the generation service supports it."""

        close_method = getattr(
            self.generation_service,
            "close",
            None,
        )

        if callable(
            close_method
        ):
            close_method()


def build_production_rag_pipeline() -> RAGPipeline:
    """Construct the currently selected production RAG stack."""

    generation_service = (
        GeminiGenerationService(
            prompt_builder=(
                GroundedPromptBuilder()
            ),
        )
    )

    return RAGPipeline(
        context_provider=(
            run_context_selection
        ),
        generation_service=(
            generation_service
        ),
    )