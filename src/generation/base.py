"""Provider-independent generation contracts for NepalGov AI.

The generation layer receives the evidence passages selected by the retrieval,
reranking, and context-selection pipeline and produces answer text without
coupling downstream RAG orchestration to a specific LLM provider.

This module intentionally contains no Gemini-specific behavior, prompt
construction, citation formatting, or insufficient-evidence policy. Those
concerns belong to later, independently testable layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from src.reranking.base import (
    RerankedResult,
)


@dataclass(frozen=True)
class GenerationRequest:
    """Provider-independent input supplied to a generation service.

    `query` is the normalized user question.

    `context` contains the selected evidence passages in context-selection
    order. Each item retains the original source passage and citation metadata.

    `answer_language` identifies the language in which the final answer should
    be generated. The generic contract does not restrict providers to only the
    current English/Nepali V1 languages.
    """

    query: str
    context: tuple[
        RerankedResult,
        ...
    ]
    answer_language: str


@dataclass(frozen=True)
class GenerationResult:
    """Provider-independent generation output.

    `answer_text` contains the generated answer.

    `provider` and `model` are optional provenance fields that concrete
    providers may populate for logging, diagnostics, and evaluation without
    requiring downstream code to understand provider-specific response types.
    """

    answer_text: str
    provider: str | None = None
    model: str | None = None


class GenerationService(ABC):
    """Abstract interface implemented by generation providers."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Generate one answer from a validated request."""


def build_generation_request(
    query: str,
    context: Sequence[
        RerankedResult
    ],
    *,
    answer_language: str,
) -> GenerationRequest:
    """Validate and normalize provider-independent generation input.

    Generation requires selected evidence. Empty-context behavior belongs to
    the later insufficient-evidence layer rather than allowing an LLM provider
    to generate an ungrounded answer.

    Evidence objects are preserved exactly and converted only to an immutable
    tuple so providers cannot accidentally alter the selected context list.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "query must contain non-whitespace text."
        )

    clean_answer_language = (
        answer_language.strip().lower()
    )

    if not clean_answer_language:
        raise ValueError(
            "answer_language must contain "
            "non-whitespace text."
        )

    if not context:
        raise ValueError(
            "at least one selected evidence "
            "passage is required."
        )

    return GenerationRequest(
        query=clean_query,
        context=tuple(
            context
        ),
        answer_language=clean_answer_language,
    )


def build_generation_result(
    answer_text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> GenerationResult:
    """Validate and normalize common generation output fields."""

    clean_answer = answer_text.strip()

    if not clean_answer:
        raise ValueError(
            "answer_text must contain "
            "non-whitespace text."
        )

    clean_provider = (
        provider.strip()
        if provider is not None
        and provider.strip()
        else None
    )

    clean_model = (
        model.strip()
        if model is not None
        and model.strip()
        else None
    )

    return GenerationResult(
        answer_text=clean_answer,
        provider=clean_provider,
        model=clean_model,
    )