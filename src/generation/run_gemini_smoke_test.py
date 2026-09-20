"""Live grounded-generation smoke test for NepalGov AI.

This module performs one real Gemini request using the production
GroundedPromptBuilder and GeminiGenerationService.

The evidence is intentionally synthetic so the smoke test measures whether the
generation stack follows supplied evidence rather than relying on model
knowledge. No credentials are printed or stored by this module.

The model can be overridden through GEMINI_SMOKE_MODEL for availability
diagnostics without changing the production model configured by
GeminiGenerationService.
"""

from __future__ import annotations

import os

from src.generation.base import (
    build_generation_request,
)
from src.generation.gemini_service import (
    MODEL_NAME,
    GeminiGenerationService,
)
from src.generation.grounded_prompt import (
    GroundedPromptBuilder,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


SMOKE_MODEL_ENV = "GEMINI_SMOKE_MODEL"


def build_smoke_test_context() -> list[
    RerankedResult
]:
    """Create deterministic synthetic evidence for the live smoke test."""

    evidence = RetrievalResult(
        point_id="smoke-test-point-1",
        score=1.0,
        chunk_id="smoke-test-chunk-1",
        document_id="synthetic_smoke_test_document",
        title="Synthetic NepalGov AI Smoke Test Document",
        organization="NepalGov AI Test Fixture",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.invalid/smoke-test",
        chunk_text=(
            "For this synthetic smoke test only, "
            "the public service desk operates Monday through Thursday "
            "from 09:00 to 15:00. "
            "It is closed on Friday."
        ),
        chunk_index=0,
        token_count=35,
        category="smoke_test",
        document_type="synthetic_test_fixture",
    )

    return [
        RerankedResult(
            result=evidence,
            rerank_score=1.0,
            original_rank=1,
        )
    ]


def resolve_smoke_model() -> str:
    """Resolve an optional diagnostic model override."""

    override = os.getenv(
        SMOKE_MODEL_ENV
    )

    if override is None:
        return MODEL_NAME

    clean_override = override.strip()

    if not clean_override:
        return MODEL_NAME

    return clean_override


def main() -> None:
    """Run one real grounded Gemini request and print safe diagnostics."""

    request = build_generation_request(
        (
            "According to the supplied evidence, "
            "when does the public service desk operate, "
            "and is it open on Friday?"
        ),
        build_smoke_test_context(),
        answer_language="en",
    )

    smoke_model = (
        resolve_smoke_model()
    )

    service = GeminiGenerationService(
        prompt_builder=GroundedPromptBuilder(),
        model_name=smoke_model,
    )

    try:
        result = service.generate(
            request
        )

        print(
            "LIVE GEMINI GROUNDED SMOKE TEST"
        )
        print(
            "Provider:",
            result.provider,
        )
        print(
            "Model:",
            result.model,
        )
        print(
            "Answer:"
        )
        print(
            result.answer_text
        )

    finally:
        service.close()


if __name__ == "__main__":
    main()