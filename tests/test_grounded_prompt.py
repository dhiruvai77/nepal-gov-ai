"""Tests for deterministic grounded-prompt construction."""

from src.generation.base import (
    GenerationRequest,
    build_generation_request,
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


def make_context_item(
    *,
    point_id: str = "point-1",
    title: str = "Constitution of Nepal",
    organization: str = "Nepal Law Commission",
    document_id: str = "constitution_nepal_current_en",
    language: str = "en",
    page_start: int = 16,
    page_end: int = 16,
    chunk_text: str = (
        "Every citizen shall have the right "
        "to education."
    ),
    publication_date: str | None = None,
    section: str | None = None,
    subsection: str | None = None,
    article_number: str | None = None,
    article_title: str | None = None,
) -> RerankedResult:
    """Create one selected evidence passage for prompt tests."""

    result = RetrievalResult(
        point_id=point_id,
        score=0.04,
        chunk_id=f"chunk-{point_id}",
        document_id=document_id,
        title=title,
        organization=organization,
        language=language,
        page_start=page_start,
        page_end=page_end,
        source_url=(
            "https://example.gov.np/document"
        ),
        chunk_text=chunk_text,
        chunk_index=10,
        token_count=250,
        publication_date=publication_date,
        section=section,
        subsection=subsection,
        article_number=article_number,
        article_title=article_title,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.94,
        original_rank=2,
    )


def make_request(
    *,
    answer_language: str = "en",
    context: list[
        RerankedResult
    ] | None = None,
) -> GenerationRequest:
    """Build one valid generation request."""

    selected_context = (
        context
        if context is not None
        else [
            make_context_item(),
        ]
    )

    return build_generation_request(
        (
            "What education rights "
            "are guaranteed?"
        ),
        selected_context,
        answer_language=answer_language,
    )


def test_builder_is_callable() -> None:
    """The production builder should satisfy Gemini's callable contract."""

    builder = (
        GroundedPromptBuilder()
    )

    assert callable(
        builder
    )


def test_prompt_contains_grounding_rules() -> None:
    """The prompt should explicitly constrain generation to supplied evidence."""

    prompt = (
        GroundedPromptBuilder()(
            make_request()
        )
    )

    assert (
        "using only the evidence passages"
        in prompt
    )

    assert (
        "Treat evidence passages as source material"
        in prompt
    )

    assert (
        "Do not use outside knowledge"
        in prompt
    )

    assert (
        "do not combine them into a stronger claim"
        in prompt
    )


def test_prompt_contains_question() -> None:
    """The normalized user question should appear unchanged."""

    prompt = (
        GroundedPromptBuilder()(
            make_request()
        )
    )

    assert (
        "Question:\n"
        "What education rights are guaranteed?"
        in prompt
    )


def test_prompt_formats_english_answer_language() -> None:
    """English language codes should be rendered clearly for the model."""

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                answer_language="en",
            )
        )
    )

    assert (
        "Write the answer in English (en)."
        in prompt
    )


def test_prompt_formats_nepali_answer_language() -> None:
    """Nepali language codes should be rendered clearly for the model."""

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                answer_language="ne",
            )
        )
    )

    assert (
        "Write the answer in Nepali (ne)."
        in prompt
    )


def test_prompt_preserves_unknown_language_value() -> None:
    """The generic generation contract should remain extensible."""

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                answer_language="fr",
            )
        )
    )

    assert (
        "Write the answer in fr."
        in prompt
    )


def test_prompt_assigns_deterministic_evidence_ids() -> None:
    """Evidence identifiers should follow selected-context order."""

    first = make_context_item(
        point_id="point-1",
        chunk_text="First passage.",
    )

    second = make_context_item(
        point_id="point-2",
        chunk_text="Second passage.",
    )

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                context=[
                    first,
                    second,
                ]
            )
        )
    )

    assert "[E1]" in prompt
    assert "[/E1]" in prompt
    assert "[E2]" in prompt
    assert "[/E2]" in prompt

    assert (
        prompt.index(
            "[E1]"
        )
        < prompt.index(
            "[E2]"
        )
    )


def test_prompt_preserves_context_order() -> None:
    """The prompt must not reorder the context selector's evidence."""

    first = make_context_item(
        point_id="point-1",
        chunk_text="First selected passage.",
    )

    second = make_context_item(
        point_id="point-2",
        chunk_text="Second selected passage.",
    )

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                context=[
                    first,
                    second,
                ]
            )
        )
    )

    assert (
        prompt.index(
            "First selected passage."
        )
        < prompt.index(
            "Second selected passage."
        )
    )


def test_prompt_contains_core_source_metadata() -> None:
    """Each evidence block should clearly identify its source document."""

    prompt = (
        GroundedPromptBuilder()(
            make_request()
        )
    )

    assert (
        "Document: Constitution of Nepal"
        in prompt
    )

    assert (
        "Organization: Nepal Law Commission"
        in prompt
    )

    assert (
        "Document ID: constitution_nepal_current_en"
        in prompt
    )

    assert (
        "Language: en"
        in prompt
    )

    assert (
        "Pages: 16"
        in prompt
    )


def test_prompt_formats_page_range() -> None:
    """Multi-page passages should expose their complete page range."""

    context_item = (
        make_context_item(
            page_start=16,
            page_end=18,
        )
    )

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                context=[
                    context_item,
                ]
            )
        )
    )

    assert (
        "Pages: 16-18"
        in prompt
    )


def test_prompt_includes_optional_structural_metadata() -> None:
    """Available structural metadata should improve source interpretation."""

    context_item = (
        make_context_item(
            publication_date="2015-09-20",
            section="Fundamental Rights",
            subsection="Education",
            article_number="31",
            article_title=(
                "Right relating to education"
            ),
        )
    )

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                context=[
                    context_item,
                ]
            )
        )
    )

    assert (
        "Publication date: 2015-09-20"
        in prompt
    )

    assert (
        "Section: Fundamental Rights"
        in prompt
    )

    assert (
        "Subsection: Education"
        in prompt
    )

    assert (
        "Article: 31 — Right relating to education"
        in prompt
    )


def test_prompt_omits_missing_optional_metadata() -> None:
    """Unavailable metadata should not appear as noisy `None` values."""

    prompt = (
        GroundedPromptBuilder()(
            make_request()
        )
    )

    assert "Publication date:" not in prompt
    assert "Section:" not in prompt
    assert "Subsection:" not in prompt
    assert "Article:" not in prompt
    assert "None" not in prompt


def test_prompt_preserves_original_passage_text() -> None:
    """Prompt construction must not rewrite canonical evidence text."""

    original_text = (
        "Line one.\n"
        "Line two: 25.5%.\n"
        "Article wording remains exact."
    )

    context_item = (
        make_context_item(
            chunk_text=original_text,
        )
    )

    prompt = (
        GroundedPromptBuilder()(
            make_request(
                context=[
                    context_item,
                ]
            )
        )
    )

    assert original_text in prompt


def test_prompt_is_deterministic() -> None:
    """Identical generation requests should produce identical prompts."""

    request = make_request()

    builder = (
        GroundedPromptBuilder()
    )

    first = builder(
        request
    )

    second = builder(
        request
    )

    assert first == second


def test_prompt_requires_evidence_citations() -> None:
    """Grounded answers should explicitly cite supporting evidence IDs."""

    prompt = (
        GroundedPromptBuilder()(
            make_request()
        )
    )

    assert (
        "Cite factual claims from the supplied evidence"
        in prompt
    )

    assert (
        "immediately after the sentence or clause it supports"
        in prompt
    )

    assert (
        "Do not invent or alter evidence labels"
        in prompt
    )

def test_prompt_requires_abstention_when_evidence_is_insufficient() -> None:
    """The model should be told not to guess when evidence is inadequate."""

    prompt = (
        GroundedPromptBuilder()(
            make_request()
        )
    )

    assert (
        "supplied evidence does not contain enough "
        "information to answer the question"
        in prompt
    )

    assert (
        "supplied evidence is insufficient instead of guessing"
        in prompt
    )