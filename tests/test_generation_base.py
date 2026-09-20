"""Tests for the provider-independent generation contract."""

import pytest

from src.generation.base import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    build_generation_request,
    build_generation_result,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_context_item(
    point_id: str = "point-1",
) -> RerankedResult:
    """Create one selected evidence passage for generation tests."""

    result = RetrievalResult(
        point_id=point_id,
        score=0.031,
        chunk_id=f"chunk-{point_id}",
        document_id="constitution_nepal_current_en",
        title="Constitution of Nepal",
        organization="Nepal Law Commission",
        language="en",
        page_start=16,
        page_end=16,
        source_url="https://example.gov.np/constitution",
        chunk_text=(
            "Every citizen shall have the right "
            "to education."
        ),
        chunk_index=10,
        token_count=250,
        category="constitution_law",
        document_type="constitution",
    )

    return RerankedResult(
        result=result,
        rerank_score=0.92,
        original_rank=3,
    )


def test_generation_service_is_abstract() -> None:
    """Concrete providers must implement the generation interface."""

    with pytest.raises(
        TypeError
    ):
        GenerationService()


def test_build_generation_request_normalizes_query_and_language() -> None:
    """Common generation input should be normalized before provider use."""

    context_item = (
        make_context_item()
    )

    request = build_generation_request(
        "  What education rights are guaranteed?  ",
        [
            context_item,
        ],
        answer_language=" EN ",
    )

    assert isinstance(
        request,
        GenerationRequest,
    )

    assert (
        request.query
        == "What education rights are guaranteed?"
    )

    assert (
        request.answer_language
        == "en"
    )

    assert (
        request.context
        == (
            context_item,
        )
    )


def test_build_generation_request_preserves_evidence_object() -> None:
    """Generation must retain original evidence and citation provenance."""

    context_item = (
        make_context_item()
    )

    request = build_generation_request(
        "education rights",
        [
            context_item,
        ],
        answer_language="en",
    )

    selected = (
        request.context[
            0
        ]
    )

    assert (
        selected
        is context_item
    )

    assert (
        selected.result.chunk_text
        == (
            "Every citizen shall have the right "
            "to education."
        )
    )

    assert (
        selected.result.title
        == "Constitution of Nepal"
    )

    assert (
        selected.result.page_start
        == 16
    )

    assert (
        selected.result.chunk_index
        == 10
    )

    assert (
        selected.result.token_count
        == 250
    )


def test_build_generation_request_preserves_context_order() -> None:
    """Generation should receive evidence in context-selector order."""

    first = make_context_item(
        "point-1"
    )

    second = make_context_item(
        "point-2"
    )

    request = build_generation_request(
        "education rights",
        [
            first,
            second,
        ],
        answer_language="en",
    )

    assert [
        item.result.point_id
        for item in request.context
    ] == [
        "point-1",
        "point-2",
    ]


def test_build_generation_request_uses_immutable_context_container() -> None:
    """Provider input should not expose a mutable selected-context list."""

    context = [
        make_context_item(
            "point-1"
        ),
    ]

    request = build_generation_request(
        "education rights",
        context,
        answer_language="en",
    )

    context.append(
        make_context_item(
            "point-2"
        )
    )

    assert len(
        request.context
    ) == 1

    assert isinstance(
        request.context,
        tuple,
    )


def test_build_generation_request_rejects_blank_query() -> None:
    """Blank questions should fail before reaching an LLM provider."""

    with pytest.raises(
        ValueError,
        match="query must contain non-whitespace text",
    ):
        build_generation_request(
            "   ",
            [
                make_context_item(),
            ],
            answer_language="en",
        )


def test_build_generation_request_rejects_blank_answer_language() -> None:
    """Generation requires an explicit nonblank answer-language value."""

    with pytest.raises(
        ValueError,
        match=(
            "answer_language must contain "
            "non-whitespace text"
        ),
    ):
        build_generation_request(
            "education rights",
            [
                make_context_item(),
            ],
            answer_language="   ",
        )


def test_build_generation_request_rejects_empty_context() -> None:
    """Ungrounded generation should not reach a provider accidentally."""

    with pytest.raises(
        ValueError,
        match=(
            "at least one selected evidence "
            "passage is required"
        ),
    ):
        build_generation_request(
            "education rights",
            [],
            answer_language="en",
        )


def test_build_generation_result_normalizes_common_fields() -> None:
    """Provider output should use one stable downstream result contract."""

    result = build_generation_result(
        "  Citizens have a right to education.  ",
        provider=" gemini ",
        model=" gemini-example ",
    )

    assert isinstance(
        result,
        GenerationResult,
    )

    assert (
        result.answer_text
        == "Citizens have a right to education."
    )

    assert (
        result.provider
        == "gemini"
    )

    assert (
        result.model
        == "gemini-example"
    )


def test_build_generation_result_allows_missing_provenance() -> None:
    """Generic generation results need not depend on provider metadata."""

    result = build_generation_result(
        "Grounded answer."
    )

    assert (
        result.provider
        is None
    )

    assert (
        result.model
        is None
    )


def test_build_generation_result_normalizes_blank_provenance_to_none() -> None:
    """Blank optional provider metadata should not leak downstream."""

    result = build_generation_result(
        "Grounded answer.",
        provider="   ",
        model="   ",
    )

    assert (
        result.provider
        is None
    )

    assert (
        result.model
        is None
    )


def test_build_generation_result_rejects_blank_answer() -> None:
    """A provider returning no usable answer should fail explicitly."""

    with pytest.raises(
        ValueError,
        match=(
            "answer_text must contain "
            "non-whitespace text"
        ),
    ):
        build_generation_result(
            "   "
        )


class FakeGenerationService(
    GenerationService
):
    """Minimal concrete provider used to verify the abstract contract."""

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Return deterministic output for the supplied request."""

        return build_generation_result(
            (
                f"Answer to: "
                f"{request.query}"
            ),
            provider="fake",
            model="fake-model",
        )


def test_concrete_generation_service_uses_shared_contract() -> None:
    """Concrete providers should consume and return generic contracts."""

    request = build_generation_request(
        "What education rights are guaranteed?",
        [
            make_context_item(),
        ],
        answer_language="en",
    )

    service = (
        FakeGenerationService()
    )

    result = service.generate(
        request
    )

    assert (
        result.answer_text
        == (
            "Answer to: "
            "What education rights are guaranteed?"
        )
    )

    assert (
        result.provider
        == "fake"
    )

    assert (
        result.model
        == "fake-model"
    )