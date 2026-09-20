"""Tests for the Gemini generation provider."""

from types import SimpleNamespace
from unittest.mock import (
    Mock,
)

import pytest

from src.generation.base import (
    GenerationRequest,
    build_generation_request,
)
from src.generation.gemini_service import (
    DEFAULT_TIMEOUT_SECONDS,
    GEMINI_API_KEY_ENV,
    MODEL_NAME,
    PROVIDER_NAME,
    GeminiGenerationService,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_context_item() -> RerankedResult:
    """Create one selected evidence passage."""

    result = RetrievalResult(
        point_id="point-1",
        score=0.04,
        chunk_id="constitution-chunk-1",
        document_id="constitution_nepal_current_en",
        title="Constitution of Nepal",
        organization="Nepal Law Commission",
        language="en",
        page_start=16,
        page_end=16,
        source_url="https://example.gov.np/constitution",
        chunk_text=(
            "Every citizen shall have "
            "the right to education."
        ),
        chunk_index=10,
        token_count=250,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.94,
        original_rank=2,
    )


def make_request() -> GenerationRequest:
    """Create one generic generation request."""

    return build_generation_request(
        "What education rights are guaranteed?",
        [
            make_context_item(),
        ],
        answer_language="en",
    )


def make_prompt(
    request: GenerationRequest,
) -> str:
    """Return deterministic provider input for unit tests."""

    return (
        f"Question: {request.query}"
    )


def make_client(
    answer_text: str = "Citizens have a right to education.",
):
    """Create a fake Gemini Interactions client."""

    create_interaction = Mock(
        return_value=SimpleNamespace(
            output_text=answer_text,
        )
    )

    client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=create_interaction,
        ),
        close=Mock(),
    )

    return (
        client,
        create_interaction,
    )


def test_default_model_is_stable_gemini_flash() -> None:
    """Production should use the selected stable Gemini model."""

    assert (
        MODEL_NAME
        == "gemini-3.8-flash"
    )


def test_default_timeout_is_sixty_seconds() -> None:
    """Provider requests should have an explicit network timeout."""

    assert (
        DEFAULT_TIMEOUT_SECONDS
        == 60.0
    )


def test_provider_name_is_gemini() -> None:
    """Generation results should expose stable provider provenance."""

    assert (
        PROVIDER_NAME
        == "gemini"
    )


def test_service_requires_api_key_without_injected_client(
    monkeypatch,
) -> None:
    """Production construction should fail before unauthenticated requests."""

    monkeypatch.delenv(
        GEMINI_API_KEY_ENV,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "GEMINI_API_KEY must be set"
        ),
    ):
        GeminiGenerationService(
            prompt_builder=make_prompt,
        )


def test_service_reads_api_key_from_environment(
    monkeypatch,
) -> None:
    """Production credentials should be resolved from the environment."""

    monkeypatch.setenv(
        GEMINI_API_KEY_ENV,
        "test-api-key",
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
        )
    )

    assert (
        service.api_key
        == "test-api-key"
    )


def test_explicit_api_key_overrides_environment(
    monkeypatch,
) -> None:
    """Explicit dependency injection should override environment config."""

    monkeypatch.setenv(
        GEMINI_API_KEY_ENV,
        "environment-key",
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            api_key="explicit-key",
        )
    )

    assert (
        service.api_key
        == "explicit-key"
    )


def test_injected_client_does_not_require_api_key(
    monkeypatch,
) -> None:
    """Unit tests should not depend on real Gemini credentials."""

    monkeypatch.delenv(
        GEMINI_API_KEY_ENV,
        raising=False,
    )

    client, _ = make_client()

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            client=client,
        )
    )

    assert (
        service.api_key
        is None
    )


def test_service_rejects_blank_model_name() -> None:
    """Gemini requests require a concrete model identifier."""

    client, _ = make_client()

    with pytest.raises(
        ValueError,
        match=(
            "model_name must contain "
            "non-whitespace text"
        ),
    ):
        GeminiGenerationService(
            prompt_builder=make_prompt,
            model_name="   ",
            client=client,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
    ],
)
def test_service_rejects_invalid_timeout(
    timeout_seconds: float,
) -> None:
    """Network timeout must always be positive."""

    client, _ = make_client()

    with pytest.raises(
        ValueError,
        match=(
            "timeout_seconds must be "
            "greater than zero"
        ),
    ):
        GeminiGenerationService(
            prompt_builder=make_prompt,
            timeout_seconds=timeout_seconds,
            client=client,
        )


def test_service_rejects_noncallable_prompt_builder() -> None:
    """Provider construction requires a prompt-building dependency."""

    client, _ = make_client()

    with pytest.raises(
        TypeError,
        match=(
            "prompt_builder must be callable"
        ),
    ):
        GeminiGenerationService(
            prompt_builder=None,
            client=client,
        )


def test_generate_uses_prompt_builder_and_model() -> None:
    """Gemini should receive exactly the provider input built upstream."""

    client, create_interaction = (
        make_client()
    )

    prompt_builder = Mock(
        return_value=(
            "Grounded provider input."
        )
    )

    service = (
        GeminiGenerationService(
            prompt_builder=prompt_builder,
            client=client,
        )
    )

    request = make_request()

    result = service.generate(
        request
    )

    prompt_builder.assert_called_once_with(
        request
    )

    create_interaction.assert_called_once_with(
        model=MODEL_NAME,
        input="Grounded provider input.",
    )

    assert (
        result.answer_text
        == "Citizens have a right to education."
    )

    assert (
        result.provider
        == PROVIDER_NAME
    )

    assert (
        result.model
        == MODEL_NAME
    )


def test_generate_preserves_custom_model_provenance() -> None:
    """Configured model identity should be retained in the result."""

    client, create_interaction = (
        make_client(
            "Answer."
        )
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            model_name="custom-gemini-model",
            client=client,
        )
    )

    result = service.generate(
        make_request()
    )

    create_interaction.assert_called_once_with(
        model="custom-gemini-model",
        input=(
            "Question: "
            "What education rights are guaranteed?"
        ),
    )

    assert (
        result.model
        == "custom-gemini-model"
    )


def test_generate_rejects_non_string_prompt() -> None:
    """A malformed prompt builder should fail before external inference."""

    client, create_interaction = (
        make_client()
    )

    service = (
        GeminiGenerationService(
            prompt_builder=lambda request: None,
            client=client,
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "prompt_builder must return "
            "a string"
        ),
    ):
        service.generate(
            make_request()
        )

    create_interaction.assert_not_called()


def test_generate_rejects_blank_prompt() -> None:
    """Empty provider input should never be sent to Gemini."""

    client, create_interaction = (
        make_client()
    )

    service = (
        GeminiGenerationService(
            prompt_builder=lambda request: "   ",
            client=client,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "prompt_builder returned "
            "empty prompt text"
        ),
    ):
        service.generate(
            make_request()
        )

    create_interaction.assert_not_called()


def test_generate_wraps_provider_failure() -> None:
    """Provider SDK errors should not leak through the generic RAG boundary."""

    client, create_interaction = (
        make_client()
    )

    create_interaction.side_effect = (
        RuntimeError(
            "provider failed"
        )
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            client=client,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Gemini generation request failed"
        ),
    ):
        service.generate(
            make_request()
        )


def test_generate_rejects_missing_response_text() -> None:
    """A Gemini interaction without text is not a valid generation result."""

    client, create_interaction = (
        make_client()
    )

    create_interaction.return_value = (
        SimpleNamespace(
            output_text=None,
        )
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            client=client,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Gemini response contained "
            "no text output"
        ),
    ):
        service.generate(
            make_request()
        )


def test_generate_rejects_blank_response_text() -> None:
    """Whitespace-only model output should fail explicitly."""

    client, _ = make_client(
        "   "
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            client=client,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Gemini response contained "
            "no usable text output"
        ),
    ):
        service.generate(
            make_request()
        )


def test_close_releases_injected_client() -> None:
    """Provider shutdown should release SDK networking resources."""

    client, _ = make_client()

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
            client=client,
        )
    )

    service.close()

    client.close.assert_called_once_with()


def test_close_before_client_creation_is_safe(
    monkeypatch,
) -> None:
    """Closing an unused lazily configured service should be harmless."""

    monkeypatch.setenv(
        GEMINI_API_KEY_ENV,
        "test-api-key",
    )

    service = (
        GeminiGenerationService(
            prompt_builder=make_prompt,
        )
    )

    service.close()

    assert (
        service._client
        is None
    )