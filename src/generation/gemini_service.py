"""Gemini generation provider for NepalGov AI.

This provider implements the generic GenerationService contract using the
Google Gen AI Python SDK.

The provider is responsible only for Gemini transport and response handling.
Grounded prompt construction remains a separate dependency so prompt design can
be implemented and evaluated independently from provider infrastructure.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from src.generation.base import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    build_generation_result,
)


GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

# Use a specific stable model rather than a moving `latest` alias so benchmark
# behavior remains reproducible across repeated evaluations.
MODEL_NAME = "gemini-3.8-flash"

DEFAULT_TIMEOUT_SECONDS = 60.0

PROVIDER_NAME = "gemini"


# Prompt construction deliberately remains outside the Gemini provider. The
# grounded-prompt milestone will provide the production implementation.
PromptBuilder = Callable[
    [
        GenerationRequest,
    ],
    str,
]


class GeminiGenerationService(
    GenerationService
):
    """Generate answers through the Gemini Developer API."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder,
        api_key: str | None = None,
        model_name: str = MODEL_NAME,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any | None = None,
    ) -> None:
        """Configure Gemini generation without making an API request.

        `prompt_builder` converts the generic GenerationRequest into provider
        input. Keeping it injectable prevents Gemini transport code from owning
        the project's grounded-prompt policy.

        `client` can be injected by tests or application code. When no client
        is provided, the official Google Gen AI SDK client is created lazily.
        """

        if not callable(
            prompt_builder
        ):
            raise TypeError(
                "prompt_builder must be callable."
            )

        clean_model_name = (
            model_name.strip()
        )

        if not clean_model_name:
            raise ValueError(
                "model_name must contain non-whitespace text."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        resolved_api_key = (
            api_key
            if api_key is not None
            else os.getenv(
                GEMINI_API_KEY_ENV
            )
        )

        # An injected fake/test client does not need credentials. Production
        # construction does, so fail before an external request is attempted.
        if (
            client is None
            and not resolved_api_key
        ):
            raise ValueError(
                f"{GEMINI_API_KEY_ENV} must be set "
                "for Gemini generation."
            )

        if (
            resolved_api_key is not None
            and not resolved_api_key.strip()
            and client is None
        ):
            raise ValueError(
                f"{GEMINI_API_KEY_ENV} must contain "
                "non-whitespace text."
            )

        self.prompt_builder = (
            prompt_builder
        )

        self.api_key = (
            resolved_api_key.strip()
            if resolved_api_key
            else None
        )

        self.model_name = (
            clean_model_name
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self._client = client

    def _get_client(
        self,
    ) -> Any:
        """Create the official Gemini client lazily."""

        if self._client is not None:
            return self._client

        try:
            from google import genai
            from google.genai import types

        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required "
                "for Gemini generation."
            ) from exc

        # Gemini 3.8 Flash is available through the stable API. Selecting v1
        # avoids silently depending on beta API behavior.
        #
        # The SDK uses httpx internally. Supplying the timeout through
        # client_args keeps connection behavior explicit while leaving
        # transient-error retries to the SDK's built-in retry handling.
        http_options = types.HttpOptions(
            api_version="v1",
            client_args={
                "timeout": (
                    self.timeout_seconds
                ),
            },
        )

        self._client = genai.Client(
            api_key=self.api_key,
            http_options=http_options,
        )

        return self._client

    def _build_prompt(
        self,
        request: GenerationRequest,
    ) -> str:
        """Build and validate provider input using the injected prompt builder."""

        prompt = self.prompt_builder(
            request
        )

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt_builder must return a string."
            )

        clean_prompt = (
            prompt.strip()
        )

        if not clean_prompt:
            raise ValueError(
                "prompt_builder returned empty prompt text."
            )

        return clean_prompt

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Generate one answer through Gemini."""

        prompt = self._build_prompt(
            request
        )

        client = self._get_client()

        try:
            response = (
                client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
            )

        except Exception as exc:
            # Provider-specific SDK exceptions must not leak into application
            # orchestration. Preserve the original error as the cause for
            # logging and debugging.
            raise RuntimeError(
                "Gemini generation request failed."
            ) from exc

        answer_text = getattr(
            response,
            "text",
            None,
        )

        if not isinstance(
            answer_text,
            str,
        ):
            raise RuntimeError(
                "Gemini response contained no text output."
            )

        try:
            return build_generation_result(
                answer_text,
                provider=PROVIDER_NAME,
                model=self.model_name,
            )

        except ValueError as exc:
            raise RuntimeError(
                "Gemini response contained no usable text output."
            ) from exc

    def close(
        self,
    ) -> None:
        """Release underlying Gemini SDK network resources when available."""

        if self._client is None:
            return

        close_method = getattr(
            self._client,
            "close",
            None,
        )

        if callable(
            close_method
        ):
            close_method()