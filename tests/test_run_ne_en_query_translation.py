"""Tests for automatic NE->EN retrieval-query translation."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from src.evaluation.run_ne_en_query_translation import (
    TRANSLATION_CONFIG_ID,
    TRANSLATION_SCHEMA_VERSION,
    build_output_record,
    build_translation_prompt,
    contains_devanagari,
    load_existing_output,
    normalize_translation,
    prompt_sha256,
    run_translation,
    translate_query,
    validate_persisted_row,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)


def make_record(
    *,
    question_id: str = "ne_en_001",
) -> EvaluationRecord:
    """Build one deterministic NE->EN benchmark record."""

    return EvaluationRecord(
        question_id=question_id,
        query=(
            "नेपालको संविधानले "
            "स्वास्थ्यसम्बन्धी अधिकारबारे "
            "के व्यवस्था गरेको छ?"
        ),
        query_language="ne",
        target_language="en",
        category="constitution_law",
        expected_document_ids=(
            "constitution_nepal_current_en",
        ),
        primary_relevant_chunk_ids=(
            "primary",
        ),
        relevant_chunk_ids=(
            "primary",
        ),
        notes="Test.",
    )


class FakeInteractions:
    """Minimal fake Gemini Interactions endpoint."""

    def __init__(
        self,
        outputs: list[str],
    ) -> None:
        self.outputs = list(
            outputs
        )

        self.calls: list[
            dict
        ] = []

    def create(
        self,
        *,
        model: str,
        input: str,
    ) -> SimpleNamespace:
        """Record the request and return the next fake output."""

        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        if not self.outputs:
            raise RuntimeError(
                "No fake outputs remain."
            )

        return (
            SimpleNamespace(
                output_text=(
                    self.outputs.pop(
                        0
                    )
                )
            )
        )


class FakeClient:
    """Minimal fake Gemini client."""

    def __init__(
        self,
        outputs: list[str],
    ) -> None:
        self.interactions = (
            FakeInteractions(
                outputs
            )
        )

        self.closed = False

    def close(
        self,
    ) -> None:
        """Track explicit close calls."""

        self.closed = True


def test_translation_prompt_preserves_translation_only_policy() -> None:
    """Prompt must prohibit answering and semantic expansion."""

    prompt = (
        build_translation_prompt(
            "नेपालको आर्थिक वृद्धि कति छ?"
        )
    )

    assert (
        "Do not answer the question."
        in prompt
    )

    assert (
        "Do not add facts"
        in prompt
    )

    assert (
        "Do not make the query more specific"
        in prompt
    )

    assert (
        "नेपालको आर्थिक वृद्धि कति छ?"
        in prompt
    )


def test_translation_prompt_rejects_blank_query() -> None:
    """Blank queries should fail before provider use."""

    with pytest.raises(
        ValueError,
        match=(
            "query must contain"
        ),
    ):
        build_translation_prompt(
            "   "
        )


def test_prompt_sha256_is_deterministic() -> None:
    """Prompt hashes should protect persisted translations from stale prompts."""

    prompt = "translation prompt"

    expected = (
        hashlib.sha256(
            prompt.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    assert (
        prompt_sha256(
            prompt
        )
        == expected
    )


def test_contains_devanagari() -> None:
    """Devanagari detection should distinguish source and translated text."""

    assert (
        contains_devanagari(
            "नेपाल"
        )
        is True
    )

    assert (
        contains_devanagari(
            "Nepal"
        )
        is False
    )


def test_normalize_translation_collapses_whitespace() -> None:
    """Provider output should be normalized to one retrieval-query line."""

    assert (
        normalize_translation(
            "  What   does Nepal\n"
            "guarantee?  "
        )
        == "What does Nepal guarantee?"
    )


def test_normalize_translation_rejects_blank() -> None:
    """Empty provider output should fail."""

    with pytest.raises(
        ValueError,
        match=(
            "non-whitespace"
        ),
    ):
        normalize_translation(
            "   "
        )


def test_normalize_translation_rejects_devanagari() -> None:
    """Automatic English output should not retain the Nepali script."""

    with pytest.raises(
        ValueError,
        match=(
            "Devanagari"
        ),
    ):
        normalize_translation(
            "What does नेपालको Constitution say?"
        )


def test_build_output_record_contains_stable_metadata() -> None:
    """Persisted translations should be reproducible and self-describing."""

    record = (
        make_record()
    )

    output = (
        build_output_record(
            record,
            translated_query=(
                "What does the Constitution "
                "of Nepal provide regarding "
                "the right to health?"
            ),
            model="test-model",
            provider="gemini",
        )
    )

    assert (
        output[
            "schema_version"
        ]
        == TRANSLATION_SCHEMA_VERSION
    )

    assert (
        output[
            "translation_config_id"
        ]
        == TRANSLATION_CONFIG_ID
    )

    assert (
        output[
            "question_id"
        ]
        == record.question_id
    )

    assert (
        output[
            "original_query"
        ]
        == record.query
    )

    assert (
        output[
            "model"
        ]
        == "test-model"
    )

    assert (
        len(
            output[
                "prompt_sha256"
            ]
        )
        == 64
    )


def test_validate_persisted_row_accepts_current_record() -> None:
    """A current persisted translation should be reusable."""

    record = (
        make_record()
    )

    row = (
        build_output_record(
            record,
            translated_query=(
                "What does the Constitution "
                "say about the right to health?"
            ),
            model="test-model",
            provider="gemini",
        )
    )

    validate_persisted_row(
        row,
        record,
        model="test-model",
        provider="gemini",
    )


def test_validate_persisted_row_rejects_stale_prompt_hash() -> None:
    """Prompt changes must invalidate old automatic translations."""

    record = (
        make_record()
    )

    row = (
        build_output_record(
            record,
            translated_query=(
                "What does the Constitution "
                "say about the right to health?"
            ),
            model="test-model",
            provider="gemini",
        )
    )

    row[
        "prompt_sha256"
    ] = (
        "0" * 64
    )

    with pytest.raises(
        ValueError,
        match=(
            "prompt_sha256"
        ),
    ):
        validate_persisted_row(
            row,
            record,
            model="test-model",
            provider="gemini",
        )


def test_translate_query_uses_interactions_api() -> None:
    """Translation should use one Interactions call and return normalized text."""

    client = (
        FakeClient(
            [
                (
                    "What does the Constitution "
                    "of Nepal guarantee regarding "
                    "the right to health?"
                )
            ]
        )
    )

    result = (
        translate_query(
            client,
            query=(
                "नेपालको संविधानले "
                "स्वास्थ्यसम्बन्धी अधिकारबारे "
                "के व्यवस्था गरेको छ?"
            ),
            model="test-model",
        )
    )

    assert result == (
        "What does the Constitution of Nepal "
        "guarantee regarding the right to health?"
    )

    assert (
        len(
            client.interactions.calls
        )
        == 1
    )

    assert (
        client.interactions.calls[
            0
        ][
            "model"
        ]
        == "test-model"
    )

    assert (
        "Do not answer the question."
        in client.interactions.calls[
            0
        ][
            "input"
        ]
    )


def test_translate_query_rejects_unusable_provider_output() -> None:
    """Provider output with Nepali script should fail explicitly."""

    client = (
        FakeClient(
            [
                "नेपालको right to health"
            ]
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "no usable English"
        ),
    ):
        translate_query(
            client,
            query="नेपाली प्रश्न",
            model="test-model",
        )


def test_load_existing_output_round_trip(
    tmp_path,
) -> None:
    """Completed translations should be reusable from disk."""

    record = (
        make_record()
    )

    row = (
        build_output_record(
            record,
            translated_query=(
                "What does the Constitution "
                "say about health rights?"
            ),
            model="test-model",
            provider="gemini",
        )
    )

    output_path = (
        tmp_path
        / "translations.jsonl"
    )

    output_path.write_text(
        __import__(
            "json"
        ).dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = (
        load_existing_output(
            output_path,
            [
                record,
            ],
            model="test-model",
            provider="gemini",
        )
    )

    assert (
        loaded[
            record.question_id
        ][
            "translated_query"
        ]
        == (
            "What does the Constitution "
            "say about health rights?"
        )
    )


def test_run_translation_reuses_existing_rows(
    tmp_path,
    monkeypatch,
) -> None:
    """Resuming should skip already persisted provider calls."""

    records = [
        make_record(
            question_id=(
                f"ne_en_00{index}"
            )
        )
        for index in range(
            1,
            7,
        )
    ]

    monkeypatch.setattr(
        (
            "src.evaluation."
            "run_ne_en_query_translation."
            "load_evaluation_records"
        ),
        lambda path: records,
    )

    output_path = (
        tmp_path
        / "translations.jsonl"
    )

    existing = (
        build_output_record(
            records[
                0
            ],
            translated_query=(
                "Existing English translation."
            ),
            model="test-model",
            provider="gemini",
        )
    )

    output_path.write_text(
        __import__(
            "json"
        ).dumps(
            existing,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    client = (
        FakeClient(
            [
                f"English translation {index}."
                for index in range(
                    2,
                    7,
                )
            ]
        )
    )

    rows = (
        run_translation(
            dataset_path="unused.jsonl",
            output_path=(
                output_path
            ),
            model="test-model",
            provider="gemini",
            client=client,
        )
    )

    assert (
        len(
            rows
        )
        == 6
    )

    assert (
        len(
            client.interactions.calls
        )
        == 5
    )

    assert (
        rows[
            0
        ][
            "translated_query"
        ]
        == (
            "Existing English translation."
        )
    )


def test_load_existing_output_rejects_duplicate_question_id(
    tmp_path,
) -> None:
    """Duplicate persisted translations must fail explicitly."""

    record = (
        make_record()
    )

    row = (
        build_output_record(
            record,
            translated_query=(
                "English translation."
            ),
            model="test-model",
            provider="gemini",
        )
    )

    serialized = (
        __import__(
            "json"
        ).dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    output_path = (
        tmp_path
        / "translations.jsonl"
    )

    output_path.write_text(
        serialized
        + "\n"
        + serialized
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate question_id"
        ),
    ):
        load_existing_output(
            output_path,
            [
                record,
            ],
            model="test-model",
            provider="gemini",
        )