"""Generate automatic English translations for the NE->EN retrieval benchmark.

This evaluation-only runner translates the six Nepali queries in the existing
NE->EN retrieval slice into English using Gemini.

The generated English queries are persisted so hosted translation calls do not
need to be repeated during later retrieval experiments.

Important boundaries:

- this module does not modify production retrieval,
- translations are evaluation artifacts only,
- it does not retrieve documents,
- it does not rerank candidates,
- it does not generate answers,
- it does not use gold evidence when constructing the translation prompt.

The manually controlled English queries from the previous experiment remain an
evaluation upper-bound/control and are never included in the translation prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from src.evaluation.compare_ne_en_retrieval_variants import (
    CONTROLLED_ENGLISH_QUERIES,
    select_ne_en_records,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.generation.gemini_service import (
    DEFAULT_TIMEOUT_SECONDS,
    GEMINI_API_KEY_ENV,
    MODEL_NAME,
    PROVIDER_NAME,
)


TRANSLATION_SCHEMA_VERSION = 1

TRANSLATION_CONFIG_ID = (
    "ne-en-query-translation-v1"
)

DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_questions.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/retrieval_runs/"
    "ne_en_query_translation_v1.jsonl"
)

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F


def build_translation_prompt(
    query: str,
) -> str:
    """Build a strict retrieval-oriented Nepali-to-English translation prompt."""

    clean_query = (
        query.strip()
    )

    if not clean_query:
        raise ValueError(
            "query must contain non-whitespace text."
        )

    return (
        "Translate the following Nepali government-information "
        "search query into natural English for document retrieval.\n\n"
        "Requirements:\n"
        "- Preserve the exact meaning of the query.\n"
        "- Preserve document names, legal concepts, dates, years, "
        "numbers, institutions, and named entities.\n"
        "- Do not answer the question.\n"
        "- Do not add facts, synonyms, keywords, explanations, "
        "or information that is not present in the Nepali query.\n"
        "- Do not make the query more specific than the original.\n"
        "- Return only the English translation as one line.\n\n"
        "Nepali query:\n"
        f"{clean_query}"
    )


def prompt_sha256(
    prompt: str,
) -> str:
    """Return the stable SHA-256 digest of one translation prompt."""

    return (
        hashlib.sha256(
            prompt.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def contains_devanagari(
    text: str,
) -> bool:
    """Return whether text contains any Devanagari code point."""

    return any(
        DEVANAGARI_START
        <= ord(
            character
        )
        <= DEVANAGARI_END
        for character in text
    )


def normalize_translation(
    value: Any,
) -> str:
    """Validate and normalize one English translation."""

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "Translation must be a string."
        )

    clean = (
        " ".join(
            value.strip().split()
        )
    )

    if not clean:
        raise ValueError(
            "Translation must contain "
            "non-whitespace text."
        )

    if contains_devanagari(
        clean
    ):
        raise ValueError(
            "Translation still contains "
            "Devanagari text."
        )

    return clean


def build_output_record(
    record: EvaluationRecord,
    *,
    translated_query: str,
    model: str,
    provider: str,
) -> dict[
    str,
    Any,
]:
    """Build one persisted automatic-translation record."""

    translation = (
        normalize_translation(
            translated_query
        )
    )

    prompt = (
        build_translation_prompt(
            record.query
        )
    )

    return {
        "schema_version": (
            TRANSLATION_SCHEMA_VERSION
        ),
        "translation_config_id": (
            TRANSLATION_CONFIG_ID
        ),
        "question_id": (
            record.question_id
        ),
        "query_language": (
            record.query_language
        ),
        "target_language": (
            record.target_language
        ),
        "original_query": (
            record.query
        ),
        "translated_query": (
            translation
        ),
        "provider": (
            provider
        ),
        "model": (
            model
        ),
        "prompt_sha256": (
            prompt_sha256(
                prompt
            )
        ),
    }


def validate_persisted_row(
    row: Mapping[
        str,
        Any,
    ],
    record: EvaluationRecord,
    *,
    model: str,
    provider: str,
) -> None:
    """Reject stale or incompatible automatic translations."""

    question_id = (
        record.question_id
    )

    expected_prompt = (
        build_translation_prompt(
            record.query
        )
    )

    expected_hash = (
        prompt_sha256(
            expected_prompt
        )
    )

    comparisons = (
        (
            "schema_version",
            row.get(
                "schema_version"
            ),
            TRANSLATION_SCHEMA_VERSION,
        ),
        (
            "translation_config_id",
            row.get(
                "translation_config_id"
            ),
            TRANSLATION_CONFIG_ID,
        ),
        (
            "question_id",
            row.get(
                "question_id"
            ),
            question_id,
        ),
        (
            "query_language",
            row.get(
                "query_language"
            ),
            record.query_language,
        ),
        (
            "target_language",
            row.get(
                "target_language"
            ),
            record.target_language,
        ),
        (
            "original_query",
            row.get(
                "original_query"
            ),
            record.query,
        ),
        (
            "provider",
            row.get(
                "provider"
            ),
            provider,
        ),
        (
            "model",
            row.get(
                "model"
            ),
            model,
        ),
        (
            "prompt_sha256",
            row.get(
                "prompt_sha256"
            ),
            expected_hash,
        ),
    )

    for (
        field_name,
        actual,
        expected,
    ) in comparisons:
        if (
            actual
            != expected
        ):
            raise ValueError(
                f"{question_id}: persisted "
                f"translation field "
                f"{field_name!r} does not "
                "match the current run."
            )

    normalize_translation(
        row.get(
            "translated_query"
        )
    )


def load_existing_output(
    path: str | Path,
    records: Sequence[
        EvaluationRecord
    ],
    *,
    model: str,
    provider: str,
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load reusable completed translations."""

    output_path = Path(
        path
    )

    if not output_path.exists():
        return {}

    records_by_id = {
        record.question_id: record
        for record in records
    }

    completed: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for (
            line_number,
            raw_line,
        ) in enumerate(
            handle,
            start=1,
        ):
            line = (
                raw_line.strip()
            )

            if not line:
                continue

            try:
                row = (
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid translation JSON "
                    f"on line {line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Translation row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            question_id = (
                row.get(
                    "question_id"
                )
            )

            if (
                not isinstance(
                    question_id,
                    str,
                )
                or not question_id
            ):
                raise ValueError(
                    "Translation row "
                    f"{line_number} has no "
                    "valid question_id."
                )

            record = (
                records_by_id.get(
                    question_id
                )
            )

            if record is None:
                raise ValueError(
                    "Translation output "
                    "contains unexpected "
                    f"question_id "
                    f"{question_id!r}."
                )

            if (
                question_id
                in completed
            ):
                raise ValueError(
                    "Translation output "
                    "contains duplicate "
                    f"question_id "
                    f"{question_id!r}."
                )

            validate_persisted_row(
                row,
                record,
                model=model,
                provider=provider,
            )

            completed[
                question_id
            ] = row

    return completed


def append_output_record(
    output_path: str | Path,
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Persist one translation immediately."""

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                dict(
                    row
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        handle.flush()


def build_gemini_client(
    *,
    api_key: str | None = None,
    timeout_seconds: float = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> Any:
    """Build the Gemini client without making a request."""

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater "
            "than zero."
        )

    resolved_api_key = (
        api_key
        if api_key is not None
        else os.getenv(
            GEMINI_API_KEY_ENV
        )
    )

    if (
        not isinstance(
            resolved_api_key,
            str,
        )
        or not resolved_api_key.strip()
    ):
        raise ValueError(
            f"{GEMINI_API_KEY_ENV} must be set "
            "and contain non-whitespace text."
        )

    try:
        from google import genai
        from google.genai import types

    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required "
            "for automatic query translation."
        ) from exc

    http_options = (
        types.HttpOptions(
            api_version="v1",
            client_args={
                "timeout": (
                    timeout_seconds
                ),
            },
        )
    )

    return (
        genai.Client(
            api_key=(
                resolved_api_key.strip()
            ),
            http_options=(
                http_options
            ),
        )
    )


def translate_query(
    client: Any,
    *,
    query: str,
    model: str,
) -> str:
    """Translate one query through the Gemini Interactions API."""

    prompt = (
        build_translation_prompt(
            query
        )
    )

    try:
        interaction = (
            client.interactions.create(
                model=model,
                input=prompt,
            )
        )

    except Exception as exc:
        raise RuntimeError(
            "Gemini query translation "
            "request failed."
        ) from exc

    output_text = getattr(
        interaction,
        "output_text",
        None,
    )

    try:
        return (
            normalize_translation(
                output_text
            )
        )

    except ValueError as exc:
        raise RuntimeError(
            "Gemini returned no usable "
            "English query translation."
        ) from exc


def run_translation(
    *,
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    model: str = MODEL_NAME,
    provider: str = PROVIDER_NAME,
    client: Any | None = None,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Generate missing NE->EN translations and reuse completed ones."""

    records = (
        select_ne_en_records(
            load_evaluation_records(
                dataset_path
            )
        )
    )

    if not records:
        raise ValueError(
            "No NE->EN benchmark "
            "records were found."
        )

    completed = (
        load_existing_output(
            output_path,
            records,
            model=model,
            provider=provider,
        )
    )

    owns_client = (
        client is None
    )

    active_client = (
        client
    )

    try:
        for (
            index,
            record,
        ) in enumerate(
            records,
            start=1,
        ):
            existing = (
                completed.get(
                    record.question_id
                )
            )

            if existing is not None:
                print(
                    f"[{index}/{len(records)}] "
                    f"{record.question_id}: "
                    "reuse"
                )

                continue

            if active_client is None:
                active_client = (
                    build_gemini_client()
                )

            print(
                f"[{index}/{len(records)}] "
                f"{record.question_id}: "
                "translate"
            )

            translated_query = (
                translate_query(
                    active_client,
                    query=(
                        record.query
                    ),
                    model=model,
                )
            )

            row = (
                build_output_record(
                    record,
                    translated_query=(
                        translated_query
                    ),
                    model=model,
                    provider=provider,
                )
            )

            append_output_record(
                output_path,
                row,
            )

            completed[
                record.question_id
            ] = row

    finally:
        if (
            owns_client
            and active_client
            is not None
        ):
            close_method = getattr(
                active_client,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                close_method()

    ordered = [
        completed[
            record.question_id
        ]
        for record in records
    ]

    return ordered


def print_translation_summary(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Print automatic translations beside the previous manual controls."""

    print()
    print(
        "=" * 100
    )

    print(
        "Automatic NE->EN retrieval-query translations"
    )

    print(
        "=" * 100
    )

    for (
        index,
        row,
    ) in enumerate(
        rows,
        start=1,
    ):
        question_id = str(
            row[
                "question_id"
            ]
        )

        print()
        print(
            f"{index}. {question_id}"
        )

        print(
            "Nepali:"
        )

        print(
            row[
                "original_query"
            ]
        )

        print(
            "Automatic English:"
        )

        print(
            row[
                "translated_query"
            ]
        )

        print(
            "Controlled English:"
        )

        print(
            CONTROLLED_ENGLISH_QUERIES[
                question_id
            ]
        )

    print()
    print(
        f"Completed translations: "
        f"{len(rows)}/6"
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the evaluation CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Generate persistent automatic "
                "English translations for the "
                "six-query NE->EN retrieval slice."
            )
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            DEFAULT_DATASET_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    return parser


def main(
) -> None:
    """Generate translations and print the inspection table."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    rows = (
        run_translation(
            dataset_path=(
                args.dataset
            ),
            output_path=(
                args.output
            ),
        )
    )

    print_translation_summary(
        rows
    )


if __name__ == "__main__":
    main()