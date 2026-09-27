"""Run the completeness prompt over the full production benchmark.

This experiment isolates generation from retrieval.

For all 30 completed production answer-quality rows:

- query text is unchanged,
- answer language is unchanged,
- selected evidence identity is unchanged,
- selected evidence order is unchanged,
- canonical passage text is unchanged,
- retrieval is not rerun,
- reranking is not rerun,
- context selection is not rerun.

Only the prompt policy changes from the current production GroundedPromptBuilder
to the evaluation-only CompletenessPromptBuilder.

The two completeness_v1 outputs already produced by the two-question pilot are
reused after validation so they do not require duplicate Gemini calls.

Production prompting remains unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from src.evaluation.answer_quality_review import (
    REVIEW_CONFIG_ID,
    load_review_rows,
)
from src.evaluation.run_completeness_prompt_experiment import (
    EXPERIMENT_CONFIG_ID as PILOT_CONFIG_ID,
    VARIANT_COMPLETENESS,
    CompletenessPromptBuilder,
    build_output_record,
    build_request_from_review_row,
    prompt_sha256,
)
from src.generation.base import (
    GenerationService,
)
from src.generation.gemini_service import (
    GeminiGenerationService,
    MODEL_NAME,
    PROVIDER_NAME,
)


FULL_SCHEMA_VERSION = 1

FULL_RUN_CONFIG_ID = (
    "production-rag-v2-completeness-prompt-full-v1"
)

DEFAULT_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_answer_quality_review_v1.jsonl"
)

DEFAULT_PILOT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_prompt_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_prompt_full_v1.jsonl"
)

EXPECTED_QUESTION_COUNT = 30


def build_full_record(
    source_row: Mapping[
        str,
        Any,
    ],
    *,
    generated_answer_text: str,
    provider: str | None,
    model: str | None,
) -> dict[
    str,
    Any,
]:
    """Build one full-benchmark completeness-prompt result."""

    request = (
        build_request_from_review_row(
            source_row
        )
    )

    builder = (
        CompletenessPromptBuilder()
    )

    prompt = (
        builder(
            request
        )
    )

    pilot_style = (
        build_output_record(
            source_row,
            variant=(
                VARIANT_COMPLETENESS
            ),
            request=request,
            prompt=prompt,
            generated_answer_text=(
                generated_answer_text
            ),
            provider=provider,
            model=model,
        )
    )

    return {
        **pilot_style,
        "schema_version": (
            FULL_SCHEMA_VERSION
        ),
        "run_config_id": (
            FULL_RUN_CONFIG_ID
        ),
        "source_pilot_config_id": (
            PILOT_CONFIG_ID
        ),
    }


def validate_full_record(
    persisted: Mapping[
        str,
        Any,
    ],
    source_row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Reject stale or incompatible full-benchmark rows."""

    question_id = (
        source_row[
            "question_id"
        ]
    )

    request = (
        build_request_from_review_row(
            source_row
        )
    )

    prompt = (
        CompletenessPromptBuilder()(
            request
        )
    )

    expected = (
        (
            "schema_version",
            FULL_SCHEMA_VERSION,
        ),
        (
            "run_config_id",
            FULL_RUN_CONFIG_ID,
        ),
        (
            "source_review_config_id",
            REVIEW_CONFIG_ID,
        ),
        (
            "source_pilot_config_id",
            PILOT_CONFIG_ID,
        ),
        (
            "question_id",
            question_id,
        ),
        (
            "query",
            request.query,
        ),
        (
            "query_language",
            source_row[
                "query_language"
            ],
        ),
        (
            "target_language",
            source_row[
                "target_language"
            ],
        ),
        (
            "answer_language",
            request.answer_language,
        ),
        (
            "variant",
            VARIANT_COMPLETENESS,
        ),
        (
            "prompt_sha256",
            prompt_sha256(
                prompt
            ),
        ),
        (
            "provider",
            PROVIDER_NAME,
        ),
        (
            "model",
            MODEL_NAME,
        ),
        (
            "selected_point_ids",
            [
                item.result.point_id
                for item in (
                    request.context
                )
            ],
        ),
        (
            "production_changed",
            False,
        ),
    )

    for (
        field_name,
        expected_value,
    ) in expected:
        if (
            persisted.get(
                field_name
            )
            != expected_value
        ):
            raise ValueError(
                f"{question_id}: persisted "
                f"field {field_name!r} does not "
                "match the current full benchmark."
            )

    for field_name in (
        "generated_answer_text",
        "answer_text",
    ):
        value = (
            persisted.get(
                field_name
            )
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"{question_id}: persisted "
                f"{field_name} is empty."
            )


def load_existing_full_output(
    path: str | Path,
    source_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load reusable completed full-benchmark rows."""

    output_path = Path(
        path
    )

    if not output_path.exists():
        return {}

    source_by_id = {
        row[
            "question_id"
        ]: row
        for row in (
            source_rows
        )
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
                persisted = (
                    json.loads(
                        line
                    )
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid full completeness "
                    f"benchmark JSON on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                persisted,
                dict,
            ):
                raise ValueError(
                    "Full completeness benchmark "
                    f"row {line_number} must be "
                    "an object."
                )

            question_id = (
                persisted.get(
                    "question_id"
                )
            )

            if (
                not isinstance(
                    question_id,
                    str,
                )
                or question_id
                not in source_by_id
            ):
                raise ValueError(
                    "Full completeness benchmark "
                    f"contains unexpected question_id "
                    f"{question_id!r}."
                )

            if (
                question_id
                in completed
            ):
                raise ValueError(
                    "Full completeness benchmark "
                    "contains duplicate question_id "
                    f"{question_id!r}."
                )

            validate_full_record(
                persisted,
                source_by_id[
                    question_id
                ],
            )

            completed[
                question_id
            ] = (
                persisted
            )

    return completed


def load_pilot_interventions(
    path: str | Path,
    source_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load reusable completeness_v1 rows from the two-question pilot."""

    pilot_path = Path(
        path
    )

    if not pilot_path.exists():
        return {}

    source_by_id = {
        row[
            "question_id"
        ]: row
        for row in (
            source_rows
        )
    }

    reusable: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    with pilot_path.open(
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
                    "Invalid pilot JSON on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    f"Pilot row {line_number} "
                    "must be an object."
                )

            if (
                row.get(
                    "run_config_id"
                )
                != PILOT_CONFIG_ID
            ):
                raise ValueError(
                    "Unexpected pilot run_config_id."
                )

            if (
                row.get(
                    "variant"
                )
                != VARIANT_COMPLETENESS
            ):
                continue

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
                or question_id
                not in source_by_id
            ):
                raise ValueError(
                    "Pilot contains unexpected "
                    f"question_id {question_id!r}."
                )

            source_row = (
                source_by_id[
                    question_id
                ]
            )

            request = (
                build_request_from_review_row(
                    source_row
                )
            )

            expected_prompt = (
                CompletenessPromptBuilder()(
                    request
                )
            )

            if (
                row.get(
                    "prompt_sha256"
                )
                != prompt_sha256(
                    expected_prompt
                )
            ):
                raise ValueError(
                    f"{question_id}: pilot prompt "
                    "hash is stale."
                )

            if (
                row.get(
                    "selected_point_ids"
                )
                != [
                    item.result.point_id
                    for item in (
                        request.context
                    )
                ]
            ):
                raise ValueError(
                    f"{question_id}: pilot selected "
                    "context does not match the "
                    "current review artifact."
                )

            if (
                row.get(
                    "provider"
                )
                != PROVIDER_NAME
                or row.get(
                    "model"
                )
                != MODEL_NAME
            ):
                raise ValueError(
                    f"{question_id}: pilot provider "
                    "or model does not match."
                )

            reusable[
                question_id
            ] = (
                build_full_record(
                    source_row,
                    generated_answer_text=(
                        row[
                            "generated_answer_text"
                        ]
                    ),
                    provider=(
                        row[
                            "provider"
                        ]
                    ),
                    model=(
                        row[
                            "model"
                        ]
                    ),
                )
            )

    return reusable


def append_output_record(
    path: str | Path,
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Append one completed generation immediately."""

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


def print_summary(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Print structural completion and citation safety."""

    accepted_count = sum(
        bool(
            row[
                "accepted"
            ]
        )
        for row in rows
    )

    invalid_count = sum(
        bool(
            row[
                "invalid_evidence_ids"
            ]
        )
        for row in rows
    )

    withheld_count = sum(
        bool(
            row[
                "withheld"
            ]
        )
        for row in rows
    )

    print()
    print(
        "=" * 92
    )

    print(
        "Full completeness-prompt generation benchmark"
    )

    print(
        "=" * 92
    )

    print(
        f"Questions:             {len(rows)}"
    )

    print(
        f"Accepted:              {accepted_count}"
    )

    print(
        f"Withheld:              {withheld_count}"
    )

    print(
        f"Invalid-citation rows: {invalid_count}"
    )

    print()
    print(
        f"{'Question':<14}"
        f"{'Pair':<10}"
        f"{'Accepted':>10}"
        f"{'Citations':>12}"
        f"{'Invalid':>10}"
    )

    print(
        "-" * 60
    )

    for row in rows:
        pair = (
            f"{row['query_language']}"
            f"->{row['target_language']}"
        )

        print(
            f"{row['question_id']:<14}"
            f"{pair:<10}"
            f"{str(row['accepted']):>10}"
            f"{len(row['cited_evidence_ids']):>12}"
            f"{len(row['invalid_evidence_ids']):>10}"
        )


def run_full_benchmark(
    *,
    review_path: str | Path = (
        DEFAULT_REVIEW_PATH
    ),
    pilot_path: str | Path = (
        DEFAULT_PILOT_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    generation_service: GenerationService | None = None,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Generate completeness-prompt answers for the full 30-question set."""

    source_rows = (
        load_review_rows(
            review_path
        )
    )

    if (
        len(
            source_rows
        )
        != EXPECTED_QUESTION_COUNT
    ):
        raise ValueError(
            "Full completeness benchmark requires "
            f"{EXPECTED_QUESTION_COUNT} source rows."
        )

    if any(
        row.get(
            "review_status"
        )
        != "completed"
        for row in (
            source_rows
        )
    ):
        raise ValueError(
            "Answer-quality review must be "
            "fully completed."
        )

    completed = (
        load_existing_full_output(
            output_path,
            source_rows,
        )
    )

    pilot_rows = (
        load_pilot_interventions(
            pilot_path,
            source_rows,
        )
    )

    for source_row in (
        source_rows
    ):
        question_id = (
            source_row[
                "question_id"
            ]
        )

        if (
            question_id
            in completed
        ):
            continue

        reusable = (
            pilot_rows.get(
                question_id
            )
        )

        if reusable is None:
            continue

        append_output_record(
            output_path,
            reusable,
        )

        completed[
            question_id
        ] = (
            reusable
        )

        print(
            f"{question_id}: reused pilot "
            "completeness_v1 output"
        )

    owns_service = (
        generation_service is None
    )

    service = (
        generation_service
    )

    try:
        missing = [
            row
            for row in source_rows
            if (
                row[
                    "question_id"
                ]
                not in completed
            )
        ]

        if (
            missing
            and service is None
        ):
            service = (
                GeminiGenerationService(
                    prompt_builder=(
                        CompletenessPromptBuilder()
                    ),
                )
            )

        for (
            index,
            source_row,
        ) in enumerate(
            missing,
            start=1,
        ):
            question_id = (
                source_row[
                    "question_id"
                ]
            )

            print(
                f"[{index}/{len(missing)}] "
                f"{question_id}: generate"
            )

            request = (
                build_request_from_review_row(
                    source_row
                )
            )

            if service is None:
                raise RuntimeError(
                    "Generation service was not initialized."
                )

            result = (
                service.generate(
                    request
                )
            )

            record = (
                build_full_record(
                    source_row,
                    generated_answer_text=(
                        result.answer_text
                    ),
                    provider=(
                        result.provider
                    ),
                    model=(
                        result.model
                    ),
                )
            )

            append_output_record(
                output_path,
                record,
            )

            completed[
                question_id
            ] = (
                record
            )

    finally:
        if (
            owns_service
            and service is not None
        ):
            close_method = getattr(
                service,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                close_method()

    ordered = [
        completed[
            row[
                "question_id"
            ]
        ]
        for row in (
            source_rows
        )
    ]

    if (
        len(
            ordered
        )
        != EXPECTED_QUESTION_COUNT
    ):
        raise RuntimeError(
            "Full completeness benchmark is incomplete."
        )

    print_summary(
        ordered
    )

    return ordered


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the full-benchmark CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run the completeness prompt against "
                "all 30 fixed production contexts."
            )
        )
    )

    parser.add_argument(
        "--review",
        type=Path,
        default=(
            DEFAULT_REVIEW_PATH
        ),
    )

    parser.add_argument(
        "--pilot",
        type=Path,
        default=(
            DEFAULT_PILOT_PATH
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
    """Run the full fixed-context completeness-prompt benchmark."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    run_full_benchmark(
        review_path=(
            args.review
        ),
        pilot_path=(
            args.pilot
        ),
        output_path=(
            args.output
        ),
    )

    print()
    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()