"""Resumable automated semantic-judge runner for NepalGov AI.

This module runs an evaluation-only Gemini semantic judge over the manually
reviewed 48-claim reference subset.

Important:

- human labels remain the reference standard,
- human labels are never included in judge prompts,
- predictions are persisted incrementally,
- completed predictions are reused after validation,
- prompt hashes prevent silently reusing predictions after prompt changes,
- the automated judge is not part of the production RAG evidence guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import (
    Any,
    Protocol,
)

from src.evaluation.semantic_judge import (
    JUDGE_SCHEMA_VERSION,
    SemanticJudgeAgreement,
    SemanticJudgePrediction,
    aggregate_judge_agreement,
    aggregate_judge_agreement_by_language_pair,
    build_semantic_judge_prompt,
    compare_prediction_to_human,
    parse_semantic_judge_response,
    prediction_to_dict,
)
from src.evaluation.semantic_review import (
    REVIEW_STATUS_COMPLETED,
    load_review_rows,
    validate_review_row,
)


DEFAULT_INPUT_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1_labeled.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1_judge.jsonl"
)

RESULT_SCHEMA_VERSION = 1

JUDGE_RUN_CONFIG_ID = (
    "production-rag-v2-human-review-v1-"
    "semantic-judge-v1"
)

DEFAULT_JUDGE_PROVIDER = "gemini"

DEFAULT_JUDGE_MODEL = "gemini-3.8-flash"

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

DEFAULT_TIMEOUT_SECONDS = 60.0

LANGUAGE_PAIR_ORDER = (
    "en->en",
    "ne->ne",
    "en->ne",
    "ne->en",
)


class SemanticJudgeService(
    Protocol
):
    """Minimal hosted service contract required by the judge runner."""

    provider: str
    model_name: str

    def judge(
        self,
        prompt: str,
    ) -> str:
        """Return raw judge response text."""

    def close(
        self,
    ) -> None:
        """Release provider resources."""


JudgeServiceFactory = Callable[
    [
        str,
    ],
    SemanticJudgeService,
]


class GeminiSemanticJudgeService:
    """Evaluation-only Gemini semantic judge transport."""

    provider = (
        DEFAULT_JUDGE_PROVIDER
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = (
            DEFAULT_JUDGE_MODEL
        ),
        timeout_seconds: float = (
            DEFAULT_TIMEOUT_SECONDS
        ),
        client: Any | None = None,
    ) -> None:
        """Configure Gemini semantic judging without making a request."""

        clean_model_name = (
            model_name.strip()
        )

        if not clean_model_name:
            raise ValueError(
                "model_name must contain "
                "non-whitespace text."
            )

        if (
            timeout_seconds
            <= 0
        ):
            raise ValueError(
                "timeout_seconds must be "
                "greater than zero."
            )

        resolved_api_key = (
            api_key
            if api_key is not None
            else os.getenv(
                GEMINI_API_KEY_ENV
            )
        )

        if (
            client is None
            and not resolved_api_key
        ):
            raise ValueError(
                f"{GEMINI_API_KEY_ENV} "
                "must be set for Gemini "
                "semantic judging."
            )

        if (
            client is None
            and isinstance(
                resolved_api_key,
                str,
            )
            and not resolved_api_key.strip()
        ):
            raise ValueError(
                f"{GEMINI_API_KEY_ENV} "
                "must contain "
                "non-whitespace text."
            )

        self.model_name = (
            clean_model_name
        )

        self.api_key = (
            resolved_api_key.strip()
            if isinstance(
                resolved_api_key,
                str,
            )
            and resolved_api_key.strip()
            else None
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self._client = (
            client
        )

    def _get_client(
        self,
    ) -> Any:
        """Create the official Gemini SDK client lazily."""

        if (
            self._client
            is not None
        ):
            return self._client

        try:
            from google import genai
            from google.genai import types

        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required "
                "for Gemini semantic judging."
            ) from exc

        http_options = (
            types.HttpOptions(
                api_version="v1",
                client_args={
                    "timeout": (
                        self.timeout_seconds
                    ),
                },
            )
        )

        self._client = (
            genai.Client(
                api_key=self.api_key,
                http_options=(
                    http_options
                ),
            )
        )

        return self._client

    def judge(
        self,
        prompt: str,
    ) -> str:
        """Run one semantic judgment through Gemini Interactions."""

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string."
            )

        clean_prompt = (
            prompt.strip()
        )

        if not clean_prompt:
            raise ValueError(
                "prompt must contain "
                "non-whitespace text."
            )

        client = (
            self._get_client()
        )

        try:
            interaction = (
                client.interactions.create(
                    model=(
                        self.model_name
                    ),
                    input=(
                        clean_prompt
                    ),
                )
            )

        except Exception as exc:
            raise RuntimeError(
                "Gemini semantic judge "
                "request failed."
            ) from exc

        response_text = (
            getattr(
                interaction,
                "output_text",
                None,
            )
        )

        if (
            not isinstance(
                response_text,
                str,
            )
            or not response_text.strip()
        ):
            raise RuntimeError(
                "Gemini semantic judge "
                "returned no usable text."
            )

        return (
            response_text.strip()
        )

    def close(
        self,
    ) -> None:
        """Release Gemini SDK resources when available."""

        if (
            self._client
            is None
        ):
            return

        close_method = (
            getattr(
                self._client,
                "close",
                None,
            )
        )

        if callable(
            close_method
        ):
            close_method()


def build_gemini_judge_service(
    model_name: str,
) -> GeminiSemanticJudgeService:
    """Build the default hosted semantic judge."""

    return (
        GeminiSemanticJudgeService(
            model_name=(
                model_name
            )
        )
    )


def validate_limit(
    limit: int | None,
) -> int | None:
    """Validate an optional maximum number of new hosted calls."""

    if (
        limit is not None
        and limit <= 0
    ):
        raise ValueError(
            "limit must be greater "
            "than zero when provided."
        )

    return limit


def prompt_sha256(
    prompt: str,
) -> str:
    """Return deterministic SHA-256 for one judge prompt."""

    return (
        hashlib.sha256(
            prompt.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Return one validated non-empty string."""

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must contain "
            "non-whitespace text."
        )

    return (
        value.strip()
    )


def build_output_record(
    human_row: Mapping[
        str,
        Any,
    ],
    prediction: SemanticJudgePrediction,
    *,
    response_text: str,
    prompt: str,
    provider: str,
    model: str,
) -> dict[
    str,
    Any,
]:
    """Build one persisted automated-judge prediction record."""

    validate_review_row(
        human_row
    )

    claim_id = (
        _require_nonempty_string(
            human_row.get(
                "claim_id"
            ),
            field_name="claim_id",
        )
    )

    if (
        human_row.get(
            "review_status"
        )
        != REVIEW_STATUS_COMPLETED
    ):
        raise ValueError(
            f"Human claim {claim_id!r} "
            "has not been completed."
        )

    if (
        prediction.claim_id
        != claim_id
    ):
        raise ValueError(
            "Prediction claim_id does "
            "not match human row."
        )

    # Deliberately do not persist human semantic labels in this file.
    # Predictions remain separate from the reference truth.
    return {
        "schema_version": (
            RESULT_SCHEMA_VERSION
        ),
        "judge_run_config_id": (
            JUDGE_RUN_CONFIG_ID
        ),
        "judge_schema_version": (
            JUDGE_SCHEMA_VERSION
        ),
        "claim_id": (
            claim_id
        ),
        "question_id": (
            human_row.get(
                "question_id"
            )
        ),
        "query_language": (
            human_row.get(
                "query_language"
            )
        ),
        "target_language": (
            human_row.get(
                "target_language"
            )
        ),
        "review_sample_config_id": (
            human_row.get(
                "review_sample_config_id"
            )
        ),
        "judge_provider": (
            provider
        ),
        "judge_model": (
            model
        ),
        "prompt_sha256": (
            prompt_sha256(
                prompt
            )
        ),
        "raw_response": (
            response_text
        ),
        "prediction": (
            prediction_to_dict(
                prediction
            )
        ),
    }


def append_output_record(
    output_path: str | Path,
    row: Mapping[
        str,
        Any,
    ],
) -> None:
    """Append one successful hosted judgment immediately."""

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
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        handle.flush()


def prediction_from_output_record(
    row: Mapping[
        str,
        Any,
    ],
    human_row: Mapping[
        str,
        Any,
    ],
) -> SemanticJudgePrediction:
    """Reconstruct and validate a persisted automated prediction."""

    response_text = (
        _require_nonempty_string(
            row.get(
                "raw_response"
            ),
            field_name=(
                "raw_response"
            ),
        )
    )

    prediction = (
        parse_semantic_judge_response(
            response_text,
            row=human_row,
        )
    )

    persisted_prediction = (
        row.get(
            "prediction"
        )
    )

    expected_prediction = (
        prediction_to_dict(
            prediction
        )
    )

    if (
        persisted_prediction
        != expected_prediction
    ):
        raise ValueError(
            "Persisted parsed prediction "
            "does not match its raw "
            "judge response."
        )

    return prediction


def validate_persisted_row(
    row: Mapping[
        str,
        Any,
    ],
    human_row: Mapping[
        str,
        Any,
    ],
    *,
    line_number: int,
    provider: str,
    model: str,
) -> None:
    """Reject stale or incompatible persisted judge predictions."""

    claim_id = (
        _require_nonempty_string(
            human_row.get(
                "claim_id"
            ),
            field_name="claim_id",
        )
    )

    if (
        row.get(
            "schema_version"
        )
        != RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Persisted semantic judge row "
            f"on line {line_number} uses an "
            "incompatible schema_version."
        )

    if (
        row.get(
            "judge_run_config_id"
        )
        != JUDGE_RUN_CONFIG_ID
    ):
        raise ValueError(
            "Persisted semantic judge row "
            f"on line {line_number} uses a "
            "different judge_run_config_id. "
            "Use --reset for a fresh run."
        )

    if (
        row.get(
            "judge_schema_version"
        )
        != JUDGE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Persisted semantic judge row "
            f"on line {line_number} uses an "
            "incompatible judge schema."
        )

    expected_fields = {
        "claim_id": (
            claim_id
        ),
        "question_id": (
            human_row.get(
                "question_id"
            )
        ),
        "query_language": (
            human_row.get(
                "query_language"
            )
        ),
        "target_language": (
            human_row.get(
                "target_language"
            )
        ),
        "review_sample_config_id": (
            human_row.get(
                "review_sample_config_id"
            )
        ),
        "judge_provider": (
            provider
        ),
        "judge_model": (
            model
        ),
    }

    for (
        field,
        expected_value,
    ) in (
        expected_fields.items()
    ):
        if (
            row.get(
                field
            )
            != expected_value
        ):
            raise ValueError(
                "Persisted semantic judge "
                f"row for {claim_id!r} "
                f"does not match current "
                f"{field!r}. Use --reset "
                "for a fresh run."
            )

    current_prompt = (
        build_semantic_judge_prompt(
            human_row
        )
    )

    if (
        row.get(
            "prompt_sha256"
        )
        != prompt_sha256(
            current_prompt
        )
    ):
        raise ValueError(
            "Persisted semantic judge "
            f"row for {claim_id!r} was "
            "generated with a different "
            "prompt. Use --reset for a "
            "fresh run."
        )

    prediction_from_output_record(
        row,
        human_row,
    )


def load_existing_output(
    output_path: str | Path,
    human_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    provider: str,
    model: str,
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load reusable completed automated-judge predictions."""

    path = Path(
        output_path
    )

    if not path.exists():
        return {}

    human_by_id = {
        str(
            row[
                "claim_id"
            ]
        ): row
        for row in (
            human_rows
        )
    }

    completed: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    with path.open(
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
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in semantic "
                    "judge output on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Semantic judge output "
                    f"row {line_number} must "
                    "be a JSON object."
                )

            claim_id = (
                row.get(
                    "claim_id"
                )
            )

            if (
                not isinstance(
                    claim_id,
                    str,
                )
                or not claim_id
            ):
                raise ValueError(
                    "Persisted semantic "
                    "judge row on line "
                    f"{line_number} has no "
                    "valid claim_id."
                )

            if (
                claim_id
                not in human_by_id
            ):
                raise ValueError(
                    "Persisted semantic "
                    "judge output contains "
                    f"unknown claim_id "
                    f"{claim_id!r}."
                )

            if (
                claim_id
                in completed
            ):
                raise ValueError(
                    "Persisted semantic "
                    "judge output contains "
                    "duplicate claim_id "
                    f"{claim_id!r}."
                )

            validate_persisted_row(
                row,
                human_by_id[
                    claim_id
                ],
                line_number=(
                    line_number
                ),
                provider=provider,
                model=model,
            )

            completed[
                claim_id
            ] = row

    return completed


def build_agreements(
    human_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    completed: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ],
) -> list[
    SemanticJudgeAgreement
]:
    """Compare all persisted predictions against human reference labels."""

    agreements: list[
        SemanticJudgeAgreement
    ] = []

    for human_row in (
        human_rows
    ):
        claim_id = str(
            human_row[
                "claim_id"
            ]
        )

        output_row = (
            completed.get(
                claim_id
            )
        )

        if (
            output_row
            is None
        ):
            continue

        prediction = (
            prediction_from_output_record(
                output_row,
                human_row,
            )
        )

        agreements.append(
            compare_prediction_to_human(
                human_row,
                prediction,
            )
        )

    return agreements


def print_judge_summary(
    agreements: Sequence[
        SemanticJudgeAgreement
    ],
    *,
    total_claim_count: int,
) -> None:
    """Print overall and language-pair automated-vs-human agreement."""

    if not agreements:
        print(
            "No completed automated "
            "judge predictions."
        )
        return

    pair_summaries = (
        aggregate_judge_agreement_by_language_pair(
            agreements
        )
    )

    overall = (
        aggregate_judge_agreement(
            agreements
        )
    )

    print()
    print(
        "=" * 75
    )

    print(
        "Automated semantic judge vs human reference"
    )

    print(
        "=" * 75
    )

    print(
        f"{'Slice':<10}"
        f"{'N':>5}"
        f"{'Semantic':>12}"
        f"{'Req':>10}"
        f"{'Individual':>13}"
    )

    print(
        "-" * 75
    )

    for pair in (
        LANGUAGE_PAIR_ORDER
    ):
        metrics = (
            pair_summaries.get(
                pair
            )
        )

        if (
            metrics
            is None
        ):
            continue

        print(
            f"{pair:<10}"
            f"{metrics['claim_count']:>5}"
            f"{metrics['semantic_exact_accuracy']:>12.3f}"
            f"{metrics['citation_requirement_accuracy']:>10.3f}"
            f"{metrics['individual_exact_accuracy']:>13.3f}"
        )

    print(
        "-" * 75
    )

    print(
        f"{'overall':<10}"
        f"{overall['claim_count']:>5}"
        f"{overall['semantic_exact_accuracy']:>12.3f}"
        f"{overall['citation_requirement_accuracy']:>10.3f}"
        f"{overall['individual_exact_accuracy']:>13.3f}"
    )

    print()

    print(
        "Completed predictions: "
        f"{len(agreements)}/"
        f"{total_claim_count}"
    )

    print()

    print(
        "Semantic confusion "
        "(human -> judge):"
    )

    print(
        json.dumps(
            overall[
                "semantic_confusion"
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Citation requirement confusion "
        "(human -> judge):"
    )

    print(
        json.dumps(
            overall[
                "citation_requirement_confusion"
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def run_semantic_judge(
    input_path: str | Path = (
        DEFAULT_INPUT_PATH
    ),
    *,
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    limit: int | None = None,
    reset: bool = False,
    provider: str = (
        DEFAULT_JUDGE_PROVIDER
    ),
    model: str = (
        DEFAULT_JUDGE_MODEL
    ),
    service_factory: JudgeServiceFactory = (
        build_gemini_judge_service
    ),
) -> tuple[
    list[
        SemanticJudgeAgreement
    ],
    dict[
        str,
        Any,
    ],
]:
    """Run or resume automated semantic judging."""

    normalized_limit = (
        validate_limit(
            limit
        )
    )

    human_rows = (
        load_review_rows(
            input_path
        )
    )

    if not human_rows:
        raise ValueError(
            "Human reference set "
            "contains no rows."
        )

    for human_row in (
        human_rows
    ):
        if (
            human_row.get(
                "review_status"
            )
            != REVIEW_STATUS_COMPLETED
        ):
            raise ValueError(
                "Automated judge input "
                "must contain only completed "
                "human review rows."
            )

    clean_provider = (
        _require_nonempty_string(
            provider,
            field_name="provider",
        )
    )

    clean_model = (
        _require_nonempty_string(
            model,
            field_name="model",
        )
    )

    path = Path(
        output_path
    )

    if (
        reset
        and path.exists()
    ):
        path.unlink()

    completed = (
        load_existing_output(
            path,
            human_rows,
            provider=(
                clean_provider
            ),
            model=(
                clean_model
            ),
        )
    )

    missing_rows = [
        row
        for row in (
            human_rows
        )
        if str(
            row[
                "claim_id"
            ]
        )
        not in completed
    ]

    if (
        normalized_limit
        is not None
    ):
        rows_to_run = (
            missing_rows[
                :normalized_limit
            ]
        )

    else:
        rows_to_run = (
            missing_rows
        )

    if rows_to_run:
        service = (
            service_factory(
                clean_model
            )
        )

        if (
            service.provider
            != clean_provider
        ):
            raise ValueError(
                "Judge service provider "
                "does not match configured "
                "provider."
            )

        if (
            service.model_name
            != clean_model
        ):
            raise ValueError(
                "Judge service model "
                "does not match configured "
                "model."
            )

        try:
            for (
                index,
                human_row,
            ) in enumerate(
                rows_to_run,
                start=1,
            ):
                claim_id = str(
                    human_row[
                        "claim_id"
                    ]
                )

                print(
                    "Judging "
                    f"{index}/"
                    f"{len(rows_to_run)}: "
                    f"{claim_id}"
                )

                prompt = (
                    build_semantic_judge_prompt(
                        human_row
                    )
                )

                response_text = (
                    service.judge(
                        prompt
                    )
                )

                prediction = (
                    parse_semantic_judge_response(
                        response_text,
                        row=human_row,
                    )
                )

                output_row = (
                    build_output_record(
                        human_row,
                        prediction,
                        response_text=(
                            response_text
                        ),
                        prompt=prompt,
                        provider=(
                            clean_provider
                        ),
                        model=(
                            clean_model
                        ),
                    )
                )

                append_output_record(
                    path,
                    output_row,
                )

        finally:
            service.close()

    completed = (
        load_existing_output(
            path,
            human_rows,
            provider=(
                clean_provider
            ),
            model=(
                clean_model
            ),
        )
    )

    agreements = (
        build_agreements(
            human_rows,
            completed,
        )
    )

    if agreements:
        overall = (
            aggregate_judge_agreement(
                agreements
            )
        )

    else:
        overall = {
            "claim_count": 0,
        }

    return (
        agreements,
        overall,
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build command-line interface."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run the automated "
                "semantic citation judge "
                "against the human-reviewed "
                "reference subset."
            )
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            DEFAULT_INPUT_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reset",
        action="store_true",
    )

    parser.add_argument(
        "--model",
        default=(
            DEFAULT_JUDGE_MODEL
        ),
    )

    return parser


def main(
) -> None:
    """Run automated semantic judging."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    agreements, _ = (
        run_semantic_judge(
            args.input,
            output_path=(
                args.output
            ),
            limit=args.limit,
            reset=args.reset,
            model=args.model,
        )
    )

    human_rows = (
        load_review_rows(
            args.input
        )
    )

    print_judge_summary(
        agreements,
        total_claim_count=len(
            human_rows
        ),
    )


if __name__ == "__main__":
    main()