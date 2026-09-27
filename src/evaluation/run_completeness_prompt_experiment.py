"""Evaluate a completeness-oriented generation prompt on fixed production context.

This experiment isolates generation behavior from retrieval.

The selected evidence passages are reconstructed from the completed production
answer-quality review. No retrieval, embedding, reranking, Qdrant access, or
context selection is performed.

Two prompt variants are compared against the exact same selected evidence:

baseline
    The current production GroundedPromptBuilder.

completeness_v1
    The current grounded prompt plus general instructions to report substantive
    supported results rather than merely describing tables, annexes, or charts.

The experiment targets the two non-complete production answers for which a
generation intervention is diagnostically meaningful:

en_en_005
    All manually verified gold evidence was selected. This is the clean
    generation-only failure.

en_en_004
    Primary evidence was selected but broader gold evidence was incomplete.
    This is a mixed case. Prompt improvement may recover material information
    already present in the selected primary passage, but cannot recover absent
    supporting evidence.

Production retrieval and production prompting remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any

from src.citations.evidence import (
    process_answer_citations,
)
from src.evaluation.answer_quality_review import (
    REVIEW_CONFIG_ID,
    load_review_rows,
)
from src.evaluation.analyze_answer_completeness_failures import (
    ANALYSIS_CONFIG_ID,
    DEFAULT_OUTPUT_PATH as DEFAULT_ANALYSIS_PATH,
)
from src.generation.base import (
    GenerationRequest,
    GenerationService,
    build_generation_request,
)
from src.generation.evidence_guard import (
    guard_generated_answer,
)
from src.generation.gemini_service import (
    GeminiGenerationService,
    MODEL_NAME,
    PROVIDER_NAME,
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


EXPERIMENT_SCHEMA_VERSION = 1

EXPERIMENT_CONFIG_ID = (
    "production-rag-v2-completeness-prompt-v1"
)

DEFAULT_REVIEW_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_answer_quality_review_v1.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/rag_runs/"
    "production_rag_v2_completeness_prompt_v1.jsonl"
)

VARIANT_BASELINE = "baseline"
VARIANT_COMPLETENESS = "completeness_v1"

VARIANT_ORDER = (
    VARIANT_BASELINE,
    VARIANT_COMPLETENESS,
)

EXPECTED_QUESTION_IDS = (
    "en_en_005",
    "en_en_004",
)

PROMPT_INSERTION_ANCHOR = (
    "\n\nQuestion:\n"
)

COMPLETENESS_RULES = (
    "Completeness rules:\n"
    "- Start with the direct substantive answer supported by the evidence "
    "before optional background or peripheral detail.\n"
    "- Cover each material part of the user's question that is directly "
    "supported by the supplied evidence.\n"
    "- When relevant evidence contains a table, annex, chart, numerical "
    "series, ratio, or examination result, report the substantive values or "
    "results that can be mapped reliably to their labels instead of merely "
    "stating that the source contains such a table or annex.\n"
    "- Do not omit material numbers, dates, legal obligations, exceptions, "
    "qualifications, or result values when they directly answer the question.\n"
    "- Do not infer a value-to-label mapping when PDF extraction is malformed, "
    "truncated, or ambiguous. State the limitation instead of guessing.\n"
    "- Prefer concise coverage of relevant evidence over unrelated detail."
)


class CompletenessPromptBuilder:
    """Add evaluation-only completeness instructions to the production prompt."""

    def __init__(
        self,
        *,
        base_builder: GroundedPromptBuilder | None = None,
    ) -> None:
        """Configure the variant without changing the production builder."""

        self.base_builder = (
            base_builder
            if base_builder is not None
            else GroundedPromptBuilder()
        )

    def __call__(
        self,
        request: GenerationRequest,
    ) -> str:
        """Build the production prompt plus completeness instructions."""

        prompt = (
            self.base_builder(
                request
            )
        )

        if (
            PROMPT_INSERTION_ANCHOR
            not in prompt
        ):
            raise RuntimeError(
                "Production prompt no longer contains "
                "the expected Question section anchor."
            )

        return prompt.replace(
            PROMPT_INSERTION_ANCHOR,
            (
                "\n\n"
                f"{COMPLETENESS_RULES}"
                f"{PROMPT_INSERTION_ANCHOR}"
            ),
            1,
        )


def prompt_sha256(
    prompt: str,
) -> str:
    """Return a stable prompt digest."""

    return hashlib.sha256(
        prompt.encode(
            "utf-8"
        )
    ).hexdigest()


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
    question_id: str,
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
            f"{question_id}: {field_name} "
            "must contain non-whitespace text."
        )

    return value


def _require_page_number(
    value: Any,
    *,
    field_name: str,
    question_id: str,
) -> int:
    """Return one validated page number."""

    if (
        not isinstance(
            value,
            int,
        )
        or value <= 0
    ):
        raise ValueError(
            f"{question_id}: {field_name} "
            "must be a positive integer."
        )

    return value


def load_analysis(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    """Load and validate the deterministic failure-analysis artifact."""

    input_path = Path(
        path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Completeness analysis does not exist: {input_path}"
        )

    try:
        with input_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Completeness analysis contains invalid JSON."
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Completeness analysis must contain one JSON object."
        )

    if (
        data.get(
            "analysis_config_id"
        )
        != ANALYSIS_CONFIG_ID
    ):
        raise ValueError(
            "Unexpected completeness-analysis config ID."
        )

    candidates = (
        data.get(
            "prompt_benchmark_question_ids"
        )
    )

    if not isinstance(
        candidates,
        list,
    ):
        raise ValueError(
            "Completeness analysis contains no valid "
            "prompt benchmark candidate list."
        )

    if (
        set(
            candidates
        )
        != set(
            EXPECTED_QUESTION_IDS
        )
    ):
        raise ValueError(
            "Completeness analysis prompt candidates "
            "do not match the expected two-question slice."
        )

    return data


def select_review_rows(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> list[
    Mapping[
        str,
        Any,
    ]
]:
    """Select the fixed prompt-intervention slice in deterministic order."""

    rows_by_id = {
        row.get(
            "question_id"
        ): row
        for row in rows
    }

    missing = [
        question_id
        for question_id in (
            EXPECTED_QUESTION_IDS
        )
        if (
            question_id
            not in rows_by_id
        )
    ]

    if missing:
        raise ValueError(
            "Answer-quality review is missing "
            "prompt-experiment questions: "
            + ", ".join(
                missing
            )
        )

    selected = [
        rows_by_id[
            question_id
        ]
        for question_id in (
            EXPECTED_QUESTION_IDS
        )
    ]

    for row in selected:
        question_id = (
            row[
                "question_id"
            ]
        )

        if (
            row.get(
                "review_config_id"
            )
            != REVIEW_CONFIG_ID
        ):
            raise ValueError(
                f"{question_id}: unexpected review_config_id."
            )

        if (
            row.get(
                "review_status"
            )
            != "completed"
        ):
            raise ValueError(
                f"{question_id}: human review is not complete."
            )

    return selected


def build_context_from_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> tuple[
    RerankedResult,
    ...,
]:
    """Reconstruct exact generation evidence from the review artifact.

    Reranker scores are intentionally synthetic because prompt construction,
    citation processing, and generation do not consume them. Evidence order,
    canonical passage text, source metadata, and evidence labels are preserved.
    """

    question_id = (
        _require_nonempty_string(
            row.get(
                "question_id"
            ),
            field_name=(
                "question_id"
            ),
            question_id="unknown",
        )
    )

    selected_evidence = (
        row.get(
            "selected_evidence"
        )
    )

    if (
        not isinstance(
            selected_evidence,
            list,
        )
        or not selected_evidence
    ):
        raise ValueError(
            f"{question_id}: selected_evidence "
            "must be a non-empty list."
        )

    context: list[
        RerankedResult
    ] = []

    for (
        index,
        evidence,
    ) in enumerate(
        selected_evidence,
        start=1,
    ):
        if not isinstance(
            evidence,
            Mapping,
        ):
            raise ValueError(
                f"{question_id}: selected evidence "
                "entries must be mappings."
            )

        expected_evidence_id = (
            f"E{index}"
        )

        if (
            evidence.get(
                "evidence_id"
            )
            != expected_evidence_id
        ):
            raise ValueError(
                f"{question_id}: selected evidence order "
                "does not match deterministic evidence IDs."
            )

        result = RetrievalResult(
            point_id=(
                _require_nonempty_string(
                    evidence.get(
                        "point_id"
                    ),
                    field_name="point_id",
                    question_id=question_id,
                )
            ),
            score=0.0,
            chunk_id=(
                _require_nonempty_string(
                    evidence.get(
                        "chunk_id"
                    ),
                    field_name="chunk_id",
                    question_id=question_id,
                )
            ),
            document_id=(
                _require_nonempty_string(
                    evidence.get(
                        "document_id"
                    ),
                    field_name="document_id",
                    question_id=question_id,
                )
            ),
            title=(
                _require_nonempty_string(
                    evidence.get(
                        "title"
                    ),
                    field_name="title",
                    question_id=question_id,
                )
            ),
            organization=(
                _require_nonempty_string(
                    evidence.get(
                        "organization"
                    ),
                    field_name="organization",
                    question_id=question_id,
                )
            ),
            language=(
                _require_nonempty_string(
                    evidence.get(
                        "language"
                    ),
                    field_name="language",
                    question_id=question_id,
                )
            ),
            page_start=(
                _require_page_number(
                    evidence.get(
                        "page_start"
                    ),
                    field_name="page_start",
                    question_id=question_id,
                )
            ),
            page_end=(
                _require_page_number(
                    evidence.get(
                        "page_end"
                    ),
                    field_name="page_end",
                    question_id=question_id,
                )
            ),
            source_url=(
                _require_nonempty_string(
                    evidence.get(
                        "source_url"
                    ),
                    field_name="source_url",
                    question_id=question_id,
                )
            ),
            chunk_text=(
                _require_nonempty_string(
                    evidence.get(
                        "chunk_text"
                    ),
                    field_name="chunk_text",
                    question_id=question_id,
                )
            ),
        )

        context.append(
            RerankedResult(
                result=result,
                rerank_score=0.0,
                original_rank=index,
            )
        )

    return tuple(
        context
    )


def build_request_from_review_row(
    row: Mapping[
        str,
        Any,
    ],
) -> GenerationRequest:
    """Build one generation request using exact persisted selected evidence."""

    question_id = (
        str(
            row.get(
                "question_id"
            )
            or "unknown"
        )
    )

    query = (
        _require_nonempty_string(
            row.get(
                "query"
            ),
            field_name="query",
            question_id=question_id,
        )
    )

    answer_language = (
        row.get(
            "answer_language"
        )
    )

    if (
        not isinstance(
            answer_language,
            str,
        )
        or not answer_language.strip()
    ):
        answer_language = (
            _require_nonempty_string(
                row.get(
                    "query_language"
                ),
                field_name="query_language",
                question_id=question_id,
            )
        )

    return build_generation_request(
        query,
        build_context_from_review_row(
            row
        ),
        answer_language=(
            answer_language
        ),
    )


def build_variant_builder(
    variant: str,
) -> GroundedPromptBuilder | CompletenessPromptBuilder:
    """Return the prompt builder associated with one experiment variant."""

    if (
        variant
        == VARIANT_BASELINE
    ):
        return (
            GroundedPromptBuilder()
        )

    if (
        variant
        == VARIANT_COMPLETENESS
    ):
        return (
            CompletenessPromptBuilder()
        )

    raise ValueError(
        f"Unsupported prompt variant {variant!r}."
    )


def build_output_record(
    row: Mapping[
        str,
        Any,
    ],
    *,
    variant: str,
    request: GenerationRequest,
    prompt: str,
    generated_answer_text: str,
    provider: str | None,
    model: str | None,
) -> dict[
    str,
    Any,
]:
    """Build one persisted generation-only experiment result."""

    question_id = (
        row[
            "question_id"
        ]
    )

    citation_result = (
        process_answer_citations(
            generated_answer_text,
            request.context,
        )
    )

    guarded = (
        guard_generated_answer(
            citation_result,
            answer_language=(
                request.answer_language
            ),
        )
    )

    cited_ids = [
        citation.evidence_id
        for citation in (
            citation_result.citations
        )
    ]

    selected_ids = [
        item.result.point_id
        for item in (
            request.context
        )
    ]

    return {
        "schema_version": (
            EXPERIMENT_SCHEMA_VERSION
        ),
        "run_config_id": (
            EXPERIMENT_CONFIG_ID
        ),
        "source_review_config_id": (
            REVIEW_CONFIG_ID
        ),
        "question_id": (
            question_id
        ),
        "query": (
            request.query
        ),
        "query_language": (
            row[
                "query_language"
            ]
        ),
        "target_language": (
            row[
                "target_language"
            ]
        ),
        "answer_language": (
            request.answer_language
        ),
        "variant": (
            variant
        ),
        "prompt_sha256": (
            prompt_sha256(
                prompt
            )
        ),
        "provider": (
            provider
        ),
        "model": (
            model
        ),
        "selected_point_ids": (
            selected_ids
        ),
        "generated_answer_text": (
            generated_answer_text
        ),
        "answer_text": (
            guarded.answer_text
        ),
        "accepted": (
            guarded.accepted
        ),
        "withheld": (
            guarded.withheld
        ),
        "guard_reason": (
            guarded.reason.value
            if guarded.reason
            is not None
            else None
        ),
        "cited_evidence_ids": (
            cited_ids
        ),
        "invalid_evidence_ids": list(
            citation_result.invalid_evidence_ids
        ),
        "production_changed": False,
    }


def validate_persisted_row(
    persisted: Mapping[
        str,
        Any,
    ],
    source_row: Mapping[
        str,
        Any,
    ],
    *,
    variant: str,
) -> None:
    """Reject stale or incompatible generated experiment rows."""

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

    builder = (
        build_variant_builder(
            variant
        )
    )

    expected_prompt = (
        builder(
            request
        )
    )

    expected = (
        (
            "schema_version",
            EXPERIMENT_SCHEMA_VERSION,
        ),
        (
            "run_config_id",
            EXPERIMENT_CONFIG_ID,
        ),
        (
            "source_review_config_id",
            REVIEW_CONFIG_ID,
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
            variant,
        ),
        (
            "prompt_sha256",
            prompt_sha256(
                expected_prompt
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
                f"{question_id}/{variant}: persisted "
                f"field {field_name!r} does not match "
                "the current experiment."
            )

    answer_text = (
        persisted.get(
            "answer_text"
        )
    )

    generated_answer_text = (
        persisted.get(
            "generated_answer_text"
        )
    )

    if (
        not isinstance(
            answer_text,
            str,
        )
        or not answer_text.strip()
    ):
        raise ValueError(
            f"{question_id}/{variant}: persisted "
            "answer_text is empty."
        )

    if (
        not isinstance(
            generated_answer_text,
            str,
        )
        or not generated_answer_text.strip()
    ):
        raise ValueError(
            f"{question_id}/{variant}: persisted "
            "generated_answer_text is empty."
        )


def load_existing_output(
    path: str | Path,
    source_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    dict[
        str,
        Any,
    ],
]:
    """Load completed variant rows so interrupted experiments can resume."""

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
        tuple[
            str,
            str,
        ],
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
                    "Invalid completeness-prompt JSON "
                    f"on line {line_number}."
                ) from exc

            if not isinstance(
                persisted,
                dict,
            ):
                raise ValueError(
                    "Completeness-prompt row "
                    f"{line_number} must be an object."
                )

            question_id = (
                persisted.get(
                    "question_id"
                )
            )

            variant = (
                persisted.get(
                    "variant"
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
                    "Completeness-prompt output "
                    f"contains unexpected question_id "
                    f"{question_id!r}."
                )

            if (
                variant
                not in VARIANT_ORDER
            ):
                raise ValueError(
                    "Completeness-prompt output "
                    f"contains unsupported variant "
                    f"{variant!r}."
                )

            key = (
                question_id,
                variant,
            )

            if (
                key
                in completed
            ):
                raise ValueError(
                    "Completeness-prompt output "
                    "contains duplicate "
                    f"{question_id}/{variant}."
                )

            validate_persisted_row(
                persisted,
                source_by_id[
                    question_id
                ],
                variant=variant,
            )

            completed[
                key
            ] = (
                persisted
            )

    return completed


def append_output_record(
    path: str | Path,
    record: Mapping[
        str,
        Any,
    ],
) -> None:
    """Persist one completed hosted generation immediately."""

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
                    record
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        handle.flush()


def print_results(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Print paired baseline and completeness-prompt answers."""

    rows_by_key = {
        (
            row[
                "question_id"
            ],
            row[
                "variant"
            ],
        ): row
        for row in rows
    }

    print()
    print(
        "=" * 110
    )

    print(
        "Completeness prompt experiment"
    )

    print(
        "=" * 110
    )

    for question_id in (
        EXPECTED_QUESTION_IDS
    ):
        baseline = (
            rows_by_key[
                (
                    question_id,
                    VARIANT_BASELINE,
                )
            ]
        )

        intervention = (
            rows_by_key[
                (
                    question_id,
                    VARIANT_COMPLETENESS,
                )
            ]
        )

        print()
        print(
            "-" * 110
        )

        print(
            question_id
        )

        print(
            "-" * 110
        )

        print()
        print(
            "BASELINE"
        )

        print(
            baseline[
                "answer_text"
            ]
        )

        print()
        print(
            "COMPLETENESS V1"
        )

        print(
            intervention[
                "answer_text"
            ]
        )

        print()
        print(
            "Structural guard:"
        )

        print(
            "  baseline accepted: "
            f"{baseline['accepted']}"
        )

        print(
            "  completeness accepted: "
            f"{intervention['accepted']}"
        )

        print(
            "  baseline invalid citations: "
            f"{baseline['invalid_evidence_ids']}"
        )

        print(
            "  completeness invalid citations: "
            f"{intervention['invalid_evidence_ids']}"
        )


def run_experiment(
    *,
    review_path: str | Path = (
        DEFAULT_REVIEW_PATH
    ),
    analysis_path: str | Path = (
        DEFAULT_ANALYSIS_PATH
    ),
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    services_by_variant: Mapping[
        str,
        GenerationService,
    ]
    | None = None,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Run missing paired prompt variants using fixed selected evidence."""

    analysis = (
        load_analysis(
            analysis_path
        )
    )

    # Accessing this field is intentional even though select_review_rows uses a
    # fixed order. It ensures the persisted analyzer still identifies this exact
    # two-question prompt benchmark.
    _ = (
        analysis[
            "prompt_benchmark_question_ids"
        ]
    )

    source_rows = (
        select_review_rows(
            load_review_rows(
                review_path
            )
        )
    )

    completed = (
        load_existing_output(
            output_path,
            source_rows,
        )
    )

    owned_services: dict[
        str,
        GenerationService,
    ] = {}

    active_services: dict[
        str,
        GenerationService,
    ] = (
        dict(
            services_by_variant
        )
        if services_by_variant
        is not None
        else {}
    )

    try:
        total = (
            len(
                source_rows
            )
            * len(
                VARIANT_ORDER
            )
        )

        position = 0

        for source_row in (
            source_rows
        ):
            request = (
                build_request_from_review_row(
                    source_row
                )
            )

            for variant in (
                VARIANT_ORDER
            ):
                position += 1

                key = (
                    source_row[
                        "question_id"
                    ],
                    variant,
                )

                if (
                    key
                    in completed
                ):
                    print(
                        f"[{position}/{total}] "
                        f"{key[0]}/{variant}: reuse"
                    )

                    continue

                print(
                    f"[{position}/{total}] "
                    f"{key[0]}/{variant}: generate"
                )

                builder = (
                    build_variant_builder(
                        variant
                    )
                )

                prompt = (
                    builder(
                        request
                    )
                )

                service = (
                    active_services.get(
                        variant
                    )
                )

                if service is None:
                    service = (
                        GeminiGenerationService(
                            prompt_builder=(
                                builder
                            ),
                        )
                    )

                    active_services[
                        variant
                    ] = (
                        service
                    )

                    owned_services[
                        variant
                    ] = (
                        service
                    )

                generation_result = (
                    service.generate(
                        request
                    )
                )

                output_record = (
                    build_output_record(
                        source_row,
                        variant=variant,
                        request=request,
                        prompt=prompt,
                        generated_answer_text=(
                            generation_result.answer_text
                        ),
                        provider=(
                            generation_result.provider
                        ),
                        model=(
                            generation_result.model
                        ),
                    )
                )

                append_output_record(
                    output_path,
                    output_record,
                )

                completed[
                    key
                ] = (
                    output_record
                )

    finally:
        for service in (
            owned_services.values()
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
            (
                question_id,
                variant,
            )
        ]
        for question_id in (
            EXPECTED_QUESTION_IDS
        )
        for variant in (
            VARIANT_ORDER
        )
    ]

    print_results(
        ordered
    )

    return ordered


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the experiment CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Compare current and completeness-oriented "
                "generation prompts using fixed production evidence."
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
        "--analysis",
        type=Path,
        default=(
            DEFAULT_ANALYSIS_PATH
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
    """Run the paired completeness-prompt experiment."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    run_experiment(
        review_path=(
            args.review
        ),
        analysis_path=(
            args.analysis
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