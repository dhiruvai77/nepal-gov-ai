"""Tests for the resumable automated semantic-judge runner."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.evaluation.run_semantic_judge import (
    JUDGE_RUN_CONFIG_ID,
    RESULT_SCHEMA_VERSION,
    append_output_record,
    build_agreements,
    build_output_record,
    load_existing_output,
    prediction_from_output_record,
    prompt_sha256,
    run_semantic_judge,
    validate_judge_run_config_id,
    validate_limit,
)
from src.evaluation.semantic_judge import (
    build_semantic_judge_prompt,
    parse_semantic_judge_response,
)


PRODUCTION_REFERENCE_CONFIG_ID = (
    "production-rag-v2-human-review-v1"
)

HARD_CASE_CONFIG_ID = (
    "semantic-judge-hard-cases-v1"
)

HARD_CASE_JUDGE_RUN_CONFIG_ID = (
    "semantic-judge-hard-cases-v1-judge"
)


def make_human_row(
    claim_id: str = "q1_c001",
    *,
    query_language: str = "en",
    target_language: str = "en",
    config_id: str = (
        PRODUCTION_REFERENCE_CONFIG_ID
    ),
) -> dict:
    """Build one completed human reference row."""

    return {
        "claim_id": (
            claim_id
        ),
        "question_id": (
            claim_id.split(
                "_c"
            )[0]
        ),
        "query": (
            "What does the law say?"
        ),
        "query_language": (
            query_language
        ),
        "target_language": (
            target_language
        ),
        "answer_language": (
            query_language
        ),
        "claim_text": (
            "The law states the provision."
        ),
        "has_citation": True,
        "evidence_ids": [
            "E1",
        ],
        "cited_evidence": [
            {
                "evidence_id": "E1",
                "title": "Example Act",
                "page_start": 1,
                "page_end": 1,
                "chunk_text": (
                    "The law directly states "
                    "the provision."
                ),
                "individual_support_label": (
                    "supported"
                ),
                "individual_support_notes": None,
            }
        ],
        "semantic_support_label": (
            "supported"
        ),
        "citation_requirement_label": (
            "required"
        ),
        "semantic_notes": None,
        "review_sample_config_id": (
            config_id
        ),
        "review_language_pair": (
            f"{query_language}"
            f"->{target_language}"
        ),
        "review_stratum": "test",
        "review_status": (
            "completed"
        ),
    }


def valid_response() -> str:
    """Return one valid automated prediction."""

    return (
        json.dumps(
            {
                "semantic_support_label": (
                    "supported"
                ),
                "citation_requirement_label": (
                    "required"
                ),
                "semantic_notes": (
                    "Direct support."
                ),
                "individual_evidence": [
                    {
                        "evidence_id": "E1",
                        "support_label": (
                            "supported"
                        ),
                        "notes": (
                            "Directly stated."
                        ),
                    }
                ],
            }
        )
    )


def write_human_rows(
    path,
    rows,
) -> None:
    """Write human reference rows to JSONL."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )


class FakeJudgeService:
    """Deterministic fake hosted judge."""

    provider = "fake"
    model_name = "fake-model"

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.calls: list[
            str
        ] = []

        self.fail_on_call = (
            fail_on_call
        )

        self.closed = False

    def judge(
        self,
        prompt: str,
    ) -> str:
        """Return one deterministic judge response."""

        self.calls.append(
            prompt
        )

        if (
            self.fail_on_call
            is not None
            and len(
                self.calls
            )
            == self.fail_on_call
        ):
            raise RuntimeError(
                "simulated hosted failure"
            )

        return (
            valid_response()
        )

    def close(
        self,
    ) -> None:
        """Record service closure."""

        self.closed = True


def make_output_row(
    human_row: dict,
    *,
    provider: str = "fake",
    model: str = "fake-model",
    judge_run_config_id: str = (
        JUDGE_RUN_CONFIG_ID
    ),
) -> dict:
    """Build one valid persisted judge row."""

    prompt = (
        build_semantic_judge_prompt(
            human_row
        )
    )

    response = (
        valid_response()
    )

    prediction = (
        parse_semantic_judge_response(
            response,
            row=human_row,
        )
    )

    return (
        build_output_record(
            human_row,
            prediction,
            response_text=(
                response
            ),
            prompt=(
                prompt
            ),
            provider=(
                provider
            ),
            model=(
                model
            ),
            judge_run_config_id=(
                judge_run_config_id
            ),
        )
    )


def test_prompt_sha256_is_deterministic() -> None:
    assert (
        prompt_sha256(
            "example"
        )
        == prompt_sha256(
            "example"
        )
    )

    assert (
        prompt_sha256(
            "example"
        )
        != prompt_sha256(
            "different"
        )
    )


def test_build_output_record_contains_no_human_labels() -> None:
    row = (
        make_human_row()
    )

    output = (
        make_output_row(
            row
        )
    )

    assert (
        output[
            "schema_version"
        ]
        == RESULT_SCHEMA_VERSION
    )

    assert (
        output[
            "judge_run_config_id"
        ]
        == JUDGE_RUN_CONFIG_ID
    )

    assert (
        "semantic_support_label"
        not in output
    )

    assert (
        "citation_requirement_label"
        not in output
    )

    assert (
        output[
            "prediction"
        ][
            "semantic_support_label"
        ]
        == "supported"
    )


def test_build_output_record_supports_custom_judge_run_config() -> None:
    row = (
        make_human_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            )
        )
    )

    output = (
        make_output_row(
            row,
            judge_run_config_id=(
                HARD_CASE_JUDGE_RUN_CONFIG_ID
            ),
        )
    )

    assert (
        output[
            "judge_run_config_id"
        ]
        == HARD_CASE_JUDGE_RUN_CONFIG_ID
    )

    assert (
        output[
            "review_sample_config_id"
        ]
        == HARD_CASE_CONFIG_ID
    )


def test_prediction_from_output_record_round_trips() -> None:
    human_row = (
        make_human_row()
    )

    output_row = (
        make_output_row(
            human_row
        )
    )

    prediction = (
        prediction_from_output_record(
            output_row,
            human_row,
        )
    )

    assert (
        prediction.claim_id
        == "q1_c001"
    )

    assert (
        prediction.semantic_support_label
        == "supported"
    )


def test_load_existing_output_returns_empty_when_missing(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "missing.jsonl"
    )

    loaded = (
        load_existing_output(
            path,
            [
                make_human_row(),
            ],
            provider="fake",
            model="fake-model",
        )
    )

    assert (
        loaded
        == {}
    )


def test_load_existing_output_reuses_valid_prediction(
    tmp_path,
) -> None:
    human_row = (
        make_human_row()
    )

    path = (
        tmp_path
        / "judge.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            human_row
        ),
    )

    loaded = (
        load_existing_output(
            path,
            [
                human_row,
            ],
            provider="fake",
            model="fake-model",
        )
    )

    assert list(
        loaded
    ) == [
        "q1_c001",
    ]


def test_load_existing_output_supports_custom_judge_run_config(
    tmp_path,
) -> None:
    human_row = (
        make_human_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            )
        )
    )

    path = (
        tmp_path
        / "judge.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            human_row,
            judge_run_config_id=(
                HARD_CASE_JUDGE_RUN_CONFIG_ID
            ),
        ),
    )

    loaded = (
        load_existing_output(
            path,
            [
                human_row,
            ],
            provider="fake",
            model="fake-model",
            judge_run_config_id=(
                HARD_CASE_JUDGE_RUN_CONFIG_ID
            ),
        )
    )

    assert list(
        loaded
    ) == [
        "q1_c001",
    ]


def test_load_existing_output_rejects_wrong_judge_run_config(
    tmp_path,
) -> None:
    human_row = (
        make_human_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            )
        )
    )

    path = (
        tmp_path
        / "judge.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            human_row,
            judge_run_config_id=(
                HARD_CASE_JUDGE_RUN_CONFIG_ID
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "different "
            "judge_run_config_id"
        ),
    ):
        load_existing_output(
            path,
            [
                human_row,
            ],
            provider="fake",
            model="fake-model",
            judge_run_config_id=(
                "different-config"
            ),
        )


def test_load_existing_output_rejects_wrong_model(
    tmp_path,
) -> None:
    human_row = (
        make_human_row()
    )

    path = (
        tmp_path
        / "judge.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            human_row
        ),
    )

    with pytest.raises(
        ValueError,
        match="judge_model",
    ):
        load_existing_output(
            path,
            [
                human_row,
            ],
            provider="fake",
            model="different-model",
        )


def test_load_existing_output_rejects_prompt_change(
    tmp_path,
) -> None:
    human_row = (
        make_human_row()
    )

    path = (
        tmp_path
        / "judge.jsonl"
    )

    output = (
        make_output_row(
            human_row
        )
    )

    output[
        "prompt_sha256"
    ] = "wrong"

    append_output_record(
        path,
        output,
    )

    with pytest.raises(
        ValueError,
        match="different prompt",
    ):
        load_existing_output(
            path,
            [
                human_row,
            ],
            provider="fake",
            model="fake-model",
        )


def test_build_agreements_compares_prediction_to_human() -> None:
    human_row = (
        make_human_row()
    )

    output = (
        make_output_row(
            human_row
        )
    )

    agreements = (
        build_agreements(
            [
                human_row,
            ],
            {
                "q1_c001": (
                    output
                ),
            },
        )
    )

    assert (
        len(
            agreements
        )
        == 1
    )

    assert (
        agreements[
            0
        ].semantic_exact
        is True
    )


def test_run_semantic_judge_executes_and_persists(
    tmp_path,
) -> None:
    input_path = (
        tmp_path
        / "human.jsonl"
    )

    output_path = (
        tmp_path
        / "judge.jsonl"
    )

    write_human_rows(
        input_path,
        [
            make_human_row(),
        ],
    )

    service = (
        FakeJudgeService()
    )

    agreements, metrics = (
        run_semantic_judge(
            input_path,
            output_path=(
                output_path
            ),
            provider="fake",
            model="fake-model",
            service_factory=(
                lambda _: service
            ),
        )
    )

    assert (
        len(
            agreements
        )
        == 1
    )

    assert (
        metrics[
            "semantic_exact_accuracy"
        ]
        == 1.0
    )

    assert (
        len(
            service.calls
        )
        == 1
    )

    assert (
        service.closed
        is True
    )

    assert (
        output_path.exists()
    )


def test_run_semantic_judge_supports_hard_case_reference_config(
    tmp_path,
) -> None:
    input_path = (
        tmp_path
        / "hard_reference.jsonl"
    )

    output_path = (
        tmp_path
        / "hard_judge.jsonl"
    )

    write_human_rows(
        input_path,
        [
            make_human_row(
                config_id=(
                    HARD_CASE_CONFIG_ID
                )
            ),
        ],
    )

    service = (
        FakeJudgeService()
    )

    agreements, metrics = (
        run_semantic_judge(
            input_path,
            output_path=(
                output_path
            ),
            judge_run_config_id=(
                HARD_CASE_JUDGE_RUN_CONFIG_ID
            ),
            provider="fake",
            model="fake-model",
            service_factory=(
                lambda _: service
            ),
        )
    )

    assert (
        len(
            agreements
        )
        == 1
    )

    assert (
        metrics[
            "semantic_exact_accuracy"
        ]
        == 1.0
    )

    persisted = (
        json.loads(
            output_path
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )
    )

    assert (
        persisted[
            "judge_run_config_id"
        ]
        == HARD_CASE_JUDGE_RUN_CONFIG_ID
    )

    assert (
        persisted[
            "review_sample_config_id"
        ]
        == HARD_CASE_CONFIG_ID
    )


def test_run_semantic_judge_resumes_without_repeating(
    tmp_path,
) -> None:
    first = (
        make_human_row(
            "q1_c001"
        )
    )

    second = (
        make_human_row(
            "q2_c001"
        )
    )

    input_path = (
        tmp_path
        / "human.jsonl"
    )

    output_path = (
        tmp_path
        / "judge.jsonl"
    )

    write_human_rows(
        input_path,
        [
            first,
            second,
        ],
    )

    append_output_record(
        output_path,
        make_output_row(
            first
        ),
    )

    service = (
        FakeJudgeService()
    )

    agreements, _ = (
        run_semantic_judge(
            input_path,
            output_path=(
                output_path
            ),
            provider="fake",
            model="fake-model",
            service_factory=(
                lambda _: service
            ),
        )
    )

    assert (
        len(
            service.calls
        )
        == 1
    )

    assert (
        len(
            agreements
        )
        == 2
    )


def test_run_semantic_judge_limit_applies_to_new_calls(
    tmp_path,
) -> None:
    rows = [
        make_human_row(
            "q1_c001"
        ),
        make_human_row(
            "q2_c001"
        ),
        make_human_row(
            "q3_c001"
        ),
    ]

    input_path = (
        tmp_path
        / "human.jsonl"
    )

    output_path = (
        tmp_path
        / "judge.jsonl"
    )

    write_human_rows(
        input_path,
        rows,
    )

    service = (
        FakeJudgeService()
    )

    agreements, _ = (
        run_semantic_judge(
            input_path,
            output_path=(
                output_path
            ),
            limit=2,
            provider="fake",
            model="fake-model",
            service_factory=(
                lambda _: service
            ),
        )
    )

    assert (
        len(
            service.calls
        )
        == 2
    )

    assert (
        len(
            agreements
        )
        == 2
    )


def test_completed_prediction_survives_later_failure(
    tmp_path,
) -> None:
    rows = [
        make_human_row(
            "q1_c001"
        ),
        make_human_row(
            "q2_c001"
        ),
    ]

    input_path = (
        tmp_path
        / "human.jsonl"
    )

    output_path = (
        tmp_path
        / "judge.jsonl"
    )

    write_human_rows(
        input_path,
        rows,
    )

    service = (
        FakeJudgeService(
            fail_on_call=2
        )
    )

    with pytest.raises(
        RuntimeError,
        match="simulated hosted failure",
    ):
        run_semantic_judge(
            input_path,
            output_path=(
                output_path
            ),
            provider="fake",
            model="fake-model",
            service_factory=(
                lambda _: service
            ),
        )

    loaded = (
        load_existing_output(
            output_path,
            rows,
            provider="fake",
            model="fake-model",
        )
    )

    assert list(
        loaded
    ) == [
        "q1_c001",
    ]

    assert (
        service.closed
        is True
    )


def test_all_completed_rows_avoid_service_creation(
    tmp_path,
) -> None:
    human_row = (
        make_human_row()
    )

    input_path = (
        tmp_path
        / "human.jsonl"
    )

    output_path = (
        tmp_path
        / "judge.jsonl"
    )

    write_human_rows(
        input_path,
        [
            human_row,
        ],
    )

    append_output_record(
        output_path,
        make_output_row(
            human_row
        ),
    )

    factory = (
        Mock()
    )

    agreements, _ = (
        run_semantic_judge(
            input_path,
            output_path=(
                output_path
            ),
            provider="fake",
            model="fake-model",
            service_factory=(
                factory
            ),
        )
    )

    factory.assert_not_called()

    assert (
        len(
            agreements
        )
        == 1
    )


def test_validate_limit_rejects_non_positive_values() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_limit(
            0
        )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_limit(
            -1
        )


def test_validate_limit_accepts_positive_and_none() -> None:
    assert (
        validate_limit(
            None
        )
        is None
    )

    assert (
        validate_limit(
            3
        )
        == 3
    )


def test_validate_judge_run_config_id_accepts_and_strips_text() -> None:
    assert (
        validate_judge_run_config_id(
            " hard-case-run "
        )
        == "hard-case-run"
    )


def test_validate_judge_run_config_id_rejects_empty_text() -> None:
    with pytest.raises(
        ValueError,
        match="judge_run_config_id",
    ):
        validate_judge_run_config_id(
            "   "
        )