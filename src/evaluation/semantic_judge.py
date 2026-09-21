"""Automated semantic citation judging for NepalGov AI.

This module defines the provider-independent semantic-judge contract used to
compare automated semantic citation labels against the human-reviewed reference
set.

The automated judge is evaluation infrastructure only.

Human labels remain the reference standard. Automated predictions must be
measured against the human-reviewed subset before they are used for larger-scale
semantic evaluation.

The judge evaluates:

1. joint semantic support for a claim,
2. whether the claim requires citation,
3. individual support from each cited evidence passage.

No external model call is performed in this module.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any

from src.evaluation.build_semantic_evaluation_dataset import (
    SUPPORTED_CITATION_REQUIREMENT_LABELS,
    SUPPORTED_SEMANTIC_LABELS,
)
from src.evaluation.semantic_review import (
    INDIVIDUAL_SUPPORT_LABELS,
    REVIEW_STATUS_COMPLETED,
    validate_review_row,
)


JUDGE_SCHEMA_VERSION = 1

SUPPORTED_JUDGE_SEMANTIC_LABELS = (
    "supported",
    "partially_supported",
    "unsupported",
    "not_a_factual_claim",
    "needs_review",
)

SUPPORTED_JUDGE_CITATION_REQUIREMENT_LABELS = (
    "required",
    "not_required",
    "unclear",
)

SUPPORTED_JUDGE_INDIVIDUAL_LABELS = (
    "supported",
    "partially_supported",
    "unsupported",
    "needs_review",
)


@dataclass(frozen=True)
class IndividualEvidencePrediction:
    """Automated support judgment for one cited evidence passage."""

    evidence_id: str
    support_label: str
    notes: str | None = None


@dataclass(frozen=True)
class SemanticJudgePrediction:
    """Automated semantic judgment for one claim."""

    claim_id: str
    semantic_support_label: str
    citation_requirement_label: str
    semantic_notes: str | None
    individual_evidence: tuple[
        IndividualEvidencePrediction,
        ...,
    ]


@dataclass(frozen=True)
class SemanticJudgeAgreement:
    """Agreement indicators for one automated prediction."""

    claim_id: str
    query_language: str
    target_language: str

    semantic_exact: bool
    citation_requirement_exact: bool

    individual_count: int
    individual_exact_count: int

    semantic_human_label: str
    semantic_judge_label: str

    citation_requirement_human_label: str
    citation_requirement_judge_label: str


def _require_nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Return a validated non-empty string."""

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

    return value.strip()


def _optional_clean_string(
    value: Any,
) -> str | None:
    """Normalize optional text."""

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "Optional notes must be "
            "a string or null."
        )

    cleaned = (
        value.strip()
    )

    return (
        cleaned
        or None
    )


def build_semantic_judge_prompt(
    row: Mapping[
        str,
        Any,
    ],
) -> str:
    """Build a deterministic semantic-judge prompt for one review row."""

    validate_review_row(
        row
    )

    claim_id = (
        _require_nonempty_string(
            row.get(
                "claim_id"
            ),
            field_name="claim_id",
        )
    )

    question = (
        _require_nonempty_string(
            row.get(
                "query"
            ),
            field_name=(
                f"query for {claim_id}"
            ),
        )
    )

    claim_text = (
        _require_nonempty_string(
            row.get(
                "claim_text"
            ),
            field_name=(
                f"claim_text for "
                f"{claim_id}"
            ),
        )
    )

    cited_evidence = (
        row.get(
            "cited_evidence"
        )
    )

    if not isinstance(
        cited_evidence,
        list,
    ):
        raise ValueError(
            f"Claim {claim_id!r} must "
            "contain cited_evidence."
        )

    evidence_sections: list[
        str
    ] = []

    if cited_evidence:
        for evidence in (
            cited_evidence
        ):
            evidence_id = (
                _require_nonempty_string(
                    evidence.get(
                        "evidence_id"
                    ),
                    field_name=(
                        f"evidence_id for "
                        f"{claim_id}"
                    ),
                )
            )

            title = (
                _require_nonempty_string(
                    evidence.get(
                        "title"
                    ),
                    field_name=(
                        f"title for "
                        f"{claim_id} "
                        f"{evidence_id}"
                    ),
                )
            )

            chunk_text = (
                _require_nonempty_string(
                    evidence.get(
                        "chunk_text"
                    ),
                    field_name=(
                        f"chunk_text for "
                        f"{claim_id} "
                        f"{evidence_id}"
                    ),
                )
            )

            page_start = (
                evidence.get(
                    "page_start"
                )
            )

            page_end = (
                evidence.get(
                    "page_end"
                )
            )

            evidence_sections.append(
                (
                    f"[{evidence_id}]\n"
                    f"Source: {title}\n"
                    f"Pages: "
                    f"{page_start}-{page_end}\n\n"
                    f"{chunk_text}\n"
                    f"[/{evidence_id}]"
                )
            )

        evidence_text = (
            "\n\n".join(
                evidence_sections
            )
        )

    else:
        evidence_text = (
            "NO CITED EVIDENCE"
        )

    evidence_ids = [
        str(
            evidence[
                "evidence_id"
            ]
        )
        for evidence in (
            cited_evidence
        )
    ]

    evidence_ids_json = (
        json.dumps(
            evidence_ids,
            ensure_ascii=False,
        )
    )

    return (
        "You are evaluating citation faithfulness "
        "for a government-document RAG system.\n\n"

        "Judge ONLY from the supplied claim and cited evidence. "
        "Do not use outside knowledge. "
        "Do not repair missing evidence using your own knowledge.\n\n"

        "You must evaluate three separate properties:\n\n"

        "1. JOINT SEMANTIC SUPPORT\n"
        "Judge whether all cited evidence taken together supports the claim.\n\n"

        "Allowed semantic_support_label values:\n"
        "- supported\n"
        "- partially_supported\n"
        "- unsupported\n"
        "- not_a_factual_claim\n"
        "- needs_review\n\n"

        "Rubric:\n"
        "- supported: all material factual content is established by the "
        "evidence. Faithful paraphrase or translation is acceptable.\n"
        "- partially_supported: the core proposition is supported, but a "
        "material number, date, actor, scope, qualification, or part of a "
        "compound claim is missing, wrong, or insufficiently evidenced.\n"
        "- unsupported: the cited evidence does not establish the material "
        "claim or conflicts with it.\n"
        "- not_a_factual_claim: the text is primarily a heading, transition, "
        "introductory framing, or organizational statement rather than an "
        "independent substantive factual claim.\n"
        "- needs_review: the evidence is genuinely too ambiguous, corrupted, "
        "or incomplete to make a defensible judgment.\n\n"

        "For a claim with NO CITED EVIDENCE, do not label it supported or "
        "partially_supported. A substantive uncited factual claim is "
        "unsupported. Pure framing may be not_a_factual_claim.\n\n"

        "2. CITATION REQUIREMENT\n\n"

        "Allowed citation_requirement_label values:\n"
        "- required\n"
        "- not_required\n"
        "- unclear\n\n"

        "Use required for substantive externally verifiable claims about "
        "law, policy, statistics, government actions, programmes, dates, "
        "rights, duties, institutions, or similar factual matters.\n"
        "Use not_required for pure headings, transitions, or organizational "
        "framing that does not independently assert a substantive fact.\n"
        "Use unclear only for genuinely ambiguous cases.\n\n"

        "3. INDIVIDUAL EVIDENCE SUPPORT\n\n"

        "Evaluate each cited evidence passage independently.\n\n"

        "Allowed support_label values:\n"
        "- supported\n"
        "- partially_supported\n"
        "- unsupported\n"
        "- needs_review\n\n"

        "A joint claim may be fully supported even if individual passages "
        "only support separate portions of the claim.\n\n"

        "Return ONLY valid JSON. Do not use Markdown fences.\n\n"

        "The JSON object must have exactly this structure:\n"
        "{\n"
        '  "semantic_support_label": "...",\n'
        '  "citation_requirement_label": "...",\n'
        '  "semantic_notes": "short explanation or null",\n'
        '  "individual_evidence": [\n'
        "    {\n"
        '      "evidence_id": "E1",\n'
        '      "support_label": "...",\n'
        '      "notes": "short explanation or null"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "The individual_evidence array must contain exactly the supplied "
        "evidence IDs, in the same order. "
        f"The required evidence ID list is: {evidence_ids_json}\n\n"

        f"QUESTION:\n{question}\n\n"

        f"CLAIM:\n{claim_text}\n\n"

        f"CITED EVIDENCE:\n{evidence_text}"
    )


def parse_semantic_judge_response(
    response_text: str,
    *,
    row: Mapping[
        str,
        Any,
    ],
) -> SemanticJudgePrediction:
    """Parse and validate one automated semantic-judge JSON response."""

    validate_review_row(
        row
    )

    claim_id = (
        _require_nonempty_string(
            row.get(
                "claim_id"
            ),
            field_name="claim_id",
        )
    )

    if not isinstance(
        response_text,
        str,
    ):
        raise TypeError(
            "response_text must be a string."
        )

    cleaned_response = (
        response_text.strip()
    )

    if not cleaned_response:
        raise ValueError(
            "Semantic judge returned "
            "empty response text."
        )

    try:
        payload = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Semantic judge response "
            "must be valid JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Semantic judge response "
            "must be a JSON object."
        )

    expected_keys = {
        "semantic_support_label",
        "citation_requirement_label",
        "semantic_notes",
        "individual_evidence",
    }

    if set(
        payload
    ) != expected_keys:
        raise ValueError(
            "Semantic judge response has "
            "unexpected JSON fields."
        )

    semantic_label = (
        payload.get(
            "semantic_support_label"
        )
    )

    if (
        semantic_label
        not in SUPPORTED_JUDGE_SEMANTIC_LABELS
    ):
        raise ValueError(
            "Semantic judge returned "
            "an invalid semantic support label."
        )

    citation_requirement = (
        payload.get(
            "citation_requirement_label"
        )
    )

    if (
        citation_requirement
        not in SUPPORTED_JUDGE_CITATION_REQUIREMENT_LABELS
    ):
        raise ValueError(
            "Semantic judge returned an invalid "
            "citation requirement label."
        )

    notes = (
        _optional_clean_string(
            payload.get(
                "semantic_notes"
            )
        )
    )

    raw_individual = (
        payload.get(
            "individual_evidence"
        )
    )

    if not isinstance(
        raw_individual,
        list,
    ):
        raise ValueError(
            "individual_evidence must "
            "be a JSON array."
        )

    cited_evidence = (
        row[
            "cited_evidence"
        ]
    )

    expected_ids = [
        str(
            evidence[
                "evidence_id"
            ]
        )
        for evidence in (
            cited_evidence
        )
    ]

    if len(
        raw_individual
    ) != len(
        expected_ids
    ):
        raise ValueError(
            f"Claim {claim_id!r} requires "
            f"{len(expected_ids)} individual "
            "evidence predictions."
        )

    individual_predictions: list[
        IndividualEvidencePrediction
    ] = []

    for index, (
        raw_prediction,
        expected_evidence_id,
    ) in enumerate(
        zip(
            raw_individual,
            expected_ids,
            strict=True,
        ),
        start=1,
    ):
        if not isinstance(
            raw_prediction,
            dict,
        ):
            raise ValueError(
                "Individual evidence prediction "
                f"{index} must be a JSON object."
            )

        expected_individual_keys = {
            "evidence_id",
            "support_label",
            "notes",
        }

        if set(
            raw_prediction
        ) != expected_individual_keys:
            raise ValueError(
                "Individual evidence prediction "
                f"{index} has unexpected fields."
            )

        evidence_id = (
            _require_nonempty_string(
                raw_prediction.get(
                    "evidence_id"
                ),
                field_name=(
                    "individual evidence_id"
                ),
            )
        )

        if (
            evidence_id
            != expected_evidence_id
        ):
            raise ValueError(
                f"Claim {claim_id!r} expected "
                f"evidence {expected_evidence_id!r} "
                f"at position {index}, "
                f"received {evidence_id!r}."
            )

        support_label = (
            raw_prediction.get(
                "support_label"
            )
        )

        if (
            support_label
            not in SUPPORTED_JUDGE_INDIVIDUAL_LABELS
        ):
            raise ValueError(
                f"Claim {claim_id!r} evidence "
                f"{evidence_id!r} has invalid "
                "support label."
            )

        individual_predictions.append(
            IndividualEvidencePrediction(
                evidence_id=(
                    evidence_id
                ),
                support_label=(
                    support_label
                ),
                notes=(
                    _optional_clean_string(
                        raw_prediction.get(
                            "notes"
                        )
                    )
                ),
            )
        )

    if (
        not row[
            "has_citation"
        ]
        and semantic_label
        in {
            "supported",
            "partially_supported",
        }
    ):
        raise ValueError(
            "Uncited claims cannot be "
            "judged supported or "
            "partially_supported."
        )

    # Keep judge and human-review vocabularies explicitly aligned.
    if (
        semantic_label
        not in SUPPORTED_SEMANTIC_LABELS
    ):
        raise ValueError(
            "Semantic judge label is not "
            "supported by the human-review schema."
        )

    if (
        citation_requirement
        not in SUPPORTED_CITATION_REQUIREMENT_LABELS
    ):
        raise ValueError(
            "Citation requirement label is not "
            "supported by the human-review schema."
        )

    for prediction in (
        individual_predictions
    ):
        if (
            prediction.support_label
            not in INDIVIDUAL_SUPPORT_LABELS
        ):
            raise ValueError(
                "Individual judge label is not "
                "supported by the human-review schema."
            )

    return SemanticJudgePrediction(
        claim_id=(
            claim_id
        ),
        semantic_support_label=(
            semantic_label
        ),
        citation_requirement_label=(
            citation_requirement
        ),
        semantic_notes=(
            notes
        ),
        individual_evidence=tuple(
            individual_predictions
        ),
    )


def prediction_to_dict(
    prediction: SemanticJudgePrediction,
) -> dict[
    str,
    Any,
]:
    """Convert one validated prediction to JSON-serializable data."""

    return {
        "judge_schema_version": (
            JUDGE_SCHEMA_VERSION
        ),
        "claim_id": (
            prediction.claim_id
        ),
        "semantic_support_label": (
            prediction.semantic_support_label
        ),
        "citation_requirement_label": (
            prediction.citation_requirement_label
        ),
        "semantic_notes": (
            prediction.semantic_notes
        ),
        "individual_evidence": [
            {
                "evidence_id": (
                    item.evidence_id
                ),
                "support_label": (
                    item.support_label
                ),
                "notes": (
                    item.notes
                ),
            }
            for item in (
                prediction.individual_evidence
            )
        ],
    }


def compare_prediction_to_human(
    human_row: Mapping[
        str,
        Any,
    ],
    prediction: SemanticJudgePrediction,
) -> SemanticJudgeAgreement:
    """Compare one automated prediction with completed human labels."""

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
            "Prediction claim_id does not "
            "match human row claim_id."
        )

    human_semantic = (
        _require_nonempty_string(
            human_row.get(
                "semantic_support_label"
            ),
            field_name=(
                "human semantic support label"
            ),
        )
    )

    human_requirement = (
        _require_nonempty_string(
            human_row.get(
                "citation_requirement_label"
            ),
            field_name=(
                "human citation requirement label"
            ),
        )
    )

    human_evidence = (
        human_row[
            "cited_evidence"
        ]
    )

    if len(
        human_evidence
    ) != len(
        prediction.individual_evidence
    ):
        raise ValueError(
            f"Claim {claim_id!r} has mismatched "
            "human and judge evidence counts."
        )

    individual_exact_count = 0

    for human_item, judge_item in zip(
        human_evidence,
        prediction.individual_evidence,
        strict=True,
    ):
        human_evidence_id = (
            _require_nonempty_string(
                human_item.get(
                    "evidence_id"
                ),
                field_name=(
                    "human evidence_id"
                ),
            )
        )

        if (
            human_evidence_id
            != judge_item.evidence_id
        ):
            raise ValueError(
                f"Claim {claim_id!r} has "
                "misaligned evidence IDs."
            )

        if (
            human_item.get(
                "individual_support_label"
            )
            == judge_item.support_label
        ):
            individual_exact_count += 1

    query_language = (
        _require_nonempty_string(
            human_row.get(
                "query_language"
            ),
            field_name=(
                "query_language"
            ),
        )
    )

    target_language = (
        _require_nonempty_string(
            human_row.get(
                "target_language"
            ),
            field_name=(
                "target_language"
            ),
        )
    )

    return SemanticJudgeAgreement(
        claim_id=(
            claim_id
        ),
        query_language=(
            query_language
        ),
        target_language=(
            target_language
        ),
        semantic_exact=(
            human_semantic
            == prediction.semantic_support_label
        ),
        citation_requirement_exact=(
            human_requirement
            == prediction.citation_requirement_label
        ),
        individual_count=len(
            human_evidence
        ),
        individual_exact_count=(
            individual_exact_count
        ),
        semantic_human_label=(
            human_semantic
        ),
        semantic_judge_label=(
            prediction.semantic_support_label
        ),
        citation_requirement_human_label=(
            human_requirement
        ),
        citation_requirement_judge_label=(
            prediction.citation_requirement_label
        ),
    )


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """Return a deterministic zero-safe ratio."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def aggregate_judge_agreement(
    agreements: Sequence[
        SemanticJudgeAgreement
    ],
) -> dict[
    str,
    Any,
]:
    """Aggregate automated-vs-human agreement metrics."""

    if not agreements:
        raise ValueError(
            "At least one judge agreement "
            "is required."
        )

    semantic_exact_count = sum(
        item.semantic_exact
        for item in agreements
    )

    requirement_exact_count = sum(
        item.citation_requirement_exact
        for item in agreements
    )

    individual_count = sum(
        item.individual_count
        for item in agreements
    )

    individual_exact_count = sum(
        item.individual_exact_count
        for item in agreements
    )

    semantic_confusion: dict[
        str,
        dict[
            str,
            int,
        ],
    ] = defaultdict(
        lambda: defaultdict(
            int
        )
    )

    requirement_confusion: dict[
        str,
        dict[
            str,
            int,
        ],
    ] = defaultdict(
        lambda: defaultdict(
            int
        )
    )

    for item in agreements:
        semantic_confusion[
            item.semantic_human_label
        ][
            item.semantic_judge_label
        ] += 1

        requirement_confusion[
            item.citation_requirement_human_label
        ][
            item.citation_requirement_judge_label
        ] += 1

    return {
        "claim_count": len(
            agreements
        ),
        "semantic_exact_count": (
            semantic_exact_count
        ),
        "semantic_exact_accuracy": (
            _safe_ratio(
                semantic_exact_count,
                len(
                    agreements
                ),
            )
        ),
        "citation_requirement_exact_count": (
            requirement_exact_count
        ),
        "citation_requirement_accuracy": (
            _safe_ratio(
                requirement_exact_count,
                len(
                    agreements
                ),
            )
        ),
        "individual_evidence_count": (
            individual_count
        ),
        "individual_exact_count": (
            individual_exact_count
        ),
        "individual_exact_accuracy": (
            _safe_ratio(
                individual_exact_count,
                individual_count,
            )
        ),
        "semantic_confusion": {
            human_label: dict(
                judge_counts
            )
            for (
                human_label,
                judge_counts,
            ) in (
                semantic_confusion.items()
            )
        },
        "citation_requirement_confusion": {
            human_label: dict(
                judge_counts
            )
            for (
                human_label,
                judge_counts,
            ) in (
                requirement_confusion.items()
            )
        },
    }


def aggregate_judge_agreement_by_language_pair(
    agreements: Sequence[
        SemanticJudgeAgreement
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Aggregate agreement separately for each query-target language pair."""

    grouped: dict[
        str,
        list[
            SemanticJudgeAgreement
        ],
    ] = defaultdict(
        list
    )

    for agreement in (
        agreements
    ):
        pair = (
            f"{agreement.query_language}"
            f"->{agreement.target_language}"
        )

        grouped[
            pair
        ].append(
            agreement
        )

    return {
        pair: (
            aggregate_judge_agreement(
                pair_agreements
            )
        )
        for (
            pair,
            pair_agreements,
        ) in sorted(
            grouped.items()
        )
    }