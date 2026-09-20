"""Build a deterministic stratified human-review subset.

The full semantic evaluation dataset contains all claim units extracted from
the production RAG benchmark. Manually reviewing every claim immediately is
expensive, so this module creates a smaller reproducible subset for semantic
citation labeling.

Sampling is stratified across:

* language pair,
* citation count,
* numeric vs non-numeric claims,
* cited vs uncited claims.

Selection inside each stratum uses a stable SHA-256 key derived from claim_id
and the sampling configuration. The result therefore does not depend on source
file ordering or Python random state.

No semantic labels are assigned by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_claims.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/evaluation/semantic/"
    "production_rag_v2_human_review_v1.jsonl"
)

SAMPLE_CONFIG_ID = (
    "production-rag-v2-human-review-v1"
)

PAIR_ORDER = (
    ("en", "en"),
    ("ne", "ne"),
    ("en", "ne"),
    ("ne", "en"),
)

STRATUM_ORDER = (
    "multi_citation",
    "single_numeric",
    "single_other",
    "uncited_numeric",
    "uncited_other",
)

STRATUM_QUOTAS = {
    "multi_citation": 2,
    "single_numeric": 3,
    "single_other": 5,
    "uncited_numeric": 1,
    "uncited_other": 1,
}

EXPECTED_PER_PAIR = sum(
    STRATUM_QUOTAS.values()
)

EXPECTED_TOTAL = (
    EXPECTED_PER_PAIR
    * len(
        PAIR_ORDER
    )
)

NUMERIC_PATTERN = re.compile(
    r"[0-9०-९]"
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

    return value


def load_semantic_rows(
    path: str | Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load the materialized semantic claim dataset."""

    source_path = Path(
        path
    )

    if not source_path.exists():
        raise FileNotFoundError(
            "Semantic evaluation dataset "
            f"does not exist: {source_path}"
        )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen_claim_ids: set[
        str
    ] = set()

    with source_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
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
                    f"dataset on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Semantic dataset row "
                    f"{line_number} must be "
                    "a JSON object."
                )

            claim_id = (
                _require_nonempty_string(
                    row.get(
                        "claim_id"
                    ),
                    field_name=(
                        "claim_id"
                    ),
                )
            )

            if claim_id in seen_claim_ids:
                raise ValueError(
                    "Semantic dataset contains "
                    "duplicate claim_id "
                    f"{claim_id!r}."
                )

            seen_claim_ids.add(
                claim_id
            )

            # Classification also validates the citation structure required by
            # the sampling protocol.
            classify_claim_stratum(
                row
            )

            rows.append(
                row
            )

    if not rows:
        raise ValueError(
            "Semantic evaluation dataset "
            "contains no rows."
        )

    return rows


def contains_numeric_content(
    claim_text: str,
) -> bool:
    """Return whether a claim contains ASCII or Devanagari digits."""

    return bool(
        NUMERIC_PATTERN.search(
            claim_text
        )
    )


def classify_claim_stratum(
    row: Mapping[
        str,
        Any,
    ],
) -> str:
    """Classify one claim into a mutually exclusive review stratum."""

    claim_id = (
        _require_nonempty_string(
            row.get(
                "claim_id"
            ),
            field_name=(
                "claim_id"
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

    evidence_ids = (
        row.get(
            "evidence_ids"
        )
    )

    if not isinstance(
        evidence_ids,
        list,
    ):
        raise ValueError(
            f"Claim {claim_id!r} must contain "
            "evidence_ids as a list."
        )

    for evidence_id in (
        evidence_ids
    ):
        _require_nonempty_string(
            evidence_id,
            field_name=(
                f"evidence_id for "
                f"{claim_id}"
            ),
        )

    has_citation = (
        row.get(
            "has_citation"
        )
    )

    if not isinstance(
        has_citation,
        bool,
    ):
        raise ValueError(
            f"Claim {claim_id!r} must contain "
            "boolean has_citation."
        )

    expected_has_citation = bool(
        evidence_ids
    )

    if (
        has_citation
        != expected_has_citation
    ):
        raise ValueError(
            f"Claim {claim_id!r} has inconsistent "
            "has_citation and evidence_ids."
        )

    numeric = (
        contains_numeric_content(
            claim_text
        )
    )

    citation_count = len(
        evidence_ids
    )

    if citation_count > 1:
        return "multi_citation"

    if citation_count == 1:
        return (
            "single_numeric"
            if numeric
            else "single_other"
        )

    return (
        "uncited_numeric"
        if numeric
        else "uncited_other"
    )


def language_pair(
    row: Mapping[
        str,
        Any,
    ],
) -> tuple[
    str,
    str,
]:
    """Return the query -> target language pair for one claim."""

    query_language = (
        _require_nonempty_string(
            row.get(
                "query_language"
            ),
            field_name=(
                "query_language"
            ),
        )
    )

    target_language = (
        _require_nonempty_string(
            row.get(
                "target_language"
            ),
            field_name=(
                "target_language"
            ),
        )
    )

    pair = (
        query_language,
        target_language,
    )

    if pair not in PAIR_ORDER:
        raise ValueError(
            "Unsupported semantic review "
            f"language pair: "
            f"{query_language}->"
            f"{target_language}."
        )

    return pair


def stable_sample_key(
    claim_id: str,
) -> str:
    """Return a stable deterministic ordering key for one claim."""

    clean_claim_id = (
        _require_nonempty_string(
            claim_id,
            field_name=(
                "claim_id"
            ),
        )
    )

    material = (
        f"{SAMPLE_CONFIG_ID}|"
        f"{clean_claim_id}"
    )

    return hashlib.sha256(
        material.encode(
            "utf-8"
        )
    ).hexdigest()


def _select_stratum(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    *,
    pair: tuple[
        str,
        str,
    ],
    stratum: str,
    quota: int,
) -> list[
    Mapping[
        str,
        Any,
    ]
]:
    """Select one deterministic stratum sample."""

    candidates = [
        row
        for row in rows
        if language_pair(
            row
        )
        == pair
        and classify_claim_stratum(
            row
        )
        == stratum
    ]

    if len(
        candidates
    ) < quota:
        raise ValueError(
            "Insufficient semantic claims for "
            f"{pair[0]}->{pair[1]} "
            f"stratum {stratum!r}: "
            f"required {quota}, "
            f"found {len(candidates)}."
        )

    ordered = sorted(
        candidates,
        key=lambda row: (
            stable_sample_key(
                str(
                    row[
                        "claim_id"
                    ]
                )
            ),
            str(
                row[
                    "claim_id"
                ]
            ),
        ),
    )

    return ordered[
        :quota
    ]


def _copy_for_review(
    row: Mapping[
        str,
        Any,
    ],
    *,
    stratum: str,
) -> dict[
    str,
    Any,
]:
    """Copy one semantic row and attach review-sampling metadata."""

    copied = dict(
        row
    )

    pair = (
        language_pair(
            row
        )
    )

    copied[
        "review_sample_config_id"
    ] = SAMPLE_CONFIG_ID

    copied[
        "review_language_pair"
    ] = (
        f"{pair[0]}->{pair[1]}"
    )

    copied[
        "review_stratum"
    ] = stratum

    copied[
        "review_status"
    ] = "pending"

    return copied


def build_review_subset(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Build the complete deterministic 48-claim review subset."""

    if not rows:
        raise ValueError(
            "At least one semantic claim "
            "is required."
        )

    seen_claim_ids: set[
        str
    ] = set()

    for row in rows:
        claim_id = (
            _require_nonempty_string(
                row.get(
                    "claim_id"
                ),
                field_name=(
                    "claim_id"
                ),
            )
        )

        if claim_id in seen_claim_ids:
            raise ValueError(
                "Semantic rows contain duplicate "
                f"claim_id {claim_id!r}."
            )

        seen_claim_ids.add(
            claim_id
        )

    selected: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for pair in PAIR_ORDER:
        for stratum in (
            STRATUM_ORDER
        ):
            quota = (
                STRATUM_QUOTAS[
                    stratum
                ]
            )

            stratum_rows = (
                _select_stratum(
                    rows,
                    pair=pair,
                    stratum=stratum,
                    quota=quota,
                )
            )

            selected.extend(
                _copy_for_review(
                    row,
                    stratum=stratum,
                )
                for row in (
                    stratum_rows
                )
            )

    if len(
        selected
    ) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Semantic review sampler "
            "produced an unexpected "
            f"row count: {len(selected)}."
        )

    pair_rank = {
        pair: index
        for index, pair
        in enumerate(
            PAIR_ORDER
        )
    }

    # Human review is easier when selected rows from the same question remain
    # adjacent. Sampling itself remains hash-based; this sort affects output
    # presentation only.
    selected.sort(
        key=lambda row: (
            pair_rank[
                language_pair(
                    row
                )
            ],
            str(
                row.get(
                    "question_id",
                    "",
                )
            ),
            int(
                row.get(
                    "claim_index",
                    0,
                )
            ),
            str(
                row[
                    "claim_id"
                ]
            ),
        )
    )

    return selected


def summarize_subset(
    rows: Sequence[
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
        int,
    ],
]:
    """Return deterministic pair/stratum counts for reporting."""

    summary: dict[
        tuple[
            str,
            str,
        ],
        dict[
            str,
            int,
        ],
    ] = {}

    for pair in PAIR_ORDER:
        summary[
            pair
        ] = {
            stratum: 0
            for stratum in (
                STRATUM_ORDER
            )
        }

    for row in rows:
        pair = language_pair(
            row
        )

        stratum = (
            row.get(
                "review_stratum"
            )
        )

        if (
            stratum
            not in STRATUM_ORDER
        ):
            raise ValueError(
                "Review row contains invalid "
                f"stratum {stratum!r}."
            )

        summary[
            pair
        ][
            stratum
        ] += 1

    return summary


def write_jsonl_atomic(
    path: str | Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    """Write the review subset atomically."""

    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
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

    temporary_path.replace(
        output_path
    )


def build_semantic_review_subset(
    source_path: str | Path = (
        DEFAULT_SOURCE_PATH
    ),
    *,
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Load the semantic dataset, sample it, and persist the review subset."""

    source_rows = (
        load_semantic_rows(
            source_path
        )
    )

    selected = (
        build_review_subset(
            source_rows
        )
    )

    write_jsonl_atomic(
        output_path,
        selected,
    )

    return selected


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the review-subset command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic "
            "human semantic-review subset."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=(
            DEFAULT_SOURCE_PATH
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


def main() -> None:
    """Build and report the semantic human-review subset."""

    args = (
        build_argument_parser()
        .parse_args()
    )

    rows = (
        build_semantic_review_subset(
            source_path=(
                args.source
            ),
            output_path=(
                args.output
            ),
        )
    )

    summary = (
        summarize_subset(
            rows
        )
    )

    print(
        "Semantic human-review subset"
    )

    print(
        f"Sampling config: "
        f"{SAMPLE_CONFIG_ID}"
    )

    print(
        f"Claims: "
        f"{len(rows)}"
    )

    for pair in PAIR_ORDER:
        pair_summary = (
            summary[
                pair
            ]
        )

        total = sum(
            pair_summary.values()
        )

        print(
            f"{pair[0]}->{pair[1]}: "
            f"{total} "
            f"(multi="
            f"{pair_summary['multi_citation']}, "
            f"single_numeric="
            f"{pair_summary['single_numeric']}, "
            f"single_other="
            f"{pair_summary['single_other']}, "
            f"uncited_numeric="
            f"{pair_summary['uncited_numeric']}, "
            f"uncited_other="
            f"{pair_summary['uncited_other']})"
        )

    print(
        f"Output: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()