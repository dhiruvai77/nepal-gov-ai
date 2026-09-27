"""Build table-aware retrieval representations for document chunks.

This module is currently used by evaluation experiments only.

The production contextual representation remains unchanged.

The table-aware representation augments the existing contextual metadata with
table, annex, or chart headings that are already present inside the original
chunk text.

No query text, benchmark annotation, gold evidence label, or manually supplied
answer vocabulary is used when constructing the representation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from src.embeddings.contextual_passage import (
    CONTEXTUAL_METADATA_FIELDS,
)


TABLE_MARKER_PATTERN = re.compile(
    r"\b(?:Annex|Table|Chart)"
    r"\s+"
    r"[0-9A-Za-z][0-9A-Za-z./()\-]*"
    r"\s*:?",
    flags=re.IGNORECASE,
)

# Stop a detected heading before common transitions from a table/annex title
# into the table body, source note, footnote, or printed page marker.
TABLE_HINT_STOP_PATTERNS = (
    re.compile(
        r"\s+-\s+\d+\s+-"
    ),
    re.compile(
        r"\s+Source\s*:",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\s+\*"
    ),
    re.compile(
        r"\s+Academic\s+Year\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\s+Indicators\s+\d+[.)]",
        flags=re.IGNORECASE,
    ),
)

DEFAULT_MAX_HINTS = 3
DEFAULT_MAX_HINT_CHARS = 220


def normalize_inline_text(
    text: str,
) -> str:
    """Collapse arbitrary whitespace into one stable inline representation."""

    return (
        " ".join(
            text.split()
        )
    )


def _trim_table_hint(
    text: str,
    *,
    max_chars: int,
) -> str:
    """Trim one candidate structural heading before obvious table-body text."""

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be greater than zero."
        )

    snippet = (
        text[
            :max_chars
        ]
    )

    stop_positions: list[int] = []

    for pattern in (
        TABLE_HINT_STOP_PATTERNS
    ):
        match = (
            pattern.search(
                snippet
            )
        )

        if (
            match is not None
            and match.start() > 0
        ):
            stop_positions.append(
                match.start()
            )

    if stop_positions:
        snippet = (
            snippet[
                :min(
                    stop_positions
                )
            ]
        )

    normalized = (
        normalize_inline_text(
            snippet
        )
        .strip(
            " \t\r\n:;,.-"
        )
    )

    return normalized


def extract_table_hints(
    chunk_text: str,
    *,
    max_hints: int = (
        DEFAULT_MAX_HINTS
    ),
    max_hint_chars: int = (
        DEFAULT_MAX_HINT_CHARS
    ),
) -> tuple[str, ...]:
    """Extract deterministic table/annex/chart headings from passage text.

    The extractor intentionally uses only text already present in the passage.
    It does not infer titles with an LLM and does not consult evaluation labels.

    Multiple headings may be returned because one child chunk can contain more
    than one annex or table after PDF extraction.
    """

    if max_hints <= 0:
        raise ValueError(
            "max_hints must be greater than zero."
        )

    if max_hint_chars <= 0:
        raise ValueError(
            "max_hint_chars must be greater than zero."
        )

    clean_text = (
        str(
            chunk_text
            or ""
        )
        .strip()
    )

    if not clean_text:
        return ()

    matches = list(
        TABLE_MARKER_PATTERN.finditer(
            clean_text
        )
    )

    if not matches:
        return ()

    hints: list[str] = []

    seen: set[str] = set()

    for (
        index,
        match,
    ) in enumerate(
        matches
    ):
        start = (
            match.start()
        )

        hard_end = min(
            len(
                clean_text
            ),
            start
            + max_hint_chars,
        )

        if (
            index + 1
            < len(
                matches
            )
        ):
            hard_end = min(
                hard_end,
                matches[
                    index + 1
                ].start(),
            )

        candidate = (
            clean_text[
                start:hard_end
            ]
        )

        hint = (
            _trim_table_hint(
                candidate,
                max_chars=(
                    max_hint_chars
                ),
            )
        )

        if not hint:
            continue

        normalized_key = (
            hint.casefold()
        )

        if (
            normalized_key
            in seen
        ):
            continue

        seen.add(
            normalized_key
        )

        hints.append(
            hint
        )

        if (
            len(
                hints
            )
            >= max_hints
        ):
            break

    return tuple(
        hints
    )


def is_table_like_passage(
    chunk_text: str,
) -> bool:
    """Return whether the passage exposes a detectable table-like heading."""

    return bool(
        extract_table_hints(
            chunk_text
        )
    )


def build_table_aware_passage_text(
    chunk: Mapping[
        str,
        Any,
    ],
) -> str:
    """Build a deterministic table-aware dense-embedding representation.

    Existing contextual metadata is preserved.

    When table-like headings are present inside ``chunk_text``, they are
    promoted ahead of the original passage as explicit ``Table:`` metadata.

    The original chunk text itself is never altered.
    """

    chunk_text = (
        str(
            chunk.get(
                "chunk_text"
            )
            or ""
        )
        .strip()
    )

    if not chunk_text:
        raise ValueError(
            "chunk_text must contain "
            "non-whitespace text."
        )

    context_lines: list[str] = []

    for (
        label,
        field_name,
    ) in (
        CONTEXTUAL_METADATA_FIELDS
    ):
        value = (
            chunk.get(
                field_name
            )
        )

        if value is None:
            continue

        clean_value = (
            str(
                value
            )
            .strip()
        )

        if not clean_value:
            continue

        context_lines.append(
            f"{label}: {clean_value}"
        )

    table_hints = (
        extract_table_hints(
            chunk_text
        )
    )

    for hint in table_hints:
        context_lines.append(
            f"Table: {hint}"
        )

    return "\n".join(
        [
            *context_lines,
            "Content:",
            chunk_text,
        ]
    )