"""Build retrieval-oriented text representations for document chunks."""

from __future__ import annotations

from typing import Any, Mapping


# Keep this metadata set aligned with the successful retrieval ablation.
# Operational metadata such as URLs and timestamps is intentionally excluded
# because it does not describe the semantic meaning of the passage.
CONTEXTUAL_METADATA_FIELDS = (
    ("Document", "title"),
    ("Organization", "organization"),
    ("Document type", "document_type"),
    ("Section", "section"),
    ("Subsection", "subsection"),
    ("Article number", "article_number"),
    ("Article title", "article_title"),
)


def build_contextualized_passage_text(
    chunk: Mapping[str, Any],
) -> str:
    """Build metadata-enriched text for dense passage embeddings.

    The original ``chunk_text`` remains unchanged in Qdrant payloads and is
    still used for citations and answer generation. This representation exists
    only to provide the dense embedding model with document and structural
    context that may otherwise be absent from an isolated child chunk.
    """

    chunk_text = str(
        chunk.get("chunk_text") or ""
    ).strip()

    if not chunk_text:
        raise ValueError(
            "chunk_text must contain non-whitespace text."
        )

    context_lines: list[str] = []

    for label, field_name in CONTEXTUAL_METADATA_FIELDS:
        value = chunk.get(field_name)

        # Optional structural metadata is included only when it contains useful
        # text. This prevents placeholders such as "None" from entering the
        # embedding representation.
        if value is None:
            continue

        clean_value = str(value).strip()

        if not clean_value:
            continue

        context_lines.append(
            f"{label}: {clean_value}"
        )

    return "\n".join(
        [
            *context_lines,
            "Content:",
            chunk_text,
        ]
    )