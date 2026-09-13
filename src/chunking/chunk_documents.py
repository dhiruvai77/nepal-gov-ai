"""Create baseline retrieval chunks from page-level extracted government documents.

The chunker preserves page/document provenance and uses the tokenizer from the
planned multilingual E5 embedding model for realistic token counting.

This is still the baseline chunking stage. Structure-aware parent/child
chunking will be added after the baseline pipeline has been validated.
"""

import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


# Resolve paths relative to the repository root so the script can be executed
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "interim" / "extracted_pages"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "child_chunks"


# Use the tokenizer belonging to the embedding model selected for NepalGov AI.
# Loading only the tokenizer does not require PyTorch or the full embedding model.
TOKENIZER_MODEL = "intfloat/multilingual-e5-large-instruct"


# Keep retrieval chunks comfortably below E5's 512-token model limit.
# A 400-token target leaves room for model-specific special tokens and provides
# a useful balance between retrieval precision and contextual completeness.
TARGET_CHUNK_SIZE = 400

# Preserve a modest amount of neighboring context without making adjacent
# chunks excessively repetitive.
CHUNK_OVERLAP = 60

# Very small fragments are usually page numbers, layout artifacts, or other
# low-value content. Heading-like fragments are handled separately.
MIN_FRAGMENT_WORDS = 5


# Some source PDFs contain known graphical pages where OCR produces unusable
# noise. Keep exclusions explicit and document-specific rather than relying on
# an aggressive generic heuristic that could delete valid Nepali text.
SKIP_PAGES_BY_DOCUMENT = {
    "economic_survey_2081_82_ne": {1},
}


def load_tokenizer():
    """Load the multilingual E5 tokenizer used for chunk-size calculation."""

    print(f"Loading tokenizer: {TOKENIZER_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL
    )

    print(
        f"Tokenizer loaded: "
        f"{tokenizer.__class__.__name__}"
    )

    return tokenizer


def normalize_whitespace(text: str) -> str:
    """Normalize line endings and excessive whitespace without flattening paragraphs."""

    # Standardize newline characters because PDF extraction may contain
    # different line-ending styles.
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Strip individual lines while retaining blank lines because they provide
    # useful paragraph boundaries.
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # More than two consecutive newlines provide no additional structure.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def remove_repeated_page_noise(text: str) -> str:
    """Remove simple standalone page-number artifacts from extracted text."""

    cleaned_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Standalone Arabic page numbers usually originate from headers or
        # footers and provide little retrieval value. Numbers within real text
        # remain untouched.
        if re.fullmatch(
            r"\d{1,4}",
            stripped,
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_page_text(text: str) -> str:
    """Apply conservative text cleanup before paragraph segmentation."""

    text = normalize_whitespace(text)
    text = remove_repeated_page_noise(text)
    text = normalize_whitespace(text)

    return text


def is_heading_like(text: str) -> bool:
    """Return True when a short fragment appears to be a structural heading."""

    lowered = text.lower()

    heading_prefixes = (
        "part ",
        "chapter ",
        "section ",
        "article ",
        "schedule ",
    )

    return (
        text.isupper()
        or text.endswith(":")
        or lowered.startswith(heading_prefixes)
    )


def split_into_paragraphs(text: str) -> list[str]:
    """Split cleaned page text into retrieval-friendly paragraph fragments."""

    raw_fragments = re.split(
        r"\n\s*\n",
        text,
    )

    paragraphs: list[str] = []

    for fragment in raw_fragments:
        # PDF extraction often wraps one logical paragraph across many physical
        # lines. Collapse those internal line breaks into regular spaces.
        paragraph = re.sub(
            r"\s+",
            " ",
            fragment,
        ).strip()

        if not paragraph:
            continue

        word_count = len(
            paragraph.split()
        )

        # Preserve normal paragraphs and short structural headings.
        if (
            word_count >= MIN_FRAGMENT_WORDS
            or is_heading_like(paragraph)
        ):
            paragraphs.append(paragraph)

    return paragraphs


def token_count(
    text: str,
    tokenizer,
) -> int:
    """Count content tokens using the actual multilingual E5 tokenizer."""

    if not text:
        return 0

    # verbose=False prevents Transformers from warning when we temporarily
    # inspect oversized source text before splitting it into safe chunks.
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )

    return len(token_ids)


def split_text_by_token_window(
    text: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    """Split oversized text into tokenizer-bounded windows.

    This is a safety fallback for OCR output or unusually long text fragments
    that contain no reliable paragraph or sentence boundaries.
    """

    # This function exists specifically to split oversized text, so the input
    # may legitimately exceed the model's 512-token limit.
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )

    pieces: list[str] = []

    for start in range(
        0,
        len(token_ids),
        max_tokens,
    ):
        window_ids = token_ids[
            start:start + max_tokens
        ]

        # Decoding can normalize some spacing. This fallback is used only when
        # a source fragment cannot be split more naturally.
        piece = tokenizer.decode(
            window_ids,
            skip_special_tokens=True,
        ).strip()

        if piece:
            pieces.append(piece)

    return pieces


def split_oversized_paragraph(
    text: str,
    tokenizer,
) -> list[str]:
    """Split a paragraph that exceeds the configured target size.

    Sentence boundaries are preferred. Token windows are used only when a
    sentence or OCR fragment is itself too large.
    """

    if token_count(
        text,
        tokenizer,
    ) <= TARGET_CHUNK_SIZE:
        return [text]

    # Support standard sentence punctuation and the Devanagari danda.
    sentences = re.split(
        r"(?<=[.!?।])\s+",
        text,
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # OCR sometimes removes nearly all sentence spacing. If sentence splitting
    # yields no useful structure, fall back to deterministic token windows.
    if len(sentences) <= 1:
        return split_text_by_token_window(
            text,
            tokenizer,
            TARGET_CHUNK_SIZE,
        )

    pieces: list[str] = []

    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_count(
            sentence,
            tokenizer,
        )

        # A single sentence may still exceed the target because of OCR or
        # missing punctuation. Split such sentences directly by tokens.
        if sentence_tokens > TARGET_CHUNK_SIZE:
            if current_sentences:
                pieces.append(
                    " ".join(
                        current_sentences
                    )
                )

                current_sentences = []
                current_tokens = 0

            pieces.extend(
                split_text_by_token_window(
                    sentence,
                    tokenizer,
                    TARGET_CHUNK_SIZE,
                )
            )

            continue

        if (
            current_sentences
            and current_tokens + sentence_tokens > TARGET_CHUNK_SIZE
        ):
            pieces.append(
                " ".join(
                    current_sentences
                )
            )

            current_sentences = [
                sentence
            ]

            current_tokens = sentence_tokens

        else:
            current_sentences.append(
                sentence
            )

            current_tokens += sentence_tokens

    if current_sentences:
        pieces.append(
            " ".join(
                current_sentences
            )
        )

    return pieces


def build_page_units(
    document: dict[str, Any],
    tokenizer,
) -> list[dict[str, Any]]:
    """Convert page text into paragraph-level units with page provenance."""

    units: list[dict[str, Any]] = []

    skipped_pages = SKIP_PAGES_BY_DOCUMENT.get(
        document["document_id"],
        set(),
    )

    for page in document["pages"]:
        # Skip only pages that were manually verified as unusable OCR output
        # for this exact source document.
        if page["page_number"] in skipped_pages:
            continue

        cleaned_text = clean_page_text(
            page.get(
                "text",
                "",
            )
        )

        if not cleaned_text:
            continue

        paragraphs = split_into_paragraphs(
            cleaned_text
        )

        for paragraph in paragraphs:
            # Ensure one very large source paragraph cannot force the final
            # retrieval chunk beyond the configured token budget.
            paragraph_parts = split_oversized_paragraph(
                paragraph,
                tokenizer,
            )

            for part in paragraph_parts:
                units.append(
                    {
                        "text": part,
                        "page_number": page["page_number"],
                        "extraction_method": page[
                            "extraction_method"
                        ],
                        "token_count": token_count(
                            part,
                            tokenizer,
                        ),
                    }
                )

    return units


def get_overlap_units(
    chunk_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select whole trailing units totaling roughly the overlap target."""

    overlap_units: list[dict[str, Any]] = []
    overlap_tokens = 0

    # Walk backwards from the completed chunk so the next chunk inherits the
    # most recent contextual information.
    for unit in reversed(
        chunk_units
    ):
        if (
            overlap_units
            and overlap_tokens >= CHUNK_OVERLAP
        ):
            break

        overlap_units.insert(
            0,
            unit,
        )

        overlap_tokens += unit[
            "token_count"
        ]

    return overlap_units


def create_baseline_chunks(
    units: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group paragraph units into approximately 400-token retrieval chunks."""

    chunks: list[
        list[dict[str, Any]]
    ] = []

    current_units: list[
        dict[str, Any]
    ] = []

    current_tokens = 0

    for unit in units:
        unit_tokens = unit[
            "token_count"
        ]

        # Close the current chunk before adding a unit that would exceed the
        # configured target.
        if (
            current_units
            and current_tokens + unit_tokens > TARGET_CHUNK_SIZE
        ):
            chunks.append(
                current_units.copy()
            )

            current_units = get_overlap_units(
                current_units
            )

            current_tokens = sum(
                overlap_unit["token_count"]
                for overlap_unit in current_units
            )

            # The overlap itself may occasionally leave insufficient room for
            # the next unit. Drop the oldest overlap units until the next unit
            # fits within the token target.
            while (
                current_units
                and current_tokens + unit_tokens > TARGET_CHUNK_SIZE
            ):
                removed_unit = current_units.pop(
                    0
                )

                current_tokens -= removed_unit[
                    "token_count"
                ]

        current_units.append(
            unit
        )

        current_tokens += unit_tokens

    if current_units:
        chunks.append(
            current_units.copy()
        )

    return chunks


def determine_extraction_method(
    chunk_units: list[dict[str, Any]],
) -> str:
    """Summarize page extraction provenance for a chunk."""

    methods = {
        unit["extraction_method"]
        for unit in chunk_units
    }

    if len(methods) == 1:
        return next(
            iter(methods)
        )

    # A chunk may span a native page and an OCR page.
    return "mixed"


def build_chunk_record(
    document: dict[str, Any],
    chunk_units: list[dict[str, Any]],
    chunk_index: int,
    tokenizer,
) -> dict[str, Any]:
    """Create one retrieval chunk with citation and provenance metadata."""

    chunk_text = "\n\n".join(
        unit["text"]
        for unit in chunk_units
    )

    page_numbers = [
        unit["page_number"]
        for unit in chunk_units
    ]

    document_id = document[
        "document_id"
    ]

    chunk_id = (
        f"{document_id}_chunk_"
        f"{chunk_index:05d}"
    )

    return {
        "chunk_id": chunk_id,

        # Parent chunks are not part of the baseline implementation yet.
        "parent_chunk_id": None,

        "document_id": document_id,
        "title": document.get(
            "title"
        ),
        "organization": document.get(
            "organization"
        ),
        "category": document.get(
            "domain"
        ),
        "document_type": document.get(
            "document_type"
        ),
        "language": document.get(
            "language"
        ),
        "publication_date": document.get(
            "publication_date"
        ),

        # source_url is the human-facing government page. download_url is the
        # exact acquired PDF endpoint.
        "source_url": document.get(
            "source_url"
        ),
        "download_url": document.get(
            "download_url"
        ),
        "retrieved_at": document.get(
            "retrieved_at"
        ),

        "page_start": min(
            page_numbers
        ),
        "page_end": max(
            page_numbers
        ),

        # These fields are reserved for the later structure-aware chunker.
        "section": None,
        "subsection": None,
        "article_number": None,
        "article_title": None,

        "chunk_index": chunk_index,
        "chunk_text": chunk_text,

        # Count the complete final chunk rather than summing component counts,
        # because tokenizer behavior around joined boundaries can differ by a
        # small number of tokens.
        "token_count": token_count(
            chunk_text,
            tokenizer,
        ),

        "extraction_method": determine_extraction_method(
            chunk_units
        ),
    }


def process_document(
    input_path: Path,
    tokenizer,
) -> None:
    """Create and save baseline chunks for one extracted document."""

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(
            file
        )

    units = build_page_units(
        document,
        tokenizer,
    )

    chunk_unit_groups = create_baseline_chunks(
        units
    )

    chunks = [
        build_chunk_record(
            document=document,
            chunk_units=chunk_units,
            chunk_index=index,
            tokenizer=tokenizer,
        )
        for index, chunk_units
        in enumerate(
            chunk_unit_groups
        )
    ]

    output_record = {
        "document_id": document[
            "document_id"
        ],
        "chunk_count": len(
            chunks
        ),
        "chunks": chunks,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{document['document_id']}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        # Keep Devanagari content human-readable in generated JSON files.
        json.dump(
            output_record,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Processed: "
        f"{document['document_id']}"
    )

    print(
        f"  Chunks: "
        f"{len(chunks)}"
    )

    print(
        f"  Saved to: "
        f"{output_path}"
    )


def main() -> None:
    """Chunk every page-level extracted document in the corpus."""

    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(
            f"Extracted document directory not found: "
            f"{EXTRACTED_DIR}"
        )

    input_paths = sorted(
        EXTRACTED_DIR.glob(
            "*.json"
        )
    )

    print(
        f"Extracted documents found: "
        f"{len(input_paths)}"
    )

    if not input_paths:
        return

    # Load once and reuse for the entire corpus.
    tokenizer = load_tokenizer()

    for input_path in input_paths:
        try:
            process_document(
                input_path,
                tokenizer,
            )

        except Exception as error:
            # Keep one malformed document from blocking the rest of the corpus.
            print(
                f"Failed: "
                f"{input_path.name}: "
                f"{error}"
            )


if __name__ == "__main__":
    main()