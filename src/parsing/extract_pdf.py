"""Extract page-level text from downloaded government PDF documents.

The parser uses native PDF text extraction by default and falls back to
Tesseract OCR when a page contains too little native text.

Some PDFs contain a native text layer with broken Unicode mappings even though
the extracted character count is high. Those documents can be explicitly
configured for full-page OCR using FORCE_OCR_DOCUMENT_IDS.
"""

import csv
import json
from pathlib import Path
from typing import Any

import pymupdf


# Resolve paths relative to the repository root so this script works regardless
# of the directory from which it is executed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "data_sources.csv"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted_pages"


# Pages containing fewer than this number of native characters are considered
# suspicious and are passed through OCR for comparison.
OCR_CHARACTER_THRESHOLD = 200

# Use both English and Nepali Tesseract language models because government
# documents may contain mixed-script text, abbreviations, and English terms.
OCR_LANGUAGES = "eng+nep"

# 300 DPI generally provides a good OCR-quality trade-off for government PDFs.
OCR_DPI = 300


# Some PDFs contain large amounts of native text but use broken font-to-Unicode
# mappings. Character count alone cannot detect this problem.
#
# Manual native-vs-OCR comparison showed that this Nepali Economic Survey has
# systematically corrupted native text such as:
#
#     वर्वकास   instead of   विकास
#     अर्वथथा   instead of   अवस्था
#     थथानीय    instead of   स्थानीय
#
# Tesseract OCR is therefore preferred for the entire document.
#
# This is deliberately document-specific rather than applying OCR to every
# Nepali document. Future Nepali PDFs may contain high-quality Unicode text.
FORCE_OCR_DOCUMENT_IDS = {
    "economic_survey_2081_82_ne",
}


def load_downloaded_documents(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """Load successfully downloaded documents from the acquisition manifest."""

    documents: list[dict[str, str]] = []

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Only documents that successfully completed acquisition should be
            # passed into the parsing pipeline.
            if row.get("acquisition_status") == "downloaded":
                documents.append(row)

    return documents


def perform_full_page_ocr(
    page: pymupdf.Page,
) -> str:
    """Run Tesseract OCR over an entire PDF page and return extracted text."""

    # full=True intentionally ignores any existing native text layer. This is
    # required for documents whose embedded text exists but is semantically
    # corrupted because of broken font encoding.
    text_page = page.get_textpage_ocr(
        language=OCR_LANGUAGES,
        dpi=OCR_DPI,
        full=True,
    )

    return page.get_text(
        "text",
        textpage=text_page,
    ).strip()


def extract_page_text(
    page: pymupdf.Page,
    force_ocr: bool = False,
) -> tuple[str, str, bool]:
    """Extract one page using native text and OCR where appropriate.

    Returns:
        tuple containing:
        - selected page text
        - extraction method: ``native`` or ``ocr``
        - whether the page still requires manual OCR review
    """

    native_text = page.get_text("text").strip()

    # If the document has a known broken Unicode text layer, OCR should be
    # attempted regardless of how many native characters were extracted.
    should_try_ocr = (
        force_ocr
        or len(native_text) < OCR_CHARACTER_THRESHOLD
    )

    # High-volume native text is normally preferred because it preserves word
    # spacing and punctuation better than OCR.
    if not should_try_ocr:
        return native_text, "native", False

    try:
        ocr_text = perform_full_page_ocr(page)

        if force_ocr:
            # For explicitly configured broken-text documents, character count
            # is not a useful quality measure. Prefer OCR whenever it produces
            # meaningful text because manual inspection already established
            # that the native Unicode mapping is unreliable.
            if ocr_text:
                return ocr_text, "ocr", False

            # An empty OCR result is not automatically better than an existing
            # native layer. Keep whatever native text exists and flag the page
            # for later inspection.
            return native_text, "native", True

        # For ordinary pages, OCR is only a fallback for sparse native text.
        # The longer extraction is used as a conservative V1 heuristic.
        if len(ocr_text) > len(native_text):
            return ocr_text, "ocr", False

        # OCR did not improve the extraction. Retain native content but mark the
        # page so it remains visible during data-quality review.
        return native_text, "native", True

    except Exception as error:
        # OCR failure should not abort extraction of an entire document.
        # Preserve native text and mark the page for inspection instead.
        print(f"    OCR failed: {error}")

        return native_text, "native", True


def extract_pdf_pages(
    pdf_path: Path,
    force_ocr: bool = False,
) -> list[dict[str, Any]]:
    """Extract text and provenance metadata from every page of one PDF."""

    pages: list[dict[str, Any]] = []

    document = pymupdf.open(pdf_path)

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1

            text, extraction_method, needs_ocr_review = extract_page_text(
                page,
                force_ocr=force_ocr,
            )

            pages.append(
                {
                    # Store human-readable page numbers rather than PyMuPDF's
                    # zero-based indexes because these values will later be used
                    # directly in RAG citations.
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                    "extraction_method": extraction_method,
                    "needs_ocr_review": needs_ocr_review,
                }
            )

    finally:
        document.close()

    return pages


def build_document_record(
    manifest_row: dict[str, str],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine manifest provenance with extracted page-level text."""

    native_page_count = sum(
        1
        for page in pages
        if page["extraction_method"] == "native"
    )

    ocr_page_count = sum(
        1
        for page in pages
        if page["extraction_method"] == "ocr"
    )

    ocr_review_page_count = sum(
        1
        for page in pages
        if page["needs_ocr_review"]
    )

    return {
        "document_id": manifest_row["document_id"],
        "title": manifest_row["title"],
        "organization": manifest_row["organization"],
        "domain": manifest_row["domain"],
        "document_type": manifest_row["document_type"],
        "language": manifest_row["language"],
        "publication_date": manifest_row.get("publication_date") or None,
        "document_status": manifest_row.get("document_status") or None,

        # Keep both the source landing page and direct PDF URL. The landing page
        # is useful for human-facing citations, while download_url preserves the
        # exact acquisition provenance.
        "source_url": manifest_row.get("source_page_url") or None,
        "download_url": manifest_row.get("download_url") or None,
        "retrieved_at": manifest_row.get("retrieved_at") or None,

        "source_file": manifest_row["local_filename"],
        "file_hash": manifest_row.get("file_hash") or None,

        # These document-level statistics make extraction quality easy to
        # inspect without scanning every page manually.
        "page_count": len(pages),
        "native_page_count": native_page_count,
        "ocr_page_count": ocr_page_count,
        "ocr_review_page_count": ocr_review_page_count,

        "pages": pages,
    }


def process_document(
    manifest_row: dict[str, str],
) -> None:
    """Extract and save one downloaded government document."""

    document_id = manifest_row["document_id"]
    local_filename = manifest_row["local_filename"]

    pdf_path = RAW_PDF_DIR / local_filename
    output_path = OUTPUT_DIR / f"{document_id}.json"

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found for {document_id}: {pdf_path}"
        )

    # Force OCR only for documents whose native text layer has already been
    # verified as unreliable.
    force_ocr = document_id in FORCE_OCR_DOCUMENT_IDS

    print(f"Processing: {document_id}")

    if force_ocr:
        print("  OCR policy: forced full-page OCR")
    else:
        print("  OCR policy: native text with sparse-page OCR fallback")

    pages = extract_pdf_pages(
        pdf_path,
        force_ocr=force_ocr,
    )

    document_record = build_document_record(
        manifest_row,
        pages,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        # ensure_ascii=False is necessary to keep Nepali Devanagari text
        # readable in the generated JSON rather than escaping every character.
        json.dump(
            document_record,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  Pages: {document_record['page_count']}")
    print(f"  Native pages: {document_record['native_page_count']}")
    print(f"  OCR pages: {document_record['ocr_page_count']}")
    print(
        "  OCR review pages: "
        f"{document_record['ocr_review_page_count']}"
    )
    print(f"  Saved to: {output_path}")


def main() -> None:
    """Extract all successfully downloaded documents in the manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    documents = load_downloaded_documents(
        MANIFEST_PATH
    )

    if not documents:
        print("No downloaded documents found in the manifest.")
        return

    print(f"Downloaded documents found: {len(documents)}")

    for document in documents:
        try:
            process_document(document)

        except Exception as error:
            # One malformed PDF should not prevent the remaining corpus from
            # being processed. The failed document remains visible in console
            # output for investigation.
            print(
                f"  Failed to process "
                f"{document['document_id']}: {error}"
            )

    print("PDF extraction complete.")


if __name__ == "__main__":
    main()