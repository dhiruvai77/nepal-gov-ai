"""Extract page-level text and metadata from downloaded government PDFs."""

import csv
import json
from pathlib import Path

import pymupdf


# Resolve all project paths relative to this script so execution does not
# depend on the directory from which the script is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "data_sources.csv"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted_pages"

# Native extraction below this threshold triggers OCR fallback.
# This is an initial heuristic that we can tune after testing more documents.
OCR_CHARACTER_THRESHOLD = 200

# OCR both English and Nepali because the project corpus is multilingual.
OCR_LANGUAGES = "eng+nep"

# Higher DPI generally improves OCR quality at the cost of processing time.
OCR_DPI = 300


def load_downloaded_documents() -> list[dict]:
    """Load manifest records for documents downloaded successfully."""

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        documents = [
            row
            for row in reader
            if row["acquisition_status"].strip().lower() == "downloaded"
        ]

    return documents


def extract_page_text(page: pymupdf.Page) -> tuple[str, str, bool]:
    """Extract page text using native extraction with OCR fallback."""

    # Native PDF text extraction is faster and usually more accurate than OCR,
    # so it is always attempted first.
    native_text = page.get_text("text").strip()

    if len(native_text) >= OCR_CHARACTER_THRESHOLD:
        return native_text, "native", False

    try:
        # full=True OCRs the complete page. This is appropriate for pages where
        # native extraction yielded very little usable text.
        text_page = page.get_textpage_ocr(
            language=OCR_LANGUAGES,
            dpi=OCR_DPI,
            full=True,
        )

        ocr_text = page.get_text(
            "text",
            textpage=text_page,
        ).strip()

        # Use OCR output only when it improves on native extraction. This avoids
        # replacing a small amount of valid native text with poorer OCR output.
        if len(ocr_text) > len(native_text):
            return ocr_text, "ocr", False

        # OCR ran successfully but did not improve extraction quality.
        return native_text, "native", True

    except Exception as error:
        # A single OCR failure should not terminate extraction for the entire
        # document. The page remains flagged for later inspection.
        print(f"  OCR failed on page {page.number + 1}: {error}")
        return native_text, "native", True


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract page-level text and quality metadata from a PDF."""

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text, extraction_method, needs_ocr_review = extract_page_text(page)

            page_record = {
                "page_number": page_index + 1,
                "text": text,
                "character_count": len(text),
                "extraction_method": extraction_method,
                "needs_ocr_review": needs_ocr_review,
            }

            pages.append(page_record)

    return pages


def process_document(document_record: dict) -> None:
    """Extract one downloaded document and save page-level JSON output."""

    document_id = document_record["document_id"]
    local_filename = document_record["local_filename"]

    input_pdf = RAW_PDF_DIR / local_filename
    output_file = OUTPUT_DIR / f"{document_id}.json"

    if not input_pdf.exists():
        print(f"Skipped {document_id}: local PDF not found.")
        return

    pages = extract_pdf_pages(input_pdf)

    # These metrics make extraction quality visible without inspecting every
    # page manually.
    native_page_count = sum(
        page["extraction_method"] == "native" for page in pages
    )
    ocr_page_count = sum(
        page["extraction_method"] == "ocr" for page in pages
    )
    review_page_count = sum(
        page["needs_ocr_review"] for page in pages
    )

    output = {
        "document_id": document_id,
        "title": document_record["title"],
        "organization": document_record["organization"],
        "domain": document_record["domain"],
        "document_type": document_record["document_type"],
        "language": document_record["language"],
        "document_status": document_record["document_status"],
        "source_url": document_record["source_page_url"],
        "source_file": local_filename,
        "file_hash": document_record["file_hash"],
        "page_count": len(pages),
        "native_page_count": native_page_count,
        "ocr_page_count": ocr_page_count,
        "ocr_review_page_count": review_page_count,
        "pages": pages,
    }

    # UTF-8 and ensure_ascii=False preserve Nepali Unicode characters.
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Processed: {document_id}")
    print(f"  Pages: {len(pages)}")
    print(f"  Native pages: {native_page_count}")
    print(f"  OCR pages: {ocr_page_count}")
    print(f"  OCR review pages: {review_page_count}")
    print(f"  Saved to: {output_file}")


def main() -> None:
    """Process every downloaded document listed in the source manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_downloaded_documents()

    if not documents:
        print("No downloaded documents found in the manifest.")
        return

    print(f"Downloaded documents found: {len(documents)}")

    for document_record in documents:
        process_document(document_record)


if __name__ == "__main__":
    main()