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

# This is an initial heuristic rather than a permanent OCR rule.
# We will refine it once we test scanned English and Nepali documents.
OCR_CHARACTER_THRESHOLD = 200


def load_downloaded_documents() -> list[dict]:
    """Load manifest records for documents that were downloaded successfully."""

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        # Parsing should only run against files that successfully completed
        # the acquisition stage.
        documents = [
            row
            for row in reader
            if row["acquisition_status"].strip().lower() == "downloaded"
        ]

    return documents


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract native page-level text and basic quality metadata from a PDF."""

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            # Native extraction is always attempted before OCR because it is
            # faster and generally cleaner for text-based PDFs.
            text = page.get_text("text").strip()

            # Pages with very little extracted text are marked for later OCR
            # review rather than being OCRed automatically at this stage.
            needs_ocr = len(text) < OCR_CHARACTER_THRESHOLD

            page_record = {
                "page_number": page_index + 1,
                "text": text,
                "character_count": len(text),
                "extraction_method": "native",
                "needs_ocr": needs_ocr,
            }

            pages.append(page_record)

    return pages


def process_document(document_record: dict) -> None:
    """Extract one downloaded document and save its page-level JSON output."""

    document_id = document_record["document_id"]
    local_filename = document_record["local_filename"]

    input_pdf = RAW_PDF_DIR / local_filename
    output_file = OUTPUT_DIR / f"{document_id}.json"

    if not input_pdf.exists():
        print(f"Skipped {document_id}: local PDF not found.")
        return

    pages = extract_pdf_pages(input_pdf)

    # Count OCR candidates now so the pipeline exposes document quality
    # immediately after extraction.
    ocr_candidate_count = sum(page["needs_ocr"] for page in pages)

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
        "ocr_candidate_count": ocr_candidate_count,
        "pages": pages,
    }

    # UTF-8 with ensure_ascii=False preserves Nepali Unicode text when we begin
    # processing Nepali-language documents.
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Processed: {document_id}")
    print(f"  Pages: {len(pages)}")
    print(f"  OCR candidates: {ocr_candidate_count}")
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