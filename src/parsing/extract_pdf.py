"""Extract page-level text and metadata from government PDF documents."""

import json
from pathlib import Path

import pymupdf


# Resolve project paths relative to this script so execution does not depend
# on the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PDF = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pdf"
    / "constitution_nepal_current_en.pdf"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted_pages"
OUTPUT_FILE = OUTPUT_DIR / "constitution_nepal_current_en.json"


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text and basic page metadata from a PDF."""

    pages = []

    # PyMuPDF opens the PDF lazily and gives us direct access to each page.
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            # Extract native text first. OCR will only be considered when
            # native extraction produces too little usable text.
            text = page.get_text("text").strip()

            # Pages with fewer than 200 extracted characters are initially
            # flagged as possible OCR candidates.
            needs_ocr = len(text) < 200

            page_record = {
                "page_number": page_index + 1,
                "text": text,
                "character_count": len(text),
                "extraction_method": "native",
                "needs_ocr": needs_ocr,
            }

            pages.append(page_record)

    return pages


def main() -> None:
    """Extract the Constitution PDF and save page-level JSON."""

    if not INPUT_PDF.exists():
        raise FileNotFoundError(f"Input PDF not found: {INPUT_PDF}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pdf_pages(INPUT_PDF)

    # Count pages flagged by the native-text quality check so we can quickly
    # determine whether OCR fallback is likely to be necessary.
    ocr_candidate_count = sum(page["needs_ocr"] for page in pages)

    output = {
        "document_id": "constitution_nepal_current_en",
        "source_file": INPUT_PDF.name,
        "page_count": len(pages),
        "pages": pages,
    }

    # UTF-8 and ensure_ascii=False are important because later documents
    # will contain Nepali Unicode text.
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Extracted {len(pages)} pages")
    print(f"OCR candidates: {ocr_candidate_count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()