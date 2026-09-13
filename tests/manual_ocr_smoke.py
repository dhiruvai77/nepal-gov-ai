"""Manual smoke test for the page-level OCR fallback."""

import sys
from pathlib import Path

import pymupdf


# Resolve the repository root so this script can import the project's
# extraction code when executed directly from the command line.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parsing.extract_pdf import extract_pdf_pages  # noqa: E402


SOURCE_PDF = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pdf"
    / "constitution_nepal_current_en.pdf"
)

# The generated PDF contains only a raster image, so native PDF text
# extraction should fail and force the production OCR fallback to run.
OCR_TEST_PDF = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pdf"
    / "ocr_smoke_test.pdf"
)


def create_image_only_pdf() -> None:
    """Convert the first Constitution page into an image-only PDF."""

    with pymupdf.open(SOURCE_PDF) as source_document:
        source_page = source_document[0]

        # Render at sufficient resolution to resemble a reasonably clear
        # scanned government document.
        pixmap = source_page.get_pixmap(dpi=200)
        image_bytes = pixmap.tobytes("png")

        output_document = pymupdf.open()

        # Preserve approximately the same physical page dimensions as the
        # original document.
        output_page = output_document.new_page(
            width=source_page.rect.width,
            height=source_page.rect.height,
        )

        output_page.insert_image(
            output_page.rect,
            stream=image_bytes,
        )

        output_document.save(OCR_TEST_PDF)
        output_document.close()


def main() -> None:
    """Generate the image-only PDF and run the production extractor."""

    create_image_only_pdf()

    pages = extract_pdf_pages(OCR_TEST_PDF)
    page = pages[0]

    print(f"Extraction method: {page['extraction_method']}")
    print(f"Character count: {page['character_count']}")
    print(f"Needs OCR review: {page['needs_ocr_review']}")
    print()
    print("Text preview:")
    print(page["text"][:500])


if __name__ == "__main__":
    main()