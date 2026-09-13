"""Download official Nepal government documents listed in the source manifest."""

import csv
import hashlib
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


# Resolve project paths relative to this file so the script works regardless
# of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "data_sources.csv"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"

USER_AGENT = (
    "Mozilla/5.0 (compatible; NepalGovAI/1.0; "
    "+https://github.com/dhiruvai77/nepal-gov-ai)"
)


class PDFLinkParser(HTMLParser):
    """Collect candidate PDF links from an HTML source page."""

    def __init__(self) -> None:
        super().__init__()
        self.pdf_links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Capture href values that appear to reference PDF documents."""

        if tag.lower() != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")

        if not href:
            return

        # Some government files have names such as ".pdf.pdf", while others
        # may include query parameters. Looking at the URL path is therefore
        # more robust than checking only whether the full URL ends in ".pdf".
        path = urlparse(href).path.lower()

        if ".pdf" in path:
            self.pdf_links.append(href)


def is_official_government_url(url: str) -> bool:
    """Return True when the URL belongs to an official Nepal government host."""

    hostname = (urlparse(url).hostname or "").lower()

    # Accept gov.np itself and any subdomain such as lawcommission.gov.np or
    # giwmscdnone.gov.np. External third-party download hosts are rejected.
    return hostname == "gov.np" or hostname.endswith(".gov.np")


def build_request(url: str) -> Request:
    """Create an HTTP request with a descriptive user agent."""

    return Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,*/*",
        },
    )


def resolve_pdf_url(source_page_url: str) -> str:
    """Find the first official PDF link exposed by a government source page."""

    if not is_official_government_url(source_page_url):
        raise ValueError(
            f"Source page is not an official .gov.np URL: {source_page_url}"
        )

    with urlopen(build_request(source_page_url), timeout=30) as response:
        html_bytes = response.read()

        # Respect the page's declared character encoding where possible.
        charset = response.headers.get_content_charset() or "utf-8"
        html = html_bytes.decode(charset, errors="replace")

    parser = PDFLinkParser()
    parser.feed(html)

    for href in parser.pdf_links:
        # Convert relative links into absolute URLs using the source page.
        candidate_url = urljoin(source_page_url, href)

        if is_official_government_url(candidate_url):
            return candidate_url

    raise ValueError(
        f"No official PDF link found on source page: {source_page_url}"
    )


def download_pdf(url: str, destination: Path) -> None:
    """Download a PDF and verify that the response contains PDF data."""

    if not is_official_government_url(url):
        raise ValueError(f"Download URL is not an official .gov.np URL: {url}")

    with urlopen(build_request(url), timeout=60) as response:
        pdf_bytes = response.read()

    # PDF files begin with the %PDF signature. Checking it prevents HTML error
    # pages or redirects from silently being stored with a .pdf extension.
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a valid PDF: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pdf_bytes)


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 hash without loading the entire file into memory."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest() -> tuple[list[str], list[dict]]:
    """Load the manifest and preserve its existing column order."""

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Manifest does not contain a header row.")

        return reader.fieldnames, list(reader)


def save_manifest(fieldnames: list[str], records: list[dict]) -> None:
    """Write updated acquisition metadata back to the manifest."""

    # newline="" gives the csv module control over line endings and also fixes
    # the current missing-newline-at-EOF issue when the manifest is rewritten.
    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def process_document(record: dict) -> None:
    """Acquire one pending government document and update its manifest row."""

    document_id = record["document_id"].strip()
    source_page_url = record["source_page_url"].strip()
    download_url = record["download_url"].strip()
    local_filename = record["local_filename"].strip()

    destination = RAW_PDF_DIR / local_filename

    print(f"Processing: {document_id}")

    try:
        # Prefer a manually verified direct URL when one already exists.
        # Otherwise discover the PDF from the official source page.
        if not download_url:
            print("  Resolving PDF from source page...")
            download_url = resolve_pdf_url(source_page_url)
            record["download_url"] = download_url

            print(f"  Resolved: {download_url}")

        download_pdf(download_url, destination)

        record["file_hash"] = calculate_sha256(destination)
        record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        record["acquisition_status"] = "downloaded"

        print(f"  Downloaded: {local_filename}")
        print(f"  SHA-256: {record['file_hash']}")

    except Exception as error:
        # Leave the record pending so a failed acquisition can be retried after
        # inspecting the source page or fixing the metadata.
        record["acquisition_status"] = "pending"
        print(f"  Failed: {error}")


def main() -> None:
    """Download every pending document in the source manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    fieldnames, records = load_manifest()

    pending_records = [
        record
        for record in records
        if record["acquisition_status"].strip().lower() == "pending"
    ]

    if not pending_records:
        print("No pending documents found.")
        return

    print(f"Pending documents found: {len(pending_records)}")

    for record in pending_records:
        process_document(record)

    # Save once after processing the batch so resolved URLs, timestamps, hashes,
    # and acquisition statuses become reproducible manifest metadata.
    save_manifest(fieldnames, records)


if __name__ == "__main__":
    main()