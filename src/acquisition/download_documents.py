"""Download official Nepal government documents listed in the source manifest."""

import csv
import hashlib
import re
import ssl
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import certifi


# Resolve project paths relative to this file so the script works regardless
# of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "data_sources.csv"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"

# Identify the project while remaining compatible with government websites
# that expect a browser-like User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (compatible; NepalGovAI/1.0; "
    "+https://github.com/dhiruvai77/nepal-gov-ai)"
)

# Use certifi's CA bundle for normal Python HTTPS requests.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class PDFLinkParser(HTMLParser):
    """Collect PDF links exposed through standard HTML anchor elements."""

    def __init__(self) -> None:
        """Initialize storage for discovered PDF href values."""
        super().__init__()
        self.pdf_links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Capture anchor href values that appear to reference PDF files."""

        if tag.lower() != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")

        if not href:
            return

        # Inspect only the URL path because government links may contain query
        # parameters or unusual names such as ".pdf.pdf".
        path = urlparse(href).path.lower()

        if ".pdf" in path:
            self.pdf_links.append(href)


def is_official_government_url(url: str) -> bool:
    """Return True when a URL belongs to Nepal's official .gov.np namespace."""

    hostname = (urlparse(url).hostname or "").lower()

    return hostname == "gov.np" or hostname.endswith(".gov.np")


def build_request(url: str) -> Request:
    """Create an HTTP request with headers suitable for government websites."""

    return Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,*/*",
        },
    )


def fetch_with_urllib(url: str, timeout: int) -> bytes:
    """Fetch a URL using Python HTTPS with certificate verification enabled."""

    with urlopen(
        build_request(url),
        timeout=timeout,
        context=SSL_CONTEXT,
    ) as response:
        return response.read()


def fetch_with_curl(url: str, timeout: int) -> bytes:
    """Fetch a URL using system curl while preserving normal TLS verification."""

    # Windows curl successfully validates some government certificate chains
    # that Python/OpenSSL may reject. We intentionally do NOT use -k/--insecure.
    result = subprocess.run(
        [
            "curl",
            "-L",
            "-sS",
            "--max-time",
            str(timeout),
            "-A",
            USER_AGENT,
            url,
        ],
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        error_message = result.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        raise RuntimeError(
            f"curl failed for {url}: {error_message}"
        )

    return result.stdout


def fetch_url(url: str, timeout: int = 60) -> bytes:
    """Fetch a URL, falling back to curl only for Python TLS failures."""

    try:
        return fetch_with_urllib(
            url,
            timeout,
        )

    except URLError as error:
        # The Ministry of Finance site currently exposes a certificate chain
        # that Windows curl accepts but Python/OpenSSL may reject. Fall back
        # only for certificate-verification errors rather than masking other
        # network problems.
        error_text = str(error)

        if "CERTIFICATE_VERIFY_FAILED" not in error_text:
            raise

        print("  Python TLS verification failed; using curl fallback...")

        return fetch_with_curl(
            url,
            timeout,
        )


def extract_javascript_pdf_links(html: str) -> list[str]:
    """Extract PDF URLs embedded inside JavaScript source code."""

    # Some Ministry of Finance pages load their document viewer using syntax
    # similar to:
    #
    #     var pdf = 'https://example.gov.np/document.pdf';
    #
    # Those URLs are invisible to a normal HTML anchor parser.
    pattern = re.compile(
        r"""['"](?P<url>https?://[^'"]+?\.pdf(?:\?[^'"]*)?)['"]""",
        flags=re.IGNORECASE,
    )

    return [
        match.group("url")
        for match in pattern.finditer(html)
    ]


def resolve_pdf_url(source_page_url: str) -> str:
    """Find an official PDF URL exposed by a government source page."""

    if not is_official_government_url(source_page_url):
        raise ValueError(
            f"Source page is not an official .gov.np URL: "
            f"{source_page_url}"
        )

    html_bytes = fetch_url(
        source_page_url,
        timeout=30,
    )

    # Government pages are Unicode-heavy, especially for Nepali content.
    # UTF-8 with replacement is safer than failing on a malformed character.
    html = html_bytes.decode(
        "utf-8",
        errors="replace",
    )

    candidate_links: list[str] = []

    # First collect conventional <a href="...pdf"> links.
    parser = PDFLinkParser()
    parser.feed(html)

    candidate_links.extend(
        urljoin(source_page_url, href)
        for href in parser.pdf_links
    )

    # Then collect PDFs embedded directly in JavaScript viewers.
    candidate_links.extend(
        extract_javascript_pdf_links(html)
    )

    # Preserve discovery order while removing duplicates.
    unique_candidates = list(
        dict.fromkeys(candidate_links)
    )

    for candidate_url in unique_candidates:
        if is_official_government_url(candidate_url):
            return candidate_url

    raise ValueError(
        f"No official PDF link found on source page: "
        f"{source_page_url}"
    )


def download_pdf(
    url: str,
    destination: Path,
) -> None:
    """Download an official PDF and validate its file signature."""

    if not is_official_government_url(url):
        raise ValueError(
            f"Download URL is not an official .gov.np URL: {url}"
        )

    pdf_bytes = fetch_url(
        url,
        timeout=120,
    )

    # Prevent HTML error pages or redirects from being silently stored as PDFs.
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(
            f"Downloaded content is not a valid PDF: {url}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_bytes(
        pdf_bytes
    )


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the full file into memory."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        # Process large reports in 1 MB blocks to keep memory use predictable.
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest() -> tuple[list[str], list[dict]]:
    """Load the source manifest while preserving its column order."""

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Manifest does not contain a valid header row."
            )

        return reader.fieldnames, list(reader)


def save_manifest(
    fieldnames: list[str],
    records: list[dict],
) -> None:
    """Write updated acquisition metadata back to the manifest."""

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


def process_document(record: dict) -> None:
    """Acquire one pending document and update its manifest metadata."""

    document_id = record["document_id"].strip()
    source_page_url = record["source_page_url"].strip()
    download_url = record["download_url"].strip()
    local_filename = record["local_filename"].strip()

    destination = RAW_PDF_DIR / local_filename

    print(f"Processing: {document_id}")

    try:
        # Prefer an existing verified direct URL when present. Otherwise resolve
        # the PDF automatically from the document's official source page.
        if not download_url:
            print("  Resolving PDF from source page...")

            download_url = resolve_pdf_url(
                source_page_url
            )

            record["download_url"] = download_url

            print(
                f"  Resolved: {download_url}"
            )

        download_pdf(
            download_url,
            destination,
        )

        # Record provenance only after the PDF has passed download validation.
        record["file_hash"] = calculate_sha256(
            destination
        )

        record["retrieved_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        record["acquisition_status"] = "downloaded"

        print(
            f"  Downloaded: {local_filename}"
        )

        print(
            f"  SHA-256: {record['file_hash']}"
        )

    except Exception as error:
        # Keep failed records retryable instead of marking them complete.
        record["acquisition_status"] = "pending"

        print(
            f"  Failed: {error}"
        )


def main() -> None:
    """Download every pending document listed in the source manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    fieldnames, records = load_manifest()

    pending_records = [
        record
        for record in records
        if record["acquisition_status"]
        .strip()
        .lower()
        == "pending"
    ]

    if not pending_records:
        print("No pending documents found.")
        return

    print(
        f"Pending documents found: {len(pending_records)}"
    )

    for record in pending_records:
        process_document(record)

    # Save once per batch so URL resolution, hashes, timestamps and acquisition
    # states remain reproducible in version-controlled metadata.
    save_manifest(
        fieldnames,
        records,
    )


if __name__ == "__main__":
    main()