"""Download pending government documents listed in the source manifest."""

import csv
import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


# Resolve paths relative to the project root so the script works
# regardless of which directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "data_sources.csv"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"


def is_official_government_url(url: str) -> bool:
    """Return True when the URL belongs to Nepal's .gov.np namespace."""

    hostname = urlparse(url).hostname or ""

    # Both gov.np itself and its subdomains are considered official.
    return hostname == "gov.np" or hostname.endswith(".gov.np")


def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the whole PDF into memory."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        # Reading in chunks keeps this safe for large government PDFs.
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def download_pdf(url: str, destination: Path) -> None:
    """Download a PDF and verify that the downloaded file is actually a PDF."""

    request = urllib.request.Request(
        url,
        headers={
            # Some government web servers reject requests without a browser-like
            # User-Agent, so we identify the project explicitly.
            "User-Agent": "NepalGovAI/1.0 document-research-project"
        },
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()

    # PDF files should begin with the %PDF signature.
    # This prevents accidentally saving an HTML error page as a .pdf file.
    if not content.startswith(b"%PDF"):
        raise ValueError("Downloaded content does not appear to be a valid PDF.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def process_manifest() -> None:
    """Download all pending documents and update their acquisition metadata."""

    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("Manifest does not contain a valid CSV header.")

    for row in rows:
        # Only process records that have not already been downloaded.
        if row["acquisition_status"].strip().lower() != "pending":
            continue

        document_id = row["document_id"]
        download_url = row["download_url"]
        local_filename = row["local_filename"]

        print(f"Processing: {document_id}")

        # Provenance rule: V1 downloads must come from an official Nepal
        # government domain rather than third-party mirrors.
        if not is_official_government_url(download_url):
            print(f"  Skipped: non-government download URL: {download_url}")
            continue

        destination = RAW_PDF_DIR / local_filename

        try:
            download_pdf(download_url, destination)

            # Record integrity and acquisition metadata only after the
            # download has successfully passed the PDF validation check.
            row["file_hash"] = calculate_sha256(destination)
            row["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            row["acquisition_status"] = "downloaded"

            print(f"  Downloaded: {destination}")
            print(f"  SHA-256: {row['file_hash']}")

        except Exception as exc:
            # Keep the record pending so it can safely be retried later.
            print(f"  Failed: {exc}")

    # Rewrite the manifest with the updated retrieval timestamp,
    # checksum, and acquisition status.
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    process_manifest()