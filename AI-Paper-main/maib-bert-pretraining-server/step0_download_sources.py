"""
STEP 0 - Download the unlabelled MAIB texts for stage 1 (masked language modelling).

Two sources on gov.uk, both published by the MAIB (Open Government Licence):
  reports   MAIB investigation reports and safety bulletins       https://www.gov.uk/maib-reports
  digests   MAIB Safety Digests (about 2 per year since 1990)     https://www.gov.uk/government/collections/maib-safety-digests

Only documents up to --until (default 2020) are downloaded. The labelled dataset was built from MAIB
occurrences of recent years; older documents describe OTHER accidents, so no test report can be read in
stage 1, not even in other words.

The script uses GOV.UK's public JSON API (no login): the search API lists the reports, the content API
gives the PDF attachments of each page. It waits between requests, skips files that already exist (so it
can be stopped and started again) and writes a list of everything it downloaded.

  uv run python step0_download_sources.py --dry-run                # only list what would be downloaded
  uv run python step0_download_sources.py                          # reports and digests up to 2020
  uv run python step0_download_sources.py --what digests           # only the Safety Digests
  uv run python step0_download_sources.py --since 2010 --max-reports 100

Output: corpus_sources/reports/*.pdf, corpus_sources/digests/*.pdf, reports/downloads.csv (list of the files)
"""
import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

import config

GOVUK = "https://www.gov.uk"
SEARCH_API = GOVUK + "/api/search.json"
CONTENT_API = GOVUK + "/api/content"
DIGEST_COLLECTION = "/government/collections/maib-safety-digests"
HEADERS = {"User-Agent": "MSc research (maritime NLP); MAIB open data download"}

parser = argparse.ArgumentParser()
parser.add_argument("--what", choices=["reports", "digests", "both"], default="both")
parser.add_argument("--since", type=int, default=1990, help="first year to download")
parser.add_argument("--until", type=int, default=2020, help="last year to download (keep it before the labelled data)")
parser.add_argument("--max-reports", type=int, help="stop after this many report pages (for a quick start)")
parser.add_argument("--dry-run", action="store_true", help="list the files without downloading them")
parser.add_argument("--delay", type=float, default=0.5, help="seconds to wait between requests")
parser.add_argument("--out", default=config.SOURCES_DIR)
args = parser.parse_args()

session = requests.Session()
session.headers.update(HEADERS)


def get(url, **kwargs):
    """GET with 3 retries; waits --delay seconds before every request to be polite to gov.uk."""
    for attempt in range(3):
        time.sleep(args.delay)
        try:
            response = session.get(url, timeout=60, **kwargs)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            if attempt == 2:
                print(f"  [failed] {url}: {error}")
                return None
            time.sleep(5 * (attempt + 1))


def year_of(*texts):
    """First plausible year (1980-2039) found in the given strings, or None."""
    for text in texts:
        match = re.search(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)", text or "")
        if match:
            return int(match.group(1))
    return None


def pdf_attachments(base_path):
    """PDF links of one gov.uk page: from the content API, else from the page's HTML."""
    response = get(CONTENT_API + base_path)
    pdfs = []
    if response is not None:
        page = response.json()
        for attachment in page.get("details", {}).get("attachments", []):
            url = attachment.get("url", "")
            if url.lower().endswith(".pdf") or "pdf" in attachment.get("content_type", ""):
                pdfs.append((urljoin(GOVUK, url), attachment.get("title", "")))
        if not pdfs:                       # some pages list their documents as HTML inside the JSON
            for html in page.get("details", {}).get("documents", []):
                pdfs += [(urljoin(GOVUK, u), "") for u in re.findall(r'href="([^"]+\.pdf)"', html)]
    if not pdfs:
        html = get(GOVUK + base_path)
        if html is not None:
            pdfs = [(urljoin(GOVUK, u), "") for u in re.findall(r'href="([^"]+\.pdf)"', html.text)]
    return list(dict.fromkeys(pdfs)), (response.json() if response is not None else {})


def report_pages():
    """Every MAIB report page (investigation reports, safety bulletins, ...), newest first."""
    pages, start = [], 0
    while True:
        response = get(SEARCH_API, params={"filter_format": "maib_report", "count": 500, "start": start,
                                           "fields": ["title", "link", "public_timestamp"],
                                           "order": "-public_timestamp"})
        if response is None:
            break
        results = response.json().get("results", [])
        pages += results
        if len(results) < 500:
            break
        start += 500
    return pages


def download(url, folder, source, title, year, rows):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(url.split("?")[0]).name)
    path = Path(args.out) / folder / name
    rows.append({"source": source, "year": year, "title": title, "url": url, "file": str(path)})
    if args.dry_run or path.exists():
        return
    response = get(url)
    if response is not None and response.content[:4] == b"%PDF":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    else:
        rows[-1]["file"] = "FAILED"


def main():
    rows = []
    in_range = lambda year: year is not None and args.since <= year <= args.until

    if args.what in ("reports", "both"):
        pages = report_pages()
        print(f"MAIB reports: {len(pages)} pages found on gov.uk")
        used = 0
        for page in pages:
            if args.max_reports and used >= args.max_reports:
                break
            # quick filter on the publication date: a report comes out after its accident, usually within
            # 1-2 years, so reports published before --since or long after --until can be skipped unread
            published = year_of(page.get("public_timestamp"))
            if published is not None and not args.since <= published <= args.until + 3:
                continue
            pdfs, content = pdf_attachments(page["link"])
            metadata = content.get("details", {}).get("metadata", {})
            year = year_of(metadata.get("date_of_occurrence"), page.get("public_timestamp"))
            if not in_range(year):
                continue
            used += 1
            print(f"  [{year}] {page.get('title', '')[:70]}  ({len(pdfs)} PDF)")
            for url, title in pdfs:
                download(url, "reports", "maib_report", page.get("title", ""), year, rows)

    if args.what in ("digests", "both"):
        response = get(CONTENT_API + DIGEST_COLLECTION)
        documents = response.json().get("links", {}).get("documents", []) if response is not None else []
        print(f"Safety Digests: {len(documents)} publication pages in the collection")
        for document in documents:
            pdfs, _ = pdf_attachments(document["base_path"])
            for url, title in pdfs:
                # the year is in the file or attachment title, e.g. "2015-SD1-MAIBSafetyDigest.pdf"
                year = year_of(Path(url).name, title, document.get("title"))
                if not in_range(year):
                    continue
                print(f"  [{year}] {title or Path(url).name}")
                download(url, "digests", "safety_digest", title or document.get("title", ""), year, rows)

    out = Path(config.REPORTS_DIR)             # not in corpus_sources/: step 2 would read it as text
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "downloads.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "year", "title", "url", "file"])
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(r["file"] == "FAILED" for r in rows)
    verb = "would be downloaded" if args.dry_run else "downloaded or already present"
    print(f"\n{len(rows) - failed} PDF files {verb} ({args.since}-{args.until}); {failed} failed.")
    print(f"List: {out / 'downloads.csv'}")


if __name__ == "__main__":
    main()
