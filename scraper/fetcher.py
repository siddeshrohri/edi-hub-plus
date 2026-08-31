"""
Fetcher — Stage 2
Responsible for:
  - Iterating over discovered resources
  - Classifying each URL (pdf / webpage)
  - Fetching webpage content
  - Detecting JS-heavy pages
  - Updating each Resource object with status and raw HTML
  - Saving results and a skipped log

It does NOT clean the text — that is Stage 3's job.
"""

import json
import time

from bs4 import BeautifulSoup

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import RATE_LIMIT_SEC
from models import Resource
from scraper.http_client import build_session, fetch
from scraper.classifier import classify_url, is_js_heavy
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_raw_text(html: str) -> str:
    """
    Pull all visible text from raw HTML.
    No cleaning yet — just strip tags so we can count words
    and detect JS-heavy pages. Stage 3 does the real cleaning.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def fetch_resources(resources: list[Resource]) -> tuple[list[Resource], list[dict]]:
    """
    Entry point for Stage 2.

    Takes the Resource list from Stage 1.
    Returns:
      - updated resources (status set, raw_html stored for Stage 3)
      - skipped list (pdfs + js-heavy + failed) for review
    """
    session  = build_session()
    skipped  = []
    total    = len(resources)

    for i, resource in enumerate(resources, 1):
        logger.info(f"[{i}/{total}] {resource.title[:55]}")
        logger.info(f"  URL: {resource.url}")

        # ── Step 1: Classify URL ──────────────────────────────────────────────
        url_type = classify_url(resource.url)
        resource.url_type = url_type

        if url_type == "pdf":
            logger.info("  → PDF detected — skipping for now (Stage 2b)")
            resource.status = "pdf-skip"
            skipped.append({
                "resource_id": resource.resource_id,
                "title":       resource.title,
                "url":         resource.url,
                "reason":      "pdf",
            })
            continue

        # ── Step 2: Fetch the page ────────────────────────────────────────────
        response, fetch_failure = fetch(resource.url, session)

        if response is None:
            logger.warning(f"  → Fetch failed: {fetch_failure}")
            resource.status = "failed"
            skipped.append({
                "resource_id":    resource.resource_id,
                "title":          resource.title,
                "url":            resource.url,
                "reason":         "fetch-failed",
                "failure_detail": fetch_failure,
            })
            continue

        # ── Step 3: Detect JS-heavy pages ────────────────────────────────────
        raw_text = _extract_raw_text(response.text)

        if is_js_heavy(raw_text):
            logger.warning(f"  → JS-heavy page ({len(raw_text.split())} words) — skipping for now (Stage 2c)")
            resource.status = "js-skip"
            skipped.append({
                "resource_id": resource.resource_id,
                "title":       resource.title,
                "url":         resource.url,
                "reason":      "js-heavy",
                "word_count":  len(raw_text.split()),
            })
            continue

        # ── Step 4: Store raw HTML for Stage 3 ───────────────────────────────
        # We store the raw HTML (not the text) so Stage 3 can do proper cleaning
        resource.raw_html = response.text
        resource.status   = "success"
        logger.info(f"  ✓ Fetched — {len(raw_text.split())} words")

    # ── Summary ───────────────────────────────────────────────────────────────
    success  = sum(1 for r in resources if r.status == "success")
    pdf_skip = sum(1 for r in resources if r.status == "pdf-skip")
    js_skip  = sum(1 for r in resources if r.status == "js-skip")
    failed   = sum(1 for r in resources if r.status == "failed")

    logger.info("-" * 60)
    logger.info(f"Stage 2 complete")
    logger.info(f"  ✓ Success   : {success}")
    logger.info(f"  ⊘ PDF skip  : {pdf_skip}")
    logger.info(f"  ⊘ JS skip   : {js_skip}")
    logger.info(f"  ✗ Failed    : {failed}")
    logger.info("-" * 60)

    return resources, skipped