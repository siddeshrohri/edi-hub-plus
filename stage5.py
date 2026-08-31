"""
Stage 5 — One-Level Deep Link Extraction
=========================================
For each resource in stage3_resources.json, follow its external_links (web pages)
and pdf_links (PDFs) one level deep, extract structured content from each, and
produce a fully traceable output: every piece of extracted content is tied back to
its parent resource and the exact link it came from.

Output: stage5_linked_content.json

Usage:
    python stage5.py
    python stage5.py --input stage3_resources.json --output stage5_linked_content.json
    python stage5.py --max-ext 10          # cap external links per resource (default 10)
    python stage5.py --max-pdf 10          # cap PDF links per resource (default 10)
    python stage5.py --resource-id 46      # only process one resource (for testing)
    python stage5.py --dry-run             # show what would be fetched, without fetching
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.robotparser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Stage 4 PDF extraction pipeline ──────────────────────────────────────────
# Import stage4 as a module so linked PDFs go through the same full pymupdf-layout
# pipeline (sectionisation, TOC detection, heading confidence, etc.) as direct PDFs.
try:
    import importlib.util as _ilu
    _s4_spec = _ilu.spec_from_file_location("stage4", Path(__file__).parent / "stage4.py")
    _s4 = _ilu.module_from_spec(_s4_spec)
    _s4_spec.loader.exec_module(_s4)
    HAS_STAGE4 = True
except Exception as _s4_err:
    HAS_STAGE4 = False
    _s4 = None

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INPUT  = "stage3_resources.json"
DEFAULT_OUTPUT = "stage5_linked_content.json"

PDF_LINKED_DIR             = "pdfs_linked"   # cache folder for Stage 5 linked PDFs
MAX_EXT_LINKS_PER_RESOURCE = 20   # cap so 87-link resources don't explode runtime
MAX_PDF_LINKS_PER_RESOURCE = 20
MAX_PDF_LINKS_PER_RESOURCE = 20
MAX_PDF_PAGES              = 100
REQUEST_TIMEOUT_WEB        = 20   # seconds
REQUEST_TIMEOUT_PDF        = 60
RATE_LIMIT_DELAY           = 1.5  # seconds between requests to same domain
USER_AGENT                 = (
    "EDIHub-Scraper/1.0 (University of Leeds research project; "
    "contact: s.shinde@leeds.ac.uk)"
)

# OLMo context window and prompt overhead — matches Stage 3 / Stage 4 threshold
OLMO_CONTEXT_WINDOW = 4096
PROMPT_OVERHEAD     = 300
TOKEN_SAFE_LIMIT    = OLMO_CONTEXT_WINDOW - PROMPT_OVERHEAD   # 3,796

# URLs that are clearly not EDI content pages — skip these silently
SKIP_URL_PATTERNS = [
    r"^mailto:",
    r"^tel:",
    r"linkedin\.com/shareArticle",
    r"twitter\.com/(intent|home\?status)",
    r"facebook\.com/sharer",
    r"forms\.cloud\.microsoft",
    r"creativecommons\.org/licenses",
    r"#$",                           # anchor-only
    r"\.(jpg|jpeg|png|gif|svg|ico|mp4|mp3|zip|exe)(\?.*)?$",
]
SKIP_COMPILED = [re.compile(p, re.IGNORECASE) for p in SKIP_URL_PATTERNS]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage5")


# ═════════════════════════════════════════════════════════════════════════════
# Tokeniser (lazy-loaded, cached at module level — same as Stage 3 / Stage 4)
# ═════════════════════════════════════════════════════════════════════════════

_tokenizer = None

def get_tokenizer():
    """
    Load OLMo-2's tokeniser once and cache it for the lifetime of the process.
    Uses allenai/OLMo-2-7B-1124 — standard HF tokeniser, no custom packages needed.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        log.info("[TOKENISER] Loading OLMo-2 tokeniser (first call only)...")
        _tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-7B-1124")
        log.info("[TOKENISER] Ready.")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Return exact OLMo token count for a string. Returns 0 for empty input."""
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def should_skip_url(url: str) -> bool:
    return any(p.search(url) for p in SKIP_COMPILED)


def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
_domain_last_request: dict[str, float] = {}


def get_robots(domain: str) -> urllib.robotparser.RobotFileParser | None:
    if domain in _robots_cache:
        return _robots_cache[domain]
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"https://{domain}/robots.txt")
    try:
        # Fetch robots.txt ourselves so we can inspect the response properly.
        # If the server returns anything other than 2xx, treat as no-robots
        # (i.e. default to allowed) rather than letting urllib misparse it.
        resp = requests.get(
            f"https://{domain}/robots.txt",
            timeout=10,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        if resp.status_code == 200 and "text/plain" in resp.headers.get("Content-Type", ""):
            rp.parse(resp.text.splitlines())
        elif resp.status_code == 404:
            # No robots.txt → everything allowed
            rp = None
        else:
            # 403, 500, proxy error, HTML page, etc. → can't trust it → assume allowed
            rp = None
    except Exception:
        rp = None
    _robots_cache[domain] = rp
    return rp


def robots_allowed(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc
    rp = get_robots(domain)
    if rp is None:
        return True
    return rp.can_fetch(USER_AGENT, url)


def rate_limit(url: str):
    domain = urlparse(url).netloc
    last = _domain_last_request.get(domain, 0)
    wait = RATE_LIMIT_DELAY - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _domain_last_request[domain] = time.time()


def ensure_linked_pdf_dir() -> Path:
    """Create pdfs_linked/ folder if it doesn't exist."""
    folder = Path(PDF_LINKED_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def linked_pdf_path(parent_resource_id: str, url: str) -> Path:
    """
    Build a deterministic cache path for a linked PDF.
    Format: pdfs_linked/{parent_resource_id}_{url_hash}.pdf

    The parent_resource_id prefix makes it immediately clear which
    resource this PDF was discovered under — critical for provenance.
    The URL hash ensures uniqueness even if two resources link to the
    same PDF (they'll each get their own cached copy, correctly labelled).
    """
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    return Path(PDF_LINKED_DIR) / f"{parent_resource_id}_{url_hash}.pdf"



def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return s


# ═════════════════════════════════════════════════════════════════════════════
# Web Page Fetching + Cleaning  (mirrors Stage 3 logic)
# ═════════════════════════════════════════════════════════════════════════════

NOISE_TAGS = {
    "nav", "header", "footer", "aside", "script", "style",
    "noscript", "form", "iframe", "button", "figure",
}
NOISE_CLASSES_RE = re.compile(
    r"(nav|menu|header|footer|sidebar|cookie|banner|ad[sv]?|popup|modal|"
    r"breadcrumb|social|share|comment|related|pagination|widget)",
    re.IGNORECASE,
)


def _is_noise(tag) -> bool:
    if tag.name in NOISE_TAGS:
        return True
    cls = " ".join(tag.get("class", []))
    _id = tag.get("id", "")
    return bool(NOISE_CLASSES_RE.search(cls) or NOISE_CLASSES_RE.search(_id))


def clean_html(html: str, base_url: str) -> dict:
    """
    Parse and clean a web page, returning a structured dict with the same
    fields as Stage 3 output: title, description, headings, body_text,
    pdf_links, external_links, clean_text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Title ────────────────────────────────────────────────────────────────
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)

    # ── Meta description ─────────────────────────────────────────────────────
    description = ""
    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta:
        description = meta.get("content", "").strip()

    # ── Remove noise ─────────────────────────────────────────────────────────
    # Collect first, then decompose — decomposing during iteration orphans child
    # nodes and causes 'NoneType has no attribute get' on the next loop pass.
    noise = [tag for tag in soup.find_all(True) if _is_noise(tag)]
    for tag in noise:
        tag.decompose()

    # ── Headings ─────────────────────────────────────────────────────────────
    headings = [
        h.get_text(strip=True)
        for h in soup.find_all(["h1", "h2", "h3", "h4"])
        if h.get_text(strip=True)
    ]

    # ── Body text ────────────────────────────────────────────────────────────
    body_parts = []
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    for elem in main.find_all(["p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5"]):
        t = elem.get_text(" ", strip=True)
        if len(t) > 20:
            body_parts.append(t)
    body_text = " ".join(body_parts)

    # ── Links ────────────────────────────────────────────────────────────────
    pdf_links = []
    ext_links = []
    base_domain = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        full_url = urljoin(base_url, href)
        link_text = a.get_text(strip=True) or full_url
        if should_skip_url(full_url):
            continue
        if is_pdf_url(full_url):
            pdf_links.append({"text": link_text, "url": full_url})
        elif urlparse(full_url).netloc != base_domain:
            ext_links.append({"text": link_text, "url": full_url})

    # ── Assemble clean_text ──────────────────────────────────────────────────
    h_str   = " | ".join(headings) if headings else "N/A"
    pdf_str = " | ".join(f"{l['text']} ({l['url']})" for l in pdf_links) if pdf_links else "None"
    ext_str = " | ".join(f"{l['text']} ({l['url']})" for l in ext_links[:20]) if ext_links else "None"

    clean_text = (
        f"TITLE: {title}\n\n"
        f"DESCRIPTION: {description}\n\n"
        f"HEADINGS: {h_str}\n\n"
        f"CONTENT: {body_text}\n\n"
        f"PDF LINKS: {pdf_str}\n\n"
        f"EXTERNAL LINKS: {ext_str}"
    )

    # Inline profiling — computed here while clean_text is in memory
    char_count  = len(clean_text)
    token_count = count_tokens(clean_text)
    route_hint  = "single_pass" if token_count <= TOKEN_SAFE_LIMIT else "map_reduce"

    return {
        "title":          title,
        "description":    description,
        "headings":       headings,
        "body_text":      body_text,
        "pdf_links":      pdf_links,
        "external_links": ext_links,
        "clean_text":     clean_text,
        "word_count":     len(body_text.split()),
        "char_count":     char_count,
        "token_count":    token_count,
        "route_hint":     route_hint,
    }


def fetch_webpage(url: str, session: requests.Session) -> dict:
    """
    Fetch a URL and return a structured result dict.
    """
    result = {
        "url":        url,
        "link_type":  "webpage",
        "status":     "pending",
        "error":      None,
        "extracted":  None,
    }

    if should_skip_url(url):
        result["status"] = "skipped"
        result["error"]  = "URL matched skip pattern"
        return result

    if not robots_allowed(url):
        result["status"] = "blocked"
        result["error"]  = "Disallowed by robots.txt"
        return result

    rate_limit(url)

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT_WEB, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "")

        if resp.status_code != 200:
            result["status"] = "http_error"
            result["error"]  = f"HTTP {resp.status_code}"
            return result

        if "pdf" in content_type.lower() or is_pdf_url(resp.url):
            # Redirect landed on a PDF — hand off to PDF extractor
            result["link_type"] = "pdf"
            result["status"]    = "redirect_to_pdf"
            result["error"]     = "URL redirected to PDF; handle via pdf extractor"
            return result

        extracted = clean_html(resp.text, url)
        result["status"]    = "success"
        result["extracted"] = extracted

    except requests.exceptions.Timeout:
        result["status"] = "timeout"
        result["error"]  = "Request timed out"
    except requests.exceptions.ConnectionError as e:
        result["status"] = "connection_error"
        result["error"]  = str(e)[:120]
    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)[:120]

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PDF Extraction  (mirrors Stage 4 logic)
# ═════════════════════════════════════════════════════════════════════════════

def _heading_detector(text: str) -> bool:
    """True if the line looks like a section heading."""
    t = text.strip()
    if not t or len(t) > 120:
        return False
    # Numbered headings: "1.", "1.2", "A." etc.
    if re.match(r"^(\d+\.|\d+\.\d+|[A-Z]\.|[IVX]+\.)\s+\S", t):
        return True
    # ALL-CAPS headings (min 3 words or ≥10 chars)
    if t.isupper() and (len(t.split()) >= 3 or len(t) >= 10):
        return True
    return False


def extract_pdf_s4(pdf_bytes: bytes, url: str, link_text: str, parent_id: str) -> dict:
    """
    Extract a linked PDF using stage4's full pymupdf-layout pipeline.
    Produces the same structured output (sections, section_headings, clean_text,
    token_count, route_hint, document_profile) as a stage4 direct PDF resource.
    """
    if not HAS_STAGE4 or _s4 is None:
        return {
            "status": "error",
            "error":  "stage4 module could not be loaded — pymupdf-layout unavailable",
            "extracted": None,
            "token_count": 0,
            "route_hint": "single_pass",
        }

    stub = {
        "resource_id": f"{parent_id}_linked",
        "url":         url,
        "title":       link_text or url,
        "author":      "",
        "description": "",
    }

    meta             = _s4.extract_metadata(pdf_bytes)
    layout_segments, total_pages, logical_pages = _s4.extract_layout_with_pymupdf(pdf_bytes)

    if not layout_segments:
        return {
            "status":      "extraction-failed",
            "error":       "No layout segments — possibly scanned/image PDF",
            "token_count": 0,
            "route_hint":  "single_pass",
        }

    toc_entries, toc_pages = _s4.extract_toc_with_pymupdf(pdf_bytes)
    outline_entries        = _s4.extract_outline_with_pymupdf(pdf_bytes)
    sections, body_text    = _s4.extract_sections_from_layout(
        layout_segments,
        toc_entries=toc_entries,
        toc_pages=toc_pages,
        outline_entries=outline_entries,
    )

    if not sections and not body_text:
        return {
            "status":      "extraction-failed",
            "error":       "No usable text structure was extracted",
            "token_count": 0,
            "route_hint":  "single_pass",
        }

    sections  = _s4._strip_redundant_front_matter(sections, stub, meta)
    sections  = _s4._split_trailing_back_matter(sections)
    sections  = _s4._classify_section_roles(sections, toc_pages=toc_pages, total_pages=total_pages)

    body_text = _s4.normalise(" ".join(
        _s4.normalise(" ".join(
            [s.get("content", "")]
            + s.get("callouts", [])
            + [_s4.normalise(" ".join(filter(None, [f.get("title",""), f.get("caption",""), f.get("text","")]))) for f in s.get("figures", [])]
            + s.get("footnotes", [])
        ))
        for s in sections
    ))

    clean_text       = _s4.build_clean_text(stub, meta, sections)
    char_count       = len(clean_text)
    token_count      = count_tokens(clean_text)
    route_hint       = "single_pass" if token_count <= TOKEN_SAFE_LIMIT else "map_reduce"
    section_headings = [s["heading"] for s in sections if s["heading"]]

    document_profile, structure_warnings = _s4._build_document_profile(
        layout_segments, sections, toc_entries, outline_entries, total_pages
    )

    return {
        "status":           "success",
        "extractor_used":   "pymupdf-layout",
        "sections":         sections,
        "section_headings": section_headings,
        "body_text":        body_text,
        "clean_text":       clean_text,
        "word_count":       len(body_text.split()),
        "char_count":       char_count,
        "token_count":      token_count,
        "route_hint":       route_hint,
        "total_pages":      total_pages,
        "pages_extracted":  min(total_pages, _s4.MAX_PAGES),
        "logical_pages":    logical_pages,
        "toc_entries_detected":   len(toc_entries),
        "outline_entries_detected": len(outline_entries),
        "document_profile": document_profile,
        "structure_warnings": structure_warnings,
        "pdf_meta":         meta,
    }


def _extract_with_pdfplumber(pdf_bytes: bytes) -> tuple[list[str], list[str]]:
    sections, content_lines = [], []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        pages = pdf.pages[:MAX_PDF_PAGES]
        for page in pages:
            page_text = page.extract_text() or ""
            for line in page_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if _heading_detector(stripped):
                    sections.append(stripped)
                content_lines.append(stripped)
    return sections, content_lines


def _extract_with_pypdf(pdf_bytes: bytes) -> tuple[list[str], list[str]]:
    sections, content_lines = [], []
    reader = PdfReader(BytesIO(pdf_bytes))
    for page in reader.pages[:MAX_PDF_PAGES]:
        page_text = page.extract_text() or ""
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _heading_detector(stripped):
                sections.append(stripped)
            content_lines.append(stripped)
    return sections, content_lines


def extract_pdf(pdf_bytes: bytes, url: str, link_text: str) -> dict:
    """
    Extract structured text from PDF bytes.
    Returns a dict with sections, content, and clean_text.
    """
    sections, content_lines = [], []
    backend_used = None

    if HAS_PDFPLUMBER:
        try:
            sections, content_lines = _extract_with_pdfplumber(pdf_bytes)
            backend_used = "pdfplumber"
        except Exception as e:
            log.warning(f"pdfplumber failed for {url}: {e}")

    if not content_lines and HAS_PYPDF:
        try:
            sections, content_lines = _extract_with_pypdf(pdf_bytes)
            backend_used = "pypdf (fallback)"
        except Exception as e:
            log.warning(f"pypdf fallback failed for {url}: {e}")

    if not content_lines:
        return {
            "status":       "no_text",
            "error":        "No text extracted — possibly scanned/image PDF",
            "backend":      backend_used,
            "extracted":    None,
            "token_count":  0,
            "route_hint":   "single_pass",
        }

    content_text = "\n".join(content_lines)
    sec_str = "\n".join(f"  • {s}" for s in sections) if sections else "  (none detected)"

    clean_text = (
        f"SOURCE: {link_text}\n"
        f"URL: {url}\n\n"
        f"SECTIONS:\n{sec_str}\n\n"
        f"CONTENT:\n{content_text}"
    )

    # Inline profiling — computed here while clean_text is in memory
    char_count  = len(clean_text)
    token_count = count_tokens(clean_text)
    route_hint  = "single_pass" if token_count <= TOKEN_SAFE_LIMIT else "map_reduce"

    return {
        "status":      "success",
        "backend":     backend_used,
        "sections":    sections,
        "content":     content_text,
        "clean_text":  clean_text,
        "word_count":  len(content_text.split()),
        "char_count":  char_count,
        "token_count": token_count,
        "route_hint":  route_hint,
    }


def fetch_pdf(url: str, link_text: str, session: requests.Session,
              parent_resource_id: str = "unknown") -> dict:
    """
    Download a linked PDF and extract structured text.
    PDFs are cached to pdfs_linked/{parent_resource_id}_{url_hash}.pdf so
    re-runs skip the download entirely. The parent_resource_id prefix in the
    filename makes provenance explicit on disk.
    """
    result = {
        "url":       url,
        "link_text": link_text,
        "link_type": "pdf",
        "status":    "pending",
        "error":     None,
        "extracted": None,
    }

    if not robots_allowed(url):
        result["status"] = "blocked"
        result["error"]  = "Disallowed by robots.txt"
        return result

    rate_limit(url)

    # ── Check disk cache first ────────────────────────────────────────────────
    cache_path = linked_pdf_path(parent_resource_id, url)

    if cache_path.exists():
        log.info(f"    [CACHED] {cache_path.name}")
        pdf_bytes = cache_path.read_bytes()
    else:
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_PDF, allow_redirects=True)
            if resp.status_code != 200:
                result["status"] = "http_error"
                result["error"]  = f"HTTP {resp.status_code}"
                return result
            pdf_bytes = resp.content
            cache_path.write_bytes(pdf_bytes)
            log.info(f"    [SAVED]  {cache_path.name} ({len(pdf_bytes)//1024} KB)")
        except requests.exceptions.Timeout:
            result["status"] = "timeout"
            result["error"]  = "Request timed out"
            return result
        except requests.exceptions.ConnectionError as e:
            result["status"] = "connection_error"
            result["error"]  = str(e)[:120]
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"]  = str(e)[:120]
            return result

    pdf_data = extract_pdf_s4(pdf_bytes, url, link_text, parent_resource_id)
    result["status"]        = pdf_data.get("status", "error")
    result["extracted"]     = pdf_data if pdf_data["status"] == "success" else None
    result["error"]         = pdf_data.get("error")
    result["cached_file"]   = str(cache_path)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Per-resource Link Filter
# ═════════════════════════════════════════════════════════════════════════════

def filter_links(links: list[dict], max_count: int, link_kind: str, resource_id: str) -> list[dict]:
    """
    Filter out skip-pattern URLs and cap at max_count.
    Logs what is kept vs dropped.
    """
    valid = [l for l in links if not should_skip_url(l["url"])]
    skipped_count = len(links) - len(valid)
    if skipped_count:
        log.info(f"  [R{resource_id}] Filtered {skipped_count} {link_kind} skip-pattern URLs")

    if len(valid) > max_count:
        log.info(f"  [R{resource_id}] Capping {link_kind} links: {len(valid)} → {max_count}")
        valid = valid[:max_count]

    return valid


# ═════════════════════════════════════════════════════════════════════════════
# Main Processing
# ═════════════════════════════════════════════════════════════════════════════

def process_resource(resource: dict, session: requests.Session, args) -> dict:
    rid   = resource["resource_id"]
    title = resource["title"]

    log.info(f"── Resource [{rid}] {title[:60]}")

    raw_ext = resource.get("external_links", [])
    raw_pdf = resource.get("pdf_links", [])

    ext_links = filter_links(raw_ext, args.max_ext, "external", rid)
    pdf_links = filter_links(raw_pdf, args.max_pdf, "PDF",      rid)

    # ── Result envelope for this resource ────────────────────────────────────
    record = {
        "resource_id":      rid,
        "resource_title":   title,
        "resource_url":     resource["url"],
        "external_link_results": [],
        "pdf_link_results":      [],
        "summary": {
            "ext_total":    len(raw_ext),
            "ext_processed": len(ext_links),
            "ext_success":  0,
            "ext_failed":   0,
            "pdf_total":    len(raw_pdf),
            "pdf_processed": len(pdf_links),
            "pdf_success":  0,
            "pdf_failed":   0,
        }
    }

    # ── Process external (web) links ──────────────────────────────────────────
    for i, link in enumerate(ext_links, 1):
        log.info(f"  [{i}/{len(ext_links)}] WEB  {link['url'][:80]}")
        if args.dry_run:
            result = {"url": link["url"], "link_type": "webpage", "status": "dry_run",
                      "error": None, "extracted": None}
        else:
            result = fetch_webpage(link["url"], session)

        # Attach provenance
        result["link_text"]         = link.get("text", "")
        result["parent_resource_id"]= rid
        result["parent_title"]      = title

        record["external_link_results"].append(result)

        if result["status"] == "success":
            record["summary"]["ext_success"] += 1
        else:
            record["summary"]["ext_failed"] += 1
            log.warning(f"    ✗ {result['status']}: {result.get('error','')}")

    # ── Process PDF links ─────────────────────────────────────────────────────
    for i, link in enumerate(pdf_links, 1):
        log.info(f"  [{i}/{len(pdf_links)}] PDF  {link['url'][:80]}")
        if args.dry_run:
            result = {"url": link["url"], "link_type": "pdf", "status": "dry_run",
                      "error": None, "extracted": None}
        else:
            result = fetch_pdf(link["url"], link.get("text", ""), session,
                              parent_resource_id=rid)

        # Attach provenance
        result["link_text"]         = link.get("text", "")
        result["parent_resource_id"]= rid
        result["parent_title"]      = title

        record["pdf_link_results"].append(result)

        if result["status"] == "success":
            record["summary"]["pdf_success"] += 1
        else:
            record["summary"]["pdf_failed"] += 1
            log.warning(f"    ✗ {result['status']}: {result.get('error','')}")

    # ── Combined token aggregation ────────────────────────────────────────────
    # Parent token count comes from Stage 3 output (already profiled there).
    # Linked piece token counts are profiled inline above in clean_html / extract_pdf.
    # Summing here gives the true combined token budget for Stage 6 routing.
    parent_tokens = resource.get("token_count", 0)

    linked_tokens = 0
    for r in record["external_link_results"]:
        ext = r.get("extracted") or {}
        linked_tokens += ext.get("token_count", 0)
    for r in record["pdf_link_results"]:
        ext = r.get("extracted") or {}
        linked_tokens += ext.get("token_count", 0)

    combined_token_count = parent_tokens + linked_tokens
    combined_route       = "single_pass" if combined_token_count <= TOKEN_SAFE_LIMIT else "map_reduce"

    record["combined_token_count"] = combined_token_count
    record["combined_route"]       = combined_route
    record["parent_token_count"]   = parent_tokens
    record["linked_token_count"]   = linked_tokens

    s = record["summary"]
    log.info(
        f"  Done  — web: {s['ext_success']}✓ {s['ext_failed']}✗  "
        f"| pdf: {s['pdf_success']}✓ {s['pdf_failed']}✗  "
        f"| tokens: {parent_tokens} parent + {linked_tokens} linked = "
        f"{combined_token_count} total → {combined_route}"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description="Stage 5: one-level deep link extraction")
    parser.add_argument("--input",       default=DEFAULT_INPUT,  help="stage3 JSON file")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT, help="output JSON file")
    parser.add_argument("--max-ext",     type=int, default=MAX_EXT_LINKS_PER_RESOURCE,
                        help="max external links to process per resource")
    parser.add_argument("--max-pdf",     type=int, default=MAX_PDF_LINKS_PER_RESOURCE,
                        help="max PDF links to process per resource")
    parser.add_argument("--resource-id", default=None,
                        help="only process a single resource by ID (for testing)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="show what would be fetched without fetching anything")
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        resources = json.load(f)

    if args.resource_id:
        resources = [r for r in resources if str(r.get("resource_id")) == str(args.resource_id)]
        if not resources:
            log.error(f"No resource with ID {args.resource_id} found")
            sys.exit(1)

    # Only process successfully scraped web pages (they're the ones with links)
    resources = [r for r in resources if r.get("clean_status") == "success"]
    log.info(f"Processing {len(resources)} resources")
    if args.dry_run:
        log.info("DRY RUN mode — no actual HTTP requests will be made")

    # ── Run ───────────────────────────────────────────────────────────────────
    session = make_session()
    ensure_linked_pdf_dir()
    log.info(f"Linked PDF cache: {Path(PDF_LINKED_DIR).resolve()}")
    all_results = []

    for resource in resources:
        record = process_resource(resource, session, args)
        all_results.append(record)

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_ext_ok  = sum(r["summary"]["ext_success"] for r in all_results)
    total_ext_fail= sum(r["summary"]["ext_failed"]  for r in all_results)
    total_pdf_ok  = sum(r["summary"]["pdf_success"] for r in all_results)
    total_pdf_fail= sum(r["summary"]["pdf_failed"]  for r in all_results)

    log.info("═" * 60)
    log.info(f"STAGE 5 COMPLETE")
    log.info(f"  Web pages  : {total_ext_ok} success  {total_ext_fail} failed")
    log.info(f"  PDFs       : {total_pdf_ok} success  {total_pdf_fail} failed")
    log.info(f"  Output     : {output_path}")
    log.info("═" * 60)


if __name__ == "__main__":
    main()