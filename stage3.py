"""
EDI Hub+ Pipeline — Stage 3: Fetch and Clean Text
==================================================
Reads stage2_resources.json, re-fetches the 'success' URLs,
strips boilerplate, and extracts structured clean text.

Now includes inline content profiling: char_count and token_count
are computed at extraction time using OLMo's own GPT-NeoX tokeniser,
so the profiling block downstream only needs to aggregate, not re-tokenise.

Usage:
    # Test on one resource (by resource_id)
    python stage3.py --test 46

    # Full batch (all successful resources)
    python stage3.py

    # Full batch with custom input/output
    python stage3.py --input stage2_resources.json --output stage3_resources.json
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGE2_INPUT_FILE  = "stage2_resources.json"
STAGE3_OUTPUT_FILE = "stage3_resources.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Tags that are pure boilerplate — strip entirely before extraction
NOISE_TAGS = [
    "script", "style", "noscript", "iframe",
    "nav", "footer", "header", "aside",
    "form", "button", "svg", "figure",
    "cookie-banner", "cookie-notice",
]

# CSS class / id fragments that indicate boilerplate
NOISE_PATTERNS = [
    "cookie", "banner", "popup", "modal",
    "sidebar", "breadcrumb", "share",
    "social", "newsletter", "subscribe",
    "advertisement", "advert", "skip-link",
    "site-header", "site-footer", "top-nav",
    "related-links", "related-content",
    "in-this-section", "in_this_section",
    "section-nav", "local-nav", "local-links",
    "page-nav", "secondary-nav", "sub-nav",
    "utility-links", "share-links", "share-page",
    "social-share", "addthis", "sharethis",
    "pagination", "pager", "tag-list",
    "was-this-page", "feedback-form",
    "follow-us", "footer-links",
]

DELAY_BETWEEN_REQUESTS = 1.5   # seconds — be polite

# OLMo context window and prompt overhead — matches the routing threshold
OLMO_CONTEXT_WINDOW = 4096
PROMPT_OVERHEAD     = 300
TOKEN_SAFE_LIMIT    = OLMO_CONTEXT_WINDOW - PROMPT_OVERHEAD   # 3,796


# ---------------------------------------------------------------------------
# Tokeniser (lazy-loaded, cached at module level)
# ---------------------------------------------------------------------------

_tokenizer = None

def get_tokenizer():
    """
    Load OLMo's GPT-NeoX tokeniser once and cache it.
    Uses the exact same tokeniser OLMo 7B uses internally,
    so token counts are precise rather than estimated.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print("  [TOKENISER] Loading OLMo tokeniser (first call only)...")
        _tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-7B-1124")
        print("  [TOKENISER] Ready.\n")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Return exact OLMo token count for a string. Returns 0 for empty input."""
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_html(url: str, timeout: int = 15) -> str | None:
    """Fetch raw HTML for a URL. Returns None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        print(f"    [WARN] HTTP {resp.status_code} for {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] {e}")
        return None


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def is_noise_element(tag) -> bool:
    """Return True if this BS4 element looks like boilerplate."""
    if not hasattr(tag, "attrs") or tag.attrs is None:
        return False
    cls_str  = " ".join(tag.attrs.get("class", [])).lower()
    id_str   = (tag.attrs.get("id") or "").lower()
    combined = cls_str + " " + id_str
    return any(pat in combined for pat in NOISE_PATTERNS)


def strip_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove noise tags and boilerplate elements in-place."""
    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove noise tags wholesale
    for tag_name in NOISE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Remove elements whose class/id looks like boilerplate
    for el in soup.find_all(True):
        if not hasattr(el, "attrs"):
            continue
        if is_noise_element(el):
            el.decompose()

    # Kill "In this section" sidebar nav blocks — only small elements (< 300 chars)
    for el in soup.find_all(["ul", "nav", "section"]):
        text = el.get_text(strip=True)
        if text.lower().startswith("in this section") and len(text) < 300:
            el.decompose()

    # Kill social share blocks — small blocks (< 80 words) with 3+ social keywords
    social_words = {"facebook", "twitter", "whatsapp", "messenger", "linkedin", "share"}
    for el in soup.find_all(["div", "ul", "section"]):
        text = el.get_text(separator=" ", strip=True)
        if len(text.split()) < 80:
            words = set(text.lower().split())
            if len(words & social_words) >= 3:
                el.decompose()

    return soup


def extract_meta_description(soup: BeautifulSoup) -> str:
    """Pull <meta name='description'> or og:description."""
    for attrs in [
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"},
    ]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def extract_headings(soup: BeautifulSoup) -> list[str]:
    """Extract h1–h3 text as a deduplicated list."""
    seen = set()
    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        text = h.get_text(separator=" ", strip=True)
        text = normalise(text)
        word_count = len(text.split())
        if text and len(text) > 3 and word_count <= 12 and text not in seen:
            seen.add(text)
            headings.append(text)
    return headings


def extract_links(soup: BeautifulSoup, base_url: str) -> dict:
    """
    Extract meaningful hyperlinks from the page, split into:
      - pdf_links:      links ending in .pdf
      - external_links: links to other domains
      - internal_links: links within the same domain (nav-style, skipped)
    Returns a dict with pdf_links and external_links as lists of
    {"text": ..., "url": ...} dicts.
    """
    from urllib.parse import urlparse, urljoin

    base_domain = urlparse(base_url).netloc
    pdf_links      = []
    external_links = []
    internal_links = []
    seen_urls      = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue

        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        anchor_text = normalise(a.get_text(separator=" ", strip=True))
        if not anchor_text or len(anchor_text) < 3:
            continue

        link_domain = urlparse(full_url).netloc

        # Classify
        if full_url.lower().endswith(".pdf") or ".pdf" in full_url.lower():
            pdf_links.append({"text": anchor_text, "url": full_url})
        elif link_domain and link_domain != base_domain:
            external_links.append({"text": anchor_text, "url": full_url})
        elif link_domain and link_domain == base_domain:
            # Only keep links that are genuine sub-pages of the current URL
            # i.e. their path starts with the parent page's path
            from urllib.parse import urlparse
            base_path = urlparse(base_url).path.rstrip("/")
            link_path = urlparse(full_url).path.rstrip("/")
            if link_path.startswith(base_path + "/") and link_path != base_path:
                internal_links.append({"text": anchor_text, "url": full_url})

    return {"pdf_links": pdf_links, "external_links": external_links, "internal_links": internal_links}


def find_main_container(soup: BeautifulSoup):
    """
    Locate the main content container element using a priority chain:
      1. <main>
      2. <article>
      3. <div id='content'> or similar
      4. Largest <div> by character count (heuristic fallback)
      5. <body> absolute fallback
    Returns the BS4 element itself (not text), so callers can walk its children.
    """
    # Priority 1 & 2: semantic tags
    for tag_name in ["main", "article"]:
        el = soup.find(tag_name)
        if el:
            return el

    # Priority 3: divs with content-ish IDs / classes
    content_hints = [
        "content", "main-content", "page-content",
        "entry-content", "post-content", "body-content",
    ]
    for hint in content_hints:
        el = soup.find("div", id=hint) or soup.find("div", class_=hint)
        if el and len(el.get_text(strip=True)) > 200:
            return el

    # Priority 4: largest <div> heuristic
    best_div, best_len = None, 0
    for div in soup.find_all("div"):
        length = len(div.get_text(strip=True))
        if length > best_len:
            best_len = length
            best_div = div
    if best_div and best_len > 200:
        return best_div

    # Absolute fallback
    return soup.find("body")


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """
    Walk the main content container and split content into sections,
    each anchored to the heading that precedes it.

    Each section dict contains:
        heading    : str  — heading text, or "" for content before first heading
        level      : str  — "h1" / "h2" / "h3", or "preamble" if no heading
        content    : str  — normalised body text of the section
        word_count : int
        char_count : int
        token_count: int  — counted with OLMo GPT-NeoX tokeniser

    Content before the first heading is captured as a preamble section.
    Sections whose content is empty after normalisation are dropped.
    """
    HEADING_TAGS = {"h1", "h2", "h3"}
    CONTENT_TAGS = {"p", "li", "td", "th", "dd", "dt", "blockquote", "figcaption"}

    container = find_main_container(soup)
    if container is None:
        return []

    # Walk direct and nested descendants, collecting (heading, level, [text_chunks])
    sections_raw = []
    current_heading = ""
    current_level   = "preamble"
    current_chunks  = []

    def flush():
        """Commit the current buffer as a section."""
        text = normalise(" ".join(current_chunks))
        if text:
            sections_raw.append({
                "heading": current_heading,
                "level":   current_level,
                "_text":   text,
            })

    for el in container.descendants:
        if not hasattr(el, "name") or el.name is None:
            # It's a NavigableString — skip, we collect via tag-level get_text
            continue

        if el.name in HEADING_TAGS:
            text = normalise(el.get_text(separator=" ", strip=True))
            # Only treat as a heading if it looks like one (not too long)
            if text and len(text.split()) <= 15:
                flush()
                current_heading = text
                current_level   = el.name
                current_chunks  = []

        elif el.name in CONTENT_TAGS:
            # Avoid double-counting nested tags (e.g. <li> inside <ul> inside <p>)
            # Only grab text if this element has no CONTENT_TAG ancestors already visited
            parent_names = {p.name for p in el.parents if hasattr(p, "name")}
            if not (CONTENT_TAGS - {el.name}) & parent_names:
                text = normalise(el.get_text(separator=" ", strip=True))
                if text:
                    current_chunks.append(text)

    flush()  # Commit the last section

    # Attach per-section metadata
    sections = []
    for raw in sections_raw:
        content    = raw["_text"]
        word_count = len(content.split())
        char_count = len(content)
        tok_count  = count_tokens(content)
        sections.append({
            "heading":     raw["heading"],
            "level":       raw["level"],
            "content":     content,
            "word_count":  word_count,
            "char_count":  char_count,
            "token_count": tok_count,
        })

    return sections


def normalise(text: str) -> str:
    """
    Normalise unicode, collapse whitespace, strip control chars.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_resource(resource: dict) -> dict:
    """
    Fetch and clean a single resource dict.
    Adds: clean_title, description, headings, body_text,
          clean_text, word_count, char_count, token_count, clean_status.

    token_count uses OLMo's exact GPT-NeoX tokeniser so downstream
    profiling only needs to aggregate — no re-tokenisation required.
    """
    url   = resource["url"]
    title = resource.get("title", "")
    print(f"  Fetching [{resource['resource_id']}] {title[:60]}...")

    html = fetch_html(url)

    if not html:
        return {
            **resource,
            "clean_status": "fetch-failed",
            "clean_title":  "",
            "description":  "",
            "headings":     [],
            "body_text":    "",
            "clean_text":   None,
            "word_count":   0,
            "char_count":   0,
            "token_count":  0,
            "route_hint":   "single_pass",
        }

    soup = BeautifulSoup(html, "html.parser")

    # --- Extract BEFORE stripping (meta tags live in <head>) ---
    raw_title = soup.title.string.strip() if soup.title else title
    import re as _re
    raw_title = _re.sub(r"^[\s»›«|·\-–—:]+|[\s»›«|·\-–—:]+$", "", raw_title).strip()
    description = extract_meta_description(soup)

    # --- Extract links BEFORE stripping (links live in body, may get removed) ---
    links = extract_links(soup, url)

    # --- Strip boilerplate ---
    soup = strip_boilerplate(soup)

    # --- Extract structure ---
    content_sections = extract_sections(soup)
    headings = [s["heading"] for s in content_sections if s["heading"]]

    # Derive flat body_text from sections (for word count + backward compat)
    body_text = normalise(
        " ".join(s["content"] for s in content_sections)
    )

    clean_title = normalise(raw_title)
    description = normalise(description)

    # --- Build structured clean_text for the LLM ---
    # Header block
    header_parts = []
    if clean_title:
        header_parts.append(f"TITLE: {clean_title}")
    if description:
        header_parts.append(f"DESCRIPTION: {description}")

    # Section blocks — each rendered as "## Heading\nContent"
    section_parts = []
    for s in content_sections:
        if s["heading"]:
            section_parts.append(f"## {s['heading']}\n{s['content']}")
        else:
            section_parts.append(s["content"])  # preamble, no heading prefix

    # Link block
    link_parts = []
    if links["pdf_links"]:
        pdf_strs = [f"{l['text']} ({l['url']})" for l in links["pdf_links"]]
        link_parts.append("PDF LINKS: " + " | ".join(pdf_strs))
    if links["external_links"]:
        ext_strs = [f"{l['text']} ({l['url']})" for l in links["external_links"][:10]]
        link_parts.append("EXTERNAL LINKS: " + " | ".join(ext_strs))

    clean_text = "\n\n".join(
        filter(None, [
            "\n".join(header_parts),
            "\n\n".join(section_parts),
            "\n".join(link_parts),
        ])
    )

    # --- Inline content profiling ---
    char_count  = len(clean_text)
    token_count = count_tokens(clean_text)
    route_hint  = "single_pass" if token_count <= TOKEN_SAFE_LIMIT else "map_reduce"

    return {
        **resource,
        "clean_status":    "success",
        "clean_title":     clean_title,
        "description":     description,
        "headings":        headings,
        "sections":        content_sections,   # structured: heading + content + metadata
        "body_text":       body_text,           # flat join for backward compat
        "clean_text":      clean_text,
        "word_count":      len(body_text.split()),
        "char_count":      char_count,
        "token_count":     token_count,
        "route_hint":      route_hint,
        "pdf_links":        links["pdf_links"],
        "external_links":   links["external_links"],
        "internal_links":   links["internal_links"],
    }


# ---------------------------------------------------------------------------
# Depth probe
# ---------------------------------------------------------------------------

def probe_depth(resources: list[dict]) -> None:
    """
    For each successful resource, fetch the page and count internal links.
    Classifies each site as single-page, shallow, or deep.
    Does NOT write any output — read-only inspection.
    """
    from urllib.parse import urlparse, urljoin

    print(f"\nProbing {len(resources)} page(s) for internal link depth...\n")
    print(f"{'ID':<6} {'Links':>6} {'Depth':>10}  {'Title':<45} URL")
    print("-" * 120)

    for resource in resources:
        url   = resource["url"]
        rid   = resource["resource_id"]
        title = resource.get("title", "")[:44]

        html = fetch_html(url)
        if not html:
            print(f"{rid:<6} {'ERROR':>6} {'':>10}  {title:<45} {url}")
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(url).netloc

        # Collect unique internal links
        internal_urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("mailto:"):
                continue
            full = urljoin(url, href)
            link_domain = urlparse(full).netloc
            if link_domain == base_domain and full.rstrip("/") != url.rstrip("/"):
                internal_urls.add(full.rstrip("/"))

        count = len(internal_urls)
        if count == 0:
            depth_label = "single-page"
        elif count <= 5:
            depth_label = "shallow"
        elif count <= 20:
            depth_label = "medium"
        else:
            depth_label = "deep"

        print(f"{rid:<6} {count:>6} {depth_label:>10}  {title:<45} {url[:60]}")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("\n" + "-" * 120)
    print("Depth guide: single-page=0 links | shallow=1-5 | medium=6-20 | deep=20+")
    print("Suggest crawling: medium and deep sites only.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EDI Hub+ Pipeline — Stage 3: Fetch and Clean Text"
    )
    parser.add_argument("--input",  default=STAGE2_INPUT_FILE)
    parser.add_argument("--output", default=STAGE3_OUTPUT_FILE)
    parser.add_argument(
        "--test",
        metavar="RESOURCE_ID",
        help="Run on a single resource ID only (e.g. --test 46)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Inspect internal link depth for all successful pages (read-only)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help=(
            "Save each resource as its own JSON file instead of one combined output. "
            "Files are written to a directory named after --output (without extension), "
            "e.g. 'stage3_resources/' with files like 'resource_46.json'."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EDI Hub+ Pipeline — Stage 3: Fetch and Clean Text")
    print("=" * 60)

    with open(args.input, encoding="utf-8") as f:
        all_resources = json.load(f)

    # Filter to only successfully fetched web pages
    targets = [r for r in all_resources if r.get("status") == "success"]

    if args.probe:
        probe_depth(targets)
        return

    if args.test:
        targets = [r for r in targets if r["resource_id"] == args.test]
        if not targets:
            print(f"[ERROR] No successful resource found with id '{args.test}'")
            return

    print(f"Processing {len(targets)} resource(s)...\n")

    results = []
    for i, resource in enumerate(targets):
        result = clean_resource(resource)
        results.append(result)

        status = result["clean_status"]
        wc     = result.get("word_count", 0)
        secs   = len(result.get("sections", []))
        tc     = result.get("token_count", 0)
        route  = result.get("route_hint", "-")
        print(f"    → {status} | {wc} words | {secs} sections | "
              f"{tc} tokens | route: {route}\n")

        if i < len(targets) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Save
    ok   = sum(1 for r in results if r["clean_status"] == "success")
    fail = sum(1 for r in results if r["clean_status"] == "fetch-failed")

    if args.separate:
        output_dir = Path(args.output).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            rid = result["resource_id"]
            file_path = output_dir / f"resource_{rid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    Saved → {file_path}")
        print(f"\nDone. {ok} cleaned, {fail} failed → {output_dir}/ ({len(results)} files)")
    else:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Done. {ok} cleaned, {fail} failed → {output_path}")

    if args.test and results:
        r = results[0]
        sep = "=" * 60
        print(f"\n{sep}")
        print("FULL CLEAN OUTPUT")
        print(sep)
        print(f"\nTITLE:\n  {r['clean_title']}")
        print(f"\nDESCRIPTION:\n  {r['description']}")
        print(f"\nSECTIONS ({len(r['sections'])} found):")
        for i, s in enumerate(r['sections']):
            heading_label = f"[{s['level']}] {s['heading']}" if s['heading'] else "[preamble]"
            print(f"  {i+1:>2}. {heading_label}")
            print(f"       words: {s['word_count']}  |  chars: {s['char_count']}  |  tokens: {s['token_count']}")
            print(f"       {s['content'][:120]}{'...' if len(s['content']) > 120 else ''}")
        print(f"\nTOTAL WORD COUNT:  {r['word_count']}")
        print(f"TOTAL CHAR COUNT:  {r['char_count']}")
        print(f"TOTAL TOKEN COUNT: {r['token_count']}  (OLMo GPT-NeoX tokeniser)")
        print(f"ROUTE HINT:        {r['route_hint']}  (threshold: {TOKEN_SAFE_LIMIT} tokens)")
        print(f"\n{sep}")
        print("FULL CLEAN_TEXT (as the LLM will see it):")
        print(sep)
        print(r['clean_text'])
        print(sep)
        print(f"\nPDF LINKS ({len(r.get('pdf_links',[]))} found):")
        for l in r.get("pdf_links", []):
            print(f"  [PDF] {l['text'][:60]} → {l['url'][:80]}")
        print(f"\nEXTERNAL LINKS ({len(r.get('external_links',[]))} found):")
        for l in r.get("external_links", [])[:8]:
            print(f"  [EXT] {l['text'][:60]} → {l['url'][:80]}")
        print(sep)


if __name__ == "__main__":
    main()