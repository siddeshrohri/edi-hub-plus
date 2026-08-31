"""
EDI Hub+ Pipeline — Stage 4: PDF Text Extraction
=================================================
Reads stage2_resources.json, downloads PDF resources to a local
pdfs/ folder (skipping already-downloaded files), extracts and
cleans text, and outputs stage4_resources.json.

Primary extraction is layout-aware: PyMuPDF retains coordinates/font metadata,
likely two-page spreads are split at the central gutter, and headings are
identified from an adaptive evidence stack: PDF outlines/bookmarks, visible TOC, explicit numbering, typography, whitespace and geometry.

Now includes inline content profiling: page_count, char_count and
token_count are computed at extraction time using OLMo's own GPT-NeoX
tokeniser, so the profiling block downstream only needs to aggregate,
not re-tokenise.

Usage:
    python stage4.py --test 39     # test one resource
    python stage4.py               # full batch
    python stage4.py --probe       # check page counts only
    python stage4.py --allow-flat-fallback  # explicit legacy fallback if needed
"""

import argparse
import io
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

try:
    import pymupdf as fitz  # preferred modern import name
except ImportError:
    try:
        import fitz  # compatibility with older PyMuPDF installs
    except ImportError:
        fitz = None

import pdfplumber
import requests
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGE2_INPUT_FILE  = "stage2_resources.json"
STAGE4_OUTPUT_FILE = "stage4_resources.json"
PDF_DIR            = "pdfs"          # local folder for cached PDFs
MAX_PAGES          = 100              # max pages to extract per PDF
DELAY              = 2.0             # seconds between downloads
STAGE4_BUILD       = "adaptive-layout-v6-2026-08-13"

# OLMo context window and prompt overhead — matches the routing threshold
OLMO_CONTEXT_WINDOW = 4096
PROMPT_OVERHEAD     = 300
TOKEN_SAFE_LIMIT    = OLMO_CONTEXT_WINDOW - PROMPT_OVERHEAD   # 3,796

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


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
# Local PDF folder
# ---------------------------------------------------------------------------

def ensure_pdf_dir() -> Path:
    """Create pdfs/ folder if it doesn't exist."""
    folder = Path(PDF_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def local_pdf_path(resource_id: str) -> Path:
    return Path(PDF_DIR) / f"{resource_id}.pdf"


# ---------------------------------------------------------------------------
# Download (with local cache)
# ---------------------------------------------------------------------------

def fetch_pdf(resource_id: str, url: str, timeout: int = 60) -> bytes | None:
    """
    Return PDF bytes for a resource.
    - If pdfs/{resource_id}.pdf already exists, reads from disk.
    - Otherwise downloads from URL, saves to disk, then returns bytes.
    """
    path = local_pdf_path(resource_id)

    if path.exists():
        print(f"    [CACHED]   {path}")
        return path.read_bytes()

    print(f"    [DOWNLOAD] {url[:75]}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            path.write_bytes(resp.content)
            print(f"    [SAVED]    {path} ({len(resp.content)//1024} KB)")
            return resp.content
        print(f"    [WARN]     HTTP {resp.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR]    {e}")
        return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_with_pdfplumber(pdf_bytes: bytes) -> tuple[list[str], int]:
    pages, total = [], 0
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)
            for page in pdf.pages[:MAX_PAGES]:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append(text.strip())
    except Exception as e:
        print(f"    [WARN] pdfplumber: {e}")
    return pages, total


def extract_with_pypdf(pdf_bytes: bytes) -> tuple[list[str], int]:
    pages, total = [], 0
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total  = len(reader.pages)
        for page in reader.pages[:MAX_PAGES]:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    except Exception as e:
        print(f"    [WARN] pypdf: {e}")
    return pages, total


# ---------------------------------------------------------------------------
# Layout-aware extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def _is_bold_font(font_name: str) -> bool:
    """
    Return True for visually emphasised font faces.

    PDF font names are not standardised. Besides names containing words such as
    "Bold" or "Semibold", Adobe/InDesign exports commonly use abbreviated weight
    suffixes such as 65Md (Medium), 75Bd (Bold), and 95Blk (Black). Treat these
    as emphasis while leaving regular faces such as 55Rg as body text.
    """
    name = (font_name or "").strip().lower()
    if any(token in name for token in (
        "bold", "semibold", "demibold", "black", "heavy", "medium"
    )):
        return True

    # Common abbreviated weight names, especially Neue Haas / Helvetica-style
    # families exported by Adobe InDesign.
    if re.search(r"(?:^|[-_])(?:\d{2})?(?:md|bd|blk)(?:$|[-_])", name):
        return True
    return False


def _join_pdf_lines(lines: list[str]) -> str:
    """Join visual PDF lines while repairing ordinary end-of-line hyphenation."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        # Only dehyphenate when a line ends with a letter-hyphen and the next
        # line starts lower-case. This avoids destroying genuine compounds.
        if re.search(r"[A-Za-z]-$", out) and re.match(r"^[a-z]", line):
            out = out[:-1] + line
        else:
            out += " " + line
    return normalise(out)


def _line_style(line: dict) -> tuple[float, bool]:
    """Return character-weighted font size and dominant boldness for a line."""
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return 0.0, False
    total_chars = sum(max(1, len(s.get("text", "").strip())) for s in spans)
    size = sum(float(s.get("size", 0.0)) * max(1, len(s.get("text", "").strip())) for s in spans) / total_chars
    bold_chars = sum(
        max(1, len(s.get("text", "").strip()))
        for s in spans if _is_bold_font(s.get("font", ""))
    )
    return round(size, 1), (bold_chars / total_chars) >= 0.55


def _logical_page_clips(page) -> list:
    """
    Detect likely two-page spreads / strongly separated two-panel layouts.

    We split only when:
      - the page is landscape-ish,
      - substantial text exists on both sides of the centre, and
      - very little text crosses the central gutter.

    This deliberately uses geometry rather than document-specific words.
    """
    w, h = float(page.rect.width), float(page.rect.height)
    full = [fitz.Rect(0, 0, w, h)]
    if not w or not h or (w / h) < 1.20:
        return full

    words = page.get_text("words", sort=False)
    if not words:
        return full

    # Ignore tiny footer/page-number material when deciding whether the page
    # contains two meaningful text regions.
    useful = []
    for wd in words:
        x0, y0, x1, y1, text = wd[:5]
        t = str(text).strip()
        if not t:
            continue
        if y0 > h * 0.94 and re.fullmatch(r"\d{1,4}", t):
            continue
        useful.append((float(x0), float(y0), float(x1), float(y1), t))

    if len(useful) < 20:
        return full

    mid = w / 2.0
    gutter = max(12.0, w * 0.018)
    left = [wd for wd in useful if wd[2] <= mid - gutter]
    right = [wd for wd in useful if wd[0] >= mid + gutter]
    crossing = [wd for wd in useful if wd[0] < mid + gutter and wd[2] > mid - gutter]

    # Require meaningful text on each side; otherwise it is more likely a
    # normal landscape page with an image or whitespace on one side.
    left_chars = sum(len(wd[4]) for wd in left)
    right_chars = sum(len(wd[4]) for wd in right)
    cross_chars = sum(len(wd[4]) for wd in crossing)
    total_chars = max(1, left_chars + right_chars + cross_chars)

    cross_ratio = cross_chars / total_chars
    if left_chars >= 120 and right_chars >= 120 and cross_ratio <= 0.035:
        return [fitz.Rect(0, 0, mid, h), fitz.Rect(mid, 0, w, h)]

    # A wide spread can legitimately have text on one logical page and a full-
    # bleed image on the other. If the centre is completely clear of text,
    # splitting is still safe and prevents the populated half being treated as
    # an over-wide single page for downstream column detection.
    if (w / h) >= 1.30 and max(left_chars, right_chars) >= 180 and cross_ratio <= 0.005:
        return [fitz.Rect(0, 0, mid, h), fitz.Rect(mid, 0, w, h)]

    return full


def _order_segments_in_clip(segments: list[dict], clip) -> list[dict]:
    """
    Reconstruct reading order for one logical page.

    Key rule: full-width text is a horizontal anchor, not something to append
    after inferred columns. This handles pages that transition between
    full-width prose and multi-column/table/callout regions.
    """
    key = lambda s: (float(s["bbox"][1]), float(s["bbox"][0]))
    if len(segments) < 4:
        return sorted(segments, key=key)

    width = float(clip.width)
    height = float(clip.height)
    x_origin = float(clip.x0)

    # Strong display headings in the top page band belong before any columns,
    # even when their bbox centre happens to fall to one side of the inferred
    # split.
    top_anchors = [
        seg for seg in segments
        if float(seg["bbox"][1]) <= float(clip.y0) + height * 0.14
        and float(seg.get("font_size", 0.0)) >= 18.0
    ]
    top_ids = {id(seg) for seg in top_anchors}
    working = [seg for seg in segments if id(seg) not in top_ids]

    # Infer possible column starts only from substantial, non-display text.
    candidates = []
    for seg in working:
        text = seg.get("text", "")
        x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
        seg_width = x1 - x0
        if len(text) >= 45 and seg_width <= width * 0.62 and float(seg.get("font_size", 0)) <= 16.0:
            candidates.append(seg)

    if len(candidates) < 4:
        return sorted(top_anchors, key=key) + sorted(working, key=key)

    raw = sorted((float(seg["bbox"][0]) - x_origin, len(seg["text"])) for seg in candidates)
    clusters: list[list[float]] = []
    cluster_tol = width * 0.07
    for x, weight in raw:
        if not clusters or abs(x - clusters[-1][0]) > cluster_tol:
            clusters.append([x, float(weight)])
        else:
            old_x, old_w = clusters[-1]
            new_w = old_w + weight
            clusters[-1] = [(old_x * old_w + x * weight) / new_w, new_w]

    if len(clusters) < 2:
        return sorted(top_anchors, key=key) + sorted(working, key=key)

    dominant = sorted(clusters, key=lambda c: c[1], reverse=True)[:2]
    dominant.sort(key=lambda c: c[0])
    if (dominant[1][0] - dominant[0][0]) < width * 0.25:
        return sorted(top_anchors, key=key) + sorted(working, key=key)

    split_local = (dominant[0][0] + dominant[1][0]) / 2.0
    split_x = x_origin + split_local

    left_chars = sum(len(seg["text"]) for seg in candidates if (float(seg["bbox"][0]) - x_origin) < split_local)
    right_chars = sum(len(seg["text"]) for seg in candidates if (float(seg["bbox"][0]) - x_origin) >= split_local)
    if min(left_chars, right_chars) < 140:
        return sorted(top_anchors, key=key) + sorted(working, key=key)

    cross: list[dict] = []
    non_cross: list[dict] = []
    for seg in working:
        x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
        # A true horizontal anchor spans the inferred split substantially.
        if x0 < split_x < x1 and (x1 - x0) >= width * 0.68:
            cross.append(seg)
        else:
            non_cross.append(seg)

    # If no full-width anchors exist, classic column order is appropriate.
    if not cross:
        left, right = [], []
        for seg in non_cross:
            centre = (float(seg["bbox"][0]) + float(seg["bbox"][2])) / 2.0
            (left if centre < split_x else right).append(seg)
        return sorted(top_anchors, key=key) + sorted(left, key=key) + sorted(right, key=key)

    cross = sorted(cross, key=key)

    def order_band(items: list[dict], y_start: float, y_end: float) -> list[dict]:
        if not items:
            return []

        # Short mixed bands and table-like bands are best read row-wise.
        band_height = max(0.0, y_end - y_start)
        row_pairs = 0
        sorted_items = sorted(items, key=key)
        for a_idx in range(len(sorted_items)):
            ay = float(sorted_items[a_idx]["bbox"][1])
            ax = (float(sorted_items[a_idx]["bbox"][0]) + float(sorted_items[a_idx]["bbox"][2])) / 2.0
            for b_idx in range(a_idx + 1, min(len(sorted_items), a_idx + 8)):
                by = float(sorted_items[b_idx]["bbox"][1])
                if by - ay > 5.0:
                    break
                bx = (float(sorted_items[b_idx]["bbox"][0]) + float(sorted_items[b_idx]["bbox"][2])) / 2.0
                if abs(ax - bx) >= width * 0.20:
                    row_pairs += 1
                    break

        if band_height <= height * 0.28 or row_pairs >= 3:
            return sorted_items

        left, right = [], []
        for seg in items:
            centre = (float(seg["bbox"][0]) + float(seg["bbox"][2])) / 2.0
            (left if centre < split_x else right).append(seg)

        if not left or not right:
            return sorted_items
        return sorted(left, key=key) + sorted(right, key=key)

    ordered: list[dict] = []
    used: set[int] = set()
    cursor_y = float(clip.y0)

    for anchor in cross:
        ay0, ay1 = float(anchor["bbox"][1]), float(anchor["bbox"][3])

        band = [
            seg for seg in non_cross
            if id(seg) not in used
            and float(seg["bbox"][1]) >= cursor_y - 1.0
            and float(seg["bbox"][1]) < ay0 - 1.0
        ]
        ordered.extend(order_band(band, cursor_y, ay0))
        used.update(id(seg) for seg in band)

        ordered.append(anchor)
        cursor_y = max(cursor_y, ay1)

    tail = [
        seg for seg in non_cross
        if id(seg) not in used and float(seg["bbox"][1]) >= cursor_y - 1.0
    ]
    ordered.extend(order_band(tail, cursor_y, float(clip.y1)))
    used.update(id(seg) for seg in tail)

    # Safety: retain any unusual overlapping items not captured by bands.
    leftovers = [seg for seg in non_cross if id(seg) not in used]
    ordered.extend(sorted(leftovers, key=key))
    return sorted(top_anchors, key=key) + ordered

def _segments_from_clip(page, clip, physical_page: int, logical_page: int) -> list[dict]:
    """
    Extract style-aware text segments from one logical page region.

    PyMuPDF often stores a heading and the following body paragraph in the
    same text block. We therefore split blocks again when font size/boldness
    changes, preserving multi-line headings such as:

        5. Vague language and hyperbolic
        language operate as barriers to
        inclusiveness
    """
    data = page.get_text("dict", clip=clip, sort=False)
    segments: list[dict] = []

    for block in data.get("blocks", []):
        lines = block.get("lines")
        if not lines:
            continue

        current_lines: list[str] = []
        current_size: float | None = None
        current_bold: bool | None = None
        current_bbox: list[float] | None = None
        current_line_records: list[dict] = []

        def flush_run():
            nonlocal current_lines, current_size, current_bold, current_bbox, current_line_records
            text = _join_pdf_lines(current_lines)
            if text and current_bbox:
                segments.append({
                    "text": text,
                    "font_size": float(current_size or 0.0),
                    "bold": bool(current_bold),
                    "bbox": tuple(current_bbox),
                    "physical_page": physical_page,
                    "logical_page": logical_page,
                    "logical_width": float(clip.width),
                    "logical_height": float(clip.height),
                    "_lines": list(current_line_records),
                })
            current_lines = []
            current_size = None
            current_bold = None
            current_bbox = None
            current_line_records = []

        for line in lines:
            line_text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not line_text:
                continue
            size, bold = _line_style(line)
            bbox = [float(v) for v in line.get("bbox", (0, 0, 0, 0))]

            # Font-size is the primary signal; boldness is secondary. Keeping
            # italic/non-italic body text together avoids excessive fragments.
            same_run = (
                current_size is not None
                and abs(size - current_size) <= 0.45
                and bold == current_bold
            )
            if not same_run:
                flush_run()
                current_size, current_bold = size, bold
                current_bbox = bbox
            else:
                current_bbox = [
                    min(current_bbox[0], bbox[0]), min(current_bbox[1], bbox[1]),
                    max(current_bbox[2], bbox[2]), max(current_bbox[3], bbox[3]),
                ]
            current_lines.append(line_text)
            current_line_records.append({
                "text": line_text,
                "bbox": tuple(bbox),
                "font_size": float(size),
                "bold": bool(bold),
            })

        flush_run()

    return _order_segments_in_clip(segments, clip)


def extract_layout_with_pymupdf(pdf_bytes: bytes) -> tuple[list[dict], int, int]:
    """
    Return typography/layout-aware segments rather than flattened page text.

    Output tuple:
      segments, physical_page_count, logical_page_count
    """
    if fitz is None:
        return [], 0, 0

    all_segments: list[dict] = []
    total_pages = 0
    logical_page_no = 0
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        for physical_idx, page in enumerate(doc[:MAX_PAGES], start=1):
            clips = _logical_page_clips(page)
            for clip in clips:
                logical_page_no += 1
                all_segments.extend(
                    _segments_from_clip(page, clip, physical_idx, logical_page_no)
                )
        doc.close()
    except Exception as e:
        print(f"    [WARN] PyMuPDF layout extraction: {e}")
        return [], total_pages, logical_page_no

    return all_segments, total_pages, logical_page_no



def _heading_key(text: str) -> str:
    """Punctuation-insensitive canonical key for matching the same heading."""
    s = normalise(text).casefold().strip()
    # Remove a trailing dotted leader + page number before punctuation folding.
    s = re.sub(r"\.{3,}\s*\d{1,4}\s*$", "", s)
    s = s.replace("–", " ").replace("—", " ").replace("-", " ")
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _heading_key_without_number(text: str) -> str:
    """Canonical key with a leading numeric section/list marker removed."""
    s = _heading_key(text)
    s = re.sub(r"^\d{1,3}(?:\s+\d{1,3})*\s+", "", s)
    return s


def _numeric_heading_depth(text: str) -> int | None:
    """
    Return explicit decimal section depth, e.g.:
      ``4.`` -> 1, ``4.1`` -> 2, ``4.1.1`` -> 3.

    A dotted numeric hierarchy is one of the strongest structure signals in a
    report and should override ambiguous font-size/TOC indentation heuristics.
    """
    text = normalise(text)
    m = re.match(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s+\S", text)
    if not m:
        return None
    return len(m.group(1).split("."))


def _numeric_heading_prefix(text: str) -> str:
    """Return a leading decimal section number without a trailing period."""
    text = normalise(text)
    m = re.match(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s+", text)
    return m.group(1) if m else ""


def extract_toc_with_pymupdf(pdf_bytes: bytes) -> tuple[list[dict], set[int]]:
    """
    Extract a visible table of contents from the PDF itself.

    Supports both common PDF encodings:
      1. ``Heading ........ 12`` stored as one visual line, and
      2. ``Heading`` and ``12`` stored as separate line objects on the same row.

    Indentation is used when it is present. If the TOC is visually flat, levels
    are refined later from the matched body-heading typography instead of
    assuming every entry is H1.
    """
    if fitz is None:
        return [], set()

    entries: list[dict] = []
    toc_pages: set[int] = set()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for pidx, page in enumerate(doc[:MAX_PAGES], start=1):
            raw_lines: list[dict] = []
            for block in page.get_text("dict", sort=False).get("blocks", []):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    txt = "".join(s.get("text", "") for s in spans).strip()
                    if not txt:
                        continue
                    size, emphasis = _line_style(line)
                    raw_lines.append({
                        "text": normalise(txt),
                        "bbox": tuple(float(v) for v in line.get("bbox", (0, 0, 0, 0))),
                        "font_size": float(size),
                        "bold": bool(emphasis),
                        "font": spans[0].get("font", "") if spans else "",
                    })

            has_contents = any(
                _heading_key(item["text"]) in {"contents", "table of contents"}
                for item in raw_lines
            )
            if not has_contents:
                continue

            # Group separate PDF line objects that visually occupy the same TOC row.
            rows: list[list[dict]] = []
            for item in sorted(raw_lines, key=lambda x: (x["bbox"][1], x["bbox"][0])):
                key = _heading_key(item["text"])
                if key in {"contents", "table of contents"}:
                    continue
                y0 = float(item["bbox"][1])
                placed = False
                for row in rows[-3:]:
                    ry = sum(float(r["bbox"][1]) for r in row) / len(row)
                    if abs(y0 - ry) <= 2.6:
                        row.append(item)
                        placed = True
                        break
                if not placed:
                    rows.append([item])

            page_entries: list[dict] = []
            page_w = float(page.rect.width)
            for row in rows:
                row = sorted(row, key=lambda x: x["bbox"][0])
                joined = normalise(" ".join(r["text"] for r in row))
                if not joined:
                    continue

                heading = ""
                page_label = None
                layout_kind = ""
                heading_parts: list[dict] = []

                # Form 1: dotted leaders, possibly contained in one line object.
                m = re.match(r"^(.*?)\.{3,}\s*(\d{1,4})\s*$", joined)
                if m:
                    heading = normalise(m.group(1))
                    page_label = int(m.group(2))
                    layout_kind = "dotted"
                    heading_parts = row
                else:
                    # Form 2: page number stored as its own object at the far right.
                    last = row[-1]
                    if re.fullmatch(r"\d{1,4}", normalise(last["text"])) and len(row) >= 2:
                        lx0 = float(last["bbox"][0])
                        previous_x1 = max(float(r["bbox"][2]) for r in row[:-1])
                        if lx0 >= page_w * 0.72 or (lx0 - previous_x1) >= page_w * 0.12:
                            heading_parts = row[:-1]
                            heading = normalise(" ".join(r["text"] for r in heading_parts))
                            page_label = int(normalise(last["text"]))
                            layout_kind = "paired"

                if not heading or page_label is None or len(heading.split()) > 30:
                    continue

                # Remove dotted leader residue/control characters from the title.
                heading = re.sub(r"\.{2,}\s*$", "", normalise(heading)).strip()
                if not heading:
                    continue

                x0 = min(float(r["bbox"][0]) for r in heading_parts)
                weights = [max(1, len(r["text"])) for r in heading_parts]
                total_w = sum(weights) or 1
                font_size = sum(float(r.get("font_size", 0.0)) * w for r, w in zip(heading_parts, weights)) / total_w
                font = max(
                    (r.get("font", "") for r in heading_parts),
                    key=lambda f: sum(w for r, w in zip(heading_parts, weights) if r.get("font", "") == f),
                    default="",
                )

                page_entries.append({
                    "text": heading,
                    "page_label": page_label,
                    "x0": x0,
                    "toc_physical_page": pidx,
                    "font_size": round(font_size, 1),
                    "font": font,
                    "layout_kind": layout_kind,
                })

            if not page_entries:
                continue

            toc_pages.add(pidx)

            # Indentation is the strongest generic TOC hierarchy signal. Keep
            # it when genuinely present; otherwise leave the TOC flat and let
            # body-heading typography refine the levels later.
            starts = sorted(e["x0"] for e in page_entries)
            clusters: list[float] = []
            for x in starts:
                if not clusters or abs(x - clusters[-1]) > 12.0:
                    clusters.append(x)
                else:
                    clusters[-1] = (clusters[-1] + x) / 2.0

            meaningful_indent = len(clusters) > 1 and (max(clusters) - min(clusters)) >= 14.0
            for e in page_entries:
                numeric_depth = _numeric_heading_depth(e["text"])
                if numeric_depth is not None and numeric_depth > 1:
                    # Explicit decimal numbering is stronger than indentation.
                    e["level"] = numeric_depth
                    e["level_source"] = "numbering"
                elif meaningful_indent:
                    nearest = min(range(len(clusters)), key=lambda i: abs(e["x0"] - clusters[i]))
                    e["level"] = nearest + 1
                    e["level_source"] = "indent"
                else:
                    e["level"] = 1
                    e["level_source"] = "flat"
                e["source"] = "visible_toc"
                e["key"] = _heading_key(e["text"])
                e["key_without_number"] = _heading_key_without_number(e["text"])
                e["numeric_prefix"] = _numeric_heading_prefix(e["text"])
                entries.append(e)

        doc.close()
    except Exception as e:
        print(f"    [WARN] TOC extraction: {e}")
        return [], set()

    return entries, toc_pages


def extract_outline_with_pymupdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract the PDF document outline/bookmarks when present.

    Outlines are valuable because they encode author-supplied hierarchy even
    when typography is ambiguous. They are treated as strong evidence, but not
    blindly trusted: visibly printed TOC entries and explicit body numbering can
    override an inconsistent bookmark level.
    """
    if fitz is None:
        return []

    entries: list[dict] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for item in doc.get_toc(simple=True):
            if len(item) < 3:
                continue
            level, title, page_no = item[:3]
            title = normalise(title)
            if not title or not page_no:
                continue

            # Drop obvious accidental paragraph bookmarks while keeping long but
            # legitimate report headings.
            wc = len(title.split())
            personal_titles = len(re.findall(
                r"\b(?:dr|prof(?:essor)?|mr|mrs|ms|miss)\.?\b",
                title,
                flags=re.IGNORECASE,
            ))
            if (
                wc > 30
                or (wc > 18 and title.endswith((".", ";", ":")))
                or personal_titles >= 3
            ):
                continue

            depth = _numeric_heading_depth(title)
            if depth is not None and depth > 1:
                resolved_level = depth
                level_source = "numbering"
            else:
                resolved_level = max(1, int(level or 1))
                level_source = "outline"

            entries.append({
                "text": title,
                "physical_page": int(page_no),
                "level": resolved_level,
                "level_source": level_source,
                "source": "pdf_outline",
                "key": _heading_key(title),
                "key_without_number": _heading_key_without_number(title),
                "numeric_prefix": _numeric_heading_prefix(title),
            })
        doc.close()
    except Exception as e:
        print(f"    [WARN] PDF outline extraction: {e}")
        return []
    return entries


def _refine_flat_toc_levels_from_body(
    toc_entries: list[dict], segments: list[dict], toc_pages: set[int], body_size: float
) -> list[dict]:
    """
    Refine a visually flat TOC using the typography of the matching body headings.

    Some designed reports show all TOC rows at the same x-position even though
    the body clearly uses, for example, 18pt H1 headings and 12pt H2 headings.
    This function only activates when indentation supplied no hierarchy and when
    there is a clear body-font-size separation, so indented TOCs are untouched.
    """
    if not toc_entries or any(int(e.get("level", 1) or 1) > 1 for e in toc_entries):
        return toc_entries

    # Infer printed-page -> physical-PDF offset from any exact body matches.
    votes: Counter[int] = Counter()
    for e in toc_entries:
        label = int(e.get("page_label", 0) or 0)
        keys = {e.get("key", ""), e.get("key_without_number", "")}
        keys.discard("")
        if not label or not keys:
            continue
        for seg in segments:
            p = int(seg.get("physical_page", 0) or 0)
            if p in toc_pages:
                continue
            if _heading_key(seg.get("text", "")) in keys or _heading_key_without_number(seg.get("text", "")) in keys:
                votes[p - label] += 1
    offset = votes.most_common(1)[0][0] if votes else None

    matched: list[tuple[dict, float]] = []
    for e in toc_entries:
        label = int(e.get("page_label", 0) or 0)
        expected = label + offset if (label and offset is not None) else None
        keys = {e.get("key", ""), e.get("key_without_number", "")}
        keys.discard("")
        cands = []
        for seg in segments:
            p = int(seg.get("physical_page", 0) or 0)
            if p in toc_pages or (expected is not None and p != expected):
                continue
            sk = _heading_key(seg.get("text", ""))
            sn = _heading_key_without_number(seg.get("text", ""))
            if sk in keys or sn in keys:
                cands.append(seg)
        if cands:
            best = min(cands, key=lambda s: (float(s["bbox"][1]), float(s["bbox"][0])))
            matched.append((e, float(best.get("font_size", 0.0))))

    if len(matched) < 3:
        return toc_entries

    # Cluster matched heading sizes. Only accept a hierarchy if the largest
    # style is clearly separated from the next style; this avoids inventing
    # levels from tiny font-rendering differences.
    sizes = sorted({round(sz * 2) / 2.0 for _, sz in matched if sz > 0}, reverse=True)
    clusters: list[float] = []
    for sz in sizes:
        if not clusters or abs(sz - clusters[-1]) > 1.0:
            clusters.append(sz)
        else:
            clusters[-1] = (clusters[-1] + sz) / 2.0
    if len(clusters) < 2 or clusters[0] < clusters[1] * 1.18:
        return toc_entries

    # Use at most three body-derived levels. Tiny styles near body text are not
    # automatically made deeper unless the TOC explicitly indented them.
    usable = [c for c in clusters if c >= body_size * 1.05][:3]
    if len(usable) < 2:
        return toc_entries

    size_by_entry = {id(e): sz for e, sz in matched}
    refined = []
    for original in toc_entries:
        sz = size_by_entry.get(id(original))
        e = dict(original)
        if sz is not None:
            level = min(range(len(usable)), key=lambda i: abs(sz - usable[i])) + 1
            e["level"] = level
            e["level_source"] = "body-typography"
        refined.append(e)
    return refined

def _estimate_body_font_size(segments: list[dict]) -> float:
    """Estimate the dominant body font from character-weighted span segments."""
    counts: Counter[float] = Counter()
    for seg in segments:
        text = seg.get("text", "")
        size = round(float(seg.get("font_size", 0.0)) * 2) / 2.0
        # Ignore obviously decorative/display text when estimating body size.
        if not text or size <= 0 or size > 16.0:
            continue
        counts[size] += min(len(text), 1200)
    return counts.most_common(1)[0][0] if counts else 11.0



def _boilerplate_signature(text: str) -> str:
    """Normalise recurring margin text while masking changing page numbers."""
    text = normalise(text).casefold()
    # Running footers often differ only by a leading / trailing page number.
    text = re.sub(r"^\d{1,4}\s+", "<page> ", text)
    text = re.sub(r"\s+\d{1,4}$", " <page>", text)
    return text


def _remove_margin_boilerplate(segments: list[dict], body_size: float) -> list[dict]:
    """
    Remove repeated running headers/footers using position + cross-page frequency.

    Bottom-margin repeats are removed regardless of font size. Top-margin repeats are
    removed only when they are body-sized/small; large repeated display headings are
    retained so the sectionizer can keep the first occurrence as a real section heading.
    """
    if not segments:
        return []

    occurrences: dict[tuple[str, str], set[int]] = {}
    for seg_index, seg in enumerate(segments):
        text = normalise(seg.get("text", ""))
        if not text or len(text.split()) > 18:
            continue
        h = float(seg.get("logical_height", 1.0) or 1.0)
        y0, y1 = float(seg["bbox"][1]), float(seg["bbox"][3])
        zone = None
        if y0 >= h * 0.86:
            zone = "bottom"
        elif y1 <= h * 0.10 and float(seg.get("font_size", 0.0)) <= body_size * 1.30:
            zone = "top"
        if zone is None:
            continue
        sig = _boilerplate_signature(text)
        occurrences.setdefault((zone, sig), set()).add(int(seg.get("logical_page", 0)))

    repeated = {key for key, pages in occurrences.items() if len(pages) >= 3}
    if not repeated:
        return segments

    filtered: list[dict] = []
    for seg in segments:
        text = normalise(seg.get("text", ""))
        h = float(seg.get("logical_height", 1.0) or 1.0)
        y0, y1 = float(seg["bbox"][1]), float(seg["bbox"][3])
        zone = None
        if y0 >= h * 0.86:
            zone = "bottom"
        elif y1 <= h * 0.10 and float(seg.get("font_size", 0.0)) <= body_size * 1.30:
            zone = "top"
        if zone and (zone, _boilerplate_signature(text)) in repeated:
            continue
        filtered.append(seg)
    return filtered


def _merge_adjacent_heading_segments(segments: list[dict], body_size: float) -> list[dict]:
    """
    Join consecutive visual heading lines before classification.

    Some authoring tools store each line of a wrapped heading as a separate PDF block,
    even though the lines have identical typography. Geometry lets us safely reassemble
    those lines without document-specific wording.
    """
    if not segments:
        return []

    merged: list[dict] = []
    for seg in segments:
        seg = dict(seg)
        text = normalise(seg.get("text", ""))
        if not text:
            continue
        seg["text"] = text

        if merged:
            prev = merged[-1]
            same_page = prev.get("logical_page") == seg.get("logical_page")
            same_style = (
                bool(prev.get("bold")) == bool(seg.get("bold"))
                and abs(float(prev.get("font_size", 0.0)) - float(seg.get("font_size", 0.0))) <= 0.55
            )
            heading_like_style = (
                bool(seg.get("bold"))
                and float(seg.get("font_size", 0.0)) >= body_size * 1.08
            )
            px0, py0, px1, py1 = [float(v) for v in prev["bbox"]]
            x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
            logical_width = float(seg.get("logical_width", 1.0) or 1.0)
            aligned = abs(px0 - x0) <= max(8.0, logical_width * 0.035)
            vertical_gap = y0 - py1
            close = -2.0 <= vertical_gap <= max(28.0, float(seg.get("font_size", 0.0)) * 1.25)
            combined_words = len(prev.get("text", "").split()) + len(text.split())

            if (
                same_page and same_style and heading_like_style and aligned and close
                and combined_words <= 28
                and not text.startswith("•")
                and not prev.get("text", "").startswith("•")
            ):
                prev["text"] = normalise(prev["text"] + " " + text)
                prev["bbox"] = (
                    min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1)
                )
                continue

        merged.append(seg)
    return merged


def _reconstruct_table_rows(segments: list[dict], body_size: float) -> tuple[list[dict], int]:
    """
    Reconstruct obvious text-native table rows from aligned PDF fragments.

    The detector is intentionally conservative: it requires repeated rows on the
    same logical page with multiple horizontally separated cells. This preserves
    numeric table values that would otherwise look like page numbers while
    avoiding a document-specific table parser.
    """
    if not segments:
        return segments, 0

    by_page: dict[int, list[int]] = {}
    for idx, seg in enumerate(segments):
        by_page.setdefault(int(seg.get("logical_page", 0) or 0), []).append(idx)

    merge_groups: dict[int, list[int]] = {}
    table_rows = 0

    for lp, indices in by_page.items():
        rows: list[list[int]] = []
        for idx in sorted(indices, key=lambda i: (float(segments[i]["bbox"][1]), float(segments[i]["bbox"][0]))):
            seg = segments[idx]
            txt = normalise(seg.get("text", ""))
            if not txt or len(txt) > 120:
                continue
            y0 = float(seg["bbox"][1])
            placed = False
            for row in rows[-4:]:
                row_boxes = [segments[j]["bbox"] for j in row]
                ry0 = min(float(b[1]) for b in row_boxes)
                ry1 = max(float(b[3]) for b in row_boxes)
                cy0, cy1 = float(seg["bbox"][1]), float(seg["bbox"][3])
                overlap = max(0.0, min(ry1, cy1) - max(ry0, cy0))
                min_h = max(1.0, min(ry1 - ry0, cy1 - cy0))
                row_center = (ry0 + ry1) / 2.0
                cand_center = (cy0 + cy1) / 2.0
                if (overlap / min_h) >= 0.35 or abs(cand_center - row_center) <= 4.5:
                    row.append(idx)
                    placed = True
                    break
            if not placed:
                rows.append([idx])

        candidates: list[list[int]] = []
        for row in rows:
            if len(row) < 2 or len(row) > 4:
                continue
            row = sorted(row, key=lambda i: float(segments[i]["bbox"][0]))
            width = float(segments[row[0]].get("logical_width", 1.0) or 1.0)
            centres = [
                (float(segments[i]["bbox"][0]) + float(segments[i]["bbox"][2])) / 2.0
                for i in row
            ]
            if max(centres) - min(centres) < width * 0.16:
                continue

            cell_texts = [normalise(segments[i].get("text", "")) for i in row]
            # Tables usually contain compact cells. This deliberately rejects
            # ordinary two-column prose and infographic sentences.
            if any(len(t.split()) > 6 or len(t) > 48 for t in cell_texts):
                continue
            candidates.append(row)

        # A single aligned pair is often just a two-column layout. Repeated,
        # geometrically stable rows are a much stronger table signal.
        if len(candidates) < 3:
            continue

        # Require stable column starts across at least three rows.
        by_cell_count: dict[int, list[list[int]]] = {}
        for row in candidates:
            by_cell_count.setdefault(len(row), []).append(row)
        stable_groups = max(by_cell_count.values(), key=len)
        if len(stable_groups) < 3:
            continue

        ncols = len(stable_groups[0])
        start_ranges = []
        for col in range(ncols):
            xs = [float(segments[row[col]]["bbox"][0]) for row in stable_groups]
            start_ranges.append(max(xs) - min(xs))
        width = float(segments[stable_groups[0][0]].get("logical_width", 1.0) or 1.0)
        if any(rng > max(28.0, width * 0.06) for rng in start_ranges):
            continue

        y0 = min(float(segments[row[0]]["bbox"][1]) for row in stable_groups)
        y1 = max(max(float(segments[i]["bbox"][3]) for i in row) for row in stable_groups)
        height = float(segments[stable_groups[0][0]].get("logical_height", 1.0) or 1.0)
        if (y1 - y0) > height * 0.38:
            continue

        for row in stable_groups:
            row = sorted(row, key=lambda i: float(segments[i]["bbox"][0]))
            first = row[0]
            merge_groups[first] = row
            table_rows += 1

    if not merge_groups:
        return segments, 0

    consumed = {j for row in merge_groups.values() for j in row[1:]}
    rebuilt: list[dict] = []
    for idx, seg in enumerate(segments):
        if idx in consumed:
            continue
        row = merge_groups.get(idx)
        if not row:
            rebuilt.append(seg)
            continue

        cells = [normalise(segments[j].get("text", "")) for j in row]
        cells = [c for c in cells if c]
        if not cells:
            rebuilt.append(seg)
            continue

        # PyMuPDF sometimes stores adjacent numeric table columns in one span
        # (e.g. ``25 50`` for Number and %). Split only purely numeric/percent
        # spans so prose cells are never fragmented.
        expanded_cells: list[str] = []
        for cell in cells:
            parts = cell.split()
            if len(parts) > 1 and all(re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?%?", p) for p in parts):
                expanded_cells.extend(parts)
            else:
                expanded_cells.append(cell)
        cells = expanded_cells

        boxes = [segments[j]["bbox"] for j in row]
        chars = [max(1, len(normalise(segments[j].get("text", "")))) for j in row]
        total_chars = sum(chars) or 1
        merged = {
            **seg,
            "text": " | ".join(cells),
            "cells": cells,
            "block_type": "table_row",
            "font_size": sum(float(segments[j].get("font_size", 0.0)) * w for j, w in zip(row, chars)) / total_chars,
            "bold": all(bool(segments[j].get("bold")) for j in row),
            "bbox": (
                min(float(b[0]) for b in boxes),
                min(float(b[1]) for b in boxes),
                max(float(b[2]) for b in boxes),
                max(float(b[3]) for b in boxes),
            ),
        }
        rebuilt.append(merged)

    # A compact label immediately followed by reconstructed rows is the table
    # header, not a document subsection. This is much safer than suppressing
    # any heading that merely has table rows somewhere below it.
    for i, seg in enumerate(rebuilt):
        if seg.get("block_type") or not normalise(seg.get("text", "")):
            continue
        if not bool(seg.get("bold")):
            continue
        if len(normalise(seg.get("text", "")).split()) > 12:
            continue
        following = []
        for cand in rebuilt[i + 1:i + 4]:
            if int(cand.get("logical_page", 0) or 0) != int(seg.get("logical_page", 0) or 0):
                break
            if cand.get("block_type") == "table_row":
                following.append(cand)
            else:
                break
        if len(following) >= 2:
            seg["block_type"] = "table_header"
            raw_lines = [l for l in seg.get("_lines", []) if normalise(l.get("text", ""))]
            if len(raw_lines) >= 2:
                # Distinct x-starts in the header are genuine column labels.
                raw_lines = sorted(raw_lines, key=lambda l: float(l["bbox"][0]))
                seg["cells"] = [normalise(l.get("text", "")) for l in raw_lines]
                seg["text"] = " | ".join(seg["cells"])
            else:
                seg["cells"] = [normalise(seg.get("text", ""))]

    return rebuilt, table_rows


def _reconstruct_text_table_regions(segments: list[dict], body_size: float) -> tuple[list[dict], int]:
    """
    Reconstruct text-heavy tables using the original PDF line geometry.

    PyMuPDF can place several table columns inside one text block. Stage 4 keeps
    the raw line boxes in ``_lines`` and uses a three-column header as strong
    evidence that the following aligned material is a table, not ordinary
    multi-column prose. This handles both numeric tables and prose-heavy tables
    without relying on document-specific column names.
    """
    if not segments:
        return segments, 0

    def clustered_lines(seg: dict) -> list[tuple[float, list[dict]]]:
        lines = [l for l in seg.get("_lines", []) if normalise(l.get("text", ""))]
        if not lines:
            return []
        width = float(seg.get("logical_width", 1.0) or 1.0)
        tol = max(16.0, width * 0.035)
        clusters: list[tuple[float, list[dict]]] = []
        for line in sorted(lines, key=lambda l: (float(l["bbox"][0]), float(l["bbox"][1]))):
            x = float(line["bbox"][0])
            if not clusters or abs(x - clusters[-1][0]) > tol:
                clusters.append((x, [line]))
            else:
                ox, ls = clusters[-1]
                ls = ls + [line]
                nx = sum(float(z["bbox"][0]) for z in ls) / len(ls)
                clusters[-1] = (nx, ls)
        return clusters

    def cells_for_starts(lines: list[dict], starts: list[float], width: float) -> list[str]:
        buckets: list[list[dict]] = [[] for _ in starts]
        max_dist = max(38.0, width * 0.09)
        for line in lines:
            x = float(line["bbox"][0])
            nearest = min(range(len(starts)), key=lambda k: abs(x - starts[k]))
            if abs(x - starts[nearest]) <= max_dist:
                buckets[nearest].append(line)
        cells: list[str] = []
        for bucket in buckets:
            bucket = sorted(bucket, key=lambda l: (float(l["bbox"][1]), float(l["bbox"][0])))
            cells.append(normalise(" ".join(l.get("text", "") for l in bucket)))
        return cells

    by_page: dict[int, list[int]] = {}
    for idx, seg in enumerate(segments):
        by_page.setdefault(int(seg.get("logical_page", 0) or 0), []).append(idx)

    merge_groups: dict[int, tuple[list[int], list[str]]] = {}
    table_headers: dict[int, list[str]] = {}
    extra_rows = 0

    for lp, ids in by_page.items():
        ids = sorted(ids, key=lambda i: (float(segments[i]["bbox"][1]), float(segments[i]["bbox"][0])))
        for pos, header_idx in enumerate(ids):
            header = segments[header_idx]
            if header.get("block_type"):
                continue
            htext = normalise(header.get("text", ""))
            if not htext or not bool(header.get("bold")):
                continue
            hsize = float(header.get("font_size", 0.0))
            if not (body_size * 0.78 <= hsize <= body_size * 1.22):
                continue

            hclusters = clustered_lines(header)
            width = float(header.get("logical_width", 1.0) or 1.0)
            # Three clearly separated header columns are strong table evidence.
            if len(hclusters) < 3:
                continue
            starts = [x for x, _ in hclusters]
            if starts[-1] - starts[0] < width * 0.28:
                continue
            header_cells = [normalise(" ".join(l.get("text", "") for l in ls)) for _, ls in hclusters]
            if len([c for c in header_cells if c]) < 3:
                continue

            region: list[int] = []
            for idx in ids[pos + 1:]:
                cand = segments[idx]
                cy0 = float(cand["bbox"][1])
                if cy0 - float(header["bbox"][3]) > float(header.get("logical_height", 1.0) or 1.0) * 0.42:
                    break
                ctext = normalise(cand.get("text", ""))
                if not ctext:
                    continue
                cclusters = clustered_lines(cand)
                # A new normal subsection heading ends the table. A repeated
                # three-column header will be handled on its own pass.
                if (
                    bool(cand.get("bold"))
                    and float(cand.get("font_size", 0.0)) >= body_size * 0.95
                    and len(cclusters) < 3
                    and len(ctext.split()) <= 12
                    and abs(float(cand["bbox"][0]) - float(header["bbox"][0])) <= max(24.0, width * 0.06)
                ):
                    break
                if cand.get("block_type"):
                    continue
                if body_size * 0.72 <= float(cand.get("font_size", 0.0)) <= body_size * 1.22:
                    region.append(idx)

            if len(region) < 3:
                continue

            # Row anchors are blocks that contain material in the first header
            # column. Their y positions define row boundaries; other column
            # fragments are attached geometrically within those boundaries.
            first_x = starts[0]
            anchors: list[int] = []
            for idx in region:
                lines = segments[idx].get("_lines", []) or [{"text": segments[idx].get("text", ""), "bbox": segments[idx]["bbox"]}]
                if any(abs(float(l["bbox"][0]) - first_x) <= max(32.0, width * 0.065) for l in lines):
                    anchors.append(idx)
            anchors = sorted(set(anchors), key=lambda i: float(segments[i]["bbox"][1]))
            # Collapse multiple anchors that are really continuation fragments
            # of the same first-column cell.
            collapsed: list[int] = []
            for idx in anchors:
                if not collapsed:
                    collapsed.append(idx)
                    continue
                prev = collapsed[-1]
                gap = float(segments[idx]["bbox"][1]) - float(segments[prev]["bbox"][3])
                if gap <= max(3.0, body_size * 0.25):
                    continue
                collapsed.append(idx)
            anchors = collapsed
            if len(anchors) < 2:
                continue

            region_ymax = max(float(segments[i]["bbox"][3]) for i in region)
            candidate_rows: list[tuple[int, list[int], list[str]]] = []
            for a_pos, anchor in enumerate(anchors):
                y0 = float(segments[anchor]["bbox"][1])
                y_next = float(segments[anchors[a_pos + 1]]["bbox"][1]) if a_pos + 1 < len(anchors) else region_ymax + body_size
                members = [
                    i for i in region
                    if y0 - 2.0 <= float(segments[i]["bbox"][1]) < y_next - 1.0
                ]
                if not members:
                    continue
                lines: list[dict] = []
                for member_idx in members:
                    raw_lines = segments[member_idx].get("_lines", [])
                    if raw_lines:
                        lines.extend(raw_lines)
                    else:
                        lines.append({"text": segments[member_idx].get("text", ""), "bbox": segments[member_idx]["bbox"]})
                cells = cells_for_starts(lines, starts, width)
                if len([c for c in cells if c]) < 2:
                    continue
                first = min(members, key=lambda i: (float(segments[i]["bbox"][1]), float(segments[i]["bbox"][0])))
                candidate_rows.append((first, sorted(set(members)), cells))

            # A real three-column table should expose at least two complete
            # rows. This rejects infographic/card bands whose labels happen to
            # form three x-clusters but whose body content does not follow the
            # same grid.
            complete_rows = sum(1 for _, _, cells in candidate_rows if len(cells) >= 3 and all(cells[:3]))
            if complete_rows < 2:
                continue

            table_headers[header_idx] = header_cells
            for first, members, cells in candidate_rows:
                merge_groups[first] = (members, cells)
                extra_rows += 1

    if not merge_groups and not table_headers:
        return segments, 0

    consumed = {j for members, _ in merge_groups.values() for j in members[1:]}
    rebuilt: list[dict] = []
    for idx, seg in enumerate(segments):
        if idx in consumed:
            continue
        if idx in table_headers:
            cells = table_headers[idx]
            rebuilt.append({**seg, "block_type": "table_header", "cells": cells, "text": " | ".join(cells)})
            continue
        row = merge_groups.get(idx)
        if not row:
            rebuilt.append(seg)
            continue
        members, cells = row
        boxes = [segments[j]["bbox"] for j in members]
        rebuilt.append({
            **seg,
            "text": " | ".join(cells),
            "cells": cells,
            "block_type": "table_row",
            "bbox": (
                min(float(b[0]) for b in boxes), min(float(b[1]) for b in boxes),
                max(float(b[2]) for b in boxes), max(float(b[3]) for b in boxes),
            ),
        })

    # Repeated three-column headers are also typed even if their subsequent
    # rows are too fused to reconstruct confidently.
    recognised = {_heading_key(" ".join(v)) for v in table_headers.values()}
    if recognised:
        for i, seg in enumerate(rebuilt):
            if seg.get("block_type"):
                continue
            clusters = clustered_lines(seg)
            if len(clusters) >= 3:
                cells = [normalise(" ".join(l.get("text", "") for l in ls)) for _, ls in clusters]
                if _heading_key(" ".join(cells)) in recognised:
                    rebuilt[i] = {**seg, "block_type": "table_header", "cells": cells, "text": " | ".join(cells)}

    return rebuilt, extra_rows


def _reconstruct_semantic_blocks(segments: list[dict], body_size: float) -> tuple[list[dict], int]:
    """
    Convert low-level PDF line fragments into semantic paragraph/list blocks.

    - joins visual lines that clearly belong to one paragraph;
    - preserves bullet and numbered list items as typed blocks;
    - attaches standalone graphical list numbers (e.g. 1..12 beside text);
    - never invents missing list numbers.
    """
    if not segments:
        return segments, 0

    bullet_re = re.compile(r"^\s*([•●▪◦‣⁃])\s*(.+)$")
    inline_num_re = re.compile(r"^\s*(\d{1,3})(?:[.)])?\s+(.+)$")

    # Pass 1: join only line-height body fragments. Existing multi-line
    # paragraph blocks are left untouched.
    coalesced: list[dict] = []
    for seg in segments:
        txt = normalise(seg.get("text", ""))
        if not txt:
            continue
        if not coalesced:
            coalesced.append(seg)
            continue
        prev = coalesced[-1]
        ptxt = normalise(prev.get("text", ""))
        same_page = int(prev.get("logical_page", 0) or 0) == int(seg.get("logical_page", 0) or 0)
        ph = float(prev["bbox"][3]) - float(prev["bbox"][1])
        ch = float(seg["bbox"][3]) - float(seg["bbox"][1])
        gap = float(seg["bbox"][1]) - float(prev["bbox"][3])
        width = float(seg.get("logical_width", 1.0) or 1.0)
        same_lane = abs(float(seg["bbox"][0]) - float(prev["bbox"][0])) <= max(8.0, width * 0.018)
        line_like = ph <= body_size * 1.65 and ch <= body_size * 1.65
        same_style = (
            not bool(prev.get("bold")) and not bool(seg.get("bold"))
            and abs(float(prev.get("font_size", 0.0)) - float(seg.get("font_size", 0.0))) <= 0.75
        )
        marker_boundary = bool(
            bullet_re.match(txt) or bullet_re.match(ptxt)
            or re.fullmatch(r"\d{1,3}", txt) or re.fullmatch(r"\d{1,3}", ptxt)
            or inline_num_re.match(txt)
        )
        if (
            same_page and line_like and same_style and same_lane and not marker_boundary
            and -1.0 <= gap <= max(14.0, body_size * 1.15)
            and not prev.get("block_type") and not seg.get("block_type")
        ):
            joined = _join_pdf_lines([ptxt, txt])
            coalesced[-1] = {
                **prev,
                "text": joined,
                "_lines": list(prev.get("_lines", [])) + list(seg.get("_lines", [])),
                "bbox": (
                    min(float(prev["bbox"][0]), float(seg["bbox"][0])),
                    min(float(prev["bbox"][1]), float(seg["bbox"][1])),
                    max(float(prev["bbox"][2]), float(seg["bbox"][2])),
                    max(float(prev["bbox"][3]), float(seg["bbox"][3])),
                ),
            }
        else:
            coalesced.append(seg)

    segments = coalesced

    # Pass 2: identify repeated standalone numeric markers aligned beside text.
    pairs: dict[int, int] = {}
    by_page: dict[int, list[tuple[int, int, float, float]]] = {}
    for i, seg in enumerate(segments):
        txt = normalise(seg.get("text", ""))
        if not re.fullmatch(r"\d{1,3}", txt):
            continue
        h = float(seg.get("logical_height", 1.0) or 1.0)
        y0, y1 = float(seg["bbox"][1]), float(seg["bbox"][3])
        if y1 >= h * 0.91:
            continue
        page = int(seg.get("logical_page", 0) or 0)
        best: tuple[float, int] | None = None
        mcy = (y0 + y1) / 2.0
        for j, cand in enumerate(segments):
            if j == i or int(cand.get("logical_page", 0) or 0) != page:
                continue
            ctxt = normalise(cand.get("text", ""))
            if not ctxt or len(ctxt) < 8 or cand.get("block_type"):
                continue
            if bool(cand.get("bold")):
                continue
            cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
            if cx0 <= float(seg["bbox"][2]) + 8.0:
                continue
            ccy = (cy0 + cy1) / 2.0
            dy = abs(ccy - mcy)
            # Large decorative numbers often sit vertically centred beside a
            # multi-line item; allow a generous centre tolerance.
            if dy > max(22.0, (y1 - y0) * 0.9, (cy1 - cy0) * 0.45):
                continue
            score = dy + max(0.0, cx0 - float(seg["bbox"][2])) * 0.02
            if best is None or score < best[0]:
                best = (score, j)
        if best:
            j = best[1]
            by_page.setdefault(page, []).append((i, j, float(seg["bbox"][0]), float(segments[j]["bbox"][0])))

    for page, candidates in by_page.items():
        if len(candidates) < 3:
            continue
        # Repeated alignment is the key evidence that these are list markers,
        # not chart values or page numbers.
        mx = [c[2] for c in candidates]
        tx = [c[3] for c in candidates]
        if (max(mx) - min(mx)) <= 24.0 and (max(tx) - min(tx)) <= 34.0:
            for i, j, _, _ in candidates:
                pairs[i] = j

    consumed: set[int] = set()
    paired: list[dict] = []
    for i, seg in enumerate(segments):
        if i in consumed:
            continue
        if i in pairs:
            j = pairs[i]
            marker = normalise(seg.get("text", ""))
            body = segments[j]
            text = normalise(body.get("text", ""))
            boxes = [seg["bbox"], body["bbox"]]
            paired.append({
                **body,
                "text": f"{marker} {text}",
                "marker": marker,
                "block_type": "list_item",
                "bbox": (
                    min(float(b[0]) for b in boxes), min(float(b[1]) for b in boxes),
                    max(float(b[2]) for b in boxes), max(float(b[3]) for b in boxes),
                ),
            })
            consumed.add(j)
        else:
            paired.append(seg)
    segments = paired

    # Pass 3: merge bullet/numbered item continuations based on indentation.
    out: list[dict] = []
    list_count = 0
    i = 0
    while i < len(segments):
        seg = segments[i]
        txt = normalise(seg.get("text", ""))
        marker = ""
        item_text = ""
        already_list = seg.get("block_type") == "list_item"
        if already_list:
            marker = str(seg.get("marker", ""))
            item_text = re.sub(r"^\s*" + re.escape(marker) + r"\s+", "", txt, count=1) if marker else txt
        else:
            bm = bullet_re.match(txt)
            nm = inline_num_re.match(txt)
            if bm and not bool(seg.get("bold")):
                marker, item_text = bm.group(1), bm.group(2)
            elif nm and not bool(seg.get("bold")) and float(seg.get("font_size", 0.0)) <= body_size * 1.20:
                # Dotted multi-level headings are handled by the hierarchy
                # resolver, not the list parser.
                if re.match(r"^\d{1,3}(?:\.\d{1,3})+", txt):
                    nm = None
                else:
                    marker, item_text = nm.group(1), nm.group(2)

        if not marker:
            out.append(seg)
            i += 1
            continue

        parts = [item_text]
        boxes = [seg["bbox"]]
        page = int(seg.get("logical_page", 0) or 0)
        start_x = float(seg["bbox"][0])
        last_y1 = float(seg["bbox"][3])
        j = i + 1
        while j < len(segments):
            cand = segments[j]
            if int(cand.get("logical_page", 0) or 0) != page:
                break
            ctxt = normalise(cand.get("text", ""))
            if not ctxt or cand.get("block_type"):
                break
            if bullet_re.match(ctxt) or inline_num_re.match(ctxt):
                break
            if bool(cand.get("bold")):
                break
            cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
            gap = cy0 - last_y1
            # Wrapped list lines are normally indented relative to the marker.
            if cx0 < start_x + 8.0 or gap > max(30.0, body_size * 2.3):
                break
            if abs(float(cand.get("font_size", 0.0)) - float(seg.get("font_size", 0.0))) > 1.0:
                break
            parts.append(ctxt)
            boxes.append(cand["bbox"])
            last_y1 = cy1
            j += 1

        full_text = normalise(" ".join(parts))
        out.append({
            **seg,
            "text": f"{marker} {full_text}" if marker not in {"•", "●", "▪", "◦", "‣", "⁃"} else f"{marker} {full_text}",
            "marker": marker,
            "item_text": full_text,
            "block_type": "list_item",
            "bbox": (
                min(float(b[0]) for b in boxes), min(float(b[1]) for b in boxes),
                max(float(b[2]) for b in boxes), max(float(b[3]) for b in boxes),
            ),
        })
        list_count += 1
        i = j

    return out, list_count


def _annotate_footnotes_and_figures(
    segments: list[dict], body_size: float, protected_heading_keys: set[str] | None = None
) -> tuple[list[dict], int, int]:
    """Annotate footnotes and graphical regions so they cannot corrupt prose hierarchy."""
    if not segments:
        return segments, 0, 0

    segments = [dict(s) for s in segments]
    protected_heading_keys = set(protected_heading_keys or set())
    footnotes = 0
    figure_groups = 0

    # Bottom-of-page small-print footnotes.
    for i, seg in enumerate(segments):
        txt = normalise(seg.get("text", ""))
        if not txt:
            continue
        size = float(seg.get("font_size", 0.0))
        h = float(seg.get("logical_height", 1.0) or 1.0)
        y0 = float(seg["bbox"][1])
        footnote_shape = bool(re.match(r"^(?:\d{1,3}|[*†‡])\s+", txt))
        linkish = "http://" in txt.casefold() or "https://" in txt.casefold() or "doi" in txt.casefold()
        if size <= body_size * 0.86 and y0 >= h * 0.70 and (footnote_shape or linkish):
            segments[i]["block_type"] = "footnote"
            footnotes += 1

    # Captioned figures: group compact text between the previous prose block and
    # a conventional Figure/Fig./Diagram/Chart caption.
    caption_re = re.compile(r"^(?:figure|fig\.?|diagram|chart)\s*\d*[\s:.-]", re.IGNORECASE)
    next_group = 1
    for i, seg in enumerate(segments):
        if seg.get("block_type"):
            continue
        txt = normalise(seg.get("text", ""))
        if not caption_re.match(txt) or len(txt.split()) > 24:
            continue
        page = int(seg.get("logical_page", 0) or 0)
        start = i
        j = i - 1
        while j >= 0 and int(segments[j].get("logical_page", 0) or 0) == page:
            cand = segments[j]
            ctext = normalise(cand.get("text", ""))
            if cand.get("block_type") in {"footnote", "table_row", "table_header"}:
                break
            csize = float(cand.get("font_size", 0.0))
            if bool(cand.get("bold")) and csize >= body_size * 1.30:
                break
            # A substantial body paragraph marks the start of the graphical band.
            if len(ctext) >= 100 and csize >= body_size * 0.90:
                break
            start = j
            j -= 1
        if i - start >= 2:
            gid = f"fig-{page}-{next_group}"
            next_group += 1
            figure_groups += 1
            for k in range(start, i):
                if segments[k].get("block_type") not in {"footnote", "table_row", "table_header"}:
                    segments[k]["block_type"] = "figure_text"
                    segments[k]["figure_group"] = gid
            segments[i]["block_type"] = "figure_caption"
            segments[i]["figure_group"] = gid

    # Uncaptioned chart titles: a modest bold title followed immediately by a
    # dense band of numeric/percentage labels is graphical, not hierarchical.
    universal_like = {
        "abstract", "foreword", "preface", "introduction", "background", "methodology",
        "methods", "results", "discussion", "conclusion", "conclusions", "references",
        "acknowledgements", "acknowledgments", "recommendations", "summary", "overview",
        "appendix", "limitations", "implications", "interventions", "testimonials",
        "key findings", "findings in depth", "contents", "table of contents",
    }
    for i, seg in enumerate(segments):
        if seg.get("block_type"):
            continue
        txt = normalise(seg.get("text", ""))
        if not txt or not bool(seg.get("bold")) or not (2 <= len(txt.split()) <= 12):
            continue
        if _heading_key(txt) in universal_like or _heading_key(txt) in protected_heading_keys:
            continue
        if _numeric_heading_depth(txt) is not None or re.match(r"^(?:[IVXLCDM]+|[A-Z]|[ivxlcdm]+)\.\s+", txt):
            continue
        size = float(seg.get("font_size", 0.0))
        if not (body_size * 0.92 <= size <= body_size * 1.38):
            continue
        page = int(seg.get("logical_page", 0) or 0)
        h = float(seg.get("logical_height", 1.0) or 1.0)
        width = float(seg.get("logical_width", 1.0) or 1.0)
        y1 = float(seg["bbox"][3])
        window: list[int] = []
        for j in range(i + 1, min(len(segments), i + 14)):
            cand = segments[j]
            if int(cand.get("logical_page", 0) or 0) != page:
                break
            if float(cand["bbox"][1]) - y1 > h * 0.23:
                break
            if cand.get("block_type") in {"footnote", "table_row", "table_header", "list_item"}:
                break
            window.append(j)
        if len(window) < 4:
            continue
        texts = [normalise(segments[j].get("text", "")) for j in window]
        numeric = sum(bool(re.search(r"\b\d+(?:\.\d+)?%?\b", t)) for t in texts)
        compact = sum(len(t.split()) <= 10 and len(t) <= 70 for t in texts)
        centres = [
            (float(segments[j]["bbox"][0]) + float(segments[j]["bbox"][2])) / 2.0
            for j in window
        ]
        long_prose = sum(len(t) >= 100 for t in texts)
        if numeric >= 2 and compact >= 4 and long_prose == 0 and (max(centres) - min(centres)) >= width * 0.18:
            gid = f"fig-{page}-{next_group}"
            next_group += 1
            figure_groups += 1
            segments[i]["block_type"] = "figure_title"
            segments[i]["figure_group"] = gid
            for j in window:
                if not segments[j].get("block_type"):
                    segments[j]["block_type"] = "figure_text"
                    segments[j]["figure_group"] = gid

    # Dense process/infographic label bands (e.g. staggered process steps).
    by_page: dict[int, list[int]] = {}
    for i, seg in enumerate(segments):
        if seg.get("block_type"):
            continue
        txt = normalise(seg.get("text", ""))
        size = float(seg.get("font_size", 0.0))
        if bool(seg.get("bold")) and 1 <= len(txt.split()) <= 5 and body_size * 0.92 <= size <= body_size * 1.35:
            if (
                _heading_key(txt) not in protected_heading_keys
                and _numeric_heading_depth(txt) is None
                and not re.match(r"^(?:[IVXLCDM]+|[A-Z]|[ivxlcdm]+)\.\s+", txt)
            ):
                by_page.setdefault(int(seg.get("logical_page", 0) or 0), []).append(i)

    for page, ids in by_page.items():
        ids = sorted(ids, key=lambda j: float(segments[j]["bbox"][1]))
        if len(ids) < 4:
            continue
        height = float(segments[ids[0]].get("logical_height", 1.0) or 1.0)
        width = float(segments[ids[0]].get("logical_width", 1.0) or 1.0)
        for start_pos in range(len(ids)):
            band = [ids[start_pos]]
            top = float(segments[ids[start_pos]]["bbox"][1])
            for j in ids[start_pos + 1:]:
                if float(segments[j]["bbox"][1]) - top > height * 0.16:
                    break
                band.append(j)
            if len(band) < 4:
                continue
            centres = [
                (float(segments[j]["bbox"][0]) + float(segments[j]["bbox"][2])) / 2.0
                for j in band
            ]
            if max(centres) - min(centres) < width * 0.42:
                continue
            gid = f"fig-{page}-{next_group}"
            next_group += 1
            figure_groups += 1
            y_min = min(float(segments[j]["bbox"][1]) for j in band)
            y_max = max(float(segments[j]["bbox"][3]) for j in band) + max(90.0, body_size * 8.0)
            for j in band:
                segments[j]["block_type"] = "figure_text"
                segments[j]["figure_group"] = gid
            # Include narrow explanatory boxes in the same process band.
            for j, cand in enumerate(segments):
                if int(cand.get("logical_page", 0) or 0) != page or cand.get("block_type"):
                    continue
                if _heading_key(normalise(cand.get("text", ""))) in protected_heading_keys:
                    continue
                cy0, cy1 = float(cand["bbox"][1]), float(cand["bbox"][3])
                cwidth = float(cand["bbox"][2]) - float(cand["bbox"][0])
                if y_min <= cy0 <= y_max and cwidth <= width * 0.50 and not bool(cand.get("bold")):
                    segments[j]["block_type"] = "figure_text"
                    segments[j]["figure_group"] = gid
            break

    return segments, footnotes, figure_groups


def extract_sections_from_layout(
    segments: list[dict],
    toc_entries: list[dict] | None = None,
    toc_pages: set[int] | None = None,
    outline_entries: list[dict] | None = None,
) -> tuple[list[dict], str]:
    """
    Sectionise with layout + typography + optional TOC evidence.

    Geometry determines reading order. Typography identifies emphasis. A visible
    table of contents, when present, is used as high-confidence evidence for
    document headings and hierarchy. Regex is used only to interpret numbering.

    This prevents large statistical callouts from becoming sections merely
    because they use a huge font, while still supporting PDFs without a TOC.
    """
    if not segments:
        return [], ""

    toc_entries = toc_entries or []
    toc_pages = set(toc_pages or set())
    outline_entries = outline_entries or []

    body_size = _estimate_body_font_size(segments)
    segments = _remove_margin_boilerplate(segments, body_size)
    segments, table_rows_detected = _reconstruct_table_rows(segments, body_size)
    segments, text_table_rows = _reconstruct_text_table_regions(segments, body_size)
    table_rows_detected += text_table_rows
    segments = _merge_adjacent_heading_segments(segments, body_size)
    segments, list_items_detected = _reconstruct_semantic_blocks(segments, body_size)
    protected_heading_keys = {
        _heading_key(e.get("text", "")) for e in (toc_entries + outline_entries) if _heading_key(e.get("text", ""))
    }
    protected_heading_keys |= {
        _heading_key_without_number(e.get("text", "")) for e in (toc_entries + outline_entries) if _heading_key_without_number(e.get("text", ""))
    }
    segments, footnotes_detected, figure_groups_detected = _annotate_footnotes_and_figures(
        segments, body_size, protected_heading_keys=protected_heading_keys
    )
    toc_entries = _refine_flat_toc_levels_from_body(toc_entries, segments, toc_pages, body_size)

    lone_page = re.compile(r"^\d{1,4}$")
    numbered_heading = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.\s+[A-Z]")
    roman_upper = re.compile(r"^[IVXLCDM]+\.\s+\S")
    letter_upper = re.compile(r"^[A-Z]\.\s+\S")
    roman_lower = re.compile(r"^[ivxlcdm]+\.\s+\S")

    universal_sections = {
        "abstract", "foreword", "preface", "introduction", "background",
        "methodology", "methods", "results", "discussion", "conclusion",
        "conclusions", "references", "acknowledgements", "acknowledgments",
        "recommendations", "summary", "overview", "appendix", "limitations",
        "implications", "interventions", "testimonials", "key findings",
        "findings in-depth", "connect with us", "contents", "table of contents", "scope",
        "about this report", "about edica",
    }

    # Visible TOC lookup. Numeric prefixes get a second alias so an entry such
    # as "9. Collective action" can match body text rendered as "Collective action".
    toc_lookup: dict[str, int] = {}
    for entry in toc_entries:
        level = max(1, int(entry.get("level", 1) or 1))
        for key in {
            entry.get("key") or _heading_key(entry.get("text", "")),
            entry.get("key_without_number") or _heading_key_without_number(entry.get("text", "")),
        }:
            if key:
                toc_lookup[key] = min(level, toc_lookup.get(key, level))

    # Infer the offset between printed page labels in the TOC and physical PDF
    # pages (front matter commonly shifts them by 1-3 pages).
    offset_votes: Counter[int] = Counter()
    for entry in toc_entries:
        if int(entry.get("level", 1) or 1) != 1:
            continue
        ekey = entry.get("key") or _heading_key(entry.get("text", ""))
        if not ekey:
            continue
        label = int(entry.get("page_label", 0) or 0)
        if not label:
            continue
        for seg in segments:
            if int(seg.get("physical_page", 0) or 0) in toc_pages:
                continue
            if _heading_key(seg.get("text", "")) == ekey:
                offset_votes[int(seg.get("physical_page", 0) or 0) - label] += 1

    page_offset = offset_votes.most_common(1)[0][0] if offset_votes else None

    # Select one concrete body occurrence for each TOC entry. This avoids
    # globally matching the same words inside diagrams/tables elsewhere.
    selected_toc_levels: dict[int, int] = {}
    selected_toc_entries: dict[int, dict] = {}
    for entry in toc_entries:
        level = max(1, int(entry.get("level", 1) or 1))
        label = int(entry.get("page_label", 0) or 0)
        expected_page = (label + page_offset) if (label and page_offset is not None) else None
        keys = {
            entry.get("key") or _heading_key(entry.get("text", "")),
            entry.get("key_without_number") or _heading_key_without_number(entry.get("text", "")),
        }
        keys.discard("")

        candidates: list[tuple[int, dict]] = []
        for idx, seg in enumerate(segments):
            p = int(seg.get("physical_page", 0) or 0)
            if p in toc_pages:
                continue
            if expected_page is not None and p != expected_page:
                continue
            skey = _heading_key(seg.get("text", ""))
            sstrip = _heading_key_without_number(seg.get("text", ""))
            if skey in keys or sstrip in keys:
                candidates.append((idx, seg))

        if not candidates and expected_page is None:
            for idx, seg in enumerate(segments):
                if int(seg.get("physical_page", 0) or 0) in toc_pages:
                    continue
                skey = _heading_key(seg.get("text", ""))
                sstrip = _heading_key_without_number(seg.get("text", ""))
                if skey in keys or sstrip in keys:
                    candidates.append((idx, seg))

        if candidates:
            # Real report headings usually align to the primary text margin.
            # For duplicate labels on the same page (e.g. infographic + body
            # heading), prefer the left-most occurrence, then larger type.
            best_idx, _ = min(
                candidates,
                key=lambda pair: (
                    float(pair[1]["bbox"][0]),
                    -float(pair[1].get("font_size", 0.0)),
                    float(pair[1]["bbox"][1]),
                ),
            )
            selected_toc_levels[best_idx] = level
            selected_toc_entries[best_idx] = entry

    def toc_level_for(index: int) -> int | None:
        return selected_toc_levels.get(index)

    # PDF outline/bookmark evidence. Match on the exact bookmarked physical page
    # and prefer exact text over number-stripped aliases. Visible TOC evidence
    # takes precedence later when the two sources disagree.
    selected_outline_levels: dict[int, int] = {}
    selected_outline_entries: dict[int, dict] = {}
    for entry in outline_entries:
        page_no = int(entry.get("physical_page", 0) or 0)
        if not page_no:
            continue
        ekey = entry.get("key") or _heading_key(entry.get("text", ""))
        estrip = entry.get("key_without_number") or _heading_key_without_number(entry.get("text", ""))
        candidates: list[tuple[int, dict, int]] = []
        for idx, seg in enumerate(segments):
            if int(seg.get("physical_page", 0) or 0) != page_no:
                continue
            skey = _heading_key(seg.get("text", ""))
            sstrip = _heading_key_without_number(seg.get("text", ""))
            score = 0
            if ekey and skey == ekey:
                score = 3
            elif estrip and (skey == estrip or sstrip == estrip):
                score = 2
            elif ekey and sstrip == ekey:
                score = 1
            if score:
                candidates.append((idx, seg, score))
        if not candidates:
            continue
        best_idx, _, _ = min(
            candidates,
            key=lambda pair: (
                -pair[2],
                float(pair[1]["bbox"][0]),
                float(pair[1]["bbox"][1]),
                -float(pair[1].get("font_size", 0.0)),
            ),
        )
        level = max(1, int(entry.get("level", 1) or 1))
        selected_outline_levels.setdefault(best_idx, level)
        selected_outline_entries.setdefault(best_idx, entry)

    def outline_level_for(index: int) -> int | None:
        return selected_outline_levels.get(index)

    def structural_level_for(index: int) -> int | None:
        # Printed TOC reflects the visible document structure and therefore
        # wins over a stale/inconsistent bookmark level.
        return toc_level_for(index) if toc_level_for(index) is not None else outline_level_for(index)

    def structural_evidence_for(index: int) -> tuple[float, list[str], str]:
        evidence: list[str] = []
        confidence = 0.80
        source_heading = ""
        if index in selected_toc_entries:
            evidence.append("visible_toc")
            confidence = max(confidence, 0.99)
            source_heading = selected_toc_entries[index].get("text", "")
        if index in selected_outline_entries:
            evidence.append("pdf_outline")
            confidence = max(confidence, 0.98)
            if not source_heading:
                source_heading = selected_outline_entries[index].get("text", "")
        return confidence, evidence, source_heading

    # Large recurring top headings are retained once.
    top_counts: Counter[str] = Counter()
    for seg in segments:
        txt = normalise(seg.get("text", ""))
        h = float(seg.get("logical_height", 1.0) or 1.0)
        y0 = float(seg.get("bbox", (0, 0, 0, 0))[1])
        if txt and y0 <= h * 0.12 and len(txt.split()) <= 12:
            top_counts[_heading_key(txt)] += 1
    repeated_top_headings = {t for t, n in top_counts.items() if n >= 2}
    seen_repeated_headings: set[str] = set()

    first_page_segments = [s for s in segments if s.get("logical_page") == 1]
    first_page_words = sum(len(normalise(s.get("text", "")).split()) for s in first_page_segments)
    cover_like_first_page = (
        bool(first_page_segments)
        and first_page_words <= 45
        and max((s.get("font_size", 0.0) for s in first_page_segments), default=0.0)
            >= body_size * 1.45
    )

    def has_following_body(
        index: int, seg: dict, *, require_alignment: bool, max_body_ratio: float
    ) -> bool:
        """Return True when a candidate heading is followed by genuine prose/list body.

        The test is geometry-based rather than relying only on reconstructed reading
        order. Short bold infographic labels are skipped. For centred display
        subheadings, horizontal overlap with the following prose is accepted even
        when the left edges do not align.
        """
        x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
        page = int(seg.get("logical_page", 0) or 0)
        h = float(seg.get("logical_height", 1.0) or 1.0)
        width = float(seg.get("logical_width", 1.0) or 1.0)
        size = float(seg.get("font_size", 0.0))
        max_gap = max(90.0, size * 4.0)

        spatial: list[tuple[float, dict]] = []
        for cand in segments:
            cpage = int(cand.get("logical_page", 0) or 0)
            if cpage != page:
                continue
            cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
            if cy0 < y1 - 1.5:
                continue
            gap = cy0 - y1
            if gap > max_gap:
                continue
            spatial.append((gap, cand))

        for gap, cand in sorted(spatial, key=lambda pair: (pair[0], pair[1]["bbox"][0])):
            ctext = normalise(cand.get("text", ""))
            if not ctext or lone_page.fullmatch(ctext):
                continue
            csize = float(cand.get("font_size", 0.0))
            cbold = bool(cand.get("bold"))
            ckey = _heading_key(ctext)
            if ckey in toc_lookup or _heading_key_without_number(ctext) in toc_lookup or ckey in universal_sections:
                continue
            if csize > body_size * max_body_ratio:
                continue
            # Infographic/callout labels are commonly short and emphasised.
            if cbold and len(ctext.split()) <= 16:
                continue
            if len(ctext) < 24 and len(ctext.split()) < 5:
                continue

            cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
            aligned = abs(cx0 - x0) <= max(18.0, width * 0.07)
            overlap = max(0.0, min(x1, cx1) - max(x0, cx0))
            min_span = max(1.0, min(x1 - x0, cx1 - cx0))
            overlaps = (overlap / min_span) >= 0.20
            if (require_alignment and aligned) or (not require_alignment and (aligned or overlaps)):
                return True

        # A heading near the bottom of a page may introduce body text on the
        # next page. Keep this conservative and alignment-based.
        if y0 >= h * 0.70:
            next_page = page + 1
            for cand in segments:
                if int(cand.get("logical_page", 0) or 0) != next_page:
                    continue
                ctext = normalise(cand.get("text", ""))
                if not ctext or len(ctext) < 24:
                    continue
                cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
                if cy0 > float(cand.get("logical_height", h)) * 0.22:
                    continue
                if float(cand.get("font_size", 0.0)) > body_size * max_body_ratio:
                    continue
                if bool(cand.get("bold")) and len(ctext.split()) <= 16:
                    continue
                aligned = abs(cx0 - x0) <= max(18.0, width * 0.08)
                if aligned:
                    return True
        return False

    # Active hierarchy is also used to disambiguate single-letter Roman
    # numerals (e.g. I.) from alphabetic subheadings (e.g. C.).
    heading_stack: dict[int, str] = {}
    # Strong structural ancestry is tracked separately from local typographic
    # subheads. Explicit Roman/alphabetic/decimal headings should not become
    # children of a chart title or other inferred display label.
    structural_stack: dict[int, str] = {}

    # Short, same-style labels placed side-by-side are usually panel/comparison
    # labels rather than document sections. Suppress only moderate-size labels;
    # very large paired headings can legitimately be separate spread sections.
    paired_panel_labels: set[int] = set()
    for i, a in enumerate(segments):
        at = normalise(a.get("text", ""))
        if not at or len(at.split()) > 4:
            continue
        asize = float(a.get("font_size", 0.0))
        if asize < body_size * 1.35 or asize >= body_size * 2.0:
            continue
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if int(a.get("physical_page", 0) or 0) != int(b.get("physical_page", 0) or 0):
                continue
            bt = normalise(b.get("text", ""))
            if not bt or len(bt.split()) > 4:
                continue
            bsize = float(b.get("font_size", 0.0))
            if abs(asize - bsize) > 0.6 or bool(a.get("bold")) != bool(b.get("bold")):
                continue
            ay = float(a["bbox"][1]); by = float(b["bbox"][1])
            if abs(ay - by) > max(8.0, asize * 0.65):
                continue
            ax = (float(a["bbox"][0]) + float(a["bbox"][2])) / 2.0
            bx = (float(b["bbox"][0]) + float(b["bbox"][2])) / 2.0
            width = max(float(a.get("logical_width", 1.0)), float(b.get("logical_width", 1.0)))
            if abs(ax - bx) >= width * 0.45:
                paired_panel_labels.update({i, j})

    # Repeated short emphasis blocks arranged across the same horizontal band are
    # usually infographic/table labels, not section headings. This catches, for
    # example, three labelled boxes across a diagram without hard-coding words.
    graphic_row_labels: set[int] = set()
    by_graphic_page: dict[int, list[int]] = {}
    for i, seg in enumerate(segments):
        txt = normalise(seg.get("text", ""))
        size = float(seg.get("font_size", 0.0))
        if (
            bool(seg.get("bold"))
            and 1 <= len(txt.split()) <= 12
            and body_size * 0.95 <= size <= body_size * 1.30
        ):
            by_graphic_page.setdefault(int(seg.get("logical_page", 0) or 0), []).append(i)

    for lp, indices in by_graphic_page.items():
        indices = sorted(indices, key=lambda i: float(segments[i]["bbox"][1]))
        for pos, i in enumerate(indices):
            a = segments[i]
            ay = float(a["bbox"][1])
            band = [i]
            for j in indices[pos + 1:]:
                by = float(segments[j]["bbox"][1])
                if by - ay > max(26.0, body_size * 2.8):
                    break
                if abs(float(segments[j].get("font_size", 0.0)) - float(a.get("font_size", 0.0))) <= 1.0:
                    band.append(j)
            if len(band) < 3:
                continue
            centres = [
                (float(segments[j]["bbox"][0]) + float(segments[j]["bbox"][2])) / 2.0
                for j in band
            ]
            width = float(a.get("logical_width", 1.0) or 1.0)
            if max(centres) - min(centres) >= width * 0.28:
                graphic_row_labels.update(band)

    # Dominant body-text left edges for each logical page. Same-size bold
    # headings aligned to these lanes are likely semantic subsections; graphical
    # callouts positioned in a separate panel are not.
    body_lanes: dict[int, list[float]] = {}
    by_page: dict[int, list[tuple[float, int]]] = {}
    for cand in segments:
        ctext = normalise(cand.get("text", ""))
        csize = float(cand.get("font_size", 0.0))
        if (
            len(ctext) >= 24
            and not bool(cand.get("bold"))
            and body_size * 0.86 <= csize <= body_size * 1.14
        ):
            lp = int(cand.get("logical_page", 0) or 0)
            by_page.setdefault(lp, []).append((float(cand["bbox"][0]), min(len(ctext), 220)))

    for lp, vals in by_page.items():
        clusters: list[list[float]] = []  # weighted x, weight
        for x, weight in sorted(vals):
            if not clusters or abs(x - clusters[-1][0]) > 20.0:
                clusters.append([x, float(weight)])
            else:
                ox, ow = clusters[-1]
                nw = ow + weight
                clusters[-1] = [(ox * ow + x * weight) / nw, nw]
        strongest = sorted(clusters, key=lambda c: c[1], reverse=True)[:2]
        body_lanes[lp] = [c[0] for c in strongest]

    def near_primary_body_lane(seg: dict) -> bool:
        lp = int(seg.get("logical_page", 0) or 0)
        lanes = body_lanes.get(lp, [])
        if not lanes:
            return False
        x0 = float(seg["bbox"][0])
        width = float(seg.get("logical_width", 1.0) or 1.0)
        return min(abs(x0 - lane) for lane in lanes) <= max(20.0, width * 0.055)

    # Callout/infographic regions are preserved separately from prose so their
    # text does not get interleaved with neighbouring paragraph sections.
    callout_indices: set[int] = set(graphic_row_labels) | set(paired_panel_labels)
    far_emphasis_by_page: dict[int, list[int]] = {}
    for idx, cand in enumerate(segments):
        if structural_level_for(idx) is not None:
            continue
        ctext = normalise(cand.get("text", ""))
        if not ctext or len(ctext.split()) > 28 or not bool(cand.get("bold")):
            continue
        csize = float(cand.get("font_size", 0.0))
        if csize < body_size * 0.98:
            continue
        if near_primary_body_lane(cand):
            continue
        lp = int(cand.get("logical_page", 0) or 0)
        far_emphasis_by_page.setdefault(lp, []).append(idx)

    for lp, indices in far_emphasis_by_page.items():
        # Require a genuine graphical panel, not one isolated side heading.
        if len(indices) >= 3:
            callout_indices.update(indices)

    def has_heading_whitespace(seg: dict) -> bool:
        """Check for a visual break before a same-size local subheading."""
        x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
        lp = int(seg.get("logical_page", 0) or 0)
        h = float(seg.get("logical_height", 1.0) or 1.0)
        width = float(seg.get("logical_width", 1.0) or 1.0)
        prev: list[float] = []
        for cand in segments:
            if int(cand.get("logical_page", 0) or 0) != lp or cand is seg:
                continue
            cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
            if cy1 > y0 + 1.0:
                continue
            aligned = abs(cx0 - x0) <= max(20.0, width * 0.055)
            overlap = max(0.0, min(x1, cx1) - max(x0, cx0))
            min_span = max(1.0, min(x1 - x0, cx1 - cx0))
            if aligned or (overlap / min_span) >= 0.35:
                prev.append(y0 - cy1)
        if not prev:
            return y0 <= h * 0.35
        return min(prev) >= max(7.5, body_size * 0.75)

    def has_immediate_following_body(seg: dict) -> bool:
        """Stricter body test used to rescue a real heading inside a graphic row."""
        x0, y0, x1, y1 = [float(v) for v in seg["bbox"]]
        lp = int(seg.get("logical_page", 0) or 0)
        width = float(seg.get("logical_width", 1.0) or 1.0)
        for cand in segments:
            if int(cand.get("logical_page", 0) or 0) != lp or cand is seg:
                continue
            cx0, cy0, cx1, cy1 = [float(v) for v in cand["bbox"]]
            gap = cy0 - y1
            if gap < -1.0 or gap > max(14.0, body_size * 1.4):
                continue
            ctext = normalise(cand.get("text", ""))
            if bool(cand.get("bold")) or len(ctext) < 30:
                continue
            if not (body_size * 0.95 <= float(cand.get("font_size", 0.0)) <= body_size * 1.14):
                continue
            aligned = abs(cx0 - x0) <= max(18.0, width * 0.07)
            if aligned:
                return True
        return False

    def follows_structural_heading(index: int) -> bool:
        """True when the immediately preceding meaningful segment is a structural heading."""
        for j in range(index - 1, max(-1, index - 4), -1):
            prev = segments[j]
            if int(prev.get("logical_page", 0) or 0) != int(segments[index].get("logical_page", 0) or 0):
                break
            ptext = normalise(prev.get("text", ""))
            if not ptext or lone_page.fullmatch(ptext):
                continue
            if structural_level_for(j) is not None:
                return True
            pdepth = _numeric_heading_depth(ptext)
            if pdepth is not None and len(ptext.split()) <= 24:
                return True
            return False
        return False

    # Deep local headings should be siblings beneath the latest TOC/major
    # structural anchor, not children of the immediately preceding local heading.
    active_anchor_level = 0

    def heading_level(index: int, seg: dict) -> int | None:
        text = normalise(seg.get("text", ""))
        if not text or lone_page.fullmatch(text) or text.startswith("•"):
            return None
        if cover_like_first_page and seg.get("logical_page") == 1:
            return None

        # Typed content blocks are never allowed to enter the structural stack.
        # They remain available downstream as lists/tables/figures/footnotes.
        if seg.get("block_type") in {
            "table_row", "table_header", "list_item", "footnote",
            "figure_text", "figure_title", "figure_caption",
        }:
            return None

        lower_early = _heading_key(text)
        if structural_level_for(index) is None and lower_early not in universal_sections:
            if index in paired_panel_labels:
                return None
            if index in graphic_row_labels:
                if not (near_primary_body_lane(seg) and has_immediate_following_body(seg)):
                    return None
            elif index in callout_indices:
                return None
        wc = len(text.split())
        if wc > 30:
            return None

        lower = _heading_key(text)
        size = float(seg.get("font_size", 0.0))
        emphasis = bool(seg.get("bold"))
        h = float(seg.get("logical_height", 1.0) or 1.0)
        y0 = float(seg.get("bbox", (0, 0, 0, 0))[1])
        physical_page = int(seg.get("physical_page", 0) or 0)

        # The Contents heading itself is valid, but entries printed on the TOC
        # page are not body-section starts.
        exact_label = normalise(text).casefold().rstrip(":").strip()
        if physical_page in toc_pages and exact_label not in {"contents", "table of contents"}:
            return None

        # Explicit decimal numbering is the strongest hierarchy signal in the
        # body. It resolves structures such as 4 -> 4.1 -> 4.1.1 even when the
        # TOC is visually flat or bookmark levels are inconsistent.
        numeric_depth = _numeric_heading_depth(text)
        structural_level = structural_level_for(index)
        if numeric_depth is not None and numeric_depth > 1:
            return numeric_depth
        if numeric_depth == 1 and structural_level == 1:
            return 1

        # High-confidence author-supplied structure: visible TOC first, then PDF
        # outline/bookmarks for headings not represented by the visible TOC.
        if structural_level is not None:
            return structural_level

        if lower in universal_sections and wc <= 10:
            return 1

        # Explicit nested report numbering. These are useful even when the TOC
        # only lists the parent section.
        if emphasis and size >= body_size * 0.95:
            if roman_lower.match(text):
                return 5
            if letter_upper.match(text):
                prefix = text.split(".", 1)[0]
                # A single Roman-looking letter is a Roman section marker when
                # there is no active level-3 Roman parent yet (e.g. "I. Overall
                # Impact..."). Under an active level-3 parent, the same shape is
                # an alphabetic child (e.g. "C. Setting Institutional...").
                if prefix in {"I", "V", "X", "L", "C", "D", "M"} and 3 not in heading_stack:
                    return 3
                return 4
            if roman_upper.match(text):
                return 3

        if numbered_heading.match(text):
            # When a TOC exists, numbered headings declared by it have already
            # been selected above. A second unmatched numbered label on the
            # same page is usually an infographic / process-diagram duplicate.
            if toc_entries:
                return None
            if emphasis and size >= body_size * 1.03 and wc <= 24:
                return 2
            return None

        # With a TOC available, TOC-selected segments define the major
        # hierarchy. Non-TOC typography can still introduce *local* subsections,
        # but it must have genuine following body text. Extremely oversized text
        # is treated as a statistical/graphic callout unless the TOC selected it.
        if toc_entries:
            if size >= body_size * 2.80:
                return None

            root_key = _heading_key(heading_stack.get(1, ""))
            blocked_root = root_key in {"acknowledgements", "acknowledgments", "references", "contents"}
            if active_anchor_level and not blocked_root:
                # Same-size/near-body bold labels can be real semantic subheads.
                # Keep this deliberately strict: require whitespace, prose-lane
                # alignment, genuine following body, at least two words, and
                # reject common personal-name/byline shapes.
                personal_label = bool(
                    "," in text
                    or re.match(r"^(?:prof(?:essor)?|dr|mr|mrs|ms|miss)\.?\s+", text, re.IGNORECASE)
                )
                if (
                    1 <= wc <= 9
                    and not personal_label
                    and emphasis
                    and body_size * 0.94 <= size <= body_size * 1.35
                    and not text.endswith(".")
                    and near_primary_body_lane(seg)
                    and (has_heading_whitespace(seg) or follows_structural_heading(index))
                    and has_following_body(index, seg, require_alignment=True, max_body_ratio=1.16)
                ):
                    return min(6, active_anchor_level + 1)

                # Moderate-size local headings retain the older conservative
                # size-step rule. This is important for reports where H2/H3
                # labels such as "Writing" or "Flooring" are larger than body
                # text but are not present in the TOC.
                if (
                    wc <= 16
                    and emphasis
                    and (size - body_size) >= 1.8
                    and size <= body_size * 1.40
                    and not text.endswith(".")
                    and near_primary_body_lane(seg)
                    and has_following_body(index, seg, require_alignment=True, max_body_ratio=1.18)
                ):
                    return min(6, active_anchor_level + 1)

                # Designed feedback/recommendation reports often use centred
                # quotation-style subsection titles. Quotation punctuation is
                # a strong semantic signal; require substantial prose beneath.
                quoted = text.startswith(("“", '"', "‘", "'")) and text.endswith(("”", '"', "’", "'"))
                if (
                    quoted
                    and wc <= 24
                    and emphasis
                    and body_size * 1.30 <= size < body_size * 2.80
                    and has_following_body(index, seg, require_alignment=False, max_body_ratio=1.18)
                ):
                    return min(6, active_anchor_level + 1)

            # A non-TOC major heading is only inferred when no structural anchor
            # is active yet. Once a TOC-defined section is active, unmatched
            # display text is much more likely to be a callout/diagram label.
            if (
                not active_anchor_level
                and wc <= 20
                and y0 <= h * 0.16
                and size >= body_size * 1.40
                and emphasis
                and has_following_body(index, seg, require_alignment=False, max_body_ratio=1.45)
            ):
                return 1
            return None

        # TOC-free fallback for reports such as designed brochures.
        if wc <= 20 and size >= body_size * 1.55 and (
            emphasis or size >= body_size * 2.10
        ):
            if has_following_body(index, seg, require_alignment=False, max_body_ratio=1.45):
                return 1

        if wc <= 16 and emphasis and size >= body_size * 1.06:
            if text.endswith("."):
                return None
            if has_following_body(index, seg, require_alignment=True, max_body_ratio=1.22):
                return 2

        return None

    def is_margin_page_number(seg: dict, text: str) -> bool:
        """Drop numeric folios only when they are actually positioned in a page margin."""
        if not (
            lone_page.fullmatch(text)
            or re.fullmatch(r"\d{1,3}\s+\d{1,3}", text)
        ):
            return False
        y0, y1 = float(seg["bbox"][1]), float(seg["bbox"][3])
        h = float(seg.get("logical_height", 1.0) or 1.0)
        return y0 <= h * 0.07 or y1 >= h * 0.92

    tagged: list[dict] = []

    for seg_index, seg in enumerate(segments):
        text = normalise(seg.get("text", ""))
        if not text or len(text) < 2:
            continue
        if is_margin_page_number(seg, text):
            continue

        level = heading_level(seg_index, seg)
        if level is not None:
            key = _heading_key(text)
            if key in repeated_top_headings:
                if key in seen_repeated_headings:
                    continue
                seen_repeated_headings.add(key)

            # Avoid impossible hierarchy jumps if a document starts mid-level.
            if heading_stack:
                max_existing = max(heading_stack)
                level = min(level, max_existing + 1)
            elif level > 1:
                level = 1

            # Update the structural anchor only for author-supplied/explicit
            # structure (visible TOC, PDF outline, decimal/Roman/alphabetic
            # numbering) or universal major sections. Local typographic subheads
            # must not become authoritative ancestors.
            explicit_nested = bool(
                _numeric_heading_depth(text) is not None
                or roman_lower.match(text)
                or roman_upper.match(text)
                or letter_upper.match(text)
            )
            strong_structural = bool(
                structural_level_for(seg_index) is not None
                or explicit_nested
                or (_heading_key(text) in universal_sections and level == 1)
            )
            if strong_structural:
                active_anchor_level = level

            # Explicit hierarchy markers take their parent from the strong
            # structural stack, not from an inferred chart/figure/local label.
            parent_source = structural_stack if explicit_nested else heading_stack
            parent = ""
            for parent_level in range(level - 1, 0, -1):
                if parent_level in parent_source:
                    parent = parent_source[parent_level]
                    break

            for k in list(heading_stack):
                if k >= level:
                    del heading_stack[k]
            heading_stack[level] = text.rstrip()

            if strong_structural:
                for k in list(structural_stack):
                    if k >= level:
                        del structural_stack[k]
                structural_stack[level] = text.rstrip()

            confidence, evidence, source_heading = structural_evidence_for(seg_index)
            if _numeric_heading_depth(text) is not None:
                evidence.append("explicit_numbering")
                confidence = max(confidence, 0.99)
            if _heading_key(text) in universal_sections:
                evidence.append("universal_section_name")
                confidence = max(confidence, 0.93)
            if not evidence:
                evidence.extend(["typography", "geometry"])
                confidence = max(confidence, 0.84)

            tagged.append({
                "kind": "heading",
                "text": text.rstrip(),
                "level": level,
                "parent_heading": parent,
                "heading_confidence": round(min(confidence, 0.999), 3),
                "heading_evidence": sorted(set(evidence)),
                "source_heading": source_heading,
                "physical_page": int(seg.get("physical_page", 0) or 0),
                "logical_page": int(seg.get("logical_page", 0) or 0),
            })
        else:
            block_type = seg.get("block_type")
            kind = "content"
            if block_type in {
                "table_row", "table_header", "list_item", "footnote",
                "figure_text", "figure_title", "figure_caption",
            }:
                kind = block_type
            elif seg_index in callout_indices:
                kind = "callout"
            tagged.append({
                "kind": kind,
                "text": text,
                "cells": seg.get("cells", []),
                "marker": seg.get("marker", ""),
                "item_text": seg.get("item_text", ""),
                "figure_group": seg.get("figure_group", ""),
                "bbox": tuple(seg.get("bbox", (0, 0, 0, 0))),
                "physical_page": int(seg.get("physical_page", 0) or 0),
                "logical_page": int(seg.get("logical_page", 0) or 0),
            })

    sections_raw: list[dict] = []
    current_heading = ""
    current_level = 0
    current_parent = ""
    current_heading_confidence = 0.0
    current_heading_evidence: list[str] = []
    current_source_heading = ""
    current_chunks: list[str] = []
    current_table_rows: list[dict] = []
    current_callouts: list[str] = []
    current_list_items: list[dict] = []
    current_footnotes: list[str] = []
    current_figures: dict[str, dict] = {}
    current_blocks: list[dict] = []
    physical_pages: set[int] = set()
    logical_pages: set[int] = set()

    def flush() -> None:
        nonlocal current_chunks, current_table_rows, current_callouts, current_list_items
        nonlocal current_footnotes, current_figures, current_blocks
        nonlocal physical_pages, logical_pages
        content = normalise(" ".join(current_chunks))
        figures = []
        for fig in current_figures.values():
            figures.append({
                "title": normalise(fig.get("title", "")),
                "caption": normalise(fig.get("caption", "")),
                "text": normalise(" ".join(fig.get("text", []))),
                "physical_page": fig.get("physical_page"),
                "logical_page": fig.get("logical_page"),
            })
        if (
            content or current_heading or current_callouts or current_table_rows
            or current_list_items or current_footnotes or figures
        ):
            sections_raw.append({
                "heading": current_heading,
                "level": current_level,
                "parent_heading": current_parent,
                "heading_confidence": current_heading_confidence,
                "heading_evidence": list(current_heading_evidence),
                "source_heading": current_source_heading,
                "_text": content,
                "table_rows": list(current_table_rows),
                "callouts": list(current_callouts),
                "list_items": list(current_list_items),
                "footnotes": list(current_footnotes),
                "figures": figures,
                "blocks": list(current_blocks),
                "physical_pages": sorted(p for p in physical_pages if p),
                "logical_pages": sorted(p for p in logical_pages if p),
            })
        current_chunks = []
        current_table_rows = []
        current_callouts = []
        current_list_items = []
        current_footnotes = []
        current_figures = {}
        current_blocks = []
        physical_pages = set()
        logical_pages = set()

    def add_pages(item: dict) -> None:
        if item.get("physical_page"):
            physical_pages.add(item["physical_page"])
        if item.get("logical_page"):
            logical_pages.add(item["logical_page"])

    for item in tagged:
        if item["kind"] == "heading":
            flush()
            current_heading = item["text"]
            current_level = int(item.get("level", 1))
            current_parent = item.get("parent_heading", "")
            current_heading_confidence = float(item.get("heading_confidence", 0.0) or 0.0)
            current_heading_evidence = list(item.get("heading_evidence", []))
            current_source_heading = item.get("source_heading", "")
            add_pages(item)
        elif item["kind"] == "callout":
            current_callouts.append(item["text"])
            current_blocks.append({
                "type": "callout", "text": item["text"],
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            add_pages(item)
        elif item["kind"] in {"table_row", "table_header"}:
            row_block = {
                "type": item["kind"],
                "text": item["text"],
                "cells": list(item.get("cells", [])),
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            }
            current_table_rows.append({
                "type": item["kind"],
                "text": item["text"],
                "cells": list(item.get("cells", [])),
                "physical_page": item.get("physical_page"),
            })
            current_blocks.append(row_block)
            current_chunks.append(item["text"])
            add_pages(item)
        elif item["kind"] == "list_item":
            marker = normalise(str(item.get("marker", "")))
            item_text = normalise(item.get("item_text", ""))
            if not item_text:
                item_text = normalise(re.sub(r"^\s*" + re.escape(marker) + r"\s+", "", item["text"], count=1)) if marker else item["text"]
            current_list_items.append({
                "marker": marker,
                "text": item_text,
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            current_blocks.append({
                "type": "list_item",
                "marker": marker,
                "text": item_text,
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            display = f"{marker} {item_text}".strip()
            current_chunks.append(display)
            add_pages(item)
        elif item["kind"] == "footnote":
            current_footnotes.append(item["text"])
            current_blocks.append({
                "type": "footnote", "text": item["text"],
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            add_pages(item)
        elif item["kind"] in {"figure_text", "figure_title", "figure_caption"}:
            gid = item.get("figure_group") or f"fig-{item.get('logical_page', 0)}-ungrouped"
            fig = current_figures.setdefault(gid, {
                "title": "", "caption": "", "text": [],
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            if item["kind"] == "figure_title":
                fig["title"] = item["text"]
            elif item["kind"] == "figure_caption":
                fig["caption"] = item["text"]
            else:
                fig["text"].append(item["text"])
            current_blocks.append({
                "type": item["kind"], "text": item["text"],
                "figure_group": gid,
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            add_pages(item)
        else:
            current_chunks.append(item["text"])
            current_blocks.append({
                "type": "paragraph",
                "text": item["text"],
                "physical_page": item.get("physical_page"),
                "logical_page": item.get("logical_page"),
            })
            add_pages(item)
    flush()

    sections: list[dict] = []
    for raw in sections_raw:
        content = raw["_text"]
        ppages = raw.get("physical_pages", [])
        lpages = raw.get("logical_pages", [])
        callouts = [normalise(x) for x in raw.get("callouts", []) if normalise(x)]
        footnotes = [normalise(x) for x in raw.get("footnotes", []) if normalise(x)]
        figures = raw.get("figures", [])
        figure_text = [
            normalise(" ".join(filter(None, [f.get("title", ""), f.get("caption", ""), f.get("text", "")])))
            for f in figures
        ]
        figure_text = [x for x in figure_text if x]
        semantic_content = normalise(" ".join([content] + callouts + figure_text + footnotes))
        source_heading = raw.get("source_heading", "")
        body_prefix = _numeric_heading_prefix(raw["heading"])
        source_prefix = _numeric_heading_prefix(source_heading)
        structure_conflict = bool(
            body_prefix and source_prefix and body_prefix != source_prefix
        )
        sections.append({
            "heading": raw["heading"],
            "level": raw.get("level", 0),
            "parent_heading": raw.get("parent_heading", ""),
            "heading_confidence": raw.get("heading_confidence", 0.0),
            "heading_evidence": raw.get("heading_evidence", []),
            "source_heading": source_heading,
            "structure_conflict": structure_conflict,
            "content": content,
            "callouts": callouts,
            "table_rows": raw.get("table_rows", []),
            "list_items": raw.get("list_items", []),
            "figures": figures,
            "footnotes": footnotes,
            "blocks": raw.get("blocks", []),
            "word_count": len(semantic_content.split()),
            "char_count": len(semantic_content),
            "token_count": count_tokens(semantic_content),
            "physical_page_start": ppages[0] if ppages else None,
            "physical_page_end": ppages[-1] if ppages else None,
            "logical_page_start": lpages[0] if lpages else None,
            "logical_page_end": lpages[-1] if lpages else None,
        })

    body_text = normalise(" ".join(
        normalise(" ".join(
            [s.get("content", "")]
            + s.get("callouts", [])
            + [normalise(" ".join(filter(None, [f.get("title", ""), f.get("caption", ""), f.get("text", "")]))) for f in s.get("figures", [])]
            + s.get("footnotes", [])
        ))
        for s in sections
    ))
    return sections, body_text

def extract_metadata(pdf_bytes: bytes) -> dict:
    meta = {"pdf_title": "", "pdf_author": "", "pdf_subject": "", "pdf_creator": ""}
    try:
        info = PdfReader(io.BytesIO(pdf_bytes)).metadata or {}
        for key, field in [("/Title","pdf_title"),("/Author","pdf_author"),
                           ("/Subject","pdf_subject"),("/Creator","pdf_creator")]:
            meta[field] = (info.get(key) or "").strip()
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Unicode-normalise text and remove invisible PDF control characters."""
    text = unicodedata.normalize("NFKC", text or "")
    # PDF exports can contain backspace / zero-width control characters inside
    # TOC lines. They are not semantic content and can break exact matching.
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf"} or ch in "\n\t"
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_sections_from_pdf(pages_text: list[str]) -> tuple[list[dict], str]:
    """
    Walk cleaned PDF page text and split into structured sections,
    each anchored to the heading that precedes it.

    Heading detection heuristics (three tiers):
      1. Numbered: "1. Introduction", "2.3 Methods"
      2. ALL CAPS: "BACKGROUND", "RECOMMENDATIONS:"
      3. Title Case short line: "Introduction", "Key findings", "The broader picture"

    Exclusions:
      - Lines > 12 words
      - Toolkit numbered list items (digit + space + long text)
      - Two merged column headings (e.g. "9. X 10. Y")

    Returns:
      sections  : list of dicts — heading, content, word_count, char_count, token_count
      body_text : flat join for backward compat
    """
    if not pages_text:
        return [], ""

    # ── Step 1: boilerplate detection ──────────────────────────────────────
    line_freq: dict[str, int] = {}
    for page in pages_text:
        for line in page.splitlines():
            s = line.strip()
            if s:
                line_freq[s] = line_freq.get(s, 0) + 1
    boilerplate = {
        line for line, count in line_freq.items()
        if count >= 3 and len(line.split()) <= 10
    }

    lone_page    = re.compile(r"^\d{1,4}$")
    paired_pages = re.compile(r"^\d{1,3}\s+\d{1,3}$")

    # Known structural section names — whitelist for title case detection
    KNOWN_SECTIONS = {
        "introduction", "background", "methodology", "methods", "results",
        "discussion", "conclusion", "conclusions", "recommendations",
        "references", "acknowledgements", "acknowledgments", "abstract",
        "summary", "overview", "appendix", "findings", "analysis",
        "interventions", "testimonials", "limitations", "implications",
        "preface", "glossary", "key findings", "the broader picture",
        "findings in-depth", "connect with us", "about you",
        "a communications toolkit",
    }

    # Numbered heading: period is REQUIRED (not optional) to exclude toolkit items
    numbered_heading     = re.compile(r"^\d{1,2}(\.\d{1,2})*\.\s+[A-Z][a-z].{1,80}$")
    # Pattern to detect a numbered heading at the START of a line
    numbered_start       = re.compile(r"^(\d{1,2}\.\s+(?:[A-Z]\w+\s+){1,10})")
    allcaps_heading      = re.compile(r"^[A-Z][A-Z\s]{3,40}:?$")
    merged_text_headings = re.compile(
        r"^(The broader picture|Key findings|Findings in.depth|Connect with us)"
        r"\s+(Methodology|Introduction|Interventions|Conclusion|Testimonials)",
        re.IGNORECASE
    )
    toolkit_item    = re.compile(r"^\d{1,2}\s+[A-Z]")
    merged_headings = re.compile(r"\d+\.\s+\w.+\d+\.\s+\w")
    body_sentence   = re.compile(r"[,;]$|,\s+\w")

    def is_heading(line: str, seen: set) -> bool:
        if line in seen:
            return False
        wc = len(line.split())
        # Numbered headings can be longer — check them first
        if numbered_heading.match(line) or allcaps_heading.match(line):
            if wc <= 15 and not merged_headings.search(line):
                return True
            return False
        if wc > 8:
            return False
        if merged_headings.search(line):
            return False
        if toolkit_item.match(line) and wc > 2:
            return False
        if body_sentence.search(line):
            return False
        if re.match(r'^(Our|This|They|We|It|In|On|By|For|As|At)\s+\w', line) and wc > 2:
            return False
        return line.lower() in KNOWN_SECTIONS

    # ── Step 1b: pre-split two-column merged lines ──────────────────────────
    # pdfplumber reads across columns and merges e.g.:
    # "5. Vague language ... 6. Using dominant ..." into one line.
    # Also handles: "1. Finding title Body text continues here..."
    numbered_in_line = re.compile(r'\d{1,2}\.\s+[A-Z]')
    # Max words a numbered heading title should be (beyond this = body text mixed in)
    MAX_HEADING_WORDS = 12

    expanded_pages = []
    for page in pages_text:
        new_lines = []
        for line in page.splitlines():
            s = line.strip()
            matches = list(numbered_in_line.finditer(s))
            if len(matches) >= 2:
                # Multiple numbered headings on one line — split at each
                parts = re.split(r'(?=\b\d{1,2}\.\s+[A-Z])', s)
                for part in parts:
                    part = part.strip()
                    if part:
                        new_lines.append(part)
            elif len(matches) == 1 and matches[0].start() == 0:
                # Single numbered heading at start — check if body text is mixed in
                words = s.split()
                if len(words) > MAX_HEADING_WORDS:
                    # Split: first MAX_HEADING_WORDS as heading, rest as body
                    heading_part = " ".join(words[:MAX_HEADING_WORDS])
                    body_part    = " ".join(words[MAX_HEADING_WORDS:])
                    new_lines.append(heading_part)
                    if body_part:
                        new_lines.append(body_part)
                else:
                    new_lines.append(s)
            else:
                new_lines.append(s)
        expanded_pages.append("\n".join(new_lines))
    pages_text = expanded_pages
    tagged: list[tuple[str, str]] = []
    seen_headings: set[str] = set()

    for page in pages_text:
        for line in page.splitlines():
            s = line.strip()
            if not s or s in boilerplate:
                continue
            if lone_page.match(s) or paired_pages.match(s):
                continue
            if len(s) < 3:
                continue
            s = re.sub(r"-\s+", "", s)   # fix hyphenated line-breaks

            # Split merged text headings e.g. "The broader picture Methodology"
            m = merged_text_headings.match(s)
            if m:
                h1 = m.group(1).strip()
                h2 = m.group(2).strip()
                for h in [h1, h2]:
                    if h not in seen_headings:
                        seen_headings.add(h)
                        tagged.append(("heading", h))
                continue

            if is_heading(s, seen_headings):
                seen_headings.add(s)
                tagged.append(("heading", s))
            else:
                tagged.append(("content", s))

    # ── Step 2b: collapse runs of consecutive short headings into content ───
    # Find all maximal runs of consecutive headings where every heading in the
    # run is ≤ 4 words — these are list items (e.g. specialism lists), not sections.
    i = 0
    collapsed: list[tuple[str, str]] = []
    while i < len(tagged):
        kind, text = tagged[i]
        if kind != "heading":
            collapsed.append((kind, text))
            i += 1
            continue
        # Find the full run of consecutive headings from position i
        run_end = i
        while run_end < len(tagged) and tagged[run_end][0] == "heading":
            run_end += 1
        run = tagged[i:run_end]
        # If every item in the run is ≤ 4 words, demote all to content
        if all(len(t.split()) <= 4 for _, t in run) and len(run) > 1:
            for _, t in run:
                collapsed.append(("content", t))
        else:
            collapsed.extend(run)
        i = run_end
    tagged = collapsed

    # ── Step 3: group into sections ─────────────────────────────────────────
    sections_raw: list[dict] = []
    current_heading = ""
    current_chunks: list[str] = []

    def flush(heading: str, chunks: list[str]) -> None:
        text = normalise(" ".join(chunks))
        if text:
            sections_raw.append({"heading": heading, "_text": text})

    for kind, text in tagged:
        if kind == "heading":
            flush(current_heading, current_chunks)
            current_heading = text
            current_chunks  = []
        else:
            current_chunks.append(text)

    flush(current_heading, current_chunks)

    # ── Step 4: attach per-section metadata, dropping trivial sections ─────
    # If a "section" has fewer than 15 words of content, its heading was likely
    # a false positive — demote it by merging into the previous section.
    sections_raw_filtered: list[dict] = []
    for raw in sections_raw:
        text = raw["_text"]
        wc   = len(text.split())
        # Merge trivial sections (< 15 words) into previous section's content
        if wc < 15 and sections_raw_filtered:
            prev = sections_raw_filtered[-1]
            merged = normalise(prev["_text"] + " " + (raw["heading"] + " " if raw["heading"] else "") + text)
            sections_raw_filtered[-1] = {**prev, "_text": merged}
        else:
            sections_raw_filtered.append(raw)

    sections: list[dict] = []
    for raw in sections_raw_filtered:
        content = raw["_text"]
        sections.append({
            "heading":     raw["heading"],
            "content":     content,
            "word_count":  len(content.split()),
            "char_count":  len(content),
            "token_count": count_tokens(content),
        })

    body_text = normalise(" ".join(s["content"] for s in sections))
    return sections, body_text



def _resolved_title(resource: dict, meta: dict) -> str:
    """Prefer curated resource metadata over often-stale embedded PDF properties."""
    return normalise(resource.get("title") or meta.get("pdf_title") or "")


def _resolved_author(resource: dict, meta: dict) -> str:
    """Prefer the catalogue/resource author over the PDF file's creator/owner metadata."""
    return normalise(resource.get("author") or meta.get("pdf_author") or "")


def _resolved_description(resource: dict, meta: dict) -> str:
    return normalise(resource.get("description") or meta.get("pdf_subject") or "")


def _strip_redundant_front_matter(sections: list[dict], resource: dict, meta: dict) -> list[dict]:
    """
    Drop a tiny unheaded cover-page fragment when it simply repeats title/author metadata.
    Meaningful unheaded preambles are preserved.
    """
    if not sections or sections[0].get("heading"):
        return sections

    first = sections[0]
    content = normalise(first.get("content", ""))
    if not content or len(content.split()) > 30:
        return sections

    reference = normalise(" ".join(filter(None, [
        resource.get("title", ""), resource.get("author", ""),
        meta.get("pdf_title", ""), meta.get("pdf_author", ""),
    ])))
    if not reference:
        return sections

    tokenise = lambda s: {t for t in re.findall(r"[a-z0-9]+", s.casefold()) if len(t) >= 3}
    cover_tokens = tokenise(content)
    ref_tokens = tokenise(reference)
    if cover_tokens and len(cover_tokens & ref_tokens) / len(cover_tokens) >= 0.75:
        return sections[1:]
    return sections



def _split_trailing_back_matter(sections: list[dict]) -> list[dict]:
    """
    Split a final contact/copyright leaf away from a substantive section.

    Designed brochures often place a contact/copyright panel on the final
    logical page without giving it a heading. Keeping it as a separate block
    preserves provenance while preventing it from being sent to the tagger.
    """
    if not sections:
        return sections

    max_logical = max((int(s.get("logical_page_end") or 0) for s in sections), default=0)
    if not max_logical:
        return sections

    backmatter_re = re.compile(
        r"(?:©|copyright|all rights reserved|printed\s+(?:in|on|may|january|february|march|april|june|july|august|september|october|november|december)|"
        r"for more information|contact details|alternative format|general enqu|https?://|www\.|@\w|\bemail\b|\be-mail\b|\bisbn\b)",
        re.IGNORECASE,
    )

    out: list[dict] = []
    for sec in sections:
        blocks = list(sec.get("blocks", []))
        if not blocks or int(sec.get("logical_page_end") or 0) != max_logical:
            out.append(sec)
            continue

        final_blocks = [b for b in blocks if int(b.get("logical_page") or 0) == max_logical]
        earlier_blocks = [b for b in blocks if int(b.get("logical_page") or 0) < max_logical]
        if not final_blocks or not earlier_blocks:
            out.append(sec)
            continue

        # Require the final page to look predominantly administrative/contact
        # oriented. This avoids stripping a substantive final paragraph merely
        # because it happens to contain a URL.
        matches = sum(bool(backmatter_re.search(normalise(b.get("text", "")))) for b in final_blocks)
        if matches < max(1, (len(final_blocks) + 1) // 2):
            out.append(sec)
            continue

        keep_texts = [normalise(b.get("text", "")) for b in earlier_blocks if normalise(b.get("text", ""))]
        tail_texts = [normalise(b.get("text", "")) for b in final_blocks if normalise(b.get("text", ""))]
        if not keep_texts or not tail_texts:
            out.append(sec)
            continue

        main = dict(sec)
        main["blocks"] = earlier_blocks
        main["content"] = normalise(" ".join(
            b.get("text", "") for b in earlier_blocks
            if b.get("type") in {"paragraph", "list_item", "table_row", "table_header"}
        ))
        main["callouts"] = [b.get("text", "") for b in earlier_blocks if b.get("type") == "callout"]
        main["footnotes"] = [b.get("text", "") for b in earlier_blocks if b.get("type") == "footnote"]
        main["logical_page_end"] = max(int(b.get("logical_page") or 0) for b in earlier_blocks)
        main["physical_page_end"] = max(int(b.get("physical_page") or 0) for b in earlier_blocks)
        semantic = normalise(" ".join([main.get("content", "")] + main.get("callouts", []) + main.get("footnotes", [])))
        main["word_count"] = len(semantic.split())
        main["char_count"] = len(semantic)
        main["token_count"] = count_tokens(semantic)
        out.append(main)

        tail_text = normalise(" ".join(tail_texts))
        out.append({
            "heading": "",
            "level": 0,
            "parent_heading": "",
            "heading_confidence": 0.0,
            "heading_evidence": [],
            "source_heading": "",
            "structure_conflict": False,
            "content": tail_text,
            "callouts": [],
            "table_rows": [],
            "list_items": [],
            "figures": [],
            "footnotes": [],
            "blocks": final_blocks,
            "word_count": len(tail_text.split()),
            "char_count": len(tail_text),
            "token_count": count_tokens(tail_text),
            "physical_page_start": min(int(b.get("physical_page") or 0) for b in final_blocks),
            "physical_page_end": max(int(b.get("physical_page") or 0) for b in final_blocks),
            "logical_page_start": max_logical,
            "logical_page_end": max_logical,
            "section_type": "back_matter",
            "include_in_llm": False,
        })
    return out


def _classify_section_roles(
    sections: list[dict],
    toc_pages: set[int] | None = None,
    total_pages: int = 0,
) -> list[dict]:
    """
    Add semantic section roles without deleting source content.

    Front matter and a visible table of contents are retained in ``sections`` for
    provenance/coverage, but flagged ``include_in_llm=False`` so repository
    wrappers and duplicated TOC text do not pollute downstream tagging.
    """
    toc_pages = set(toc_pages or set())
    toc_end = max(toc_pages) if toc_pages else 0

    for sec in sections:
        heading = normalise(sec.get("heading", ""))
        key = _heading_key(heading)
        p0 = int(sec.get("physical_page_start") or 0)
        p1 = int(sec.get("physical_page_end") or p0 or 0)

        section_type = "section"
        include = True

        if key in {"contents", "table of contents"}:
            section_type = "toc"
            include = False
        elif key in {"about this report", "reference"} and toc_end and p0 <= toc_end:
            section_type = "front_matter"
            include = False
        elif not heading and toc_end and p1 and p1 <= toc_end:
            section_type = "front_matter"
            include = False
        elif key in {"references", "bibliography"}:
            section_type = "references"
            include = False
        elif key.startswith("appendix"):
            section_type = "appendix"
        elif key in {"acknowledgements", "acknowledgments"}:
            section_type = "acknowledgements"
            include = False
        elif key.startswith("about ") and total_pages and p0 >= max(1, total_pages - 2):
            section_type = "back_matter"
            include = False
        elif sec.get("section_type") == "back_matter":
            section_type = "back_matter"
            include = False
        elif not heading and total_pages and p0 >= max(1, total_pages - 1) and re.search(
            r"(?:©|all rights reserved|alternative format|general enqu|https?://|www\.)",
            normalise(sec.get("content", "")), re.IGNORECASE
        ):
            section_type = "back_matter"
            include = False
        elif heading and not sec.get("content") and not sec.get("callouts") and not sec.get("figures") and not sec.get("table_rows") and not sec.get("list_items"):
            section_type = "container"
            include = False

        sec["section_type"] = section_type
        sec["include_in_llm"] = include

    return sections


def _build_document_profile(
    layout_segments: list[dict],
    sections: list[dict],
    toc_entries: list[dict],
    outline_entries: list[dict],
    total_pages: int,
) -> tuple[dict, list[str]]:
    """Summarise which adaptive parser capabilities were actually used."""
    text_chars = sum(len(normalise(s.get("text", ""))) for s in layout_segments)
    avg_chars = (text_chars / max(1, total_pages)) if total_pages else 0.0
    likely_scanned = bool(total_pages and avg_chars < 90)

    body_size = _estimate_body_font_size(layout_segments) if layout_segments else 0.0
    multicol_pages = 0
    by_page: dict[int, list[dict]] = {}
    for seg in layout_segments:
        by_page.setdefault(int(seg.get("logical_page", 0) or 0), []).append(seg)

    for _, page_segs in by_page.items():
        vals: list[tuple[float, int, float]] = []
        for seg in page_segs:
            txt = normalise(seg.get("text", ""))
            size = float(seg.get("font_size", 0.0))
            if len(txt) < 45 or bool(seg.get("bold")):
                continue
            if body_size and not (body_size * 0.82 <= size <= body_size * 1.20):
                continue
            vals.append((float(seg["bbox"][0]), len(txt), float(seg.get("logical_width", 1.0) or 1.0)))
        if len(vals) < 4:
            continue
        width = vals[0][2]
        xs = sorted(vals, key=lambda v: v[0])
        clusters: list[list[float]] = []
        for x, weight, _ in xs:
            if not clusters or abs(x - clusters[-1][0]) > max(25.0, width * 0.08):
                clusters.append([x, float(weight)])
            else:
                ox, ow = clusters[-1]
                nw = ow + weight
                clusters[-1] = [(ox * ow + x * weight) / nw, nw]
        strongest = sorted(clusters, key=lambda c: c[1], reverse=True)[:2]
        if (
            len(strongest) == 2
            and min(c[1] for c in strongest) >= 140
            and abs(strongest[0][0] - strongest[1][0]) >= width * 0.22
        ):
            multicol_pages += 1

    heading_confs = [
        float(s.get("heading_confidence", 0.0) or 0.0)
        for s in sections
        if s.get("heading") and s.get("section_type") not in {"front_matter", "toc"}
    ]
    avg_heading_conf = sum(heading_confs) / len(heading_confs) if heading_confs else 0.0
    table_rows = sum(len(s.get("table_rows", [])) for s in sections)
    callouts = sum(len(s.get("callouts", [])) for s in sections)
    list_items = sum(len(s.get("list_items", [])) for s in sections)
    figures = sum(len(s.get("figures", [])) for s in sections)
    footnotes = sum(len(s.get("footnotes", [])) for s in sections)
    conflicts = sum(1 for s in sections if s.get("structure_conflict"))

    sources = ["geometry", "typography"]
    if toc_entries:
        sources.append("visible_toc")
    if outline_entries:
        sources.append("pdf_outline")
    if any(_numeric_heading_depth(s.get("heading", "")) for s in sections if s.get("heading")):
        sources.append("explicit_numbering")
    if table_rows:
        sources.append("table_geometry")
    if list_items:
        sources.append("list_geometry")
    if figures:
        sources.append("figure_geometry")
    if footnotes:
        sources.append("footnote_geometry")

    profile = {
        "text_native": bool(layout_segments) and not likely_scanned,
        "likely_scanned_or_low_text": likely_scanned,
        "average_extracted_chars_per_page": round(avg_chars, 1),
        "multi_column_pages_detected": multicol_pages,
        "table_rows_detected": table_rows,
        "callout_blocks_detected": callouts,
        "list_items_detected": list_items,
        "figure_groups_detected": figures,
        "footnotes_detected": footnotes,
        "structure_sources_used": sources,
        "average_heading_confidence": round(avg_heading_conf, 3),
        "structure_conflicts_detected": conflicts,
    }

    warnings: list[str] = []
    if likely_scanned:
        warnings.append("Very little selectable text was found; this PDF may require OCR.")
    if not toc_entries and not outline_entries:
        warnings.append("No author-supplied TOC/bookmarks were available; hierarchy relies on layout/typography.")
    if heading_confs and avg_heading_conf < 0.78:
        warnings.append("Average heading confidence is low; review section boundaries before downstream use.")
    if conflicts:
        warnings.append(
            f"{conflicts} body heading(s) disagree with TOC/bookmark numbering; body text was preserved and the source heading recorded."
        )

    return profile, warnings


def build_clean_text(resource: dict, meta: dict, sections: list[dict]) -> str:
    """Build hierarchy-preserving clean text for the downstream LLM tagger."""
    parts = []

    title = _resolved_title(resource, meta)
    if title:
        parts.append(f"TITLE: {title}")

    author = _resolved_author(resource, meta)
    if author:
        parts.append(f"AUTHOR: {author}")

    description = _resolved_description(resource, meta)
    if description:
        parts.append(f"DESCRIPTION: {description}")

    section_parts = []
    for s in sections:
        if not s.get("include_in_llm", True):
            continue

        content = s.get("content", "")
        figure_lines: list[str] = []
        for fig in s.get("figures", []):
            label = normalise(fig.get("title", "") or fig.get("caption", ""))
            ftext = normalise(fig.get("text", ""))
            if label:
                figure_lines.append(f"FIGURE: {label}")
            if ftext:
                figure_lines.append(f"FIGURE_TEXT: {ftext}")
        if figure_lines:
            content = "\n".join(filter(None, [content, "\n".join(figure_lines)]))

        callouts = [normalise(x) for x in s.get("callouts", []) if normalise(x)]
        if callouts:
            callout_text = "\n".join(f"CALLOUT: {x}" for x in callouts)
            content = "\n".join(filter(None, [content, callout_text]))

        footnotes = [normalise(x) for x in s.get("footnotes", []) if normalise(x)]
        if footnotes:
            footnote_text = "\n".join(f"FOOTNOTE: {x}" for x in footnotes)
            content = "\n".join(filter(None, [content, footnote_text]))

        if s.get("heading"):
            level = int(s.get("level", 1) or 1)
            # TITLE metadata acts as the document's H1, so extracted level 1
            # begins at Markdown H2 and deeper hierarchy is preserved.
            marker = "#" * min(6, max(2, level + 1))
            section_parts.append(f"{marker} {s['heading']}\n{content}")
        elif content:
            section_parts.append(content)

    return "\n\n".join(filter(None, [
        "\n".join(parts),
        "\n\n".join(section_parts),
    ]))


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_resource(resource: dict, allow_flat_fallback: bool = False) -> dict:
    url   = resource["url"]
    title = resource.get("title", "")
    rid   = resource["resource_id"]

    print(f"  [{rid}] {title[:60]}")

    pdf_bytes = fetch_pdf(rid, url)
    if not pdf_bytes:
        return {
            **resource,
            "clean_status":    "download-failed",
            "clean_title":     "", "description": "",
            "sections":        [], "section_headings": [],
            "body_text":       "", "clean_text": None,
            "word_count":      0,  "char_count": 0,
            "token_count":     0,  "route_hint": "single_pass",
            "total_pages":     0,  "pages_extracted": 0,
            "logical_pages":   0,
            "extractor_used":  None, "stage4_build": STAGE4_BUILD,
            "toc_entries_detected": 0, "toc_pages_detected": [],
            "outline_entries_detected": 0,
            "document_profile": {}, "structure_warnings": ["PDF download failed."],
            "pdf_meta": {},
        }

    meta = extract_metadata(pdf_bytes)

    # Layout-aware path first. This retains x/y position and typography, splits
    # likely two-page spreads at the gutter, and only then sectionises text.
    layout_segments, total_pages, logical_pages = extract_layout_with_pymupdf(pdf_bytes)
    extractor = "pymupdf-layout"
    toc_entries: list[dict] = []
    toc_pages: set[int] = set()
    outline_entries: list[dict] = []

    if layout_segments:
        toc_entries, toc_pages = extract_toc_with_pymupdf(pdf_bytes)
        outline_entries = extract_outline_with_pymupdf(pdf_bytes)
        if toc_entries:
            print(f"    [TOC]      {len(toc_entries)} visible entries from page(s): {sorted(toc_pages)}")
        if outline_entries:
            print(f"    [OUTLINE]  {len(outline_entries)} bookmark entries")
        sections, body_text = extract_sections_from_layout(
            layout_segments,
            toc_entries=toc_entries,
            toc_pages=toc_pages,
            outline_entries=outline_entries,
        )
    elif allow_flat_fallback:
        print("    [WARN] Layout extraction unavailable - explicit flat fallback enabled.")
        pages_text, total_pages = extract_with_pdfplumber(pdf_bytes)
        extractor = "pdfplumber-fallback"
        if not pages_text:
            pages_text, total_pages = extract_with_pypdf(pdf_bytes)
            extractor = "pypdf-fallback"
        logical_pages = len(pages_text)
        sections, body_text = extract_sections_from_pdf(pages_text) if pages_text else ([], "")
    else:
        print("    [ERROR] Layout extraction unavailable. Flat fallback is disabled by default.")
        print("            Install PyMuPDF or rerun with --allow-flat-fallback if you explicitly want flat extraction.")
        sections, body_text = [], ""
        extractor = "layout-unavailable"

    if not sections and not body_text:
        return {
            **resource,
            "clean_status":    "extraction-failed",
            "clean_title":     _resolved_title(resource, meta),
            "description":     _resolved_description(resource, meta),
            "sections":        [], "section_headings": [],
            "body_text":       "", "clean_text": None,
            "word_count":      0,  "char_count": 0,
            "token_count":     0,  "route_hint": "single_pass",
            "total_pages":     total_pages, "pages_extracted": 0,
            "logical_pages":   logical_pages,
            "extractor_used":  extractor, "stage4_build": STAGE4_BUILD,
            "toc_entries_detected": len(toc_entries), "toc_pages_detected": sorted(toc_pages),
            "outline_entries_detected": len(outline_entries),
            "document_profile": {}, "structure_warnings": ["No usable text structure was extracted."],
            "pdf_meta": meta,
        }

    sections = _strip_redundant_front_matter(sections, resource, meta)
    sections = _split_trailing_back_matter(sections)
    sections = _classify_section_roles(sections, toc_pages=toc_pages, total_pages=total_pages)
    body_text = normalise(" ".join(
        normalise(" ".join(
            [s.get("content", "")]
            + s.get("callouts", [])
            + [normalise(" ".join(filter(None, [f.get("title", ""), f.get("caption", ""), f.get("text", "")]))) for f in s.get("figures", [])]
            + s.get("footnotes", [])
        ))
        for s in sections
    ))

    document_profile, structure_warnings = _build_document_profile(
        layout_segments,
        sections,
        toc_entries,
        outline_entries,
        total_pages,
    )

    clean_text  = build_clean_text(resource, meta, sections)
    clean_title = _resolved_title(resource, meta)

    char_count  = len(clean_text)
    token_count = count_tokens(clean_text)
    route_hint  = "single_pass" if token_count <= TOKEN_SAFE_LIMIT else "map_reduce"
    section_headings = [s["heading"] for s in sections if s["heading"]]

    return {
        **resource,
        "clean_status":     "success",
        "clean_title":      clean_title,
        "description":      _resolved_description(resource, meta),
        "sections":         sections,
        "section_headings": section_headings,
        "body_text":        body_text,
        "clean_text":       clean_text,
        "word_count":       len(body_text.split()),
        "char_count":       char_count,
        "token_count":      token_count,
        "route_hint":       route_hint,
        "total_pages":      total_pages,
        "pages_extracted":  min(total_pages, MAX_PAGES),
        "logical_pages":    logical_pages,
        "extractor_used":   extractor,
        "stage4_build":     STAGE4_BUILD,
        "toc_entries_detected": len(toc_entries),
        "toc_pages_detected": sorted(toc_pages),
        "outline_entries_detected": len(outline_entries),
        "document_profile": document_profile,
        "structure_warnings": structure_warnings,
        "pdf_meta":         meta,
    }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe_pdfs(targets: list[dict]) -> None:
    """Check page count and metadata for all PDFs (read-only, still caches)."""
    print(f"\nProbing {len(targets)} PDF(s)...\n")
    print(f"{'ID':<6} {'Pages':>6} {'Size':>8}  {'PDF Title':<40} Resource Title")
    print("-" * 110)

    for r in targets:
        rid, url, title = r["resource_id"], r["url"], r.get("title","")[:45]
        pdf_bytes = fetch_pdf(rid, url)
        if not pdf_bytes:
            print(f"{rid:<6} {'ERROR':>6} {'':>8}  {'':40} {title}")
            time.sleep(DELAY)
            continue

        size_kb, total, pdf_title = len(pdf_bytes)//1024, 0, ""
        try:
            reader    = PdfReader(io.BytesIO(pdf_bytes))
            total     = len(reader.pages)
            pdf_title = ((reader.metadata or {}).get("/Title") or "")[:38].strip()
        except Exception as e:
            pdf_title = f"[{e}]"[:38]

        flag = "✓" if total <= MAX_PAGES else f"! capped at {MAX_PAGES}"
        print(f"{rid:<6} {total:>5}p {size_kb:>6}KB  {pdf_title:<40} {title} {flag}")
        time.sleep(DELAY)

    print("-" * 110)
    print(f"Page cap: {MAX_PAGES}. '!' = extraction will be capped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EDI Hub+ Pipeline — Stage 4: PDF Text Extraction"
    )
    parser.add_argument("--input",  default=STAGE2_INPUT_FILE)
    parser.add_argument("--output", default=STAGE4_OUTPUT_FILE)
    parser.add_argument("--test",   metavar="RESOURCE_ID",
                        help="Run on a single resource ID only (e.g. --test 39)")
    parser.add_argument("--probe",  action="store_true",
                        help="Check page counts for all PDFs (read-only)")
    parser.add_argument("--allow-flat-fallback", action="store_true",
                        help="Explicitly allow pdfplumber/pypdf flat extraction if layout extraction fails")
    args = parser.parse_args()

    print("=" * 60)
    print("EDI Hub+ Pipeline — Stage 4: PDF Text Extraction")
    print("=" * 60)
    print(f"BUILD: {STAGE4_BUILD}")
    if fitz is not None:
        version = getattr(fitz, "__version__", getattr(fitz, "VersionBind", "unknown"))
        print(f"PyMuPDF: {version} (layout-aware extraction enabled)")
    elif args.allow_flat_fallback:
        print("[WARN] PyMuPDF is unavailable; explicit flat fallback is enabled.")
    else:
        print("[ERROR] PyMuPDF is unavailable and flat fallback is disabled.")
        print("        Install with: python -m pip install --upgrade PyMuPDF")
        return

    with open(args.input, encoding="utf-8") as f:
        all_resources = json.load(f)

    targets = [r for r in all_resources if r.get("status") == "pdf-skip"]

    folder = ensure_pdf_dir()
    print(f"PDF cache folder: {folder.resolve()}\n")

    if args.probe:
        probe_pdfs(targets)
        return

    if args.test:
        targets = [r for r in targets if r["resource_id"] == args.test]
        if not targets:
            print(f"[ERROR] No pdf-skip resource with id '{args.test}'")
            return

    print(f"Processing {len(targets)} PDF resource(s)...\n")

    results = []
    for i, resource in enumerate(targets):
        result = extract_resource(resource, allow_flat_fallback=args.allow_flat_fallback)
        results.append(result)

        status = result["clean_status"]
        wc     = result.get("word_count", 0)
        pages  = result.get("pages_extracted", 0)
        total  = result.get("total_pages", 0)
        secs   = len(result.get("sections", []))
        extr   = result.get("extractor_used", "-")
        tc     = result.get("token_count", 0)
        route  = result.get("route_hint", "-")
        print(f"    → {status} | {wc} words | {pages}/{total} pages | "
              f"{secs} sections | {tc} tokens | route: {route} | via {extr}\n")

        if i < len(targets) - 1:
            time.sleep(DELAY)

    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok   = sum(1 for r in results if r["clean_status"] == "success")
    fail = sum(1 for r in results if r["clean_status"] != "success")
    print(f"Done. {ok} extracted, {fail} failed → {args.output}")

    # Full preview in --test mode
    if args.test and results:
        r   = results[0]
        sep = "=" * 60
        print(f"\n{sep}\nFULL EXTRACTION OUTPUT\n{sep}")
        print(f"\nTITLE:          {r['clean_title']}")
        print(f"AUTHOR:         {_resolved_author(r, r.get('pdf_meta', {}))}")
        print(f"PAGES:          {r['pages_extracted']}/{r['total_pages']}")
        print(f"EXTRACTOR:      {r['extractor_used']}")
        print(f"WORD COUNT:     {r['word_count']}")
        print(f"CHAR COUNT:     {r['char_count']}")
        print(f"TOKEN COUNT:    {r['token_count']}  (OLMo GPT-NeoX tokeniser)")
        print(f"ROUTE HINT:     {r['route_hint']}  (threshold: {TOKEN_SAFE_LIMIT} tokens)")
        print(f"\nSECTIONS ({len(r.get('sections', []))} found):")
        for i, s in enumerate(r.get("sections", [])):
            label = s["heading"] if s["heading"] else "[preamble]"
            print(f"  {i+1:>3}. {label}")
            print(f"        words: {s['word_count']}  |  chars: {s['char_count']}  |  tokens: {s['token_count']}")
            print(f"        {s['content'][:120]}{'...' if len(s['content']) > 120 else ''}")
        print(f"\n{sep}\nFULL CLEAN_TEXT:\n{sep}")
        print(r.get("clean_text") or "[EMPTY]")
        print(sep)


if __name__ == "__main__":
    main()