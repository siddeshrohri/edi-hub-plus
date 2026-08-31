"""
EDI Hub+ Pipeline — Stage 6 Chunker
=====================================
Rhetorical chunking with section-aligned boundaries and a three-tier
fallback hierarchy. Chunks are always split at a natural language boundary —
never mid-sentence, never mid-paragraph, never mid-section.

Tier 1  — Section boundary   (heading + body = one atomic unit)
Tier 2  — Paragraph boundary (fallback when a section exceeds threshold)
Tier 3  — Sentence boundary  (fallback when a paragraph exceeds threshold)

Outputs a pretty-printed JSON file (stage6_chunks.json) for human inspection.

Usage — inspect chunking for a specific resource:
    python stage6_chunker.py --resource-id 46 --source stage3
    python stage6_chunker.py --resource-id 46 --source stage4
    python stage6_chunker.py --resource-id 46 --source stage5
    python stage6_chunker.py --resource-id 46 --source stage3 --threshold 800

    --source controls which stage JSON to pull the text from:
        stage3  -> stage3_resources.json  (web page clean text)
        stage4  -> stage4_resources.json  (PDF clean text)
        stage5  -> stage5_linked_content.json (all linked pages/PDFs for a resource)

Import into stage6.py:
    from stage6_chunker import chunk_document, Chunk
    chunks = chunk_document(text, token_threshold=800, tokenizer=tok)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage6_chunker")

# ---------------------------------------------------------------------------
# Default file paths (mirror stage6.py constants)
# ---------------------------------------------------------------------------

STAGE3_INPUT  = "stage3_resources.json"
STAGE4_INPUT  = "stage4_resources.json"
STAGE5_INPUT  = "stage5_linked_content.json"
CHUNKS_OUTPUT = "stage6_chunks.json"

# ---------------------------------------------------------------------------
# Public dataclass — one chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single rhetorical chunk ready to be passed to OLMo."""
    index:                int             # 0-based chunk index within the document
    text:                 str             # full text OLMo will see (with label prefix)
    token_count:          int             # approximate token count
    tier:                 int             # 1=section  2=paragraph  3=sentence
    section_title:        Optional[str]   # heading of the section this chunk belongs to
    part_label:           str             # human-readable label prepended to text
    is_continuation:      bool = False    # True if this chunk continues a prior one
    parent_section_part:  int  = 1        # e.g. 2 out of "Part 2 of 3"
    parent_section_total: int  = 1        # total parts this section was split into
    tier_reason:          str  = ""       # why this tier was chosen (logged + stored)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens using the provided HuggingFace tokenizer if available,
    otherwise fall back to word-count * 1.35 approximation.
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            log.warning("Tokenizer failed (%s) — falling back to word approximation", exc)
    return int(len(text.split()) * 1.35)


# ---------------------------------------------------------------------------
# Tier 1 — Section detection
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(r'^\s{0,3}#{1,6}\s+\S', re.MULTILINE),           # Markdown # heading
    re.compile(r'^[A-Z][A-Z\s\d\-:]{3,}$', re.MULTILINE),       # ALL CAPS line
    re.compile(r'^\s*\d+(\.\d+)*[\.\)]\s+\S', re.MULTILINE),    # 1.  / 1.1  / 2.3.1
    re.compile(r'^([A-Z][a-z]+\s+){1,8}[A-Z][a-z]+\s*\n\s*\n', re.MULTILINE),  # Title Case + blank
]
_MAX_HEADING_LEN = 120


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_LEN:
        return False
    for pat in _HEADING_PATTERNS:
        if pat.match(line + "\n"):
            return True
    return False


def _split_into_sections(text: str) -> list[tuple[Optional[str], str]]:
    """
    Split document into (heading, body) pairs.
    A leading block before any heading gets heading=None.
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    current_body:    list[str]     = []

    for line in lines:
        if _is_heading(line):
            body = "".join(current_body).strip()
            if body or current_heading is not None:
                sections.append((current_heading, body))
            current_heading = line.strip()
            current_body    = []
        else:
            current_body.append(line)

    body = "".join(current_body).strip()
    if body or current_heading is not None:
        sections.append((current_heading, body))

    if not sections:
        sections = [(None, text.strip())]

    return sections


# ---------------------------------------------------------------------------
# Tier 2 — Paragraph splitting
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> list[str]:
    raw = re.split(r'\n\s*\n', text)
    return [p.strip() for p in raw if p.strip()]


# ---------------------------------------------------------------------------
# Tier 3 — Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Greedy packer (used at all tiers)
# ---------------------------------------------------------------------------

def _greedy_pack(units: list[str], threshold: int, tokenizer) -> list[list[str]]:
    """
    Pack units into groups where total tokens <= threshold.
    A unit that exceeds threshold on its own gets its own group.
    """
    groups:         list[list[str]] = []
    current_group:  list[str]       = []
    current_tokens: int             = 0

    for unit in units:
        unit_tokens = _count_tokens(unit, tokenizer)
        if current_group and current_tokens + unit_tokens > threshold:
            groups.append(current_group)
            current_group  = [unit]
            current_tokens = unit_tokens
        else:
            current_group.append(unit)
            current_tokens += unit_tokens

    if current_group:
        groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Part-label helper
# ---------------------------------------------------------------------------

def _make_part_label(
    section_title: Optional[str],
    part_idx:      int,
    total_parts:   int,
    tier:          int,
    is_para_cont:  bool = False,
) -> str:
    title = section_title if section_title else "Preamble"
    if total_parts == 1:
        return f"[Section: {title}]"
    label = f"[Section: {title} — Part {part_idx + 1} of {total_parts}"
    if tier == 3 and is_para_cont:
        label += ", Paragraph continued"
    label += "]"
    return label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    text:            str,
    token_threshold: int  = 800,
    tokenizer              = None,
    source_label:    str  = "",   # e.g. "resource 46 / stage3" — for log messages
) -> list[Chunk]:
    """
    Split `text` into rhetorical chunks respecting the three-tier hierarchy.

    Parameters
    ----------
    text            : full document text
    token_threshold : max tokens per chunk
    tokenizer       : HuggingFace tokenizer (or None → word approximation)
    source_label    : descriptive label used only in log messages

    Returns
    -------
    list[Chunk] — ordered, ready for OLMo
    """
    src = f"[{source_label}] " if source_label else ""

    if not text or not text.strip():
        log.warning("%sEmpty text — returning no chunks", src)
        return []

    total_tokens = _count_tokens(text, tokenizer)
    log.info("%sDocument: %d chars | ~%d tokens | threshold: %d",
             src, len(text), total_tokens, token_threshold)

    sections = _split_into_sections(text)
    log.info("%sDetected %d section(s)", src, len(sections))

    raw_chunks: list[Chunk] = []
    chunk_index = 0

    for sec_num, (heading, body) in enumerate(sections, 1):
        section_text   = f"{heading}\n{body}" if heading else body
        section_tokens = _count_tokens(section_text, tokenizer)
        sec_label      = heading or "(preamble)"

        log.debug("%sSection %d/%d: '%s' | %d tokens",
                  src, sec_num, len(sections), sec_label[:60], section_tokens)

        # ── Tier 1: whole section fits ─────────────────────────────────────
        if section_tokens <= token_threshold:
            reason = f"Tier 1 — section fits ({section_tokens} <= {token_threshold} tokens)"
            log.debug("%s  → %s", src, reason)

            part_label   = _make_part_label(heading, 0, 1, tier=1)
            labeled_text = f"{part_label}\n{section_text}"

            raw_chunks.append(Chunk(
                index=chunk_index,
                text=labeled_text,
                token_count=section_tokens,
                tier=1,
                section_title=heading,
                part_label=part_label,
                is_continuation=False,
                parent_section_part=1,
                parent_section_total=1,
                tier_reason=reason,
            ))
            chunk_index += 1
            continue

        # ── Tier 2: section too large — try paragraph splits ───────────────
        log.info("%s  Section '%s' exceeds threshold (%d > %d) — falling back to Tier 2 (paragraphs)",
                 src, sec_label[:60], section_tokens, token_threshold)

        paragraphs = _split_into_paragraphs(body)
        log.debug("%s  Split into %d paragraph(s)", src, len(paragraphs))

        oversized_paras = [
            (i, p) for i, p in enumerate(paragraphs)
            if _count_tokens(p, tokenizer) > token_threshold
        ]

        if not oversized_paras:
            # All paragraphs fit — pack greedily at Tier 2
            groups      = _greedy_pack(paragraphs, token_threshold, tokenizer)
            total_parts = len(groups)
            log.info("%s  Tier 2: packed %d paragraph(s) into %d chunk(s)",
                     src, len(paragraphs), total_parts)

            for part_idx, group in enumerate(groups):
                reason = (
                    f"Tier 2 — section too large ({section_tokens} tokens); "
                    f"packed {len(group)} paragraph(s) into part {part_idx+1}/{total_parts}"
                )
                log.debug("%s    chunk %d: %d para(s), ~%d tokens",
                          src, part_idx+1, len(group),
                          _count_tokens("\n\n".join(group), tokenizer))

                part_label   = _make_part_label(heading, part_idx, total_parts, tier=2)
                group_text   = "\n\n".join(group)
                labeled_text = f"{part_label}\n{group_text}"

                raw_chunks.append(Chunk(
                    index=chunk_index,
                    text=labeled_text,
                    token_count=_count_tokens(labeled_text, tokenizer),
                    tier=2,
                    section_title=heading,
                    part_label=part_label,
                    is_continuation=(part_idx > 0),
                    parent_section_part=part_idx + 1,
                    parent_section_total=total_parts,
                    tier_reason=reason,
                ))
                chunk_index += 1
            continue

        # ── Tier 3: at least one paragraph too large — sentence splits ─────
        log.info("%s  %d paragraph(s) individually exceed threshold — falling back to Tier 3 (sentences)",
                 src, len(oversized_paras))

        all_sentences: list[str] = []
        for i, para in enumerate(paragraphs):
            para_tokens = _count_tokens(para, tokenizer)
            if para_tokens > token_threshold:
                sents = _split_into_sentences(para)
                log.debug("%s    Para %d: %d tokens — split into %d sentence(s)",
                          src, i+1, para_tokens, len(sents))
                all_sentences.extend(sents)
            else:
                all_sentences.append(para)

        groups      = _greedy_pack(all_sentences, token_threshold, tokenizer)
        total_parts = len(groups)
        log.info("%s  Tier 3: packed into %d chunk(s)", src, total_parts)

        for part_idx, group in enumerate(groups):
            reason = (
                f"Tier 3 — paragraph(s) too large even after section split; "
                f"sentence-packed part {part_idx+1}/{total_parts}"
            )
            log.debug("%s    chunk %d: %d sentence(s), ~%d tokens",
                      src, part_idx+1, len(group),
                      _count_tokens(" ".join(group), tokenizer))

            part_label   = _make_part_label(
                heading, part_idx, total_parts, tier=3,
                is_para_cont=(part_idx > 0),
            )
            group_text   = " ".join(group)
            labeled_text = f"{part_label}\n{group_text}"

            raw_chunks.append(Chunk(
                index=chunk_index,
                text=labeled_text,
                token_count=_count_tokens(labeled_text, tokenizer),
                tier=3,
                section_title=heading,
                part_label=part_label,
                is_continuation=(part_idx > 0),
                parent_section_part=part_idx + 1,
                parent_section_total=total_parts,
                tier_reason=reason,
            ))
            chunk_index += 1

    # ── Merge consecutive small Tier-1 chunks to reduce OLMo call count ───
    log.info("%sMerging small adjacent Tier-1 chunks...", src)
    merged = _merge_small_chunks(raw_chunks, token_threshold, tokenizer, src)

    # Re-index
    for i, c in enumerate(merged):
        c.index = i

    log.info("%sFinal chunk count: %d  (was %d before merge)",
             src, len(merged), len(raw_chunks))

    tier_counts = {1: 0, 2: 0, 3: 0}
    for c in merged:
        tier_counts[c.tier] += 1
    log.info("%sTier breakdown — Tier1: %d  Tier2: %d  Tier3: %d",
             src, tier_counts[1], tier_counts[2], tier_counts[3])

    return merged


# ---------------------------------------------------------------------------
# Merge adjacent small Tier-1 chunks
# ---------------------------------------------------------------------------

def _merge_small_chunks(
    chunks:    list[Chunk],
    threshold: int,
    tokenizer,
    src:       str = "",
) -> list[Chunk]:
    if not chunks:
        return chunks

    merged:        list[Chunk] = []
    buffer:        list[Chunk] = []
    buffer_tokens: int         = 0

    for chunk in chunks:
        if chunk.tier != 1:
            if buffer:
                combined = _combine_tier1_buffer(buffer)
                log.debug("%s  Merged %d Tier-1 chunk(s) → 1 chunk (%d tokens)",
                          src, len(buffer), combined.token_count)
                merged.append(combined)
                buffer        = []
                buffer_tokens = 0
            merged.append(chunk)
            continue

        if buffer and buffer_tokens + chunk.token_count > threshold:
            combined = _combine_tier1_buffer(buffer)
            log.debug("%s  Merged %d Tier-1 chunk(s) → 1 chunk (%d tokens)",
                      src, len(buffer), combined.token_count)
            merged.append(combined)
            buffer        = [chunk]
            buffer_tokens = chunk.token_count
        else:
            buffer.append(chunk)
            buffer_tokens += chunk.token_count

    if buffer:
        combined = _combine_tier1_buffer(buffer)
        log.debug("%s  Merged %d Tier-1 chunk(s) → 1 chunk (%d tokens)",
                  src, len(buffer), combined.token_count)
        merged.append(combined)

    return merged


def _combine_tier1_buffer(buffer: list[Chunk]) -> Chunk:
    if len(buffer) == 1:
        return buffer[0]

    combined_text = "\n\n".join(c.text for c in buffer)
    titles        = [c.section_title for c in buffer if c.section_title]
    part_label    = (
        f"[Sections: {' | '.join(titles)}]" if titles else "[Document Preamble]"
    )
    labeled_text  = f"{part_label}\n\n{combined_text}"
    reason        = (
        f"Tier 1 merge — {len(buffer)} small sections combined "
        f"({', '.join(str(c.token_count) for c in buffer)} tokens)"
    )

    return Chunk(
        index=buffer[0].index,
        text=labeled_text,
        token_count=sum(c.token_count for c in buffer),
        tier=1,
        section_title=buffer[0].section_title,
        part_label=part_label,
        is_continuation=False,
        parent_section_part=1,
        parent_section_total=1,
        tier_reason=reason,
    )


# ---------------------------------------------------------------------------
# JSON output writer
# ---------------------------------------------------------------------------

def write_chunks_json(
    resource_id:     str,
    source:          str,
    title:           str,
    url:             str,
    token_threshold: int,
    chunks:          list[Chunk],
    output_path:     str = CHUNKS_OUTPUT,
) -> None:
    """
    Write chunks to a pretty-printed JSON file with a human-readable
    summary block at the very top so numbers are visible on first open.
    Each chunk text is stored as a list of lines for easy reading.
    """
    tier_counts  = {1: 0, 2: 0, 3: 0}
    over_thresh  = []
    tiny_chunks  = []
    token_counts = []

    for c in chunks:
        tier_counts[c.tier] += 1
        token_counts.append(c.token_count)
        if c.token_count > token_threshold:
            over_thresh.append({
                "chunk_number": c.index + 1,
                "token_count":  c.token_count,
                "overage":      c.token_count - token_threshold,
                "reason":       c.tier_reason,
                "label":        c.part_label,
            })
        if c.token_count < 50:
            tiny_chunks.append({
                "chunk_number": c.index + 1,
                "token_count":  c.token_count,
                "label":        c.part_label,
            })

    avg_tokens = int(sum(token_counts) / len(token_counts)) if token_counts else 0

    # Build a plain-text chunk index — one line per chunk, readable at a glance
    chunk_index_lines = []
    for c in chunks:
        cont   = " [CONT]" if c.is_continuation else ""
        over   = f" *** OVER THRESHOLD by {c.token_count - token_threshold}" if c.token_count > token_threshold else ""
        tiny   = " (tiny)" if c.token_count < 50 else ""
        chunk_index_lines.append(
            f"  Chunk {c.index+1:02d} | T{c.tier} | {c.token_count:>5} tokens{cont}{over}{tiny} | {c.part_label[:80]}"
        )

    doc = {
        # ── SUMMARY — visible immediately on open ────────────────────────────
        "SUMMARY": {
            "resource_id":      resource_id,
            "source":           source,
            "title":            title,
            "url":              url,
            "token_threshold":  token_threshold,
            "total_chunks":     len(chunks),
            "tier_breakdown": {
                "tier1_section_level":   tier_counts[1],
                "tier2_paragraph_level": tier_counts[2],
                "tier3_sentence_level":  tier_counts[3],
            },
            "token_stats": {
                "min":     min(token_counts) if token_counts else 0,
                "max":     max(token_counts) if token_counts else 0,
                "average": avg_tokens,
            },
            "warnings": {
                "chunks_over_threshold": len(over_thresh),
                "chunks_under_50_tokens": len(tiny_chunks),
                "over_threshold_detail":  over_thresh  if over_thresh  else "none",
                "tiny_chunk_detail":      tiny_chunks  if tiny_chunks  else "none",
            },
            "chunk_index": chunk_index_lines,
        },
        # ── FULL CHUNK DATA ──────────────────────────────────────────────────
        "chunks": [],
    }

    for c in chunks:
        chunk_dict = asdict(c)
        chunk_dict["chunk_number"]  = c.index + 1    # 1-based for readability
        chunk_dict["over_threshold"] = c.token_count > token_threshold
        chunk_dict["text_lines"]    = c.text.splitlines()
        chunk_dict.pop("text")      # replaced by text_lines
        doc["chunks"].append(chunk_dict)

    output = Path(output_path)
    output.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Chunks written to: %s", output)

    # ── Print readable summary to console ────────────────────────────────────
    _print_chunk_summary(
        resource_id=resource_id,
        source=source,
        title=title,
        token_threshold=token_threshold,
        chunks=chunks,
        output_path=str(output),
        tier_counts=tier_counts,
        token_counts=token_counts,
        over_thresh=over_thresh,
        tiny_chunks=tiny_chunks,
    )


def _print_chunk_summary(
    resource_id:     str,
    source:          str,
    title:           str,
    token_threshold: int,
    chunks:          list,
    output_path:     str,
    tier_counts:     dict,
    token_counts:    list,
    over_thresh:     list,
    tiny_chunks:     list,
) -> None:
    """Print a formatted chunk summary table to the console."""
    W = 72   # line width

    print()
    print("┌" + "─" * (W - 2) + "┐")
    print(f"│  CHUNK SUMMARY — Resource {resource_id} ({source})".ljust(W - 1) + "│")
    print("├" + "─" * (W - 2) + "┤")
    print(f"│  Title     : {title[:W - 16]}".ljust(W - 1) + "│")
    print(f"│  Threshold : {token_threshold} tokens".ljust(W - 1) + "│")
    print(f"│  Output    : {output_path}".ljust(W - 1) + "│")
    print("├" + "─" * (W - 2) + "┤")

    # Tier breakdown
    total = len(chunks)
    t1, t2, t3 = tier_counts[1], tier_counts[2], tier_counts[3]
    mn  = min(token_counts) if token_counts else 0
    mx  = max(token_counts) if token_counts else 0
    avg = int(sum(token_counts) / len(token_counts)) if token_counts else 0

    print(f"│  Total chunks : {total:<5}  (Tier1={t1}  Tier2={t2}  Tier3={t3})".ljust(W - 1) + "│")
    print(f"│  Token stats  : min={mn}  avg={avg}  max={mx}  (threshold={token_threshold})".ljust(W - 1) + "│")

    # Warnings
    warn_lines = []
    if over_thresh:
        for o in over_thresh:
            warn_lines.append(
                f"  ⚠  Chunk {o['chunk_number']:02d} OVER threshold by {o['overage']} tokens ({o['token_count']} total)"
            )
    if tiny_chunks:
        for t in tiny_chunks:
            warn_lines.append(
                f"  ·  Chunk {t['chunk_number']:02d} tiny ({t['token_count']} tokens) — may be noise"
            )
    if warn_lines:
        print("├" + "─" * (W - 2) + "┤")
        print(f"│  WARNINGS".ljust(W - 1) + "│")
        for w in warn_lines:
            print(f"│{w}".ljust(W - 1) + "│")

    # Chunk index table
    print("├" + "─" * (W - 2) + "┤")
    print(f"│  {'#':>3}  {'Tier':4}  {'Tokens':>7}  {'Flags':<10}  Label".ljust(W - 1) + "│")
    print("│" + "  " + "─" * (W - 4) + "  │")

    for c in chunks:
        flags = []
        if c.is_continuation:              flags.append("CONT")
        if c.token_count > token_threshold: flags.append("OVER!")
        if c.token_count < 50:             flags.append("tiny")
        flag_str = ",".join(flags) if flags else ""

        tier_label = f"T{c.tier}"
        # Truncate label to fit
        label_space = W - 32
        label = c.part_label[:label_space] + ("…" if len(c.part_label) > label_space else "")

        row = f"  {c.index+1:>3}  {tier_label:<4}  {c.token_count:>7}  {flag_str:<10}  {label}"
        print(f"│{row}".ljust(W - 1) + "│")

    print("└" + "─" * (W - 2) + "┘")
    print()


# ---------------------------------------------------------------------------
# Content extractors (for CLI --source modes)
# ---------------------------------------------------------------------------

def _load_stage3_text(resource_id: str, path: str) -> tuple[str, str, str]:
    """Returns (title, url, clean_text) from stage3_resources.json."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        if str(r.get("resource_id")) == str(resource_id):
            title = r.get("clean_title") or r.get("title", "")
            url   = r.get("url", "")
            text  = r.get("clean_text", "")
            if not text:
                log.warning("Resource %s found in Stage 3 but clean_text is empty", resource_id)
            return title, url, text
    raise ValueError(f"Resource ID {resource_id} not found in {path}")


def _load_stage4_text(resource_id: str, path: str) -> tuple[str, str, str]:
    """Returns (title, url, clean_text) from stage4_resources.json."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        if str(r.get("resource_id")) == str(resource_id):
            title = r.get("title", "")
            url   = r.get("url", "")
            text  = r.get("clean_text", "")
            if not text:
                log.warning("Resource %s found in Stage 4 but clean_text is empty", resource_id)
            return title, url, text
    raise ValueError(f"Resource ID {resource_id} not found in {path}")


def _load_stage5_documents(resource_id: str, path: str) -> tuple[str, str, list]:
    """
    Returns (resource_title, resource_url, documents) from stage5_linked_content.json.

    Each linked PDF and linked web page is returned as a SEPARATE document dict:
        {
            "doc_id"   : "pdf_1" / "web_1" etc.
            "doc_type" : "pdf" / "webpage"
            "title"    : link text or page title
            "url"      : source url
            "text"     : clean extracted text
        }

    This ensures each linked document is chunked independently — no concatenation.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    for r in records:
        if str(r.get("resource_id")) == str(resource_id):
            resource_title = r.get("resource_title", "")
            resource_url   = r.get("resource_url", "")
            documents      = []

            # Linked PDFs only — each PDF is its own independent document.
            # Linked web pages are intentionally excluded: the parent webpage
            # content comes from Stage 3, and external web links (social media,
            # campus pages etc.) are not EDI guideline content.
            pdf_results = r.get("pdf_link_results", [])
            pdf_ok = [
                x for x in pdf_results
                if x.get("status") == "success" and x.get("extracted")
                and x["extracted"].get("clean_text", "").strip()
            ]
            for i, x in enumerate(pdf_ok, 1):
                ext   = x["extracted"]
                title = x.get("link_text", ext.get("title", f"Linked PDF {i}"))
                documents.append({
                    "doc_id":   f"pdf_{i}",
                    "doc_type": "pdf",
                    "title":    title,
                    "url":      x.get("url", ""),
                    "text":     ext.get("clean_text", ""),
                })

            if not documents:
                log.warning("Resource %s found in Stage 5 but no linked PDFs extracted",
                            resource_id)

            log.info("Resource %s: found %d linked PDF(s) (linked web pages excluded — use Stage 3 for parent content)",
                     resource_id, len(documents))

            return resource_title, resource_url, documents

    raise ValueError(f"Resource ID {resource_id} not found in {path}")


# ---------------------------------------------------------------------------
# Stage 5 multi-document JSON writer
# ---------------------------------------------------------------------------

def _write_stage5_chunks_json(
    resource_id:      str,
    resource_title:   str,
    resource_url:     str,
    token_threshold:  int,
    doc_chunk_groups: list,
    output_path:      str = CHUNKS_OUTPUT,
) -> None:
    """
    Write chunks for a Stage 5 resource to JSON.
    Each linked document (PDF or webpage) has its own chunk group.
    Structure:
        SUMMARY
            resource_id, title, url, threshold
            total_documents, total_chunks
            per_document summary table
        documents[]
            doc_id, doc_type, title, url
            chunk_count, tier_breakdown, token_stats, warnings
            chunks[]
    """
    W = 72

    total_chunks   = sum(len(g["chunks"]) for g in doc_chunk_groups)
    total_docs     = len(doc_chunk_groups)

    # Build per-document summary rows for the SUMMARY block
    doc_summaries  = []
    for g in doc_chunk_groups:
        chunks      = g["chunks"]
        tiers       = {1: 0, 2: 0, 3: 0}
        tok_counts  = []
        over        = []
        tiny        = []
        for c in chunks:
            tiers[c.tier] += 1
            tok_counts.append(c.token_count)
            if c.token_count > token_threshold:
                over.append({"chunk_number": c.index+1, "token_count": c.token_count,
                             "overage": c.token_count - token_threshold})
            if c.token_count < 50:
                tiny.append({"chunk_number": c.index+1, "token_count": c.token_count})

        doc_summaries.append({
            "doc_id":          g["doc_id"],
            "doc_type":        g["doc_type"],
            "title":           g["title"][:70],
            "url":             g["url"],
            "chunk_count":     len(chunks),
            "tier_breakdown":  {"T1": tiers[1], "T2": tiers[2], "T3": tiers[3]},
            "token_stats":     {
                "min": min(tok_counts) if tok_counts else 0,
                "max": max(tok_counts) if tok_counts else 0,
                "avg": int(sum(tok_counts)/len(tok_counts)) if tok_counts else 0,
            },
            "warnings": {
                "over_threshold": len(over),
                "tiny_chunks":    len(tiny),
                "over_detail":    over  if over  else "none",
                "tiny_detail":    tiny  if tiny  else "none",
            },
            "chunk_index": [
                f"  {c.index+1:02d} | T{c.tier} | {c.token_count:>5} tok"
                + (" OVER!" if c.token_count > token_threshold else "")
                + (" CONT"  if c.is_continuation else "")
                + (" tiny"  if c.token_count < 50 else "")
                + f" | {c.part_label[:55]}"
                for c in chunks
            ],
        })

    doc = {
        "SUMMARY": {
            "resource_id":      resource_id,
            "resource_title":   resource_title,
            "resource_url":     resource_url,
            "token_threshold":  token_threshold,
            "total_documents":  total_docs,
            "total_chunks":     total_chunks,
            "note":             "Each linked document is chunked independently. No cross-document concatenation.",
            "documents":        doc_summaries,
        },
        "documents": [],
    }

    for g, ds in zip(doc_chunk_groups, doc_summaries):
        chunk_list = []
        for c in g["chunks"]:
            cd = asdict(c)
            cd["chunk_number"]   = c.index + 1
            cd["over_threshold"] = c.token_count > token_threshold
            cd["text_lines"]     = c.text.splitlines()
            cd.pop("text")
            chunk_list.append(cd)

        doc["documents"].append({
            "doc_id":   g["doc_id"],
            "doc_type": g["doc_type"],
            "title":    g["title"],
            "url":      g["url"],
            "chunks":   chunk_list,
        })

    output = Path(output_path)
    output.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Stage 5 chunks written to: %s", output)

    # ── Console summary ───────────────────────────────────────────────────
    print()
    print("┌" + "─" * (W-2) + "┐")
    print(f"│  STAGE 5 CHUNK SUMMARY — Resource {resource_id}".ljust(W-1) + "│")
    print("├" + "─" * (W-2) + "┤")
    print(f"│  Title      : {resource_title[:W-18]}".ljust(W-1) + "│")
    print(f"│  Threshold  : {token_threshold} tokens".ljust(W-1) + "│")
    print(f"│  Documents  : {total_docs}  |  Total chunks: {total_chunks}".ljust(W-1) + "│")
    print(f"│  Output     : {output_path}".ljust(W-1) + "│")
    print("├" + "─" * (W-2) + "┤")
    print(f"│  {'Doc':<8}  {'Type':<8}  {'Chunks':>6}  {'Min':>5}  {'Avg':>5}  {'Max':>5}  {'Warn'}".ljust(W-1) + "│")
    print("│  " + "─" * (W-4) + "  │")

    for ds in doc_summaries:
        warn = ""
        if ds["warnings"]["over_threshold"]: warn += f"⚠{ds['warnings']['over_threshold']}over "
        if ds["warnings"]["tiny_chunks"]:    warn += f"·{ds['warnings']['tiny_chunks']}tiny"
        row = (
            f"  {ds['doc_id']:<8}  {ds['doc_type']:<8}  "
            f"{ds['chunk_count']:>6}  "
            f"{ds['token_stats']['min']:>5}  "
            f"{ds['token_stats']['avg']:>5}  "
            f"{ds['token_stats']['max']:>5}  "
            f"{warn}"
        )
        print(f"│{row}".ljust(W-1) + "│")
        title_row = f"    └─ {ds['title'][:W-12]}"
        print(f"│{title_row}".ljust(W-1) + "│")

    print("└" + "─" * (W-2) + "┘")
    print(f"\nChunks saved → {output}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Inspect rhetorical chunking for a specific resource."
    )
    parser.add_argument(
        "--resource-id", required=True, metavar="ID",
        help="Resource ID to chunk (e.g. 46)",
    )
    parser.add_argument(
        "--source", required=True,
        choices=["stage3", "stage4", "stage5"],
        help="Which stage JSON to pull text from.",
    )
    parser.add_argument(
        "--threshold", type=int, default=800,
        help="Token threshold per chunk (default: 800)",
    )
    parser.add_argument(
        "--stage3",  default=STAGE3_INPUT,  help=f"Stage 3 JSON path (default: {STAGE3_INPUT})"
    )
    parser.add_argument(
        "--stage4",  default=STAGE4_INPUT,  help=f"Stage 4 JSON path (default: {STAGE4_INPUT})"
    )
    parser.add_argument(
        "--stage5",  default=STAGE5_INPUT,  help=f"Stage 5 JSON path (default: {STAGE5_INPUT})"
    )
    parser.add_argument(
        "--output",  default=CHUNKS_OUTPUT, help=f"Output JSON path (default: {CHUNKS_OUTPUT})"
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="HuggingFace tokenizer name (optional). If omitted, word-count approx is used.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress DEBUG logs — show only INFO and above.",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.INFO)

    # Load tokenizer if specified
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        log.info("Loading tokenizer: %s", args.tokenizer)
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Extract text from the chosen source
    source_path_map = {
        "stage3": args.stage3,
        "stage4": args.stage4,
        "stage5": args.stage5,
    }
    # ── Stage 5: each linked doc chunked independently ───────────────────
    if args.source == "stage5":
        try:
            resource_title, resource_url, documents = _load_stage5_documents(
                args.resource_id, args.stage5
            )
        except (ValueError, FileNotFoundError) as exc:
            log.error("%s", exc)
            sys.exit(1)

        if not documents:
            log.error("No linked documents found for resource %s", args.resource_id)
            sys.exit(1)

        log.info("Resource: [%s] %s", args.resource_id, resource_title[:80])
        log.info("Linked documents: %d", len(documents))

        # Chunk each document independently
        all_doc_chunks = []
        for doc in documents:
            source_label = f"resource {args.resource_id} / {doc['doc_id']} ({doc['doc_type']})"
            log.info("Chunking %s: %s", doc["doc_id"], doc["title"][:60])
            doc_chunks = chunk_document(
                doc["text"],
                token_threshold=args.threshold,
                tokenizer=tok,
                source_label=source_label,
            )
            all_doc_chunks.append({
                "doc_id":   doc["doc_id"],
                "doc_type": doc["doc_type"],
                "title":    doc["title"],
                "url":      doc["url"],
                "chunks":   doc_chunks,
            })

        # Write combined JSON
        _write_stage5_chunks_json(
            resource_id=args.resource_id,
            resource_title=resource_title,
            resource_url=resource_url,
            token_threshold=args.threshold,
            doc_chunk_groups=all_doc_chunks,
            output_path=args.output,
        )
        return

    # ── Stage 3 / Stage 4: single document ───────────────────────────────
    loader_map = {
        "stage3": _load_stage3_text,
        "stage4": _load_stage4_text,
    }

    source_path = source_path_map[args.source]
    loader      = loader_map[args.source]

    try:
        title, url, text = loader(args.resource_id, source_path)
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    if not text.strip():
        log.error("No text content found for resource %s in %s",
                  args.resource_id, args.source)
        sys.exit(1)

    log.info("Resource: [%s] %s", args.resource_id, title[:80])
    log.info("Source:   %s (%s)", args.source, source_path)
    log.info("URL:      %s", url)

    source_label = f"resource {args.resource_id} / {args.source}"
    chunks = chunk_document(
        text,
        token_threshold=args.threshold,
        tokenizer=tok,
        source_label=source_label,
    )

    write_chunks_json(
        resource_id=args.resource_id,
        source=args.source,
        title=title,
        url=url,
        token_threshold=args.threshold,
        chunks=chunks,
        output_path=args.output,
    )


if __name__ == "__main__":
    _cli()