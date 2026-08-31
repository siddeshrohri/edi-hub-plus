"""
EDI Hub+ Pipeline — Stage 6: LLM Tagging + Summarisation
=========================================================
Reads stage3_resources.json, stage4_resources.json, and
stage5_linked_content.json, merges all content per resource,
and runs OLMo-2 7B locally via Ollama to produce structured
tags and a plain-English summary for each resource.

Routing:
  - combined_token_count <= 3,796  →  single pass (one OLMo call)
  - combined_token_count  > 3,796  →  map-reduce  (chunk → mini-summaries → final call)

Map-Reduce chunking strategy (v2):
  1. Structural split  — text is first divided on === SECTION === markers so
     parent content, linked web pages, and linked PDFs are never mixed within
     a single chunk.
  2. Overlap chunking  — within any section that exceeds CHUNK_TOKEN_SIZE,
     chunks are built with a CHUNK_OVERLAP_TOKENS tail carried forward from the
     previous chunk, ensuring boundary-spanning content is seen in full by at
     least one chunk.
  3. Enlarged map summaries — each chunk produces 5-8 bullets (up from 3-5),
     with an explicit instruction to focus on content unique to that chunk and
     not re-summarise the overlapping region.
  4. Reduce budget guard — before the reduce call, combined mini-summary tokens
     are checked against the safe limit; least-informative summaries are dropped
     if the budget would be exceeded.
  5. Chunk documentation — every map-reduce output record includes a full
     'chunks' array so tagging decisions can be cross-verified against the
     exact text each chunk saw.

Output: stage6_tagged.json

Usage:
    python stage6.py                        # full batch
    python stage6.py --test 46             # single resource
    python stage6.py --dry-run             # assemble text only, no OLMo calls
    python stage6.py --model olmo2:latest  # override Ollama model name
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGE3_INPUT   = "stage3_resources.json"
STAGE4_INPUT   = "stage4_resources.json"
STAGE5_INPUT   = "stage5_linked_content.json"
STAGE6_OUTPUT  = "stage6_tagged.json"

OLLAMA_URL     = "http://localhost:11434/api/generate"
DEFAULT_MODEL  = "olmo2:7b"

TOKEN_SAFE_LIMIT            = 3796   # matches Stage 3/4/5 threshold
CHUNK_TOKEN_SIZE            = 3000   # max tokens per map-reduce chunk
CHUNK_OVERLAP_TOKENS        = 175    # tokens carried forward as overlap between chunks
REDUCE_PROMPT_OVERHEAD      = 1200   # conservative token estimate for reduce prompt
                                     # overhead (taxonomy lists + instructions + title/url)
OLLAMA_TIMEOUT              = 300    # seconds — 7B models can be slow on long inputs
MIN_LINKED_CONTENT_TOKENS = 50

# ---------------------------------------------------------------------------
# Taxonomy — full validated tag lists for all four dimensions
# ---------------------------------------------------------------------------

INDIVIDUAL_CHARACTERISTICS = [
    "Age", "Career Stage", "Caring Responsibilities", "Disability",
    "Education Status", "Gender", "Intersectionality", "Job Roles",
    "Linguistic", "Marriage and Civil Partnership", "Nationality",
    "Neurodiversity", "Physical Characteristics", "Pregnancy and Maternity",
    "Race and Ethnicity", "Religion and Belief", "Sex", "Sexual Orientation",
    "Socio-Economic Disadvantage", "Temporary Impairment", "Trans Identity",
]

CAREER_PATHWAY = [
    "Career Mobility", "Career Progression", "Early Career", "Leadership",
    "Mentoring", "Mid-career", "Networking and Collaboration",
    "Recruitment", "Retention", "Visibility and Recognition",
]

ORGANISATIONAL_CULTURE = [
    "Accessibility", "Allyship and Advocacy", "Belonging and Engagement",
    "Bias", "Cultural Competency", "Data Collection and Monitoring",
    "Harassment Bullying and Microaggression", "Inclusive Leadership",
    "Inclusivity", "Representation", "Research Culture", "Research Design",
    "Strategy or Policy", "Work-Life Balance",
]

RESEARCH_FUNDING_PROCESS = [
    "Advertising", "Application Process", "Assessment Process",
    "Bias Mitigation", "Data Collection and Monitoring",
    "Eligibility Criteria", "Funding Guidance",
    "Pre and Post Award Support", "University Selection",
]

RESOURCE_TYPES = [
    "Academic Paper", "Blog Post", "Book Chapter", "Case Study/Testimonial",
    "EDI Intervention", "Evidence Document", "Guidance", "Legislation",
    "People Profile", "Policy Document", "Presentation or Poster",
    "Report or Article", "Resource Database", "Training Material",
]

RESOURCE_FORMATS = [
    "Database", "Dataset", "Document", "Mixed format",
    "Podcast", "Video", "Webinar", "Webpage",
]

SECTORS = [
    "Academia", "Government", "Industry", "International",
    "Learned society", "Not-for-profit", "Other", "Not Specified",
]

ADOPTION_READINESS = [
    "Initial Design", "Needs Identification", "Not applicable",
    "Pilot in Single Setting", "Proof of Concept",
    "Scaling in Multiple Settings", "Widespread Adoption",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an EDI (Equality, Diversity and Inclusion) resource analyst specialising in UK engineering, physical and mathematical sciences research and innovation.
Your task is to tag and summarise resources against a structured taxonomy used by the EDI Hub+ Resource Centre, funded by EPSRC.

Rules:
- Base ALL tagging and summarisation SOLELY on information explicitly present in the provided text.
- Do NOT use prior knowledge, assumptions, or information not present in the resource content.
- If a tag cannot be justified by a specific piece of text in the content, do not apply it.
- If you are unsure whether a tag applies, leave it out — omission is better than guessing.
- For Research Funding Process tags: these refer ONLY to the process of funding research projects by a research funding body such as EPSRC. Do NOT apply recruitment or HR context to this dimension.
- Always respond with valid JSON only. No preamble, no explanation, no markdown fences."""


def build_tagging_prompt(title: str, url: str, text: str) -> str:
    ic  = ", ".join(INDIVIDUAL_CHARACTERISTICS)
    cp  = ", ".join(CAREER_PATHWAY)
    oc  = ", ".join(ORGANISATIONAL_CULTURE)
    rfp = ", ".join(RESEARCH_FUNDING_PROCESS)
    rt  = ", ".join(RESOURCE_TYPES)
    rf  = ", ".join(RESOURCE_FORMATS)
    sec = ", ".join(SECTORS)
    arl = ", ".join(ADOPTION_READINESS)

    return f"""Analyse this EDI resource and return ONLY a JSON object with these exact fields.

CRITICAL GROUNDING RULES:
- This resource is being tagged for the EDI Hub+ Resource Centre, which focuses on equality, diversity and inclusion in UK engineering, physical and mathematical sciences research and innovation.
- Base every tag and every sentence of the summary SOLELY on what is explicitly written in the Resource Content below.
- Do NOT apply a tag because the resource title sounds relevant — the actual content must discuss it.
- Do NOT draw on outside knowledge about the organisation, author, or topic.
- If a tag cannot be traced to a specific piece of text in the content, do not apply it.
- If the content does not mention a dimension at all, return an empty list for that dimension.
- The summary must describe only what the content actually says, not what such a resource typically covers.
- For Research Funding Process: this dimension refers ONLY to the process of funding research projects by a body such as EPSRC. Do NOT apply tags from this dimension based on general recruitment, HR, or people management content.
- Before applying each tag, ask yourself: "Is there a specific sentence or section in this text that directly supports this tag?" If not, exclude it.

Resource Title: {title}
Resource URL: {url}
Resource Content:
{text}

Return this exact JSON structure with no other text:
{{
  "resource_type": "<one value from: {rt}>",
  "resource_format": "<one value from: {rf}>",
  "sector": "<one or more from: {sec} — comma separated>",
  "summary": "<Plain English summary of 80-120 words covering: what this resource is, the key themes and content it addresses, and who it is most useful for. Be specific — reference actual content, frameworks, or topics covered rather than speaking in generalities>",
  "individual_characteristics": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {ic}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "career_pathway": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {cp}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "organisational_culture": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {oc}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "research_funding_process": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {rfp}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "adoption_readiness_level": "<one value from: {arl}>",
  "rationale": "<2-3 sentences explaining the most important tagging decisions. For each key tag applied, quote or closely paraphrase the specific text that justified it>"
}}
CRITICAL: Every tag MUST appear verbatim in the lists provided. Do not invent new tags."""


def build_chunk_summary_prompt(
    chunk_text: str,
    chunk_num: int,
    total_chunks: int,
    section_label: str = "",
    has_overlap: bool = False,
) -> str:
    """
    Prompt for the MAP phase — summarise one chunk.

    Changes from v1:
    - Now requests 5-8 bullets (up from 3-5) to better cover longer chunks.
    - Includes the structural section the chunk came from so OLMo has context.
    - When has_overlap=True, explicitly instructs OLMo to focus on content that
      is new/unique to this chunk and not re-summarise the overlapping region.
    """
    section_note = (
        f"\nThis chunk is from the '{section_label}' section of the resource."
        if section_label else ""
    )
    overlap_note = (
        "\nNOTE: The opening paragraphs of this chunk overlap with the preceding chunk. "
        "Focus your bullets on content that is new or distinct in this chunk — "
        "do not re-summarise points already covered by the overlapping region."
    ) if has_overlap else ""

    return f"""You are summarising part {chunk_num} of {total_chunks} of an EDI resource.{section_note}{overlap_note}

CRITICAL GROUNDING RULES:
- Extract ONLY points that are explicitly stated in the text below.
- Do NOT infer, assume, or add context from outside knowledge.
- If a sentence is not in the text, do not include it.
- Prioritise content that is unique to this chunk — avoid repeating points that
  are likely covered in the overlapping region or in other chunks.

Respond with 5-8 bullet points of the most important EDI-relevant content from
this chunk. Each bullet should be a complete, specific point. No preamble.

Content:
{chunk_text}"""


def build_reduce_prompt(title: str, url: str, mini_summaries: list[str]) -> str:
    """
    Prompt for the REDUCE phase — final tagging from combined mini-summaries.
    Uses a richer 150-250 word summary target and enforces taxonomy-only tags.
    """
    combined = "\n\n".join(
        f"[Section {i+1} summary]\n{s}" for i, s in enumerate(mini_summaries)
    )

    ic  = ", ".join(INDIVIDUAL_CHARACTERISTICS)
    cp  = ", ".join(CAREER_PATHWAY)
    oc  = ", ".join(ORGANISATIONAL_CULTURE)
    rfp = ", ".join(RESEARCH_FUNDING_PROCESS)
    rt  = ", ".join(RESOURCE_TYPES)
    rf  = ", ".join(RESOURCE_FORMATS)
    sec = ", ".join(SECTORS)
    arl = ", ".join(ADOPTION_READINESS)

    return f"""Analyse this EDI resource and return ONLY a JSON object with these exact fields.
The content below is drawn from the main resource page AND all its linked web pages and PDFs.
Use ALL sections to inform your tagging and summary — do not focus only on the first section.

CRITICAL GROUNDING RULES:
- This resource is being tagged for the EDI Hub+ Resource Centre, which focuses on equality, diversity and inclusion in UK engineering, physical and mathematical sciences research and innovation.
- Base every tag and every sentence of the summary SOLELY on what is explicitly stated in the section summaries below.
- Do NOT apply a tag because the resource title sounds relevant — the actual content must discuss it.
- Do NOT draw on outside knowledge about the organisation, author, or topic.
- If a tag cannot be traced to a specific point in the section summaries, do not apply it.
- If a dimension is not mentioned anywhere in the summaries, return an empty list for it.
- The summary must describe only what the summaries actually say, not what such a resource typically covers.
- For Research Funding Process: this dimension refers ONLY to the process of funding research projects by a body such as EPSRC. Do NOT apply tags from this dimension based on general recruitment, HR, or people management content.
- Before applying each tag, ask yourself: "Is there a specific bullet point in these summaries that directly supports this tag?" If not, exclude it.

Resource Title: {title}
Resource URL: {url}
Section Summaries:
{combined}

Return this exact JSON structure with no other text:
{{
  "resource_type": "<one value from: {rt}>",
  "resource_format": "<one value from: {rf}>",
  "sector": "<one or more from: {sec} — comma separated>",
  "summary": "<Plain English summary of 150-250 words. Cover: (1) what this resource is and its origin, (2) the key themes frameworks or tools it provides, (3) what the linked content adds such as supporting PDFs related pages or evidence, (4) who this is most useful for and how they can use it. Be specific and reference actual content — do not speak in generalities>",
  "individual_characteristics": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {ic}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "career_pathway": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {cp}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "organisational_culture": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {oc}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "research_funding_process": {{
    "primary": ["<tags DIRECTLY addressed — choose ONLY from: {rfp}>"],
    "secondary": ["<tags relevant but not the focus — choose ONLY from the same list>"]
  }},
  "adoption_readiness_level": "<one value from: {arl}>",
  "rationale": "<2-3 sentences explaining the most important tagging decisions. For each key tag applied, quote or closely paraphrase the specific section summary text that justified it>"
}}
CRITICAL: Every tag MUST appear verbatim in the lists provided. Do not invent new tags."""


# ---------------------------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------------------------

def load_inputs(s3_path: str, s4_path: str, s5_path: str) -> dict:
    """
    Load all three stage outputs and index them by resource_id.
    Returns a dict keyed by resource_id with all content ready to merge.
    """
    with open(s3_path, encoding="utf-8") as f:
        s3 = {str(r["resource_id"]): r for r in json.load(f)}
    with open(s4_path, encoding="utf-8") as f:
        s4 = {str(r["resource_id"]): r for r in json.load(f)}
    with open(s5_path, encoding="utf-8") as f:
        s5 = {str(r["resource_id"]): r for r in json.load(f)}

    print(f"  Loaded: {len(s3)} web resources (Stage 3)")
    print(f"  Loaded: {len(s4)} PDF resources (Stage 4)")
    print(f"  Loaded: {len(s5)} linked content records (Stage 5)\n")

    return {"stage3": s3, "stage4": s4, "stage5": s5}


def assemble_resource_text(resource_id: str, data: dict) -> dict:
    """
    Merge all content for a single resource into one structured text block.
    Returns a dict with: title, url, combined_text, combined_token_count, route.
    """
    rid  = str(resource_id)
    s3   = data["stage3"]
    s4   = data["stage4"]
    s5   = data["stage5"]

    # ── Identify parent content (Stage 3 web page or Stage 4 PDF) ────────────
    parent = s3.get(rid) or s4.get(rid)
    if not parent:
        return None

    title = parent.get("clean_title") or parent.get("title", "")
    url   = parent.get("url", "")
    parts = []

    # Parent content
    parent_text = parent.get("clean_text", "")
    if parent_text:
        parts.append("=== PARENT CONTENT ===")
        parts.append(parent_text)

    # Stage 5 linked pieces for this resource
    s5_record = s5.get(rid)
    if s5_record:
        # Linked web pages
        ext_results = s5_record.get("external_link_results", [])
        web_pieces = [
            r for r in ext_results
            if r.get("status") == "success"
            and r.get("extracted")
            and count_tokens(r["extracted"].get("clean_text", "")) >= MIN_LINKED_CONTENT_TOKENS
        ]
        if web_pieces:
            parts.append("\n=== LINKED WEB PAGES ===")
            for i, r in enumerate(web_pieces, 1):
                ext = r["extracted"]
                link_title = ext.get("title", r.get("link_text", f"Link {i}"))
                parts.append(f"\n--- Linked page {i}: {link_title} ---")
                parts.append(ext.get("clean_text", ""))

        # Linked PDFs
        pdf_results = s5_record.get("pdf_link_results", [])
        pdf_pieces  = [
            r for r in pdf_results
            if r.get("status") == "success" and r.get("extracted")
        ]
        if pdf_pieces:
            parts.append("\n=== LINKED PDFs ===")
            for i, r in enumerate(pdf_pieces, 1):
                ext = r["extracted"]
                link_title = r.get("link_text", f"PDF {i}")
                parts.append(f"\n--- Linked PDF {i}: {link_title} ---")
                parts.append(ext.get("clean_text", ""))

        combined_token_count = s5_record.get("combined_token_count", 0)
        combined_route       = s5_record.get("combined_route", "single_pass")
    else:
        # Resource has no Stage 5 record — use parent token count only
        combined_token_count = parent.get("token_count", 0)
        combined_route       = (
            "single_pass" if combined_token_count <= TOKEN_SAFE_LIMIT
            else "map_reduce"
        )

    combined_text = "\n\n".join(parts)

    return {
        "resource_id":          rid,
        "title":                title,
        "url":                  url,
        "combined_text":        combined_text,
        "combined_token_count": combined_token_count,
        "route":                combined_route,
    }


# ---------------------------------------------------------------------------
# Tokeniser (same lazy-loaded pattern as Stage 3/4/5)
# ---------------------------------------------------------------------------

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-7B-1124")
    return _tokenizer


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Structural chunking with overlap  (replaces the old flat chunk_text)
# ---------------------------------------------------------------------------

# Regex that matches the === SECTION === markers written by assemble_resource_text
_SECTION_MARKER = re.compile(r"(=== .+? ===)")


def split_into_sections(combined_text: str) -> list[tuple[str, str]]:
    """
    Split combined_text on '=== LABEL ===' markers produced by
    assemble_resource_text.  Returns a list of (label, text) pairs in order.

    Example output:
        [("PARENT CONTENT",    "..."),
         ("LINKED WEB PAGES",  "..."),
         ("LINKED PDFs",       "...")]
    """
    parts = _SECTION_MARKER.split(combined_text)
    sections: list[tuple[str, str]] = []
    current_label = "PREAMBLE"

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if _SECTION_MARKER.fullmatch(stripped):
            # This part IS a marker — update the label for following content
            current_label = stripped.strip("= ").strip()
        else:
            # This part is content belonging to current_label
            sections.append((current_label, stripped))

    return sections


def _build_overlap_tail(paragraphs: list[str], max_tokens: int) -> tuple[str, int]:
    """
    Walk backwards through paragraphs, accumulating text until max_tokens is
    reached.  Returns (overlap_text, actual_token_count).
    """
    tail_paras: list[str] = []
    tail_tokens = 0
    for para in reversed(paragraphs):
        pt = count_tokens(para)
        if tail_tokens + pt > max_tokens:
            break
        tail_paras.insert(0, para)
        tail_tokens += pt
    return "\n\n".join(tail_paras), tail_tokens


def _chunk_section_with_overlap(
    section_text: str,
    section_label: str,
    chunk_size: int = CHUNK_TOKEN_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Split a single section into overlapping paragraph-boundary chunks.

    Each returned dict contains:
        section        — the section label this chunk came from
        text           — the chunk text (including any overlap prefix)
        token_count    — tokens in this chunk
        overlap_tokens — tokens carried forward from the previous chunk
                         (0 for the first chunk of a section)
    """
    paragraphs = [p for p in section_text.split("\n\n") if p.strip()]
    chunks: list[dict] = []

    current_paras: list[str] = []
    current_tokens = 0
    overlap_text   = ""
    overlap_count  = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > chunk_size and current_paras:
            # Emit the current chunk
            chunks.append({
                "section":        section_label,
                "text":           "\n\n".join(current_paras),
                "token_count":    current_tokens,
                "overlap_tokens": overlap_count,
            })

            # Build overlap tail from the chunk we just emitted
            overlap_text, overlap_count = _build_overlap_tail(
                current_paras, overlap_tokens
            )

            # Start the next chunk: overlap prefix + new paragraph
            if overlap_text:
                current_paras  = [overlap_text, para]
                current_tokens = overlap_count + para_tokens
            else:
                current_paras  = [para]
                current_tokens = para_tokens
                overlap_count  = 0
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    # Flush the final chunk
    if current_paras:
        chunks.append({
            "section":        section_label,
            "text":           "\n\n".join(current_paras),
            "token_count":    current_tokens,
            "overlap_tokens": overlap_count,
        })

    return chunks


def chunk_text_structured(
    combined_text: str,
    chunk_size: int = CHUNK_TOKEN_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Two-stage chunking strategy:

    1. Structural split — divide on === SECTION === markers so parent content,
       linked web pages, and linked PDFs are never mixed in a single chunk.
    2. Overlap chunking — within any section that exceeds chunk_size, build
       overlapping chunks carrying CHUNK_OVERLAP_TOKENS of context forward.

    Returns a list of chunk dicts, each with:
        chunk_index    — 1-based global index across all sections
        section        — which top-level section this chunk came from
        text           — exact text OLMo will see
        token_count    — token count of this chunk
        overlap_tokens — tokens of overlap prefix (0 = no overlap)
    """
    sections = split_into_sections(combined_text)
    all_chunks: list[dict] = []

    for section_label, section_text in sections:
        section_tokens = count_tokens(section_text)

        if section_tokens <= chunk_size:
            # Whole section fits — one chunk, no overlap needed
            all_chunks.append({
                "section":        section_label,
                "text":           section_text,
                "token_count":    section_tokens,
                "overlap_tokens": 0,
            })
        else:
            # Section is too large — chunk it with overlap
            section_chunks = _chunk_section_with_overlap(
                section_text, section_label, chunk_size, overlap_tokens
            )
            all_chunks.extend(section_chunks)

    # Attach global 1-based index
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i + 1

    return all_chunks


# ---------------------------------------------------------------------------
# Reduce budget guard
# ---------------------------------------------------------------------------

def _guard_reduce_budget(
    mini_summaries: list[str],
    title: str,
    url: str,
) -> list[str]:
    """
    Ensure the combined mini-summaries won't push the reduce prompt over the
    TOKEN_SAFE_LIMIT.

    Strategy: if the total exceeds budget, drop the shortest summaries first
    (they are the least informative) until the budget is met.  Original order
    is preserved in the final list.

    The budget is:
        TOKEN_SAFE_LIMIT - REDUCE_PROMPT_OVERHEAD - tokens(title) - tokens(url)
    """
    overhead = REDUCE_PROMPT_OVERHEAD + count_tokens(title) + count_tokens(url)
    budget   = TOKEN_SAFE_LIMIT - overhead

    total = sum(count_tokens(s) for s in mini_summaries)
    if total <= budget:
        return mini_summaries  # nothing to trim

    print(
        f"      [REDUCE BUDGET] Mini-summaries total {total} tokens "
        f"exceeds budget {budget} — trimming shortest summaries"
    )

    # Sort ascending by token count — drop the shortest ones first
    indexed   = sorted(enumerate(mini_summaries), key=lambda x: count_tokens(x[1]))
    kept      = list(range(len(mini_summaries)))   # all indices initially kept
    running   = total

    for orig_idx, summary in indexed:
        if running <= budget:
            break
        st       = count_tokens(summary)
        running -= st
        kept.remove(orig_idx)
        print(
            f"      [REDUCE BUDGET] Dropped chunk summary #{orig_idx + 1} "
            f"({st} tokens) — {running} tokens remaining"
        )

    # Restore original order
    kept.sort()
    return [mini_summaries[i] for i in kept]


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, system: str = SYSTEM_PROMPT) -> str | None:
    """
    Send a prompt to Ollama and return the response text.
    Returns None on failure.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temperature for consistent structured output
            "num_predict": 1024,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        print(f"    [WARN] Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    except requests.exceptions.ConnectionError:
        print("    [ERROR] Could not connect to Ollama. Is it running? (ollama serve)")
        return None
    except requests.exceptions.Timeout:
        print("    [ERROR] Ollama timed out.")
        return None
    except Exception as e:
        print(f"    [ERROR] Ollama call failed: {e}")
        return None


def parse_json_response(raw: str) -> dict | None:
    """
    Extract and parse JSON from OLMo's response.
    Handles cases where the model wraps the JSON in markdown fences.
    """
    if not raw:
        return None
    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract just the JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Taxonomy validation
# ---------------------------------------------------------------------------

# Master lookup — maps each dimension to its valid tag set (lowercase for comparison)
_VALID_TAGS = {
    "individual_characteristics": {t.lower(): t for t in INDIVIDUAL_CHARACTERISTICS},
    "career_pathway":             {t.lower(): t for t in CAREER_PATHWAY},
    "organisational_culture":     {t.lower(): t for t in ORGANISATIONAL_CULTURE},
    "research_funding_process":   {t.lower(): t for t in RESEARCH_FUNDING_PROCESS},
}
_VALID_RESOURCE_TYPES     = {t.lower(): t for t in RESOURCE_TYPES}
_VALID_RESOURCE_FORMATS   = {t.lower(): t for t in RESOURCE_FORMATS}
_VALID_SECTORS            = {t.lower(): t for t in SECTORS}
_VALID_ADOPTION_READINESS = {t.lower(): t for t in ADOPTION_READINESS}


def _clean_tags(tags, valid_map, dimension, tier, rid):
    """
    Filter a list of tags against a valid set.
    Returns (valid_tags, removed_tags).
    Matching is case-insensitive so minor capitalisation drift is forgiven.
    """
    valid, removed = [], []
    for tag in tags:
        normalised = tag.strip().lower()
        if normalised in valid_map:
            valid.append(valid_map[normalised])
        else:
            removed.append(tag)
            print(f"    [TAXONOMY] [{rid}] Removed invalid {dimension}.{tier}: '{tag}'")
    return valid, removed


def _clean_scalar(value, valid_map, field, rid, fallback=""):
    """Validate a single-value field. Returns canonical value or fallback."""
    if not value:
        return fallback
    normalised = value.strip().lower()
    if normalised in valid_map:
        return valid_map[normalised]
    print(f"    [TAXONOMY] [{rid}] Invalid {field}: '{value}' -> set to '{fallback}'")
    return fallback


def validate_tags(parsed, rid):
    """
    Post-process OLMo JSON to enforce full taxonomy compliance.
    - Strips any tag not in the validated list for its dimension, logs removals.
    - Deduplicates across primary/secondary — primary wins.
    - Validates all scalar fields against their allowed lists.
    - Attaches a _validation block summarising what was fixed.
    """
    validation_log = []

    # Dimension tags
    for dim, valid_map in _VALID_TAGS.items():
        block = parsed.get(dim, {})
        if not isinstance(block, dict):
            parsed[dim] = {"primary": [], "secondary": [], "validation_notes": ["Malformed block reset"]}
            continue

        primary_raw   = block.get("primary",   []) if isinstance(block.get("primary"),   list) else []
        secondary_raw = block.get("secondary", []) if isinstance(block.get("secondary"), list) else []

        primary_valid,   primary_removed   = _clean_tags(primary_raw,   valid_map, dim, "primary",   rid)
        secondary_valid, secondary_removed = _clean_tags(secondary_raw, valid_map, dim, "secondary", rid)

        # Primary wins on duplicates
        primary_set     = {t.lower() for t in primary_valid}
        secondary_valid = [t for t in secondary_valid if t.lower() not in primary_set]

        removed_all = primary_removed + secondary_removed
        if removed_all:
            validation_log.append(f"{dim}: removed {removed_all}")

        parsed[dim] = {
            "primary":          primary_valid,
            "secondary":        secondary_valid,
            "validation_notes": [f"Removed: {removed_all}"] if removed_all else [],
        }

    # Scalar fields
    parsed["resource_type"] = _clean_scalar(
        parsed.get("resource_type", ""), _VALID_RESOURCE_TYPES, "resource_type", rid)
    parsed["resource_format"] = _clean_scalar(
        parsed.get("resource_format", ""), _VALID_RESOURCE_FORMATS, "resource_format", rid)
    parsed["adoption_readiness_level"] = _clean_scalar(
        parsed.get("adoption_readiness_level", ""), _VALID_ADOPTION_READINESS,
        "adoption_readiness_level", rid)

    # Sector (can be comma-separated string or list)
    raw_sector    = parsed.get("sector", "")
    sector_tokens = (
        raw_sector if isinstance(raw_sector, list)
        else [s.strip() for s in str(raw_sector).split(",")]
    )
    valid_sectors, removed_sectors = _clean_tags(
        sector_tokens, _VALID_SECTORS, "sector", "value", rid
    )
    parsed["sector"] = valid_sectors
    if removed_sectors:
        validation_log.append(f"sector: removed {removed_sectors}")

    parsed["_validation"] = {
        "passed":       len(validation_log) == 0,
        "issues_found": len(validation_log),
        "log":          validation_log,
    }
    return parsed


# ---------------------------------------------------------------------------
# Single pass
# ---------------------------------------------------------------------------

def run_single_pass(resource: dict, model: str) -> dict:
    """
    Send all content in one OLMo call. Used when combined_token_count <= 3,796.
    """
    print(f"    → Single pass ({resource['combined_token_count']} tokens)")

    prompt = build_tagging_prompt(
        resource["title"],
        resource["url"],
        resource["combined_text"],
    )
    raw    = call_ollama(prompt, model)
    parsed = parse_json_response(raw)

    if parsed:
        parsed = validate_tags(parsed, resource["resource_id"])
        validation_passed = parsed["_validation"]["passed"]
        issues = parsed["_validation"]["issues_found"]
        print(f"    [VALIDATION] {'OK - clean' if validation_passed else str(issues) + ' issue(s) fixed'}")

    return {
        "status":        "success" if parsed else "parse_failed",
        "route_taken":   "single_pass",
        "chunk_count":   1,
        "chunks":        None,   # not applicable for single pass
        "raw_response":  raw,
        "tags":          parsed,
        "mini_summaries": None,
    }


# ---------------------------------------------------------------------------
# Map-Reduce
# ---------------------------------------------------------------------------

def run_map_reduce(resource: dict, model: str) -> dict:
    """
    Chunk → per-chunk mini-summaries (Map) → final tagging call (Reduce).
    Used when combined_token_count > 3,796.

    v2 changes:
    - Uses chunk_text_structured() for structural + overlap chunking.
    - Each map prompt includes section label and overlap awareness.
    - Mini-summaries are budget-checked before the reduce call.
    - Full chunk documentation is written to the output record.
    """
    chunks = chunk_text_structured(resource["combined_text"])
    total  = len(chunks)
    print(f"    → Map-Reduce ({resource['combined_token_count']} tokens → {total} chunks)")

    # ── Map phase ────────────────────────────────────────────────────────────
    mini_summaries: list[str]  = []
    chunk_docs:     list[dict] = []

    for chunk in chunks:
        idx = chunk["chunk_index"]
        print(
            f"      [Map {idx}/{total}] "
            f"section='{chunk['section']}' "
            f"~{chunk['token_count']} tokens "
            f"(overlap: {chunk['overlap_tokens']})"
        )

        prompt = build_chunk_summary_prompt(
            chunk["text"],
            idx,
            total,
            section_label=chunk["section"],
            has_overlap=chunk["overlap_tokens"] > 0,
        )
        raw = call_ollama(prompt, model, system="You are a helpful EDI research assistant.")

        # Document this chunk fully for cross-verification
        chunk_doc = {
            "chunk_index":    idx,
            "section":        chunk["section"],
            "token_count":    chunk["token_count"],
            "overlap_tokens": chunk["overlap_tokens"],
            "text":           chunk["text"],           # exact text OLMo saw
            "mini_summary":   raw if raw else None,    # OLMo's bullet-point output
            "map_status":     "success" if raw else "failed",
        }
        chunk_docs.append(chunk_doc)

        if raw:
            mini_summaries.append(raw)
        else:
            print(f"      [WARN] Map chunk {idx} failed — skipping from reduce input")

    if not mini_summaries:
        return {
            "status":        "map_failed",
            "route_taken":   "map_reduce",
            "chunk_count":   total,
            "chunks":        chunk_docs,
            "mini_summaries": [],
            "raw_response":  None,
            "tags":          None,
        }

    # ── Reduce budget guard ───────────────────────────────────────────────────
    mini_summaries_for_reduce = _guard_reduce_budget(
        mini_summaries, resource["title"], resource["url"]
    )

    # ── Reduce phase ─────────────────────────────────────────────────────────
    print(
        f"      [Reduce] Combining {len(mini_summaries_for_reduce)} "
        f"of {len(mini_summaries)} summaries → final tags"
    )
    reduce_prompt = build_reduce_prompt(
        resource["title"],
        resource["url"],
        mini_summaries_for_reduce,
    )
    raw    = call_ollama(reduce_prompt, model)
    parsed = parse_json_response(raw)

    if parsed:
        parsed = validate_tags(parsed, resource["resource_id"])
        validation_passed = parsed["_validation"]["passed"]
        issues = parsed["_validation"]["issues_found"]
        print(
            f"      [VALIDATION] "
            f"{'OK - clean' if validation_passed else str(issues) + ' issue(s) fixed'}"
        )

    return {
        "status":        "success" if parsed else "parse_failed",
        "route_taken":   "map_reduce",
        "chunk_count":   total,
        "chunks":        chunk_docs,       # full per-chunk record for cross-verification
        "mini_summaries": mini_summaries,  # all successful map outputs (pre-budget trim)
        "raw_response":  raw,
        "tags":          parsed,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_resource(resource: dict, model: str, dry_run: bool) -> dict:
    """
    Run tagging + summarisation for one assembled resource.
    """
    rid   = resource["resource_id"]
    title = resource["title"]
    route = resource["route"]

    print(f"\n[{rid}] {title[:65]}")
    print(f"       tokens: {resource['combined_token_count']} | route: {route}")

    result = {
        "resource_id":          rid,
        "title":                title,
        "url":                  resource["url"],
        "combined_token_count": resource["combined_token_count"],
        "route_taken":          route,
        "chunk_count":          None,
        "chunks":               None,   # populated for map-reduce runs
        "status":               None,
        "tags":                 None,
        "mini_summaries":       None,
        "raw_response":         None,
    }

    if dry_run:
        print(f"    [DRY RUN] Would run {route} — skipping OLMo call")
        result["status"] = "dry_run"
        return result

    if route == "single_pass":
        outcome = run_single_pass(resource, model)
    else:
        outcome = run_map_reduce(resource, model)

    result.update(outcome)

    status_symbol = "✓" if result["status"] == "success" else "✗"
    print(f"    {status_symbol} {result['status']}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EDI Hub+ Pipeline — Stage 6: LLM Tagging + Summarisation"
    )
    parser.add_argument("--stage3",  default=STAGE3_INPUT)
    parser.add_argument("--stage4",  default=STAGE4_INPUT)
    parser.add_argument("--stage5",  default=STAGE5_INPUT)
    parser.add_argument("--output",  default=STAGE6_OUTPUT)
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help="Ollama model name (default: olmo2:7b)")
    parser.add_argument("--test",    metavar="RESOURCE_ID",
                        help="Run on a single resource ID only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Assemble and show text only — no OLMo calls")
    args = parser.parse_args()

    print("=" * 60)
    print("EDI Hub+ Pipeline — Stage 6: LLM Tagging + Summarisation")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Ollama:  {OLLAMA_URL}\n")

    # ── Load all inputs ───────────────────────────────────────────────────────
    data = load_inputs(args.stage3, args.stage4, args.stage5)

    # All resource IDs come from Stage 5 (it has the routing decisions)
    with open(args.stage5, encoding="utf-8") as f:
        s5_records = json.load(f)

    resource_ids = [str(r["resource_id"]) for r in s5_records]

    # Also include any Stage 4 PDFs that Stage 5 didn't process
    # (Stage 5 only processes Stage 3 web pages, not direct PDFs from Stage 4)
    with open(args.stage4, encoding="utf-8") as f:
        s4_records = json.load(f)
    for r in s4_records:
        rid = str(r["resource_id"])
        if rid not in resource_ids and r.get("clean_status") == "success":
            resource_ids.append(rid)

    if args.test:
        resource_ids = [r for r in resource_ids if r == str(args.test)]
        if not resource_ids:
            print(f"[ERROR] Resource ID {args.test} not found")
            sys.exit(1)

    print(f"Processing {len(resource_ids)} resource(s)...\n")

    # ── Assemble and process ──────────────────────────────────────────────────
    results = []
    for rid in resource_ids:
        assembled = assemble_resource_text(rid, data)
        if not assembled:
            print(f"[SKIP] Resource {rid} — no parent content found")
            continue

        if args.dry_run:
            # In dry-run mode, show the assembled text and chunk plan for inspection
            print(f"\n{'='*60}")
            print(f"[{rid}] {assembled['title']}")
            print(f"Route: {assembled['route']} | Tokens: {assembled['combined_token_count']}")
            print(f"{'='*60}")

            if assembled["route"] == "map_reduce":
                chunks = chunk_text_structured(assembled["combined_text"])
                print(f"Chunk plan ({len(chunks)} chunks):")
                for c in chunks:
                    print(
                        f"  [{c['chunk_index']}] section='{c['section']}' "
                        f"tokens={c['token_count']} overlap={c['overlap_tokens']}"
                    )
                print()

            print(assembled["combined_text"][:2000])
            if len(assembled["combined_text"]) > 2000:
                print(f"\n... [{len(assembled['combined_text']) - 2000} chars truncated]")

        result = process_resource(assembled, args.model, args.dry_run)
        results.append(result)

    # ── Save output ───────────────────────────────────────────────────────────
    if not args.dry_run:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ok       = sum(1 for r in results if r["status"] == "success")
        failed   = sum(1 for r in results if r["status"] == "parse_failed")
        sp_count = sum(1 for r in results if r["route_taken"] == "single_pass")
        mr_count = sum(1 for r in results if r["route_taken"] == "map_reduce")

        print(f"\n{'='*60}")
        print(f"STAGE 6 COMPLETE")
        print(f"  Success:      {ok}")
        print(f"  Parse failed: {failed}")
        print(f"  Single pass:  {sp_count}")
        print(f"  Map-Reduce:   {mr_count}")
        print(f"  Output:       {output_path}")
        print(f"{'='*60}")

    else:
        print(f"\n[DRY RUN complete — {len(results)} resources assembled, no OLMo calls made]")


if __name__ == "__main__":
    main()
