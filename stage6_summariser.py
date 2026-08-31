"""
EDI Hub+ Pipeline — Stage 6 Summariser
========================================
Generates per-chunk summaries of EDI resources using OLMo-2 7B via Ollama.

Mechanism:
    1. Load chunk JSON file (from stage6_chunker.py output)
    2. Merge continuation chunks with their parent before summarising
    3. For each chunk (or merged group), call OLMo to produce a structured summary:
         - core_topic    : one sentence on what this section is about
         - key_points    : 2-3 bullet points of main arguments or recommendations
         - edi_relevance : who is affected and how, from an EDI perspective
    4. Write all chunk summaries to a JSON output file

Usage:
    python stage6_summariser.py --resource-id 39  --source stage4
    python stage6_summariser.py --resource-id 44  --source stage4
    python stage6_summariser.py --resource-id 202 --source stage4
    python stage6_summariser.py --resource-id 203 --source stage5
    python stage6_summariser.py --resource-id 39  --source stage4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import requests

from summary_templates import get_template

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage6_summariser")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL     = "http://localhost:11434/api/generate"
DEFAULT_MODEL  = "olmo2:7b"
OLLAMA_TIMEOUT = 300
CHUNKS_DIR     = "."
OUTPUT_DIR     = "."

# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str, model: str) -> str | None:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 600},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        log.warning("Ollama HTTP %d: %s", resp.status_code, resp.text[:200])
        return None
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama — is it running? (ollama serve)")
        return None
    except requests.exceptions.Timeout:
        log.error("Ollama timed out after %ds", OLLAMA_TIMEOUT)
        return None
    except Exception as exc:
        log.error("Ollama call failed: %s", exc)
        return None


def _parse_json_response(raw: str) -> dict | None:
    if not raw:
        return None
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(clean):
        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(clean[start:i+1])
                except json.JSONDecodeError:
                    pass
    return None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_SUMMARISE = (
    "You are an expert EDI (Equality, Diversity and Inclusion) research analyst "
    "specialising in STEM workforce equity. You summarise sections of EDI resources "
    "clearly and concisely, capturing the key points and their EDI relevance. "
    "You always respond with valid JSON only — no preamble, no explanation outside the JSON."
)

# ---------------------------------------------------------------------------
# Merge continuation chunks
# ---------------------------------------------------------------------------

def merge_continuation_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge continuation chunks with their parent chunk.

    A continuation chunk has is_continuation=True and shares the same
    section_title as the preceding chunk. Merged chunks are combined
    into one unit before summarising so OLMo sees the full section.

    Returns a new list of chunk groups — each group is a list of one
    or more raw chunks to be summarised together.
    """
    groups   = []
    current  = []

    for chunk in chunks:
        is_cont = chunk.get("is_continuation", False)

        if is_cont and current:
            # Append to current group — this is a continuation of the previous section
            current.append(chunk)
        else:
            # Start a new group
            if current:
                groups.append(current)
            current = [chunk]

    if current:
        groups.append(current)

    merged_groups = []
    for group in groups:
        if len(group) == 1:
            merged_groups.append({
                "chunk_numbers":    [group[0]["chunk_number"]],
                "part_label":       group[0]["part_label"],
                "section_title":    group[0].get("section_title", ""),
                "is_merged":        False,
                "token_count":      group[0]["token_count"],
                "text":             "\n".join(group[0].get("text_lines", [])),
            })
        else:
            # Merge all continuation chunks into one text block
            combined_text = "\n\n".join(
                "\n".join(c.get("text_lines", [])) for c in group
            )
            total_tokens  = sum(c["token_count"] for c in group)
            labels        = " + ".join(c["part_label"] for c in group)
            merged_groups.append({
                "chunk_numbers":    [c["chunk_number"] for c in group],
                "part_label":       labels,
                "section_title":    group[0].get("section_title", ""),
                "is_merged":        True,
                "token_count":      total_tokens,
                "text":             combined_text,
            })
            log.info("  Merged chunks %s → one summarisation unit (%d tokens)",
                     [c["chunk_number"] for c in group], total_tokens)

    return merged_groups

# ---------------------------------------------------------------------------
# Step 1 — Fact extraction (anti-hallucination)
# ---------------------------------------------------------------------------

def _build_fact_extraction_prompt(chunk_text: str) -> str:
    """
    Extract verified facts from the chunk text before summarisation.
    This creates a reference list OLMo must respect when writing the summary.
    """
    return f"""Read the following section of an EDI resource and extract only facts that are explicitly stated in the text.

--- SECTION TEXT ---
{chunk_text[:3000]}
--- END SECTION ---

Extract and list:
1. All numbers and percentages (e.g. "55 companies", "70% increase", "1.4 million samples")
2. All named organisations, people, reports, or programmes mentioned
3. All specific named examples, tools, or interventions described

Be precise — only include what is explicitly written. Do not infer or combine figures.
If no facts of a type exist, return an empty list for that field.

Respond with this JSON only:
{{
  "numbers_and_statistics": ["list every number or percentage exactly as written in the text"],
  "named_entities": ["list every organisation, person, report, or programme named in the text"],
  "specific_examples": ["list concrete examples, tools, or interventions named in the text"]
}}"""


def _extract_facts(chunk_text: str, model: str) -> dict:
    """
    Run fact extraction on a chunk and return the verified facts dict.
    Falls back to empty facts if parse fails.
    """
    prompt = _build_fact_extraction_prompt(chunk_text)
    raw    = _call_ollama(prompt, SYSTEM_SUMMARISE, model)
    parsed = _parse_json_response(raw)
    if parsed and isinstance(parsed, dict):
        return {
            "numbers_and_statistics": parsed.get("numbers_and_statistics", []),
            "named_entities":         parsed.get("named_entities", []),
            "specific_examples":      parsed.get("specific_examples", []),
        }
    log.warning("  Fact extraction parse failed — proceeding without verified facts")
    return {"numbers_and_statistics": [], "named_entities": [], "specific_examples": []}


# ---------------------------------------------------------------------------
# Summarisation prompt
# ---------------------------------------------------------------------------

def _build_summary_prompt(
    chunk_text:     str,
    section_label:  str,
    verified_facts: dict,
) -> str:
    """
    Build the summarisation prompt including the verified facts list
    to prevent OLMo from hallucinating statistics or named entities.
    """
    # Format the verified facts block
    stats   = verified_facts.get("numbers_and_statistics", [])
    names   = verified_facts.get("named_entities", [])
    examples= verified_facts.get("specific_examples", [])

    facts_block = ""
    if stats or names or examples:
        facts_lines = []
        if stats:
            facts_lines.append("Numbers and statistics in this section:")
            facts_lines.extend(f"  - {s}" for s in stats)
        if names:
            facts_lines.append("Named organisations, people, and reports:")
            facts_lines.extend(f"  - {n}" for n in names)
        if examples:
            facts_lines.append("Specific examples and tools:")
            facts_lines.extend(f"  - {e}" for e in examples)
        facts_block = (
            "\nVERIFIED FACTS FROM THIS SECTION (extracted directly from the text):\n"
            + "\n".join(facts_lines)
            + "\n\nCRITICAL: When writing key_points, you may only use numbers, "
            "percentages, and named entities from the VERIFIED FACTS list above. "
            "Do not generate any statistic, percentage, or named claim that is not in this list. "
            "If you are unsure whether a number is correct, do not include it.\n"
        )

    return f"""Summarise the following section of an EDI resource.

Section: {section_label}
{facts_block}
--- SECTION TEXT ---
{chunk_text[:3000]}
--- END SECTION ---

Produce a JSON object with exactly these keys:

{{
  "core_topic": "One precise sentence describing what this specific section is fundamentally about",
  "key_points": [
    "First key point",
    "Second key point",
    "Third key point (if applicable)"
  ],
  "edi_relevance": "One or two sentences on who is affected and how this section relates to EDI in STEM research"
}}

Rules for core_topic:
- One sentence only
- Capture the main INSIGHT or ARGUMENT of the section — what does this section
  show, argue, or recommend?
- Do NOT restate or paraphrase the section label shown above
- Do NOT start with "This section..." or "The section..."
- Be precise and grounded in the actual content of the text

Rules for key_points:
- 2 to 3 items — no more, no less
- Each point should be an INSIGHT or FINDING, not a list item or heading copied from the text
- Explain what the evidence shows or what the recommendation achieves, not just what it is
- Only use numbers and named entities from the VERIFIED FACTS list — never invent statistics
- Where content is more conceptual, summarise the argument precisely
- Every point must be grounded in something actually written in this section

Rules for edi_relevance:
Before writing edi_relevance, mentally answer these three questions from the text:
  1. Who exactly is mentioned? (specific people, groups, or conditions — not categories)
  2. What specific problem or barrier do they face according to this section?
  3. What does this section argue, show, or recommend that addresses them?
Write edi_relevance using only your answers to those three questions.
Do not add anything not grounded in your answers.

Respond with the JSON object only"""

# ---------------------------------------------------------------------------
# Step 3 — Verification pass (anti-hallucination)
# ---------------------------------------------------------------------------

def _build_verification_prompt(
    chunk_text:   str,
    summary:      dict,
    verified_facts: dict,
    section_label:str,
) -> str:
    """
    Ask OLMo to verify every specific claim in the summary against the chunk text.
    Pays special attention to numbers — checks they are not inverted, rounded wrongly,
    or confused with other numbers in the text.
    """
    key_points_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(summary.get("key_points", [])))

    # Build the verified numbers list for explicit checking
    verified_nums = verified_facts.get("numbers_and_statistics", [])
    nums_block = ""
    if verified_nums:
        nums_str = "\n".join(f"  - {n}" for n in verified_nums)
        nums_block = f"""
VERIFIED NUMBERS FROM THE TEXT (the ONLY numbers that may appear in the summary):
{nums_str}

CRITICAL NUMBER CHECK: For every number or percentage in the summary below,
verify it appears EXACTLY in the verified numbers list above.
Common errors to catch:
  - Inverted numbers (e.g. writing 9% when the text says 91%)
  - Wrong numbers (e.g. writing 55% when the text says 55 companies)
  - Combined numbers (e.g. writing 55% increase by mixing two separate facts)
If a number in the summary does not appear exactly in the verified list → remove or correct it.
"""

    return f"""You are verifying a summary of an EDI resource section for factual accuracy.

Your job: check every specific claim against the original text and the verified numbers list.

Section: {section_label}
{nums_block}
--- ORIGINAL SECTION TEXT ---
{chunk_text[:3000]}
--- END ORIGINAL TEXT ---

--- SUMMARY TO VERIFY ---
core_topic: {summary.get("core_topic", "")}

key_points:
{key_points_str}

edi_relevance: {summary.get("edi_relevance", "")}
--- END SUMMARY ---

For each key point:
- Check every number against the VERIFIED NUMBERS list — if wrong, inverted, or fabricated → fix or remove it
- If the point is fully supported → keep as-is
- If a claim cannot be verified → rewrite without the unverified claim
- If the whole point is unsupported → replace with empty string ""

For core_topic and edi_relevance:
- Remove any specific claims not in the original text
- Keep general arguments that are supported

Respond with this JSON only:
{{
  "core_topic": "verified or corrected core topic",
  "key_points": [
    "verified or corrected point 1",
    "verified or corrected point 2",
    "verified or corrected point 3 or empty string if unsupported"
  ],
  "edi_relevance": "verified or corrected edi relevance"
}}"""


def _verify_summary(
    chunk_text:     str,
    summary:        dict,
    section_label:  str,
    model:          str,
    verified_facts: dict = None,
) -> tuple[dict, bool]:
    """
    Run the verification pass on a generated summary.
    Passes verified_facts to the prompt for explicit number checking.
    Returns (verified_summary, was_corrected).
    """
    prompt = _build_verification_prompt(
        chunk_text, summary,
        verified_facts or {"numbers_and_statistics": [], "named_entities": [], "specific_examples": []},
        section_label,
    )
    raw    = _call_ollama(prompt, SYSTEM_SUMMARISE, model)
    parsed = _parse_json_response(raw)

    if not parsed:
        log.warning("  Verification parse failed — keeping original summary")
        return summary, False

    # Clean up: remove empty key points
    key_points = [p for p in parsed.get("key_points", []) if p and p.strip()]
    if not key_points:
        log.warning("  Verification removed all key points — keeping original")
        return summary, False

    verified = {
        "core_topic":    parsed.get("core_topic", summary.get("core_topic", "")),
        "key_points":    key_points,
        "edi_relevance": parsed.get("edi_relevance", summary.get("edi_relevance", "")),
    }

    # Check if anything actually changed
    was_corrected = (
        verified["core_topic"]    != summary.get("core_topic")    or
        verified["key_points"]    != summary.get("key_points")    or
        verified["edi_relevance"] != summary.get("edi_relevance")
    )

    if was_corrected:
        log.info("  Verification corrected the summary — hallucinations removed")
    else:
        log.info("  Verification passed — no hallucinations detected")

    return verified, was_corrected


# ---------------------------------------------------------------------------
# EDI relevance quality check — rewrite if generic phrases detected
# ---------------------------------------------------------------------------

# Phrases that indicate OLMo produced a generic EDI relevance statement
# rather than one grounded in the specific text
_GENERIC_EDI_PHRASES = [
    "underrepresented groups",
    "diverse groups",
    "diverse candidates",
    "all potential employees",
    "broader community",
    "specific groups",
    "targeted communities",
    "marginalised groups",
    "all individuals",
    "various groups",
]


def _edi_relevance_is_generic(text: str) -> bool:
    """Return True if the edi_relevance contains generic placeholder phrases."""
    lower = text.lower()
    return any(phrase in lower for phrase in _GENERIC_EDI_PHRASES)


def _build_edi_rewrite_prompt(
    chunk_text:      str,
    current_edi:     str,
    section_label:   str,
) -> str:
    return f"""The following EDI relevance statement is too generic.
Rewrite it using a structured reasoning approach.

Section: {section_label}

--- SECTION TEXT ---
{chunk_text[:2000]}
--- END SECTION ---

Current EDI relevance (too generic):
"{current_edi}"

Before writing the new edi_relevance, answer these three questions using ONLY what is written in the section text:

Q1: Who exactly is mentioned in this section?
    Name the specific people, groups, or communities — for example: Black women academics,
    people with ADHD, transgender applicants, early career researchers, disabled scientists.
    Do not answer with a category — answer with who the text actually names or describes.

Q2: What specific problem, barrier, or challenge does this section say they face?
    Quote or closely paraphrase from the text.

Q3: What does this section specifically argue, show, or recommend that addresses them?
    Be concrete — what does it say should happen or what evidence does it present?

Now write the edi_relevance in one or two sentences using ONLY your answers to Q1, Q2, and Q3.
Do not add anything that is not in your answers.

Respond with this JSON only:
{{
  "q1_who": "your answer to Q1",
  "q2_problem": "your answer to Q2",
  "q3_argument": "your answer to Q3",
  "edi_relevance": "one or two sentences built from your answers above"
}}"""


# ---------------------------------------------------------------------------
# Summarise one chunk group
# ---------------------------------------------------------------------------

def _build_chunk_template_prompt(
    chunk_text:    str,
    section_label: str,
    headings:      list[str],
    template:      str,
    resource_type: str,
    core_topic:    str,
    key_points:    list[str],
) -> str:
    """
    Ask OLMo to fill in the template sections for a single chunk.
    Only sections relevant to this chunk get real content — others get
    'Not described in this section.' so the aggregation step can combine
    across chunks later.
    """
    # Build exact JSON schema
    json_schema = "{\n"
    for h in headings:
        json_schema += f'  "{h}": "Your response or Not described in this section.",\n'
    json_schema = json_schema.rstrip(",\n") + "\n}"

    # Extract sub-questions per heading from the template
    sections_guidance = []
    lines = template.splitlines()
    current_heading   = None
    current_questions = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("* "):
            if current_heading:
                sections_guidance.append((current_heading, current_questions))
            current_heading   = stripped[2:].strip()
            current_questions = []
        elif stripped and current_heading:
            current_questions.append(stripped)
    if current_heading:
        sections_guidance.append((current_heading, current_questions))

    guidance_block = ""
    for heading, questions in sections_guidance:
        guidance_block += f"\n  \"{heading}\"\n"
        for q in questions:
            guidance_block += f"    → {q}\n"

    key_points_str = "\n".join(f"  - {p}" for p in key_points)

    return f"""You are filling in a structured summary template for ONE SECTION of an EDI resource.

Resource type:  {resource_type}
Section label:  {section_label}
Section topic:  {core_topic}
Key points from this section:
{key_points_str}

--- SECTION TEXT ---
{chunk_text[:2500]}
--- END SECTION ---

For each template heading below, write 1-3 sentences based ONLY on what this specific section covers.
If this section does not contain relevant content for a heading, write exactly: "Not described in this section."

Template headings and their guidance:
{guidance_block}

You must respond with a JSON object containing EXACTLY these {len(headings)} keys:

{json_schema}

RULES:
- Only use content from THIS section — not general knowledge
- Be specific and grounded in the section text
- If a heading is not covered by this section, write: "Not described in this section."
- Do NOT rename, skip, or add keys
- Write in clear professional prose — no bullet points

Respond with the JSON object only."""


def summarise_chunk_group(
    group:        dict,
    model:        str,
    dry_run:      bool      = False,
    template:     str       = None,
    headings:     list[str] = None,
    resource_type:str       = None,
) -> dict:
    """
    Summarise one chunk group (which may be a single chunk or merged continuations).
    Returns a summary dict.
    """
    chunk_nums = group["chunk_numbers"]
    label      = group["part_label"]
    log.info("  Summarising chunk(s) %s: %s", chunk_nums, label[:60])

    result = {
        "chunk_numbers":    chunk_nums,
        "part_label":       label,
        "section_title":    group.get("section_title", ""),
        "is_merged":        group["is_merged"],
        "token_count":      group["token_count"],
        "summary":          None,
        "template_sections": None,
        "olmo_calls":       0,
        "error":            None,
    }

    if dry_run:
        log.info("  [DRY RUN] Skipping OLMo call")
        result["summary"] = {
            "core_topic":     "Dry run — no summary generated",
            "key_points":     ["Point 1", "Point 2"],
            "edi_relevance":  "Dry run — no EDI relevance generated",
        }
        result["template_sections"] = {h: "Dry run." for h in (headings or [])}
        return result

    # Step 1 — Extract verified facts to prevent hallucination
    log.info("  Chunk(s) %s: extracting verified facts...", chunk_nums)
    verified_facts = _extract_facts(group["text"], model)
    result["olmo_calls"] += 1
    log.debug("  Chunk(s) %s: facts — stats=%s names=%s",
              chunk_nums,
              verified_facts["numbers_and_statistics"][:3],
              verified_facts["named_entities"][:3])

    # Step 2 — Summarise using verified facts as grounding reference
    prompt  = _build_summary_prompt(group["text"], label, verified_facts)
    raw     = _call_ollama(prompt, SYSTEM_SUMMARISE, model)
    result["olmo_calls"] += 1

    parsed = _parse_json_response(raw)
    if not parsed:
        log.warning("  Chunk(s) %s: summary parse failed", chunk_nums)
        result["error"] = "parse_failed"
        result["summary"] = {
            "core_topic":    "Could not parse summary",
            "key_points":    [],
            "edi_relevance": "",
        }
        return result

    # Validate key_points — ensure it's a list
    if not isinstance(parsed.get("key_points"), list):
        parsed["key_points"] = [str(parsed.get("key_points", ""))]

    # Enforce 2–3 key points — trim to max 3, retry only when subsections are distinct
    parsed["key_points"] = [p for p in parsed["key_points"] if p and p.strip()]
    parsed["key_points"] = parsed["key_points"][:3]
    if len(parsed["key_points"]) < 2:

        # Extract subsection headings from part_label
        import re as _re2
        subsections = _re2.findall(r'##+ ([^\|^\]]+)', label)
        subsections = [s.strip() for s in subsections if s.strip()]

        # Only retry if there are 2+ distinct subsections — if it's a single subsection
        # or all subsections are variations of the same type (e.g. all guidance items),
        # 1 point may genuinely be correct — accept it without retrying
        if len(subsections) >= 2:
            log.warning("  Chunk(s) %s: only %d key point(s) across %d subsections — retrying...",
                        chunk_nums, len(parsed["key_points"]), len(subsections))
            subsection_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subsections))
            retry_prompt = (
                f"This EDI resource chunk covers {len(subsections)} subsections but only produced "
                f"{len(parsed['key_points'])} key point(s). You must produce one distinct key point "
                f"per subsection — each capturing the main insight or finding from that subsection.\n\n"
                f"The subsections are:\n{subsection_block}\n\n"
                f"Section label: {label}\n\n"
                f"--- SECTION TEXT ---\n{group['text'][:3000]}\n--- END ---\n\n"
                f"Write exactly {min(len(subsections), 3)} key points, one per subsection, "
                f"each as a complete sentence stating what that subsection specifically argues or finds.\n"
                f"If two subsections genuinely say the same thing, combine them into one point.\n\n"
                f"Respond with JSON only:\n"
                f"{{\"key_points\": [\"insight from subsection 1\", \"insight from subsection 2\""
                + (", \"insight from subsection 3\"" if len(subsections) >= 3 else "")
                + "]}}"
            )
            raw_retry = _call_ollama(retry_prompt, SYSTEM_SUMMARISE, model)
            result["olmo_calls"] += 1
            retry_parsed = _parse_json_response(raw_retry)
            if retry_parsed and isinstance(retry_parsed.get("key_points"), list):
                retry_points = [p for p in retry_parsed["key_points"] if p and p.strip()][:3]
                if len(retry_points) >= 2:
                    parsed["key_points"] = retry_points
                    log.info("  Chunk(s) %s: retry produced %d key points", chunk_nums, len(retry_points))
                else:
                    log.info("  Chunk(s) %s: retry still produced %d — subsections likely homogeneous, accepting",
                             chunk_nums, len(retry_points))
        else:
            log.info("  Chunk(s) %s: single subsection or thin content — 1 key point accepted",
                     chunk_nums)

    # Post-process key points: strip leaked verification reasoning
    # e.g. "...though the exact percentage is not specified as X in the original text"
    import re as _re
    cleaned_points = []
    for point in parsed.get("key_points", []):
        # Strip parenthetical caveats about the source text
        point = _re.sub(r",?\s*though the exact [^,.]+ (?:is not|are not)[^.]*\.", ".", point)
        point = _re.sub(r",?\s*though (?:this|it|the) (?:figure|number|percentage|statistic)[^.]*\.", ".", point)
        point = _re.sub(r"\s*\(note:[^)]*\)", "", point, flags=_re.IGNORECASE)
        cleaned_points.append(point.strip())
    parsed["key_points"] = cleaned_points

    # Step 3 — Verification pass: check every claim against the source text
    log.info("  Chunk(s) %s: running verification pass...", chunk_nums)
    verified_summary, was_corrected = _verify_summary(
        chunk_text=group["text"],
        summary=parsed,
        section_label=label,
        model=model,
        verified_facts=verified_facts,
    )
    result["olmo_calls"] += 1
    parsed = verified_summary

    # Step 4 — EDI rewrite if generic phrases still present after verification
    edi_text = parsed.get("edi_relevance", "")
    if edi_text and _edi_relevance_is_generic(edi_text):
        log.info("  Chunk(s) %s: edi_relevance still generic — rewriting...", chunk_nums)
        rewrite_prompt = _build_edi_rewrite_prompt(
            chunk_text=group["text"],
            current_edi=edi_text,
            section_label=label,
        )
        raw_rewrite = _call_ollama(rewrite_prompt, SYSTEM_SUMMARISE, model)
        result["olmo_calls"] += 1
        rewrite_parsed = _parse_json_response(raw_rewrite)
        if rewrite_parsed and rewrite_parsed.get("edi_relevance"):
            rewritten = rewrite_parsed["edi_relevance"]
            # Log the reasoning for transparency
            log.debug("  Chunk(s) %s: Q1=%s | Q2=%s",
                      chunk_nums,
                      rewrite_parsed.get("q1_who", "")[:60],
                      rewrite_parsed.get("q2_problem", "")[:60])
            if not _edi_relevance_is_generic(rewritten):
                parsed["edi_relevance"] = rewritten
                log.info("  Chunk(s) %s: edi_relevance rewritten — who: %s",
                         chunk_nums, rewrite_parsed.get("q1_who", "")[:60])
            else:
                log.warning("  Chunk(s) %s: rewrite still generic — keeping original",
                            chunk_nums)
        else:
            log.warning("  Chunk(s) %s: rewrite parse failed — keeping original",
                        chunk_nums)

    result["summary"] = parsed

    # Step 5 — Per-chunk template sections
    if template and headings and resource_type:
        log.info("  Chunk(s) %s: generating template sections (%s)...", chunk_nums, resource_type)
        tmpl_prompt = _build_chunk_template_prompt(
            chunk_text=group["text"],
            section_label=label,
            headings=headings,
            template=template,
            resource_type=resource_type,
            core_topic=parsed.get("core_topic", ""),
            key_points=parsed.get("key_points", []),
        )
        raw_tmpl = _call_ollama(tmpl_prompt, SYSTEM_SUMMARISE, model)
        result["olmo_calls"] += 1
        tmpl_parsed = _parse_json_response(raw_tmpl)
        if tmpl_parsed and isinstance(tmpl_parsed, dict):
            result["template_sections"] = tmpl_parsed
            log.info("  Chunk(s) %s: template sections done", chunk_nums)
        else:
            log.warning("  Chunk(s) %s: template section parse failed", chunk_nums)
            result["template_sections"] = {h: "Parse failed." for h in headings}

    log.info("  Chunk(s) %s: done — topic: %s",
             chunk_nums, parsed.get("core_topic", "")[:60])
    return result

# ---------------------------------------------------------------------------
# Document-level structured summary (template-driven)
# ---------------------------------------------------------------------------

def _extract_template_headings(template: str) -> list[str]:
    """Extract section headings from a template — lines starting with '* '."""
    headings = []
    for line in template.splitlines():
        line = line.strip()
        if line.startswith("* "):
            heading = line[2:].strip()
            if heading:
                headings.append(heading)
    return headings


def _build_document_summary_prompt(
    chunk_summaries: list[dict],
    resource_type:   str,
    title:           str,
    template:        str,
) -> str:
    """
    Build the document-level summary prompt.
    Explicitly passes required JSON keys so OLMo cannot guess wrong.
    """
    # Extract headings and their sub-questions from the template
    headings = _extract_template_headings(template)

    sections_guidance = []
    lines = template.splitlines()
    current_heading   = None
    current_questions = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("* "):
            if current_heading:
                sections_guidance.append((current_heading, current_questions))
            current_heading   = stripped[2:].strip()
            current_questions = []
        elif stripped and current_heading:
            current_questions.append(stripped)
    if current_heading:
        sections_guidance.append((current_heading, current_questions))

    # Guidance block — heading + sub-questions
    guidance_block = ""
    for heading, questions in sections_guidance:
        guidance_block += f"\n  \"{heading}\"\n"
        for q in questions:
            guidance_block += f"    → {q}\n"

    # Exact JSON schema OLMo must return
    json_schema = "{\n"
    for heading in headings:
        json_schema += f'  "{heading}": "Your 2-4 sentence response here",\n'
    json_schema = json_schema.rstrip(",\n") + "\n}"

    # Flatten chunk summaries into a digest
    digest_lines = []
    for s in chunk_summaries:
        if not s.get("summary"):
            continue
        label  = s.get("part_label", "")
        topic  = s["summary"].get("core_topic", "")
        points = s["summary"].get("key_points", [])
        edi    = s["summary"].get("edi_relevance", "")
        digest_lines.append(f"Section: {label}")
        if topic:
            digest_lines.append(f"  Topic: {topic}")
        for p in points:
            digest_lines.append(f"  - {p}")
        if edi:
            digest_lines.append(f"  EDI: {edi}")
        digest_lines.append("")

    digest = "\n".join(digest_lines)

    return f"""You are writing a structured summary of an EDI resource for the EDI Hub+ database.

Resource title: {title}
Resource type:  {resource_type}

Below is a digest of all sections of this resource from an earlier analysis:

--- RESOURCE DIGEST ---
{digest}
--- END DIGEST ---

You must respond with a JSON object containing EXACTLY these {len(headings)} keys, in this order:

{json_schema}

For each key, write 2 to 4 sentences addressing the questions listed below for that section:
{guidance_block}

RULES:
- Use ONLY information from the RESOURCE DIGEST — no outside knowledge
- If information for a section is not in the digest, end your response with a new sentence that reads exactly: "Not described in the document." — do NOT embed this phrase mid-sentence or mid-paragraph
- Write in clear professional prose — no bullet points inside your answers
- Do NOT rename, skip, or add any keys — the JSON must contain exactly the keys shown above
- Do NOT include the question text in your answer — answer directly in prose
- Every sentence must be complete and standalone — do not trail off or leave partial sentences

Respond with the JSON object only. No preamble, no explanation."""


def generate_document_summary(
    chunk_summaries: list[dict],
    resource_type:   str,
    title:           str,
    model:           str,
    dry_run:         bool = False,
) -> dict:
    """
    Generate the final structured document-level summary using the
    resource-type-specific template.
    """
    template = get_template(resource_type)
    headings = _extract_template_headings(template)
    log.info("  [Doc summary] Resource type: %s — %d sections", resource_type, len(headings))

    if dry_run:
        return {
            "resource_type": resource_type,
            "sections":      {h: "Dry run — no summary generated" for h in headings},
            "olmo_calls":    0,
        }

    prompt = _build_document_summary_prompt(chunk_summaries, resource_type, title, template)
    raw    = _call_ollama(prompt, SYSTEM_SUMMARISE, model)
    parsed = _parse_json_response(raw)

    if not parsed or not isinstance(parsed, dict):
        log.warning("  [Doc summary] Parse failed — returning empty summary")
        return {
            "resource_type": resource_type,
            "sections":      {},
            "olmo_calls":    1,
            "error":         "parse_failed",
        }

    # Post-process: clean up "Not described in the document." handling
    import re as _re
    cleaned = {}
    for heading, text in parsed.items():
        if not isinstance(text, str):
            cleaned[heading] = text
            continue

        # Fix: "Not described in the document is/are ..." → standalone sentence
        text = _re.sub(
            r"[Nn]ot described in the document (?:is|are)[^.]*\.",
            "Not described in the document.",
            text,
        )
        # Fix: embedded mid-sentence via comma
        text = _re.sub(
            r",\s*not described in the document\.",
            ". Not described in the document.",
            text,
            flags=_re.IGNORECASE,
        )

        # Remove "Not described in the document." if the section already has
        # 3 or more real sentences of content — it's being appended unnecessarily
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        real_sentences = [s for s in sentences if s.lower() != "not described in the document."]
        if len(real_sentences) >= 3:
            # Section has enough content — drop the unnecessary fallback
            text = " ".join(real_sentences)
        else:
            # Keep it but ensure it's at the end and not duplicated
            not_desc_count = sum(1 for s in sentences if s.lower() == "not described in the document.")
            if not_desc_count > 1:
                sentences = [s for s in sentences if s.lower() != "not described in the document."]
                sentences.append("Not described in the document.")
            text = " ".join(sentences)

        cleaned[heading] = text.strip()

    log.info("  [Doc summary] Done — %d sections filled", len(cleaned))
    return {
        "resource_type": resource_type,
        "sections":      cleaned,
        "olmo_calls":    1,
    }


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_summaries_json(
    resource_id:      str,
    source:           str,
    title:            str,
    url:              str,
    summaries:        list[dict],
    output_path:      str,
    document_summary: dict = None,
) -> None:
    """Write chunk summaries and document summary to a JSON file."""
    total_calls  = sum(s["olmo_calls"] for s in summaries)
    total_merged = sum(1 for s in summaries if s["is_merged"])
    total_errors = sum(1 for s in summaries if s["error"])

    doc = {
        "SUMMARY": {
            "resource_id":      resource_id,
            "source":           source,
            "title":            title,
            "url":              url,
            "total_groups":     len(summaries),
            "merged_groups":    total_merged,
            "total_olmo_calls": total_calls,
            "errors":           total_errors,
        },
        "document_summary":  document_summary or {},
        "chunk_summaries": [],
    }

    for s in summaries:
        entry = {
            "chunk_numbers":    s["chunk_numbers"],
            "part_label":       s["part_label"],
            "is_merged":        s["is_merged"],
            "token_count":      s["token_count"],
            "core_topic":       s["summary"].get("core_topic", "") if s["summary"] else "",
            "key_points":       s["summary"].get("key_points", []) if s["summary"] else [],
            "edi_relevance":    s["summary"].get("edi_relevance", "") if s["summary"] else "",
            "template_sections": s.get("template_sections") or {},
            "error":            s["error"],
        }
        doc["chunk_summaries"].append(entry)

    Path(output_path).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Summaries written → %s", output_path)

    # Console output
    W = 72
    print()
    print("┌" + "─"*(W-2) + "┐")
    print(f"│  SUMMARIES COMPLETE — Resource {resource_id}".ljust(W-1) + "│")
    print("├" + "─"*(W-2) + "┤")
    print(f"│  Title   : {title[:W-12]}".ljust(W-1) + "│")
    print(f"│  Groups  : {len(summaries)}  |  Merged: {total_merged}  |  OLMo calls: {total_calls}".ljust(W-1) + "│")
    if total_errors:
        print(f"│  ⚠ Errors: {total_errors}".ljust(W-1) + "│")
    print("├" + "─"*(W-2) + "┤")
    for s in summaries:
        nums  = str(s["chunk_numbers"])
        topic = s["summary"].get("core_topic", "parse failed")[:W-22] if s["summary"] else "error"
        merge = " [MERGED]" if s["is_merged"] else ""
        print(f"│  Chunks {nums:<10}{merge}".ljust(W-1) + "│")
        print(f"│    {topic}".ljust(W-1) + "│")
    print("└" + "─"*(W-2) + "┘")
    print(f"\nOutput → {output_path}\n")

# ---------------------------------------------------------------------------
# Chunk file loaders
# ---------------------------------------------------------------------------

def _load_single_doc_chunks(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s    = d["SUMMARY"]
    meta = {
        "resource_id": s["resource_id"],
        "title":       s["title"],
        "url":         s["url"],
        "source":      s["source"],
    }
    return meta, d["chunks"]


def _load_stage5_doc_chunks(path: str) -> tuple[dict, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s    = d["SUMMARY"]
    meta = {
        "resource_id": s["resource_id"],
        "title":       s["resource_title"],
        "url":         s["resource_url"],
        "source":      "stage5",
    }
    doc_chunks = {}
    for doc in d["documents"]:
        doc_chunks[doc["doc_id"]] = {
            "title":  doc["title"],
            "chunks": doc["chunks"],
        }
    return meta, doc_chunks

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Stage 6 summariser — generates per-chunk summaries using OLMo."
    )
    parser.add_argument("--resource-id", required=True, metavar="ID")
    parser.add_argument(
        "--source", required=True,
        choices=["stage3", "stage4", "stage5"],
    )
    parser.add_argument("--chunks-dir", default=CHUNKS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument(
        "--tagged-json", metavar="FILE", default=None,
        help="Path to tagged_<id>.json from stage6_tagger — used to pick the correct "
             "summary template. If omitted, defaults to 'Report or Article'.",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    rid    = args.resource_id
    source = args.source

    # ── Resolve resource type and template from tagged JSON ──────────────────
    resource_type = "Report or Article"  # default
    if args.tagged_json:
        try:
            with open(args.tagged_json, encoding="utf-8") as f:
                tagged_data = json.load(f)
            rt_tags = tagged_data.get("tags", {}).get("resource_type", [])
            if rt_tags:
                resource_type = rt_tags[0]["tag"]
                log.info("Resource type from tagger: %s", resource_type)
            else:
                log.warning("No resource_type tags found in %s — using default: %s",
                            args.tagged_json, resource_type)
        except Exception as e:
            log.warning("Could not load --tagged-json (%s) — using default: %s",
                        e, resource_type)
    else:
        log.info("No --tagged-json provided — using default resource type: %s", resource_type)

    template = get_template(resource_type)
    headings = _extract_template_headings(template)
    log.info("Template: %s — %d sections: %s", resource_type, len(headings), headings)

    # ── Stage 5: multi-document ──────────────────────────────────────────────
    if source == "stage5":
        chunk_file = Path(args.chunks_dir) / f"chunks_{rid}_pdfs.json"
        if not chunk_file.exists():
            chunk_file = Path(args.chunks_dir) / f"chunks_{rid}_1500.json"
        if not chunk_file.exists():
            log.error("Chunk file not found: %s", chunk_file)
            sys.exit(1)

        meta, doc_chunk_groups = _load_stage5_doc_chunks(str(chunk_file))
        log.info("Resource [%s]: %s — %d linked documents",
                 rid, meta["title"][:60], len(doc_chunk_groups))

        all_summaries = []
        for doc_id, doc_info in doc_chunk_groups.items():
            log.info("Summarising %s: %s (%d chunks)",
                     doc_id, doc_info["title"][:50], len(doc_info["chunks"]))
            groups = merge_continuation_chunks(doc_info["chunks"])
            for group in groups:
                s = summarise_chunk_group(group, args.model, args.dry_run,
                                          template=template, headings=headings,
                                          resource_type=resource_type)
                s["doc_id"]    = doc_id
                s["doc_title"] = doc_info["title"]
                all_summaries.append(s)

        suffix = "_pdfs"

    # ── Stage 3 / Stage 4: single document ───────────────────────────────────
    else:
        chunk_file = Path(args.chunks_dir) / f"chunks_{rid}.json"
        if not chunk_file.exists():
            chunk_file = Path(args.chunks_dir) / f"chunks_{rid}_1500.json"
        if not chunk_file.exists():
            log.error("Chunk file not found: %s", chunk_file)
            sys.exit(1)

        meta, raw_chunks = _load_single_doc_chunks(str(chunk_file))
        log.info("Resource [%s]: %s — %d chunks",
                 rid, meta["title"][:60], len(raw_chunks))

        groups        = merge_continuation_chunks(raw_chunks)
        log.info("After merging continuations: %d summarisation groups", len(groups))
        all_summaries = [
            summarise_chunk_group(g, args.model, args.dry_run,
                                  template=template, headings=headings,
                                  resource_type=resource_type)
            for g in groups
        ]
        suffix = "_web" if source == "stage3" else ""

    output_path = Path(args.output_dir) / f"summaries_{rid}{suffix}.json"
    write_summaries_json(
        resource_id=rid,
        source=source,
        title=meta["title"],
        url=meta["url"],
        summaries=all_summaries,
        output_path=str(output_path),
        document_summary={"resource_type": resource_type, "note": "Aggregation pending."},
    )


if __name__ == "__main__":
    _cli()