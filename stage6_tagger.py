"""
EDI Hub+ Pipeline — Stage 6 Tagger
=====================================
Semantic tagging of chunked EDI resources using OLMo-2 7B via Ollama.

Mechanism (per chunk):
    Step 1  — Semantic Analysis       : OLMo reads the chunk and builds a
                                        structured semantic profile (1 call)
    Step 2  — Dimension Tagging       : For each of the 7 taxonomy dimensions,
                                        OLMo is given the semantic profile +
                                        allowed tag list and proposes tags with
                                        evidence (7 calls, parallel if --workers>1)
    Step 3  — Hallucination Rescue    : OLMo self-reasons the closest valid tag
                                        for any proposed tag not in the allowed list
    Step 4  — Confidence Scoring      : high / medium / low / rejected per tag
    Step 5  — Map-Reduce Aggregation  : across all chunks, union tags, flag conflicts
    Step 6  — Final Output Record     : structured JSON per resource

Parallelism:
    Set OLLAMA_NUM_PARALLEL=N before starting Ollama, then pass --workers N.
    The semantic analysis (Step 1) always runs first sequentially.
    All 7 dimension calls then fire in parallel up to --workers concurrency.

Usage:
    python stage6_tagger.py --resource-id 39  --source stage4
    python stage6_tagger.py --resource-id 39  --source stage4 --workers 3
    python stage6_tagger.py --resource-id 203 --source stage3
    python stage6_tagger.py --resource-id 203 --source stage5 --workers 3
    python stage6_tagger.py --resource-id 39  --source stage4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage6_tagger")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL      = "http://localhost:11434/api/generate"
DEFAULT_MODEL   = "olmo2:7b"
OLLAMA_TIMEOUT  = 300
CHUNKS_DIR      = "."
OUTPUT_DIR      = "."

CONF_HIGH_EXACT  = "high"
CONF_MED_EXACT   = "medium"
CONF_LOW_RESCUED = "low"
CONF_REJECTED    = "rejected"

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY = {
    "resource_type": [
        "Academic Paper", "Blog Post", "Book Chapter", "Case Study/Testimonial",
        "EDI Intervention", "Evidence Document", "Guidance", "Legislation",
        "People Profile", "Policy Document", "Presentation or Poster",
        "Report or Article", "Resource Database", "Training Material",
    ],
    "resource_format": [
        "Database", "Dataset", "Document", "Mixed format",
        "Podcast", "Video", "Webinar", "Webpage",
    ],
    "individual_characteristics": [
        "Age", "Career Stage", "Caring Responsibilities", "Disability",
        "Education Status", "Gender", "Intersectionality", "Job Roles",
        "Linguistic", "Marriage and Civil Partnership", "Nationality",
        "Neurodiversity", "Physical Characteristics", "Pregnancy and Maternity",
        "Race and Ethnicity", "Religion and Belief", "Sex", "Sexual Orientation",
        "Socio-Economic Disadvantage", "Temporary Impairment", "Trans Identity",
        "Not Specified",
    ],
    "career_pathway": [
        "Career Mobility", "Career Progression", "Early Career", "Leadership",
        "Mentoring", "Mid-career", "Networking and Collaboration", "Recruitment",
        "Retention", "Visibility and Recognition",
    ],
    "organisational_culture": [
        "Accessibility", "Allyship and Advocacy", "Belonging and Engagement",
        "Bias", "Cultural Competency", "Data Collection and Monitoring",
        "Harassment Bullying and Microaggression", "Inclusive Leadership",
        "Inclusivity", "Representation", "Research Culture", "Research Design",
        "Strategy or Policy", "Work-Life Balance",
    ],
    "research_funding_process": [
        "Advertising", "Application Process", "Assessment Process",
        "Bias Mitigation", "Data Collection and Monitoring", "Eligibility Criteria",
        "Funding Guidance", "Pre and Post Award Support", "University Selection",
    ],
    "adoption_readiness_level": [
        "Initial Design", "Needs Identification", "Not applicable",
        "Pilot in Single Setting", "Proof of Concept",
        "Scaling in Multiple Settings", "Widespread Adoption",
    ],
}

MULTI_TAG_DIMENSIONS = {
    "individual_characteristics",
    "career_pathway",
    "organisational_culture",
    "research_funding_process",
}
SINGLE_TAG_DIMENSIONS = {
    "resource_type",
    "resource_format",
    "adoption_readiness_level",
}

_TYPO_MAP = {
    "caring responsiblities":                  "Caring Responsibilities",
    "netwotking and collaboration":            "Networking and Collaboration",
    "harassment, bullying and microagression": "Harassment Bullying and Microaggression",
    "harassment bullying and microagression":  "Harassment Bullying and Microaggression",
}


def _normalise_tag(tag: str) -> str:
    """
    Normalise a proposed tag for comparison.
    Converts snake_case/hyphen-case to spaced lowercase,
    then applies known typo corrections.
    """
    clean = tag.strip()
    # Convert snake_case and hyphen-case to space-separated lowercase
    # Handle mixed cases like work-life_balance
    if "_" in clean or "-" in clean:
        clean = clean.replace("_", " ").replace("-", " ")
    lower = clean.lower().strip()
    # Apply typo corrections
    if lower in _TYPO_MAP:
        return _TYPO_MAP[lower]
    return clean.strip()


def _find_valid_tag(proposed: str, dimension: str) -> str | None:
    """
    Find the canonical tag for a proposed string.
    Pass 1: raw case-insensitive match.
    Pass 2: normalise (snake_case fix + typo map) then case-insensitive match.
    Returns None if no match — never guesses.
    """
    allowed = TAXONOMY.get(dimension, [])

    # Pass 1: raw lowercase match
    raw_lower = proposed.strip().lower()
    for tag in allowed:
        if tag.lower() == raw_lower:
            return tag

    # Pass 2: normalised lowercase match (normalise both sides for comparison)
    norm_lower = _normalise_tag(proposed).lower().strip()
    for tag in allowed:
        # Normalise the allowed tag the same way for fair comparison
        tag_norm = tag.replace("-", " ").replace("_", " ").lower().strip()
        if tag_norm == norm_lower:
            return tag

    return None


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str, model: str) -> str | None:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "system":  system,
        "stream":  False,
        "options": {"temperature": 0.05, "num_predict": 1500},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        log.warning("Ollama HTTP %d: %s", resp.status_code, resp.text[:200])
        return None
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama. Is it running? (ollama serve)")
        return None
    except requests.exceptions.Timeout:
        log.error("Ollama timed out after %ds", OLLAMA_TIMEOUT)
        return None
    except Exception as exc:
        log.error("Ollama call failed: %s", exc)
        return None


def _parse_json_response(raw: str) -> dict | list | None:
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

SYSTEM_EDI = (
    "You are an expert EDI (Equality, Diversity and Inclusion) research analyst "
    "specialising in STEM workforce equity. You analyse academic and policy documents "
    "to understand their content deeply and classify them accurately. "
    "You always respond with valid JSON only — no preamble, no explanation outside the JSON."
)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_semantic_prompt(chunk_text: str, chunk_label: str) -> str:
    return f"""Analyse the following EDI resource chunk and produce a structured semantic profile.

Chunk label: {chunk_label}

--- CHUNK TEXT ---
{chunk_text}
--- END CHUNK ---

Produce a JSON object with exactly these keys:

{{
  "core_topic": "One sentence describing the fundamental subject of this chunk",
  "groups_mentioned": ["list of people, groups, roles, or communities explicitly or implicitly discussed"],
  "barriers_described": ["specific barriers, challenges, or problems mentioned"],
  "interventions_described": ["specific solutions, recommendations, or interventions mentioned"],
  "organisational_context": "The institutional or organisational setting (e.g. university, lab, funding body)",
  "career_research_stage": "Which stage of career or research process this relates to",
  "edi_themes": ["high-level EDI themes present, e.g. disability access, racial equity, gender inclusion"]
}}

IMPORTANT: Base your analysis on what the document ITSELF is arguing or recommending.
Do NOT include themes from quoted examples, sample texts, or illustrative excerpts cited within the chunk.
These show what the document is critiquing, not what it is about.

Respond with the JSON object only."""


# Tag definitions for single-value dimensions
_RESOURCE_TYPE_DEFS = {
    "Academic Paper":          "A peer-reviewed or scholarly research article",
    "Blog Post":               "An informal online article or opinion piece",
    "Book Chapter":            "A chapter from an academic or professional book",
    "Case Study/Testimonial":  "A real-world example or personal account of an EDI experience",
    "EDI Intervention":        "A programme, initiative, tool, or practical scheme actively designed to improve EDI outcomes",
    "Evidence Document":       "A document compiling research evidence or data to inform EDI policy",
    "Guidance":                "A practical how-to guide or set of recommendations for practitioners",
    "Legislation":             "A law, regulation, or legal framework",
    "People Profile":          "A profile or biography of an individual related to EDI",
    "Policy Document":         "An official institutional or organisational policy statement",
    "Presentation or Poster":  "Slides, poster, or visual presentation material",
    "Report or Article":       "A report or article summarising findings — NOT an active intervention or programme",
    "Resource Database":       "A collection or database of EDI resources",
    "Training Material":       "Content specifically designed for training or education",
}

_RESEARCH_FUNDING_SCOPE = """SCOPE OF THIS DIMENSION:
Research Funding Process refers to the process for funding research projects,
usually by a research funding body such as EPSRC, UKRI, Wellcome Trust, or similar.

Use only the Research Funding Process context.
Do NOT use People Recruitment context.

Valid — tag these:
  - How funding bodies advertise grants or funding calls
  - The application process for research grants
  - How grant applications are assessed or reviewed
  - Bias mitigation in the funding assessment process
  - Eligibility criteria for research funding
  - Guidance for applying to or managing research funding
  - Support provided to researchers before or after grant awards (pre/post award)
  - University selection processes used by funding bodies

NOT valid — do NOT tag these even if the word appears:
  - General workplace inclusion, HR practices, or people recruitment
  - Organisational culture or strategy documents
  - University selection of students or staff (not research funding)
  - Documents that only mention a funding body as commissioner or funder of the research
    but do not discuss the funding process itself
  - Lab access, language in job adverts, neuro-inclusion audits, or similar

If this chunk does not specifically discuss the research grant funding process,
return an empty proposed_tags list."""


_ADOPTION_DEFS = {
    "Initial Design":              "The resource is at the earliest design/planning stage — no implementation yet",
    "Needs Identification":        "The resource identifies EDI needs but has not designed a solution",
    "Not applicable":              "The resource is NOT an intervention — it is a report, evidence document, guidance, or legislation. If the semantic profile shows no concrete interventions or programmes, choose this.",
    "Pilot in Single Setting":     "The intervention has been tested in one organisation or setting",
    "Proof of Concept":            "A small-scale test has been completed to show the idea works",
    "Scaling in Multiple Settings":"The intervention has been rolled out across multiple organisations",
    "Widespread Adoption":         "The intervention is widely adopted across the sector",
}


# Option A — dedicated resource type intent prompt
def _build_resource_type_intent_prompt(
    semantic_profile: dict,
    chunk_text:       str,
) -> str:
    """
    Pre-classification prompt for Resource Type.
    Asks OLMo two questions about intent and purpose before choosing a tag.
    """
    defs_lines = "\n".join(f"  - {tag}: {defn}" for tag, defn in _RESOURCE_TYPE_DEFS.items())
    return f"""You are classifying the Resource Type of an EDI document chunk.

SEMANTIC PROFILE:
{json.dumps(semantic_profile, indent=2)}

CHUNK TEXT:
---
{chunk_text[:2000]}
---

Before choosing a tag, answer these two questions about the PRIMARY PURPOSE of this resource:

Q1: Does this resource primarily describe, report, analyse, or summarise findings about EDI?
    (e.g. presents data, describes a problem, reviews literature, reports on a study)
    Answer: yes or no

Q2: Does this resource primarily provide a practical tool, framework, intervention, programme,
    or actionable guidance that organisations can directly use to improve their EDI practice?
    (e.g. a toolkit, checklist, training programme, specific intervention)
    Answer: yes or no

TAG DEFINITIONS:
{defs_lines}

INSTRUCTIONS:
- If Q1=yes and Q2=no  → most likely Report or Article, Evidence Document, or Academic Paper
- If Q1=no  and Q2=yes → most likely EDI Intervention, Guidance, or Training Material
- If Q1=yes and Q2=yes → consider whether the PRIMARY purpose is reporting or intervening
- Use the semantic profile's "interventions_described" field — if it lists concrete tools
  or programmes, lean toward EDI Intervention or Guidance
- Choose EXACTLY ONE tag from the list below:
{chr(10).join(f"  - {t}" for t in _RESOURCE_TYPE_DEFS.keys())}

Respond with this JSON only:
{{
  "q1_reports_findings": true or false,
  "q2_provides_tool": true or false,
  "primary_purpose": "one sentence describing the primary purpose of this chunk",
  "proposed_tags": [
    {{
      "tag": "exact tag name from the list",
      "evidence": "direct quote or paraphrase supporting this choice",
      "reasoning": "why this tag best captures the primary purpose",
      "confidence": "high or medium"
    }}
  ],
  "not_applicable": false
}}"""


# Organisational culture tag definitions for Strategy or Policy
_ORG_CULTURE_DEFS = {
    "Strategy or Policy": (
        "The chunk discusses or recommends changes to institutional strategies, "
        "policies, hiring practices, or systemic organisational approaches — "
        "not just individual behaviours or attitudes"
    ),
}

_CAREER_PATHWAY_DEFS = {
    "Career Mobility": (
        "Movement between different roles, institutions, sectors, or career stages. "
        "NOT physical mobility or accessibility. Only tag if the chunk specifically "
        "discusses people moving between jobs, switching sectors, or transitioning careers."
    ),
    "Early Career": (
        "Specifically about people at the START of their career — PhD students, "
        "postdocs, new graduates, or early-stage researchers. "
        "NOT about early stages of a programme, project, or initiative."
    ),
}


def _build_dimension_prompt(
    semantic_profile: dict,
    chunk_text:       str,
    dimension:        str,
    allowed_tags:     list,
    is_multi:         bool,
) -> str:
    # Resource Type uses a dedicated intent-first prompt (Option A)
    if dimension == "resource_type":
        return _build_resource_type_intent_prompt(semantic_profile, chunk_text)

    # Resource Format — simple structural classification, no semantic inference needed
    if dimension == "resource_format":
        title   = semantic_profile.get("core_topic", "")
        org     = semantic_profile.get("organisational_context", "")
        preview = chunk_text[:300].replace("\n", " ")
        return (
            f'''Classify the format of this EDI resource. Choose exactly one from the list below.

Resource title/topic: {title}
Text preview: {preview}

FORMAT OPTIONS:
  - Document    : a written document, PDF, report, guide, paper, or audit
  - Webpage     : a web page or online article
  - Video       : a video or film
  - Podcast     : an audio recording or podcast
  - Webinar     : an online seminar or recorded webinar
  - Database    : a searchable collection of resources
  - Dataset     : a structured data file or dataset
  - Mixed format: combines two or more of the above

Respond with this JSON and nothing else:
{{"dimension": "resource_format", "proposed_tags": [{{"tag": "Document", "evidence": "this is a written document", "reasoning": "the resource is a written document", "confidence": "high"}}], "not_applicable": false}}

Replace the tag value with whichever format applies. Most EDI resources are Documents.''')

    tags_str = "\n".join(f"  - {t}" for t in allowed_tags)

    # Add definitions for single-value dimensions
    defs_block = ""
    if dimension == "adoption_readiness_level":
        defs_lines = "\n".join(f"  - {tag}: {defn}" for tag, defn in _ADOPTION_DEFS.items())
        pre_check  = (
            "STEP 1 — BEFORE choosing any tag, answer: Is this resource an active "
            "intervention, programme, or initiative organisations can implement to improve EDI? "
            "(yes/no)\n"
            "- If NO  → return 'Not applicable' only. Do not choose any other tag.\n"
            "- If YES → proceed to choose from the adoption scale below.\n\n"
            "NOT an intervention: reports, evidence reviews, guidance, legislation, audits.\n"
            "IS an intervention: a programme, toolkit, training scheme, structured initiative.\n\n"
        )
        defs_block = f"\n{pre_check}TAG DEFINITIONS:\n{defs_lines}\n"
    elif dimension == "organisational_culture":
        # Add specific definition for Strategy or Policy — most commonly missed
        extra_defs = "\n".join(f"  - {tag}: {defn}" for tag, defn in _ORG_CULTURE_DEFS.items())
        defs_block = f"\nKEY TAG CLARIFICATIONS:\n{extra_defs}\n"
    elif dimension == "career_pathway":
        extra_defs = "\n".join(f"  - {tag}: {defn}" for tag, defn in _CAREER_PATHWAY_DEFS.items())
        defs_block = f"\nKEY TAG CLARIFICATIONS (read carefully):\n{extra_defs}\n"
    elif dimension == "research_funding_process":
        # Inject the strict scope restriction to prevent over-tagging
        defs_block = f"\n{_RESEARCH_FUNDING_SCOPE}\n"

    if is_multi:
        multi_str = (
            "You may propose MULTIPLE tags — but ONLY if the chunk is genuinely and "
            "actively discussing each tag concept. Do not add a tag just because the "
            "topic could be loosely related. Ask yourself: is this chunk specifically "
            "about this concept, or is it just tangentially connected?"
        )
    else:
        multi_str = (
            "You must propose EXACTLY ONE tag from the list. "
            "Choose the single most accurate tag based on the definitions provided."
        )

    return f"""You are tagging an EDI resource chunk for the dimension: "{dimension}".

SEMANTIC PROFILE OF THE CHUNK:
{json.dumps(semantic_profile, indent=2)}

ORIGINAL CHUNK TEXT (for direct evidence):
---
{chunk_text[:2000]}
---

ALLOWED TAGS FOR THIS DIMENSION:
{tags_str}
{defs_block}
INSTRUCTIONS:
1. Use the semantic profile to understand what this specific chunk is genuinely about.
2. {multi_str}
3. CRITICAL: Only propose tags from the allowed list. Copy the tag name exactly as written — do NOT use underscores, do NOT change capitalisation.
4. For each proposed tag provide:
   - "tag": exact tag name from the allowed list
   - "evidence": a quote or paraphrase from the chunk text that relates to this tag
   - "reasoning": one sentence explaining the semantic connection between the text and this tag
   - "confidence": "high" if explicitly discussed, "medium" if requires semantic interpretation
5. If this chunk is not genuinely about this dimension, return an empty proposed_tags list.
6. Do NOT tag something just because the document broadly relates to it. Only tag what
   this specific chunk is actively discussing or arguing.

Respond with this JSON structure only:
{{
  "dimension": "{dimension}",
  "proposed_tags": [
    {{
      "tag": "exact tag name from allowed list",
      "evidence": "direct quote or paraphrase from chunk text",
      "reasoning": "why the chunk semantically maps to this tag",
      "confidence": "high or medium"
    }}
  ],
  "not_applicable": false
}}"""

def _build_rescue_prompt(
    proposed_tag: str,
    dimension:    str,
    allowed_tags: list,
    evidence:     str,
) -> str:
    tags_str = "\n".join(f"  - {t}" for t in allowed_tags)
    return f"""An EDI tagging model proposed the tag "{proposed_tag}" for the dimension "{dimension}".
This tag is NOT in the allowed list. Your job is to find the closest valid tag semantically.

EVIDENCE that caused this tag to be proposed:
"{evidence}"

ALLOWED TAGS FOR "{dimension}":
{tags_str}

INSTRUCTIONS:
1. First check: is "{proposed_tag}" simply a reformatted or misspelled version of an allowed tag?
   For example "race_and_ethnicity" is just "Race and Ethnicity" with underscores — return that tag immediately.
2. If not a reformatting, reason about the core semantic meaning of "{proposed_tag}" given the evidence.
3. Compare against each allowed tag. Only return a match if the meaning genuinely overlaps.
4. CRITICAL: Do not force a match. If "{proposed_tag}" does not meaningfully correspond
   to any allowed tag, return null. A wrong match is worse than no match.
5. Do not return a tag just because it is in the same general topic area.

Respond with this JSON only:
{{
  "proposed": "{proposed_tag}",
  "rescued_tag": "closest valid tag from the allowed list, or null if no genuine match",
  "rescue_reasoning": "step-by-step: is it a reformatting? if not, what is the semantic match or why is there none?"
}}"""




# ---------------------------------------------------------------------------
# Option 1 — Verification pass
# ---------------------------------------------------------------------------

def _build_verify_prompt(
    tag:        str,
    dimension:  str,
    evidence:   str,
    reasoning:  str,
    chunk_text: str,
) -> str:
    return f"""An EDI tagging model assigned the tag "{tag}" to the dimension "{dimension}" for this chunk.

EVIDENCE used to assign this tag:
"{evidence}"

REASONING given:
"{reasoning}"

CHUNK TEXT:
---
{chunk_text[:1500]}
---

Your job: verify whether this tag is genuinely justified.

A tag is JUSTIFIED if the chunk specifically and actively discusses the concept that "{tag}" represents.
A tag is NOT JUSTIFIED if:
- The chunk only mentions this concept in passing
- The chunk is about a related but different topic
- The connection requires multiple inferential steps
- The concept is only implied by the general EDI theme, not this specific chunk

Answer with this JSON only:
{{
  "tag": "{tag}",
  "dimension": "{dimension}",
  "justified": true or false,
  "reason": "one sentence explaining why this tag is or is not justified"
}}"""


def _verify_dimension_tags(
    accepted_tags: list,
    dimension:     str,
    chunk_text:    str,
    chunk_index:   int,
    model:         str,
) -> tuple:
    """
    Verification pass: drop any tag OLMo says is not genuinely justified.
    Returns (verified_tags, extra_calls_made).
    """
    if not accepted_tags:
        return accepted_tags, 0

    dim_label     = dimension.replace("_", " ").title()
    verified_tags = []
    extra_calls   = 0

    for tag_entry in accepted_tags:
        tag       = tag_entry["tag"]
        evidence  = tag_entry.get("evidence", "")
        reasoning = tag_entry.get("reasoning", "")

        verify_prompt = _build_verify_prompt(
            tag, dimension, evidence, reasoning, chunk_text
        )
        raw_verify  = _call_ollama(verify_prompt, SYSTEM_EDI, model)
        extra_calls += 1

        verify_resp = _parse_json_response(raw_verify)
        if not verify_resp:
            log.warning("  Chunk %02d [%s]: verify parse failed for '%s' — keeping",
                        chunk_index, dim_label, tag)
            verified_tags.append(tag_entry)
            continue

        justified = verify_resp.get("justified", True)
        reason    = verify_resp.get("reason", "")

        if justified:
            tag_entry["verified"]      = True
            tag_entry["verify_reason"] = reason
            verified_tags.append(tag_entry)
            log.debug("  Chunk %02d [%s]: ✓ verified '%s'", chunk_index, dim_label, tag)
        else:
            log.info("  Chunk %02d [%s]: ✗ dropped '%s' — %s",
                     chunk_index, dim_label, tag, reason)

    return verified_tags, extra_calls


# ---------------------------------------------------------------------------
# Single dimension tagger — extracted so it can run in parallel
# ---------------------------------------------------------------------------

def _tag_single_dimension(
    dim:              str,
    allowed_tags:     list,
    semantic_profile: dict,
    chunk_text:       str,
    chunk_index:      int,
    model:            str,
    verify:           bool = False,
) -> dict:
    """
    Tag one dimension for a chunk.
    Returns {"dim_result": {...}, "calls": N, "errors": [...]}
    If verify=True, runs a second OLMo call per tag to drop unjustified ones.
    Safe to call from multiple threads simultaneously.
    """
    dim_label = dim.replace("_", " ").title()
    is_multi  = dim in MULTI_TAG_DIMENSIONS
    calls     = 0
    errors    = []

    log.info("  Chunk %02d: tagging [%s]...", chunk_index, dim_label)

    dim_prompt   = _build_dimension_prompt(
        semantic_profile, chunk_text, dim, allowed_tags, is_multi
    )
    raw_dim = _call_ollama(dim_prompt, SYSTEM_EDI, model)
    calls  += 1

    dim_response = _parse_json_response(raw_dim)
    if not dim_response:
        log.warning("  Chunk %02d [%s]: parse failed", chunk_index, dim_label)
        errors.append(f"{dim}_parse_failed")
        return {
            "dim_result": {"tags": [], "status": "parse_failed", "raw": raw_dim},
            "calls": calls,
            "errors": errors,
        }

    not_applicable = dim_response.get("not_applicable", False)
    proposed_tags  = dim_response.get("proposed_tags", [])
    accepted_tags  = []

    for pt in proposed_tags:
        raw_tag   = pt.get("tag", "").strip()
        evidence  = pt.get("evidence", "")
        reasoning = pt.get("reasoning", "")
        conf      = pt.get("confidence", "medium").lower()

        valid_tag = _find_valid_tag(raw_tag, dim)

        if valid_tag:
            confidence = CONF_HIGH_EXACT if conf == "high" else CONF_MED_EXACT
            accepted_tags.append({
                "tag":        valid_tag,
                "evidence":   evidence,
                "reasoning":  reasoning,
                "confidence": confidence,
                "rescued":    False,
            })
            log.debug("  Chunk %02d [%s]: ✓ %s (%s)",
                      chunk_index, dim_label, valid_tag, confidence)
        else:
            log.info("  Chunk %02d [%s]: '%s' not in list — rescuing...",
                     chunk_index, dim_label, raw_tag)

            rescue_prompt = _build_rescue_prompt(raw_tag, dim, allowed_tags, evidence)
            raw_rescue    = _call_ollama(rescue_prompt, SYSTEM_EDI, model)
            calls        += 1

            rescue_resp = _parse_json_response(raw_rescue)
            if rescue_resp and rescue_resp.get("rescued_tag"):
                rescued_valid = _find_valid_tag(rescue_resp["rescued_tag"], dim)
                if rescued_valid:
                    accepted_tags.append({
                        "tag":               rescued_valid,
                        "evidence":          evidence,
                        "reasoning":         reasoning,
                        "rescue_reasoning":  rescue_resp.get("rescue_reasoning", ""),
                        "confidence":        CONF_LOW_RESCUED,
                        "rescued":           True,
                        "original_proposed": raw_tag,
                    })
                    log.info("  Chunk %02d [%s]: rescued '%s' → '%s'",
                             chunk_index, dim_label, raw_tag, rescued_valid)
                else:
                    log.warning("  Chunk %02d [%s]: rescue failed for '%s'",
                                chunk_index, dim_label, raw_tag)
                    errors.append(f"{dim}_rescue_failed:{raw_tag}")
            else:
                log.warning("  Chunk %02d [%s]: '%s' rejected",
                            chunk_index, dim_label, raw_tag)
                errors.append(f"{dim}_rejected:{raw_tag}")

    # Single-tag: keep highest-confidence only
    if dim in SINGLE_TAG_DIMENSIONS and len(accepted_tags) > 1:
        conf_order = {CONF_HIGH_EXACT: 3, CONF_MED_EXACT: 2,
                      CONF_LOW_RESCUED: 1, CONF_REJECTED: 0}
        accepted_tags = sorted(
            accepted_tags,
            key=lambda t: conf_order.get(t["confidence"], 0),
            reverse=True
        )[:1]

    # Option 1 — Verification pass
    if verify and accepted_tags and dim in MULTI_TAG_DIMENSIONS:
        accepted_tags, extra_calls = _verify_dimension_tags(
            accepted_tags, dim, chunk_text, chunk_index, model
        )
        calls += extra_calls

    return {
        "dim_result": {
            "tags":           accepted_tags,
            "not_applicable": not_applicable,
            "status":         "ok",
        },
        "calls":  calls,
        "errors": errors,
    }


def _tag_all_dimensions(
    semantic_profile: dict,
    chunk_text:       str,
    chunk_index:      int,
    model:            str,
    workers:          int  = 1,
    verify:           bool = False,
    dims_override:    dict = None,
) -> dict:
    """
    Tag dimensions — sequentially (workers=1) or in parallel (workers>1).
    dims_override: if provided, only tag these dimensions (used to skip doc-level dims).
    Returns {dim: {dim_result, calls, errors}}.
    """
    dims = list((dims_override or TAXONOMY).items())

    if workers <= 1:
        return {
            dim: _tag_single_dimension(
                dim, allowed_tags, semantic_profile, chunk_text, chunk_index, model,
                verify=verify,
            )
            for dim, allowed_tags in dims
        }

    log.info("  Chunk %02d: firing %d dimensions in parallel (workers=%d, verify=%s)",
             chunk_index, len(dims), workers, verify)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_dim = {
            executor.submit(
                _tag_single_dimension,
                dim, allowed_tags, semantic_profile, chunk_text, chunk_index, model,
                verify,
            ): dim
            for dim, allowed_tags in dims
        }
        for future in as_completed(future_to_dim):
            dim = future_to_dim[future]
            try:
                results[dim] = future.result()
            except Exception as exc:
                log.error("  Chunk %02d [%s]: parallel error: %s", chunk_index, dim, exc)
                results[dim] = {
                    "dim_result": {"tags": [], "status": "exception", "error": str(exc)},
                    "calls":  0,
                    "errors": [f"{dim}_exception:{exc}"],
                }

    # Restore taxonomy order
    return {dim: results[dim] for dim, _ in dims if dim in results}


# ---------------------------------------------------------------------------
# Document-level dimension cache
# Dims that are facts about the whole document — tag once, reuse across all chunks
# ---------------------------------------------------------------------------

# resource_format is a structural fact — tag once, reuse
# resource_type is tagged per-chunk so aggregation can majority-vote
_DOC_LEVEL_DIMS = {"resource_format"}


def tag_doc_level_dims(
    first_chunk_text:    str,
    first_chunk_label:   str,
    first_chunk_index:   int,
    model:               str,
    dry_run:             bool = False,
) -> dict:
    """
    Tag resource_type and resource_format using the first chunk only.
    Returns {dim: dim_result} to be reused across all chunks.
    """
    log.info("  [Doc-level] Tagging Resource Type and Resource Format once on chunk %02d",
             first_chunk_index)

    cached = {}
    # Need a minimal semantic profile for the format prompt
    sem_prompt = _build_semantic_prompt(first_chunk_text, first_chunk_label)
    raw_sem    = _call_ollama(sem_prompt, SYSTEM_EDI, model) if not dry_run else None
    semantic_profile = _parse_json_response(raw_sem) if raw_sem else {
        "core_topic": "", "groups_mentioned": [], "barriers_described": [],
        "interventions_described": [], "organisational_context": "",
        "career_research_stage": "", "edi_themes": [],
    }

    for dim in _DOC_LEVEL_DIMS:
        if dry_run:
            cached[dim] = {"dim_result": {"tags": [], "dry_run": True}, "calls": 0, "errors": []}
            continue
        result = _tag_single_dimension(
            dim=dim,
            allowed_tags=TAXONOMY[dim],
            semantic_profile=semantic_profile,
            chunk_text=first_chunk_text,
            chunk_index=first_chunk_index,
            model=model,
            verify=False,  # never verify structural dims
        )
        cached[dim] = result
        tags = [t["tag"] for t in result["dim_result"].get("tags", [])]
        log.info("  [Doc-level] %s → %s", dim, tags)

    return cached


# ---------------------------------------------------------------------------
# Core per-chunk tagger
# ---------------------------------------------------------------------------

def tag_chunk(
    chunk_text:       str,
    chunk_label:      str,
    chunk_index:      int,
    model:            str,
    dry_run:          bool  = False,
    workers:          int   = 1,
    verify:           bool  = False,
    doc_level_cache:  dict  = None,
) -> dict:
    """
    Full tagging pipeline for one chunk.
    workers=1        → sequential dimension calls (safe, default)
    workers>1        → parallel dimension calls via ThreadPoolExecutor
    verify=True      → Option 1: second OLMo call per tag to drop unjustified ones
    doc_level_cache  → pre-computed results for resource_type and resource_format
                       (tagged once on first chunk, reused for all subsequent chunks)
    """
    log.info("  Chunk %02d: semantic analysis...", chunk_index)
    result = {
        "chunk_index":      chunk_index,
        "chunk_label":      chunk_label,
        "semantic_profile": None,
        "dimensions":       {},
        "olmo_calls":       0,
        "errors":           [],
    }

    if dry_run:
        log.info("  [DRY RUN] Skipping OLMo calls")
        result["semantic_profile"] = {"dry_run": True}
        for dim in TAXONOMY:
            result["dimensions"][dim] = {"tags": [], "dry_run": True}
        return result

    # Step 1 — Semantic analysis (always sequential, must complete first)
    sem_prompt = _build_semantic_prompt(chunk_text, chunk_label)
    raw_sem    = _call_ollama(sem_prompt, SYSTEM_EDI, model)
    result["olmo_calls"] += 1

    semantic_profile = _parse_json_response(raw_sem)
    if not semantic_profile:
        log.warning("  Chunk %02d: semantic parse failed — using fallback", chunk_index)
        semantic_profile = {
            "core_topic":              "Could not parse semantic profile",
            "groups_mentioned":        [],
            "barriers_described":      [],
            "interventions_described": [],
            "organisational_context":  "",
            "career_research_stage":   "",
            "edi_themes":              [],
        }
        result["errors"].append("semantic_parse_failed")

    result["semantic_profile"] = semantic_profile
    log.info("  Chunk %02d: profile built — themes: %s",
             chunk_index, semantic_profile.get("edi_themes", [])[:3])

    # Steps 2-4 — Dimension tagging (parallel if workers > 1)
    # Skip doc-level dims if cached — they are document facts, not chunk facts
    dims_to_tag = {
        dim: tags for dim, tags in TAXONOMY.items()
        if dim not in _DOC_LEVEL_DIMS or doc_level_cache is None
    }

    dim_results = _tag_all_dimensions(
        semantic_profile=semantic_profile,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        model=model,
        workers=workers,
        verify=verify,
        dims_override=dims_to_tag,
    )

    # Inject cached doc-level results
    if doc_level_cache:
        for dim, cached_data in doc_level_cache.items():
            dim_results[dim] = cached_data
            log.debug("  Chunk %02d: using cached result for [%s]", chunk_index, dim)

    for dim, dim_data in dim_results.items():
        result["dimensions"][dim]  = dim_data["dim_result"]
        result["olmo_calls"]      += dim_data["calls"]
        result["errors"].extend(dim_data["errors"])

    log.info("  Chunk %02d: done — %d OLMo calls, %d error(s)",
             chunk_index, result["olmo_calls"], len(result["errors"]))
    return result


# ---------------------------------------------------------------------------
# Map-Reduce aggregation
# ---------------------------------------------------------------------------


# Funding body names — presence alone is not enough
_FUNDING_BODY_NAMES = [
    "epsrc", "ukri", "wellcome trust", "research council",
    "mrc ", "ahrc", "esrc", "nerc", "innovate uk", "horizon europe",
    "national institutes", "nsf ", "funding body", "funding bodies",
]

# Process keywords — at least one must appear alongside a funding body name
# to confirm the resource is actually about the funding process itself
_FUNDING_PROCESS_KEYWORDS = [
    "grant application", "grant proposal", "funding application",
    "funding call", "funding scheme", "funding round",
    "research grant", "grant award", "award process",
    "eligibility criteria", "peer review panel", "assessment panel",
    "funding guidance", "pre-award", "post-award",
    "application process", "funding process", "proposal submission",
]


def _document_mentions_funding_body(chunk_results: list) -> bool:
    """
    Return True only if the document discusses the research FUNDING PROCESS —
    not just mentions a funding body as a client, commissioner, or reference.

    Scans BOTH semantic profiles AND actual chunk text.
    Requires at least one funding process keyword to appear.
    A funding body name alone is not sufficient.
    """
    all_text = ""
    for chunk_res in chunk_results:
        # Scan semantic profile fields
        profile  = chunk_res.get("semantic_profile", {})
        themes   = " ".join(profile.get("edi_themes", [])).lower()
        org      = profile.get("organisational_context", "").lower()
        barriers = " ".join(profile.get("barriers_described", [])).lower()
        interv   = " ".join(profile.get("interventions_described", [])).lower()
        stage    = profile.get("career_research_stage", "").lower()

        # Also scan actual chunk text for process keywords
        chunk_text = " ".join(chunk_res.get("chunk_label", "").split()).lower()

        all_text += " " + themes + " " + org + " " + barriers + " " + interv + " " + stage + " " + chunk_text

    has_process = any(kw in all_text for kw in _FUNDING_PROCESS_KEYWORDS)

    if has_process:
        log.info("  [Funding check] Funding process keyword found — Research Funding applies")
        return True

    log.info("  [Funding check] No funding process keywords found — clearing Research Funding tags")
    return False


def aggregate_chunk_results(
    chunk_results:  list,
    resource_meta:  dict,
    min_chunks:     int  = 1,
    min_confidence: str  = "medium",
) -> dict:
    """
    Union all chunk tags per dimension, promote confidence on repeats.
    Option 3 filtering (min_chunks, min_confidence) applied after aggregation.
    """
    _CONF_ORDER = {"high": 3, "medium": 2, "low": 1, "rejected": 0}
    min_conf_score = _CONF_ORDER.get(min_confidence, 2)
    aggregated = {dim: {} for dim in TAXONOMY}

    for chunk_res in chunk_results:
        chunk_idx = chunk_res["chunk_index"]
        for dim, dim_data in chunk_res.get("dimensions", {}).items():
            for tag_entry in dim_data.get("tags", []):
                tag  = tag_entry["tag"]
                conf = tag_entry["confidence"]
                if tag not in aggregated[dim]:
                    aggregated[dim][tag] = {
                        "tag":           tag,
                        "confidence":    conf,
                        "chunk_count":   1,
                        "chunk_indices": [chunk_idx],
                        "evidence":      [tag_entry.get("evidence", "")],
                        "rescued":       tag_entry.get("rescued", False),
                    }
                else:
                    existing = aggregated[dim][tag]
                    existing["chunk_count"] += 1
                    existing["chunk_indices"].append(chunk_idx)
                    ev = tag_entry.get("evidence", "")
                    if ev and ev not in existing["evidence"]:
                        existing["evidence"].append(ev)
                    if existing["chunk_count"] > 1:
                        existing["confidence"] = CONF_HIGH_EXACT

    final_tags     = {}
    conflict_flags = []

    for dim, tag_map in aggregated.items():
        tags_list = list(tag_map.values())

        # Option 3 — smarter threshold filter for multi-value dimensions
        # Rule: keep a tag if EITHER:
        #   (a) it appears in 2+ chunks (seen consistently)  OR
        #   (b) it appears in 1 chunk BUT with high confidence (specific, direct evidence)
        # This prevents dropping valid single-mention tags (e.g. Trans Identity in one section)
        # while still removing loose medium/low confidence over-tags
        if dim in MULTI_TAG_DIMENSIONS:
            before    = len(tags_list)
            tags_list = [
                t for t in tags_list
                if t["chunk_count"] >= min_chunks                              # (a) seen in 2+ chunks
                or (
                    t["chunk_count"] == 1                                       # (b) single chunk but
                    and _CONF_ORDER.get(t["confidence"], 0) >= 3               #     high confidence only
                )
            ]
            dropped = before - len(tags_list)
            if dropped > 0:
                log.info("  [Threshold] %s: dropped %d tag(s) "
                         "(rule: 2+ chunks OR single-chunk high-conf)",
                         dim, dropped)

        if dim in SINGLE_TAG_DIMENSIONS and len(tags_list) > 1:
            conflict_flags.append({
                "dimension": dim,
                "conflict":  [t["tag"] for t in tags_list],
                "note":      "Multiple tags for single-value dimension — human review needed",
            })
            tags_list = sorted(
                tags_list,
                key=lambda t: (
                    t["chunk_count"],
                    {"high": 3, "medium": 2, "low": 1}.get(t["confidence"], 0)
                ),
                reverse=True
            )[:1]
        final_tags[dim] = tags_list

    # Fix 2 — clear Research Funding if no funding process keywords found
    if final_tags.get("research_funding_process"):
        if not _document_mentions_funding_body(chunk_results):
            dropped = [t["tag"] for t in final_tags["research_funding_process"]]
            log.info("  [Post-process] Research Funding cleared — no funding process keywords. "
                     "Dropped: %s", dropped)
            final_tags["research_funding_process"] = []

    # Fix 7 — Adoption Readiness post-processing
    _NOT_APPLICABLE_TYPES = {
        "Guidance", "Legislation", "Evidence Document",
        "Academic Paper", "Policy Document",
    }
    _IMPLEMENTATION_KEYWORDS = [
        "piloted", "pilot study", "rolled out", "implemented at",
        "scaling across", "participating institution", "programme delivered",
        "cohort", "participants completed", "has been adopted",
        "widespread adoption", "multiple settings", "multiple organisations",
    ]

    resource_type_tags = [t["tag"] for t in final_tags.get("resource_type", [])]
    winning_type       = resource_type_tags[0] if resource_type_tags else ""
    force_not_applicable = False

    if winning_type in _NOT_APPLICABLE_TYPES:
        force_not_applicable = True
        log.info("  [Post-process] Adoption Readiness → Not applicable (Resource Type: %s)",
                 winning_type)
    elif winning_type in {"Report or Article", "Resource Database"}:
        all_chunk_text = " ".join(
            cr.get("chunk_label", "") for cr in chunk_results
        ).lower()
        for cr in chunk_results:
            profile = cr.get("semantic_profile", {})
            all_chunk_text += " " + " ".join(
                profile.get("interventions_described", [])
            ).lower()
        has_impl = any(kw in all_chunk_text for kw in _IMPLEMENTATION_KEYWORDS)
        if not has_impl:
            force_not_applicable = True
            log.info("  [Post-process] Adoption Readiness → Not applicable "
                     "(Report or Article, no implementation keywords)")

    if force_not_applicable:
        existing = [t["tag"] for t in final_tags.get("adoption_readiness_level", [])]
        if existing != ["Not applicable"]:
            log.info("  [Post-process] Overriding Adoption Readiness: %s → Not applicable",
                     existing)
            final_tags["adoption_readiness_level"] = [{
                "tag":           "Not applicable",
                "confidence":    "high",
                "chunk_count":   1,
                "chunk_indices": [1],
                "evidence":      [f"Resource Type is {winning_type or 'non-intervention'} "
                                  "— not an active intervention"],
                "rescued":       False,
            }]

    total_tags    = sum(len(v) for v in final_tags.values())
    total_rescued = sum(1 for v in final_tags.values() for t in v if t.get("rescued"))
    total_errors  = sum(len(r.get("errors", [])) for r in chunk_results)
    total_calls   = sum(r.get("olmo_calls", 0) for r in chunk_results)

    return {
        "resource_id":  resource_meta.get("resource_id", ""),
        "title":        resource_meta.get("title", ""),
        "url":          resource_meta.get("url", ""),
        "source":       resource_meta.get("source", ""),
        "chunk_count":  len(chunk_results),
        "tags":         final_tags,
        "conflicts":    conflict_flags,
        "stats": {
            "total_tags_assigned": total_tags,
            "tags_rescued":        total_rescued,
            "total_errors":        total_errors,
            "total_olmo_calls":    total_calls,
        },
    }


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_tagged_json(result: dict, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Tagged output written → %s", output_path)

    W = 72
    print()
    print("┌" + "─" * (W-2) + "┐")
    print(f"│  TAGGING COMPLETE — Resource {result['resource_id']}".ljust(W-1) + "│")
    print("├" + "─" * (W-2) + "┤")
    print(f"│  Title   : {result['title'][:W-12]}".ljust(W-1) + "│")
    print(f"│  Chunks  : {result['chunk_count']}  |  OLMo calls: {result['stats']['total_olmo_calls']}".ljust(W-1) + "│")
    print(f"│  Tags    : {result['stats']['total_tags_assigned']} assigned  |  {result['stats']['tags_rescued']} rescued  |  {result['stats']['total_errors']} error(s)".ljust(W-1) + "│")
    if result["conflicts"]:
        print(f"│  Conflicts: {len(result['conflicts'])} dimension(s) need human review".ljust(W-1) + "│")
    print("├" + "─" * (W-2) + "┤")
    print(f"│  {'Dimension':<30}  Tags".ljust(W-1) + "│")
    print("│  " + "─" * (W-4) + "  │")
    for dim, tags in result["tags"].items():
        tag_names = ", ".join(t["tag"] for t in tags) if tags else "—"
        dim_label = dim.replace("_", " ").title()[:28]
        print(f"│  {dim_label:<30}  {tag_names[:W-36]}".ljust(W-1) + "│")
    print("└" + "─" * (W-2) + "┘")
    print(f"\nOutput → {output_path}\n")


# ---------------------------------------------------------------------------
# Chunk file loaders
# ---------------------------------------------------------------------------

def _load_single_doc_chunks(path: str) -> tuple:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s    = d["SUMMARY"]
    meta = {
        "resource_id": s["resource_id"],
        "title":       s["title"],
        "url":         s["url"],
        "source":      s["source"],
    }
    chunks = []
    for c in d["chunks"]:
        text = "\n".join(c.get("text_lines", []))
        chunks.append({
            "index": c["chunk_number"],
            "label": c["part_label"],
            "text":  text,
        })
    return meta, chunks


def _load_stage5_doc_chunks(path: str) -> tuple:
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
        chunks = []
        for c in doc["chunks"]:
            text = "\n".join(c.get("text_lines", []))
            chunks.append({
                "index":     c["chunk_number"],
                "label":     c["part_label"],
                "text":      text,
                "doc_id":    doc["doc_id"],
                "doc_title": doc["title"],
            })
        doc_chunks[doc["doc_id"]] = chunks
    return meta, doc_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Stage 6 semantic tagger — tags EDI resource chunks using OLMo."
    )
    parser.add_argument("--resource-id", required=True, metavar="ID")
    parser.add_argument(
        "--source", required=True,
        choices=["stage3", "stage4", "stage5"],
        help="stage3=parent webpage  stage4=PDF  stage5=linked PDFs",
    )
    parser.add_argument("--chunks-dir",  default=CHUNKS_DIR)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Parallel dimension calls per chunk (default: 1 = sequential). "
            "Set OLLAMA_NUM_PARALLEL to the same value before starting Ollama. "
            "Recommended: 3 for RTX 3060."
        ),
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Option 1: verification OLMo call per tag to drop unjustified tags. "
             "Best precision — adds up to 7 extra OLMo calls per chunk.",
    )
    parser.add_argument(
        "--min-chunks", type=int, default=1, metavar="N",
        help="Option 3: only keep tags seen in at least N chunks (default 1). "
             "Try --min-chunks 2 to reduce over-tagging without extra OLMo calls.",
    )
    parser.add_argument(
        "--min-confidence", default="medium", choices=["high", "medium", "low"],
        help="Option 3: minimum confidence to keep a tag (default medium).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Load chunks but skip all OLMo calls")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress DEBUG logs")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    rid    = args.resource_id
    source = args.source

    log.info("Workers=%d | Verify=%s | MinChunks=%d | MinConf=%s | Model=%s | DryRun=%s",
             args.workers, args.verify, args.min_chunks,
             args.min_confidence, args.model, args.dry_run)

    # ── Stage 5 — multi-document ─────────────────────────────────────────────
    if source == "stage5":
        chunk_file = Path(args.chunks_dir) / f"chunks_{rid}_pdfs.json"
        if not chunk_file.exists():
            chunk_file = Path(args.chunks_dir) / f"chunks_{rid}_1500.json"
        if not chunk_file.exists():
            log.error("Chunk file not found: %s", chunk_file)
            sys.exit(1)

        log.info("Loading stage5 chunks from: %s", chunk_file)
        meta, doc_chunk_groups = _load_stage5_doc_chunks(str(chunk_file))

        all_chunk_results = []
        for doc_id, chunks in doc_chunk_groups.items():
            doc_title = chunks[0].get("doc_title", doc_id) if chunks else doc_id
            log.info("Tagging %s: %s (%d chunks)", doc_id, doc_title[:50], len(chunks))
            # Tag doc-level dims once per PDF document
            doc_cache_s5 = tag_doc_level_dims(
                first_chunk_text=chunks[0]["text"],
                first_chunk_label=f"{doc_id} | {chunks[0]['label']}",
                first_chunk_index=chunks[0]["index"],
                model=args.model,
                dry_run=args.dry_run,
            ) if chunks else {}
            for chunk in chunks:
                chunk_result = tag_chunk(
                    chunk_text=chunk["text"],
                    chunk_label=f"{doc_id} | {chunk['label']}",
                    chunk_index=chunk["index"],
                    model=args.model,
                    dry_run=args.dry_run,
                    workers=args.workers,
                    verify=args.verify,
                    doc_level_cache=doc_cache_s5,
                )
                chunk_result["doc_id"]    = doc_id
                chunk_result["doc_title"] = doc_title
                all_chunk_results.append(chunk_result)

        final = aggregate_chunk_results(all_chunk_results, meta,
                    min_chunks=args.min_chunks, min_confidence=args.min_confidence)

    # ── Stage 3 / Stage 4 — single document ──────────────────────────────────
    else:
        chunk_file = Path(args.chunks_dir) / f"chunks_{rid}.json"
        if not chunk_file.exists():
            chunk_file = Path(args.chunks_dir) / f"chunks_{rid}_1500.json"
        if not chunk_file.exists():
            log.error("Chunk file not found: %s", chunk_file)
            sys.exit(1)

        log.info("Loading %s chunks from: %s", source, chunk_file)
        meta, chunks = _load_single_doc_chunks(str(chunk_file))

        log.info("Resource [%s]: %s", rid, meta["title"][:60])
        log.info("Chunks: %d | Workers: %d | Model: %s | Dry run: %s",
                 len(chunks), args.workers, args.model, args.dry_run)

        # Tag resource_type and resource_format once on first chunk
        doc_cache = tag_doc_level_dims(
            first_chunk_text=chunks[0]["text"],
            first_chunk_label=chunks[0]["label"],
            first_chunk_index=chunks[0]["index"],
            model=args.model,
            dry_run=args.dry_run,
        ) if chunks else {}

        chunk_results = []
        for chunk in chunks:
            chunk_result = tag_chunk(
                chunk_text=chunk["text"],
                chunk_label=chunk["label"],
                chunk_index=chunk["index"],
                model=args.model,
                dry_run=args.dry_run,
                workers=args.workers,
                verify=args.verify,
                doc_level_cache=doc_cache,
            )
            chunk_results.append(chunk_result)

        final = aggregate_chunk_results(chunk_results, meta,
                    min_chunks=args.min_chunks, min_confidence=args.min_confidence)

    # ── Write output ──────────────────────────────────────────────────────────
    suffix      = "_web" if source == "stage3" else ("_pdfs" if source == "stage5" else "")
    output_path = Path(args.output_dir) / f"tagged_{rid}{suffix}.json"
    write_tagged_json(final, str(output_path))


if __name__ == "__main__":
    _cli()