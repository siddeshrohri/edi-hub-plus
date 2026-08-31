# EDI Hub+ Resource Pipeline

An automated, end-to-end pipeline for discovering, scraping, extracting, tagging, and summarising Equality, Diversity and Inclusion (EDI) resources for the **EDI Hub+ Resource Centre**, funded by EPSRC's *Inclusion Matters* programme as part of the broader STEMM Change portfolio.

The pipeline takes a spreadsheet of EDI resource URLs, fetches and cleans their content, follows linked documents one level deep, and uses **OLMo-2 7B** (running locally via Ollama) to generate structured taxonomy tags and multi-section summaries, all without sending any data to external APIs.

> **Pilot evaluation results (3 resources):** F1 0.59 · Precision 0.58 · Recall 0.60

---

## Table of Contents

1. [Background](#background)
2. [Pipeline Overview](#pipeline-overview)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the Pipeline](#running-the-pipeline)
7. [Stage Reference](#stage-reference)
8. [EDI Taxonomy](#edi-taxonomy)
9. [LLM Architecture](#llm-architecture)
10. [Evaluation](#evaluation)
11. [Configuration Reference](#configuration-reference)
12. [Status Codes and Failure Reasons](#status-codes-and-failure-reasons)
13. [Ethical Scraping](#ethical-scraping)
14. [Known Limitations](#known-limitations)
15. [Team](#team)

---

## Background

The EDI Hub+ Resource Centre surfaces curated EDI resources for researchers, academics, and institutions within UK engineering, physical, and mathematical sciences (EPSRC remit). Previously, resources were tagged and summarised entirely by hand. This pipeline automates that process using a locally-run open-source LLM, chosen specifically for its strong ethical sourcing credentials.

**Model selection:** Five candidate models were evaluated against a 22-criterion ethical sourcing framework (covering provenance/consent/licensing, dataset documentation, transparency/auditability, labour conditions, and environmental impact). OLMo-2 7B scored highest at 94/110, which is why it was selected. See the EPSRC presentation materials for the full scoring breakdown.

---

## Pipeline Overview

```
Excel spreadsheet (resource URLs)
        │
        ▼
┌──────────────┐
│   Stage 1    │  Discover — read Excel, validate URLs, build Resource list
└──────┬───────┘
       │  stage1_resources.json
       ▼
┌──────────────┐
│   Stage 2    │  Fetch & Classify — robots.txt check, rate-limited fetch,
└──────┬───────┘  JS-heavy detection, raw HTML storage
       │  stage2_resources.json  +  stage2_skipped.json
       ▼
┌──────────────┐
│   Stage 3    │  Web Cleaning — BeautifulSoup, boilerplate removal,
└──────┬───────┘  structured text output, inline token profiling
       │  stage3_resources.json  (+  stage3_resources/{id}.json if --separate)
       ▼
┌──────────────┐
│   Stage 4    │  PDF Extraction — PyMuPDF → pdfplumber → pypdf fallback,
└──────┬───────┘  local disk cache, inline token profiling
       │  stage4_resources.json
       ▼
┌──────────────┐
│   Stage 5    │  Link Following — external links + linked PDFs one level deep,
└──────┬───────┘  full provenance tracing, disk cache for linked PDFs
       │  stage5_resources.json
       ▼
┌──────────────────────────────┐
│          Stage 6             │
│  ┌────────┐  ┌────────────┐  │  Chunker  — structural split + 175-token overlap
│  │Chunker │→ │  Tagger   │  │  Tagger   — 6-step per-chunk taxonomy tagging
│  └────────┘  └─────┬──────┘  │  Summariser — 3-pass fact extraction + summarisation
│               ┌────▼──────┐  │
│               │Summariser │  │
│               └───────────┘  │
└──────────────────────────────┘
       │  stage6_tagged.json
       ▼
┌──────────────┐
│   eval.py    │  Evaluation — P / R / F1 / Exact Match vs ground truth Excel
└──────────────┘
       │  eval_results.json  +  eval_results.txt
```

---

## Project Structure

```
edi-hub-plus/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config.py                    # Global settings — rate limits, paths, model config
├── models.py                    # Resource dataclass shared across all stages
│
├── main.py                      # Stage 1 entry point
├── stage2.py                    # Stage 2 entry point
├── stage3.py                    # Stage 3 entry point
├── stage4.py                    # Stage 4 entry point
├── stage5.py                    # Stage 5 entry point
├── stage6.py                    # Stage 6 orchestrator (calls chunker, tagger, summariser)
├── stage6_chunker.py            # Sliding window chunker (175-token overlap)
├── stage6_tagger.py             # Per-chunk taxonomy tagger (ThreadPoolExecutor)
├── stage6_summariser.py         # 3-pass summarisation (fact extract → summarise → verify)
├── eval.py                      # Evaluation: P / R / F1 / Exact Match vs ground truth
│
├── scraper/
│   ├── __init__.py
│   ├── classifier.py            # URL type detection + JS-heavy page detection
│   ├── discovery.py             # Excel → Resource list (Stage 1)
│   ├── fetcher.py               # Fetch orchestrator (Stage 2)
│   └── http_client.py           # robots.txt, rate limiting, session management
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Dual console (INFO) + file (DEBUG) logger
│   └── summary_templates.py     # 14 resource-type-specific summary templates
│
├── data/                        # Source Excel (not committed — obtain from team)
│   └── resources.xlsm
│
├── pdfs/                        # Downloaded PDFs, local disk cache (gitignored)
├── pdfs_linked/                 # Linked PDFs: {parent_id}_{url_hash}.pdf (gitignored)
├── stage3_resources/            # Per-resource Stage 3 JSON if --separate (gitignored)
│
├── outputs/                     # All pipeline JSON outputs (gitignored)
│   ├── stage1_resources.json
│   ├── stage2_resources.json
│   ├── stage2_skipped.json
│   ├── stage3_resources.json
│   ├── stage4_resources.json
│   ├── stage5_resources.json
│   ├── stage6_chunks.json
│   └── stage6_tagged.json
│
└── logs/
    └── pipeline.log             # Full DEBUG log of every pipeline run (gitignored)
```

---

## Prerequisites

### Python 3.11+

Check your version:
```bash
python --version
```

### Ollama + OLMo-2 7B

Stage 6 calls OLMo-2 7B via Ollama's local HTTP API (`localhost:11434`). No data leaves your machine.

1. Install Ollama from https://ollama.com
2. Pull the model (approx 4.5 GB download):
   ```bash
   ollama pull olmo2:7b
   ```
3. Start the Ollama server (must be running before Stage 6):
   ```bash
   ollama serve
   ```

### Source Data

Obtain the Excel resource file from the project team (Prof. Vania Dimitrova / Gabby Keating) and place it at:

```
data/resources.xlsm
```

The pipeline reads from a sheet named **`Resources-to-show`** with these columns:

| Column | Description |
|--------|-------------|
| `Resource ID` | Unique identifier (e.g. `46`) |
| `Title` | Display title of the resource |
| `Author` | Author or organisation |
| `URL` | Direct URL to the resource or PDF |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/siddeshrohri/edi-hub-plus.git
cd edi-hub-plus

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch (choose one based on your hardware)
# CPU only (sufficient — OLMo-2 runs via Ollama, not PyTorch directly):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Create output directories
mkdir outputs logs pdfs pdfs_linked stage3_resources data
```

---

## Running the Pipeline

Run each stage in order. Each stage reads the previous stage's JSON and writes its own.

### Quick start (run everything)

```bash
python main.py --excel data/resources.xlsm
python stage2.py
python stage3.py
python stage4.py
python stage5.py
ollama serve &   # ensure Ollama is running
python stage6.py
python eval.py
```

---

## Stage Reference

### Stage 1 — Discovery

Reads the Excel file, validates URLs, and produces a clean list of `Resource` objects.

```bash
python main.py --excel data/resources.xlsm
```

| Output | Description |
|--------|-------------|
| `outputs/stage1_resources.json` | All resources with title, author, URL, resource ID |

Resources without a valid `http(s)://` URL are skipped and logged.

---

### Stage 2 — Fetch and Classify

For each resource: checks `robots.txt`, applies rate limiting, fetches the page, detects JS-heavy pages, and stores raw HTML for Stage 3.

```bash
python stage2.py
```

| Output | Description |
|--------|-------------|
| `outputs/stage2_resources.json` | All resources with status and raw HTML |
| `outputs/stage2_skipped.json` | PDFs, JS-heavy pages, and failed fetches |

**Typical results on the full resource set:**
- ✅ ~15 successful fetches
- 📄 ~8 PDF skips (handled by Stage 4)
- ⚡ ~1 JS-heavy skip
- ❌ ~8 failures (mostly `robots-blocked` from rsc.org, raeng.org.uk, qub.ac.uk, ntu.ac.uk, britishscienceassociation.org, northernpowerinclusion.org)

---

### Stage 3 — Web Page Cleaning

Strips boilerplate (nav bars, footers, cookie banners, scripts) from raw HTML and produces structured plain text. Also profiles character count and token count inline for Stage 6 routing.

```bash
python stage3.py

# Write a separate JSON file per resource (useful for debugging):
python stage3.py --separate
```

Output text structure per resource:
```
TITLE: ...
DESCRIPTION: ...
HEADINGS: ...
CONTENT: ...
PDF LINKS: ...
EXTERNAL LINKS: ...
INTERNAL LINKS: ...
```

| Output | Description |
|--------|-------------|
| `outputs/stage3_resources.json` | Cleaned text + token counts for all web resources |
| `stage3_resources/{id}.json` | Per-resource file (only with `--separate`) |

---

### Stage 4 — PDF Extraction

Extracts text from PDF resources using a three-extractor fallback chain. PDFs are downloaded once and cached to disk — re-runs use the local copy.

**Extractor chain:** PyMuPDF (layout-aware, primary) → pdfplumber → pypdf

```bash
python stage4.py
```

| Output | Description |
|--------|-------------|
| `outputs/stage4_resources.json` | Extracted text + token counts for all PDF resources |
| `pdfs/{filename}.pdf` | Local PDF cache |

> Resources 65 (Warwick, login-walled) and 216 (Tech Talent Charter, HTTP 403) are flagged for manual follow-up.

---

### Stage 5 — Link Following

Follows external links and linked PDFs one level deep from each Stage 3 resource. Every piece of extracted content is traceable back to its parent resource ID, parent title, link text, URL, and link type.

```bash
python stage5.py

# Limit links followed per resource (default: 20 each):
python stage5.py --max-ext 10 --max-pdf 10

# Run for a single resource only:
python stage5.py --resource-id 46

# List links without fetching (dry run):
python stage5.py --dry-run
```

| Output | Description |
|--------|-------------|
| `outputs/stage5_resources.json` | Linked content with full provenance |
| `pdfs_linked/{id}_{hash}.pdf` | Linked PDFs cached with provenance in filename |

---

### Stage 6 — LLM Tagging and Summarisation

The most complex stage. Internally calls three sub-modules in sequence:

#### Stage 6a — Chunker (`stage6_chunker.py`)

Splits each resource's combined text (parent + linked content) into chunks using structural section markers first, then 175-token sliding window overlap for continuity. Document-level metadata (resource format) is tagged once on the first chunk and reused.

**Routing:** resources with `combined_token_count` above threshold → map-reduce; below threshold → single pass.

#### Stage 6b — Tagger (`stage6_tagger.py`)

Six-step per-chunk process:
1. Semantic analysis — identify EDI themes in the chunk
2. Parallel dimension tagging — all 7 taxonomy dimensions via `ThreadPoolExecutor`
3. Hallucination rescue — re-prompt on low-confidence or empty dimensions
4. Confidence scoring — each tag rated `high` / `medium` / `low` / `rejected`
5. Verification pass — cross-check tags against chunk text, remove unsupported tags
6. Map-reduce aggregation — merge chunk-level tags into resource-level tags

Fabrication detection: tags are checked against a 60% word-overlap threshold against the mini-summaries corpus before validation.

`.gov.uk` URLs trigger a hard sector override to `Government` in Pass 3 scalar validation.

#### Stage 6c — Summariser (`stage6_summariser.py`)

Three-pass architecture:
1. **Fact extraction** — identify key claims, statistics, and evidence from each chunk
2. **Constrained summarisation** — generate a structured summary using one of 14 resource-type-specific templates (from `summary_templates.py`)
3. **Verification** — check summary claims are supported by the extracted facts; remove any unsupported statements

```bash
# Ensure Ollama is running: ollama serve

python stage6.py

# Test on a single resource (recommended before a full run):
python stage6.py --test 46
```

| Output | Description |
|--------|-------------|
| `outputs/stage6_chunks.json` | Chunked text per resource |
| `outputs/stage6_tagged.json` | Final tags, tag evidence, and summaries per resource |

---

### Evaluation (`eval.py`)

Computes Precision, Recall, F1, and Exact Match for each of the 7 taxonomy dimensions against human ground truth annotations from the Excel file. Outputs both a human-readable text report and a machine-readable JSON.

```bash
python eval.py
```

| Output | Description |
|--------|-------------|
| `outputs/eval_results.json` | Per-resource, per-dimension TP/FP/FN + P/R/F1 |
| `outputs/eval_results.txt` | Human-readable evaluation grid |

---

## EDI Taxonomy

Stage 6 tags each resource against 7 dimensions. **Sub-tags only are valid** — top-level dimension names alone are never used as tags.

| Dimension | What it captures |
|-----------|-----------------|
| **Resource Type** | What kind of resource it is — report, toolkit, case study, training programme, policy document, research paper, dataset… |
| **Resource Format** | How it is delivered — PDF, webpage, video, interactive tool, workshop… |
| **Individual Characteristics** | Protected characteristics and identities addressed — gender, race/ethnicity, disability, age, sexuality, religion, socioeconomic background… |
| **Career Pathway** | Stage of academic or professional career addressed — PhD, early career researcher, mid-career, senior leadership, professional services… |
| **Organisational Culture** | Institutional and structural EDI themes — inclusive leadership, unconscious bias, allyship, policy reform, culture change… |
| **Research Funding Process** | Funding-related EDI aspects in the EPSRC / UKRI context — grant access, reviewer diversity, funding criteria equity. **Do not apply to general people recruitment content.** |
| **Adoption Readiness Level** | How immediately actionable the resource is — awareness raising through to implementation-ready toolkit |

> **Scope:** This pipeline targets UK engineering, physical, and mathematical sciences research within the EPSRC remit. Tags should be grounded exclusively in the text of the resource — no external knowledge should be applied.

---

## LLM Architecture

### Model
- **OLMo-2 7B** (`olmo2:7b` via Ollama)
- Context window: 4,096 tokens
- Runs entirely locally — no data sent externally

### Tokeniser
- `allenai/OLMo-2-7B-1124` via HuggingFace `transformers`
- Used for accurate token counting in Stages 3, 4, and 6 routing
- Loaded lazily — only when needed

### Routing
| Condition | Strategy |
|-----------|----------|
| `combined_token_count` ≤ threshold | Single-pass tagging + summarisation |
| `combined_token_count` > threshold | Map-reduce: tag/summarise per chunk, then reduce |

### Prompting
- System prompt establishes EPSRC context, semantic grounding rules, and output format
- All three task prompts (tagging, chunk summary, reduce) are consistent with the system prompt
- All prompts explicitly prohibit fabrication — OLMo must base every tag and summary claim solely on the provided text
- Grounding enforced with `tag_evidence` field: every tag must cite an exact quote from the source

---

## Evaluation

Pilot evaluation was run on three resources (IDs 39, 44, 202):

| Metric | Score |
|--------|-------|
| F1 | 0.59 |
| Precision | 0.58 |
| Recall | 0.60 |

**Per-dimension F1 (corrected ground truth):**

| Dimension | F1 |
|-----------|-----|
| Career Pathway | 0.67 |
| Adoption Readiness Level | 0.67 |
| Organisational Culture | 0.61 |
| Individual Characteristics | 0.56 |
| Research Funding Process | 0.00 |

Research Funding Process F1 of 0.00 reflects the difficulty of applying a funding-specific dimension to general EDI documents — this is an expected and known limitation at this stage.

**Planned future evaluation:** BERTScore and SummaC for summary quality; human spot-check by Prof. Vania Dimitrova.

---

## Configuration Reference

All key settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `RATE_LIMIT_SEC` | `2` | Seconds to wait between HTTP requests |
| `REQUEST_TIMEOUT_SEC` | `15` | Per-request timeout in seconds |
| `USER_AGENT` | `EDIHubBot/1.0` | Bot identifier used in requests and robots.txt checks |
| `LOG_FILE` | `logs/pipeline.log` | Path for the full DEBUG log |
| `JS_WORD_COUNT_THRESHOLD` | `150` | Word count below which a page is flagged as JS-heavy |
| `MAX_EXT_LINKS_PER_RESOURCE` | `20` | Max external links followed per resource in Stage 5 |
| `MAX_PDF_LINKS_PER_RESOURCE` | `20` | Max PDF links followed per resource in Stage 5 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `olmo2:7b` | Model name passed to Ollama |
| `CHUNK_TOKEN_OVERLAP` | `175` | Overlap tokens between adjacent chunks |

---

## Status Codes and Failure Reasons

Resources are assigned a `status` field as they move through the pipeline:

| Status | Stage set | Meaning |
|--------|-----------|---------|
| `success` | 2 | Fetched and processed successfully |
| `pdf-skip` | 2 | Direct PDF URL — handled by Stage 4 instead |
| `js-skip` | 2 | Page returned too little text — likely JS-rendered |
| `failed` | 2 | Fetch failed — see `failure_detail` |

`failure_detail` values (from `http_client.py`):

| Value | Cause |
|-------|-------|
| `robots-blocked` | `robots.txt` disallows our bot for this URL |
| `http-error (4xx/5xx)` | Server returned an error status code |
| `timeout` | Request exceeded `REQUEST_TIMEOUT_SEC` |
| `connection-error` | Could not connect to the server |
| `unknown-error (ExceptionType)` | Any other unexpected exception |

---

## Ethical Scraping

The pipeline is designed to scrape responsibly:

- **`robots.txt` compliance** — checked before every fetch; disallowed URLs are skipped, never bypassed. Results are cached at domain level to avoid repeated lookups.
- **Rate limiting** — a configurable delay (`RATE_LIMIT_SEC`) is enforced between every request, regardless of domain.
- **Transparent User-Agent** — the bot identifies itself with a descriptive `User-Agent` string so site owners can identify and contact us.
- **No depth crawling** — each resource URL is fetched as a single page only (plus one level of linked documents in Stage 5). We do not spider entire sites.
- **Local model** — OLMo-2 7B runs entirely on-device via Ollama. No resource content is sent to any external API or third-party service.

---

## Known Limitations

- **JS-rendered pages** — resources behind JavaScript rendering (React, Angular, etc.) return too little text to process. These are logged in `stage2_skipped.json` for manual review.
- **Login-walled resources** — resource 65 (Warwick) requires authentication and cannot be scraped automatically.
- **robots.txt blocks** — approximately 8 domains block our bot. These resources require manual content input or direct liaison with site owners.
- **OLMo-2 context window** — at 4,096 tokens, very long documents must go through map-reduce. Some nuance can be lost in the reduce step.
- **Research Funding Process dimension** — this dimension is difficult to apply correctly to general EDI documents; F1 of 0.00 in pilot evaluation. Prompt refinement and more ground truth examples are needed.
- **PDF extraction quality** — scanned PDFs (image-only) cannot be extracted by any of the three extractors. These will produce empty or near-empty text and are flagged in logs.

---

## Team

| Name | Role |
|------|------|
| **Siddesh Rohri** | Pipeline development (summer intern) |
| **Prof. Vania Dimitrova** | Project supervisor, University of Leeds |
| **Gabby Keating** | Taxonomy definition, ground truth annotation |
| **Louise Jennings** | Ground truth annotation |

Part of the **STEMM Change** consortium, funded by EPSRC's **Inclusion Matters** portfolio (£5.5M national investment).

---

## Licence

*To be confirmed with the project team before making the repository public.*  
